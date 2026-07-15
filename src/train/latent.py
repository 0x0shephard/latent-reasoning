"""Phase-2 CODI/KaVa task builder for the shared session-safe trainer."""
from __future__ import annotations

import math

import torch

from src.data.datasets import load_train_set
from src.data.prompts import PromptStyle
from src.data.teacher_cache import (
    collate_latent_rows,
    extract_teacher_hidden,
    extract_teacher_targets,
)
from src.losses.kv_compress import random_compress, rkv_compress, uniform_compress
from src.losses.trajectory_match import TrajectoryMatchLoss
from src.models.latent_lm import LatentCausalLM, add_latent_tokens
from src.train.batching import StepBatcher
from src.train.sft import resolve_total_steps
from src.utils.config import load_config


LATENT_METHODS = {
    "codi",
    "kava",
    "latent_nodistill",
    "kava_random",
    "kava_uniform",
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _method_defaults(method: str) -> tuple[float, float, str]:
    if method == "codi":
        return 1.0, 0.0, "rkv"
    if method == "kava":
        return 1.0, 1.0, "rkv"
    if method == "latent_nodistill":
        return 0.0, 0.0, "rkv"
    if method == "kava_random":
        return 1.0, 1.0, "random"
    if method == "kava_uniform":
        return 1.0, 1.0, "uniform"
    raise ValueError(f"unknown latent method {method!r}")


def _lr_at(step: int, total_steps: int, train_cfg) -> float:
    base = float(train_cfg.lr)
    warmup = int(train_cfg.get("warmup_steps", 0))
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    if train_cfg.get("lr_schedule", "cosine") == "constant":
        return base
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    minimum = float(train_cfg.get("min_lr_ratio", 0.0))
    factor = minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * factor


def _teacher_target_forward(model: LatentCausalLM, batch, need_kv: bool):
    kwargs = {
        "input_ids": batch.teacher_ids,
        "attention_mask": batch.teacher_mask,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if need_kv:
        kwargs.update({"use_cache": True, "output_attentions": True})
    else:
        kwargs.update({"use_cache": False})
    with torch.no_grad():
        return model.backbone(**kwargs)


def build_latent_task(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tcfg = cfg.task
    method = tcfg.get("method", "codi")
    if method not in LATENT_METHODS:
        raise ValueError(
            f"unknown latent method {method!r}; expected one of {sorted(LATENT_METHODS)}"
        )
    default_hidden, default_kv, default_compression = _method_defaults(method)
    distill = tcfg.get("distillation", {})
    hidden_weight = float(distill.get("hidden_weight", default_hidden))
    kv_weight = float(distill.get("kv_weight", default_kv))
    compression = distill.get("compression", default_compression)
    if compression not in {"rkv", "random", "uniform"}:
        raise ValueError("distillation.compression must be rkv, random, or uniform")

    data_cfg = load_config(cfg.data_config)
    style = PromptStyle.from_config(data_cfg["prompt"])
    trace_style = tcfg.get("trace_style", "eq_only")
    dataset = load_train_set(data_cfg, trace_style)
    total_steps = resolve_total_steps(cfg, len(dataset))
    revision = tcfg.get("backbone_revision")
    pretrained_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(tcfg.backbone, **pretrained_kwargs)
    # KaVa needs answer-to-trace attention weights. Eager attention is the only HF backend
    # guaranteed to expose them across supported Transformers versions.
    if kv_weight:
        pretrained_kwargs["attn_implementation"] = "eager"
    backbone = AutoModelForCausalLM.from_pretrained(tcfg.backbone, **pretrained_kwargs)
    if kv_weight and hasattr(backbone, "set_attn_implementation"):
        backbone.set_attn_implementation("eager")
    bot_token_id, eot_token_id = add_latent_tokens(tokenizer, backbone)
    device = _device()
    model = LatentCausalLM(
        backbone,
        bot_token_id=bot_token_id,
        eot_token_id=eot_token_id,
        latent_steps=int(tcfg.get("latent_steps", 6)),
        mechanism=tcfg.get("mechanism", "autoregressive"),
        jacobi_iterations=int(tcfg.get("jacobi_iterations", 3)),
        projection_dim=tcfg.get("projection_dim"),
        projection_dropout=float(tcfg.get("projection_dropout", 0.0)),
    ).to(device)
    model.train()

    cfg["task"]["resolved_backbone_revision"] = (
        getattr(backbone.config, "_commit_hash", None) or revision or "unresolved"
    )
    cfg["task"]["train_dataset_fingerprint"] = getattr(
        dataset, "_fingerprint", "unavailable"
    )
    cfg["task"]["train_examples"] = len(dataset)
    cfg["task"]["bot_token_id"] = bot_token_id
    cfg["task"]["eot_token_id"] = eot_token_id

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    batcher = StepBatcher(len(dataset), int(cfg.train.batch_size), seed=int(cfg.seed))
    loss_module = TrajectoryMatchLoss(
        hidden_weight=hidden_weight,
        kv_weight=kv_weight,
        hidden_layers=distill.get("hidden_layers", "all"),
        kv_layers=distill.get("kv_layers", "all"),
        hidden_metric=distill.get("hidden_metric", "l1"),
        kv_metric=distill.get("kv_metric", "l1"),
        normalize_teacher_std=bool(distill.get("normalize_teacher_std", True)),
        hidden_layer_reduction=distill.get("hidden_layer_reduction", "sum"),
    )
    max_length = int(tcfg.get("max_length", 256))
    student_ce_weight = float(tcfg.get("student_ce_weight", 1.0))
    teacher_ce_weight = float(tcfg.get("teacher_ce_weight", 1.0))
    importance_weight = float(distill.get("importance_weight", 0.1))
    effective_epochs = total_steps * cfg.train.batch_size / len(dataset)
    print(
        f"[data] train_examples={len(dataset)} method={method} M={model.latent_steps} "
        f"mechanism={model.mechanism} planned_epochs={effective_epochs:.3f}"
    )

    stats = {"seen": 0, "truncated": 0}

    def step_fn(step: int) -> float:
        for group in optimizer.param_groups:
            group["lr"] = _lr_at(step, total_steps, cfg.train)
        rows = [dataset[int(index)] for index in batcher.batch_indices(step)]
        batch = collate_latent_rows(
            tokenizer,
            rows,
            style,
            bot_token_id=bot_token_id,
            eot_token_id=eot_token_id,
            trace_style=trace_style,
            max_length=max_length,
            latent_steps=model.latent_steps,
        ).to(device)
        stats["seen"] += len(rows)
        stats["truncated"] += batch.reasoning_truncated

        optimizer.zero_grad(set_to_none=True)
        target_outputs = None
        if hidden_weight or kv_weight:
            target_outputs = _teacher_target_forward(model, batch, need_kv=bool(kv_weight))
        if kv_weight:
            assert target_outputs is not None
            teacher_targets = extract_teacher_targets(target_outputs, batch)
            if compression == "rkv":
                compressed = rkv_compress(
                    teacher_targets.trace_keys,
                    teacher_targets.trace_values,
                    teacher_targets.importance,
                    teacher_targets.trace_mask,
                    model.latent_steps,
                    importance_weight=importance_weight,
                )
            elif compression == "uniform":
                compressed = uniform_compress(
                    teacher_targets.trace_keys,
                    teacher_targets.trace_values,
                    teacher_targets.trace_mask,
                    model.latent_steps,
                )
            else:
                generator = torch.Generator(device=device).manual_seed(
                    int(cfg.seed) * 1_000_003 + step
                )
                compressed = random_compress(
                    teacher_targets.trace_keys,
                    teacher_targets.trace_values,
                    teacher_targets.trace_mask,
                    model.latent_steps,
                    generator=generator,
                )
            teacher_hidden = teacher_targets.hidden_endpoint
            # Do not retain full teacher attentions/caches through the gradient-bearing
            # teacher and student passes; the compressed tensors are independent gathers.
            del teacher_targets, target_outputs
        elif hidden_weight:
            assert target_outputs is not None
            teacher_hidden = extract_teacher_hidden(
                target_outputs, batch.teacher_endpoint
            )
            compressed = None
            del target_outputs
        else:
            teacher_hidden = torch.empty(0, device=device)
            compressed = None

        # Shared teacher model receives its own explicit-CoT language-model objective.
        teacher = model.backbone(
            input_ids=batch.teacher_ids,
            attention_mask=batch.teacher_mask,
            labels=batch.teacher_labels,
            use_cache=False,
            return_dict=True,
        )
        student = model.forward_student(batch)
        trajectory = loss_module(
            student_hidden=student.hidden_endpoint,
            teacher_hidden=teacher_hidden,
            student_keys=student.latent_keys,
            student_values=student.latent_values,
            teacher_keys=None if compressed is None else compressed.keys,
            teacher_values=None if compressed is None else compressed.values,
            kv_mask=None if compressed is None else compressed.mask,
        )
        total = (
            student_ce_weight * student.answer_loss
            + teacher_ce_weight * teacher.loss
            + trajectory.total
        )
        total.backward()
        grad_clip = float(cfg.train.get("grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        log_every = int(cfg.train.get("log_every", cfg.train.ckpt_every))
        if (step + 1) % log_every == 0:
            rate = stats["truncated"] / max(1, stats["seen"])
            print(
                f"[loss] student_ce={float(student.answer_loss.detach()):.5f} "
                f"teacher_ce={float(teacher.loss.detach()):.5f} "
                f"hidden={float(trajectory.hidden.detach()):.5f} "
                f"kv={float(trajectory.kv.detach()):.5f} "
                f"reasoning_truncated={rate:.2%}"
            )
        return float(total.detach())

    return model, optimizer, step_fn
