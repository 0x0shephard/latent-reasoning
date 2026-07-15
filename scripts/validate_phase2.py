"""Preflight a Phase-2 CODI/KaVa config before starting a Kaggle GPU run.

Checks the controlled-comparison contract, real tokenizer/data sequence construction,
anti-shortcut trace removal, latent/teacher context lengths, and loss/compression settings.

    python scripts/validate_phase2.py --config configs/codi.yaml --peer-config configs/kava.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from src.data.datasets import load_train_set
from src.data.prompts import PromptStyle
from src.data.teacher_cache import encode_latent_example
from src.models.latent_lm import BOT_TOKEN, EOT_TOKEN
from src.train.latent import LATENT_METHODS, _method_defaults
from src.train.sft import resolve_total_steps
from src.utils.config import load_config


_CONTROLLED_TASK_KEYS = (
    "backbone",
    "backbone_revision",
    "trace_style",
    "max_length",
    "latent_steps",
    "mechanism",
    "jacobi_iterations",
    "projection_dim",
    "projection_dropout",
    "student_ce_weight",
    "teacher_ce_weight",
)
_CONTROLLED_TRAIN_KEYS = (
    "epochs",
    "batch_size",
    "lr",
    "lr_schedule",
    "min_lr_ratio",
    "weight_decay",
    "warmup_steps",
    "grad_clip",
)
_CONTROLLED_DISTILL_KEYS = (
    "hidden_weight",
    "hidden_layers",
    "hidden_metric",
    "hidden_layer_reduction",
    "normalize_teacher_std",
)


def _sample_indices(size: int, count: int, seed: int) -> list[int]:
    if count <= 0 or count >= size:
        return list(range(size))
    return sorted(random.Random(seed).sample(range(size), count))


def _controlled_differences(cfg, peer) -> list[dict]:
    differences = []
    for section, keys in (
        ("task", _CONTROLLED_TASK_KEYS),
        ("train", _CONTROLLED_TRAIN_KEYS),
    ):
        left, right = cfg[section], peer[section]
        for key in keys:
            if left.get(key) != right.get(key):
                differences.append(
                    {
                        "field": f"{section}.{key}",
                        "configured": left.get(key),
                        "peer": right.get(key),
                    }
                )
    left_distill = cfg.task.get("distillation", {})
    right_distill = peer.task.get("distillation", {})
    for key in _CONTROLLED_DISTILL_KEYS:
        if left_distill.get(key) != right_distill.get(key):
            differences.append(
                {
                    "field": f"task.distillation.{key}",
                    "configured": left_distill.get(key),
                    "peer": right_distill.get(key),
                }
            )
    for key in ("seed", "data_config"):
        if cfg.get(key) != peer.get(key):
            differences.append(
                {"field": key, "configured": cfg.get(key), "peer": peer.get(key)}
            )
    return differences


def validate(cfg, sample_size: int, peer_cfg=None) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    method = cfg.task.get("method", "codi")
    failures = []
    if cfg.task.get("type") != "latent":
        failures.append("task.type must be 'latent'")
    if method not in LATENT_METHODS:
        failures.append(f"unsupported method {method!r}")
    default_hidden, default_kv, default_compression = _method_defaults(method)
    distill = cfg.task.get("distillation", {})
    hidden_weight = float(distill.get("hidden_weight", default_hidden))
    kv_weight = float(distill.get("kv_weight", default_kv))
    compression = distill.get("compression", default_compression)
    if method == "codi" and kv_weight != 0:
        failures.append("CODI must set distillation.kv_weight=0")
    if method.startswith("kava") and kv_weight <= 0:
        failures.append("KaVa must use a positive distillation.kv_weight")
    if compression not in {"rkv", "random", "uniform"}:
        failures.append(f"unsupported KV compression {compression!r}")

    revision = cfg.task.get("backbone_revision")
    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg.task.backbone, **kwargs)
    model_config = AutoConfig.from_pretrained(cfg.task.backbone, **kwargs)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [BOT_TOKEN, EOT_TOKEN]}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bot = tokenizer.convert_tokens_to_ids(BOT_TOKEN)
    eot = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    cue_ids = tokenizer(
        load_config(cfg.data_config)["prompt"]["answer_prefix"],
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.decode([cue_ids[-1]]).strip() != ":":
        failures.append("answer_prefix does not end at a standalone ':' tokenizer position")

    data_cfg = load_config(cfg.data_config)
    style = PromptStyle.from_config(data_cfg["prompt"])
    trace_style = cfg.task.get("trace_style", "eq_only")
    dataset = load_train_set(data_cfg, trace_style)
    total_steps = resolve_total_steps(cfg, len(dataset))
    lengths, short_traces, empty_traces, truncated = [], 0, 0, 0
    sequence_failures = []
    latent_steps = int(cfg.task.get("latent_steps", 6))
    for index in _sample_indices(len(dataset), sample_size, int(cfg.seed)):
        try:
            encoded = encode_latent_example(
                tokenizer,
                dataset[index],
                style,
                bot_token_id=bot,
                eot_token_id=eot,
                trace_style=trace_style,
                max_length=int(cfg.task.get("max_length", 256)),
                latent_steps=latent_steps,
            )
        except ValueError as exc:
            sequence_failures.append({"index": index, "error": str(exc)})
            continue
        trace_tokens = encoded.teacher_trace_end - encoded.teacher_trace_start
        lengths.append(trace_tokens)
        empty_traces += int(trace_tokens == 0)
        short_traces += int(trace_tokens < latent_steps)
        truncated += int(encoded.reasoning_truncated)
    if sequence_failures:
        failures.append(f"{len(sequence_failures)} sampled sequences failed construction")

    peer_differences = _controlled_differences(cfg, peer_cfg) if peer_cfg else []
    if peer_differences:
        failures.append("CODI/KaVa peer configs differ in controlled architecture/training fields")
    context_window = getattr(model_config, "max_position_embeddings", None)
    max_length = int(cfg.task.get("max_length", 256))
    if context_window and max_length > context_window:
        failures.append(
            f"task.max_length={max_length} exceeds backbone context window {context_window}"
        )

    report = {
        "status": "failed" if failures else "ok",
        "method": method,
        "backbone": cfg.task.backbone,
        "resolved_backbone_revision": getattr(model_config, "_commit_hash", None),
        "tokenizer_class": type(tokenizer).__name__,
        "bot_token_id": bot,
        "eot_token_id": eot,
        "latent_steps": latent_steps,
        "mechanism": cfg.task.get("mechanism", "autoregressive"),
        "hidden_weight": hidden_weight,
        "kv_weight": kv_weight,
        "compression": compression,
        "train_examples": len(dataset),
        "train_dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "sampled_examples": len(lengths),
        "teacher_trace_tokens": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": sum(lengths) / len(lengths) if lengths else None,
            "empty_after_drop_last": empty_traces,
            "shorter_than_latent_budget": short_traces,
        },
        "reasoning_truncated": truncated,
        "sequence_failures": sequence_failures[:20],
        "total_steps": total_steps,
        "effective_epochs": total_steps * cfg.train.batch_size / len(dataset),
        "controlled_peer_differences": peer_differences,
    }
    if failures:
        report["failures"] = failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Phase-2 experiment contract.")
    parser.add_argument("--config", default="configs/codi.yaml")
    parser.add_argument("--peer-config", default=None)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--set", nargs="*", default=[], help="Config dot-overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    peer = load_config(args.peer_config) if args.peer_config else None
    if cfg.get("offline", False):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        report = validate(cfg, args.sample_size, peer_cfg=peer)
    except Exception as exc:
        report = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "phase2_validation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[validate] wrote {path}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
