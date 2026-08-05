"""Train and evaluate one arm of the rank-matched endpoint-retention experiment."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import (
    _atomic_json as _write_atomic_json,
    _atomic_torch_save as _write_atomic_torch_save,
)
from scripts.collect_official_codi_endpoint_answer_conditioned import (
    LEGACY_EXCLUSION,
    sample_fresh_answer_conditioned_partitions,
)
from scripts.collect_official_codi_endpoint_parameter_aware import (
    PREVIOUS_ANSWER_CONDITIONED_EXCLUSION,
    sample_fresh_parameter_aware_partitions,
)
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    sample_endpoint_tsvc_partitions,
    verify_full_reproduction_gate,
)
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set, load_train_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
    official_codi_answer_is_eligible,
)
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.endpoint_retention import (
    RETENTION_COMMON_RANK,
    RETENTION_COMMON_STATES,
    RETENTION_CONTRACT,
    RETENTION_SCHEMA_VERSION,
    RETENTION_TRAINING_ARMS,
    endpoint_retention_loss,
    load_retention_bases,
    retention_bases_state,
    retention_basis_for_arm,
)
from src.mech.endpoint_tsvc import match_gradient_norm
from src.mech.kv_target_utility import autograd_gradients, combine_gradients
from src.mech.official_codi_target_utility import (
    OfficialCODIAnswerScorer,
    extract_official_teacher_endpoint_targets,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


PARAMETER_AWARE_EXCLUSION = {
    "residual_fit_examples": 1024,
    "direction_selection_examples": 1024,
    "update_examples": 256,
    "validation_examples": 256,
    "seed": 41,
}


def _atomic_json(path: Path, payload: dict) -> None:
    """Path-first wrapper used consistently by this runner."""
    _write_atomic_json(payload, path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    """Path-first wrapper used consistently by this runner."""
    _write_atomic_torch_save(payload, path)


def _sha256_json(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _amp_context(device: torch.device, precision: str):
    normalized = precision.casefold()
    if normalized in {"float32", "fp32"} or device.type != "cuda":
        return nullcontext()
    dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(normalized)
    if dtype is None:
        raise ValueError("precision must be float32, float16, or bfloat16")
    return torch.autocast(device_type="cuda", dtype=dtype)


def sample_fresh_retention_training(
    dataset,
    *,
    training_examples: int,
    seed: int,
) -> tuple[list[int], dict]:
    """Exclude every normalized question used by the three completed selectors."""
    if training_examples <= 0:
        raise ValueError("training_examples must be positive")
    legacy, legacy_metadata = sample_endpoint_tsvc_partitions(
        dataset,
        calibration_examples=LEGACY_EXCLUSION["calibration_examples"],
        update_examples=LEGACY_EXCLUSION["update_examples"],
        validation_examples=LEGACY_EXCLUSION["validation_examples"],
        seed=LEGACY_EXCLUSION["seed"],
    )
    answer, answer_metadata = sample_fresh_answer_conditioned_partitions(
        dataset,
        residual_fit_examples=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION[
            "residual_fit_examples"
        ],
        direction_selection_examples=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION[
            "direction_selection_examples"
        ],
        update_examples=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION["update_examples"],
        validation_examples=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION[
            "validation_examples"
        ],
        seed=PREVIOUS_ANSWER_CONDITIONED_EXCLUSION["seed"],
    )
    parameter, parameter_metadata = sample_fresh_parameter_aware_partitions(
        dataset,
        residual_fit_examples=PARAMETER_AWARE_EXCLUSION["residual_fit_examples"],
        direction_selection_examples=PARAMETER_AWARE_EXCLUSION[
            "direction_selection_examples"
        ],
        update_examples=PARAMETER_AWARE_EXCLUSION["update_examples"],
        validation_examples=PARAMETER_AWARE_EXCLUSION["validation_examples"],
        seed=PARAMETER_AWARE_EXCLUSION["seed"],
    )
    completed = {"energy": legacy, "answer_conditioned": answer, "parameter_aware": parameter}
    excluded_indices = [
        int(index)
        for partitions in completed.values()
        for indices in partitions.values()
        for index in indices
    ]
    excluded_questions = {
        _normalized_question(dataset[index]["question"]) for index in excluded_indices
    }
    groups: dict[str, list[int]] = {}
    for index, (question, answer_value) in enumerate(
        zip(dataset["question"], dataset["answer"])
    ):
        normalized = _normalized_question(question)
        if (
            normalized not in excluded_questions
            and official_codi_answer_is_eligible(answer_value)
        ):
            groups.setdefault(normalized, []).append(index)
    if len(groups) < training_examples:
        raise ValueError(
            f"need {training_examples} fresh eligible questions, found {len(groups)}"
        )
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    indices = [generator.choice(groups[key]) for key in keys[:training_examples]]
    metadata = {
        "excluded_unique_questions": len(excluded_questions),
        "expected_excluded_unique_questions": 10_632,
        "selected_unique_questions": len(indices),
        "data_seed": seed,
        "completed_partition_metadata": {
            "energy": legacy_metadata,
            "answer_conditioned": answer_metadata,
            "parameter_aware": parameter_metadata,
        },
        "training_indices_sha256": _sha256_json(indices),
    }
    if getattr(dataset, "_fingerprint", None) is not None and len(excluded_questions) != 10_632:
        raise RuntimeError("completed selector exclusions no longer contain 10,632 questions")
    return indices, metadata


def _restore_trainable(parameters_by_name, values: dict[str, torch.Tensor]) -> None:
    if set(values) != set(parameters_by_name):
        raise RuntimeError("retention checkpoint trainable parameter names changed")
    with torch.no_grad():
        for name, parameter in parameters_by_name.items():
            parameter.copy_(values[name].to(device=parameter.device, dtype=parameter.dtype))


def _trainable_state(parameters_by_name) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in parameters_by_name.items()
    }


def _save_progress(
    path: Path,
    *,
    request_sha256: str,
    next_step: int,
    parameters_by_name,
    optimizer,
    device: torch.device,
) -> None:
    payload = {
        "schema_version": RETENTION_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "next_step": next_step,
        "trainable": _trainable_state(parameters_by_name),
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    _atomic_torch_save(path, payload)


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    if args.arm not in RETENTION_TRAINING_ARMS:
        raise ValueError(f"unknown arm {args.arm!r}")
    if args.batch_size < 2 or args.training_examples % args.batch_size:
        raise ValueError("training examples must be divisible by a batch size of at least two")
    if args.epochs <= 0 or args.learning_rate <= 0 or args.save_every <= 0:
        raise ValueError("epochs, learning rate, and save interval must be positive")
    if args.weight_decay < 0 or args.eval_limit < 0 or args.eval_batch_size <= 0:
        raise ValueError("weight decay/eval limit must be non-negative and eval batch positive")

    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("endpoint retention training requires CUDA")
    dtype = resolve_torch_dtype(args.precision, device)
    torch.manual_seed(args.training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.training_seed)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id),
        revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename),
        expected_sha256=str(cfg.checkpoint.sha256),
        token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model),
        base_revision=str(cfg.model.base_revision),
        dtype=dtype,
        settings=cfg.model,
        token=token,
    )
    load_report = load_official_checkpoint(
        model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
    )
    bases = load_retention_bases(
        energy_path=args.energy_basis,
        answer_conditioned_path=args.answer_conditioned_basis,
        parameter_aware_path=args.parameter_aware_basis,
        checkpoint_sha256=load_report.checkpoint_sha256,
    )
    selected_basis, arm_mode = retention_basis_for_arm(args.arm, bases)
    model.to(device=device, dtype=dtype)
    if selected_basis is not None:
        selected_basis = replace(
            selected_basis,
            basis=selected_basis.basis.to(device=device, dtype=dtype),
            ranks=selected_basis.ranks.to(device=device),
        )
    scorer = OfficialCODIAnswerScorer(
        model, latent_positions=int(cfg.eval.latent_iterations)
    )
    parameters_by_name = {
        name: parameter
        for name, parameter in scorer.named_parameters()
        if parameter.requires_grad
    }
    if not parameters_by_name:
        raise RuntimeError("official CODI exposes no trainable parameters")
    parameters = list(parameters_by_name.values())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    dataset = load_train_set(data_cfg, "eq_only")
    training_indices, exclusion = sample_fresh_retention_training(
        dataset, training_examples=args.training_examples, seed=args.data_seed
    )
    order_generator = random.Random(args.training_seed)
    epoch_orders = []
    for _ in range(args.epochs):
        order = list(training_indices)
        order_generator.shuffle(order)
        epoch_orders.extend(order)
    batches = [
        epoch_orders[start : start + args.batch_size]
        for start in range(0, len(epoch_orders), args.batch_size)
    ]
    request = {
        "schema_version": RETENTION_SCHEMA_VERSION,
        "contract": RETENTION_CONTRACT,
        "arm": args.arm,
        "arm_mode": arm_mode,
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "basis_sources": retention_bases_state(bases),
        "common_states": list(RETENTION_COMMON_STATES),
        "common_rank_per_state": RETENTION_COMMON_RANK,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", "unavailable"),
        "exclusion": exclusion,
        "training_examples": args.training_examples,
        "training_indices": training_indices,
        "training_seed": args.training_seed,
        "data_seed": args.data_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
        "eval_precision": args.eval_precision,
        "eval_limit": args.eval_limit,
        "eval_batch_size": args.eval_batch_size,
        "auxiliary_norm_matching": "selected/complement matched to full-common every batch",
        "base_objective": "official student gold-answer NLL",
    }
    # Tensors in the frozen bases are represented by immutable source hashes above.
    request["basis_sources"] = {
        name: {
            key: value
            for key, value in state.items()
            if key not in {"basis", "ranks"}
        }
        for name, state in request["basis_sources"].items()
    }
    request_sha256 = _sha256_json(request)
    compatible_progress_hashes = {request_sha256}
    # Migrations from the first two notebook revisions. Evaluation dtype/batching cannot
    # change already-trained weights, so those completed training states remain valid.
    for legacy_batch_size in (128, 64):
        legacy_request = dict(request)
        legacy_request.pop("eval_precision", None)
        legacy_request["eval_batch_size"] = legacy_batch_size
        compatible_progress_hashes.add(_sha256_json(legacy_request))
    request["request_sha256"] = request_sha256
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if completed.get("request_sha256") != request_sha256:
            raise RuntimeError("completed output belongs to a different request")
        if not (output_dir / "gsm8k.jsonl").is_file():
            raise RuntimeError("completed retention summary is missing paired predictions")
        print(f"[resume] already complete: {output_dir}")
        return completed
    _atomic_json(
        output_dir / "run_manifest.json",
        {**request, "state": "running", "created_at_utc": datetime.now(timezone.utc).isoformat()},
    )

    progress_path = output_dir / "training_state.pt"
    start_step = 0
    if progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("request_sha256") not in compatible_progress_hashes:
            raise RuntimeError("training checkpoint belongs to a different request")
        if progress.get("request_sha256") != request_sha256:
            print(
                "[resume] accepting pre-memory-fix training state; only evaluation "
                "dtype/batching changed"
            )
        _restore_trainable(parameters_by_name, progress["trainable"])
        optimizer.load_state_dict(progress["optimizer"])
        torch.random.set_rng_state(progress["cpu_rng_state"])
        if device.type == "cuda" and progress.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(progress["cuda_rng_state"])
        start_step = int(progress["next_step"])
        if not 0 <= start_step <= len(batches):
            raise RuntimeError("retention checkpoint step is outside the frozen schedule")

    model.train()
    train_started = time.perf_counter()
    norm_records = []
    iterator = tqdm(
        range(start_step, len(batches)),
        desc=f"Retention training: {args.arm}/seed{args.training_seed}",
        unit="batch",
    )
    for step in iterator:
        batch = collate_official_codi_kv_rows(
            tokenizer,
            [dataset[index] for index in batches[step]],
            bot_token_id=model.bot_id,
        ).to(device)
        teacher = extract_official_teacher_endpoint_targets(model, batch).all_hidden
        with _amp_context(device, args.precision):
            output = scorer(batch, return_answer_endpoint_hidden=True)
        student = output.student_answer_endpoint_hidden
        if student is None:
            raise RuntimeError("student answer-cue endpoint states are missing")
        needs_auxiliary = args.arm != "answer_only"
        base_gradients = autograd_gradients(
            output.mean_loss, parameters, retain_graph=needs_auxiliary
        )
        matching = None
        if needs_auxiliary:
            full_loss = endpoint_retention_loss(student, teacher, mode="full")
            if args.arm == "full_common":
                raw_auxiliary = autograd_gradients(
                    full_loss, parameters, retain_graph=False
                )
                reference_auxiliary = raw_auxiliary
            else:
                reference_auxiliary = autograd_gradients(
                    full_loss, parameters, retain_graph=True
                )
                arm_loss = endpoint_retention_loss(
                    student,
                    teacher,
                    mode=arm_mode,
                    basis=selected_basis.basis,
                    ranks=selected_basis.ranks,
                )
                raw_auxiliary = autograd_gradients(
                    arm_loss, parameters, retain_graph=False
                )
            matched_auxiliary, matching = match_gradient_norm(
                raw_auxiliary, reference_auxiliary
            )
            total_gradients = combine_gradients(base_gradients, matched_auxiliary)
        else:
            total_gradients = combine_gradients(base_gradients)
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, total_gradients):
            parameter.grad = None if gradient is None else gradient.detach()
        optimizer.step()
        del parameter, gradient
        if matching is not None:
            norm_records.append(matching)
        iterator.set_postfix(answer_loss=float(output.mean_loss.detach().float()))
        if (step + 1) % args.save_every == 0 and step + 1 < len(batches):
            _save_progress(
                progress_path,
                request_sha256=request_sha256,
                next_step=step + 1,
                parameters_by_name=parameters_by_name,
                optimizer=optimizer,
                device=device,
            )
        del batch, teacher, output, student, base_gradients, total_gradients
        if needs_auxiliary:
            del full_loss, raw_auxiliary, reference_auxiliary, matched_auxiliary
            if args.arm != "full_common":
                del arm_loss
    training_seconds = time.perf_counter() - train_started
    # Preserve the fully trained state if generation is interrupted after training.
    _save_progress(
        progress_path,
        request_sha256=request_sha256,
        next_step=len(batches),
        parameters_by_name=parameters_by_name,
        optimizer=optimizer,
        device=device,
    )

    # AdamW keeps two parameter-sized moment buffers, and the final batch may retain
    # auxiliary gradient tuples. Neither is needed for inference. Releasing them before
    # prompt logits are allocated is required for full-GSM8K evaluation on a 16 GiB T4.
    optimizer.zero_grad(set_to_none=True)
    for parameter in parameters:
        parameter.grad = None
    del optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # The official reproduction config uses dtype=auto, which resolves to float16 on a
    # T4. Training remains float32; only the already-frozen evaluation copy is cast.
    eval_dtype = resolve_torch_dtype(args.eval_precision, device)
    model.to(device=device, dtype=eval_dtype).eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    eval_cfg = load_config(str(cfg.data_config))
    examples = load_eval_set("gsm8k", eval_cfg.eval.gsm8k)
    if len(examples) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("GSM8K evaluation count drifted")
    if args.eval_limit:
        examples = examples[: args.eval_limit]
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_started = time.perf_counter()
    generations = generate_official_codi(
        model,
        tokenizer,
        [example["question"] for example in examples],
        latent_iterations=int(cfg.eval.latent_iterations),
        max_new_tokens=int(cfg.eval.max_new_tokens),
        batch_size=args.eval_batch_size,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_started
    official_correct = [
        bool(official_answers_match(generation, example["gold"]))
        for generation, example in zip(generations, examples)
    ]
    numeric_correct = [
        bool(answers_match(generation, example["gold"]))
        for generation, example in zip(generations, examples)
    ]
    records = [
        {
            "index": index,
            "question": example["question"],
            "gold": str(example["gold"]),
            "generation": generation,
            "correct": correct,
            "numeric_exact_match_correct": numeric,
        }
        for index, (example, generation, correct, numeric) in enumerate(
            zip(examples, generations, official_correct, numeric_correct)
        )
    ]
    predictions_path = output_dir / "gsm8k.jsonl"
    temporary = predictions_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(predictions_path)
    result = {
        "schema_version": RETENTION_SCHEMA_VERSION,
        "contract": RETENTION_CONTRACT,
        "request_sha256": request_sha256,
        "arm": args.arm,
        "training_seed": args.training_seed,
        "training": {
            "steps": len(batches),
            "seconds": training_seconds,
            "mean_auxiliary_scale": (
                sum(item["auxiliary_scale"] for item in norm_records) / len(norm_records)
                if norm_records else None
            ),
        },
        "evaluation": {
            "dataset": "gsm8k",
            "count": len(examples),
            "correct": sum(official_correct),
            "accuracy": sum(official_correct) / len(examples),
            "numeric_exact_match_accuracy": sum(numeric_correct) / len(examples),
            "seconds": eval_seconds,
            "examples_per_second": len(examples) / eval_seconds,
            "predictions": str(predictions_path),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.save_trained_state:
        _atomic_torch_save(output_dir / "trainable_final.pt", _trainable_state(parameters_by_name))
    _atomic_json(summary_path, result)
    _atomic_json(output_dir / "run_manifest.json", {**request, "state": "complete"})
    if progress_path.is_file():
        progress_path.unlink()
    print(
        f"[complete] {args.arm}/seed{args.training_seed}: "
        f"{100 * result['evaluation']['accuracy']:.2f}%"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--energy-basis", type=Path, required=True)
    parser.add_argument("--answer-conditioned-basis", type=Path, required=True)
    parser.add_argument("--parameter-aware-basis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=RETENTION_TRAINING_ARMS, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=53)
    parser.add_argument("--training-examples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-precision", default="auto")
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--save-trained-state", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
