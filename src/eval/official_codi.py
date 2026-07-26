"""Evaluate the author-released CODI GPT-2 checkpoint without retraining.

The command is intentionally separate from ``src.eval.run_eval`` because the official
checkpoint does not use this repository's ``LatentCausalLM`` architecture or prompting.

Example:
    python -u -m src.eval.official_codi \
        --config configs/official_codi_gpt2.yaml \
        --limit 32
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import torch

from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set
from src.eval.official_codi_gate import build_accuracy_gate, official_answers_match
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


def select_device(requested: str = "auto") -> torch.device:
    normalized = requested.casefold()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _bounded_load_report(report) -> dict:
    payload = report.to_dict()
    for key in ("missing_keys", "unexpected_keys"):
        values = list(payload[key])
        payload[f"{key}_count"] = len(values)
        payload[key] = values[:50]
    return payload


def evaluate_official_codi(
    cfg,
    *,
    limit: int | None = None,
    datasets: list[str] | None = None,
    device_name: str = "auto",
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    device = select_device(device_name)
    dtype = resolve_torch_dtype(str(cfg.model.dtype), device)
    token = os.environ.get("HF_TOKEN") or None

    selected = datasets or list(cfg.eval.datasets)
    if not selected:
        raise ValueError("at least one evaluation dataset is required")
    unknown = sorted(set(selected) - set(cfg.eval.expected_counts))
    if unknown:
        raise ValueError(f"unknown official CODI datasets: {unknown}")

    data_cfg_path = Path(cfg.data_config)
    data_cfg = load_config(data_cfg_path)
    all_sets = {
        name: load_eval_set(name, data_cfg["eval"][name])
        for name in selected
    }
    for name, expected in cfg.eval.expected_counts.items():
        if name in selected and len(all_sets[name]) != int(expected):
            raise RuntimeError(
                f"{name} benchmark drift: loaded {len(all_sets[name])}, "
                f"expected {expected}"
            )

    effective_limit = limit if limit is not None else cfg.eval.get("limit")
    if effective_limit is not None and int(effective_limit) < 0:
        raise ValueError("limit must be non-negative; use 0 or omit it for full evaluation")
    cap = None if effective_limit in (None, 0) else int(effective_limit)

    if checkpoint_path is None:
        checkpoint = download_official_checkpoint(
            repo_id=str(cfg.checkpoint.repo_id),
            revision=str(cfg.checkpoint.revision),
            filename=str(cfg.checkpoint.filename),
            expected_sha256=str(cfg.checkpoint.sha256),
            token=token,
        )
    else:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"official checkpoint not found: {checkpoint}")

    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model),
        base_revision=str(cfg.model.base_revision),
        dtype=dtype,
        settings=cfg.model,
        token=token,
    )
    load_report = load_official_checkpoint(
        model,
        checkpoint,
        expected_sha256=str(cfg.checkpoint.sha256),
    )
    model.to(device=device, dtype=dtype).eval()

    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))

    root = Path(output_dir or cfg.output_dir)
    evaluation_scope = "full" if cap is None else f"limit_{cap:06d}"
    dataset_scope = "all" if selected == list(cfg.eval.datasets) else "-".join(selected)
    eval_dir = (
        root
        / "eval"
        / f"revision_{str(cfg.checkpoint.revision)[:8]}"
        / f"{evaluation_scope}_{dataset_scope}"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": str(cfg.run_name),
        "official_source": dict(cfg.official_source),
        "checkpoint": {
            **dict(cfg.checkpoint),
            "resolved_path": str(checkpoint),
        },
        "model": dict(cfg.model),
        "evaluation": {
            **dict(cfg.eval),
            "selected_datasets": selected,
            "effective_limit": cap,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
        },
        "data_config": str(data_cfg_path),
        "data_config_sha256": sha256_file(data_cfg_path),
        "checkpoint_load_report": _bounded_load_report(load_report),
        "packages": {
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "peft": _version("peft"),
            "datasets": _version("datasets"),
            "huggingface_hub": _version("huggingface-hub"),
        },
    }
    _atomic_json(eval_dir / "run_manifest.json", manifest)

    results: dict[str, float] = {}
    numeric_exact_match_results: dict[str, float] = {}
    evaluated_counts: dict[str, int] = {}
    for name in selected:
        examples = all_sets[name][:cap] if cap is not None else all_sets[name]
        print(f"[official-codi] evaluating {name}: {len(examples)} examples")
        generations = generate_official_codi(
            model,
            tokenizer,
            [example["question"] for example in examples],
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=int(cfg.eval.max_new_tokens),
            batch_size=int(cfg.eval.batch_size),
            device=device,
        )
        official_correctness = [
            official_answers_match(generation, example["gold"])
            for generation, example in zip(generations, examples)
        ]
        numeric_correctness = [
            answers_match(generation, example["gold"])
            for generation, example in zip(generations, examples)
        ]
        correct = sum(official_correctness)
        numeric_correct = sum(numeric_correctness)
        accuracy = correct / max(1, len(examples))
        numeric_accuracy = numeric_correct / max(1, len(examples))
        results[name] = accuracy
        numeric_exact_match_results[name] = numeric_accuracy
        evaluated_counts[name] = len(examples)
        print(
            f"[official-codi] {name:12s} official_acc={accuracy:.4f} "
            f"({correct}/{len(examples)}) numeric_exact={numeric_accuracy:.4f}"
        )
        _atomic_jsonl(
            eval_dir / f"{name}.jsonl",
            (
                {
                    "question": example["question"],
                    "gold": str(example["gold"]),
                    "generation": generation,
                    # ``correct`` remains the released scorer for compatibility with the
                    # paper gate and generic paired-analysis tooling.
                    "correct": official_correct,
                    "official_correct": official_correct,
                    "numeric_exact_match_correct": numeric_correct,
                }
                for example, generation, official_correct, numeric_correct in zip(
                    examples,
                    generations,
                    official_correctness,
                    numeric_correctness,
                )
            ),
        )

    macro = sum(results.values()) / max(1, len(results))
    gate = build_accuracy_gate(
        results=results,
        evaluated_counts=evaluated_counts,
        expected_counts=cfg.eval.expected_counts,
        published_accuracy=cfg.accuracy_gate.published_accuracy,
        primary_dataset=str(cfg.accuracy_gate.primary_dataset),
        absolute_tolerance=float(cfg.accuracy_gate.absolute_tolerance),
    )
    summary = {
        "checkpoint_repo": str(cfg.checkpoint.repo_id),
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "metric": "official_last_number_exact_match",
        "datasets": results,
        "numeric_exact_match_datasets": numeric_exact_match_results,
        "evaluated_counts": evaluated_counts,
        "macro_mean": macro,
        "accuracy_gate": gate,
    }
    _atomic_json(eval_dir / "summary.json", summary)
    print(f"[official-codi] macro_mean={macro:.4f}")
    print(f"[official-codi] gate={gate['status']}")
    print(f"[official-codi] wrote durable outputs to {eval_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the author-released CODI GPT-2 checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Examples per dataset. Use 0 or omit for the complete benchmarks.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated subset such as gsm8k,svamp.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Config override in dotted.key=value form.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    selected = (
        [value.strip() for value in args.datasets.split(",") if value.strip()]
        if args.datasets
        else None
    )
    evaluate_official_codi(
        cfg,
        limit=args.limit,
        datasets=selected,
        device_name=args.device,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
