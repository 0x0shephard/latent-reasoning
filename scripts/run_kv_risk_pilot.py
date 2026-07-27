"""Run the preregistered inference-only KV-compression risk pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.kv_risk_pilot import (
    DecodeRequest,
    PILOT_SCHEMA_VERSION,
    atomic_json,
    consolidate_records,
    deterministic_sample,
    extracted_answer,
    generate_one,
    load_candidate_dataset,
    record_identity,
    retention_from_condition,
    score_answer,
    select_screen_dataset,
    sha256_json,
)
from src.utils.config import load_config


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _dtype(value: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unsupported precision: {value}") from exc


def _load_model(cfg, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    model_cfg = cfg.model
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg.repo_id),
        revision=str(model_cfg.revision),
        token=token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_cfg.repo_id),
        revision=str(model_cfg.revision),
        token=token,
        torch_dtype=_dtype(str(model_cfg.precision), device),
        attn_implementation=str(model_cfg.attention_implementation),
        low_cpu_mem_usage=True,
    )
    model.to(device).eval()
    resolved_revision = str(
        getattr(model.config, "_commit_hash", None)
        or getattr(tokenizer, "_commit_hash", None)
        or model_cfg.revision
    )
    return model, tokenizer, resolved_revision


def _manifest_identity(
    *,
    cfg,
    stage: str,
    model_revision: str,
    examples: list[dict],
    conditions: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "stage": stage,
        "model_repo": str(cfg.model.repo_id),
        "model_revision": model_revision,
        "config_sha256": sha256_json(cfg.to_dict()),
        "prompt_instruction": str(cfg.prompt.instruction),
        "example_sha256": sha256_json(
            [
                {
                    "example_id": value["example_id"],
                    "question": value["question"],
                    "gold": value["gold"],
                }
                for value in examples
            ]
        ),
        "conditions": conditions,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "compression": dict(cfg.compression),
    }


def _condition_summary(condition_dir: Path, expected: int) -> dict:
    records = consolidate_records(condition_dir)
    if len(records) != expected:
        raise RuntimeError(
            f"{condition_dir.name} has {len(records)} records, expected {expected}"
        )
    return {
        "condition": condition_dir.name,
        "examples": len(records),
        "correct": sum(bool(record["correct"]) for record in records),
        "accuracy": float(np.mean([record["correct"] for record in records])),
        "median_generated_tokens": float(
            np.median([record["generated_tokens"] for record in records])
        ),
        "median_realized_generated_retention": float(
            np.median(
                [record["realized_generated_retention"] for record in records]
            )
        ),
        "total_retained_kv_token_steps": int(
            sum(record["retained_total_token_steps"] for record in records)
        ),
    }


def _run_condition(
    *,
    cfg,
    model,
    tokenizer,
    model_revision: str,
    device: torch.device,
    examples: list[dict],
    condition: str,
    condition_dir: Path,
    stage: str,
    max_new_tokens: int,
    early_entropy_tokens: int,
    temperature: float,
    top_p: float,
    stochastic_seed: int,
    started: float,
    max_seconds: float | None,
) -> bool:
    retention = retention_from_condition(condition)
    records_dir = condition_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    decoding_base = {
        "retention": retention,
        "max_new_tokens": max_new_tokens,
        "recent_window": int(cfg.compression.recent_window),
        "heavy_fraction": float(cfg.compression.heavy_fraction),
        "early_entropy_tokens": early_entropy_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stochastic_seed": stochastic_seed,
    }
    manifest_identity = _manifest_identity(
        cfg=cfg,
        stage=stage,
        model_revision=model_revision,
        examples=examples,
        conditions=[condition],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    request_sha256 = sha256_json({**manifest_identity, **decoding_base})
    manifest_path = condition_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise RuntimeError(
                f"refusing incompatible resume under {condition_dir}"
            )
    manifest = {
        **manifest_identity,
        "request_sha256": request_sha256,
        "condition": condition,
        "state": "running",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_examples": len(list(records_dir.glob("*.json"))),
        "expected_examples": len(examples),
    }
    atomic_json(manifest_path, manifest)

    for ordinal, example in enumerate(examples):
        output_path = records_dir / f"{ordinal:05d}.json"
        per_example_seed = (
            stochastic_seed * 1_000_003 + int(example["dataset_index"])
        )
        decoding = {
            **decoding_base,
            "sampling_seed": per_example_seed,
        }
        identity = record_identity(
            experiment=stage,
            condition=condition,
            example_id=str(example["example_id"]),
            model_revision=model_revision,
            decoding=decoding,
        )
        if output_path.is_file():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("record_sha256") != identity:
                raise RuntimeError(f"incompatible record at {output_path}")
            continue
        if max_seconds is not None and time.monotonic() - started >= 0.95 * max_seconds:
            manifest.update(
                {
                    "state": "resume_needed",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_examples": len(list(records_dir.glob("*.json"))),
                }
            )
            atomic_json(manifest_path, manifest)
            return False

        request = DecodeRequest(
            retention=retention,
            max_new_tokens=max_new_tokens,
            recent_window=int(cfg.compression.recent_window),
            heavy_fraction=float(cfg.compression.heavy_fraction),
            early_entropy_tokens=early_entropy_tokens,
            temperature=temperature,
            top_p=top_p,
            sampling_seed=per_example_seed,
        )
        result = generate_one(
            model,
            tokenizer,
            question=str(example["question"]),
            instruction=str(cfg.prompt.instruction),
            request=request,
            device=device,
        )
        correct = score_answer(
            str(result["generation"]),
            str(example["gold"]),
            str(example["grader"]),
        )
        record = {
            "schema_version": PILOT_SCHEMA_VERSION,
            "record_sha256": identity,
            "condition": condition,
            "example_id": example["example_id"],
            "dataset": example["dataset"],
            "dataset_index": example["dataset_index"],
            "question": example["question"],
            "question_tokens": result["prompt_tokens"],
            "gold": example["gold"],
            "grader": example["grader"],
            "level": example.get("level"),
            "prediction": extracted_answer(
                str(result["generation"]),
                str(example["grader"]),
            ),
            "correct": correct,
            **result,
        }
        atomic_json(output_path, record)
        manifest.update(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_examples": ordinal + 1,
            }
        )
        atomic_json(manifest_path, manifest)
        print(
            f"[{stage}] {condition} {ordinal + 1}/{len(examples)} "
            f"correct={int(correct)} tokens={result['generated_tokens']} "
            f"realized_r={result['realized_generated_retention']:.3f}",
            flush=True,
        )

    summary = _condition_summary(condition_dir, len(examples))
    atomic_json(condition_dir / "summary.json", summary)
    manifest.update(
        {
            "state": "complete",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_examples": len(examples),
            "summary": summary,
        }
    )
    atomic_json(manifest_path, manifest)
    return True


def _run_screen(args, cfg, model, tokenizer, model_revision, device) -> int:
    screen_root = args.output_dir / "screen"
    summaries: dict[str, dict] = {}
    screen_ids: dict[str, list[str]] = {}
    started = time.monotonic()
    for dataset_offset, (name, spec) in enumerate(cfg.datasets.items()):
        records = load_candidate_dataset(name, spec)
        configured_count = int(cfg.screen.examples_per_eligible_dataset)
        count = min(
            len(records),
            args.screen_examples or configured_count,
        )
        examples = deterministic_sample(
            records,
            count,
            seed=int(cfg.screen.seed) + dataset_offset,
        )
        screen_ids[name] = [str(value["example_id"]) for value in examples]
        condition_dir = screen_root / name / "full"
        completed = _run_condition(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            model_revision=model_revision,
            device=device,
            examples=examples,
            condition="full",
            condition_dir=condition_dir,
            stage=f"screen:{name}",
            max_new_tokens=args.max_new_tokens,
            early_entropy_tokens=int(cfg.analysis.early_entropy_tokens),
            temperature=0.0,
            top_p=1.0,
            stochastic_seed=0,
            started=started,
            max_seconds=args.max_seconds,
        )
        if not completed:
            return 42
        summary = json.loads(
            (condition_dir / "summary.json").read_text(encoding="utf-8")
        )
        summaries[name] = {
            **summary,
            "total_examples": len(records),
            "unused_examples": len(records) - len(examples),
            "screen_example_ids": screen_ids[name],
        }
    selection = select_screen_dataset(
        summaries,
        accuracy_min=float(cfg.screen.accuracy_min),
        accuracy_max=float(cfg.screen.accuracy_max),
        accuracy_midpoint=float(cfg.screen.accuracy_midpoint),
        minimum_median_generated_tokens=int(
            cfg.screen.minimum_median_generated_tokens
        ),
        pilot_examples=int(cfg.pilot.examples),
    )
    selection.update(
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_repo": str(cfg.model.repo_id),
            "model_revision": model_revision,
            "screen_seed": int(cfg.screen.seed),
            "selection_rule": (
                "eligible accuracy 0.60-0.85, median trace >=512, enough "
                "disjoint examples; closest accuracy to 0.725 wins"
            ),
        }
    )
    atomic_json(screen_root / "dataset_selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)
    if selection["status"] != "selected" and not args.allow_selection_failure:
        return 3
    return 0


def _selected_examples(args, cfg) -> tuple[str, list[dict]]:
    selection_path = args.output_dir / "screen" / "dataset_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            "dataset selection is missing; run --stage screen first"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = args.dataset_override or selection.get("selected_dataset")
    if not selected:
        raise RuntimeError("no dataset passed the preregistered selection gate")
    if selected not in cfg.datasets:
        raise ValueError(f"unknown selected dataset: {selected}")
    all_records = load_candidate_dataset(selected, cfg.datasets[selected])
    excluded = set(
        selection["datasets"][selected].get("screen_example_ids", [])
    )
    count = args.pilot_examples or int(cfg.pilot.examples)
    examples = deterministic_sample(
        all_records,
        count,
        seed=int(cfg.pilot.seed),
        excluded_ids=excluded,
    )
    return str(selected), examples


def _write_stage_manifest(
    path: Path,
    *,
    stage: str,
    selected: str,
    examples: list[dict],
    model_revision: str,
    conditions: list[str],
    state: str,
) -> None:
    atomic_json(
        path,
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "stage": stage,
            "state": state,
            "selected_dataset": selected,
            "model_revision": model_revision,
            "example_ids": [value["example_id"] for value in examples],
            "conditions": conditions,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _run_pilot(args, cfg, model, tokenizer, model_revision, device) -> int:
    selected, examples = _selected_examples(args, cfg)
    conditions = [str(value) for value in cfg.pilot.conditions]
    root = args.output_dir / "pilot"
    manifest_path = root / "run_manifest.json"
    _write_stage_manifest(
        manifest_path,
        stage="pilot",
        selected=selected,
        examples=examples,
        model_revision=model_revision,
        conditions=conditions,
        state="running",
    )
    started = time.monotonic()
    for condition in conditions:
        completed = _run_condition(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            model_revision=model_revision,
            device=device,
            examples=examples,
            condition=condition,
            condition_dir=root / "conditions" / condition,
            stage="pilot",
            max_new_tokens=args.max_new_tokens,
            early_entropy_tokens=int(cfg.analysis.early_entropy_tokens),
            temperature=0.0,
            top_p=1.0,
            stochastic_seed=0,
            started=started,
            max_seconds=args.max_seconds,
        )
        if not completed:
            _write_stage_manifest(
                manifest_path,
                stage="pilot",
                selected=selected,
                examples=examples,
                model_revision=model_revision,
                conditions=conditions,
                state="resume_needed",
            )
            return 42
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_stage_manifest(
        manifest_path,
        stage="pilot",
        selected=selected,
        examples=examples,
        model_revision=model_revision,
        conditions=conditions,
        state="complete",
    )
    return 0


def _run_stochastic(args, cfg, model, tokenizer, model_revision, device) -> int:
    selected, pilot_examples = _selected_examples(args, cfg)
    count = min(
        args.stochastic_examples or int(cfg.stochastic.examples),
        len(pilot_examples),
    )
    examples = pilot_examples[:count]
    retention = float(cfg.stochastic.retention)
    base_condition = f"retain_{retention:.2f}"
    seeds = [int(value) for value in cfg.stochastic.seeds]
    conditions = [
        condition
        for seed in seeds
        for condition in (f"full_seed{seed}", f"{base_condition}_seed{seed}")
    ]
    root = args.output_dir / "stochastic"
    manifest_path = root / "run_manifest.json"
    _write_stage_manifest(
        manifest_path,
        stage="stochastic",
        selected=selected,
        examples=examples,
        model_revision=model_revision,
        conditions=conditions,
        state="running",
    )
    started = time.monotonic()
    for seed in seeds:
        for condition in (f"full_seed{seed}", f"{base_condition}_seed{seed}"):
            completed = _run_condition(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                model_revision=model_revision,
                device=device,
                examples=examples,
                condition=condition,
                condition_dir=root / "conditions" / condition,
                stage="stochastic",
                max_new_tokens=args.max_new_tokens,
                early_entropy_tokens=int(cfg.analysis.early_entropy_tokens),
                temperature=float(cfg.stochastic.temperature),
                top_p=float(cfg.stochastic.top_p),
                stochastic_seed=seed,
                started=started,
                max_seconds=args.max_seconds,
            )
            if not completed:
                _write_stage_manifest(
                    manifest_path,
                    stage="stochastic",
                    selected=selected,
                    examples=examples,
                    model_revision=model_revision,
                    conditions=conditions,
                    state="resume_needed",
                )
                return 42
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_stage_manifest(
        manifest_path,
        stage="stochastic",
        selected=selected,
        examples=examples,
        model_revision=model_revision,
        conditions=conditions,
        state="complete",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("screen", "pilot", "stochastic"),
        required=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--screen-examples", type=int)
    parser.add_argument("--pilot-examples", type=int)
    parser.add_argument("--stochastic-examples", type=int)
    parser.add_argument("--dataset-override")
    parser.add_argument("--allow-selection-failure", action="store_true")
    return parser


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("enable a Kaggle GPU or pass --allow-cpu for a tiny smoke test")
    if args.max_new_tokens is None:
        args.max_new_tokens = int(cfg.model.max_new_tokens)
    torch.manual_seed(int(cfg.pilot.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.pilot.seed))
    model, tokenizer, model_revision = _load_model(cfg, device)
    print(
        f"[setup] model={cfg.model.repo_id} revision={model_revision} "
        f"device={device}",
        flush=True,
    )
    if args.stage == "screen":
        return _run_screen(
            args,
            cfg,
            model,
            tokenizer,
            model_revision,
            device,
        )
    if args.stage == "pilot":
        return _run_pilot(
            args,
            cfg,
            model,
            tokenizer,
            model_revision,
            device,
        )
    return _run_stochastic(
        args,
        cfg,
        model,
        tokenizer,
        model_revision,
        device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
