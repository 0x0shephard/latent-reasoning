"""Run full-GSM8K causal KV-subspace interventions on official CODI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_official_codi_kv_subspaces import _verify_reproduction_gate
from scripts.export_official_codi_student_subspaces import (
    STUDENT_SUBSPACE_ARTIFACT_SCHEMA_VERSION,
)
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import build_accuracy_gate, official_answers_match
from src.mech.official_codi_kv_intervention import (
    OfficialCODIKVSubspaceIntervention,
    build_intervention_specs,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


CAUSAL_EVALUATION_SCHEMA_VERSION = 1


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


def _parse_positions(value: str) -> list[int]:
    try:
        positions = sorted(
            {int(item.strip()) for item in value.split(",") if item.strip()}
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "positions must be comma-separated integers"
        ) from exc
    if not positions:
        raise argparse.ArgumentTypeError("at least one position is required")
    return positions


def _condition_priority(name: str) -> tuple:
    # Finish the preregistered positions first so a disconnect preserves the primary
    # comparison even if the optional full sweep must be resumed later.
    scope_order = {
        "p4": 0,
        "p5": 1,
        "all": 2,
        "p0": 3,
        "p1": 4,
        "p2": 5,
        "p3": 6,
    }
    scope = name.rsplit("_", 1)[-1]
    basis_order = 0 if "_learned_" in name else 1
    mode_order = 0 if name.startswith("retain_") else 1
    return (scope_order.get(scope, 99), mode_order, basis_order, name)


def _completed_condition(
    condition_dir: Path,
    *,
    expected_count: int,
    checkpoint_revision: str,
    artifact_sha256: str,
) -> dict | None:
    summary_path = condition_dir / "summary.json"
    predictions_path = condition_dir / "gsm8k.jsonl"
    if not summary_path.is_file() or not predictions_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        str(summary.get("condition")) != condition_dir.name
        or int(summary.get("evaluated_count", -1)) != expected_count
        or str(summary.get("checkpoint_revision")) != checkpoint_revision
        or str(summary.get("subspace_artifact_sha256")) != artifact_sha256
    ):
        return None
    with predictions_path.open(encoding="utf-8") as handle:
        if sum(1 for _ in handle) != expected_count:
            return None
    return summary


def _resolve_reproduction_gate(
    reproduction_summary: Path | None,
    *,
    source: dict,
    cfg,
) -> dict:
    if reproduction_summary is not None:
        return _verify_reproduction_gate(reproduction_summary, cfg)
    embedded_gate = source.get("reproduction_gate")
    if (
        not isinstance(embedded_gate, dict)
        or embedded_gate.get("status") != "passed"
    ):
        raise RuntimeError(
            "no external reproduction summary was supplied and the subspace "
            "artifact does not contain a passed embedded reproduction gate"
        )
    return {
        **embedded_gate,
        "path": "embedded in completed calibration statistics",
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "full official CODI causal evaluation requires CUDA; use --allow-cpu "
            "only for a tiny smoke test"
        )
    dtype = resolve_torch_dtype(args.precision, device)
    artifact_path = args.subspace_artifact
    if not artifact_path.is_file():
        raise FileNotFoundError(f"subspace artifact does not exist: {artifact_path}")
    artifact_sha256 = sha256_file(artifact_path)
    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        int(artifact.get("schema_version", -1))
        != STUDENT_SUBSPACE_ARTIFACT_SCHEMA_VERSION
    ):
        raise RuntimeError("subspace artifact uses an incompatible schema")
    if int(artifact.get("rank", -1)) != args.rank:
        raise RuntimeError("subspace artifact rank does not match this request")
    source = artifact.get("source", {})
    if str(source.get("checkpoint_revision")) != str(cfg.checkpoint.revision):
        raise RuntimeError("subspace artifact uses a different checkpoint revision")
    if str(source.get("checkpoint_sha256")) != str(cfg.checkpoint.sha256):
        raise RuntimeError("subspace artifact uses a different checkpoint")
    reproduction = _resolve_reproduction_gate(
        args.reproduction_summary,
        source=source,
        cfg=cfg,
    )
    latent_positions = int(cfg.eval.latent_iterations)
    for kind in ("key", "value"):
        basis = artifact["kinds"][kind]["learned_basis"]
        if basis.ndim != 5 or int(basis.shape[2]) != latent_positions:
            raise RuntimeError(f"{kind} subspace trajectory shape is incompatible")

    token = os.environ.get("HF_TOKEN") or None
    checkpoint = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else download_official_checkpoint(
            repo_id=str(cfg.checkpoint.repo_id),
            revision=str(cfg.checkpoint.revision),
            filename=str(cfg.checkpoint.filename),
            expected_sha256=str(cfg.checkpoint.sha256),
            token=token,
        )
    )
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
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_cfg = load_config(Path(cfg.data_config))
    examples = load_eval_set("gsm8k", data_cfg["eval"]["gsm8k"])
    expected_full = int(cfg.eval.expected_counts.gsm8k)
    if len(examples) != expected_full:
        raise RuntimeError(
            f"GSM8K benchmark drift: loaded {len(examples)}, expected {expected_full}"
        )
    cap = None if args.limit in (None, 0) else int(args.limit)
    if cap is not None:
        examples = examples[:cap]
    expected_count = len(examples)
    questions = [example["question"] for example in examples]

    specs = build_intervention_specs(
        positions=args.positions,
        include_all=args.include_all,
        latent_positions=latent_positions,
    )
    specs = sorted(specs, key=lambda spec: _condition_priority(spec.name))
    conditions = ["baseline", *[spec.name for spec in specs]]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": CAUSAL_EVALUATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "running",
        "analysis": "official_codi_student_kv_causal_intervention",
        "scientific_scope": (
            "inference-only centered rank-four interventions on newly appended "
            "latent K/V cache entries"
        ),
        "reproduction_gate": reproduction,
        "checkpoint_repo": str(cfg.checkpoint.repo_id),
        "checkpoint_revision": str(cfg.checkpoint.revision),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "checkpoint_matched_numel_fraction": load_report.matched_numel_fraction,
        "subspace_artifact": str(artifact_path),
        "subspace_artifact_sha256": artifact_sha256,
        "subspace_source": source,
        "rank": args.rank,
        "random_seed": artifact.get("random_seed"),
        "positions": args.positions,
        "include_all": args.include_all,
        "conditions": conditions,
        "evaluated_count": expected_count,
        "full_gsm8k": cap is None,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "precision": args.precision,
        "device": str(device),
        "intervention_timing": (
            "immediately after each selected latent KV entry is appended and before "
            "later latent steps or answer decoding consume the cache"
        ),
        "completed_conditions": [],
    }
    manifest_path = output_dir / "run_manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    if existing_manifest is not None:
        identity_keys = (
            "checkpoint_revision",
            "checkpoint_sha256",
            "subspace_artifact_sha256",
            "rank",
            "positions",
            "include_all",
            "conditions",
            "evaluated_count",
            "batch_size",
            "max_new_tokens",
            "precision",
        )
        if any(
            existing_manifest.get(key) != manifest.get(key)
            for key in identity_keys
        ):
            raise RuntimeError(
                "existing causal evaluation uses a different scientific contract"
            )
    _atomic_json(manifest_path, manifest)

    summaries = {}
    spec_by_name = {spec.name: spec for spec in specs}
    for condition in conditions:
        condition_dir = output_dir / condition
        completed = _completed_condition(
            condition_dir,
            expected_count=expected_count,
            checkpoint_revision=str(cfg.checkpoint.revision),
            artifact_sha256=artifact_sha256,
        )
        if completed is not None:
            print(f"[resume] verified {condition}: {completed['accuracy']:.4f}")
            summaries[condition] = completed
            manifest["completed_conditions"].append(condition)
            continue
        print(f"[causal-eval] {condition}: {expected_count} GSM8K examples")
        intervention = (
            None
            if condition == "baseline"
            else OfficialCODIKVSubspaceIntervention(
                artifact,
                spec_by_name[condition],
                device=device,
            )
        )
        generations = generate_official_codi(
            model,
            tokenizer,
            questions,
            latent_iterations=latent_positions,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            device=device,
            kv_intervention=intervention,
        )
        official_correctness = [
            official_answers_match(generation, example["gold"])
            for generation, example in zip(generations, examples)
        ]
        numeric_correctness = [
            answers_match(generation, example["gold"])
            for generation, example in zip(generations, examples)
        ]
        correct = int(sum(official_correctness))
        numeric_correct = int(sum(numeric_correctness))
        accuracy = correct / expected_count
        numeric_accuracy = numeric_correct / expected_count
        _atomic_jsonl(
            condition_dir / "gsm8k.jsonl",
            (
                {
                    "example_index": index,
                    "question": example["question"],
                    "gold": str(example["gold"]),
                    "generation": generation,
                    "correct": official_correct,
                    "official_correct": official_correct,
                    "numeric_exact_match_correct": numeric_correctness[index],
                }
                for index, (example, generation, official_correct) in enumerate(
                    zip(examples, generations, official_correctness)
                )
            ),
        )
        summary = {
            "schema_version": CAUSAL_EVALUATION_SCHEMA_VERSION,
            "condition": condition,
            "checkpoint_revision": str(cfg.checkpoint.revision),
            "subspace_artifact_sha256": artifact_sha256,
            "evaluated_count": expected_count,
            "correct": correct,
            "accuracy": accuracy,
            "numeric_exact_match_correct": numeric_correct,
            "numeric_exact_match_accuracy": numeric_accuracy,
            "spec": (
                None
                if intervention is None
                else {
                    "mode": intervention.spec.mode,
                    "basis_kind": intervention.spec.basis_kind,
                    "positions": sorted(intervention.spec.positions),
                }
            ),
        }
        if condition == "baseline":
            summary["accuracy_gate"] = build_accuracy_gate(
                results={"gsm8k": accuracy},
                evaluated_counts={"gsm8k": expected_count},
                expected_counts={"gsm8k": expected_full},
                published_accuracy=cfg.accuracy_gate.published_accuracy,
                primary_dataset="gsm8k",
                absolute_tolerance=float(cfg.accuracy_gate.absolute_tolerance),
            )
            if cap is None and summary["accuracy_gate"]["status"] != "passed":
                raise RuntimeError(
                    "causal baseline failed the official GSM8K reproduction gate"
                )
        _atomic_json(condition_dir / "summary.json", summary)
        summaries[condition] = summary
        manifest["completed_conditions"].append(condition)
        _atomic_json(manifest_path, manifest)
        print(
            f"[causal-eval] {condition:28s} acc={accuracy:.4f} "
            f"({correct}/{expected_count})"
        )
        del intervention, generations
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest["state"] = "complete"
    manifest["completed_conditions"] = conditions
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        output_dir / "summary.json",
        {
            "schema_version": CAUSAL_EVALUATION_SCHEMA_VERSION,
            "state": "complete",
            "checkpoint_revision": str(cfg.checkpoint.revision),
            "subspace_artifact_sha256": artifact_sha256,
            "evaluated_count": expected_count,
            "conditions": {
                condition: {
                    "accuracy": summary["accuracy"],
                    "correct": summary["correct"],
                }
                for condition, summary in summaries.items()
            },
        },
    )
    print(f"[complete] wrote causal evaluations to {output_dir}")
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate centered learned and energy-matched random rank-four student "
            "KV interventions on official CODI GSM8K."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--reproduction-summary", type=Path)
    parser.add_argument("--subspace-artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--positions", type=_parse_positions, default=list(range(6)))
    parser.add_argument("--include-all", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="cuda",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.rank <= 0:
        parser.error("rank must be positive")
    if args.limit < 0:
        parser.error("limit must be non-negative")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
