"""Test whether MATH-500 screening failed because the 2,048-token cap bound."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_kv_risk_pilot import (
    _run_condition,
    load_model_and_tokenizer,
    resolve_device,
)
from src.eval.kv_risk_pilot import (
    PILOT_SCHEMA_VERSION,
    atomic_json,
    deterministic_sample,
    load_candidate_dataset,
    load_records,
)
from src.utils.config import load_config


EXPERIMENT_SCHEMA_VERSION = 1


def find_reference_pilot_root(source: Path) -> Path:
    """Locate one scientifically unique completed third-run pilot tree."""
    source = source.resolve()
    direct = source / "screen/math500/full/run_manifest.json"
    candidates = [direct] if direct.is_file() else list(
        source.rglob("kv_compression_risk_pilot/screen/math500/full/run_manifest.json")
    )
    discovered = sorted(
        {
            path.parents[3].resolve()
            for path in candidates
            if path.is_file()
        },
        key=str,
    )
    valid: list[tuple[Path, tuple]] = []
    for root in discovered:
        manifest_path = root / "screen/math500/full/run_manifest.json"
        summary_path = root / "screen/math500/full/summary.json"
        selection_path = root / "screen/dataset_selection.json"
        predictions_path = root / "screen/math500/full/predictions.jsonl"
        records_dir = root / "screen/math500/full/records"
        if not all(
            path.is_file()
            for path in (
                manifest_path,
                summary_path,
                selection_path,
                predictions_path,
            )
        ):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            manifest.get("state") != "complete"
            or manifest.get("model_dtype") != "torch.float32"
            or int(manifest.get("max_new_tokens", -1)) != 2048
            or int(summary.get("examples", -1)) != 64
            or len(list(records_dir.glob("*.json"))) != 64
        ):
            continue
        fingerprint = (
            str(manifest.get("model_revision")),
            str(manifest.get("request_sha256")),
            str(manifest.get("example_sha256")),
            sha256_file(predictions_path),
        )
        valid.append((root, fingerprint))
    if not valid:
        raise RuntimeError(
            "no complete float32 third-run MATH-500 reference found "
            f"below {source}; discovered {discovered}"
        )
    fingerprints = {fingerprint for _, fingerprint in valid}
    if len(fingerprints) != 1:
        raise RuntimeError(
            "multiple conflicting third-run references were found; set "
            f"--reference-root explicitly after auditing them: {valid}"
        )
    roots = [root for root, _ in valid]
    return min(roots, key=lambda root: (len(root.parts), len(str(root)), str(root)))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_examples(
    reference_root: Path,
    *,
    cfg,
) -> tuple[list[dict], dict]:
    """Validate and reconstruct the exact MATH-500 third-run screen."""
    condition_dir = reference_root / "screen/math500/full"
    manifest_path = condition_dir / "run_manifest.json"
    summary_path = condition_dir / "summary.json"
    selection_path = reference_root / "screen/dataset_selection.json"
    for path in (manifest_path, summary_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required reference artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = load_records(condition_dir)
    expected = int(cfg.token_budget_diagnostic.expected_reference_examples)
    errors: list[str] = []
    if manifest.get("state") != "complete":
        errors.append("reference condition is not complete")
    if manifest.get("model_repo") != str(cfg.model.repo_id):
        errors.append("reference model repository differs")
    if manifest.get("model_revision") != str(cfg.model.revision):
        errors.append("reference model revision differs")
    if manifest.get("model_dtype") != "torch.float32":
        errors.append("reference dtype is not torch.float32")
    if int(manifest.get("max_new_tokens", -1)) != int(
        cfg.token_budget_diagnostic.reference_max_new_tokens
    ):
        errors.append("reference generation cap differs")
    if len(records) != expected:
        errors.append(
            f"reference contains {len(records)} records, expected {expected}"
        )
    if int(summary.get("examples", -1)) != expected:
        errors.append("reference summary count differs")
    selected = selection.get("datasets", {}).get("math500", {})
    if int(selected.get("examples", -1)) != expected:
        errors.append("selection manifest MATH-500 count differs")
    if errors:
        raise RuntimeError("; ".join(errors))

    examples: list[dict] = []
    for record in records:
        if record.get("dataset") != "math500":
            raise RuntimeError("reference record is not from MATH-500")
        examples.append(
            {
                "example_id": str(record["example_id"]),
                "dataset": "math500",
                "dataset_index": int(record["dataset_index"]),
                "question": str(record["question"]),
                "gold": str(record["gold"]),
                "grader": str(record["grader"]),
                "level": record.get("level"),
            }
        )
    examples.sort(key=lambda value: int(value["dataset_index"]))
    return examples, {
        "manifest": manifest,
        "summary": summary,
        "selection": selection,
        "condition_dir": str(condition_dir),
    }


def paired_accuracy_bootstrap(
    reference: list[dict],
    candidate: list[dict],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    reference_by_id = {str(row["example_id"]): row for row in reference}
    candidate_by_id = {str(row["example_id"]): row for row in candidate}
    if reference_by_id.keys() != candidate_by_id.keys():
        raise ValueError("reference and candidate contain different examples")
    ids = sorted(reference_by_id)
    deltas = np.asarray(
        [
            float(bool(candidate_by_id[key]["correct"]))
            - float(bool(reference_by_id[key]["correct"]))
            for key in ids
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(deltas), size=(samples, len(deltas)))
    estimates = deltas[draws].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def screen_gate(
    records: list[dict],
    *,
    cfg,
    total_examples: int,
    excluded_examples: int,
) -> dict:
    accuracy = float(np.mean([bool(row["correct"]) for row in records]))
    median_tokens = float(
        np.median([int(row["generated_tokens"]) for row in records])
    )
    length_limited = sum(row.get("finish_reason") == "length" for row in records)
    unused = total_examples - excluded_examples
    reasons: list[str] = []
    if not float(cfg.screen.accuracy_min) <= accuracy <= float(
        cfg.screen.accuracy_max
    ):
        reasons.append("accuracy_outside_preregistered_band")
    if median_tokens < int(cfg.screen.minimum_median_generated_tokens):
        reasons.append("reasoning_trace_too_short")
    if unused < int(cfg.pilot.examples):
        reasons.append("insufficient_disjoint_pilot_examples")
    return {
        "examples": len(records),
        "correct": sum(bool(row["correct"]) for row in records),
        "accuracy": accuracy,
        "median_generated_tokens": median_tokens,
        "length_limited_examples": length_limited,
        "length_limited_fraction": length_limited / len(records),
        "unused_examples": unused,
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
    }


def paired_diagnostic(
    reference: list[dict],
    candidate: list[dict],
    *,
    cfg,
    total_examples: int,
) -> dict:
    reference_by_id = {str(row["example_id"]): row for row in reference}
    candidate_by_id = {str(row["example_id"]): row for row in candidate}
    if reference_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired diagnostic contains different examples")
    ids = sorted(reference_by_id)
    reference_cap = int(cfg.token_budget_diagnostic.reference_max_new_tokens)
    candidate_cap = int(cfg.token_budget_diagnostic.candidate_max_new_tokens)
    reference_limited = {
        key
        for key in ids
        if reference_by_id[key].get("finish_reason") == "length"
        and int(reference_by_id[key]["generated_tokens"]) >= reference_cap
    }
    reference_limited_incorrect = {
        key
        for key in reference_limited
        if not bool(reference_by_id[key]["correct"])
    }
    corrected_limited = {
        key
        for key in reference_limited_incorrect
        if bool(candidate_by_id[key]["correct"])
    }
    regressed = {
        key
        for key in ids
        if bool(reference_by_id[key]["correct"])
        and not bool(candidate_by_id[key]["correct"])
    }
    improved = {
        key
        for key in ids
        if not bool(reference_by_id[key]["correct"])
        and bool(candidate_by_id[key]["correct"])
    }
    candidate_gate = screen_gate(
        candidate,
        cfg=cfg,
        total_examples=total_examples,
        excluded_examples=len(ids),
    )
    ci = paired_accuracy_bootstrap(
        reference,
        candidate,
        samples=int(cfg.token_budget_diagnostic.bootstrap_samples),
        seed=int(cfg.token_budget_diagnostic.bootstrap_seed),
    )
    if candidate_gate["eligible"]:
        decision = "candidate_cap_supported_pending_fresh_confirmation"
    elif (
        candidate_gate["length_limited_fraction"]
        >= float(cfg.token_budget_diagnostic.binding_length_limit_fraction)
    ):
        decision = "candidate_cap_still_binding"
    else:
        decision = "capacity_or_prompt_limit_more_likely"
    return {
        "reference": screen_gate(
            reference,
            cfg=cfg,
            total_examples=total_examples,
            excluded_examples=len(ids),
        ),
        "candidate": candidate_gate,
        "accuracy_delta": (
            candidate_gate["accuracy"]
            - float(np.mean([bool(row["correct"]) for row in reference]))
        ),
        "accuracy_delta_95ci": list(ci),
        "incorrect_to_correct": len(improved),
        "correct_to_incorrect": len(regressed),
        "reference_length_limited_examples": len(reference_limited),
        "reference_length_limited_incorrect_examples": len(
            reference_limited_incorrect
        ),
        "reference_length_limited_incorrect_to_correct": len(corrected_limited),
        "reference_length_limited_recovery_fraction": (
            None
            if not reference_limited_incorrect
            else len(corrected_limited) / len(reference_limited_incorrect)
        ),
        "decision": decision,
        "reference_max_new_tokens": reference_cap,
        "candidate_max_new_tokens": candidate_cap,
    }


def write_markdown(path: Path, report: dict) -> None:
    diagnostic = report["paired_diagnostic"]
    lines = [
        "# MATH-500 token-budget sensitivity screen",
        "",
        "## Outcome",
        "",
        f"**{report['decision'].replace('_', ' ')}**",
        "",
        "The same fixed MATH-500 questions were decoded at 2,048 and 4,096 "
        "tokens with the model, prompt, grader, precision, and greedy "
        "decoding held fixed.",
        "",
        "## Paired diagnostic",
        "",
        "| Metric | 2,048 tokens | 4,096 tokens |",
        "| --- | ---: | ---: |",
        (
            f"| Accuracy | {diagnostic['reference']['accuracy']:.4f} | "
            f"{diagnostic['candidate']['accuracy']:.4f} |"
        ),
        (
            f"| Median generated tokens | "
            f"{diagnostic['reference']['median_generated_tokens']:.1f} | "
            f"{diagnostic['candidate']['median_generated_tokens']:.1f} |"
        ),
        (
            f"| Length-limited fraction | "
            f"{diagnostic['reference']['length_limited_fraction']:.4f} | "
            f"{diagnostic['candidate']['length_limited_fraction']:.4f} |"
        ),
        "",
        (
            f"Paired accuracy delta: {diagnostic['accuracy_delta']:+.4f} "
            f"(95% bootstrap CI {diagnostic['accuracy_delta_95ci'][0]:+.4f} "
            f"to {diagnostic['accuracy_delta_95ci'][1]:+.4f})."
        ),
        "",
        (
            f"Incorrect-to-correct flips: {diagnostic['incorrect_to_correct']}. "
            f"Correct-to-incorrect flips: {diagnostic['correct_to_incorrect']}."
        ),
        "",
    ]
    confirmation = report.get("fresh_confirmation")
    if confirmation is not None:
        lines.extend(
            [
                "## Fresh disjoint confirmation",
                "",
                "| Accuracy | Median tokens | Length-limited | Eligible |",
                "| ---: | ---: | ---: | :---: |",
                (
                    f"| {confirmation['accuracy']:.4f} | "
                    f"{confirmation['median_generated_tokens']:.1f} | "
                    f"{confirmation['length_limited_fraction']:.4f} | "
                    f"{'yes' if confirmation['eligible'] else 'no'} |"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This screen diagnoses whether the original MATH-500 rejection was caused "
            "by the 2,048-token ceiling. It does not test KV compression risk. The "
            "150-question compression sweep is authorized only if the fresh "
            "confirmation passes the unchanged dataset-eligibility gate.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report.resolve()
    reference_root = find_reference_pilot_root(args.reference_root)
    examples, reference_audit = reference_examples(reference_root, cfg=cfg)
    reference_records = load_records(
        reference_root / "screen/math500/full"
    )
    all_math = load_candidate_dataset("math500", cfg.datasets.math500)

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("this diagnostic requires a Kaggle GPU")
    model, tokenizer, revision, dtype = load_model_and_tokenizer(cfg, device)
    if str(dtype) != "torch.float32":
        raise RuntimeError(f"expected torch.float32, resolved {dtype}")
    if revision != str(cfg.model.revision):
        raise RuntimeError(
            f"resolved model revision {revision} does not match {cfg.model.revision}"
        )
    started = time.monotonic()
    top_manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "state": "running",
        "model_repo": str(cfg.model.repo_id),
        "model_revision": revision,
        "model_dtype": str(dtype),
        "reference_root": str(reference_root),
        "reference_manifest": reference_audit["manifest"],
        "candidate_max_new_tokens": int(
            cfg.token_budget_diagnostic.candidate_max_new_tokens
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "run_manifest.json"
    atomic_json(manifest_path, top_manifest)

    diagnostic_dir = output_dir / (
        "diagnostic/cap_"
        f"{int(cfg.token_budget_diagnostic.candidate_max_new_tokens):06d}/full"
    )
    complete = _run_condition(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        model_revision=revision,
        device=device,
        examples=examples,
        condition="full",
        condition_dir=diagnostic_dir,
        stage="math_token_budget_diagnostic",
        max_new_tokens=int(cfg.token_budget_diagnostic.candidate_max_new_tokens),
        early_entropy_tokens=int(cfg.analysis.early_entropy_tokens),
        temperature=0.0,
        top_p=1.0,
        stochastic_seed=0,
        started=started,
        max_seconds=args.max_seconds,
    )
    if not complete:
        top_manifest.update(
            {
                "state": "resume_needed",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_json(manifest_path, top_manifest)
        return 42

    candidate_records = load_records(diagnostic_dir)
    diagnostic = paired_diagnostic(
        reference_records,
        candidate_records,
        cfg=cfg,
        total_examples=len(all_math),
    )
    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo": str(cfg.model.repo_id),
        "model_revision": revision,
        "model_dtype": str(dtype),
        "reference_root": str(reference_root),
        "paired_diagnostic": diagnostic,
        "fresh_confirmation": None,
        "decision": diagnostic["decision"],
    }

    if diagnostic["decision"] == "candidate_cap_supported_pending_fresh_confirmation":
        excluded = {str(row["example_id"]) for row in examples}
        confirmation_examples = deterministic_sample(
            all_math,
            int(cfg.token_budget_diagnostic.confirmation_examples),
            seed=int(cfg.token_budget_diagnostic.confirmation_seed),
            excluded_ids=excluded,
        )
        confirmation_dir = output_dir / (
            "confirmation/cap_"
            f"{int(cfg.token_budget_diagnostic.candidate_max_new_tokens):06d}/full"
        )
        complete = _run_condition(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            model_revision=revision,
            device=device,
            examples=confirmation_examples,
            condition="full",
            condition_dir=confirmation_dir,
            stage="math_token_budget_confirmation",
            max_new_tokens=int(
                cfg.token_budget_diagnostic.candidate_max_new_tokens
            ),
            early_entropy_tokens=int(cfg.analysis.early_entropy_tokens),
            temperature=0.0,
            top_p=1.0,
            stochastic_seed=0,
            started=started,
            max_seconds=args.max_seconds,
        )
        if not complete:
            top_manifest.update(
                {
                    "state": "resume_needed",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            atomic_json(manifest_path, top_manifest)
            return 42
        confirmation_records = load_records(confirmation_dir)
        confirmation = screen_gate(
            confirmation_records,
            cfg=cfg,
            total_examples=len(all_math),
            excluded_examples=len(examples) + len(confirmation_examples),
        )
        confirmation["screen_example_ids"] = [
            str(row["example_id"]) for row in confirmation_examples
        ]
        report["fresh_confirmation"] = confirmation
        if confirmation["eligible"]:
            report["decision"] = "math500_selected_at_4096_tokens"
            combined_ids = sorted(
                excluded
                | {
                    str(row["example_id"])
                    for row in confirmation_examples
                }
            )
            selection = {
                "schema_version": PILOT_SCHEMA_VERSION,
                "status": "selected",
                "selected_dataset": "math500",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_repo": str(cfg.model.repo_id),
                "model_revision": revision,
                "screen_seed": int(
                    cfg.token_budget_diagnostic.confirmation_seed
                ),
                "selection_rule": (
                    "fresh confirmation at 4096 tokens: accuracy 0.60-0.85, "
                    "median trace >=512, >=150 disjoint examples remain"
                ),
                "datasets": {
                    "math500": {
                        **confirmation,
                        "total_examples": len(all_math),
                        "screen_example_ids": combined_ids,
                    }
                },
            }
            atomic_json(output_dir / "screen/dataset_selection.json", selection)
        else:
            report["decision"] = "fresh_confirmation_failed"

    atomic_json(report_path, report)
    write_markdown(report_path.with_suffix(".md"), report)
    top_manifest.update(
        {
            "state": "complete",
            "decision": report["decision"],
            "report": str(report_path),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_json(manifest_path, top_manifest)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[complete] decision={report['decision']}", flush=True)
    print(f"[complete] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
