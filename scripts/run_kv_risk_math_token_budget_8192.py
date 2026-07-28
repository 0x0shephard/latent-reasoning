"""Run the final targeted 8,192-token MATH-500 eligibility diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_kv_risk_math_token_budget import (
    paired_accuracy_bootstrap,
    screen_gate,
    sha256_file,
)
from scripts.run_kv_risk_pilot import (
    _run_condition,
    load_model_and_tokenizer,
    resolve_device,
)
from src.eval.kv_risk_pilot import (
    PILOT_SCHEMA_VERSION,
    atomic_json,
    atomic_jsonl,
    deterministic_sample,
    load_candidate_dataset,
    load_records,
)
from src.utils.config import load_config


EXPERIMENT_SCHEMA_VERSION = 1


def find_4096_reference_root(source: Path) -> Path:
    """Find one scientifically unique complete 4,096-token output tree."""
    source = source.resolve()
    relative_manifest = "diagnostic/cap_004096/full/run_manifest.json"
    direct = source / relative_manifest
    candidates = [direct] if direct.is_file() else list(
        source.rglob(
            "kv_risk_math_token_budget/"
            "diagnostic/cap_004096/full/run_manifest.json"
        )
    )
    discovered = sorted(
        {path.parents[3].resolve() for path in candidates if path.is_file()},
        key=str,
    )
    valid: list[tuple[Path, tuple]] = []
    for root in discovered:
        top_manifest_path = root / "run_manifest.json"
        condition_dir = root / "diagnostic/cap_004096/full"
        manifest_path = condition_dir / "run_manifest.json"
        summary_path = condition_dir / "summary.json"
        predictions_path = condition_dir / "predictions.jsonl"
        records_dir = condition_dir / "records"
        if not all(
            path.is_file()
            for path in (
                top_manifest_path,
                manifest_path,
                summary_path,
                predictions_path,
            )
        ):
            continue
        top_manifest = json.loads(
            top_manifest_path.read_text(encoding="utf-8")
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            top_manifest.get("state") != "complete"
            or top_manifest.get("decision") != "candidate_cap_still_binding"
            or manifest.get("state") != "complete"
            or manifest.get("model_dtype") != "torch.float32"
            or int(manifest.get("max_new_tokens", -1)) != 4096
            or int(summary.get("examples", -1)) != 64
            or int(summary.get("correct", -1)) != 37
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
            "no complete 4,096-token MATH-500 reference found below "
            f"{source}; discovered {discovered}"
        )
    fingerprints = {fingerprint for _, fingerprint in valid}
    if len(fingerprints) != 1:
        raise RuntimeError(
            "conflicting 4,096-token references found; audit and set "
            f"--reference-root explicitly: {valid}"
        )
    roots = [root for root, _ in valid]
    return min(roots, key=lambda root: (len(root.parts), len(str(root)), str(root)))


def validate_reference(reference_root: Path, *, cfg) -> tuple[list[dict], dict]:
    condition_dir = reference_root / "diagnostic/cap_004096/full"
    manifest = json.loads(
        (condition_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (condition_dir / "summary.json").read_text(encoding="utf-8")
    )
    records = load_records(condition_dir)
    expected_examples = int(
        cfg.token_budget_extension.expected_reference_examples
    )
    expected_correct = int(
        cfg.token_budget_extension.expected_reference_correct
    )
    expected_limited = int(
        cfg.token_budget_extension.expected_reference_length_limited
    )
    errors: list[str] = []
    if manifest.get("model_repo") != str(cfg.model.repo_id):
        errors.append("model repository differs")
    if manifest.get("model_revision") != str(cfg.model.revision):
        errors.append("model revision differs")
    if manifest.get("model_dtype") != "torch.float32":
        errors.append("reference dtype differs")
    if int(manifest.get("max_new_tokens", -1)) != int(
        cfg.token_budget_extension.reference_max_new_tokens
    ):
        errors.append("reference token cap differs")
    if len(records) != expected_examples:
        errors.append("reference record count differs")
    if int(summary.get("correct", -1)) != expected_correct:
        errors.append("reference correct count differs")
    limited = [row for row in records if row.get("finish_reason") == "length"]
    if len(limited) != expected_limited:
        errors.append("reference length-limited count differs")
    if any(bool(row.get("degenerate_generation")) for row in records):
        errors.append("reference contains degenerate generations")
    if errors:
        raise RuntimeError("; ".join(errors))
    return records, {
        "manifest": manifest,
        "summary": summary,
        "predictions_sha256": sha256_file(
            condition_dir / "predictions.jsonl"
        ),
        "length_limited_ids": sorted(
            str(row["example_id"]) for row in limited
        ),
    }


def example_from_record(record: dict) -> dict:
    return {
        "example_id": str(record["example_id"]),
        "dataset": "math500",
        "dataset_index": int(record["dataset_index"]),
        "question": str(record["question"]),
        "gold": str(record["gold"]),
        "grader": str(record["grader"]),
        "level": record.get("level"),
    }


def compose_8192_records(
    reference_records: list[dict],
    extension_records: list[dict],
    *,
    reference_cap: int,
) -> tuple[list[dict], dict]:
    """Combine unchanged EOS rows with extended rows after prefix verification."""
    extension = {
        str(row["example_id"]): row for row in extension_records
    }
    expected = {
        str(row["example_id"])
        for row in reference_records
        if row.get("finish_reason") == "length"
    }
    if set(extension) != expected:
        raise ValueError(
            "extension IDs differ from the 4,096-token length-limited set"
        )
    combined: list[dict] = []
    prefix_matches = 0
    for reference in reference_records:
        key = str(reference["example_id"])
        if key not in extension:
            combined.append(dict(reference))
            continue
        candidate = extension[key]
        reference_ids = [
            int(value) for value in reference["generated_token_ids"]
        ]
        candidate_ids = [
            int(value) for value in candidate["generated_token_ids"]
        ]
        if len(reference_ids) != reference_cap:
            raise ValueError(
                f"{key} was length-limited but has {len(reference_ids)} tokens"
            )
        if candidate_ids[:reference_cap] != reference_ids:
            raise ValueError(
                f"8,192-token output does not preserve the 4,096-token prefix: {key}"
            )
        prefix_matches += 1
        combined.append(dict(candidate))
    return combined, {
        "reference_examples": len(reference_records),
        "extended_examples": len(extension_records),
        "reused_eos_examples": len(reference_records) - len(extension_records),
        "exact_prefix_matches": prefix_matches,
        "all_extended_prefixes_exact": prefix_matches == len(extension_records),
    }


def extension_diagnostic(
    reference: list[dict],
    candidate: list[dict],
    *,
    cfg,
    total_examples: int,
    composition_audit: dict,
) -> dict:
    reference_by_id = {str(row["example_id"]): row for row in reference}
    candidate_by_id = {str(row["example_id"]): row for row in candidate}
    if reference_by_id.keys() != candidate_by_id.keys():
        raise ValueError("reference and composed candidate IDs differ")
    ids = sorted(reference_by_id)
    improved = {
        key for key in ids
        if not bool(reference_by_id[key]["correct"])
        and bool(candidate_by_id[key]["correct"])
    }
    regressed = {
        key for key in ids
        if bool(reference_by_id[key]["correct"])
        and not bool(candidate_by_id[key]["correct"])
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
        samples=int(cfg.token_budget_extension.bootstrap_samples),
        seed=int(cfg.token_budget_extension.bootstrap_seed),
    )
    if candidate_gate["eligible"]:
        decision = "candidate_cap_supported_pending_fresh_confirmation"
    elif (
        candidate_gate["length_limited_fraction"]
        >= float(cfg.token_budget_extension.binding_length_limit_fraction)
    ):
        decision = "final_cap_still_binding_close_1p5b_configuration"
    else:
        decision = "final_cap_failed_eligibility_close_1p5b_configuration"
    reference_gate = screen_gate(
        reference,
        cfg=cfg,
        total_examples=total_examples,
        excluded_examples=len(ids),
    )
    return {
        "reference": reference_gate,
        "candidate": candidate_gate,
        "accuracy_delta": (
            candidate_gate["accuracy"] - reference_gate["accuracy"]
        ),
        "accuracy_delta_95ci": list(ci),
        "incorrect_to_correct": len(improved),
        "correct_to_incorrect": len(regressed),
        "composition_audit": composition_audit,
        "decision": decision,
        "reference_max_new_tokens": int(
            cfg.token_budget_extension.reference_max_new_tokens
        ),
        "candidate_max_new_tokens": int(
            cfg.token_budget_extension.candidate_max_new_tokens
        ),
    }


def write_markdown(path: Path, report: dict) -> None:
    diagnostic = report["paired_extension"]
    reference = diagnostic["reference"]
    candidate = diagnostic["candidate"]
    audit = diagnostic["composition_audit"]
    lines = [
        "# Final MATH-500 token-budget eligibility screen",
        "",
        "## Outcome",
        "",
        f"**{report['decision'].replace('_', ' ')}**",
        "",
        "Only the 4,096-token length-limited questions were regenerated at "
        "8,192 tokens. Previously terminated greedy outputs were reused.",
        "",
        "## Paired extension",
        "",
        "| Metric | 4,096 tokens | 8,192-token composition |",
        "| --- | ---: | ---: |",
        f"| Accuracy | {reference['accuracy']:.4f} | {candidate['accuracy']:.4f} |",
        (
            f"| Median generated tokens | "
            f"{reference['median_generated_tokens']:.1f} | "
            f"{candidate['median_generated_tokens']:.1f} |"
        ),
        (
            f"| Length-limited fraction | "
            f"{reference['length_limited_fraction']:.4f} | "
            f"{candidate['length_limited_fraction']:.4f} |"
        ),
        "",
        (
            f"Extended examples: {audit['extended_examples']}. Reused completed "
            f"examples: {audit['reused_eos_examples']}. Exact prefix checks: "
            f"{audit['exact_prefix_matches']}/{audit['extended_examples']}."
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
            "This is the final token-cap diagnostic for the 1.5B configuration. "
            "It does not test KV-compression risk. The compression sweep is "
            "authorized only if the fresh disjoint confirmation passes the "
            "unchanged dataset gate.",
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
    if bool(cfg.token_budget_extension.further_cap_escalation_allowed):
        raise RuntimeError("this must remain the final cap diagnostic")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report.resolve()
    reference_root = find_4096_reference_root(args.reference_root)
    reference_records, reference_audit = validate_reference(
        reference_root,
        cfg=cfg,
    )
    limited_examples = [
        example_from_record(row)
        for row in reference_records
        if row.get("finish_reason") == "length"
    ]
    limited_examples.sort(key=lambda row: int(row["dataset_index"]))
    all_math = load_candidate_dataset("math500", cfg.datasets.math500)

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("this diagnostic requires a Kaggle GPU")
    model, tokenizer, revision, dtype = load_model_and_tokenizer(cfg, device)
    if str(dtype) != "torch.float32":
        raise RuntimeError(f"expected torch.float32, resolved {dtype}")
    if revision != str(cfg.model.revision):
        raise RuntimeError(
            f"resolved revision {revision} does not match {cfg.model.revision}"
        )

    started = time.monotonic()
    manifest_path = output_dir / "run_manifest.json"
    top_manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "state": "running",
        "model_repo": str(cfg.model.repo_id),
        "model_revision": revision,
        "model_dtype": str(dtype),
        "reference_root": str(reference_root),
        "reference_audit": reference_audit,
        "candidate_max_new_tokens": int(
            cfg.token_budget_extension.candidate_max_new_tokens
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest_path, top_manifest)

    extension_dir = output_dir / "extension/cap_008192/full"
    complete = _run_condition(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        model_revision=revision,
        device=device,
        examples=limited_examples,
        condition="full",
        condition_dir=extension_dir,
        stage="math_token_budget_extension_8192",
        max_new_tokens=int(cfg.token_budget_extension.candidate_max_new_tokens),
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

    combined_records, composition_audit = compose_8192_records(
        reference_records,
        load_records(extension_dir),
        reference_cap=int(
            cfg.token_budget_extension.reference_max_new_tokens
        ),
    )
    combined_dir = output_dir / "combined/cap_008192"
    atomic_jsonl(combined_dir / "predictions.jsonl", combined_records)
    diagnostic = extension_diagnostic(
        reference_records,
        combined_records,
        cfg=cfg,
        total_examples=len(all_math),
        composition_audit=composition_audit,
    )
    atomic_json(
        combined_dir / "summary.json",
        {
            **diagnostic["candidate"],
            "composition_audit": composition_audit,
        },
    )
    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo": str(cfg.model.repo_id),
        "model_revision": revision,
        "model_dtype": str(dtype),
        "reference_root": str(reference_root),
        "paired_extension": diagnostic,
        "fresh_confirmation": None,
        "decision": diagnostic["decision"],
    }
    atomic_json(report_path, report)
    write_markdown(report_path.with_suffix(".md"), report)

    if diagnostic["decision"] == "candidate_cap_supported_pending_fresh_confirmation":
        excluded = {str(row["example_id"]) for row in reference_records}
        confirmation_examples = deterministic_sample(
            all_math,
            int(cfg.token_budget_extension.confirmation_examples),
            seed=int(cfg.token_budget_extension.confirmation_seed),
            excluded_ids=excluded,
        )
        confirmation_dir = output_dir / "confirmation/cap_008192/full"
        complete = _run_condition(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            model_revision=revision,
            device=device,
            examples=confirmation_examples,
            condition="full",
            condition_dir=confirmation_dir,
            stage="math_token_budget_confirmation_8192",
            max_new_tokens=int(
                cfg.token_budget_extension.candidate_max_new_tokens
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
                    "decision": report["decision"],
                    "report": str(report_path),
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
            excluded_examples=len(reference_records)
            + len(confirmation_examples),
        )
        confirmation["screen_example_ids"] = [
            str(row["example_id"]) for row in confirmation_examples
        ]
        report["fresh_confirmation"] = confirmation
        if confirmation["eligible"]:
            report["decision"] = "math500_selected_at_8192_tokens"
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
                    cfg.token_budget_extension.confirmation_seed
                ),
                "selection_rule": (
                    "fresh confirmation at 8192 tokens: accuracy 0.60-0.85, "
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
            report["decision"] = "fresh_confirmation_failed_close_1p5b_configuration"
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
