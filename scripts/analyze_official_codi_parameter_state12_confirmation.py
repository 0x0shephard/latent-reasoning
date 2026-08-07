"""Analyze all paired arms in the parameter-aware state-12 confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_parameter_state12_confirmation_analysis import (
    analyze_parameter_state12_confirmation,
)
from src.mech.endpoint_state12_confirmation import STATE12_CONFIRMATION_CONTRACT


def _load_run(summary_path: Path) -> tuple[dict, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract") != STATE12_CONFIRMATION_CONTRACT:
        raise RuntimeError(f"wrong state-12 confirmation contract: {summary_path}")
    if summary.get("state") != "complete":
        raise RuntimeError(f"incomplete state-12 confirmation run: {summary_path}")
    records = [
        json.loads(line)
        for line in (summary_path.parent / summary["predictions_file"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(records) != int(summary["evaluated_examples"]):
        raise RuntimeError(f"prediction count mismatch: {summary_path}")
    identity = [
        (int(record["index"]), str(record["question"]), str(record["gold"]))
        for record in records
    ]
    return {
        "arm": summary["arm"],
        "spec": summary.get("spec"),
        "intervention_diagnostics": summary.get("intervention_diagnostics"),
        "native_reproduction_accuracy": summary.get("reproduction_gate", {}).get(
            "gsm8k_accuracy"
        ),
        "correctness": [bool(record["correct"]) for record in records],
        "endpoint_reached": [
            bool(record["answer_cue_endpoint_reached"]) for record in records
        ],
    }, {
        "identity": identity,
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "activation_stats_sha256": summary.get("activation_stats_sha256"),
        "basis_sources": summary.get("basis_sources"),
        "random_replicates": summary.get("random_replicates"),
        "random_seed": summary.get("random_seed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-replicates", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--minimum-endpoint-coverage", type=float, default=0.95)
    parser.add_argument("--maximum-rms-ratio-deviation", type=float, default=0.10)
    args = parser.parse_args()
    loaded = [_load_run(path) for path in sorted(args.runs_root.rglob("summary.json"))]
    if not loaded:
        raise RuntimeError(f"no completed confirmation summaries under {args.runs_root}")
    runs = [value[0] for value in loaded]
    invariants = [value[1] for value in loaded]
    reference = invariants[0]
    for invariant in invariants[1:]:
        if invariant != reference:
            raise RuntimeError("confirmation arms do not share exact paired inputs/artifacts")
    family_counts = {}
    for run in runs:
        family = "baseline" if run["spec"] is None else run["spec"]["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    expected = {
        "baseline": 1,
        "selected_primary": 1,
        "matched_random_primary": args.random_replicates,
    }
    if family_counts != expected:
        raise RuntimeError(
            f"analysis requires the registered 502-arm family, found {family_counts}"
        )
    report = analyze_parameter_state12_confirmation(
        runs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        alpha=args.alpha,
        minimum_endpoint_coverage=args.minimum_endpoint_coverage,
        maximum_rms_ratio_deviation=args.maximum_rms_ratio_deviation,
    )
    report["input_invariants"] = {
        key: value for key, value in reference.items() if key != "identity"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "baseline_accuracy": report["forced_cue_baseline_accuracy"],
                "selected_accuracy_loss_pp": report["primary_result"][
                    "accuracy_loss_percentage_points"
                ],
                "empirical_matched_random_p": report["primary_result"][
                    "empirical_matched_random_p"
                ],
                "evaluation_rms_transport_passed": report["matched_random_null"][
                    "evaluation_rms_transport_passed"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
