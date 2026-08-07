"""Analyze the complete frozen CODI endpoint accuracy-localization experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_endpoint_accuracy_localization_analysis import (
    analyze_endpoint_accuracy_localization,
)
from src.mech.endpoint_accuracy_localization import ACCURACY_LOCALIZATION_CONTRACT


def _load_run(summary_path: Path) -> tuple[dict, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract") != ACCURACY_LOCALIZATION_CONTRACT:
        raise RuntimeError(f"wrong localization contract: {summary_path}")
    if summary.get("state") != "complete":
        raise RuntimeError(f"incomplete localization run: {summary_path}")
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
    parser.add_argument("--random-replicates", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-endpoint-coverage", type=float, default=0.95)
    args = parser.parse_args()
    summaries = sorted(args.runs_root.rglob("summary.json"))
    loaded = [_load_run(path) for path in summaries]
    if not loaded:
        raise RuntimeError(f"no completed localization runs under {args.runs_root}")
    runs = [value[0] for value in loaded]
    invariants = [value[1] for value in loaded]
    reference = invariants[0]
    for invariant in invariants[1:]:
        if invariant != reference:
            raise RuntimeError("localization arms do not share exact paired inputs/artifacts")
    family_counts = {}
    for run in runs:
        family = "baseline" if run["spec"] is None else run["spec"]["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    expected = {
        "baseline": 1,
        "negative_control_joint": 1,
        "selected_joint": 2,
        "selected_state": 4,
        "selected_single": 12,
        "selected_joint_minus_one": 12,
        "matched_random_joint": 2 * args.random_replicates,
    }
    if family_counts != expected:
        raise RuntimeError(
            f"analysis requires the registered localization family, found {family_counts}"
        )
    report = analyze_endpoint_accuracy_localization(
        runs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        familywise_alpha=args.familywise_alpha,
        minimum_endpoint_coverage=args.minimum_endpoint_coverage,
    )
    report["input_invariants"] = {
        key: value for key, value in reference.items() if key != "identity"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, args.output)
    print(
        json.dumps(
            {
                "baseline_accuracy": report["forced_cue_baseline_accuracy"],
                "critical_joint_subspaces": report["critical_joint_subspaces"],
                "accuracy_core_directions": {
                    method: [
                        name
                        for name, value in result["directions"].items()
                        if value["accuracy_core_direction"]
                    ]
                    for method, result in report["localization"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
