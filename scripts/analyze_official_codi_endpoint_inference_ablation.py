"""Analyze all paired frozen-checkpoint CODI endpoint-ablation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_endpoint_inference_ablation_analysis import (
    analyze_endpoint_inference_ablation,
)
from src.mech.endpoint_inference_ablation import ENDPOINT_ABLATION_CONTRACT


def _load_run(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract") != ENDPOINT_ABLATION_CONTRACT or summary.get("state") != "complete":
        raise RuntimeError(f"not a completed endpoint-ablation summary: {summary_path}")
    predictions_path = summary_path.parent / summary["predictions_file"]
    records = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != int(summary["evaluated_examples"]):
        raise RuntimeError(f"prediction count mismatch: {predictions_path}")
    if [int(record["index"]) for record in records] != list(range(len(records))):
        raise RuntimeError(f"prediction order changed: {predictions_path}")
    return {
        "arm": summary["arm"],
        "spec": summary.get("spec"),
        "intervention_diagnostics": summary.get("intervention_diagnostics"),
        "correctness": [bool(record["correct"]) for record in records],
        "endpoint_reached": [
            bool(record["answer_cue_endpoint_reached"]) for record in records
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-endpoint-coverage", type=float, default=0.95)
    args = parser.parse_args()
    summaries = sorted(args.runs_root.rglob("summary.json"))
    if not summaries:
        raise RuntimeError(f"no completed summaries under {args.runs_root}")
    runs = [_load_run(path) for path in summaries]
    family_counts = {}
    for run in runs:
        family = "baseline" if run["spec"] is None else run["spec"]["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    expected = {
        "baseline": 1,
        "selected_joint": 3,
        "selected_single": 18,
        "random_joint": 20,
        "random_single": 40,
    }
    if family_counts != expected:
        raise RuntimeError(
            f"analysis requires the registered 82-arm family, found {family_counts}"
        )
    report = analyze_endpoint_inference_ablation(
        runs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        familywise_alpha=args.familywise_alpha,
        minimum_endpoint_coverage=args.minimum_endpoint_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, args.output)
    print(json.dumps({
        "baseline_accuracy": report["baseline_accuracy"],
        "endpoint_coverage": report["answer_cue_endpoint_coverage"],
        "accuracy_critical": report["accuracy_critical_directions_or_groups"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
