"""Aggregate completed rank-matched CODI endpoint-retention runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_endpoint_retention_analysis import (
    analyze_endpoint_retention,
)
from src.mech.endpoint_retention import RETENTION_CONTRACT


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--noninferiority-margin", type=float, default=0.01)
    args = parser.parse_args()
    runs = []
    question_reference = None
    for summary_path in sorted(args.runs_root.glob("**/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("contract") != RETENTION_CONTRACT:
            continue
        prediction_path = summary_path.parent / "gsm8k.jsonl"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"missing paired predictions: {prediction_path}")
        records = _records(prediction_path)
        questions = [str(record["question"]) for record in records]
        if question_reference is None:
            question_reference = questions
        elif questions != question_reference:
            raise RuntimeError("retention runs evaluated different GSM8K question orders")
        runs.append(
            {
                "arm": summary["arm"],
                "training_seed": summary["training_seed"],
                "correctness": [bool(record["correct"]) for record in records],
                "examples_per_second": summary["evaluation"]["examples_per_second"],
                "summary_path": str(summary_path),
            }
        )
    report = analyze_endpoint_retention(
        runs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        noninferiority_margin=args.noninferiority_margin,
    )
    report["run_count"] = len(runs)
    report["runs_root"] = str(args.runs_root)
    _atomic_json(report, args.output)
    print(f"[complete] analyzed {len(runs)} runs -> {args.output}")
    print(
        "[complete] highest selected accuracy: "
        f"{report['highest_selected_accuracy']['method']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
