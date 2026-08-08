"""Apply the preregistered exact-match gates to the PC-band confirmation arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_endpoint_band_confirmation_analysis import (
    analyze_band_confirmation,
    gsm8k_accuracy_from_summary,
)
from src.utils.config import load_config


BAND_PATTERN = re.compile(r"^band_p(\d{3})_(\d{3})_s\d+$")


def load_runs(roots) -> list[dict]:
    runs = []
    for root in roots:
        for summary_path in sorted(Path(root).rglob("summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            predictions = summary_path.parent / summary.get("predictions_file", "gsm8k.jsonl")
            if not predictions.is_file():
                raise RuntimeError(f"{summary_path} has no paired predictions")
            rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
            match = BAND_PATTERN.match(str(summary.get("arm", "")))
            runs.append(
                {
                    "arm": summary["arm"],
                    "mode": summary.get("mode", "remove"),
                    "band": [int(match.group(1)), int(match.group(2))] if match else None,
                    "correctness": [bool(row["correct"]) for row in rows],
                    "endpoint_reached": [
                        bool(row["answer_cue_endpoint_reached"]) for row in rows
                    ],
                    "accuracy": summary.get("accuracy"),
                }
            )
    if not runs:
        raise RuntimeError("no completed confirmation arms were found")
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--reproduction-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = cfg.endpoint_band_confirmation
    reproduction_accuracy = None
    if args.reproduction_summary and args.reproduction_summary.is_file():
        reproduction_accuracy = gsm8k_accuracy_from_summary(
            json.loads(args.reproduction_summary.read_text(encoding="utf-8"))
        )

    report = analyze_band_confirmation(
        load_runs(args.runs_root),
        primary_band=[int(v) for v in settings.primary_band],
        control_band=[int(v) for v in settings.control_band],
        majority_band=[int(v) for v in settings.majority_band],
        minimum_primary_retention=float(settings.minimum_primary_retention),
        maximum_control_retention=float(settings.maximum_control_retention),
        minimum_primary_removal_points=float(settings.minimum_primary_removal_points),
        bootstrap_samples=int(settings.bootstrap_samples),
        bootstrap_seed=int(settings.bootstrap_seed),
        alpha=float(settings.alpha),
        reproduction_accuracy=reproduction_accuracy,
        maximum_baseline_accuracy_drift=float(
            cfg.endpoint_margin_geometry.maximum_baseline_accuracy_drift
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, args.output)

    print(f"[status] {report['status']}")
    print(
        f"[baseline] exact match {report['baseline_accuracy']:.6f} "
        f"({report['baseline_correct']}/{report['evaluated_examples']}), "
        f"drift {report['baseline_accuracy_drift']}"
    )
    for name, gate in report["gates"].items():
        print(f"[{name}] passed={gate['passed']} {json.dumps({k: v for k, v in gate.items() if k != 'passed'})}")
    for name, arm in report["arms"].items():
        if arm:
            print(
                f"  {name:18s} band {arm['band']} {arm['mode']:6s} "
                f"acc {arm['accuracy']:.4f} retained {arm['retained_fraction']:.3f}"
            )
    print(f"[report] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
