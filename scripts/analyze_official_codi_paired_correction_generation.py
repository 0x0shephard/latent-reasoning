"""Summarize paired exact-match generation for the conditioned correction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_paired_correction_analysis import paired_interval
from src.mech.endpoint_paired_correction import PAIRED_CORRECTION_CONTRACT
from src.utils.config import load_config


def _load(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("contract") != PAIRED_CORRECTION_CONTRACT:
        raise ValueError(f"{directory} belongs to another contract")
    records = [
        json.loads(line)
        for line in (directory / summary["predictions_file"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return {
        "summary": summary,
        "indices": [int(record["index"]) for record in records],
        "correct": np.asarray([record["correct"] for record in records], dtype=bool),
        "reached": np.asarray(
            [record["answer_cue_endpoint_reached"] for record in records], dtype=bool
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = load_config(args.config).endpoint_paired_correction
    names = ("baseline", "conditioned", "global_mean", "shuffled_target")
    runs = {name: _load(args.runs_root / name) for name in names}
    baseline = runs["baseline"]
    for name, run in runs.items():
        if run["indices"] != baseline["indices"]:
            raise RuntimeError(f"{name} is not paired to baseline")
        if run["summary"]["partition_sha256"] != baseline["summary"]["partition_sha256"]:
            raise RuntimeError(f"{name} uses another partition")
        if float(run["reached"].mean()) < 0.99:
            raise RuntimeError(f"{name} answer-cue coverage is below 99%")

    def contrast(left, right, offset):
        interval = paired_interval(
            runs[left]["correct"].astype(float),
            runs[right]["correct"].astype(float),
            samples=int(settings.bootstrap_samples),
            seed=int(settings.bootstrap_seed) + offset,
            alpha=float(settings.alpha),
        )
        difference = float(runs[left]["correct"].mean() - runs[right]["correct"].mean())
        return {
            "left": left,
            "right": right,
            "difference_points": 100 * difference,
            "bootstrap_ci_points": [100 * interval[0], 100 * interval[1]],
        }

    report = {
        "analysis": "official_codi_paired_correction_generation",
        "contract": PAIRED_CORRECTION_CONTRACT,
        "status": "complete",
        "evaluated_examples": len(baseline["indices"]),
        "accuracy": {
            name: float(run["correct"].mean()) for name, run in runs.items()
        },
        "paired_contrasts": {
            "conditioned_vs_baseline": contrast("conditioned", "baseline", 11),
            "conditioned_vs_global": contrast("conditioned", "global_mean", 12),
            "conditioned_vs_shuffled": contrast("conditioned", "shuffled_target", 13),
        },
        "intervention_diagnostics": {
            name: run["summary"].get("intervention_diagnostics")
            for name, run in runs.items()
        },
        "note": "The analytic first-token gate is primary; this is paired exact-match confirmation.",
    }
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
