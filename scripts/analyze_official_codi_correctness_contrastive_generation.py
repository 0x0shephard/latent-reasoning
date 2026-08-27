"""Summarise paired exact-match generation for contrastive covariance arms."""
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
from src.eval.official_codi_correctness_contrastive_covariance_analysis import (
    paired_bootstrap_interval,
)
from src.mech.endpoint_correctness_contrastive_covariance import (
    CONTRASTIVE_COVARIANCE_CONTRACT,
)
from src.utils.config import load_config


def load_run(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("contract") != CONTRASTIVE_COVARIANCE_CONTRACT:
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
        "endpoint": np.asarray(
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
    settings = load_config(args.config).endpoint_correctness_contrastive_covariance
    names = (
        "baseline",
        "contrastive_correct_retain",
        "correct_only_pca_retain",
        "accuracy_band_pca_retain",
        "contrastive_wrong_remove",
    )
    runs = {name: load_run(args.runs_root / name) for name in names}
    reference = runs["baseline"]
    for name, run in runs.items():
        if run["indices"] != reference["indices"]:
            raise RuntimeError(f"{name} is not paired to baseline")
        if run["summary"]["partition_sha256"] != reference["summary"]["partition_sha256"]:
            raise RuntimeError(f"{name} has the wrong partition")
        if float(run["endpoint"].mean()) < 0.99:
            raise RuntimeError(f"{name} answer-cue coverage is below 99%")

    def contrast(left, right, offset):
        low, high = paired_bootstrap_interval(
            runs[left]["correct"].astype(float),
            runs[right]["correct"].astype(float),
            samples=int(settings.bootstrap_samples),
            seed=int(settings.bootstrap_seed) + offset,
            alpha=float(settings.alpha),
        )
        delta = float(runs[left]["correct"].mean() - runs[right]["correct"].mean())
        return {
            "left": left,
            "right": right,
            "difference_points": 100 * delta,
            "bootstrap_ci_points": [100 * low, 100 * high],
        }

    report = {
        "analysis": "official_codi_correctness_contrastive_generation",
        "contract": CONTRASTIVE_COVARIANCE_CONTRACT,
        "status": "complete",
        "evaluated_examples": len(reference["indices"]),
        "accuracy": {
            name: float(run["correct"].mean()) for name, run in runs.items()
        },
        "paired_contrasts": {
            "correct_vs_correct_only_pca": contrast(
                "contrastive_correct_retain", "correct_only_pca_retain", 11
            ),
            "correct_vs_accuracy_band_pca": contrast(
                "contrastive_correct_retain", "accuracy_band_pca_retain", 12
            ),
            "wrong_remove_vs_baseline": contrast(
                "contrastive_wrong_remove", "baseline", 13
            ),
        },
        "note": (
            "This is a paired exact-match confirmation on the frozen final split. "
            "The first-token preregistered gate remains the primary decision."
        ),
    }
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
