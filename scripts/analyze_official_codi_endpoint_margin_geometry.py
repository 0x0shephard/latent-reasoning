"""Apply the preregistered margin-geometry gates to a completed sweep."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_endpoint_margin_geometry_analysis import (
    analyze_margin_geometry,
)
from src.mech.endpoint_margin_geometry import MARGIN_GEOMETRY_CONTRACT
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = load_config(args.config).endpoint_margin_geometry
    sweep = torch.load(args.sweep, map_location="cpu", weights_only=False)
    if sweep.get("contract") != MARGIN_GEOMETRY_CONTRACT:
        raise RuntimeError("sweep results belong to another contract")
    report = analyze_margin_geometry(
        sweep,
        bootstrap_samples=int(settings.bootstrap_samples),
        bootstrap_seed=int(settings.bootstrap_seed),
        alpha=float(settings.alpha),
        primary_rank=int(settings.primary_rank),
        state=int(settings.analytic_state),
        retention_threshold=float(settings.retention_threshold),
        maximum_calibration_relative_energy_error=float(
            settings.maximum_calibration_relative_energy_error
        ),
        maximum_selected_overlap=float(settings.maximum_selected_overlap),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, args.output)
    primary = report["primary_margin_specificity"]
    effective = report["effective_dimensionality"]["effective_rank_by_family"]
    print(f"[status] {report['status']}")
    print(
        f"[primary] margin rank {report['primary_rank']} mean gold-NLL damage "
        f"{primary['result']['mean_delta']:+.6f} "
        f"(CI {primary['result']['bootstrap_95_ci'][0]:+.6f}, "
        f"{primary['result']['bootstrap_95_ci'][1]:+.6f}), "
        f"empirical matched-random p={primary['empirical_matched_random_p']:.4f}"
    )
    print(f"[effective rank] {effective}")
    print(f"[report] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
