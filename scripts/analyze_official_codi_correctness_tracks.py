"""Apply the preregistered correctness-track gates and print a readable verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_correctness_tracks_analysis import analyze_correctness_tracks
from src.mech.endpoint_correctness_geometry import CORRECTNESS_CONTRACT
from src.utils.config import load_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml")
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _verdict(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_config(args.config).endpoint_correctness_tracks
    payload = json.loads(args.sweep.read_text())
    if payload.get("contract") != CORRECTNESS_CONTRACT:
        raise RuntimeError("sweep belongs to another contract")
    vectors_path = args.vectors or args.sweep.with_suffix(".pt")
    outcomes = torch.load(vectors_path, map_location="cpu", weights_only=False)["outcomes"]

    report = analyze_correctness_tracks(payload, outcomes, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, args.output)

    geometry = report["geometry"]
    print("=" * 74)
    print("CODI answer-cue correctness tracks")
    print("=" * 74)
    print(
        f"splits  fit={report['splits']['fit']}  select={report['splits']['select']}  "
        f"test={report['splits']['test']}"
    )
    print(
        f"class split: between-class variance {100 * geometry['between_class_fraction']:.2f}% "
        f"of total; mean difference sits {100 * geometry['lift_band_share']:.1f}% in PCs 0-3 "
        f"(random-split median {100 * geometry['null']['median_lift_band_share']:.1f}%, "
        f"{geometry['null']['share_exceedances']}/{geometry['null']['replicates']} exceed it)"
    )
    print()

    detect = report["detect"]
    print(f"[{_verdict(detect['passed'])}] detect — {detect['primary_probe']}")
    print(
        f"      probe AUC {detect['probe_auc']:.4f} vs margin-only {detect['margin_auc']:.4f}"
        f"  delta {detect['delta_auc']:+.4f} "
        f"[{detect['delta_ci'][0]:+.4f}, {detect['delta_ci'][1]:+.4f}]"
    )

    steer = report["steer"]
    print(f"[{_verdict(steer['passed'])}] steer — {steer['primary_arm']}")
    print(
        f"      accuracy {steer['arm_accuracy']:.4f} vs baseline {steer['baseline_accuracy']:.4f}"
        f"  gain {steer['gain_points']:+.2f} pts "
        f"[{steer['gain_ci_points'][0]:+.2f}, {steer['gain_ci_points'][1]:+.2f}]"
        f"  alpha {steer['selected_alpha']:g}"
    )
    print(
        f"      best matched random direction in the same band "
        f"{steer['best_random_band_accuracy']:.4f} "
        f"(margin {steer['margin_over_random_points']:+.2f} pts)"
    )

    project = report["project"]
    print(f"[{_verdict(project['passed'])}] project — rank {project['rank']}")
    print(
        f"      correct-only {project['correct_only_accuracy']:.4f} vs class-blind "
        f"{project['class_blind_accuracy']:.4f}  advantage "
        f"{project['advantage_points']:+.2f} pts "
        f"[{project['advantage_ci_points'][0]:+.2f}, {project['advantage_ci_points'][1]:+.2f}]"
    )
    print(
        f"      subspace overlap: mean principal-angle cosine "
        f"{project['overlap_with_class_blind']['mean_cosine']:.4f}"
    )
    print("=" * 74)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
