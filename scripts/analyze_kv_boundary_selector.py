"""CPU-only confirmation analysis for the boundary-aware R-KV selector."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_kv_selector_specificity import _atomic_text, _parse_ranks
from scripts.collect_official_codi_selector_subspaces import (
    SELECTOR_COLLECTION_SCHEMA_VERSION,
)
from src.mech.kv_cross_subspace import cross_moment_collection_from_state
from src.mech.kv_reduced_rank import (
    analyze_reduced_rank_prediction,
    render_reduced_rank_markdown,
)
from src.mech.kv_selector_specificity import (
    analyze_candidate_selector_specificity,
    render_candidate_selector_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test boundary-aware R-KV against unchanged R-KV, uniform, and "
            "seeded-random teacher-trace selectors."
        )
    )
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ranks", type=_parse_ranks, default=(1, 2, 4, 8, 16))
    parser.add_argument("--gate-rank", type=int, default=4)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--required-group-fraction", type=float, default=0.60)
    parser.add_argument("--r2-margin", type=float, default=0.02)
    parser.add_argument("--minimum-median-r2", type=float, default=0.05)
    parser.add_argument("--minimum-full-retention", type=float, default=0.80)
    parser.add_argument("--selector-signal-margin", type=float, default=0.01)
    parser.add_argument("--selector-win-fraction", type=float, default=0.60)
    args = parser.parse_args()

    statistics_path = args.statistics
    if statistics_path.is_dir():
        statistics_path = statistics_path / "selector_statistics.pt"
    if not statistics_path.is_file():
        parser.error(f"statistics file does not exist: {statistics_path}")
    payload = torch.load(statistics_path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != SELECTOR_COLLECTION_SCHEMA_VERSION:
        parser.error("selector statistics use an incompatible schema")
    if not bool(payload.get("complete")):
        parser.error("selector collection is incomplete; resume extraction")
    metadata = payload.get("metadata", {})
    expected = int(metadata.get("requested_examples", -1))
    processed = int(payload.get("processed_examples", -2))
    if expected <= 0 or processed != expected:
        parser.error(
            f"statistics count mismatch: processed {processed}, expected {expected}"
        )
    if not bool(metadata.get("include_boundary_rkv")):
        parser.error("statistics do not include the boundary-aware selector")
    if int(metadata.get("sample_overlap_with_exclusion", -1)) != 0:
        parser.error("calibration sample is not disjoint from the prior experiment")
    if int(metadata.get("excluded_indices_count", 0)) <= 0:
        parser.error("no prior calibration indices were excluded")

    selector_states = payload.get("selectors", {})
    expected_selectors = list(metadata.get("selectors", []))
    if set(selector_states) != set(expected_selectors):
        parser.error("selector state names do not match collection metadata")
    required = {"boundary_rkv", "rkv", "uniform"}
    if not required.issubset(selector_states):
        parser.error(
            f"selector statistics are missing {sorted(required - set(selector_states))}"
        )

    details_dir = args.output.parent / f"{args.output.stem}_details"
    selector_reports = {}
    detail_files = {}
    for selector in expected_selectors:
        print(f"[analysis] reduced-rank prediction for {selector}")
        collection = cross_moment_collection_from_state(selector_states[selector])
        report = analyze_reduced_rank_prediction(
            collection,
            ranks=args.ranks,
            gate_rank=args.gate_rank,
            ridge_ratio=args.ridge_ratio,
            required_group_fraction=args.required_group_fraction,
            r2_margin=args.r2_margin,
            minimum_median_r2=args.minimum_median_r2,
            minimum_full_retention=args.minimum_full_retention,
        )
        report["calibration"] = {**metadata, "selector": selector}
        detail_path = details_dir / f"{selector}_reduced_rank.json"
        _atomic_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            detail_path,
        )
        _atomic_text(
            render_reduced_rank_markdown(report, report["calibration"]),
            detail_path.with_suffix(".md"),
        )
        selector_reports[selector] = report
        detail_files[selector] = str(detail_path)
        del collection

    random_selectors = [
        f"random_seed{seed}" for seed in metadata["random_selector_seeds"]
    ]
    summary = analyze_candidate_selector_specificity(
        selector_reports,
        candidate_selector="boundary_rkv",
        structured_controls=("rkv", "uniform"),
        random_selectors=random_selectors,
        rank=args.gate_rank,
        signal_margin=args.selector_signal_margin,
        required_win_fraction=args.selector_win_fraction,
    )
    summary["calibration"] = metadata
    summary["detailed_reports"] = detail_files
    _atomic_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        args.output,
    )
    _atomic_text(
        render_candidate_selector_markdown(summary, metadata),
        args.output.with_suffix(".md"),
    )
    print(f"[analysis] gate={summary['gate']['status']}")
    print(f"[analysis] wrote {args.output}")
    print(f"[analysis] wrote {args.output.with_suffix('.md')}")
    print(f"[analysis] detailed reports: {details_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
