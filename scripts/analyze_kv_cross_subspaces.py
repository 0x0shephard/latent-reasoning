"""CPU-only CCA/SVD analysis for Stage 1b paired KV cross-moments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mech.kv_cross_subspace import (
    CROSS_STATISTICS_SCHEMA_VERSION,
    analyze_cross_moment_collection,
    cross_moment_collection_from_state,
    render_cross_subspace_markdown,
)


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from exc
    if not ranks or min(ranks) <= 0:
        raise argparse.ArgumentTypeError("ranks must be positive")
    return ranks


def _atomic_text(text: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze whitened teacher/student KV cross-covariance."
    )
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ranks",
        type=_parse_ranks,
        default=(1, 2, 4, 8, 16),
    )
    parser.add_argument("--gate-rank", type=int, default=4)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--correlation-margin", type=float, default=0.05)
    parser.add_argument("--overlap-margin", type=float, default=0.10)
    parser.add_argument("--required-group-fraction", type=float, default=0.60)
    args = parser.parse_args()

    statistics_path = args.statistics
    if statistics_path.is_dir():
        statistics_path = statistics_path / "statistics.pt"
    if not statistics_path.is_file():
        parser.error(f"statistics file does not exist: {statistics_path}")
    output = args.output or statistics_path.with_name(
        "cross_subspace_report.json"
    )

    payload = torch.load(
        statistics_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(payload.get("schema_version", -1)) != CROSS_STATISTICS_SCHEMA_VERSION:
        parser.error("statistics file uses an incompatible schema")
    if not bool(payload.get("complete")):
        parser.error("cross-moment collection is incomplete; resume extraction")
    metadata = payload.get("metadata", {})
    expected = int(metadata.get("requested_examples", -1))
    processed = int(payload.get("processed_examples", -2))
    if expected <= 0 or processed != expected:
        parser.error(
            f"statistics count mismatch: processed {processed}, expected {expected}"
        )

    report = analyze_cross_moment_collection(
        cross_moment_collection_from_state(payload["moments"]),
        ranks=args.ranks,
        gate_rank=args.gate_rank,
        ridge_ratio=args.ridge_ratio,
        correlation_margin=args.correlation_margin,
        overlap_margin=args.overlap_margin,
        required_group_fraction=args.required_group_fraction,
    )
    report["calibration"] = metadata
    _atomic_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        output,
    )
    markdown = output.with_suffix(".md")
    _atomic_text(render_cross_subspace_markdown(report, metadata), markdown)
    print(f"[analysis] gate={report['gate']['status']}")
    print(f"[analysis] wrote {output}")
    print(f"[analysis] wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
