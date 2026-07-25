"""Export frozen learned and random key projection bases from Stage 1b statistics."""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mech.kv_cross_subspace import (
    CROSS_STATISTICS_SCHEMA_VERSION,
    cross_moment_collection_from_state,
)
from src.mech.kv_reduced_rank import (
    fit_position_conditioned_teacher_bases,
    random_orthonormal_bases_like,
)


PROJECTION_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection_error(basis: torch.Tensor) -> float:
    identity = torch.eye(basis.shape[-1], dtype=basis.dtype)
    gram = basis.transpose(-1, -2) @ basis
    return float((gram - identity).abs().max())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export position-conditioned teacher key projection bases."
    )
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=20260725)
    args = parser.parse_args()

    statistics_path = args.statistics
    if statistics_path.is_dir():
        statistics_path = statistics_path / "statistics.pt"
    if not statistics_path.is_file():
        parser.error(f"statistics file does not exist: {statistics_path}")
    payload = torch.load(
        statistics_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(payload.get("schema_version", -1)) != CROSS_STATISTICS_SCHEMA_VERSION:
        parser.error("statistics file uses an incompatible Stage 1b schema")
    if not bool(payload.get("complete")):
        parser.error("Stage 1b statistics are incomplete")
    metadata = payload.get("metadata", {})
    processed = int(payload.get("processed_examples", -1))
    requested = int(metadata.get("requested_examples", -2))
    if processed <= 0 or processed != requested:
        parser.error(
            f"statistics count mismatch: processed {processed}, expected {requested}"
        )

    collection = cross_moment_collection_from_state(payload["moments"])
    learned = fit_position_conditioned_teacher_bases(
        collection["actual"]["key"],
        rank=args.rank,
        ridge_ratio=args.ridge_ratio,
    )
    random = random_orthonormal_bases_like(learned, seed=args.random_seed)
    artifact = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "key",
        "rank": args.rank,
        "ridge_ratio": args.ridge_ratio,
        "fit_granularity": "layer_head_position",
        "fit_pairing": "actual",
        "fit_splits": "all",
        "processed_examples": processed,
        "checkpoint_step": metadata.get("checkpoint_step"),
        "source_statistics_sha256": _sha256(statistics_path),
        "random_seed": args.random_seed,
        "learned_basis": learned,
        "random_basis": random,
        "maximum_orthonormality_error": {
            "learned": _projection_error(learned),
            "random": _projection_error(random),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(args.output)
    print(f"[projection] learned basis shape={tuple(learned.shape)}")
    print(
        "[projection] maximum orthonormality error "
        f"learned={artifact['maximum_orthonormality_error']['learned']:.3e} "
        f"random={artifact['maximum_orthonormality_error']['random']:.3e}"
    )
    print(f"[projection] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
