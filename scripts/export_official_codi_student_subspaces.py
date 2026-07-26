"""Export compact student-side KV intervention bases from official CODI moments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from src.mech.kv_cross_subspace import (
    CROSS_STATISTICS_SCHEMA_VERSION,
    cross_moment_collection_from_state,
)
from src.mech.kv_reduced_rank import (
    fit_position_conditioned_student_subspaces,
    random_orthonormal_bases_like,
    subspace_expected_energy,
)
from src.models.official_codi import sha256_file


STUDENT_SUBSPACE_ARTIFACT_SCHEMA_VERSION = 1


def _orthonormal_error(basis: torch.Tensor) -> float:
    identity = torch.eye(basis.shape[-1], dtype=basis.dtype)
    observed = basis.transpose(-1, -2) @ basis
    return float((observed - identity).abs().max())


def export(args: argparse.Namespace) -> dict:
    statistics_path = args.statistics
    if statistics_path.is_dir():
        statistics_path = statistics_path / "statistics.pt"
    if not statistics_path.is_file():
        raise FileNotFoundError(f"statistics file does not exist: {statistics_path}")
    payload = torch.load(statistics_path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != CROSS_STATISTICS_SCHEMA_VERSION:
        raise RuntimeError("statistics use an incompatible cross-moment schema")
    if not bool(payload.get("complete")):
        raise RuntimeError("cross-moment collection is incomplete")
    metadata = payload.get("metadata", {})
    expected = int(metadata.get("requested_examples", -1))
    processed = int(payload.get("processed_examples", -2))
    if expected <= 0 or processed != expected:
        raise RuntimeError(
            f"statistics count mismatch: processed {processed}, expected {expected}"
        )
    if expected < args.minimum_examples:
        raise RuntimeError(
            f"at least {args.minimum_examples} calibration examples are required"
        )
    collection = cross_moment_collection_from_state(payload["moments"])

    kinds = {}
    diagnostics = {}
    for kind in ("key", "value"):
        learned = fit_position_conditioned_student_subspaces(
            collection["actual"][kind],
            rank=args.rank,
            ridge_ratio=args.ridge_ratio,
        )
        learned_basis = learned["basis"]
        random_basis = random_orthonormal_bases_like(
            learned_basis,
            seed=args.random_seed + (0 if kind == "key" else 1_000_003),
        )
        learned_energy = subspace_expected_energy(
            learned["covariance"], learned_basis
        )
        random_unscaled_energy = subspace_expected_energy(
            learned["covariance"], random_basis
        )
        random_scale = torch.sqrt(
            learned_energy / random_unscaled_energy.clamp_min(1e-12)
        )
        random_matched_energy = random_unscaled_energy * random_scale.square()
        relative_energy_error = (
            (random_matched_energy - learned_energy).abs()
            / learned_energy.clamp_min(1e-12)
        )
        if not all(
            torch.isfinite(value).all()
            for value in (
                learned_basis,
                learned["mean"],
                random_basis,
                random_scale,
            )
        ):
            raise RuntimeError(
                f"{kind} intervention artifact contains non-finite values"
            )
        learned_error = _orthonormal_error(learned_basis)
        random_error = _orthonormal_error(random_basis)
        if max(learned_error, random_error) > 1e-4:
            raise RuntimeError(f"{kind} bases are not groupwise orthonormal")
        kinds[kind] = {
            "learned_basis": learned_basis.contiguous(),
            "student_mean": learned["mean"].contiguous(),
            "random_basis": random_basis.contiguous(),
            "random_energy_scale": random_scale.contiguous(),
            "learned_expected_energy": learned_energy.contiguous(),
            "random_unscaled_expected_energy": (
                random_unscaled_energy.contiguous()
            ),
        }
        diagnostics[kind] = {
            "basis_shape": list(learned_basis.shape),
            "mean_shape": list(learned["mean"].shape),
            "learned_orthonormal_max_error": learned_error,
            "random_orthonormal_max_error": random_error,
            "energy_match_max_relative_error": float(relative_energy_error.max()),
            "random_scale_min": float(random_scale.min()),
            "random_scale_median": float(random_scale.median()),
            "random_scale_max": float(random_scale.max()),
        }

    source_sha256 = sha256_file(statistics_path)
    artifact = {
        "schema_version": STUDENT_SUBSPACE_ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "official_codi_student_kv_causal_subspaces",
        "rank": args.rank,
        "ridge_ratio": args.ridge_ratio,
        "random_seed": args.random_seed,
        "source": {
            "statistics_path": str(statistics_path),
            "statistics_sha256": source_sha256,
            "checkpoint_repo": metadata.get("checkpoint_repo"),
            "checkpoint_revision": metadata.get("checkpoint_revision"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "official_source_revision": metadata.get("official_source_revision"),
            "processed_examples": processed,
            "indices_sha256": metadata.get("indices_sha256"),
            "data_seed": metadata.get("seed"),
            "alignment": metadata.get("alignment"),
        },
        "contract": {
            "space": "raw centered student KV feature space",
            "learned_basis": (
                "leading left singular vectors of the rank-r student-to-teacher "
                "reduced-rank map fitted on all calibration splits"
            ),
            "centering": (
                "subtract the calibration student mean before intervention and "
                "restore it afterward"
            ),
            "random_control": (
                "groupwise random orthonormal rank-r basis scaled so its expected "
                "projected calibration energy matches the learned basis"
            ),
            "causal_application": (
                "intervene on each newly appended latent KV cache entry before it "
                "is consumed by later latent steps or answer decoding"
            ),
        },
        "kinds": kinds,
        "diagnostics": diagnostics,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(artifact, output)
    artifact_sha256 = sha256_file(output)
    manifest = {
        "schema_version": STUDENT_SUBSPACE_ARTIFACT_SCHEMA_VERSION,
        "state": "complete",
        "artifact": str(output),
        "artifact_sha256": artifact_sha256,
        "source_statistics": str(statistics_path),
        "source_statistics_sha256": source_sha256,
        "rank": args.rank,
        "random_seed": args.random_seed,
        "checkpoint_revision": metadata.get("checkpoint_revision"),
        "processed_examples": processed,
        "indices_sha256": metadata.get("indices_sha256"),
        "diagnostics": diagnostics,
    }
    _atomic_json(manifest, output.with_suffix(".json"))
    print(f"[export] wrote {output}")
    print(f"[export] wrote {output.with_suffix('.json')}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit compact student-side learned and energy-matched random KV "
            "subspaces from official CODI cross-moment statistics."
        )
    )
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=20260727)
    parser.add_argument("--minimum-examples", type=int, default=5_000)
    args = parser.parse_args()
    if args.rank <= 0:
        parser.error("rank must be positive")
    if args.ridge_ratio < 0:
        parser.error("ridge-ratio must be non-negative")
    if args.minimum_examples < 2:
        parser.error("minimum-examples must be at least two")
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
