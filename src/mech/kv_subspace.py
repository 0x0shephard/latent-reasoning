"""Streaming spectral analysis for aligned teacher/student KV trajectories.

Stage 1 asks whether the difference between KaVa's explicit-CoT teacher targets and
student latent states contains stable low-rank directions.  A raw calibration tensor is
large even for GPT-2, so this module accumulates the exact sufficient statistics needed
for centered covariance/SVD:

    count, sum(x), and sum(x x^T)

Statistics are retained both after pooling the aligned latent positions and separately
for every position.  The leading covariance eigenvectors are the right singular vectors
of the centered residual matrix, without retaining that matrix in memory.
"""
from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass
from typing import Iterable

import torch


STATISTICS_SCHEMA_VERSION = 1
BASELINES = ("actual", "shuffled", "random")
KV_KINDS = ("key", "value")
GRANULARITIES = ("pooled", "position")


def _validate_shape(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    *,
    layers: int,
    heads: int,
    positions: int,
    head_dim: int,
) -> None:
    expected = (tensor.shape[0], layers, heads, positions, head_dim)
    if tensor.shape != expected:
        raise ValueError(f"expected KV tensor {expected}, got {tuple(tensor.shape)}")
    if mask.shape != expected[:-1]:
        raise ValueError(
            f"expected KV mask {expected[:-1]}, got {tuple(mask.shape)}"
        )


@dataclass
class SplitMomentAccumulator:
    """Exact split-wise covariance sufficient statistics for ``[B,L,H,M,D]``."""

    num_splits: int
    layers: int
    heads: int
    positions: int
    head_dim: int
    pooled_count: torch.Tensor
    pooled_sum: torch.Tensor
    pooled_gram: torch.Tensor
    position_count: torch.Tensor
    position_sum: torch.Tensor
    position_gram: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        num_splits: int,
        layers: int,
        heads: int,
        positions: int,
        head_dim: int,
    ) -> "SplitMomentAccumulator":
        if num_splits < 2:
            raise ValueError("num_splits must be at least two")
        if min(layers, heads, positions, head_dim) <= 0:
            raise ValueError("all KV dimensions must be positive")
        pooled = (num_splits, layers, heads)
        positional = (num_splits, layers, heads, positions)
        return cls(
            num_splits=num_splits,
            layers=layers,
            heads=heads,
            positions=positions,
            head_dim=head_dim,
            pooled_count=torch.zeros(pooled, dtype=torch.int64),
            pooled_sum=torch.zeros((*pooled, head_dim), dtype=torch.float32),
            pooled_gram=torch.zeros(
                (*pooled, head_dim, head_dim), dtype=torch.float32
            ),
            position_count=torch.zeros(positional, dtype=torch.int64),
            position_sum=torch.zeros(
                (*positional, head_dim), dtype=torch.float32
            ),
            position_gram=torch.zeros(
                (*positional, head_dim, head_dim), dtype=torch.float32
            ),
        )

    def update(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        split_ids: torch.Tensor,
    ) -> None:
        """Add one CPU or GPU batch without retaining individual observations."""
        _validate_shape(
            tensor,
            mask,
            layers=self.layers,
            heads=self.heads,
            positions=self.positions,
            head_dim=self.head_dim,
        )
        if split_ids.shape != (tensor.shape[0],):
            raise ValueError("split_ids must have shape [batch]")
        if split_ids.numel() and (
            int(split_ids.min()) < 0 or int(split_ids.max()) >= self.num_splits
        ):
            raise ValueError("split_ids contain an out-of-range split")

        values = tensor.detach().to(device="cpu", dtype=torch.float32)
        valid = mask.detach().to(device="cpu", dtype=torch.bool)
        assignments = split_ids.detach().to(device="cpu", dtype=torch.long)
        for split in range(self.num_splits):
            selected = assignments == split
            if not bool(selected.any()):
                continue
            x = values[selected]
            weights = valid[selected].to(dtype=torch.float32)
            weighted = x * weights.unsqueeze(-1)

            self.pooled_count[split] += weights.sum(dim=(0, 3)).to(torch.int64)
            self.pooled_sum[split] += weighted.sum(dim=(0, 3))
            self.pooled_gram[split] += torch.einsum(
                "blhmd,blhme->lhde", weighted, x
            )

            self.position_count[split] += weights.sum(dim=0).to(torch.int64)
            self.position_sum[split] += weighted.sum(dim=0)
            self.position_gram[split] += torch.einsum(
                "blhmd,blhme->lhmde", weighted, x
            )

    def state_dict(self) -> dict:
        return {
            "num_splits": self.num_splits,
            "layers": self.layers,
            "heads": self.heads,
            "positions": self.positions,
            "head_dim": self.head_dim,
            "pooled_count": self.pooled_count,
            "pooled_sum": self.pooled_sum,
            "pooled_gram": self.pooled_gram,
            "position_count": self.position_count,
            "position_sum": self.position_sum,
            "position_gram": self.position_gram,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SplitMomentAccumulator":
        required = {
            "num_splits",
            "layers",
            "heads",
            "positions",
            "head_dim",
            "pooled_count",
            "pooled_sum",
            "pooled_gram",
            "position_count",
            "position_sum",
            "position_gram",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"moment state is missing fields: {sorted(missing)}")
        accumulator = cls(
            num_splits=int(state["num_splits"]),
            layers=int(state["layers"]),
            heads=int(state["heads"]),
            positions=int(state["positions"]),
            head_dim=int(state["head_dim"]),
            pooled_count=state["pooled_count"].cpu(),
            pooled_sum=state["pooled_sum"].cpu(),
            pooled_gram=state["pooled_gram"].cpu(),
            position_count=state["position_count"].cpu(),
            position_sum=state["position_sum"].cpu(),
            position_gram=state["position_gram"].cpu(),
        )
        expected = cls.create(
            num_splits=accumulator.num_splits,
            layers=accumulator.layers,
            heads=accumulator.heads,
            positions=accumulator.positions,
            head_dim=accumulator.head_dim,
        )
        for field in (
            "pooled_count",
            "pooled_sum",
            "pooled_gram",
            "position_count",
            "position_sum",
            "position_gram",
        ):
            if getattr(accumulator, field).shape != getattr(expected, field).shape:
                raise ValueError(f"invalid shape for moment field {field}")
        return accumulator


def create_moment_collection(
    *,
    num_splits: int,
    layers: int,
    heads: int,
    positions: int,
    head_dim: int,
) -> dict[str, dict[str, SplitMomentAccumulator]]:
    """Create independent accumulators for actual, shuffled, and random K/V."""
    return {
        baseline: {
            kind: SplitMomentAccumulator.create(
                num_splits=num_splits,
                layers=layers,
                heads=heads,
                positions=positions,
                head_dim=head_dim,
            )
            for kind in KV_KINDS
        }
        for baseline in BASELINES
    }


def moment_collection_state(
    collection: dict[str, dict[str, SplitMomentAccumulator]],
) -> dict:
    return {
        baseline: {
            kind: collection[baseline][kind].state_dict() for kind in KV_KINDS
        }
        for baseline in BASELINES
    }


def moment_collection_from_state(
    state: dict,
) -> dict[str, dict[str, SplitMomentAccumulator]]:
    for baseline in BASELINES:
        if baseline not in state:
            raise ValueError(f"moment collection is missing baseline {baseline!r}")
        for kind in KV_KINDS:
            if kind not in state[baseline]:
                raise ValueError(
                    f"moment collection is missing {baseline}.{kind}"
                )
    return {
        baseline: {
            kind: SplitMomentAccumulator.from_state_dict(state[baseline][kind])
            for kind in KV_KINDS
        }
        for baseline in BASELINES
    }


def deterministic_derangement(
    size: int,
    *,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a seeded permutation with no example paired to itself."""
    if size < 2:
        raise ValueError("a derangement requires at least two examples")
    identity = torch.arange(size)
    for _ in range(100):
        candidate = torch.randperm(size, generator=generator)
        if bool((candidate != identity).all()):
            return candidate.to(device=device)
    # Rejection is effectively guaranteed for calibration batches, but the cyclic
    # fallback makes the contract total and deterministic for every supported size.
    return identity.roll(1).to(device=device)


def energy_matched_random(
    reference: torch.Tensor,
    mask: torch.Tensor,
    *,
    generator: torch.Generator,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Create an isotropic null with equal per-layer/head/position residual energy."""
    if reference.ndim != 5 or mask.shape != reference.shape[:-1]:
        raise ValueError("expected reference [B,L,H,M,D] and mask [B,L,H,M]")
    values = reference.detach().float()
    valid = mask.to(device=reference.device, dtype=torch.bool)
    noise = torch.randn(
        values.shape,
        generator=generator,
        device=values.device,
        dtype=torch.float32,
    )
    weights = valid.unsqueeze(-1).to(dtype=torch.float32)
    dimensions = values.shape[-1]
    counts = valid.sum(dim=0).to(torch.float32) * dimensions
    reference_energy = (values.square() * weights).sum(dim=(0, 4))
    noise_energy = (noise.square() * weights).sum(dim=(0, 4))
    scale = torch.sqrt(
        reference_energy.clamp_min(0.0)
        / noise_energy.clamp_min(eps)
    )
    randomized = noise * scale.unsqueeze(0).unsqueeze(-1)
    randomized = randomized * weights
    # Groups with no valid rows have zero reference and are kept exactly zero.
    randomized = torch.where(
        (counts > 0).unsqueeze(0).unsqueeze(-1),
        randomized,
        torch.zeros_like(randomized),
    )
    return randomized


def _batched_eigensystems(
    counts: torch.Tensor,
    totals: torch.Tensor,
    grams: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized eigendecomposition over arbitrary leading group dimensions."""
    valid = counts >= 2
    counts64 = counts.to(dtype=torch.float64)
    totals64 = totals.to(dtype=torch.float64)
    covariance = grams.to(dtype=torch.float64) - (
        totals64.unsqueeze(-1) * totals64.unsqueeze(-2)
    ) / counts64.clamp_min(1).unsqueeze(-1).unsqueeze(-1)
    covariance = covariance / (counts64 - 1).clamp_min(1).unsqueeze(
        -1
    ).unsqueeze(-1)
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    covariance = torch.where(
        valid.unsqueeze(-1).unsqueeze(-1),
        covariance,
        torch.zeros_like(covariance),
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
    eigenvectors = eigenvectors.flip(-1)
    return eigenvalues, eigenvectors, valid


def _safe_mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) > 1 else None


def _spectrum_metrics(eigenvalues: torch.Tensor, ranks: tuple[int, ...]) -> dict:
    total = float(eigenvalues.sum())
    if total <= 0:
        return {
            "total_variance": 0.0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "explained_variance": {str(rank): 0.0 for rank in ranks},
            "top_eigenvalues": [0.0 for _ in range(min(16, eigenvalues.numel()))],
        }
    probabilities = eigenvalues / eigenvalues.sum()
    nonzero = probabilities > 0
    entropy = -(
        probabilities[nonzero] * torch.log(probabilities[nonzero])
    ).sum()
    effective_rank = float(torch.exp(entropy))
    stable_rank = total / max(float(eigenvalues[0]), 1e-30)
    return {
        "total_variance": total,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "explained_variance": {
            str(rank): float(eigenvalues[:rank].sum() / eigenvalues.sum())
            for rank in ranks
        },
        "top_eigenvalues": [
            float(value) for value in eigenvalues[: min(16, eigenvalues.numel())]
        ],
    }


def _split_stability(
    systems: list[tuple[torch.Tensor, torch.Tensor] | None],
    ranks: tuple[int, ...],
) -> dict:
    overlaps = {rank: [] for rank in ranks}
    largest_angles = {rank: [] for rank in ranks}
    spectrum_cosines: list[float] = []
    for left_index, right_index in itertools.combinations(range(len(systems)), 2):
        left = systems[left_index]
        right = systems[right_index]
        if left is None or right is None:
            continue
        left_values, left_vectors = left
        right_values, right_vectors = right
        left_spectrum = left_values / left_values.sum().clamp_min(1e-30)
        right_spectrum = right_values / right_values.sum().clamp_min(1e-30)
        denominator = (
            torch.linalg.vector_norm(left_spectrum)
            * torch.linalg.vector_norm(right_spectrum)
        ).clamp_min(1e-30)
        spectrum_cosines.append(
            float(torch.dot(left_spectrum, right_spectrum) / denominator)
        )
        for rank in ranks:
            left_basis = left_vectors[:, :rank]
            right_basis = right_vectors[:, :rank]
            singular_values = torch.linalg.svdvals(left_basis.T @ right_basis)
            singular_values = singular_values.clamp(0.0, 1.0)
            overlaps[rank].append(float(singular_values.square().mean()))
            largest_angles[rank].append(
                math.degrees(math.acos(float(singular_values.min())))
            )
    return {
        "split_pairs": len(spectrum_cosines),
        "spectrum_cosine": {
            "mean": _safe_mean(spectrum_cosines),
            "stdev": _safe_stdev(spectrum_cosines),
            "minimum": min(spectrum_cosines) if spectrum_cosines else None,
        },
        "subspace_overlap": {
            str(rank): {
                "mean": _safe_mean(overlaps[rank]),
                "stdev": _safe_stdev(overlaps[rank]),
                "minimum": min(overlaps[rank]) if overlaps[rank] else None,
                "largest_principal_angle_degrees_mean": _safe_mean(
                    largest_angles[rank]
                ),
            }
            for rank in ranks
        },
    }


def _granularity_tensors(
    accumulator: SplitMomentAccumulator,
    granularity: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if granularity == "pooled":
        return (
            accumulator.pooled_count,
            accumulator.pooled_sum,
            accumulator.pooled_gram,
        )
    if granularity == "position":
        return (
            accumulator.position_count,
            accumulator.position_sum,
            accumulator.position_gram,
        )
    raise ValueError(f"unknown granularity {granularity!r}")


def _analyze_accumulator(
    accumulator: SplitMomentAccumulator,
    *,
    granularity: str,
    ranks: tuple[int, ...],
) -> list[dict]:
    counts, totals, grams = _granularity_tensors(accumulator, granularity)
    group_shape = tuple(counts.shape[1:])
    split_values, split_vectors, split_valid = _batched_eigensystems(
        counts,
        totals,
        grams,
    )
    full_counts = counts.sum(dim=0)
    full_values, _, full_valid = _batched_eigensystems(
        full_counts,
        totals.sum(dim=0),
        grams.sum(dim=0),
    )
    groups = []
    for coordinates in itertools.product(*(range(size) for size in group_shape)):
        if not bool(full_valid[coordinates]):
            continue
        split_systems = []
        for split in range(accumulator.num_splits):
            index = (split, *coordinates)
            if bool(split_valid[index]):
                split_systems.append(
                    (split_values[index], split_vectors[index])
                )
            else:
                split_systems.append(None)
        eigenvalues = full_values[coordinates]
        group = {
            "layer": coordinates[0],
            "head": coordinates[1],
            "count": int(full_counts[coordinates]),
            **_spectrum_metrics(eigenvalues, ranks),
            "stability": _split_stability(split_systems, ranks),
        }
        if granularity == "position":
            group["position"] = coordinates[2]
        groups.append(group)
    return groups


def _group_key(group: dict, granularity: str) -> tuple[int, ...]:
    key = (int(group["layer"]), int(group["head"]))
    if granularity == "position":
        key += (int(group["position"]),)
    return key


def _aggregate_groups(groups: list[dict], ranks: tuple[int, ...]) -> dict:
    return {
        "groups": len(groups),
        "median_effective_rank": _safe_median(
            [float(group["effective_rank"]) for group in groups]
        ),
        "median_stable_rank": _safe_median(
            [float(group["stable_rank"]) for group in groups]
        ),
        "median_spectrum_cosine": _safe_median(
            [
                float(group["stability"]["spectrum_cosine"]["mean"])
                for group in groups
                if group["stability"]["spectrum_cosine"]["mean"] is not None
            ]
        ),
        "ranks": {
            str(rank): {
                "median_explained_variance": _safe_median(
                    [
                        float(group["explained_variance"][str(rank)])
                        for group in groups
                    ]
                ),
                "median_split_overlap": _safe_median(
                    [
                        float(
                            group["stability"]["subspace_overlap"][str(rank)][
                                "mean"
                            ]
                        )
                        for group in groups
                        if group["stability"]["subspace_overlap"][str(rank)][
                            "mean"
                        ]
                        is not None
                    ]
                ),
            }
            for rank in ranks
        },
    }


def _position_summary(groups: list[dict], ranks: tuple[int, ...]) -> list[dict]:
    if not groups or "position" not in groups[0]:
        return []
    positions = sorted({int(group["position"]) for group in groups})
    return [
        {
            "position": position,
            **_aggregate_groups(
                [group for group in groups if int(group["position"]) == position],
                ranks,
            ),
        }
        for position in positions
    ]


def analyze_moment_collection(
    collection: dict[str, dict[str, SplitMomentAccumulator]],
    *,
    ranks: Iterable[int] = (1, 2, 4, 8, 16),
    gate_rank: int = 4,
    overlap_margin: float = 0.10,
    explained_variance_margin: float = 0.05,
    required_group_fraction: float = 0.60,
) -> dict:
    """Analyze spectra and apply a transparent, descriptive Stage-1 gate.

    The gate is deliberately stricter than merely observing low effective rank.  For a
    KV kind to pass, its pooled rank-``gate_rank`` subspace must be more stable than both
    cross-example shuffling and isotropic noise in at least ``required_group_fraction``
    of layer/head groups.  The median overlap and explained-variance advantages must
    also clear the configured margins.
    """
    ranks_tuple = tuple(sorted({int(rank) for rank in ranks}))
    if not ranks_tuple or min(ranks_tuple) <= 0:
        raise ValueError("ranks must contain positive integers")
    exemplar = collection["actual"]["key"]
    if max(ranks_tuple) > exemplar.head_dim:
        raise ValueError("a requested rank exceeds the KV head dimension")
    if gate_rank not in ranks_tuple:
        raise ValueError("gate_rank must be included in ranks")
    if not 0.0 <= required_group_fraction <= 1.0:
        raise ValueError("required_group_fraction must be in [0, 1]")

    detailed: dict[str, dict] = {}
    for baseline in BASELINES:
        detailed[baseline] = {}
        for kind in KV_KINDS:
            detailed[baseline][kind] = {}
            for granularity in GRANULARITIES:
                groups = _analyze_accumulator(
                    collection[baseline][kind],
                    granularity=granularity,
                    ranks=ranks_tuple,
                )
                detailed[baseline][kind][granularity] = {
                    "summary": _aggregate_groups(groups, ranks_tuple),
                    "by_position": _position_summary(groups, ranks_tuple),
                    "groups": groups,
                }

    comparisons = {}
    kind_gate = {}
    rank_key = str(gate_rank)
    for kind in KV_KINDS:
        comparisons[kind] = {}
        for granularity in GRANULARITIES:
            actual_groups = detailed["actual"][kind][granularity]["groups"]
            shuffled_groups = {
                _group_key(group, granularity): group
                for group in detailed["shuffled"][kind][granularity]["groups"]
            }
            random_groups = {
                _group_key(group, granularity): group
                for group in detailed["random"][kind][granularity]["groups"]
            }
            overlap_deltas = []
            variance_deltas = []
            wins = 0
            matched = 0
            for actual in actual_groups:
                key = _group_key(actual, granularity)
                shuffled = shuffled_groups.get(key)
                random = random_groups.get(key)
                if shuffled is None or random is None:
                    continue
                actual_overlap = actual["stability"]["subspace_overlap"][rank_key][
                    "mean"
                ]
                shuffled_overlap = shuffled["stability"]["subspace_overlap"][
                    rank_key
                ]["mean"]
                random_overlap = random["stability"]["subspace_overlap"][rank_key][
                    "mean"
                ]
                if None in (actual_overlap, shuffled_overlap, random_overlap):
                    continue
                strongest_null = max(float(shuffled_overlap), float(random_overlap))
                overlap_delta = float(actual_overlap) - strongest_null
                actual_variance = float(actual["explained_variance"][rank_key])
                random_variance = float(random["explained_variance"][rank_key])
                variance_delta = actual_variance - random_variance
                matched += 1
                wins += int(
                    overlap_delta > 0.0
                    and actual_variance > random_variance
                )
                overlap_deltas.append(overlap_delta)
                variance_deltas.append(variance_delta)
            comparison = {
                "matched_groups": matched,
                "fraction_above_both_nulls": wins / matched if matched else None,
                "median_overlap_delta_vs_strongest_null": _safe_median(
                    overlap_deltas
                ),
                "median_explained_variance_delta_vs_random": _safe_median(
                    variance_deltas
                ),
            }
            comparisons[kind][granularity] = comparison

        pooled = comparisons[kind]["pooled"]
        supported = bool(
            pooled["matched_groups"]
            and pooled["fraction_above_both_nulls"] is not None
            and pooled["fraction_above_both_nulls"] >= required_group_fraction
            and pooled["median_overlap_delta_vs_strongest_null"] is not None
            and pooled["median_overlap_delta_vs_strongest_null"] >= overlap_margin
            and pooled["median_explained_variance_delta_vs_random"] is not None
            and pooled["median_explained_variance_delta_vs_random"]
            >= explained_variance_margin
        )
        kind_gate[kind] = {
            "supported": supported,
            **pooled,
        }

    if all(item["supported"] for item in kind_gate.values()):
        status = "supported_for_keys_and_values"
    elif any(item["supported"] for item in kind_gate.values()):
        status = "supported_for_one_kv_kind"
    else:
        status = "not_supported_by_preregistered_gate"

    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "analysis": "teacher_minus_student_kv_residual_subspaces",
        "ranks": list(ranks_tuple),
        "gate": {
            "status": status,
            "rank": gate_rank,
            "overlap_margin": overlap_margin,
            "explained_variance_margin": explained_variance_margin,
            "required_group_fraction": required_group_fraction,
            "by_kind": kind_gate,
            "interpretation": (
                "This is a calibration-stage diagnostic, not an accuracy or causal "
                "performance claim."
            ),
        },
        "comparisons": comparisons,
        "baselines": detailed,
    }


def render_subspace_markdown(report: dict, metadata: dict | None = None) -> str:
    """Render a compact reader-facing Stage-1 report."""
    metadata = metadata or {}
    gate = report["gate"]
    lines = [
        "# Stage 1 KV residual subspace analysis",
        "",
        "## Outcome",
        "",
        f"Predefined diagnostic gate: **{gate['status'].replace('_', ' ')}**.",
        "",
        (
            "The analysis tests whether centered teacher-minus-student KV residuals "
            "have split-stable low-rank directions beyond cross-example shuffling and "
            "energy-matched isotropic noise. It does not test downstream accuracy."
        ),
        "",
        "## Calibration contract",
        "",
        f"- Checkpoint step: {metadata.get('checkpoint_step', 'unknown')}",
        f"- Requested examples: {metadata.get('requested_examples', 'unknown')}",
        f"- Processed examples: {metadata.get('processed_examples', 'unknown')}",
        f"- Independent splits: {metadata.get('num_splits', 'unknown')}",
        f"- Alignment: {metadata.get('alignment', 'R-KV chronological selected slots')}",
        f"- Seed: {metadata.get('seed', 'unknown')}",
        "",
        "## Pooled layer-head results",
        "",
        "| KV kind | Actual overlap | Shuffled overlap | Random overlap | "
        "Actual top-r variance | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    rank = str(gate["rank"])
    for kind in KV_KINDS:
        actual = report["baselines"]["actual"][kind]["pooled"]["summary"]["ranks"][
            rank
        ]
        shuffled = report["baselines"]["shuffled"][kind]["pooled"]["summary"][
            "ranks"
        ][rank]
        random = report["baselines"]["random"][kind]["pooled"]["summary"]["ranks"][
            rank
        ]
        kind_gate = gate["by_kind"][kind]
        lines.append(
            "| {kind} | {actual_overlap} | {shuffled_overlap} | {random_overlap} | "
            "{variance} | {status} |".format(
                kind=kind,
                actual_overlap=_format_float(actual["median_split_overlap"]),
                shuffled_overlap=_format_float(shuffled["median_split_overlap"]),
                random_overlap=_format_float(random["median_split_overlap"]),
                variance=_format_float(actual["median_explained_variance"]),
                status="supported" if kind_gate["supported"] else "not supported",
            )
        )

    lines.extend(
        [
            "",
            "## Position-resolved actual residuals",
            "",
            "| KV kind | Position | Median overlap | Top-r explained variance | "
            "Effective rank |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for kind in KV_KINDS:
        positions = report["baselines"]["actual"][kind]["position"]["by_position"]
        for position in positions:
            rank_metrics = position["ranks"][rank]
            lines.append(
                "| {kind} | {position} | {overlap} | {variance} | {effective} |".format(
                    kind=kind,
                    position=position["position"],
                    overlap=_format_float(rank_metrics["median_split_overlap"]),
                    variance=_format_float(
                        rank_metrics["median_explained_variance"]
                    ),
                    effective=_format_float(position["median_effective_rank"]),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "A positive gate supports proceeding to low-rank target filtering. A "
                "negative gate means this residual definition and alignment did not "
                "separate stable paired structure from the nulls. Either outcome should "
                "be confirmed with a second calibration seed before retraining."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
