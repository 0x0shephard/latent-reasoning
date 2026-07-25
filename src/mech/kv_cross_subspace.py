"""Whitened teacher/student KV cross-subspace analysis.

Residual covariance can remain strongly low-rank after teacher examples are shuffled
because it contains the teacher and student marginal covariances.  This module measures
the paired term directly.  For each layer, KV head, and optional trajectory position it
streams:

    count, sum(T), sum(S), sum(TT^T), sum(SS^T), and sum(TS^T)

The centered cross-covariance is whitened by the teacher and student covariances and
decomposed with SVD.  Its singular values are ridge-regularized canonical correlations.
"""
from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass
from typing import Iterable

import torch


CROSS_STATISTICS_SCHEMA_VERSION = 1
PAIRINGS = ("actual", "shuffled")
KV_KINDS = ("key", "value")
GRANULARITIES = ("pooled", "position")


@dataclass
class SplitCrossMomentAccumulator:
    """Split-wise paired moments for teacher/student tensors ``[B,L,H,M,D]``."""

    num_splits: int
    layers: int
    heads: int
    positions: int
    head_dim: int
    pooled_count: torch.Tensor
    pooled_teacher_sum: torch.Tensor
    pooled_student_sum: torch.Tensor
    pooled_teacher_gram: torch.Tensor
    pooled_student_gram: torch.Tensor
    pooled_cross: torch.Tensor
    position_count: torch.Tensor
    position_teacher_sum: torch.Tensor
    position_student_sum: torch.Tensor
    position_teacher_gram: torch.Tensor
    position_student_gram: torch.Tensor
    position_cross: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        num_splits: int,
        layers: int,
        heads: int,
        positions: int,
        head_dim: int,
    ) -> "SplitCrossMomentAccumulator":
        if num_splits < 2:
            raise ValueError("num_splits must be at least two")
        if min(layers, heads, positions, head_dim) <= 0:
            raise ValueError("all KV dimensions must be positive")
        pooled = (num_splits, layers, heads)
        positional = (num_splits, layers, heads, positions)

        def vector(shape):
            return torch.zeros((*shape, head_dim), dtype=torch.float32)

        def matrix(shape):
            return torch.zeros(
                (*shape, head_dim, head_dim), dtype=torch.float32
            )

        return cls(
            num_splits=num_splits,
            layers=layers,
            heads=heads,
            positions=positions,
            head_dim=head_dim,
            pooled_count=torch.zeros(pooled, dtype=torch.int64),
            pooled_teacher_sum=vector(pooled),
            pooled_student_sum=vector(pooled),
            pooled_teacher_gram=matrix(pooled),
            pooled_student_gram=matrix(pooled),
            pooled_cross=matrix(pooled),
            position_count=torch.zeros(positional, dtype=torch.int64),
            position_teacher_sum=vector(positional),
            position_student_sum=vector(positional),
            position_teacher_gram=matrix(positional),
            position_student_gram=matrix(positional),
            position_cross=matrix(positional),
        )

    def update(
        self,
        teacher: torch.Tensor,
        student: torch.Tensor,
        mask: torch.Tensor,
        split_ids: torch.Tensor,
    ) -> None:
        expected = (
            teacher.shape[0],
            self.layers,
            self.heads,
            self.positions,
            self.head_dim,
        )
        if teacher.shape != expected or student.shape != expected:
            raise ValueError(
                f"teacher/student must both have shape {expected}, got "
                f"{tuple(teacher.shape)} and {tuple(student.shape)}"
            )
        if mask.shape != expected[:-1]:
            raise ValueError(f"mask must have shape {expected[:-1]}")
        if split_ids.shape != (teacher.shape[0],):
            raise ValueError("split_ids must have shape [batch]")
        if split_ids.numel() and (
            int(split_ids.min()) < 0 or int(split_ids.max()) >= self.num_splits
        ):
            raise ValueError("split_ids contain an out-of-range split")

        target = teacher.detach().to(device="cpu", dtype=torch.float32)
        source = student.detach().to(device="cpu", dtype=torch.float32)
        valid = mask.detach().to(device="cpu", dtype=torch.bool)
        assignments = split_ids.detach().to(device="cpu", dtype=torch.long)
        for split in range(self.num_splits):
            selected = assignments == split
            if not bool(selected.any()):
                continue
            teacher_split = target[selected]
            student_split = source[selected]
            weights = valid[selected].to(torch.float32)
            weighted_teacher = teacher_split * weights.unsqueeze(-1)
            weighted_student = student_split * weights.unsqueeze(-1)

            self.pooled_count[split] += weights.sum(dim=(0, 3)).to(torch.int64)
            self.pooled_teacher_sum[split] += weighted_teacher.sum(dim=(0, 3))
            self.pooled_student_sum[split] += weighted_student.sum(dim=(0, 3))
            self.pooled_teacher_gram[split] += torch.einsum(
                "blhmd,blhme->lhde", weighted_teacher, teacher_split
            )
            self.pooled_student_gram[split] += torch.einsum(
                "blhmd,blhme->lhde", weighted_student, student_split
            )
            self.pooled_cross[split] += torch.einsum(
                "blhmd,blhme->lhde", weighted_teacher, student_split
            )

            self.position_count[split] += weights.sum(dim=0).to(torch.int64)
            self.position_teacher_sum[split] += weighted_teacher.sum(dim=0)
            self.position_student_sum[split] += weighted_student.sum(dim=0)
            self.position_teacher_gram[split] += torch.einsum(
                "blhmd,blhme->lhmde", weighted_teacher, teacher_split
            )
            self.position_student_gram[split] += torch.einsum(
                "blhmd,blhme->lhmde", weighted_student, student_split
            )
            self.position_cross[split] += torch.einsum(
                "blhmd,blhme->lhmde", weighted_teacher, student_split
            )

    def state_dict(self) -> dict:
        return {
            field: getattr(self, field)
            for field in (
                "num_splits",
                "layers",
                "heads",
                "positions",
                "head_dim",
                "pooled_count",
                "pooled_teacher_sum",
                "pooled_student_sum",
                "pooled_teacher_gram",
                "pooled_student_gram",
                "pooled_cross",
                "position_count",
                "position_teacher_sum",
                "position_student_sum",
                "position_teacher_gram",
                "position_student_gram",
                "position_cross",
            )
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SplitCrossMomentAccumulator":
        exemplar = cls.create(
            num_splits=int(state["num_splits"]),
            layers=int(state["layers"]),
            heads=int(state["heads"]),
            positions=int(state["positions"]),
            head_dim=int(state["head_dim"]),
        )
        values = {}
        for field in exemplar.state_dict():
            if field not in state:
                raise ValueError(f"cross-moment state is missing {field!r}")
            value = state[field]
            if isinstance(value, torch.Tensor):
                value = value.cpu()
                if value.shape != getattr(exemplar, field).shape:
                    raise ValueError(f"invalid shape for cross-moment field {field}")
            values[field] = value
        return cls(**values)


def create_cross_moment_collection(
    *,
    num_splits: int,
    layers: int,
    heads: int,
    positions: int,
    head_dim: int,
) -> dict[str, dict[str, SplitCrossMomentAccumulator]]:
    return {
        pairing: {
            kind: SplitCrossMomentAccumulator.create(
                num_splits=num_splits,
                layers=layers,
                heads=heads,
                positions=positions,
                head_dim=head_dim,
            )
            for kind in KV_KINDS
        }
        for pairing in PAIRINGS
    }


def cross_moment_collection_state(collection: dict) -> dict:
    return {
        pairing: {
            kind: collection[pairing][kind].state_dict() for kind in KV_KINDS
        }
        for pairing in PAIRINGS
    }


def cross_moment_collection_from_state(state: dict) -> dict:
    return {
        pairing: {
            kind: SplitCrossMomentAccumulator.from_state_dict(
                state[pairing][kind]
            )
            for kind in KV_KINDS
        }
        for pairing in PAIRINGS
    }


def _granularity_tensors(
    accumulator: SplitCrossMomentAccumulator,
    granularity: str,
) -> tuple[torch.Tensor, ...]:
    prefix = "pooled" if granularity == "pooled" else "position"
    if granularity not in GRANULARITIES:
        raise ValueError(f"unknown granularity {granularity!r}")
    return tuple(
        getattr(accumulator, f"{prefix}_{suffix}")
        for suffix in (
            "count",
            "teacher_sum",
            "student_sum",
            "teacher_gram",
            "student_gram",
            "cross",
        )
    )


def _batched_cca(
    counts: torch.Tensor,
    teacher_sum: torch.Tensor,
    student_sum: torch.Tensor,
    teacher_gram: torch.Tensor,
    student_gram: torch.Tensor,
    cross: torch.Tensor,
    *,
    ridge_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if ridge_ratio < 0:
        raise ValueError("ridge_ratio must be non-negative")
    valid = counts >= 2
    n = counts.to(torch.float64)
    teacher_sum = teacher_sum.to(torch.float64)
    student_sum = student_sum.to(torch.float64)
    denom = (n - 1).clamp_min(1).unsqueeze(-1).unsqueeze(-1)
    teacher_cov = (
        teacher_gram.to(torch.float64)
        - teacher_sum.unsqueeze(-1) * teacher_sum.unsqueeze(-2)
        / n.clamp_min(1).unsqueeze(-1).unsqueeze(-1)
    ) / denom
    student_cov = (
        student_gram.to(torch.float64)
        - student_sum.unsqueeze(-1) * student_sum.unsqueeze(-2)
        / n.clamp_min(1).unsqueeze(-1).unsqueeze(-1)
    ) / denom
    cross_cov = (
        cross.to(torch.float64)
        - teacher_sum.unsqueeze(-1) * student_sum.unsqueeze(-2)
        / n.clamp_min(1).unsqueeze(-1).unsqueeze(-1)
    ) / denom
    teacher_cov = 0.5 * (teacher_cov + teacher_cov.transpose(-1, -2))
    student_cov = 0.5 * (student_cov + student_cov.transpose(-1, -2))

    def inverse_root(covariance: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp_min(0.0)
        scale = eigenvalues.sum(dim=-1) / covariance.shape[-1]
        ridge = ridge_ratio * scale + 1e-10
        inverse = torch.rsqrt(eigenvalues + ridge.unsqueeze(-1))
        return (eigenvectors * inverse.unsqueeze(-2)) @ eigenvectors.transpose(
            -1, -2
        )

    teacher_inverse_root = inverse_root(teacher_cov)
    student_inverse_root = inverse_root(student_cov)
    whitened = teacher_inverse_root @ cross_cov @ student_inverse_root
    whitened = torch.where(
        valid.unsqueeze(-1).unsqueeze(-1),
        whitened,
        torch.zeros_like(whitened),
    )
    teacher_vectors, correlations, student_vectors_t = torch.linalg.svd(
        whitened,
        full_matrices=False,
    )
    correlations = correlations.clamp(0.0, 1.0)
    # Map canonical axes back to original feature coordinates, then orthonormalize
    # sequentially. The first r QR columns span the first r canonical coefficients.
    teacher_coefficients = teacher_inverse_root @ teacher_vectors
    student_coefficients = (
        student_inverse_root @ student_vectors_t.transpose(-1, -2)
    )
    teacher_basis, _ = torch.linalg.qr(teacher_coefficients)
    student_basis, _ = torch.linalg.qr(student_coefficients)
    return correlations, teacher_basis, student_basis, valid


def _safe_mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) > 1 else None


def _subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    singular = torch.linalg.svdvals(left.T @ right).clamp(0.0, 1.0)
    overlap = float(singular.square().mean())
    largest_angle = math.degrees(math.acos(float(singular.min())))
    return overlap, largest_angle


def _split_stability(
    systems: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None],
    ranks: tuple[int, ...],
) -> dict:
    spectrum_cosines = []
    teacher_overlaps = {rank: [] for rank in ranks}
    student_overlaps = {rank: [] for rank in ranks}
    teacher_angles = {rank: [] for rank in ranks}
    student_angles = {rank: [] for rank in ranks}
    for left_index, right_index in itertools.combinations(range(len(systems)), 2):
        left = systems[left_index]
        right = systems[right_index]
        if left is None or right is None:
            continue
        left_corr, left_teacher, left_student = left
        right_corr, right_teacher, right_student = right
        left_spectrum = left_corr.square()
        right_spectrum = right_corr.square()
        denominator = (
            torch.linalg.vector_norm(left_spectrum)
            * torch.linalg.vector_norm(right_spectrum)
        ).clamp_min(1e-30)
        spectrum_cosines.append(
            float(torch.dot(left_spectrum, right_spectrum) / denominator)
        )
        for rank in ranks:
            teacher_overlap, teacher_angle = _subspace_overlap(
                left_teacher[:, :rank],
                right_teacher[:, :rank],
            )
            student_overlap, student_angle = _subspace_overlap(
                left_student[:, :rank],
                right_student[:, :rank],
            )
            teacher_overlaps[rank].append(teacher_overlap)
            student_overlaps[rank].append(student_overlap)
            teacher_angles[rank].append(teacher_angle)
            student_angles[rank].append(student_angle)
    return {
        "split_pairs": len(spectrum_cosines),
        "spectrum_cosine": {
            "mean": _safe_mean(spectrum_cosines),
            "stdev": _safe_stdev(spectrum_cosines),
        },
        "ranks": {
            str(rank): {
                "teacher_overlap_mean": _safe_mean(teacher_overlaps[rank]),
                "student_overlap_mean": _safe_mean(student_overlaps[rank]),
                "teacher_largest_principal_angle_degrees_mean": _safe_mean(
                    teacher_angles[rank]
                ),
                "student_largest_principal_angle_degrees_mean": _safe_mean(
                    student_angles[rank]
                ),
            }
            for rank in ranks
        },
    }


def _correlation_metrics(
    correlations: torch.Tensor,
    ranks: tuple[int, ...],
) -> dict:
    energy = correlations.square()
    total = float(energy.sum())
    if total > 0:
        probabilities = energy / energy.sum()
        nonzero = probabilities > 0
        effective_rank = float(
            torch.exp(
                -(probabilities[nonzero] * torch.log(probabilities[nonzero])).sum()
            )
        )
    else:
        effective_rank = 0.0
    return {
        "top_canonical_correlations": [
            float(value) for value in correlations[: min(16, correlations.numel())]
        ],
        "effective_correlation_rank": effective_rank,
        "ranks": {
            str(rank): {
                "mean_canonical_correlation": float(correlations[:rank].mean()),
                "mean_squared_canonical_correlation": float(
                    correlations[:rank].square().mean()
                ),
            }
            for rank in ranks
        },
    }


def _analyze_accumulator(
    accumulator: SplitCrossMomentAccumulator,
    *,
    granularity: str,
    ranks: tuple[int, ...],
    ridge_ratio: float,
) -> list[dict]:
    tensors = _granularity_tensors(accumulator, granularity)
    counts = tensors[0]
    split_corr, split_teacher, split_student, split_valid = _batched_cca(
        *tensors,
        ridge_ratio=ridge_ratio,
    )
    full_tensors = tuple(tensor.sum(dim=0) for tensor in tensors)
    full_corr, _, _, full_valid = _batched_cca(
        *full_tensors,
        ridge_ratio=ridge_ratio,
    )
    group_shape = tuple(counts.shape[1:])
    groups = []
    for coordinates in itertools.product(*(range(size) for size in group_shape)):
        if not bool(full_valid[coordinates]):
            continue
        split_systems = []
        for split in range(accumulator.num_splits):
            index = (split, *coordinates)
            if bool(split_valid[index]):
                split_systems.append(
                    (
                        split_corr[index],
                        split_teacher[index],
                        split_student[index],
                    )
                )
            else:
                split_systems.append(None)
        group = {
            "layer": coordinates[0],
            "head": coordinates[1],
            "count": int(full_tensors[0][coordinates]),
            **_correlation_metrics(full_corr[coordinates], ranks),
            "stability": _split_stability(split_systems, ranks),
        }
        if granularity == "position":
            group["position"] = coordinates[2]
        groups.append(group)
    return groups


def _group_key(group: dict, granularity: str) -> tuple[int, ...]:
    key = (int(group["layer"]), int(group["head"]))
    return key + ((int(group["position"]),) if granularity == "position" else ())


def _aggregate_groups(groups: list[dict], ranks: tuple[int, ...]) -> dict:
    return {
        "groups": len(groups),
        "median_effective_correlation_rank": _safe_median(
            [float(group["effective_correlation_rank"]) for group in groups]
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
                "median_mean_canonical_correlation": _safe_median(
                    [
                        float(
                            group["ranks"][str(rank)][
                                "mean_canonical_correlation"
                            ]
                        )
                        for group in groups
                    ]
                ),
                "median_teacher_split_overlap": _safe_median(
                    [
                        float(
                            group["stability"]["ranks"][str(rank)][
                                "teacher_overlap_mean"
                            ]
                        )
                        for group in groups
                        if group["stability"]["ranks"][str(rank)][
                            "teacher_overlap_mean"
                        ]
                        is not None
                    ]
                ),
                "median_student_split_overlap": _safe_median(
                    [
                        float(
                            group["stability"]["ranks"][str(rank)][
                                "student_overlap_mean"
                            ]
                        )
                        for group in groups
                        if group["stability"]["ranks"][str(rank)][
                            "student_overlap_mean"
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
    return [
        {
            "position": position,
            **_aggregate_groups(
                [group for group in groups if int(group["position"]) == position],
                ranks,
            ),
        }
        for position in sorted({int(group["position"]) for group in groups})
    ]


def analyze_cross_moment_collection(
    collection: dict,
    *,
    ranks: Iterable[int] = (1, 2, 4, 8, 16),
    gate_rank: int = 4,
    ridge_ratio: float = 1e-4,
    correlation_margin: float = 0.05,
    overlap_margin: float = 0.10,
    required_group_fraction: float = 0.60,
) -> dict:
    ranks = tuple(sorted({int(rank) for rank in ranks}))
    exemplar = collection["actual"]["key"]
    if not ranks or min(ranks) <= 0 or max(ranks) > exemplar.head_dim:
        raise ValueError("ranks must be positive and no larger than head_dim")
    if gate_rank not in ranks:
        raise ValueError("gate_rank must be included in ranks")

    detailed = {}
    for pairing in PAIRINGS:
        detailed[pairing] = {}
        for kind in KV_KINDS:
            detailed[pairing][kind] = {}
            for granularity in GRANULARITIES:
                groups = _analyze_accumulator(
                    collection[pairing][kind],
                    granularity=granularity,
                    ranks=ranks,
                    ridge_ratio=ridge_ratio,
                )
                detailed[pairing][kind][granularity] = {
                    "summary": _aggregate_groups(groups, ranks),
                    "by_position": _position_summary(groups, ranks),
                    "groups": groups,
                }

    rank_key = str(gate_rank)
    comparisons = {}
    gates = {}
    for kind in KV_KINDS:
        comparisons[kind] = {}
        for granularity in GRANULARITIES:
            actual = detailed["actual"][kind][granularity]["groups"]
            shuffled = {
                _group_key(group, granularity): group
                for group in detailed["shuffled"][kind][granularity]["groups"]
            }
            correlation_deltas = []
            overlap_deltas = []
            wins = 0
            matched = 0
            for group in actual:
                null = shuffled.get(_group_key(group, granularity))
                if null is None:
                    continue
                actual_rank = group["ranks"][rank_key]
                null_rank = null["ranks"][rank_key]
                actual_stability = group["stability"]["ranks"][rank_key]
                null_stability = null["stability"]["ranks"][rank_key]
                values = (
                    actual_stability["teacher_overlap_mean"],
                    actual_stability["student_overlap_mean"],
                    null_stability["teacher_overlap_mean"],
                    null_stability["student_overlap_mean"],
                )
                if any(value is None for value in values):
                    continue
                correlation_delta = float(
                    actual_rank["mean_canonical_correlation"]
                    - null_rank["mean_canonical_correlation"]
                )
                teacher_delta = float(values[0] - values[2])
                student_delta = float(values[1] - values[3])
                minimum_overlap_delta = min(teacher_delta, student_delta)
                matched += 1
                wins += int(
                    correlation_delta > 0
                    and teacher_delta > 0
                    and student_delta > 0
                )
                correlation_deltas.append(correlation_delta)
                overlap_deltas.append(minimum_overlap_delta)
            comparison = {
                "matched_groups": matched,
                "fraction_stronger_and_more_stable_than_shuffle": (
                    wins / matched if matched else None
                ),
                "median_canonical_correlation_delta": _safe_median(
                    correlation_deltas
                ),
                "median_minimum_teacher_student_overlap_delta": _safe_median(
                    overlap_deltas
                ),
            }
            comparisons[kind][granularity] = comparison

        pooled = comparisons[kind]["pooled"]
        supported = bool(
            pooled["matched_groups"]
            and pooled["fraction_stronger_and_more_stable_than_shuffle"]
            >= required_group_fraction
            and pooled["median_canonical_correlation_delta"]
            >= correlation_margin
            and pooled["median_minimum_teacher_student_overlap_delta"]
            >= overlap_margin
        )
        gates[kind] = {"supported": supported, **pooled}

    if all(gate["supported"] for gate in gates.values()):
        status = "paired_signal_supported_for_keys_and_values"
    elif any(gate["supported"] for gate in gates.values()):
        status = "paired_signal_supported_for_one_kv_kind"
    else:
        status = "paired_signal_not_supported_by_gate"
    return {
        "schema_version": CROSS_STATISTICS_SCHEMA_VERSION,
        "analysis": "whitened_teacher_student_kv_cross_covariance",
        "ranks": list(ranks),
        "ridge_ratio": ridge_ratio,
        "gate": {
            "status": status,
            "rank": gate_rank,
            "correlation_margin": correlation_margin,
            "overlap_margin": overlap_margin,
            "required_group_fraction": required_group_fraction,
            "by_kind": gates,
            "interpretation": (
                "This gate isolates reproducible paired linear dependence. It does "
                "not establish answer causality or downstream accuracy."
            ),
        },
        "comparisons": comparisons,
        "pairings": detailed,
    }


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def render_cross_subspace_markdown(report: dict, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    gate = report["gate"]
    rank = str(gate["rank"])
    lines = [
        "# Stage 1b paired KV cross-subspace analysis",
        "",
        "## Outcome",
        "",
        f"Predefined diagnostic gate: **{gate['status'].replace('_', ' ')}**.",
        "",
        (
            "Whitened teacher–student cross-covariance isolates paired linear "
            "dependence that residual covariance could not distinguish from stable "
            "marginal structure."
        ),
        "",
        "## Calibration contract",
        "",
        f"- Checkpoint step: {metadata.get('checkpoint_step', 'unknown')}",
        f"- Processed examples: {metadata.get('processed_examples', 'unknown')}",
        f"- Split halves: {metadata.get('num_splits', 'unknown')}",
        f"- Shuffle repeats per batch: {metadata.get('shuffle_repeats', 'unknown')}",
        f"- Ridge ratio: {report['ridge_ratio']}",
        "",
        "## Pooled layer-head comparison",
        "",
        "| KV kind | Paired correlation | Shuffled correlation | "
        "Paired teacher overlap | Paired student overlap | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for kind in KV_KINDS:
        actual = report["pairings"]["actual"][kind]["pooled"]["summary"]["ranks"][
            rank
        ]
        shuffled = report["pairings"]["shuffled"][kind]["pooled"]["summary"][
            "ranks"
        ][rank]
        lines.append(
            "| {kind} | {actual_corr} | {null_corr} | {teacher} | {student} | "
            "{status} |".format(
                kind=kind,
                actual_corr=_format(
                    actual["median_mean_canonical_correlation"]
                ),
                null_corr=_format(
                    shuffled["median_mean_canonical_correlation"]
                ),
                teacher=_format(actual["median_teacher_split_overlap"]),
                student=_format(actual["median_student_split_overlap"]),
                status=(
                    "supported"
                    if gate["by_kind"][kind]["supported"]
                    else "not supported"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Position-resolved paired signal",
            "",
            "| KV kind | Position | Mean top-r correlation | Teacher overlap | "
            "Student overlap |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for kind in KV_KINDS:
        for position in report["pairings"]["actual"][kind]["position"][
            "by_position"
        ]:
            values = position["ranks"][rank]
            lines.append(
                "| {kind} | {position} | {corr} | {teacher} | {student} |".format(
                    kind=kind,
                    position=position["position"],
                    corr=_format(
                        values["median_mean_canonical_correlation"]
                    ),
                    teacher=_format(values["median_teacher_split_overlap"]),
                    student=_format(values["median_student_split_overlap"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "A positive result supports a subsequent low-rank distillation "
                "ablation. It still requires a projection-versus-compute-matched "
                "training experiment before making a performance claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)
