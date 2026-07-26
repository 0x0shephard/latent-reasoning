"""Cross-validated reduced-rank prediction for paired teacher/student KV states.

Stage 1b tests whether teacher and student KV states share canonical directions.
This module asks the stricter predictive question: can a low-rank map fitted from
student KV states on one data split predict teacher KV states on an untouched split?

Only sufficient statistics are required.  No examples, model weights, or GPU are
needed after Stage 1b extraction.
"""
from __future__ import annotations

import itertools
import statistics
from typing import Iterable

import torch

from src.mech.kv_cross_subspace import (
    GRANULARITIES,
    KV_KINDS,
    PAIRINGS,
    SplitCrossMomentAccumulator,
    _granularity_tensors,
)


REDUCED_RANK_REPORT_SCHEMA_VERSION = 1


def _safe_mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _safe_median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) > 1 else None


def _centered_training_moments(
    count: torch.Tensor,
    teacher_sum: torch.Tensor,
    student_sum: torch.Tensor,
    teacher_gram: torch.Tensor,
    student_gram: torch.Tensor,
    cross: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return population moments centered on the training means.

    ``cross`` is stored as teacher-transpose times student.  The returned cross
    moment is student-transpose times teacher, matching ``student @ W``.
    """
    n = float(count)
    if n < 2:
        raise ValueError("at least two training observations are required")
    teacher_sum = teacher_sum.to(torch.float64)
    student_sum = student_sum.to(torch.float64)
    teacher_mean = teacher_sum / n
    student_mean = student_sum / n
    teacher_cov = teacher_gram.to(torch.float64) / n - torch.outer(
        teacher_mean, teacher_mean
    )
    student_cov = student_gram.to(torch.float64) / n - torch.outer(
        student_mean, student_mean
    )
    student_teacher_cov = cross.to(torch.float64).T / n - torch.outer(
        student_mean, teacher_mean
    )
    teacher_cov = 0.5 * (teacher_cov + teacher_cov.T)
    student_cov = 0.5 * (student_cov + student_cov.T)
    return (
        teacher_mean,
        student_mean,
        teacher_cov,
        student_cov,
        student_teacher_cov,
    )


def _test_moments_about_training_means(
    count: torch.Tensor,
    teacher_sum: torch.Tensor,
    student_sum: torch.Tensor,
    teacher_gram: torch.Tensor,
    student_gram: torch.Tensor,
    cross: torch.Tensor,
    *,
    teacher_mean: torch.Tensor,
    student_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center held-out raw moments using means learned on the training split."""
    n = float(count)
    if n < 2:
        raise ValueError("at least two test observations are required")
    teacher_test_mean = teacher_sum.to(torch.float64) / n
    student_test_mean = student_sum.to(torch.float64) / n
    teacher_second = teacher_gram.to(torch.float64) / n
    student_second = student_gram.to(torch.float64) / n
    student_teacher_second = cross.to(torch.float64).T / n

    teacher_cov = (
        teacher_second
        - torch.outer(teacher_test_mean, teacher_mean)
        - torch.outer(teacher_mean, teacher_test_mean)
        + torch.outer(teacher_mean, teacher_mean)
    )
    student_cov = (
        student_second
        - torch.outer(student_test_mean, student_mean)
        - torch.outer(student_mean, student_test_mean)
        + torch.outer(student_mean, student_mean)
    )
    student_teacher_cov = (
        student_teacher_second
        - torch.outer(student_test_mean, teacher_mean)
        - torch.outer(student_mean, teacher_test_mean)
        + torch.outer(student_mean, teacher_mean)
    )
    teacher_cov = 0.5 * (teacher_cov + teacher_cov.T)
    student_cov = 0.5 * (student_cov + student_cov.T)
    return teacher_cov, student_cov, student_teacher_cov


def _fit_reduced_rank_maps(
    student_cov: torch.Tensor,
    student_teacher_cov: torch.Tensor,
    *,
    ranks: tuple[int, ...],
    ridge_ratio: float,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    torch.Tensor,
    float,
]:
    """Fit ridge reduced-rank maps for ``teacher_hat = student @ W``.

    If ``A = (Cov(S) + ridge I)^(1/2) W``, the ridge objective differs from
    ``||A - (Cov(S) + ridge I)^(-1/2) Cov(S,T)||_F^2`` only by a constant.
    Truncated SVD therefore gives the optimal map at every requested rank.
    """
    if ridge_ratio < 0:
        raise ValueError("ridge_ratio must be non-negative")
    dimension = int(student_cov.shape[-1])
    scale = float(torch.trace(student_cov).clamp_min(0.0) / dimension)
    ridge = ridge_ratio * scale + 1e-10
    regularized = student_cov + ridge * torch.eye(
        dimension, dtype=torch.float64
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(regularized)
    inverse_root = (
        eigenvectors
        * torch.rsqrt(eigenvalues.clamp_min(1e-12)).unsqueeze(0)
    ) @ eigenvectors.T
    target = inverse_root @ student_teacher_cov
    left, singular, right_t = torch.linalg.svd(target, full_matrices=False)
    maps = {}
    teacher_bases = {}
    for rank in ranks:
        approximation = (
            left[:, :rank] * singular[:rank].unsqueeze(0)
        ) @ right_t[:rank]
        maps[rank] = inverse_root @ approximation
        teacher_bases[rank] = right_t[:rank].T.contiguous()
    full_map = torch.linalg.solve(regularized, student_teacher_cov)
    return maps, teacher_bases, full_map, ridge


def _prediction_metrics(
    teacher_cov: torch.Tensor,
    student_cov: torch.Tensor,
    student_teacher_cov: torch.Tensor,
    mapping: torch.Tensor,
) -> dict:
    baseline_mse = float(torch.trace(teacher_cov).clamp_min(1e-30))
    mse = float(
        torch.trace(teacher_cov)
        - 2.0 * torch.trace(mapping.T @ student_teacher_cov)
        + torch.trace(mapping.T @ student_cov @ mapping)
    )
    # Numerical cancellation can produce tiny negative values for near-perfect fits.
    mse = max(mse, 0.0)
    return {
        "heldout_r2": float(1.0 - mse / baseline_mse),
        "normalized_mse": float(mse / baseline_mse),
        "teacher_baseline_mse": baseline_mse,
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
    group_shape = tuple(counts.shape[1:])
    groups = []
    for coordinates in itertools.product(*(range(size) for size in group_shape)):
        folds = []
        for train_split, test_split in itertools.permutations(
            range(accumulator.num_splits), 2
        ):
            train_index = (train_split, *coordinates)
            test_index = (test_split, *coordinates)
            if int(counts[train_index]) < 2 or int(counts[test_index]) < 2:
                continue
            training = _centered_training_moments(
                *(tensor[train_index] for tensor in tensors)
            )
            teacher_mean, student_mean, _, student_cov, cross_cov = training
            maps, _, full_map, ridge = _fit_reduced_rank_maps(
                student_cov,
                cross_cov,
                ranks=ranks,
                ridge_ratio=ridge_ratio,
            )
            heldout = _test_moments_about_training_means(
                *(tensor[test_index] for tensor in tensors),
                teacher_mean=teacher_mean,
                student_mean=student_mean,
            )
            teacher_test_cov, student_test_cov, cross_test_cov = heldout
            rank_metrics = {
                str(rank): _prediction_metrics(
                    teacher_test_cov,
                    student_test_cov,
                    cross_test_cov,
                    maps[rank],
                )
                for rank in ranks
            }
            full_metrics = _prediction_metrics(
                teacher_test_cov,
                student_test_cov,
                cross_test_cov,
                full_map,
            )
            folds.append(
                {
                    "train_split": train_split,
                    "test_split": test_split,
                    "train_count": int(counts[train_index]),
                    "test_count": int(counts[test_index]),
                    "ridge": ridge,
                    "ranks": rank_metrics,
                    "full": full_metrics,
                }
            )
        if not folds:
            continue
        group = {
            "layer": coordinates[0],
            "head": coordinates[1],
            "folds": folds,
            "ranks": {},
            "full": {},
        }
        if granularity == "position":
            group["position"] = coordinates[2]
        full_r2 = [float(fold["full"]["heldout_r2"]) for fold in folds]
        group["full"] = {
            "mean_heldout_r2": _safe_mean(full_r2),
            "stdev_heldout_r2": _safe_stdev(full_r2),
        }
        for rank in ranks:
            rank_key = str(rank)
            r2_values = [
                float(fold["ranks"][rank_key]["heldout_r2"]) for fold in folds
            ]
            retention = [
                float(rank_r2 / fold_full)
                for rank_r2, fold_full in zip(r2_values, full_r2)
                if fold_full > 0 and rank_r2 >= 0
            ]
            group["ranks"][rank_key] = {
                "mean_heldout_r2": _safe_mean(r2_values),
                "stdev_heldout_r2": _safe_stdev(r2_values),
                "mean_fraction_of_full_r2": _safe_mean(retention),
            }
        groups.append(group)
    return groups


def fit_position_conditioned_teacher_bases(
    accumulator: SplitCrossMomentAccumulator,
    *,
    rank: int,
    ridge_ratio: float = 1e-4,
) -> torch.Tensor:
    """Fit teacher-space RRR bases from all splits.

    Returns an orthonormal tensor shaped ``[L,H,M,D,R]``. These bases identify
    teacher key or value directions that are most predictable from the aligned
    student states under the Stage 1c ridge objective.
    """
    if rank <= 0 or rank > accumulator.head_dim:
        raise ValueError("rank must be positive and no larger than head_dim")
    tensors = tuple(
        tensor.sum(dim=0)
        for tensor in _granularity_tensors(accumulator, "position")
    )
    basis = torch.empty(
        (
            accumulator.layers,
            accumulator.heads,
            accumulator.positions,
            accumulator.head_dim,
            rank,
        ),
        dtype=torch.float32,
    )
    group_shape = tuple(tensors[0].shape)
    for coordinates in itertools.product(*(range(size) for size in group_shape)):
        training = _centered_training_moments(
            *(tensor[coordinates] for tensor in tensors)
        )
        _, _, _, student_cov, cross_cov = training
        _, teacher_bases, _, _ = _fit_reduced_rank_maps(
            student_cov,
            cross_cov,
            ranks=(rank,),
            ridge_ratio=ridge_ratio,
        )
        basis[coordinates] = teacher_bases[rank].to(torch.float32)
    return basis


def random_orthonormal_bases_like(
    basis: torch.Tensor, *, seed: int
) -> torch.Tensor:
    """Generate deterministic group-wise random orthonormal control bases."""
    if basis.ndim != 5:
        raise ValueError("basis must have shape [L,H,M,D,R]")
    if basis.shape[-1] > basis.shape[-2]:
        raise ValueError("basis rank cannot exceed its feature dimension")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(
        basis.shape,
        generator=generator,
        dtype=torch.float64,
    )
    orthonormal, _ = torch.linalg.qr(random, mode="reduced")
    return orthonormal.to(torch.float32)


def _group_key(group: dict, granularity: str) -> tuple[int, ...]:
    key = (int(group["layer"]), int(group["head"]))
    return key + ((int(group["position"]),) if granularity == "position" else ())


def _aggregate_groups(groups: list[dict], ranks: tuple[int, ...]) -> dict:
    full_values = [
        float(group["full"]["mean_heldout_r2"])
        for group in groups
        if group["full"]["mean_heldout_r2"] is not None
    ]
    return {
        "groups": len(groups),
        "full": {
            "median_heldout_r2": _safe_median(full_values),
            "positive_group_fraction": (
                sum(value > 0 for value in full_values) / len(full_values)
                if full_values
                else None
            ),
        },
        "ranks": {
            str(rank): {
                "median_heldout_r2": _safe_median(
                    [
                        float(group["ranks"][str(rank)]["mean_heldout_r2"])
                        for group in groups
                        if group["ranks"][str(rank)]["mean_heldout_r2"]
                        is not None
                    ]
                ),
                "positive_group_fraction": (
                    sum(
                        float(group["ranks"][str(rank)]["mean_heldout_r2"]) > 0
                        for group in groups
                        if group["ranks"][str(rank)]["mean_heldout_r2"]
                        is not None
                    )
                    / sum(
                        group["ranks"][str(rank)]["mean_heldout_r2"] is not None
                        for group in groups
                    )
                    if any(
                        group["ranks"][str(rank)]["mean_heldout_r2"] is not None
                        for group in groups
                    )
                    else None
                ),
                "median_fraction_of_full_r2": _safe_median(
                    [
                        float(
                            group["ranks"][str(rank)][
                                "mean_fraction_of_full_r2"
                            ]
                        )
                        for group in groups
                        if group["ranks"][str(rank)][
                            "mean_fraction_of_full_r2"
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
                [
                    group
                    for group in groups
                    if int(group["position"]) == position
                ],
                ranks,
            ),
        }
        for position in sorted({int(group["position"]) for group in groups})
    ]


def analyze_reduced_rank_prediction(
    collection: dict,
    *,
    ranks: Iterable[int] = (1, 2, 4, 8, 16),
    gate_rank: int = 4,
    ridge_ratio: float = 1e-4,
    required_group_fraction: float = 0.60,
    r2_margin: float = 0.02,
    minimum_median_r2: float = 0.05,
    minimum_full_retention: float = 0.80,
) -> dict:
    """Analyze held-out reduced-rank prediction and apply a position-wise gate."""
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
            actual_r2_values = []
            r2_deltas = []
            retentions = []
            wins = 0
            matched = 0
            for group in actual:
                null = shuffled.get(_group_key(group, granularity))
                actual_r2 = group["ranks"][rank_key]["mean_heldout_r2"]
                null_r2 = (
                    None
                    if null is None
                    else null["ranks"][rank_key]["mean_heldout_r2"]
                )
                if actual_r2 is None or null_r2 is None:
                    continue
                actual_r2 = float(actual_r2)
                null_r2 = float(null_r2)
                matched += 1
                wins += int(actual_r2 > null_r2)
                actual_r2_values.append(actual_r2)
                r2_deltas.append(actual_r2 - null_r2)
                retention = group["ranks"][rank_key][
                    "mean_fraction_of_full_r2"
                ]
                if retention is not None:
                    retentions.append(float(retention))
            comparisons[kind][granularity] = {
                "matched_groups": matched,
                "fraction_actual_r2_above_shuffle": (
                    wins / matched if matched else None
                ),
                "fraction_actual_r2_positive": (
                    sum(value > 0 for value in actual_r2_values) / matched
                    if matched
                    else None
                ),
                "median_actual_heldout_r2": _safe_median(actual_r2_values),
                "median_actual_minus_shuffle_r2": _safe_median(r2_deltas),
                "median_rank_retention_of_full_r2": _safe_median(retentions),
            }

        # Stage 1b showed that pooling six positions mixes distinct paired
        # subspaces. Stage 1c therefore preregisters position as the primary unit.
        primary = comparisons[kind]["position"]
        supported = bool(
            primary["matched_groups"]
            and primary["fraction_actual_r2_above_shuffle"]
            >= required_group_fraction
            and primary["fraction_actual_r2_positive"]
            >= required_group_fraction
            and primary["median_actual_minus_shuffle_r2"] >= r2_margin
            and primary["median_actual_heldout_r2"] >= minimum_median_r2
            and primary["median_rank_retention_of_full_r2"]
            >= minimum_full_retention
        )
        gates[kind] = {"supported": supported, **primary}

    if all(gate["supported"] for gate in gates.values()):
        status = "low_rank_prediction_supported_for_keys_and_values"
    elif any(gate["supported"] for gate in gates.values()):
        status = "low_rank_prediction_supported_for_one_kv_kind"
    else:
        status = "low_rank_prediction_not_supported_by_gate"
    return {
        "schema_version": REDUCED_RANK_REPORT_SCHEMA_VERSION,
        "analysis": "cross_validated_position_conditioned_reduced_rank_prediction",
        "ranks": list(ranks),
        "ridge_ratio": ridge_ratio,
        "gate": {
            "status": status,
            "primary_granularity": "position",
            "rank": gate_rank,
            "required_group_fraction": required_group_fraction,
            "r2_margin": r2_margin,
            "minimum_median_r2": minimum_median_r2,
            "minimum_full_retention": minimum_full_retention,
            "by_kind": gates,
            "interpretation": (
                "A positive gate supports a low-rank distillation ablation. It "
                "does not establish answer causality or improved task accuracy."
            ),
        },
        "comparisons": comparisons,
        "pairings": detailed,
    }


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def render_reduced_rank_markdown(
    report: dict, metadata: dict | None = None
) -> str:
    metadata = metadata or {}
    gate = report["gate"]
    rank = str(gate["rank"])
    checkpoint = metadata.get("checkpoint_step")
    if checkpoint is None:
        revision = metadata.get("checkpoint_revision")
        checkpoint = (
            f"official release {str(revision)[:8]}" if revision else "unknown"
        )
    lines = [
        "# Stage 1c cross-validated reduced-rank KV prediction",
        "",
        "## Outcome",
        "",
        f"Predefined diagnostic gate: **{gate['status'].replace('_', ' ')}**.",
        "",
        (
            "Maps are fitted on one split and evaluated on the untouched split "
            "in both directions. The primary analysis preserves latent-position "
            "identity because Stage 1b showed that pooling positions mixes "
            "distinct paired subspaces."
        ),
        "",
        "## Calibration contract",
        "",
        f"- Checkpoint: {checkpoint}",
        f"- Processed examples: {metadata.get('processed_examples', 'unknown')}",
        f"- Split halves: {metadata.get('num_splits', 'unknown')}",
        f"- Gate rank: {rank}",
        f"- Ridge ratio: {report['ridge_ratio']}",
        "",
        "## Position-conditioned held-out prediction",
        "",
        "| KV kind | Actual R² | Actual minus shuffle R² | "
        "Groups above shuffle | Positive groups | Rank/full retention | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for kind in KV_KINDS:
        values = report["comparisons"][kind]["position"]
        lines.append(
            "| {kind} | {actual} | {delta} | {wins} | {positive} | "
            "{retention} | {status} |".format(
                kind=kind,
                actual=_format(values["median_actual_heldout_r2"]),
                delta=_format(values["median_actual_minus_shuffle_r2"]),
                wins=_format(values["fraction_actual_r2_above_shuffle"]),
                positive=_format(values["fraction_actual_r2_positive"]),
                retention=_format(
                    values["median_rank_retention_of_full_r2"]
                ),
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
            "## Position-resolved actual prediction",
            "",
            "| KV kind | Position | Rank-r held-out R² | "
            "Rank/full retention | Full-rank held-out R² |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for kind in KV_KINDS:
        for position in report["pairings"]["actual"][kind]["position"][
            "by_position"
        ]:
            rank_values = position["ranks"][rank]
            lines.append(
                "| {kind} | {position} | {r2} | {retention} | {full} |".format(
                    kind=kind,
                    position=position["position"],
                    r2=_format(rank_values["median_heldout_r2"]),
                    retention=_format(
                        rank_values["median_fraction_of_full_r2"]
                    ),
                    full=_format(
                        position["full"]["median_heldout_r2"]
                    ),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "Passing supports testing a position-conditioned spectral target "
                "inside training. Only a projection-versus-full-target, "
                "compute-matched training experiment can establish an accuracy "
                "benefit."
            ),
            "",
        ]
    )
    return "\n".join(lines)
