"""Frozen selection rules for the post-rank-16 xKV fidelity frontier."""
from __future__ import annotations

from collections.abc import Mapping

import torch


def normalized_positive_utilities(
    utilities: Mapping[tuple[int, ...], float],
) -> dict[tuple[int, ...], float]:
    if not utilities:
        raise ValueError("utilities cannot be empty")
    values = {tuple(key): float(value) for key, value in utilities.items()}
    if any(not value > 0 for value in values.values()):
        raise ValueError("utilities must be positive")
    mean = sum(values.values()) / len(values)
    return {key: value / mean for key, value in values.items()}


def blend_group_utilities(
    answer_utilities: Mapping[tuple[int, ...], float],
    variance_utilities: Mapping[tuple[int, ...], float],
    *,
    task_weight: float,
) -> dict[tuple[int, ...], float]:
    """Blend normalized answer-Fisher and cache-variance group utilities."""
    weight = float(task_weight)
    if not 0 <= weight <= 1:
        raise ValueError("task_weight must lie in [0,1]")
    answer = normalized_positive_utilities(answer_utilities)
    variance = normalized_positive_utilities(variance_utilities)
    if set(answer) != set(variance):
        raise ValueError("answer and variance utilities must use identical groups")
    return {
        group: weight * answer[group] + (1 - weight) * variance[group]
        for group in answer
    }


def blend_feature_weights(
    feature_weights: Mapping[tuple[int, ...], torch.Tensor],
    *,
    task_weight: float,
) -> dict[tuple[int, ...], torch.Tensor] | None:
    """Interpolate Fisher feature weights toward unweighted reconstruction."""
    weight = float(task_weight)
    if not 0 <= weight <= 1:
        raise ValueError("task_weight must lie in [0,1]")
    if weight == 0:
        return None
    result = {}
    for group, values in feature_weights.items():
        if values.ndim != 1 or not bool(torch.isfinite(values).all()):
            raise ValueError("feature weights must be finite vectors")
        result[tuple(group)] = torch.ones_like(values) + weight * (values - 1)
    return result


def select_frontier_candidate(records: list[dict]) -> dict | None:
    """Select the highest-compression preregistered candidate that passes screening."""
    eligible = [record for record in records if bool(record.get("screen_passed"))]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            float(record["modelled_compression_ratio"]),
            float(record["nll_bootstrap_95ci"][0]),
            float(record["task_weight"]),
        ),
    )


def fidelity_gate(
    *,
    nll_interval: list[float],
    cache_bits_ratio: float,
    first_token_fidelity: float,
    dense_accuracy_interval: list[float],
    ordinary_accuracy_interval: list[float],
    storage_tolerance: float = 0.001,
    minimum_first_token_fidelity: float = 0.95,
    accuracy_noninferiority_margin: float = 0.02,
) -> dict[str, bool]:
    checks = {
        "positive_nll_interval": float(nll_interval[0]) > 0,
        "storage_matched": abs(float(cache_bits_ratio) - 1.0)
        <= float(storage_tolerance),
        "first_token_fidelity": float(first_token_fidelity)
        >= float(minimum_first_token_fidelity),
        "dense_accuracy_noninferior": float(dense_accuracy_interval[0])
        > -float(accuracy_noninferiority_margin),
        "ordinary_accuracy_noninferior": float(ordinary_accuracy_interval[0])
        > -float(accuracy_noninferiority_margin),
    }
    return {**checks, "passed": all(checks.values())}
