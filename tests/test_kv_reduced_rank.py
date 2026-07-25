"""Regression tests for cross-validated reduced-rank KV prediction."""
from __future__ import annotations

import torch

from src.mech.kv_cross_subspace import create_cross_moment_collection
from src.mech.kv_reduced_rank import analyze_reduced_rank_prediction


def _synthetic_collection(*, paired: bool) -> dict:
    torch.manual_seed(37)
    batch, layers, heads, positions, dimension = 1200, 2, 2, 3, 8
    split_ids = torch.cat(
        [torch.zeros(batch // 2), torch.ones(batch // 2)]
    ).long()
    mask = torch.ones(batch, layers, heads, positions, dtype=torch.bool)
    student = torch.randn(batch, layers, heads, positions, dimension)
    teacher = torch.empty_like(student)
    for position in range(positions):
        # Each position has a different rank-two map. Preserving position should
        # recover it; pooling intentionally mixes incompatible maps.
        mapping = torch.zeros(dimension, dimension)
        mapping[0, position] = 1.8
        mapping[1, (position + 3) % dimension] = 1.4
        teacher[..., position, :] = (
            student[..., position, :] @ mapping
            + 0.10
            * torch.randn(batch, layers, heads, dimension)
        )
    if not paired:
        teacher = torch.randn_like(teacher)

    permutation = torch.arange(batch)
    permutation[: batch // 2] = permutation[: batch // 2].roll(1)
    permutation[batch // 2 :] = permutation[batch // 2 :].roll(1)
    collection = create_cross_moment_collection(
        num_splits=2,
        layers=layers,
        heads=heads,
        positions=positions,
        head_dim=dimension,
    )
    for kind in ("key", "value"):
        collection["actual"][kind].update(
            teacher, student, mask, split_ids
        )
        collection["shuffled"][kind].update(
            teacher.index_select(0, permutation),
            student,
            mask.index_select(0, permutation),
            split_ids,
        )
    return collection


def test_position_conditioned_rank_two_prediction_passes_for_low_rank_signal():
    report = analyze_reduced_rank_prediction(
        _synthetic_collection(paired=True),
        ranks=(1, 2, 4),
        gate_rank=2,
        r2_margin=0.02,
        minimum_median_r2=0.05,
        minimum_full_retention=0.80,
    )
    assert (
        report["gate"]["status"]
        == "low_rank_prediction_supported_for_keys_and_values"
    )
    comparison = report["comparisons"]["key"]["position"]
    assert comparison["fraction_actual_r2_above_shuffle"] == 1.0
    assert comparison["median_actual_heldout_r2"] > 0.95
    assert comparison["median_rank_retention_of_full_r2"] > 0.99


def test_independent_noise_is_rejected():
    report = analyze_reduced_rank_prediction(
        _synthetic_collection(paired=False),
        ranks=(1, 2, 4),
        gate_rank=2,
    )
    assert report["gate"]["status"] == "low_rank_prediction_not_supported_by_gate"
    assert report["comparisons"]["value"]["position"][
        "median_actual_heldout_r2"
    ] < 0.05
