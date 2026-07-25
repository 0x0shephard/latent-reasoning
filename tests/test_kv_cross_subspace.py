"""Regression tests for paired teacher/student KV cross-subspace analysis."""
from __future__ import annotations

import torch

from src.mech.kv_cross_subspace import (
    SplitCrossMomentAccumulator,
    analyze_cross_moment_collection,
    create_cross_moment_collection,
    cross_moment_collection_from_state,
    cross_moment_collection_state,
)


def test_cross_moments_equal_direct_valid_pair_statistics():
    teacher = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(
        4, 1, 1, 2, 3
    )
    student = teacher.flip(-1) + 2
    mask = torch.tensor(
        [[1, 1], [1, 0], [1, 1], [0, 1]], dtype=torch.bool
    ).reshape(4, 1, 1, 2)
    splits = torch.tensor([0, 1, 0, 1])
    accumulator = SplitCrossMomentAccumulator.create(
        num_splits=2,
        layers=1,
        heads=1,
        positions=2,
        head_dim=3,
    )
    accumulator.update(teacher, student, mask, splits)

    teacher_rows = teacher.reshape(-1, 3)[mask.reshape(-1)]
    student_rows = student.reshape(-1, 3)[mask.reshape(-1)]
    assert int(accumulator.pooled_count.sum()) == len(teacher_rows)
    assert torch.equal(
        accumulator.pooled_cross.sum(dim=0)[0, 0],
        teacher_rows.T @ student_rows,
    )
    assert torch.equal(
        accumulator.pooled_teacher_gram.sum(dim=0)[0, 0],
        teacher_rows.T @ teacher_rows,
    )
    restored = SplitCrossMomentAccumulator.from_state_dict(
        accumulator.state_dict()
    )
    assert torch.equal(restored.position_cross, accumulator.position_cross)


def test_cca_gate_recovers_stable_paired_signal_and_rejects_shuffle():
    torch.manual_seed(19)
    batch, layers, heads, positions, dimension = 800, 2, 2, 2, 8
    split_ids = torch.cat(
        [torch.zeros(batch // 2), torch.ones(batch // 2)]
    ).long()
    mask = torch.ones(batch, layers, heads, positions, dtype=torch.bool)
    latent = torch.randn(batch, layers, heads, positions, 2)
    teacher = 0.15 * torch.randn(
        batch, layers, heads, positions, dimension
    )
    student = 0.15 * torch.randn_like(teacher)
    teacher[..., 0] += 2.0 * latent[..., 0]
    teacher[..., 1] += 1.5 * latent[..., 1]
    student[..., 0] += 1.6 * latent[..., 0] + 0.2 * latent[..., 1]
    student[..., 1] += 1.3 * latent[..., 1]

    # Derange independently inside each split so the null preserves both marginals.
    permutation = torch.arange(batch)
    permutation[: batch // 2] = permutation[: batch // 2].roll(1)
    permutation[batch // 2 :] = permutation[batch // 2 :].roll(1)
    shuffled_teacher = teacher.index_select(0, permutation)
    shuffled_mask = mask.index_select(0, permutation)

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
            shuffled_teacher,
            student,
            shuffled_mask,
            split_ids,
        )
    restored = cross_moment_collection_from_state(
        cross_moment_collection_state(collection)
    )
    report = analyze_cross_moment_collection(
        restored,
        ranks=(1, 2, 4),
        gate_rank=2,
        correlation_margin=0.05,
        overlap_margin=0.10,
        required_group_fraction=0.60,
    )
    assert (
        report["gate"]["status"]
        == "paired_signal_supported_for_keys_and_values"
    )
    assert (
        report["comparisons"]["key"]["pooled"][
            "median_canonical_correlation_delta"
        ]
        > 0.5
    )
