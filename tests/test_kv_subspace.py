"""Stage-1 streaming covariance, null, and spectral-gate regression tests."""
from __future__ import annotations

import torch

from scripts.collect_kv_subspaces import _sample_indices
from src.mech.kv_subspace import (
    SplitMomentAccumulator,
    analyze_moment_collection,
    create_moment_collection,
    deterministic_derangement,
    energy_matched_random,
    moment_collection_from_state,
    moment_collection_state,
)


def test_calibration_sample_can_extend_without_reprocessing_prefix():
    first = _sample_indices(100, 20, seed=7)
    extended = _sample_indices(100, 50, seed=7)
    assert extended[:20] == first
    assert len(set(extended)) == 50


def test_streaming_moments_equal_direct_valid_row_statistics():
    values = torch.arange(4 * 1 * 1 * 2 * 3, dtype=torch.float32).reshape(
        4, 1, 1, 2, 3
    )
    mask = torch.tensor(
        [
            [[[[1], [1]]]],
            [[[[1], [0]]]],
            [[[[1], [1]]]],
            [[[[0], [1]]]],
        ],
        dtype=torch.bool,
    ).reshape(4, 1, 1, 2)
    splits = torch.tensor([0, 1, 0, 1])
    accumulator = SplitMomentAccumulator.create(
        num_splits=2,
        layers=1,
        heads=1,
        positions=2,
        head_dim=3,
    )
    accumulator.update(values, mask, splits)

    valid_rows = values.reshape(-1, 3)[mask.reshape(-1)]
    assert int(accumulator.pooled_count.sum()) == valid_rows.shape[0]
    assert torch.equal(
        accumulator.pooled_sum.sum(dim=0)[0, 0],
        valid_rows.sum(dim=0),
    )
    assert torch.equal(
        accumulator.pooled_gram.sum(dim=0)[0, 0],
        valid_rows.T @ valid_rows,
    )
    restored = SplitMomentAccumulator.from_state_dict(accumulator.state_dict())
    assert torch.equal(restored.position_gram, accumulator.position_gram)


def test_derangement_and_energy_matched_random_null_contracts():
    generator = torch.Generator().manual_seed(9)
    permutation = deterministic_derangement(8, generator=generator)
    assert sorted(permutation.tolist()) == list(range(8))
    assert (permutation != torch.arange(8)).all()

    torch.manual_seed(3)
    reference = torch.randn(5, 2, 2, 3, 4)
    mask = torch.ones(5, 2, 2, 3, dtype=torch.bool)
    mask[:, :, :, 2] = False
    randomized = energy_matched_random(
        reference,
        mask,
        generator=torch.Generator().manual_seed(4),
    )
    reference_energy = (
        reference.square() * mask.unsqueeze(-1)
    ).sum(dim=(0, 4))
    randomized_energy = randomized.square().sum(dim=(0, 4))
    assert torch.allclose(reference_energy, randomized_energy, rtol=1e-5)
    assert (randomized[:, :, :, 2] == 0).all()


def test_spectral_gate_recovers_shared_low_rank_subspace_against_nulls():
    torch.manual_seed(31)
    batch = 512
    layers, heads, positions, head_dim = 2, 2, 2, 8
    splits = torch.arange(batch).remainder(2)
    mask = torch.ones(batch, layers, heads, positions, dtype=torch.bool)
    coefficients = torch.randn(batch, layers, heads, positions, 2)
    actual = torch.zeros(batch, layers, heads, positions, head_dim)
    actual[..., :2] = coefficients
    actual += 0.01 * torch.randn_like(actual)
    shuffled_null = torch.randn_like(actual)
    random_null = torch.randn_like(actual)

    collection = create_moment_collection(
        num_splits=2,
        layers=layers,
        heads=heads,
        positions=positions,
        head_dim=head_dim,
    )
    for kind in ("key", "value"):
        collection["actual"][kind].update(actual, mask, splits)
        collection["shuffled"][kind].update(shuffled_null, mask, splits)
        collection["random"][kind].update(random_null, mask, splits)

    restored = moment_collection_from_state(moment_collection_state(collection))
    report = analyze_moment_collection(
        restored,
        ranks=(2, 4),
        gate_rank=4,
        overlap_margin=0.10,
        explained_variance_margin=0.05,
        required_group_fraction=0.60,
    )
    assert report["gate"]["status"] == "supported_for_keys_and_values"
    assert report["gate"]["by_kind"]["key"]["supported"]
    assert (
        report["comparisons"]["key"]["pooled"][
            "median_overlap_delta_vs_strongest_null"
        ]
        > 0.10
    )
