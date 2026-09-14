import torch

from src.mech.causal_xkv import (
    causal_protected_factorization,
    compress_reconstruct_cache,
    concatenate_group_cache,
    mapped_group_causal_basis,
    mapped_group_independent_kv_basis,
    mapped_group_variable_causal_basis,
    reduced_attention,
    split_group_cache,
    truncated_svd_factorization,
)
from scripts.run_codi_causal_xkv import groups_around_validated_run


def test_group_cache_round_trip():
    keys = [torch.randn(9, 8) for _ in range(3)]
    values = [torch.randn(9, 8) for _ in range(3)]
    matrix = concatenate_group_cache(keys, values)
    actual_keys, actual_values = split_group_cache(matrix, layers=3, width=8)
    for expected, actual in zip(keys + values, actual_keys + actual_values):
        assert torch.equal(expected, actual)


def test_protected_factorization_exactly_preserves_core_coordinates():
    generator = torch.Generator().manual_seed(13)
    matrix = torch.randn(40, 24, generator=generator)
    protected, _ = torch.linalg.qr(torch.randn(24, 4, generator=generator))
    factor = causal_protected_factorization(matrix, protected, rank=10)
    reconstructed = factor.reconstruct()
    assert factor.protected_rank == 4
    assert torch.allclose(reconstructed @ protected, matrix @ protected, atol=2e-5)
    ordinary = truncated_svd_factorization(matrix, 10)
    # Both arms obey exactly the same rank/storage budget.
    assert factor.storage_elements == ordinary.storage_elements


def test_mapped_group_basis_and_reduced_attention_identity():
    generator = torch.Generator().manual_seed(17)
    keys = [torch.randn(8, 3, generator=generator) for _ in range(2)]
    values = [torch.randn(8, 3, generator=generator) for _ in range(2)]
    protected = mapped_group_causal_basis(keys, values)
    assert protected.shape == (32, 3)
    assert torch.allclose(protected.T @ protected, torch.eye(3), atol=1e-5)

    code = torch.randn(23, 5, generator=generator)
    key_decoder = torch.randn(5, 8, generator=generator)
    value_decoder = torch.randn(5, 8, generator=generator)
    query = torch.randn(4, 8, generator=generator)
    expected = torch.softmax(query @ (code @ key_decoder).T / (8**0.5), dim=-1) @ (code @ value_decoder)
    assert torch.allclose(
        reduced_attention(query, code, key_decoder, value_decoder), expected, atol=2e-5
    )


def test_variable_group_basis_keeps_every_layer_local_direction():
    keys = [torch.eye(8)[:, :1], torch.eye(8)[:, :3]]
    values = [torch.eye(8)[:, 1:2], torch.eye(8)[:, 3:6]]
    protected = mapped_group_variable_causal_basis(keys, values)
    assert protected.shape == (32, 4)
    assert torch.allclose(protected.T @ protected, torch.eye(4), atol=1e-5)


def test_independent_group_basis_preserves_distinct_key_and_value_ranks():
    keys = [torch.eye(8)[:, :1], torch.eye(8)[:, :2]]
    values = [torch.eye(8)[:, 2:5], torch.eye(8)[:, 5:6]]
    protected = mapped_group_independent_kv_basis(keys, values)
    assert protected.shape == (32, 7)
    assert torch.allclose(protected.T @ protected, torch.eye(7), atol=1e-5)


def test_legacy_cache_compression_preserves_shape_and_reports_modelled_storage():
    generator = torch.Generator().manual_seed(31)
    cache = tuple(
        (torch.randn(1, 2, 12, 4, generator=generator),
         torch.randn(1, 2, 12, 4, generator=generator))
        for _ in range(4)
    )
    reconstructed, report = compress_reconstruct_cache(
        cache, groups=((0, 1), (2, 3)), rank=6
    )
    assert len(reconstructed) == len(cache)
    assert all(a.shape == b.shape for old, new in zip(cache, reconstructed) for a, b in zip(old, new))
    assert report["runtime_representation"] == "dense_reconstruction_quality_proxy"
    assert report["modelled_compression_ratio"] > 1


def test_groups_keep_validated_run_intact_and_partition_layers():
    groups = groups_around_validated_run((7, 8, 9))
    assert (7, 8, 9) in groups
    flattened = [layer for group in groups for layer in group]
    assert flattened == list(range(12))
