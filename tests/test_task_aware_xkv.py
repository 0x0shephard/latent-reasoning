import torch

from src.mech.task_aware_xkv import (
    allocate_rank_budget,
    compress_reconstruct_task_aware_cache,
    gradient_rms_group_metrics,
    quantize_reconstruct_cache,
    weighted_svd_factorization,
)


def _flatten(cache, layer, kind):
    tensor = cache[layer][0 if kind == "key" else 1]
    return tensor[0].transpose(0, 1).reshape(tensor.shape[2], -1)


def test_weighted_factorization_obeys_rank_and_storage_contract():
    generator = torch.Generator().manual_seed(3)
    matrix = torch.randn(12, 10, generator=generator)
    factor = weighted_svd_factorization(
        matrix, 4, feature_weights=torch.linspace(0.5, 2.0, 10)
    )
    assert factor.code.shape == (12, 4)
    assert factor.decoder.shape == (4, 10)
    assert factor.storage_bits == (12 * 4 + 4 * 10) * 16
    quantized = weighted_svd_factorization(matrix, 4, quantization_bits=4)
    assert quantized.storage_bits < factor.storage_bits


def test_rank_allocator_preserves_total_budget_and_caps():
    ranks = allocate_rank_budget(
        [9.0, 1.0, 4.0], total_rank=12, caps=[8, 8, 8], minimum_rank=1
    )
    assert sum(ranks) == 12
    assert all(1 <= rank <= 8 for rank in ranks)
    assert ranks[0] > ranks[1]


def test_gradient_metrics_return_group_feature_weights_and_utility():
    generator = torch.Generator().manual_seed(5)
    key = torch.randn(6, 4, 3, 8, generator=generator)
    value = 2 * torch.randn(6, 4, 3, 8, generator=generator)
    groups = ((0, 1), (2, 3))
    weights, utilities = gradient_rms_group_metrics(key, value, groups)
    assert weights[(0, 1)].shape == (32,)
    assert weights[(2, 3)].shape == (32,)
    assert all(bool((row > 0).all()) for row in weights.values())
    assert all(value > 0 for value in utilities.values())


def test_sparse_residual_exactly_restores_protected_latent_coordinates():
    generator = torch.Generator().manual_seed(7)
    cache = tuple(
        (
            torch.randn(1, 2, 9, 4, generator=generator),
            torch.randn(1, 2, 9, 4, generator=generator),
        )
        for _ in range(4)
    )
    basis = torch.eye(8)[:, :2]
    reconstructed, report = compress_reconstruct_task_aware_cache(
        cache,
        groups=((0, 1), (2, 3)),
        rank=3,
        protected_bases={(1, "value"): basis},
        latent_positions=2,
    )
    original = _flatten(cache, 1, "value")
    actual = _flatten(reconstructed, 1, "value")
    assert torch.allclose(actual[-2:] @ basis, original[-2:] @ basis, atol=2e-5)
    assert report["protected_residual_bits"] == 2 * 2 * 16
    assert report["modelled_compression_ratio"] > 1
    assert {tuple(row["group"]) for row in report["records"]} == {(0, 1), (2, 3)}


def test_dense_quantization_preserves_cache_shape_and_models_scale_overhead():
    generator = torch.Generator().manual_seed(11)
    cache = tuple(
        (
            torch.randn(1, 2, 7, 4, generator=generator),
            torch.randn(1, 2, 7, 4, generator=generator),
        )
        for _ in range(3)
    )
    reconstructed, report = quantize_reconstruct_cache(cache, bits=4)
    assert all(
        old_tensor.shape == new_tensor.shape
        for old_entry, new_entry in zip(cache, reconstructed)
        for old_tensor, new_tensor in zip(old_entry, new_entry)
    )
    # Four data bits plus one FP16 scale per four-value head = 8 bits/value.
    assert report["modelled_compression_ratio"] == 2
    assert report["quantization_bits"] == 4
