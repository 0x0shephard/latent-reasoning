from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.mech.cross_model_head_benchmark import (
    benchmark_head_latency,
    diagonal_activation_scale,
    evaluate_head_quality,
    head_from_factors,
    head_macs,
    randomized_scaled_svd_factors,
    rare_token_mask_from_counts,
)


def test_full_rank_weight_svd_reconstructs_dense_readout():
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(13, 6, generator=generator, dtype=torch.float64)
    bias = torch.randn(13, generator=generator, dtype=torch.float64)
    centre, down, up, output_bias, report = randomized_scaled_svd_factors(
        weight, 6, readout_bias=bias, oversample=0, power_iterations=0,
        seed=2, compute_dtype=torch.float64,
    )
    head = head_from_factors(centre, down, up, output_bias)
    states = torch.randn(19, 6, generator=generator, dtype=torch.float64)
    assert report.method == "weight_svd"
    assert torch.allclose(head(states), F.linear(states, weight, bias), atol=1e-8, rtol=1e-8)


def test_diagonal_scale_is_positive_and_geometrically_normalized():
    states = torch.randn(40, 8, generator=torch.Generator().manual_seed(3))
    centre, scale = diagonal_activation_scale(states)
    assert centre.shape == scale.shape == (8,)
    assert bool((scale > 0).all())
    assert math.isclose(float(scale.log().mean()), 0.0, abs_tol=1e-5)


def test_quality_metrics_are_exact_for_dense_equivalent_head():
    generator = torch.Generator().manual_seed(19)
    states = torch.randn(32, 5, generator=generator)
    weight = torch.randn(17, 5, generator=generator)
    targets = torch.randint(0, 17, (32,), generator=generator)
    centre, down, up, output_bias, _ = randomized_scaled_svd_factors(
        weight, 5, oversample=0, power_iterations=0, seed=4,
    )
    head = head_from_factors(centre, down, up, output_bias)
    counts = torch.bincount(targets, minlength=17)
    metrics = evaluate_head_quality(
        head, states, targets, weight,
        rare_token_mask=rare_token_mask_from_counts(counts, maximum_count=1),
        batch_size=7, top_k=5,
    )
    assert metrics["top1_agreement"] == 1.0
    assert metrics["topk_overlap_fraction"] == 1.0
    assert metrics["teacher_kl"] < 1e-7
    assert math.isclose(metrics["perplexity_ratio"], 1.0, rel_tol=1e-6)


def test_head_operation_formula_and_cpu_benchmark():
    assert head_macs(768, 50257) == 38_597_376
    assert head_macs(768, 50257, 96) == 4_898_400
    module = torch.nn.Linear(8, 11)
    latency = benchmark_head_latency(
        module, 8, batch_size=1, iterations=3, warmup=1,
        device="cpu", dtype=torch.float32,
    )
    assert latency > 0
