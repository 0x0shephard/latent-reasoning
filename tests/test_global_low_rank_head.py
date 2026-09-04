from __future__ import annotations

import torch
import torch.nn.functional as F

from src.mech.global_low_rank_head import (
    NestedLowRankVocabularyHead,
    activation_cholesky,
    activation_whitened_factors,
    distil_nested_head,
    evaluate_nested_head,
)


def test_activation_cholesky_reconstructs_regularized_covariance():
    generator = torch.Generator().manual_seed(4)
    states = torch.randn(80, 7, generator=generator, dtype=torch.float64)
    centre, factor, ridge = activation_cholesky(
        states, compute_dtype=torch.float64, ridge_relative=1e-5
    )
    centred = states - centre
    expected = centred.T @ centred / (len(states) - 1)
    expected += ridge * torch.eye(7, dtype=torch.float64)
    assert torch.allclose(factor @ factor.T, expected, atol=1e-9, rtol=1e-9)


def test_full_rank_whitened_factors_reconstruct_linear_head():
    generator = torch.Generator().manual_seed(9)
    states = torch.randn(96, 6, generator=generator, dtype=torch.float64)
    weight = torch.randn(11, 6, generator=generator, dtype=torch.float64)
    bias = torch.randn(11, generator=generator, dtype=torch.float64)
    centre, down, up, output_bias, _ = activation_whitened_factors(
        states,
        weight,
        rank=6,
        readout_bias=bias,
        oversample=0,
        power_iterations=0,
        seed=3,
        compute_dtype=torch.float64,
    )
    head = NestedLowRankVocabularyHead.from_whitened_factors(
        centre, down, up, output_bias, ranks=(3, 6)
    )
    assert torch.allclose(head.forward_rank(states, 6), F.linear(states, weight, bias),
                          atol=1e-8, rtol=1e-8)


def test_prefix_rank_uses_only_the_requested_factor_columns():
    generator = torch.Generator().manual_seed(12)
    states = torch.randn(40, 5, generator=generator)
    weight = torch.randn(13, 5, generator=generator)
    centre, down, up, output_bias, _ = activation_whitened_factors(
        states, weight, rank=4, oversample=1, power_iterations=1, seed=2
    )
    head = NestedLowRankVocabularyHead.from_whitened_factors(
        centre, down, up, output_bias, ranks=(2, 4)
    )
    observed = head.forward_rank(states, 2)
    coordinates = F.linear(states, head.down.weight[:2], head.down.bias[:2])
    expected = F.linear(coordinates, head.up.weight[:, :2], head.up.bias)
    assert torch.equal(observed, expected)


def test_adaptive_fallback_matches_the_larger_prefix_when_every_row_is_uncertain():
    generator = torch.Generator().manual_seed(15)
    states = torch.randn(32, 5, generator=generator)
    weight = torch.randn(17, 5, generator=generator)
    centre, down, up, output_bias, _ = activation_whitened_factors(
        states, weight, rank=4, oversample=1, seed=8
    )
    head = NestedLowRankVocabularyHead.from_whitened_factors(
        centre, down, up, output_bias, ranks=(2, 4)
    )
    head.configure_adaptive(base_rank=2, fallback_rank=4, margin_threshold=1e9)
    observed = head(states)
    assert head.last_fallback_fraction == 1.0
    assert torch.allclose(observed, head.forward_rank(states, 4), atol=1e-5, rtol=1e-5)


def test_margin_distillation_improves_a_small_whitened_head():
    generator = torch.Generator().manual_seed(21)
    states = torch.randn(160, 8, generator=generator)
    weight = torch.randn(19, 8, generator=generator)
    train, validation = states[:120], states[120:]
    centre, down, up, output_bias, _ = activation_whitened_factors(
        train, weight, rank=4, oversample=4, seed=1
    )
    head = NestedLowRankVocabularyHead.from_whitened_factors(
        centre, down, up, output_bias, ranks=(2, 4)
    )
    before = evaluate_nested_head(head, validation, weight, rank=4, batch_size=10)
    result = distil_nested_head(
        head,
        train,
        validation,
        weight,
        epochs=4,
        batch_size=12,
        learning_rate=3e-3,
        nested_weight=0.25,
        seed=7,
    )
    after = evaluate_nested_head(head, validation, weight, rank=4, batch_size=10)
    assert result.best_epoch >= 0
    assert after["kl"] <= before["kl"]
