import torch

from src.mech.adaptive_kv_allocation import (
    AdaptiveComponentFactorizer,
    allocate_component_ranks,
    blend_component_utilities,
    compress_reconstruct_adaptive_component_cache,
    ordinary_per_layer_budget_bits,
    select_adaptive_candidate,
)


def _utilities(layers=2, width=8):
    result = {}
    for layer in range(layers):
        result[(layer, "key")] = torch.linspace(1.0, 0.1, width)
        result[(layer, "value")] = torch.linspace(4.0, 0.4, width)
    return result


def test_blended_utilities_are_normalized_concave_marginals():
    reconstruction = _utilities()
    fisher = {key: value.flip(0) + 0.2 for key, value in reconstruction.items()}
    blended = blend_component_utilities(
        reconstruction, fisher, answer_weight=0.5
    )
    assert set(blended) == set(reconstruction)
    assert all(bool((curve[:-1] >= curve[1:]).all()) for curve in blended.values())
    assert sum(float(curve.sum()) for curve in blended.values()) > 0


def test_allocator_preserves_budget_and_gives_values_more_rank():
    utilities = _utilities()
    ranks = allocate_component_ranks(
        utilities,
        total_rank=16,
        caps={component: 8 for component in utilities},
        minimum_rank=1,
    )
    assert sum(ranks.values()) == 16
    assert sum(ranks[(layer, "value")] for layer in range(2)) > sum(
        ranks[(layer, "key")] for layer in range(2)
    )


def test_component_factorization_exactly_matches_ordinary_modeled_bits():
    generator = torch.Generator().manual_seed(19)
    cache = tuple(
        (
            torch.randn(1, 2, 10, 4, generator=generator),
            torch.randn(1, 2, 10, 4, generator=generator),
        )
        for _ in range(2)
    )
    utilities = _utilities(layers=2, width=8)
    reconstructed, report = compress_reconstruct_adaptive_component_cache(
        cache,
        baseline_rank=3,
        utilities=utilities,
        maximum_component_rank=8,
    )
    expected = ordinary_per_layer_budget_bits(
        layer_count=2, tokens=10, width=8, rank=3
    )
    assert report["cache_bits"] == expected
    assert report["factor_bits"] + report["budget_padding_bits"] == expected
    assert report["factor_bits"] <= expected
    assert len(report["records"]) == 4
    assert all(
        old.shape == new.shape
        for old_entry, new_entry in zip(cache, reconstructed)
        for old, new in zip(old_entry, new_entry)
    )


def test_factorizer_only_runs_at_the_final_latent_step():
    utilities = _utilities(layers=2, width=8)
    factorizer = AdaptiveComponentFactorizer(
        latent_positions=3,
        baseline_rank=3,
        utilities=utilities,
        maximum_component_rank=8,
    )
    cache = tuple(
        (torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4))
        for _ in range(2)
    )
    assert factorizer(cache, 0) is cache
    factorizer(cache, 2)
    assert factorizer.summary()["batches"] == 1


def test_selection_prefers_compression_then_confirmatory_nll_evidence():
    records = [
        {"name": "a", "screen_passed": True, "modelled_compression_ratio": 1.5,
         "nll_bootstrap_95ci": [0.01, 0.03], "first_token_fidelity": 0.96,
         "answer_weight": 0.0},
        {"name": "b", "screen_passed": True, "modelled_compression_ratio": 2.0,
         "nll_bootstrap_95ci": [0.005, 0.02], "first_token_fidelity": 0.95,
         "answer_weight": 0.5},
        {"name": "c", "screen_passed": False, "modelled_compression_ratio": 3.0,
         "nll_bootstrap_95ci": [0.02, 0.04], "first_token_fidelity": 0.99,
         "answer_weight": 1.0},
    ]
    assert select_adaptive_candidate(records)["name"] == "b"
