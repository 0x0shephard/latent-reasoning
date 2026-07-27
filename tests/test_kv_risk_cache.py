import torch

from src.mech.kv_risk_cache import (
    attention_to_kv_heads,
    gather_cache_axis,
    select_generated_indices,
)


def test_attention_to_kv_heads_averages_query_groups():
    attention = torch.tensor(
        [[[[1.0, 2.0]], [[3.0, 4.0]], [[10.0, 20.0]], [[30.0, 40.0]]]]
    )
    grouped = attention_to_kv_heads(attention, kv_heads=2)
    assert grouped.shape == (1, 2, 2)
    assert torch.equal(
        grouped,
        torch.tensor([[[2.0, 3.0], [20.0, 30.0]]]),
    )


def test_selector_preserves_prompt_heavy_hitters_and_recent_token():
    positions = torch.arange(7).reshape(1, 1, 7)
    scores = torch.tensor([[[0.0, 0.0, 0.1, 9.0, 8.0, 0.2, 0.0]]])
    indices = select_generated_indices(
        scores,
        positions,
        prompt_length=2,
        target_generated=3,
        recent_window=1,
        heavy_fraction=2 / 3,
    )
    assert indices.tolist() == [[[0, 1, 3, 4, 6]]]


def test_cache_gather_can_select_different_positions_per_head():
    tensor = torch.arange(1 * 2 * 4 * 1).reshape(1, 2, 4, 1)
    indices = torch.tensor([[[0, 3], [1, 2]]])
    gathered = gather_cache_axis(tensor, indices)
    assert gathered.shape == (1, 2, 2, 1)
    assert gathered.flatten().tolist() == [0, 3, 5, 6]

