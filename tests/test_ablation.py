"""Causal latent-state ablation tests."""
from __future__ import annotations

import pytest
import torch

from src.mech.ablation import LatentAblation, parse_positions


def test_parse_positions_validates_latent_budget():
    assert parse_positions("all", 6) is None
    assert parse_positions("0,2,5", 6) == frozenset({0, 2, 5})
    with pytest.raises(ValueError, match="out of range"):
        parse_positions("6", 6)


def test_zero_ablation_only_changes_selected_position():
    state = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    ablation = LatentAblation("zero", positions=frozenset({1}))
    assert torch.equal(ablation(state, 0), state)
    assert torch.equal(ablation(state, 1), torch.zeros_like(state))


def test_batch_mean_removes_example_specific_content():
    state = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    result = LatentAblation("batch_mean")(state, 0)
    assert torch.equal(result, torch.tensor([[3.0, 5.0], [3.0, 5.0]]))


def test_batch_shuffle_is_deterministic_and_not_identity():
    state = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    ablation = LatentAblation("batch_shuffle", seed=13)
    first = ablation(state, 2)
    second = ablation(state, 2)
    assert torch.equal(first, second)
    assert not torch.equal(first, state)
    assert {tuple(row.tolist()) for row in first} == {
        tuple(row.tolist()) for row in state
    }
