import torch

from src.mech.xkv_fidelity_frontier import (
    blend_feature_weights,
    blend_group_utilities,
    fidelity_gate,
    select_frontier_candidate,
)


GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))


def test_task_variance_blend_has_frozen_endpoints():
    answer = dict(zip(GROUPS, (1.0, 2.0, 6.0)))
    variance = dict(zip(GROUPS, (6.0, 2.0, 1.0)))
    assert blend_group_utilities(answer, variance, task_weight=1) == {
        GROUPS[0]: 1 / 3,
        GROUPS[1]: 2 / 3,
        GROUPS[2]: 2.0,
    }
    assert blend_group_utilities(answer, variance, task_weight=0) == {
        GROUPS[0]: 2.0,
        GROUPS[1]: 2 / 3,
        GROUPS[2]: 1 / 3,
    }
    weights = {GROUPS[0]: torch.tensor([0.5, 2.0])}
    assert blend_feature_weights(weights, task_weight=0) is None
    assert torch.allclose(
        blend_feature_weights(weights, task_weight=0.5)[GROUPS[0]],
        torch.tensor([0.75, 1.5]),
    )


def test_frontier_selects_highest_compression_then_strongest_interval():
    records = [
        {"name": "a", "screen_passed": True, "modelled_compression_ratio": 3.0,
         "nll_bootstrap_95ci": [0.02, 0.2], "task_weight": 1.0},
        {"name": "b", "screen_passed": True, "modelled_compression_ratio": 4.0,
         "nll_bootstrap_95ci": [0.01, 0.2], "task_weight": 0.5},
        {"name": "c", "screen_passed": False, "modelled_compression_ratio": 5.0,
         "nll_bootstrap_95ci": [0.10, 0.2], "task_weight": 0.0},
    ]
    assert select_frontier_candidate(records)["name"] == "b"


def test_fidelity_gate_requires_every_preregistered_check():
    passed = fidelity_gate(
        nll_interval=[0.01, 0.1], cache_bits_ratio=1.0002,
        first_token_fidelity=0.96, dense_accuracy_interval=[-0.01, 0.02],
        ordinary_accuracy_interval=[0.0, 0.04],
    )
    assert passed["passed"]
    failed = fidelity_gate(
        nll_interval=[0.01, 0.1], cache_bits_ratio=1.0002,
        first_token_fidelity=0.94, dense_accuracy_interval=[-0.01, 0.02],
        ordinary_accuracy_interval=[0.0, 0.04],
    )
    assert not failed["passed"]
    assert not failed["first_token_fidelity"]
