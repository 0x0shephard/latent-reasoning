from __future__ import annotations

import torch

from src.mech.kv_gradient_signal import (
    GradientAlignmentAccumulator,
    mask_gradients,
    random_mask_like,
    rescale_gradients_to_norm,
)
from src.mech.kv_target_utility import gradient_norm


def test_alignment_accumulator_selects_consistently_positive_coordinates():
    parameters = [torch.zeros(4), torch.zeros(2)]
    accumulator = GradientAlignmentAccumulator.from_parameters(parameters)
    for _ in range(4):
        answer = (torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.ones(2))
        kv = (
            torch.tensor([3.0, 2.0, -4.0, -1.0]),
            torch.tensor([-2.0, -3.0]),
        )
        accumulator.update(answer, kv)
    masks, summary = accumulator.build_mask(
        sparsity=1 / 3,
        minimum_positive_fraction=0.75,
    )
    assert summary["selected_coordinates"] == 2
    assert masks[0].tolist() == [True, True, False, False]
    assert not bool(masks[1].any())


def test_random_mask_is_cardinality_matched_and_deterministic():
    shapes = (torch.zeros(5, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))
    left = random_mask_like(shapes, selected_coordinates=3, seed=17)
    right = random_mask_like(shapes, selected_coordinates=3, seed=17)
    assert sum(int(value.sum()) for value in left) == 3
    assert all(torch.equal(a, b) for a, b in zip(left, right))


def test_mask_complement_and_norm_matching():
    gradients = (torch.tensor([3.0, 4.0]),)
    masks = (torch.tensor([True, False]),)
    selected = mask_gradients(gradients, masks)
    complement = mask_gradients(gradients, masks, complement=True)
    assert selected[0].tolist() == [3.0, 0.0]
    assert complement[0].tolist() == [0.0, 4.0]
    scaled, diagnostic = rescale_gradients_to_norm(
        selected,
        target_norm=5.0,
    )
    assert abs(gradient_norm(scaled) - 5.0) < 1e-6
    assert diagnostic["raw_gradient_norm"] == 3.0
