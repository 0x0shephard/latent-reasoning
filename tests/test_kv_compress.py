"""R-KV compression shape, mask, ordering, and selection tests."""
from __future__ import annotations

import torch

from src.losses.kv_compress import random_compress, rkv_compress, uniform_compress


def _fixture():
    keys = torch.zeros(2, 2, 2, 5, 3)
    # Put the source position in every vector so gathers can be checked directly.
    for position in range(5):
        keys[:, :, :, position] = position + 1
    values = keys + 10
    importance = torch.zeros(2, 2, 2, 5)
    importance[..., 3] = 1.0
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    return keys, values, importance, mask


def test_rkv_returns_exact_slot_shape_and_masks_short_traces():
    keys, values, importance, mask = _fixture()
    result = rkv_compress(
        keys, values, importance, mask, slots=4, importance_weight=1.0
    )
    assert result.keys.shape == result.values.shape == (2, 2, 2, 4, 3)
    assert result.mask.shape == result.indices.shape == (2, 2, 2, 4)
    assert result.mask[0].all()
    assert result.mask[1].sum().item() == 2 * 2 * 2
    assert (result.indices[..., 1:] >= result.indices[..., :-1]).all()
    assert (result.indices[0] == 3).any()  # answer-important token is retained
    assert (result.keys[~result.mask] == 0).all()


def test_uniform_and_seeded_random_compression_are_deterministic():
    keys, values, _, mask = _fixture()
    first = uniform_compress(keys, values, mask, slots=3)
    second = uniform_compress(keys, values, mask, slots=3)
    assert torch.equal(first.indices, second.indices)
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    random_a = random_compress(keys, values, mask, slots=3, generator=gen_a)
    random_b = random_compress(keys, values, mask, slots=3, generator=gen_b)
    assert torch.equal(random_a.indices, random_b.indices)
