"""Regression coverage for portable checkpoint RNG restoration."""
from __future__ import annotations

import random

import numpy as np
import torch

from scripts.resume_training import restore_rng_state_portably
from src.utils.seeding import rng_state


def test_portable_rng_restore_replays_cpu_sequences():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng_state_portably(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
