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


def test_portable_rng_restore_handles_fewer_active_gpus(monkeypatch):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    state = rng_state()
    state["torch_cuda"] = [torch.arange(8, dtype=torch.uint8), torch.arange(9, dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device: restored.append((value.clone(), device)),
    )

    restore_rng_state_portably(state)

    assert len(restored) == 1
    assert restored[0][1] == 0
    assert torch.equal(restored[0][0], state["torch_cuda"][0])
