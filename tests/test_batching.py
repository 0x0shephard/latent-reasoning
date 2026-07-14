"""Unit tests for step-deterministic batching + label masking (CPU, no torch/downloads)."""
from __future__ import annotations

from src.train.batching import StepBatcher, build_labels


def test_batcher_is_deterministic():
    a = StepBatcher(100, 8, seed=0)
    b = StepBatcher(100, 8, seed=0)
    for step in [0, 1, 5, 37, 500]:
        assert a.batch_indices(step) == b.batch_indices(step)


def test_batcher_seed_changes_order():
    a = StepBatcher(100, 8, seed=0)
    b = StepBatcher(100, 8, seed=1)
    # Extremely unlikely to match across all early steps if the seed matters.
    assert any(a.batch_indices(s) != b.batch_indices(s) for s in range(5))


def test_epoch_is_a_permutation():
    """One epoch (n/bs steps) covers every example exactly once."""
    n, bs = 20, 4
    batcher = StepBatcher(n, bs, seed=3)
    seen = []
    for step in range(n // bs):
        seen.extend(batcher.batch_indices(step))
    assert sorted(seen) == list(range(n))


def test_resume_indices_independent_of_history():
    """batch_indices(step) is a pure function of step — the resume guarantee."""
    fresh = StepBatcher(50, 6, seed=7)
    warmed = StepBatcher(50, 6, seed=7)
    for s in range(10):  # "consume" earlier steps on the warmed instance
        warmed.batch_indices(s)
    assert fresh.batch_indices(10) == warmed.batch_indices(10)


def test_build_labels_masks_prompt():
    assert build_labels([1, 2, 3, 4], 2) == [-100, -100, 3, 4]
    assert build_labels([5, 6], 0) == [5, 6]           # nothing masked
    assert build_labels([5, 6], 10) == [-100, -100]    # clamp: over-long prompt_len
