"""Step-deterministic batching + label masking (dependency-free, CPU-testable).

Batches are a pure function of the step index: every step maps to a fixed set of example
indices via a per-epoch permutation seeded once. This is what lets a run killed by a
Kaggle session cap resume with no duplicated or skipped data — the same guarantee the
Phase 0 synthetic task proved, now for real data.
"""
from __future__ import annotations

import random


class StepBatcher:
    """Maps a global step index to example indices, reshuffling each epoch.

    Deterministic given (num_examples, batch_size, seed): `batch_indices(step)` returns the
    same indices regardless of how training was interrupted or resumed.
    """

    def __init__(self, num_examples: int, batch_size: int, seed: int = 0):
        if num_examples <= 0 or batch_size <= 0:
            raise ValueError("num_examples and batch_size must be positive")
        self.n = num_examples
        self.bs = batch_size
        self.seed = seed
        self._perm_cache: dict[int, list[int]] = {}

    def _perm(self, epoch: int) -> list[int]:
        perm = self._perm_cache.get(epoch)
        if perm is None:
            rng = random.Random(self.seed * 1_000_003 + epoch)  # int seed: hash-salt independent
            perm = list(range(self.n))
            rng.shuffle(perm)
            self._perm_cache[epoch] = perm
        return perm

    def batch_indices(self, step: int) -> list[int]:
        base = step * self.bs
        out = []
        for j in range(base, base + self.bs):
            epoch, pos = divmod(j, self.n)
            out.append(self._perm(epoch)[pos])
        return out


def build_labels(input_ids: list[int], prompt_len: int) -> list[int]:
    """Copy input_ids, masking the first `prompt_len` tokens with -100.

    Masked positions are ignored by the causal-LM cross-entropy, so loss is computed only on
    the completion (CoT + answer for CoT-SFT, or just the answer for No-CoT-SFT).
    """
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    return labels
