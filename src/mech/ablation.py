"""Causal interventions on continuous latent states.

Interventions operate on the vector that is about to enter a latent slot.  Applying one
there changes that slot's KV cache, all later latent states, and answer generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


ABLATION_MODES = ("zero", "batch_mean", "batch_shuffle")


def parse_positions(value: str, latent_steps: int) -> frozenset[int] | None:
    """Parse ``all`` or a comma-separated zero-based list of latent positions."""
    if latent_steps <= 0:
        raise ValueError("latent_steps must be positive")
    if value.strip().casefold() == "all":
        return None
    try:
        positions = frozenset(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("positions must be 'all' or comma-separated integers") from exc
    if not positions:
        raise ValueError("at least one latent position is required")
    invalid = sorted(position for position in positions if not 0 <= position < latent_steps)
    if invalid:
        raise ValueError(
            f"latent positions out of range for M={latent_steps}: {invalid}"
        )
    return positions


@dataclass(frozen=True)
class LatentAblation:
    """Callable latent-state intervention used by ``LatentCausalLM``.

    ``batch_mean`` removes example-specific content while preserving a typical vector at
    that slot. ``batch_shuffle`` transfers states between examples using a deterministic
    permutation. A batch of one cannot be shuffled and is therefore left unchanged.
    """

    mode: str
    positions: frozenset[int] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ABLATION_MODES:
            raise ValueError(
                f"unknown latent ablation {self.mode!r}; expected one of {ABLATION_MODES}"
            )
        if self.positions is not None and not self.positions:
            raise ValueError("positions cannot be empty")
        if self.positions is not None and min(self.positions) < 0:
            raise ValueError("positions must be non-negative")

    def applies_to(self, step: int) -> bool:
        return self.positions is None or step in self.positions

    def __call__(self, state: torch.Tensor, step: int) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError(f"latent state must have shape [batch, hidden], got {state.shape}")
        if not self.applies_to(step):
            return state
        if self.mode == "zero":
            return torch.zeros_like(state)
        if self.mode == "batch_mean":
            return state.mean(dim=0, keepdim=True).expand_as(state)
        if state.shape[0] < 2:
            return state
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + step)
        permutation = torch.randperm(state.shape[0], generator=generator).to(state.device)
        # Avoid a no-op permutation, especially for two-example test/eval batches.
        if bool(torch.equal(permutation, torch.arange(state.shape[0], device=state.device))):
            permutation = permutation.roll(1)
        return state.index_select(0, permutation)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "positions": "all" if self.positions is None else sorted(self.positions),
            "seed": self.seed,
        }
