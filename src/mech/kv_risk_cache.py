"""Small decode-time KV policy used by the compression-risk pilot.

This is deliberately a simple heavy-hitter plus recent-token policy. It is not
presented as an exact reproduction of H2O and is not a candidate method. Prompt
tokens are immutable; only generated reasoning-token entries can be removed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


def cache_to_legacy(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return a legacy tuple without assuming one Transformers cache version."""
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if not isinstance(cache, (tuple, list)):
        raise TypeError(f"unsupported cache type: {type(cache)!r}")
    legacy = tuple((layer[0], layer[1]) for layer in cache)
    if not legacy:
        raise ValueError("cache has no layers")
    return legacy


def cache_from_legacy(
    legacy: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    template: Any,
) -> Any:
    """Rebuild the cache in the representation returned by the model."""
    if isinstance(template, tuple):
        return legacy
    if isinstance(template, list):
        return list(legacy)
    cache_type = type(template)
    factory = getattr(cache_type, "from_legacy_cache", None)
    if callable(factory):
        return factory(legacy)
    try:
        from transformers import DynamicCache

        return DynamicCache.from_legacy_cache(legacy)
    except (ImportError, AttributeError) as exc:
        raise TypeError(
            f"cannot rebuild Transformers cache type {cache_type!r}"
        ) from exc


def attention_to_kv_heads(
    attention: torch.Tensor,
    kv_heads: int,
) -> torch.Tensor:
    """Average query-head attention within each grouped-query KV head."""
    if attention.ndim == 4:
        attention = attention[:, :, -1, :]
    if attention.ndim != 3:
        raise ValueError(
            "attention must have shape [batch, heads, sequence] or "
            "[batch, heads, query, sequence]"
        )
    batch, query_heads, sequence = attention.shape
    if query_heads == kv_heads:
        return attention
    if query_heads % kv_heads:
        raise ValueError(
            f"{query_heads} query heads are not divisible by {kv_heads} KV heads"
        )
    groups = query_heads // kv_heads
    return attention.reshape(batch, kv_heads, groups, sequence).mean(dim=2)


def select_generated_indices(
    scores: torch.Tensor,
    absolute_positions: torch.Tensor,
    *,
    prompt_length: int,
    target_generated: int,
    recent_window: int,
    heavy_fraction: float,
) -> torch.Tensor:
    """Select ordered indices independently for every batch item and KV head."""
    if scores.shape != absolute_positions.shape or scores.ndim != 3:
        raise ValueError("scores and positions must share [batch, heads, sequence]")
    if not 0.0 <= heavy_fraction <= 1.0:
        raise ValueError("heavy_fraction must lie in [0, 1]")
    if target_generated < 1:
        raise ValueError("target_generated must be positive")

    batch, heads, _ = scores.shape
    selections: list[torch.Tensor] = []
    expected_width: int | None = None
    recent_slots = min(
        recent_window,
        target_generated,
        max(1, math.ceil(target_generated * (1.0 - heavy_fraction))),
    )
    heavy_slots = target_generated - recent_slots

    for batch_index in range(batch):
        head_selections: list[torch.Tensor] = []
        for head_index in range(heads):
            positions = absolute_positions[batch_index, head_index]
            head_scores = scores[batch_index, head_index]
            prompt_indices = torch.nonzero(
                positions < prompt_length,
                as_tuple=False,
            ).flatten()
            generated_indices = torch.nonzero(
                positions >= prompt_length,
                as_tuple=False,
            ).flatten()
            if generated_indices.numel() <= target_generated:
                selected = torch.cat((prompt_indices, generated_indices))
            else:
                generated_positions = positions[generated_indices]
                recent_order = torch.argsort(
                    generated_positions,
                    descending=True,
                    stable=True,
                )
                recent = generated_indices[recent_order[:recent_slots]]

                candidate_mask = torch.ones(
                    generated_indices.numel(),
                    dtype=torch.bool,
                    device=generated_indices.device,
                )
                candidate_mask[recent_order[:recent_slots]] = False
                candidates = generated_indices[candidate_mask]
                if heavy_slots:
                    candidate_scores = head_scores[candidates]
                    heavy_order = torch.argsort(
                        candidate_scores,
                        descending=True,
                        stable=True,
                    )
                    heavy = candidates[heavy_order[:heavy_slots]]
                else:
                    heavy = candidates[:0]
                selected = torch.cat((prompt_indices, heavy, recent))

            selected_positions = positions[selected]
            selected = selected[
                torch.argsort(selected_positions, stable=True)
            ]
            if expected_width is None:
                expected_width = int(selected.numel())
            elif int(selected.numel()) != expected_width:
                raise RuntimeError("cache selection produced inconsistent widths")
            head_selections.append(selected)
        selections.append(torch.stack(head_selections, dim=0))
    return torch.stack(selections, dim=0)


def gather_cache_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a [batch, heads, sequence, dim] cache along its sequence axis."""
    if tensor.ndim != 4 or indices.ndim != 3:
        raise ValueError("cache/indices shapes are incompatible")
    gather_indices = indices.unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        tensor.shape[-1],
    )
    return torch.gather(tensor, dim=2, index=gather_indices)


@dataclass
class CacheStepMetrics:
    retained_generated: int
    seen_generated: int
    retained_total: int
    requested_retention: float


class HeavyHitterRecentCache:
    """Track attention mass and prune generated cache entries online."""

    def __init__(
        self,
        cache: Any,
        *,
        prompt_length: int,
        retention: float,
        recent_window: int,
        heavy_fraction: float,
    ) -> None:
        if not 0.0 < retention <= 1.0:
            raise ValueError("retention must lie in (0, 1]")
        self.prompt_length = int(prompt_length)
        self.retention = float(retention)
        self.recent_window = int(recent_window)
        self.heavy_fraction = float(heavy_fraction)
        legacy = cache_to_legacy(cache)
        self.scores: list[torch.Tensor] = []
        self.positions: list[torch.Tensor] = []
        for key, _ in legacy:
            batch, heads, sequence, _ = key.shape
            self.scores.append(
                torch.zeros(
                    (batch, heads, sequence),
                    dtype=torch.float32,
                    device=key.device,
                )
            )
            self.positions.append(
                torch.arange(sequence, device=key.device)
                .view(1, 1, sequence)
                .expand(batch, heads, sequence)
                .clone()
            )

    def update(
        self,
        cache: Any,
        attentions: Sequence[torch.Tensor | None],
        *,
        appended_absolute_position: int,
    ) -> None:
        legacy = cache_to_legacy(cache)
        if len(legacy) != len(self.scores) or len(attentions) != len(legacy):
            raise ValueError("cache, attention, and score layer counts differ")
        for layer_index, ((key, _), attention) in enumerate(
            zip(legacy, attentions)
        ):
            if attention is None:
                raise RuntimeError(
                    "the compressor requires attention weights; load the model "
                    "with eager attention"
                )
            current_length = int(key.shape[2])
            previous_length = int(self.scores[layer_index].shape[2])
            if current_length != previous_length + 1:
                raise RuntimeError(
                    "expected exactly one new cache entry before pruning"
                )
            batch, heads = key.shape[:2]
            zero = torch.zeros(
                (batch, heads, 1),
                dtype=torch.float32,
                device=key.device,
            )
            self.scores[layer_index] = torch.cat(
                (self.scores[layer_index], zero),
                dim=2,
            )
            appended = torch.full(
                (batch, heads, 1),
                int(appended_absolute_position),
                dtype=torch.long,
                device=key.device,
            )
            self.positions[layer_index] = torch.cat(
                (self.positions[layer_index], appended),
                dim=2,
            )
            kv_attention = attention_to_kv_heads(
                attention.detach().float(),
                kv_heads=heads,
            )
            if kv_attention.shape != self.scores[layer_index].shape:
                raise RuntimeError(
                    "attention weights do not align with the current cache"
                )
            self.scores[layer_index].add_(kv_attention)

    def prune(
        self,
        cache: Any,
        *,
        seen_generated: int,
    ) -> tuple[Any, CacheStepMetrics]:
        legacy = cache_to_legacy(cache)
        target_generated = min(
            int(seen_generated),
            max(
                min(self.recent_window, int(seen_generated)),
                math.ceil(self.retention * int(seen_generated)),
            ),
        )
        target_generated = max(1, target_generated)
        rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
        retained_generated: int | None = None
        retained_total: int | None = None

        for layer_index, (key, value) in enumerate(legacy):
            indices = select_generated_indices(
                self.scores[layer_index],
                self.positions[layer_index],
                prompt_length=self.prompt_length,
                target_generated=target_generated,
                recent_window=self.recent_window,
                heavy_fraction=self.heavy_fraction,
            )
            key = gather_cache_axis(key, indices)
            value = gather_cache_axis(value, indices)
            self.scores[layer_index] = torch.gather(
                self.scores[layer_index],
                dim=2,
                index=indices,
            )
            self.positions[layer_index] = torch.gather(
                self.positions[layer_index],
                dim=2,
                index=indices,
            )
            generated_count = int(
                (self.positions[layer_index][0, 0] >= self.prompt_length)
                .sum()
                .item()
            )
            total_count = int(key.shape[2])
            if retained_generated is None:
                retained_generated = generated_count
                retained_total = total_count
            elif (
                generated_count != retained_generated
                or total_count != retained_total
            ):
                raise RuntimeError("layers retained inconsistent cache lengths")
            rebuilt.append((key, value))

        assert retained_generated is not None and retained_total is not None
        metrics = CacheStepMetrics(
            retained_generated=retained_generated,
            seen_generated=int(seen_generated),
            retained_total=retained_total,
            requested_retention=self.retention,
        )
        return cache_from_legacy(tuple(rebuilt), cache), metrics

