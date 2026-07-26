"""Teacher KV-cache compression used by KaVa.

R-KV ranks each explicit-CoT token by a mixture of answer-query importance and low
redundancy, independently for every layer and attention head.  Selected indices are sorted
back into chronological order before matching them to autoregressive student latent slots.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CompressedKV:
    keys: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor
    indices: torch.Tensor
    scores: torch.Tensor


def _validate(
    keys: torch.Tensor,
    values: torch.Tensor,
    importance: torch.Tensor,
    mask: torch.Tensor,
    slots: int,
) -> None:
    if keys.ndim != 5:
        raise ValueError("keys/values must have shape [B,L,H,N,D]")
    if values.shape != keys.shape:
        raise ValueError("key/value shapes must match")
    if importance.shape != keys.shape[:-1]:
        raise ValueError("importance must have shape [B,L,H,N]")
    if mask.shape != (keys.shape[0], keys.shape[3]):
        raise ValueError("mask must have shape [B,N]")
    if slots <= 0:
        raise ValueError("slots must be positive")


def redundancy_scores(keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a per-token novelty distribution from negative pairwise cosine similarity."""
    if keys.ndim != 5 or mask.shape != (keys.shape[0], keys.shape[3]):
        raise ValueError("expected keys [B,L,H,N,D] and mask [B,N]")
    normalized = F.normalize(keys.float(), p=2, dim=-1, eps=1e-8)
    cosine = torch.einsum("blhnd,blhmd->blhnm", normalized, normalized)
    valid_pair = mask[:, None, None, :, None] & mask[:, None, None, None, :]
    eye = torch.eye(keys.shape[3], device=keys.device, dtype=torch.bool)
    valid_pair = valid_pair & ~eye[None, None, None]
    # More negative mean cosine means less redundancy. A singleton receives zero novelty.
    counts = valid_pair.sum(dim=-1)
    novelty = -(cosine * valid_pair).sum(dim=-1) / counts.clamp_min(1)
    valid_token = mask[:, None, None, :]
    novelty = novelty.masked_fill(~valid_token, -torch.finfo(novelty.dtype).max)
    singleton = counts == 0
    novelty = torch.where(singleton & mask[:, None, None, :], 0.0, novelty)
    distribution = torch.softmax(novelty, dim=-1) * valid_token
    distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return distribution.to(dtype=keys.dtype)


def _gather_slots(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # tensor [B,L,H,N,D], indices [B,L,H,M]
    expanded = indices.unsqueeze(-1).expand(*indices.shape, tensor.shape[-1])
    return torch.gather(tensor, dim=3, index=expanded)


def _select_from_scores(
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    mask: torch.Tensor,
    slots: int,
) -> CompressedKV:
    available = keys.shape[3]
    take = min(slots, available)
    valid = mask[:, None, None, :]
    ranked = scores.masked_fill(~valid, float("-inf"))
    top = torch.topk(ranked, k=take, dim=-1).indices
    top = torch.sort(top, dim=-1).values  # student latent step order is chronological
    selected_mask = torch.gather(valid.expand_as(ranked), dim=3, index=top)
    selected_scores = torch.gather(scores, dim=3, index=top)
    selected_scores = torch.where(selected_mask, selected_scores, torch.zeros_like(selected_scores))
    selected_keys = _gather_slots(keys, top)
    selected_values = _gather_slots(values, top)
    selected_keys = selected_keys * selected_mask.unsqueeze(-1)
    selected_values = selected_values * selected_mask.unsqueeze(-1)

    if take < slots:
        pad_shape = (*top.shape[:-1], slots - take)
        top = torch.cat([top, torch.zeros(pad_shape, dtype=top.dtype, device=top.device)], -1)
        selected_mask = torch.cat(
            [selected_mask, torch.zeros(pad_shape, dtype=torch.bool, device=top.device)], -1
        )
        selected_scores = torch.cat(
            [selected_scores, torch.zeros(pad_shape, dtype=scores.dtype, device=top.device)],
            -1,
        )
        kv_pad = (*selected_keys.shape[:-2], slots - take, selected_keys.shape[-1])
        selected_keys = torch.cat([selected_keys, keys.new_zeros(kv_pad)], dim=3)
        selected_values = torch.cat([selected_values, values.new_zeros(kv_pad)], dim=3)

    return CompressedKV(
        keys=selected_keys,
        values=selected_values,
        mask=selected_mask,
        indices=top,
        scores=selected_scores,
    )


def rkv_compress(
    keys: torch.Tensor,
    values: torch.Tensor,
    importance: torch.Tensor,
    mask: torch.Tensor,
    slots: int,
    *,
    importance_weight: float = 0.1,
) -> CompressedKV:
    """Compress an explicit trace to ``slots`` using KaVa's R-KV score."""
    _validate(keys, values, importance, mask, slots)
    if not 0.0 <= importance_weight <= 1.0:
        raise ValueError("importance_weight must be in [0, 1]")
    valid = mask[:, None, None, :]
    normalized_importance = importance.masked_fill(~valid, 0.0)
    normalized_importance = normalized_importance / normalized_importance.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    redundancy = redundancy_scores(keys, mask)
    scores = importance_weight * normalized_importance + (1.0 - importance_weight) * redundancy
    return _select_from_scores(keys, values, scores, mask, slots)


def uniform_compress(
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    slots: int,
) -> CompressedKV:
    """Select chronologically uniform trace locations (compression ablation)."""
    importance = keys.new_zeros(keys.shape[:-1])
    _validate(keys, values, importance, mask, slots)
    scores = keys.new_full(keys.shape[:-1], float("-inf"))
    for batch_index, count_tensor in enumerate(mask.sum(dim=-1)):
        count = int(count_tensor)
        if count == 0:
            continue
        take = min(slots, count)
        indices = torch.linspace(0, count - 1, take, device=keys.device).round().long().unique()
        scores[batch_index, :, :, indices] = torch.arange(
            len(indices), 0, -1, device=keys.device, dtype=keys.dtype
        )
    return _select_from_scores(keys, values, scores, mask, slots)


def random_compress(
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    slots: int,
    *,
    generator: torch.Generator | None = None,
    score_dtype: torch.dtype | None = None,
) -> CompressedKV:
    """Select random valid locations independently per layer/head (seedable ablation)."""
    importance = keys.new_zeros(keys.shape[:-1])
    _validate(keys, values, importance, mask, slots)
    scores = torch.rand(
        keys.shape[:-1],
        device=keys.device,
        dtype=score_dtype or keys.dtype,
        generator=generator,
    )
    return _select_from_scores(keys, values, scores, mask, slots)
