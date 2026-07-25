"""Configurable hidden-state and KV-trajectory distillation losses.

CODI is hidden/all-layers/endpoint with per-layer teacher-std normalization.  KaVa adds
keys+values/all-layers/all-latent-steps.  The functions here also cover the intermediate
supervision-granularity ablations without changing the student forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TrajectoryLossOutput:
    total: torch.Tensor
    hidden: torch.Tensor
    kv: torch.Tensor


def _layer_indices(spec: str | Iterable[int], layer_count: int) -> list[int]:
    if spec == "all":
        return list(range(layer_count))
    if isinstance(spec, str):
        raise ValueError("layers must be 'all' or a sequence of integer indices")
    indices = [int(index) % layer_count for index in spec]
    if not indices:
        raise ValueError("at least one layer must be selected")
    return indices


def _distance(student: torch.Tensor, teacher: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "l1":
        return (student - teacher).abs()
    if metric == "mse":
        return (student - teacher).square()
    if metric == "smooth_l1":
        return F.smooth_l1_loss(student, teacher, reduction="none")
    raise ValueError(f"unknown trajectory metric {metric!r}")


def hidden_match_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    layers: str | Iterable[int] = "all",
    metric: str = "l1",
    normalize_teacher_std: bool = True,
    layer_reduction: str = "sum",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match hidden targets with shapes ``[B,L,D]`` or ``[B,L,P,D]``."""
    if student.shape != teacher.shape or student.ndim not in (3, 4):
        raise ValueError("hidden targets must have equal [B,L,D] or [B,L,P,D] shapes")
    selected = _layer_indices(layers, student.shape[1])
    per_layer = []
    target = teacher.detach()
    for layer in selected:
        distance = _distance(student[:, layer], target[:, layer], metric)
        value = distance.mean()
        if normalize_teacher_std:
            std = target[:, layer].float().std(unbiased=False).clamp_min(eps)
            value = value / std.to(value.dtype)
        per_layer.append(value)
    stacked = torch.stack(per_layer)
    if layer_reduction == "sum":
        return stacked.sum()
    if layer_reduction == "mean":
        return stacked.mean()
    raise ValueError("layer_reduction must be 'sum' or 'mean'")


def kv_match_loss(
    student_keys: torch.Tensor,
    student_values: torch.Tensor,
    teacher_keys: torch.Tensor,
    teacher_values: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    layers: str | Iterable[int] = "all",
    metric: str = "l1",
) -> torch.Tensor:
    """Match KV tensors shaped ``[B,L,H,M,D]`` with an optional ``[B,L,H,M]`` mask."""
    tensors = (student_keys, student_values, teacher_keys, teacher_values)
    if any(tensor.ndim != 5 for tensor in tensors):
        raise ValueError("KV targets must have shape [B,L,H,M,D]")
    if any(tensor.shape != student_keys.shape for tensor in tensors[1:]):
        raise ValueError("all KV target shapes must match")
    indices = _layer_indices(layers, student_keys.shape[1])
    sk = student_keys[:, indices]
    sv = student_values[:, indices]
    tk = teacher_keys.detach()[:, indices]
    tv = teacher_values.detach()[:, indices]
    if mask is None:
        valid = torch.ones(sk.shape[:-1], dtype=torch.bool, device=sk.device)
    else:
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        if mask.ndim != 4:
            raise ValueError("KV mask must have shape [B,M] or [B,L,H,M]")
        valid = torch.broadcast_to(mask, student_keys.shape[:-1])[:, indices]
    weights = valid.unsqueeze(-1).to(sk.dtype)
    denom = (weights.sum() * sk.shape[-1]).clamp_min(1)
    key_loss = (_distance(sk, tk, metric) * weights).sum() / denom
    value_loss = (_distance(sv, tv, metric) * weights).sum() / denom
    return 0.5 * (key_loss + value_loss)


def key_match_loss(
    student_keys: torch.Tensor,
    teacher_keys: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    layers: str | Iterable[int] = "all",
    metric: str = "mse",
) -> torch.Tensor:
    """Match full key targets shaped ``[B,L,H,M,D]``."""
    if student_keys.ndim != 5 or teacher_keys.shape != student_keys.shape:
        raise ValueError("key targets must have equal [B,L,H,M,D] shapes")
    indices = _layer_indices(layers, student_keys.shape[1])
    student = student_keys[:, indices]
    teacher = teacher_keys.detach()[:, indices]
    if mask is None:
        valid = torch.ones(
            student.shape[:-1], dtype=torch.bool, device=student.device
        )
    else:
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        if mask.ndim != 4:
            raise ValueError("KV mask must have shape [B,M] or [B,L,H,M]")
        valid = torch.broadcast_to(mask, student_keys.shape[:-1])[:, indices]
    weights = valid.unsqueeze(-1).to(student.dtype)
    denom = (weights.sum() * student.shape[-1]).clamp_min(1)
    return (_distance(student, teacher, metric) * weights).sum() / denom


def projected_key_match_loss(
    student_keys: torch.Tensor,
    teacher_keys: torch.Tensor,
    projection: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    layers: str | Iterable[int] = "all",
    metric: str = "mse",
) -> torch.Tensor:
    """Match key coefficients in frozen bases shaped ``[L,H,M,D,R]``."""
    if student_keys.ndim != 5 or teacher_keys.shape != student_keys.shape:
        raise ValueError("key targets must have equal [B,L,H,M,D] shapes")
    expected = (
        student_keys.shape[1],
        student_keys.shape[2],
        student_keys.shape[3],
        student_keys.shape[4],
    )
    if projection.ndim != 5 or projection.shape[:-1] != expected:
        raise ValueError(
            "projection must have shape [L,H,M,D,R] matching the key targets"
        )
    indices = _layer_indices(layers, student_keys.shape[1])
    basis = projection[indices].to(
        device=student_keys.device,
        dtype=student_keys.dtype,
    )
    student = torch.einsum(
        "blhmd,lhmdr->blhmr", student_keys[:, indices], basis
    )
    teacher = torch.einsum(
        "blhmd,lhmdr->blhmr", teacher_keys.detach()[:, indices], basis
    )
    if mask is None:
        valid = torch.ones(
            student.shape[:-1], dtype=torch.bool, device=student.device
        )
    else:
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        if mask.ndim != 4:
            raise ValueError("KV mask must have shape [B,M] or [B,L,H,M]")
        valid = torch.broadcast_to(mask, student_keys.shape[:-1])[:, indices]
    weights = valid.unsqueeze(-1).to(student.dtype)
    denom = (weights.sum() * student.shape[-1]).clamp_min(1)
    return (_distance(student, teacher, metric) * weights).sum() / denom


class TrajectoryMatchLoss(nn.Module):
    """Composable CODI hidden loss plus optional KaVa KV trajectory loss."""

    def __init__(
        self,
        *,
        hidden_weight: float = 1.0,
        kv_weight: float = 0.0,
        hidden_layers: str | Iterable[int] = "all",
        kv_layers: str | Iterable[int] = "all",
        hidden_metric: str = "l1",
        kv_metric: str = "l1",
        kv_target: str = "both",
        key_projection: torch.Tensor | None = None,
        normalize_teacher_std: bool = True,
        hidden_layer_reduction: str = "sum",
    ) -> None:
        super().__init__()
        if hidden_weight < 0 or kv_weight < 0:
            raise ValueError("trajectory weights must be non-negative")
        self.hidden_weight = hidden_weight
        self.kv_weight = kv_weight
        self.hidden_layers = hidden_layers
        self.kv_layers = kv_layers
        self.hidden_metric = hidden_metric
        self.kv_metric = kv_metric
        if kv_target not in {"both", "key", "projected_key"}:
            raise ValueError("kv_target must be both, key, or projected_key")
        if kv_target == "projected_key" and key_projection is None:
            raise ValueError("projected_key requires a key_projection")
        if kv_target != "projected_key" and key_projection is not None:
            raise ValueError("key_projection is only valid for projected_key")
        self.kv_target = kv_target
        self.register_buffer(
            "key_projection",
            key_projection,
            persistent=False,
        )
        self.normalize_teacher_std = normalize_teacher_std
        self.hidden_layer_reduction = hidden_layer_reduction

    def forward(
        self,
        *,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        student_keys: torch.Tensor | None = None,
        student_values: torch.Tensor | None = None,
        teacher_keys: torch.Tensor | None = None,
        teacher_values: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> TrajectoryLossOutput:
        zero = student_hidden.sum() * 0.0
        hidden = zero
        kv = zero
        if self.hidden_weight:
            hidden = hidden_match_loss(
                student_hidden,
                teacher_hidden,
                layers=self.hidden_layers,
                metric=self.hidden_metric,
                normalize_teacher_std=self.normalize_teacher_std,
                layer_reduction=self.hidden_layer_reduction,
            )
        if self.kv_weight:
            if student_keys is None or teacher_keys is None:
                raise ValueError("key tensors are required when kv_weight is non-zero")
            if self.kv_target == "both":
                if student_values is None or teacher_values is None:
                    raise ValueError("value tensors are required for kv_target=both")
                kv = kv_match_loss(
                    student_keys,
                    student_values,
                    teacher_keys,
                    teacher_values,
                    mask=kv_mask,
                    layers=self.kv_layers,
                    metric=self.kv_metric,
                )
            elif self.kv_target == "key":
                kv = key_match_loss(
                    student_keys,
                    teacher_keys,
                    mask=kv_mask,
                    layers=self.kv_layers,
                    metric=self.kv_metric,
                )
            else:
                assert self.key_projection is not None
                kv = projected_key_match_loss(
                    student_keys,
                    teacher_keys,
                    self.key_projection,
                    mask=kv_mask,
                    layers=self.kv_layers,
                    metric=self.kv_metric,
                )
        total = self.hidden_weight * hidden + self.kv_weight * kv
        return TrajectoryLossOutput(total=total, hidden=hidden, kv=kv)
