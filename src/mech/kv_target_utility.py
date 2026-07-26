"""Utilities for hierarchical KV-target marginal-utility screening.

The screen asks whether adding a correctly paired teacher KV target produces a more
helpful parameter update than omitting the target or replacing it with a shuffled
teacher target.  Raw distillation-loss magnitude is deliberately not used as utility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


KV_KINDS = ("key", "value")
TARGET_GRANULARITIES = ("kind", "position", "layer_band")


@dataclass(frozen=True)
class KVTargetGroup:
    """One prespecified teacher-target family."""

    name: str
    kind: str
    layers: tuple[int, ...]
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in KV_KINDS:
            raise ValueError(f"unknown KV kind {self.kind!r}")
        if not self.name:
            raise ValueError("target group name cannot be empty")
        if not self.layers or min(self.layers) < 0:
            raise ValueError("target group must contain non-negative layers")
        if not self.positions or min(self.positions) < 0:
            raise ValueError("target group must contain non-negative positions")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("target group layers must be unique")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("target group positions must be unique")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "layers": list(self.layers),
            "positions": list(self.positions),
        }


def default_layer_bands(layer_count: int) -> dict[str, tuple[int, ...]]:
    """Split layers into deterministic early/middle/late contiguous bands."""
    if layer_count < 3:
        raise ValueError("layer-band screening requires at least three layers")
    boundaries = [round(index * layer_count / 3) for index in range(4)]
    names = ("early", "middle", "late")
    bands = {
        name: tuple(range(boundaries[index], boundaries[index + 1]))
        for index, name in enumerate(names)
    }
    if any(not values for values in bands.values()):
        raise RuntimeError("failed to construct non-empty layer bands")
    return bands


def build_target_groups(
    *,
    granularity: str,
    layer_count: int,
    position_count: int,
    kinds: Iterable[str] = KV_KINDS,
    positions: Iterable[int] | None = None,
    layer_bands: Mapping[str, Sequence[int]] | None = None,
) -> list[KVTargetGroup]:
    """Build the first three levels of the hierarchical target inventory."""
    if granularity not in TARGET_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {TARGET_GRANULARITIES}, got {granularity!r}"
        )
    if layer_count <= 0 or position_count <= 0:
        raise ValueError("layer_count and position_count must be positive")
    selected_kinds = tuple(dict.fromkeys(str(kind) for kind in kinds))
    if not selected_kinds or any(kind not in KV_KINDS for kind in selected_kinds):
        raise ValueError("kinds must be a non-empty subset of key,value")
    selected_positions = tuple(
        sorted(
            set(
                range(position_count)
                if positions is None
                else (int(value) for value in positions)
            )
        )
    )
    if (
        not selected_positions
        or selected_positions[0] < 0
        or selected_positions[-1] >= position_count
    ):
        raise ValueError("positions are empty or outside the latent trajectory")
    all_layers = tuple(range(layer_count))
    groups: list[KVTargetGroup] = []
    if granularity == "kind":
        for kind in selected_kinds:
            groups.append(
                KVTargetGroup(
                    name=f"{kind}_all",
                    kind=kind,
                    layers=all_layers,
                    positions=selected_positions,
                )
            )
        return groups
    if granularity == "position":
        for kind in selected_kinds:
            for position in selected_positions:
                groups.append(
                    KVTargetGroup(
                        name=f"{kind}_p{position}",
                        kind=kind,
                        layers=all_layers,
                        positions=(position,),
                    )
                )
        return groups

    resolved_bands = (
        default_layer_bands(layer_count)
        if layer_bands is None
        else {
            str(name): tuple(int(layer) for layer in values)
            for name, values in layer_bands.items()
        }
    )
    if not resolved_bands:
        raise ValueError("layer_bands cannot be empty")
    used_layers: set[int] = set()
    for name, band_layers in resolved_bands.items():
        if (
            not name
            or not band_layers
            or min(band_layers) < 0
            or max(band_layers) >= layer_count
        ):
            raise ValueError(f"invalid layer band {name!r}")
        if used_layers.intersection(band_layers):
            raise ValueError("layer bands must not overlap")
        used_layers.update(band_layers)
    for kind in selected_kinds:
        for position in selected_positions:
            for band_name, band_layers in resolved_bands.items():
                groups.append(
                    KVTargetGroup(
                        name=f"{kind}_p{position}_{band_name}",
                        kind=kind,
                        layers=tuple(band_layers),
                        positions=(position,),
                    )
                )
    return groups


def kv_group_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
    group: KVTargetGroup,
    *,
    metric: str,
) -> torch.Tensor:
    """Return a mean matched loss for tensors shaped ``[B,L,H,M,D]``."""
    if student.ndim != 5 or teacher.shape != student.shape:
        raise ValueError("student and teacher must have equal [B,L,H,M,D] shapes")
    if mask.shape != student.shape[:-1]:
        raise ValueError("mask must have shape [B,L,H,M]")
    if max(group.layers) >= student.shape[1]:
        raise ValueError("target group contains an out-of-range layer")
    if max(group.positions) >= student.shape[3]:
        raise ValueError("target group contains an out-of-range position")
    layer_index = torch.tensor(group.layers, device=student.device)
    position_index = torch.tensor(group.positions, device=student.device)
    selected_student = student.index_select(1, layer_index).index_select(
        3, position_index
    )
    selected_teacher = teacher.detach().index_select(1, layer_index).index_select(
        3, position_index
    )
    selected_mask = mask.index_select(1, layer_index).index_select(3, position_index)
    weights = selected_mask.unsqueeze(-1).to(dtype=student.dtype)
    denominator = (weights.sum() * student.shape[-1]).clamp_min(1)
    if metric == "l1":
        distance = (selected_student - selected_teacher).abs()
    elif metric == "mse":
        distance = (selected_student - selected_teacher).square()
    elif metric == "smooth_l1":
        distance = F.smooth_l1_loss(
            selected_student,
            selected_teacher,
            reduction="none",
        )
    else:
        raise ValueError("metric must be l1, mse, or smooth_l1")
    return (distance * weights).sum() / denominator


def autograd_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    """Differentiate while retaining explicit ``None`` entries for unused params."""
    return tuple(
        torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def gradient_inner_product(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> dict[str, float]:
    """Return dot product, norms, and cosine for two gradient tuples."""
    if len(left) != len(right):
        raise ValueError("gradient tuples must have equal length")
    dot = torch.zeros((), dtype=torch.float64)
    left_square = torch.zeros((), dtype=torch.float64)
    right_square = torch.zeros((), dtype=torch.float64)
    for left_value, right_value in zip(left, right):
        if left_value is not None:
            left_square += left_value.detach().double().square().sum().cpu()
        if right_value is not None:
            right_square += right_value.detach().double().square().sum().cpu()
        if left_value is not None and right_value is not None:
            dot += (
                left_value.detach().double()
                .mul(right_value.detach().double())
                .sum()
                .cpu()
            )
    left_norm = left_square.sqrt()
    right_norm = right_square.sqrt()
    denominator = left_norm * right_norm
    cosine = dot / denominator if float(denominator) > 0.0 else dot.new_zeros(())
    return {
        "dot": float(dot),
        "left_norm": float(left_norm),
        "right_norm": float(right_norm),
        "cosine": float(cosine),
    }


def combine_gradients(
    base: Sequence[torch.Tensor | None],
    auxiliary: Sequence[torch.Tensor | None] | None = None,
    *,
    auxiliary_weight: float = 1.0,
) -> tuple[torch.Tensor | None, ...]:
    """Combine base and optional auxiliary gradients without mutating either."""
    if auxiliary is None:
        return tuple(
            None if value is None else value.detach().clone() for value in base
        )
    if len(base) != len(auxiliary):
        raise ValueError("gradient tuples must have equal length")
    combined = []
    for base_value, auxiliary_value in zip(base, auxiliary):
        if base_value is None and auxiliary_value is None:
            combined.append(None)
        elif base_value is None:
            combined.append(auxiliary_value.detach() * auxiliary_weight)
        elif auxiliary_value is None:
            combined.append(base_value.detach().clone())
        else:
            combined.append(
                base_value.detach() + auxiliary_weight * auxiliary_value.detach()
            )
    return tuple(combined)


def gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    square = torch.zeros((), dtype=torch.float64)
    for value in gradients:
        if value is not None:
            square += value.detach().double().square().sum().cpu()
    return float(square.sqrt())


def parameter_norm(parameters: Sequence[torch.Tensor]) -> float:
    square = torch.zeros((), dtype=torch.float64)
    for value in parameters:
        square += value.detach().double().square().sum().cpu()
    return float(square.sqrt())


def updated_parameter_mapping(
    names: Sequence[str],
    parameters: Sequence[torch.Tensor],
    gradients: Sequence[torch.Tensor | None],
    *,
    update_norm: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build a stateless equal-norm gradient update for ``functional_call``."""
    if not (len(names) == len(parameters) == len(gradients)):
        raise ValueError("names, parameters, and gradients must have equal length")
    observed_norm = gradient_norm(gradients)
    if observed_norm <= 0.0:
        raise ValueError("cannot apply a zero gradient")
    if update_norm <= 0.0:
        raise ValueError("update_norm must be positive")
    scale = update_norm / observed_norm
    mapping: dict[str, torch.Tensor] = {}
    for name, parameter, gradient in zip(names, parameters, gradients):
        if gradient is None:
            continue
        mapping[name] = (
            parameter.detach() - scale * gradient.detach().to(parameter.dtype)
        )
    return mapping, {
        "raw_gradient_norm": observed_norm,
        "requested_update_norm": float(update_norm),
        "scale": float(scale),
    }

