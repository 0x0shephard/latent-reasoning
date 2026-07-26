"""Sparse, consistently answer-aligned KV gradient components.

The selector is fitted only on calibration batches.  For each trainable parameter
coordinate it records the mean contribution ``g_answer * g_kv`` and the fraction of
calibration batches on which that contribution is positive.  A frozen top-fraction mask
can then be tested on disjoint update and validation batches.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from src.mech.kv_target_utility import gradient_norm


GradientTuple = tuple[torch.Tensor | None, ...]


@dataclass
class GradientAlignmentAccumulator:
    """Device-resident sufficient statistics for coordinatewise alignment."""

    contribution_sum: list[torch.Tensor]
    positive_count: list[torch.Tensor]
    batches: int = 0

    @classmethod
    def from_parameters(
        cls,
        parameters: Sequence[torch.Tensor],
    ) -> "GradientAlignmentAccumulator":
        return cls(
            contribution_sum=[
                torch.zeros_like(parameter, dtype=torch.float32)
                for parameter in parameters
            ],
            positive_count=[
                torch.zeros_like(parameter, dtype=torch.int32)
                for parameter in parameters
            ],
        )

    def update(
        self,
        answer_gradients: Sequence[torch.Tensor | None],
        kv_gradients: Sequence[torch.Tensor | None],
    ) -> None:
        if not (
            len(answer_gradients)
            == len(kv_gradients)
            == len(self.contribution_sum)
        ):
            raise ValueError("gradient/statistic tuples must have equal length")
        with torch.no_grad():
            for index, (answer, kv) in enumerate(
                zip(answer_gradients, kv_gradients)
            ):
                if answer is None or kv is None:
                    continue
                product = answer.detach().float() * kv.detach().float()
                self.contribution_sum[index].add_(product)
                self.positive_count[index].add_(product > 0)
        self.batches += 1

    def build_mask(
        self,
        *,
        sparsity: float,
        minimum_positive_fraction: float,
    ) -> tuple[tuple[torch.Tensor, ...], dict]:
        if self.batches <= 0:
            raise ValueError("cannot build a mask without calibration batches")
        if not 0 < sparsity <= 1:
            raise ValueError("sparsity must be in (0, 1]")
        if not 0.5 <= minimum_positive_fraction <= 1:
            raise ValueError("minimum_positive_fraction must be in [0.5, 1]")

        score_parts = []
        positive_fraction_parts = []
        shapes = []
        for total, count in zip(
            self.contribution_sum,
            self.positive_count,
        ):
            mean_contribution = total / float(self.batches)
            positive_fraction = count.float() / float(self.batches)
            consistency = (positive_fraction - 0.5).clamp_min(0.0)
            score = mean_contribution.clamp_min(0.0) * consistency
            score = torch.where(
                positive_fraction >= minimum_positive_fraction,
                score,
                torch.zeros_like(score),
            )
            score_parts.append(score.detach().flatten().cpu())
            positive_fraction_parts.append(
                positive_fraction.detach().flatten().cpu()
            )
            shapes.append(tuple(total.shape))

        flat_score = torch.cat(score_parts)
        flat_positive_fraction = torch.cat(positive_fraction_parts)
        total_numel = int(flat_score.numel())
        requested = max(1, int(math.ceil(total_numel * sparsity)))
        eligible = torch.nonzero(flat_score > 0, as_tuple=False).flatten()
        selected = min(requested, int(eligible.numel()))
        if selected <= 0:
            raise RuntimeError(
                "no parameter coordinates have consistently positive alignment"
            )
        chosen_within_eligible = torch.topk(
            flat_score.index_select(0, eligible),
            k=selected,
            largest=True,
            sorted=False,
        ).indices
        chosen = eligible.index_select(0, chosen_within_eligible)
        flat_mask = torch.zeros(total_numel, dtype=torch.bool)
        flat_mask[chosen] = True

        masks = []
        per_parameter = []
        offset = 0
        for shape in shapes:
            count = math.prod(shape)
            mask = flat_mask[offset : offset + count].reshape(shape)
            masks.append(mask)
            per_parameter.append(int(mask.sum()))
            offset += count
        selected_scores = flat_score.index_select(0, chosen)
        selected_positive = flat_positive_fraction.index_select(0, chosen)
        return tuple(masks), {
            "calibration_batches": self.batches,
            "total_coordinates": total_numel,
            "eligible_coordinates": int(eligible.numel()),
            "requested_coordinates": requested,
            "selected_coordinates": selected,
            "requested_sparsity": float(sparsity),
            "realized_sparsity": selected / total_numel,
            "minimum_positive_fraction": float(minimum_positive_fraction),
            "selected_score_min": float(selected_scores.min()),
            "selected_score_median": float(selected_scores.median()),
            "selected_score_max": float(selected_scores.max()),
            "selected_positive_fraction_median": float(
                selected_positive.median()
            ),
            "selected_per_parameter": per_parameter,
        }


def random_mask_like(
    masks: Sequence[torch.Tensor],
    *,
    selected_coordinates: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Return a fixed random coordinate mask with identical cardinality."""
    total = sum(int(mask.numel()) for mask in masks)
    if not 0 < selected_coordinates <= total:
        raise ValueError("selected_coordinates is outside the mask size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = torch.randperm(total, generator=generator)[:selected_coordinates]
    flat = torch.zeros(total, dtype=torch.bool)
    flat[chosen] = True
    values = []
    offset = 0
    for mask in masks:
        count = int(mask.numel())
        values.append(flat[offset : offset + count].reshape(mask.shape))
        offset += count
    return tuple(values)


def mask_gradients(
    gradients: Sequence[torch.Tensor | None],
    masks: Sequence[torch.Tensor],
    *,
    complement: bool = False,
) -> GradientTuple:
    """Keep selected coordinates, or their complement, without changing values."""
    if len(gradients) != len(masks):
        raise ValueError("gradient and mask tuples must have equal length")
    output = []
    for gradient, mask in zip(gradients, masks):
        if gradient is None:
            output.append(None)
            continue
        if tuple(gradient.shape) != tuple(mask.shape):
            raise ValueError("gradient and mask shapes differ")
        active = ~mask if complement else mask
        output.append(
            gradient.detach() * active.to(
                device=gradient.device,
                dtype=gradient.dtype,
            )
        )
    return tuple(output)


def rescale_gradients_to_norm(
    gradients: Sequence[torch.Tensor | None],
    *,
    target_norm: float,
) -> tuple[GradientTuple, dict[str, float]]:
    """Scale one auxiliary component to a prespecified L2 norm."""
    if target_norm <= 0:
        raise ValueError("target_norm must be positive")
    observed = gradient_norm(gradients)
    if observed <= 0:
        raise ValueError("cannot rescale a zero gradient component")
    scale = target_norm / observed
    return tuple(
        None if value is None else value.detach() * scale
        for value in gradients
    ), {
        "raw_gradient_norm": observed,
        "target_gradient_norm": float(target_norm),
        "scale": float(scale),
    }


def masks_to_device(
    masks: Sequence[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(mask.to(device=device, dtype=torch.bool) for mask in masks)
