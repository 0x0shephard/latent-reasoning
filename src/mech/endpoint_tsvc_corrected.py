"""Source-faithful TSV-C-inspired filtering at CODI's answer-cue endpoint.

The released CODI implementation distils the teacher and student activations at
the colon in their respective ``The answer is:`` sequences.  Hugging Face returns
the embedding state followed by one state for each transformer block, and the
released loop consumes that complete tuple.  This module therefore treats the
13-state GPT-2 tuple as the native primary scope.  It intentionally remains
separate from :mod:`src.mech.endpoint_tsvc`, whose historical experiment paired
the teacher colon with the student's pre-cue sixth latent state.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from src.mech.endpoint_tsvc import (
    EndpointTSVCBases,
    project_endpoint_residual,
)


CORRECTED_ENDPOINT_TSVC_SCHEMA_VERSION = 2
CORRECTED_ENDPOINT_SCOPES = ("endpoint_all_states", "endpoint_layer11")
CORRECTED_ENDPOINT_ARMS = (
    "full",
    "learned_top77",
    "random_rank77",
    "bottom_rank77",
    "shuffled_top77",
    "complement",
)
GPT2_HIDDEN_STATE_COUNT = 13
GPT2_BLOCK11_STATE_INDEX = 12


def validate_corrected_endpoint_hidden(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[int, int, int]:
    if student.ndim != 3 or teacher.shape != student.shape:
        raise ValueError("corrected endpoint states must have equal [B,S,D] shapes")
    if student.shape[1] != GPT2_HIDDEN_STATE_COUNT:
        raise ValueError("official GPT-2 endpoint must contain 13 hidden states")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("corrected endpoint states must be finite")
    return tuple(int(value) for value in student.shape)


def corrected_scope_indices(scope: str, state_count: int) -> tuple[int, ...]:
    if state_count != GPT2_HIDDEN_STATE_COUNT:
        raise ValueError("corrected official-CODI scope requires 13 GPT-2 states")
    if scope == "endpoint_all_states":
        return tuple(range(state_count))
    if scope == "endpoint_layer11":
        return (GPT2_BLOCK11_STATE_INDEX,)
    raise ValueError(f"scope must be one of {CORRECTED_ENDPOINT_SCOPES}")


def source_faithful_native_endpoint_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    scope: str = "endpoint_all_states",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Direct loop mirroring the released CODI SmoothL1/std objective.

    This deliberately uses a separate implementation from the projected-loss path
    so a smoke test can compare both values and parameter gradients before any
    calibration is permitted.
    """
    _, state_count, _ = validate_corrected_endpoint_hidden(student, teacher)
    indices = corrected_scope_indices(scope, state_count)
    values = []
    for index in indices:
        selected_teacher = teacher[:, index, :].detach()
        value = F.smooth_l1_loss(
            student[:, index, :],
            selected_teacher,
            reduction="mean",
            beta=1.0,
        )
        # ``Tensor.std()`` in the released source uses the unbiased estimator.
        scale = selected_teacher.std(unbiased=True).clamp_min(eps)
        values.append(value / scale)
    # Preserve the released Python-loop reduction order for the parity reference.
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total / len(values)


def corrected_endpoint_tsvc_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    scope: str,
    mode: str,
    basis: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the native CODI loss to full, projected, or complementary residuals."""
    _, state_count, _ = validate_corrected_endpoint_hidden(student, teacher)
    indices = corrected_scope_indices(scope, state_count)
    index = torch.tensor(indices, device=student.device)
    selected_student = student.index_select(1, index)
    selected_teacher = teacher.detach().index_select(1, index)
    residual = selected_student - selected_teacher

    if mode == "full":
        filtered = residual
    else:
        if basis is None:
            raise ValueError(f"mode {mode!r} requires a basis")
        if basis.ndim != 3 or tuple(basis.shape[:2]) != (
            state_count,
            student.shape[-1],
        ):
            raise ValueError("corrected endpoint basis must have shape [13,D,R]")
        selected_basis = basis.index_select(0, index.to(device=basis.device))
        projected = project_endpoint_residual(residual, selected_basis)
        if mode == "projected":
            filtered = projected
        elif mode == "complement":
            filtered = residual - projected
        else:
            raise ValueError("mode must be full, projected, or complement")

    per_state = []
    for local_index in range(len(indices)):
        value = F.smooth_l1_loss(
            filtered[:, local_index, :],
            torch.zeros_like(filtered[:, local_index, :]),
            reduction="mean",
            beta=1.0,
        )
        scale = selected_teacher[:, local_index, :].std(
            unbiased=True
        ).clamp_min(eps)
        per_state.append(value / scale)
    # Match the released CODI Python-loop accumulation exactly.  ``torch.mean`` is
    # mathematically equivalent but can differ by one float32 ULP because its reduction
    # order is different, which would make the preregistered parity gate platform
    # dependent.
    total = per_state[0]
    for value in per_state[1:]:
        total = total + value
    return total / len(per_state)


def relative_gradient_error(
    observed: Sequence[torch.Tensor | None],
    reference: Sequence[torch.Tensor | None],
    *,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Return relative L2 error and cosine for two aligned gradient tuples."""
    if len(observed) != len(reference):
        raise ValueError("gradient tuples must have equal length")
    dot = torch.zeros((), dtype=torch.float64)
    observed_square = torch.zeros((), dtype=torch.float64)
    reference_square = torch.zeros((), dtype=torch.float64)
    difference_square = torch.zeros((), dtype=torch.float64)
    for left, right in zip(observed, reference):
        if left is None and right is None:
            continue
        if left is None or right is None or left.shape != right.shape:
            raise ValueError("gradient usage or shape differs between parity paths")
        left_cpu = left.detach().double().cpu()
        right_cpu = right.detach().double().cpu()
        dot += (left_cpu * right_cpu).sum()
        observed_square += left_cpu.square().sum()
        reference_square += right_cpu.square().sum()
        difference_square += (left_cpu - right_cpu).square().sum()
    denominator = max(float(reference_square.sqrt()), eps)
    relative = float(difference_square.sqrt()) / denominator
    cosine_denominator = max(
        float(observed_square.sqrt() * reference_square.sqrt()), eps
    )
    cosine = float(dot) / cosine_denominator
    return relative, cosine


def validate_corrected_bases(
    bases: EndpointTSVCBases,
    *,
    hidden_size: int = 768,
) -> None:
    from src.mech.endpoint_tsvc import validate_endpoint_bases

    validate_endpoint_bases(
        bases,
        layers=GPT2_HIDDEN_STATE_COUNT,
        hidden_size=hidden_size,
    )
