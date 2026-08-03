"""TSV-C-inspired filtering for official-CODI endpoint hidden residuals.

The original TSV-C method decomposes per-layer weight-difference matrices.  This
module deliberately adapts only its truncated-SVD idea to activation residuals:
each row is one paired student-minus-teacher endpoint residual and each layer is
decomposed independently.  It must not be described as unchanged TSV-C.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


ENDPOINT_TSVC_SCHEMA_VERSION = 1
ENDPOINT_SCOPES = ("endpoint_all_layers", "endpoint_layer11")
ENDPOINT_ARMS = (
    "full",
    "learned_top77",
    "random_rank77",
    "bottom_rank77",
    "shuffled_top77",
    "complement",
)


@dataclass(frozen=True)
class EndpointTSVCBases:
    top: torch.Tensor
    bottom: torch.Tensor
    random: torch.Tensor
    eigenvalues: torch.Tensor
    count: int
    rank: int


def create_endpoint_moments(layers: int, hidden_size: int) -> dict:
    if layers <= 0 or hidden_size <= 0:
        raise ValueError("layers and hidden_size must be positive")
    return {
        "count": 0,
        "gram": torch.zeros(
            layers,
            hidden_size,
            hidden_size,
            dtype=torch.float64,
        ),
        "teacher_sum": torch.zeros(layers, hidden_size, dtype=torch.float64),
        "teacher_square_sum": torch.zeros(
            layers, hidden_size, dtype=torch.float64
        ),
    }


def validate_endpoint_hidden(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[int, int, int]:
    if student.ndim != 3 or teacher.shape != student.shape:
        raise ValueError("endpoint states must have equal [B,L,D] shapes")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("endpoint states must be finite")
    return tuple(int(value) for value in student.shape)


def update_endpoint_moments(
    moments: dict,
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> None:
    batch, layers, hidden = validate_endpoint_hidden(student, teacher)
    gram = moments.get("gram")
    if gram is None or tuple(gram.shape) != (layers, hidden, hidden):
        raise ValueError("moment state does not match endpoint tensor shape")
    residual = (student.detach() - teacher.detach()).double().cpu()
    target = teacher.detach().double().cpu()
    moments["gram"].add_(torch.einsum("bld,ble->lde", residual, residual))
    moments["teacher_sum"].add_(target.sum(dim=0))
    moments["teacher_square_sum"].add_(target.square().sum(dim=0))
    moments["count"] = int(moments.get("count", 0)) + batch


def endpoint_moments_state(moments: dict) -> dict:
    return {
        "count": int(moments["count"]),
        "gram": moments["gram"].detach().cpu(),
        "teacher_sum": moments["teacher_sum"].detach().cpu(),
        "teacher_square_sum": moments["teacher_square_sum"].detach().cpu(),
    }


def endpoint_moments_from_state(state: dict) -> dict:
    required = {"count", "gram", "teacher_sum", "teacher_square_sum"}
    if not required.issubset(state):
        raise ValueError("endpoint moment state is incomplete")
    moments = endpoint_moments_state(state)
    if moments["gram"].ndim != 3 or moments["gram"].shape[-1] != moments["gram"].shape[-2]:
        raise ValueError("endpoint gram matrix must have shape [L,D,D]")
    return moments


def _random_bases(
    *,
    layers: int,
    hidden_size: int,
    rank: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = []
    for _ in range(layers):
        sample = torch.randn(
            hidden_size,
            rank,
            generator=generator,
            dtype=torch.float64,
        )
        q, _ = torch.linalg.qr(sample, mode="reduced")
        values.append(q.float())
    return torch.stack(values)


def fit_endpoint_tsvc_bases(
    moments: dict,
    *,
    rank: int,
    random_seed: int,
) -> EndpointTSVCBases:
    count = int(moments.get("count", 0))
    gram = moments.get("gram")
    if count <= 0 or not isinstance(gram, torch.Tensor) or gram.ndim != 3:
        raise ValueError("cannot fit endpoint TSV-C from empty moments")
    layers, hidden_size, other = (int(value) for value in gram.shape)
    if hidden_size != other:
        raise ValueError("endpoint gram matrix must be square")
    if not 0 < rank <= hidden_size:
        raise ValueError("rank must lie in [1, hidden_size]")
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double() / count)
    order = torch.arange(hidden_size - 1, -1, -1)
    descending_values = eigenvalues.index_select(-1, order).clamp_min(0.0)
    descending_vectors = eigenvectors.index_select(-1, order)
    top = descending_vectors[:, :, :rank].float()
    bottom = descending_vectors[:, :, -rank:].float()
    random = _random_bases(
        layers=layers,
        hidden_size=hidden_size,
        rank=rank,
        seed=random_seed,
    )
    bases = EndpointTSVCBases(
        top=top,
        bottom=bottom,
        random=random,
        eigenvalues=descending_values.float(),
        count=count,
        rank=rank,
    )
    validate_endpoint_bases(bases, layers=layers, hidden_size=hidden_size)
    return bases


def validate_endpoint_bases(
    bases: EndpointTSVCBases,
    *,
    layers: int,
    hidden_size: int,
    atol: float = 2e-4,
) -> None:
    if int(bases.count) <= 0 or not 0 < int(bases.rank) <= hidden_size:
        raise ValueError("basis count and rank must be positive and in range")
    expected = (layers, hidden_size, int(bases.rank))
    for name in ("top", "bottom", "random"):
        value = getattr(bases, name)
        if tuple(value.shape) != expected or not torch.isfinite(value).all():
            raise ValueError(f"{name} basis must be finite with shape {expected}")
        identity = torch.eye(bases.rank, dtype=torch.float32)
        observed = torch.einsum("ldr,lds->lrs", value.float(), value.float())
        if not torch.allclose(observed, identity.expand_as(observed), atol=atol, rtol=0):
            raise ValueError(f"{name} basis is not orthonormal")
    if (
        tuple(bases.eigenvalues.shape) != (layers, hidden_size)
        or not torch.isfinite(bases.eigenvalues).all()
        or bool((bases.eigenvalues < -1e-7).any())
    ):
        raise ValueError("eigenvalues must be finite non-negative values shaped [L,D]")


def bases_to_state(bases: EndpointTSVCBases) -> dict:
    return {
        "top": bases.top.detach().cpu(),
        "bottom": bases.bottom.detach().cpu(),
        "random": bases.random.detach().cpu(),
        "eigenvalues": bases.eigenvalues.detach().cpu(),
        "count": int(bases.count),
        "rank": int(bases.rank),
    }


def bases_from_state(state: dict) -> EndpointTSVCBases:
    return EndpointTSVCBases(
        top=state["top"],
        bottom=state["bottom"],
        random=state["random"],
        eigenvalues=state["eigenvalues"],
        count=int(state["count"]),
        rank=int(state["rank"]),
    )


def scope_layers(scope: str, layer_count: int) -> tuple[int, ...]:
    if scope == "endpoint_all_layers":
        return tuple(range(layer_count))
    if scope == "endpoint_layer11":
        if layer_count != 12:
            raise ValueError("endpoint_layer11 requires the 12-layer GPT-2 contract")
        return (11,)
    raise ValueError(f"scope must be one of {ENDPOINT_SCOPES}")


def project_endpoint_residual(
    residual: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    if residual.ndim != 3:
        raise ValueError("residual must have shape [B,L,D]")
    if basis.ndim != 3 or basis.shape[:2] != residual.shape[1:]:
        raise ValueError("basis must have shape [L,D,R]")
    resolved = basis.to(device=residual.device, dtype=residual.dtype)
    coefficients = torch.einsum("bld,ldr->blr", residual, resolved)
    return torch.einsum("blr,ldr->bld", coefficients, resolved)


def endpoint_tsvc_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    scope: str,
    mode: str,
    basis: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CODI-normalized L1 loss on full, projected, or complementary residuals."""
    _, layer_count, _ = validate_endpoint_hidden(student, teacher)
    layers = scope_layers(scope, layer_count)
    index = torch.tensor(layers, device=student.device)
    selected_student = student.index_select(1, index)
    selected_teacher = teacher.detach().index_select(1, index)
    residual = selected_student - selected_teacher
    if mode == "full":
        filtered = residual
    else:
        if basis is None:
            raise ValueError(f"mode {mode!r} requires a basis")
        basis_index = index.to(device=basis.device)
        selected_basis = basis.index_select(0, basis_index).to(student.device)
        projected = project_endpoint_residual(residual, selected_basis)
        if mode == "projected":
            filtered = projected
        elif mode == "complement":
            filtered = residual - projected
        else:
            raise ValueError("mode must be full, projected, or complement")
    per_layer = []
    for local_layer in range(len(layers)):
        teacher_std = selected_teacher[:, local_layer].float().std(
            unbiased=False
        ).clamp_min(eps)
        value = filtered[:, local_layer].abs().mean()
        per_layer.append(value / teacher_std.to(value.dtype))
    return torch.stack(per_layer).sum()


def gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    square = torch.zeros((), dtype=torch.float64)
    for value in gradients:
        if value is not None:
            square += value.detach().double().square().sum().cpu()
    return float(square.sqrt())


def match_gradient_norm(
    gradients: Sequence[torch.Tensor | None],
    reference: Sequence[torch.Tensor | None],
) -> tuple[tuple[torch.Tensor | None, ...], dict[str, float]]:
    """Scale an auxiliary gradient tuple to the full-target auxiliary norm."""
    observed = gradient_norm(gradients)
    target = gradient_norm(reference)
    if observed <= 0.0 or target <= 0.0:
        raise ValueError("auxiliary and reference gradients must be non-zero")
    scale = target / observed
    matched = tuple(
        None if value is None else value.detach() * scale for value in gradients
    )
    return matched, {
        "raw_auxiliary_gradient_norm": observed,
        "reference_full_gradient_norm": target,
        "auxiliary_scale": scale,
        "matched_auxiliary_gradient_norm": gradient_norm(matched),
    }


def explained_energy(eigenvalues: torch.Tensor, rank: int) -> list[float]:
    if eigenvalues.ndim != 2 or not 0 < rank <= eigenvalues.shape[-1]:
        raise ValueError("invalid eigenvalue tensor or rank")
    total = eigenvalues.double().sum(dim=-1).clamp_min(1e-30)
    retained = eigenvalues[:, :rank].double().sum(dim=-1)
    return [float(value) for value in (retained / total)]
