"""Parameter-aware residual-PC selection at CODI's answer-cue endpoint.

The earlier answer-conditioned screen ranked residual PCs in activation coordinates.
This module instead estimates the cosine between each PC target's induced LoRA
parameter gradient and the gold-answer parameter gradient.  Candidate norms use a
deterministic Hutchinson sketch; dot products with the answer gradient are exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.mech.endpoint_answer_conditioned import (
    GPT2_HIDDEN_SIZE,
    GPT2_STATE_COUNT,
    answer_alignment_moments_from_state,
    answer_conditioned_endpoint_loss,
    split_z_scores,
)


PARAMETER_AWARE_SCHEMA_VERSION = 1
PARAMETER_AWARE_UTILITY_SCHEMA_VERSION = 1
PARAMETER_AWARE_SCOPE = "endpoint_final_two_blocks_parameter_aware"
PARAMETER_AWARE_CANDIDATE_STATES = (11, 12)
PARAMETER_AWARE_ARMS = (
    "full_blocks",
    "parameter_aware",
    "energy_rank_matched",
    "random_rank_matched",
    "shuffled_answer_rank_matched",
    "shuffled_teacher",
    "complement",
)
PARAMETER_AWARE_PRIMARY_CONTROLS = (
    "answer_only",
    "energy_rank_matched",
    "random_rank_matched",
    "shuffled_answer_rank_matched",
    "shuffled_teacher",
)


@dataclass(frozen=True)
class ParameterAwareBases:
    parameter_aware: torch.Tensor
    energy: torch.Tensor
    random: torch.Tensor
    shuffled_answer: torch.Tensor
    ranks: torch.Tensor
    selected_pc_indices: torch.Tensor
    shuffled_pc_indices: torch.Tensor
    eigenvalues: torch.Tensor
    split_cosine_means: torch.Tensor
    split_z_scores: torch.Tensor
    shuffled_split_cosine_means: torch.Tensor
    shuffled_split_z_scores: torch.Tensor
    residual_fit_count: int
    direction_selection_batches: int
    direction_selection_examples: int
    candidate_states: tuple[int, ...]
    candidate_pc_count: int
    hutchinson_probes: int
    minimum_split_z: float
    selection_fdr: float
    maximum_rank_per_state: int

    @property
    def active_states(self) -> tuple[int, ...]:
        return tuple(
            state for state, rank in enumerate(self.ranks.tolist()) if int(rank) > 0
        )

    @property
    def total_rank(self) -> int:
        return int(self.ranks.sum())


def residual_pc_candidate_losses(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    candidate_states: Sequence[int] = PARAMETER_AWARE_CANDIDATE_STATES,
    candidate_pc_count: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one native-scaled rank-one endpoint loss per candidate residual PC."""
    if student.ndim != 3 or student.shape != teacher.shape:
        raise ValueError("student and teacher endpoints must have equal [B,S,D] shapes")
    states, hidden = student.shape[1:]
    if eigenvectors.shape != (states, hidden, hidden):
        raise ValueError("residual eigenvectors must have shape [S,D,D]")
    resolved_states = tuple(int(value) for value in candidate_states)
    if not resolved_states or len(set(resolved_states)) != len(resolved_states):
        raise ValueError("candidate states must be non-empty and unique")
    if min(resolved_states) < 1 or max(resolved_states) >= states:
        raise ValueError("candidate states must be transformer-block state indices")
    if not 0 < candidate_pc_count <= hidden:
        raise ValueError("candidate PC count must lie in [1, hidden size]")

    detached_teacher = teacher.detach()
    losses = []
    identities = []
    for state in resolved_states:
        basis = eigenvectors[state, :, :candidate_pc_count].to(
            device=student.device, dtype=student.dtype
        )
        residual = student[:, state, :] - detached_teacher[:, state, :]
        coefficients = residual @ basis
        projected = coefficients.T.unsqueeze(-1) * basis.T.unsqueeze(1)
        distance = F.smooth_l1_loss(
            projected,
            torch.zeros_like(projected),
            reduction="none",
            beta=1.0,
        ).mean(dim=(1, 2))
        scale = detached_teacher[:, state, :].std(unbiased=True).clamp_min(eps)
        losses.append(distance / scale)
        identities.extend((state, pc) for pc in range(candidate_pc_count))
    return (
        torch.cat(losses, dim=0),
        torch.tensor(identities, dtype=torch.int64),
    )


def _gradient_norm(
    gradients: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    total = None
    for value in gradients:
        if value is None:
            continue
        term = value.detach().double().square().sum()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("gradient direction is empty")
    return total.sqrt()


def _rademacher_probe(
    parameters: Sequence[torch.Tensor],
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    values = []
    for parameter in parameters:
        cpu = torch.randint(
            0,
            2,
            parameter.shape,
            generator=generator,
            dtype=torch.int8,
            device="cpu",
        )
        values.append(
            cpu.to(device=parameter.device, dtype=parameter.dtype).mul_(2).sub_(1)
        )
    return tuple(values)


def parameter_gradient_cosines(
    candidate_losses: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    reference_gradients: Mapping[str, Sequence[torch.Tensor | None]],
    *,
    hutchinson_probes: int,
    seed: int,
    probe_directions: Sequence[Sequence[torch.Tensor]] | None = None,
) -> dict:
    """Estimate all candidate/reference parameter-gradient cosines in one graph.

    For candidate losses ``l_c`` and a fixed direction ``v``, differentiating
    ``<grad_theta sum(w_c l_c), v>`` with respect to ``w`` returns every
    ``<grad_theta l_c, v>`` simultaneously.  Rademacher parameter directions then
    estimate each candidate-gradient norm without one backward pass per PC.
    """
    if candidate_losses.ndim != 1 or not candidate_losses.numel():
        raise ValueError("candidate losses must be a non-empty vector")
    resolved_parameters = tuple(parameters)
    if not resolved_parameters or any(not value.requires_grad for value in resolved_parameters):
        raise ValueError("all supplied parameters must require gradients")
    if not reference_gradients:
        raise ValueError("at least one reference gradient is required")
    if any(len(values) != len(resolved_parameters) for values in reference_gradients.values()):
        raise ValueError("reference gradients must align with parameters")
    if hutchinson_probes <= 0:
        raise ValueError("Hutchinson probe count must be positive")
    if probe_directions is not None and len(probe_directions) != hutchinson_probes:
        raise ValueError("explicit probes must match the registered probe count")

    weights = torch.zeros_like(candidate_losses, requires_grad=True)
    combined = torch.autograd.grad(
        torch.dot(weights, candidate_losses),
        resolved_parameters,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    if all(value is None for value in combined):
        raise RuntimeError("candidate losses are disconnected from trainable parameters")

    def directional_values(
        direction: Sequence[torch.Tensor | None], *, retain_graph: bool
    ) -> torch.Tensor:
        scalar = None
        for candidate_value, direction_value in zip(combined, direction):
            if candidate_value is None or direction_value is None:
                continue
            term = (candidate_value * direction_value.detach()).sum()
            scalar = term if scalar is None else scalar + term
        if scalar is None:
            raise RuntimeError("parameter direction has no candidate-gradient overlap")
        value = torch.autograd.grad(
            scalar,
            weights,
            retain_graph=retain_graph,
            allow_unused=False,
        )[0]
        return value.detach().double().cpu()

    numerators = {
        name: directional_values(values, retain_graph=True)
        for name, values in reference_gradients.items()
    }
    reference_norms = {
        name: float(_gradient_norm(values).detach().cpu())
        for name, values in reference_gradients.items()
    }
    norm_square = torch.zeros(candidate_losses.numel(), dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for index in range(hutchinson_probes):
        if probe_directions is None:
            probe = _rademacher_probe(resolved_parameters, generator=generator)
        else:
            probe = tuple(probe_directions[index])
            if len(probe) != len(resolved_parameters):
                raise ValueError("probe direction does not align with parameters")
        projection = directional_values(
            probe,
            retain_graph=index + 1 < hutchinson_probes,
        )
        norm_square += projection.square()
    candidate_norms = (norm_square / hutchinson_probes).clamp_min(0.0).sqrt()

    cosines = {}
    for name, numerator in numerators.items():
        denominator = candidate_norms * reference_norms[name]
        cosines[name] = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.zeros_like(numerator),
        ).clamp(-1.0, 1.0)
    return {
        "cosines": cosines,
        "dots": numerators,
        "candidate_norm_estimates": candidate_norms,
        "reference_norms": reference_norms,
        "hutchinson_probes": int(hutchinson_probes),
        "seed": int(seed),
    }


def _padded_bases(
    eigenvectors: torch.Tensor,
    indices: list[list[int]],
    *,
    padded_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states, hidden, _ = eigenvectors.shape
    bases = torch.zeros(states, hidden, padded_rank, dtype=torch.float32)
    stored = torch.full((states, padded_rank), -1, dtype=torch.int64)
    for state, selected in enumerate(indices):
        if not selected:
            continue
        index = torch.tensor(selected, dtype=torch.long)
        bases[state, :, : len(selected)] = eigenvectors[state].index_select(1, index)
        stored[state, : len(selected)] = index
    return bases, stored


def _random_bases(
    ranks: torch.Tensor,
    *,
    hidden_size: int,
    padded_rank: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    result = torch.zeros(len(ranks), hidden_size, padded_rank, dtype=torch.float32)
    for state, value in enumerate(ranks.tolist()):
        rank = int(value)
        if not rank:
            continue
        sample = torch.randn(hidden_size, rank, generator=generator, dtype=torch.float64)
        result[state, :, :rank] = torch.linalg.qr(sample, mode="reduced")[0].float()
    return result


def _bh_mask(pvalues: torch.Tensor, fdr: float) -> torch.Tensor:
    order = torch.argsort(pvalues, stable=True)
    ordered = pvalues.index_select(0, order)
    thresholds = fdr * torch.arange(
        1, pvalues.numel() + 1, dtype=torch.float64
    ) / pvalues.numel()
    passing = ordered <= thresholds
    result = torch.zeros_like(pvalues, dtype=torch.bool)
    if bool(passing.any()):
        cutoff = ordered[int(torch.nonzero(passing, as_tuple=False)[-1])]
        result = pvalues <= cutoff
    return result


def fit_parameter_aware_bases(
    residual_eigenvalues: torch.Tensor,
    residual_eigenvectors: torch.Tensor,
    alignment_moments: Mapping,
    *,
    candidate_states: Sequence[int],
    candidate_pc_count: int,
    hutchinson_probes: int,
    minimum_split_z: float,
    selection_fdr: float,
    maximum_rank_per_state: int,
    random_seed: int,
    residual_fit_count: int,
    direction_selection_examples: int,
) -> ParameterAwareBases:
    if residual_eigenvalues.ndim != 2 or residual_eigenvectors.ndim != 3:
        raise ValueError("residual eigensystem must be [S,D] and [S,D,D]")
    states, hidden = residual_eigenvalues.shape
    if residual_eigenvectors.shape != (states, hidden, hidden):
        raise ValueError("residual eigenvector tensor shape is inconsistent")
    resolved_states = tuple(int(value) for value in candidate_states)
    if (
        not resolved_states
        or len(set(resolved_states)) != len(resolved_states)
        or min(resolved_states) < 1
        or max(resolved_states) >= states
    ):
        raise ValueError("candidate states are invalid")
    if not 0 < candidate_pc_count <= hidden:
        raise ValueError("candidate PC count must lie in [1, hidden size]")
    if not 0 < maximum_rank_per_state <= candidate_pc_count:
        raise ValueError("maximum rank must not exceed the candidate PC count")
    if not 0 < selection_fdr <= 1:
        raise ValueError("selection FDR must lie in (0, 1]")

    moments = answer_alignment_moments_from_state(alignment_moments)
    means, zscores = split_z_scores(
        moments["count"], moments["sum"], moments["square_sum"]
    )
    shuffled_means, shuffled_zscores = split_z_scores(
        moments["count"], moments["shuffled_sum"], moments["shuffled_square_sum"]
    )
    candidate_locations = [
        (state, pc)
        for state in resolved_states
        for pc in range(candidate_pc_count)
    ]
    split_masks = []
    for split in (0, 1):
        values = torch.tensor(
            [zscores[split, state, pc] for state, pc in candidate_locations],
            dtype=torch.float64,
        )
        pvalues = 0.5 * torch.erfc(values / (2.0**0.5))
        split_masks.append(_bh_mask(pvalues, selection_fdr))

    selected: list[list[int]] = [[] for _ in range(states)]
    shuffled_selected: list[list[int]] = [[] for _ in range(states)]
    ranks = torch.zeros(states, dtype=torch.int64)
    offset = 0
    for state in resolved_states:
        eligible = torch.arange(candidate_pc_count)
        stable = (
            (means[0, state, :candidate_pc_count] > 0)
            & (means[1, state, :candidate_pc_count] > 0)
            & (zscores[0, state, :candidate_pc_count] >= minimum_split_z)
            & (zscores[1, state, :candidate_pc_count] >= minimum_split_z)
            & split_masks[0][offset : offset + candidate_pc_count]
            & split_masks[1][offset : offset + candidate_pc_count]
        )
        candidates = eligible[stable]
        if candidates.numel():
            strength = torch.minimum(
                zscores[0, state, candidates], zscores[1, state, candidates]
            )
            order = torch.argsort(strength, descending=True, stable=True)
            chosen = candidates.index_select(0, order)[:maximum_rank_per_state]
            selected[state] = [int(value) for value in chosen]
            ranks[state] = len(selected[state])
        rank = int(ranks[state])
        if rank:
            shuffled_strength = torch.minimum(
                shuffled_zscores[0, state, :candidate_pc_count],
                shuffled_zscores[1, state, :candidate_pc_count],
            )
            shuffled_order = torch.argsort(
                shuffled_strength, descending=True, stable=True
            )
            shuffled_selected[state] = [int(value) for value in shuffled_order[:rank]]
        offset += candidate_pc_count

    padded_rank = max(1, int(ranks.max()))
    parameter_basis, selected_indices = _padded_bases(
        residual_eigenvectors, selected, padded_rank=padded_rank
    )
    shuffled_basis, shuffled_indices = _padded_bases(
        residual_eigenvectors, shuffled_selected, padded_rank=padded_rank
    )
    energy_indices = [list(range(int(value))) for value in ranks.tolist()]
    energy_basis, _ = _padded_bases(
        residual_eigenvectors, energy_indices, padded_rank=padded_rank
    )
    result = ParameterAwareBases(
        parameter_aware=parameter_basis,
        energy=energy_basis,
        random=_random_bases(
            ranks,
            hidden_size=hidden,
            padded_rank=padded_rank,
            seed=random_seed,
        ),
        shuffled_answer=shuffled_basis,
        ranks=ranks,
        selected_pc_indices=selected_indices,
        shuffled_pc_indices=shuffled_indices,
        eigenvalues=residual_eigenvalues.float(),
        split_cosine_means=means.float(),
        split_z_scores=zscores.float(),
        shuffled_split_cosine_means=shuffled_means.float(),
        shuffled_split_z_scores=shuffled_zscores.float(),
        residual_fit_count=int(residual_fit_count),
        direction_selection_batches=int(moments["count"].sum()),
        direction_selection_examples=int(direction_selection_examples),
        candidate_states=resolved_states,
        candidate_pc_count=int(candidate_pc_count),
        hutchinson_probes=int(hutchinson_probes),
        minimum_split_z=float(minimum_split_z),
        selection_fdr=float(selection_fdr),
        maximum_rank_per_state=int(maximum_rank_per_state),
    )
    validate_parameter_aware_bases(
        result, states=states, hidden_size=hidden, require_candidate=False
    )
    return result


def validate_parameter_aware_bases(
    bases: ParameterAwareBases,
    *,
    states: int = GPT2_STATE_COUNT,
    hidden_size: int = GPT2_HIDDEN_SIZE,
    require_candidate: bool = True,
    atol: float = 3e-4,
) -> None:
    if bases.ranks.shape != (states,) or bool((bases.ranks < 0).any()):
        raise ValueError("parameter-aware ranks must be non-negative [S]")
    allowed = set(bases.candidate_states)
    if any(int(rank) and state not in allowed for state, rank in enumerate(bases.ranks)):
        raise ValueError("selection escaped the registered candidate states")
    if require_candidate and bases.total_rank <= 0:
        raise ValueError("no split-stable parameter-aware directions were selected")
    if int(bases.ranks.max()) > bases.maximum_rank_per_state:
        raise ValueError("selected rank exceeds the registered cap")
    padded_rank = bases.parameter_aware.shape[-1]
    expected = (states, hidden_size, padded_rank)
    for name in ("parameter_aware", "energy", "random", "shuffled_answer"):
        value = getattr(bases, name)
        if value.shape != expected or not torch.isfinite(value).all():
            raise ValueError(f"{name} basis must be finite with shape {expected}")
        for state, rank_value in enumerate(bases.ranks.tolist()):
            rank = int(rank_value)
            if not rank:
                continue
            active = value[state, :, :rank].float()
            if not torch.allclose(
                active.T @ active,
                torch.eye(rank),
                atol=atol,
                rtol=0,
            ):
                raise ValueError(f"{name} basis is not orthonormal at state {state}")
    score_shape = (2, states, hidden_size)
    for name in (
        "split_cosine_means",
        "split_z_scores",
        "shuffled_split_cosine_means",
        "shuffled_split_z_scores",
    ):
        if getattr(bases, name).shape != score_shape:
            raise ValueError(f"{name} must have shape {score_shape}")


def parameter_aware_bases_to_state(bases: ParameterAwareBases) -> dict:
    tensors = (
        "parameter_aware",
        "energy",
        "random",
        "shuffled_answer",
        "ranks",
        "selected_pc_indices",
        "shuffled_pc_indices",
        "eigenvalues",
        "split_cosine_means",
        "split_z_scores",
        "shuffled_split_cosine_means",
        "shuffled_split_z_scores",
    )
    return {name: getattr(bases, name).detach().cpu() for name in tensors} | {
        "residual_fit_count": int(bases.residual_fit_count),
        "direction_selection_batches": int(bases.direction_selection_batches),
        "direction_selection_examples": int(bases.direction_selection_examples),
        "candidate_states": list(bases.candidate_states),
        "candidate_pc_count": int(bases.candidate_pc_count),
        "hutchinson_probes": int(bases.hutchinson_probes),
        "minimum_split_z": float(bases.minimum_split_z),
        "selection_fdr": float(bases.selection_fdr),
        "maximum_rank_per_state": int(bases.maximum_rank_per_state),
    }


def parameter_aware_bases_from_state(state: Mapping) -> ParameterAwareBases:
    return ParameterAwareBases(
        parameter_aware=state["parameter_aware"],
        energy=state["energy"],
        random=state["random"],
        shuffled_answer=state["shuffled_answer"],
        ranks=state["ranks"],
        selected_pc_indices=state["selected_pc_indices"],
        shuffled_pc_indices=state["shuffled_pc_indices"],
        eigenvalues=state["eigenvalues"],
        split_cosine_means=state["split_cosine_means"],
        split_z_scores=state["split_z_scores"],
        shuffled_split_cosine_means=state["shuffled_split_cosine_means"],
        shuffled_split_z_scores=state["shuffled_split_z_scores"],
        residual_fit_count=int(state["residual_fit_count"]),
        direction_selection_batches=int(state["direction_selection_batches"]),
        direction_selection_examples=int(state["direction_selection_examples"]),
        candidate_states=tuple(int(value) for value in state["candidate_states"]),
        candidate_pc_count=int(state["candidate_pc_count"]),
        hutchinson_probes=int(state["hutchinson_probes"]),
        minimum_split_z=float(state["minimum_split_z"]),
        selection_fdr=float(state["selection_fdr"]),
        maximum_rank_per_state=int(state["maximum_rank_per_state"]),
    )


def parameter_aware_endpoint_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    mode: str,
    basis: torch.Tensor | None = None,
    ranks: torch.Tensor | None = None,
) -> torch.Tensor:
    return answer_conditioned_endpoint_loss(
        student,
        teacher,
        mode=mode,
        basis=basis,
        ranks=ranks,
    )
