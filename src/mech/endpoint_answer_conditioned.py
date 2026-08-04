"""Answer-conditioned spectral filtering at CODI's answer-cue endpoint.

This module is intentionally separate from the completed fixed-rank TSV-C-inspired
experiment.  Residual principal directions are fitted on one partition, scored on a
second partition by their activation-level alignment with gold-answer gradients, and
only split-stable positive directions are retained.  The embedding state is collected
for diagnosis but is never eligible for the primary block-only target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


ANSWER_CONDITIONED_SCHEMA_VERSION = 1
ANSWER_CONDITIONED_UTILITY_SCHEMA_VERSION = 1
ANSWER_CONDITIONED_SCOPE = "endpoint_blocks_answer_conditioned"
ANSWER_CONDITIONED_ARMS = (
    "full_blocks",
    "answer_conditioned",
    "energy_rank_matched",
    "random_rank_matched",
    "shuffled_answer_rank_matched",
    "shuffled_teacher",
    "complement",
)
ANSWER_CONDITIONED_PRIMARY_CONTROLS = (
    "answer_only",
    "energy_rank_matched",
    "random_rank_matched",
    "shuffled_answer_rank_matched",
    "shuffled_teacher",
)
GPT2_STATE_COUNT = 13
GPT2_HIDDEN_SIZE = 768
FIRST_BLOCK_STATE = 1


@dataclass(frozen=True)
class AnswerConditionedBases:
    answer_conditioned: torch.Tensor
    energy: torch.Tensor
    random: torch.Tensor
    shuffled_answer: torch.Tensor
    ranks: torch.Tensor
    selected_pc_indices: torch.Tensor
    shuffled_pc_indices: torch.Tensor
    eigenvalues: torch.Tensor
    split_z_scores: torch.Tensor
    shuffled_split_z_scores: torch.Tensor
    residual_fit_count: int
    direction_selection_count: int
    minimum_split_z: float
    selection_fdr: float
    maximum_rank_per_state: int

    @property
    def active_states(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, value in enumerate(self.ranks.tolist())
            if int(value) > 0
        )

    @property
    def total_rank(self) -> int:
        return int(self.ranks.sum())


def create_answer_alignment_moments(states: int, hidden_size: int) -> dict:
    if states <= 0 or hidden_size <= 0:
        raise ValueError("states and hidden_size must be positive")
    shape = (2, states, hidden_size)
    return {
        "count": torch.zeros(2, dtype=torch.int64),
        "sum": torch.zeros(shape, dtype=torch.float64),
        "square_sum": torch.zeros(shape, dtype=torch.float64),
        "shuffled_sum": torch.zeros(shape, dtype=torch.float64),
        "shuffled_square_sum": torch.zeros(shape, dtype=torch.float64),
    }


def update_answer_alignment_moments(
    moments: dict,
    products: torch.Tensor,
    shuffled_products: torch.Tensor,
    split_ids: torch.Tensor,
) -> None:
    if products.ndim != 3 or shuffled_products.shape != products.shape:
        raise ValueError("alignment products must have equal [B,S,D] shapes")
    if split_ids.shape != (products.shape[0],):
        raise ValueError("split ids must contain one entry per example")
    if bool(((split_ids != 0) & (split_ids != 1)).any()):
        raise ValueError("split ids must be zero or one")
    values = products.detach().double().cpu()
    shuffled = shuffled_products.detach().double().cpu()
    resolved_splits = split_ids.detach().cpu()
    for split in (0, 1):
        selected = resolved_splits == split
        count = int(selected.sum())
        if not count:
            continue
        current = values[selected]
        current_shuffled = shuffled[selected]
        moments["count"][split] += count
        moments["sum"][split] += current.sum(dim=0)
        moments["square_sum"][split] += current.square().sum(dim=0)
        moments["shuffled_sum"][split] += current_shuffled.sum(dim=0)
        moments["shuffled_square_sum"][split] += current_shuffled.square().sum(dim=0)


def answer_alignment_moments_state(moments: Mapping) -> dict:
    required = (
        "count",
        "sum",
        "square_sum",
        "shuffled_sum",
        "shuffled_square_sum",
    )
    if any(name not in moments for name in required):
        raise ValueError("answer-alignment moment state is incomplete")
    return {
        name: (
            moments[name].detach().cpu()
            if isinstance(moments[name], torch.Tensor)
            else moments[name]
        )
        for name in required
    }


def answer_alignment_moments_from_state(state: Mapping) -> dict:
    moments = answer_alignment_moments_state(state)
    if moments["count"].shape != (2,) or moments["sum"].ndim != 3:
        raise ValueError("invalid answer-alignment moment shapes")
    if any(
        moments[name].shape != moments["sum"].shape
        for name in ("square_sum", "shuffled_sum", "shuffled_square_sum")
    ):
        raise ValueError("answer-alignment moment tensors must have equal shapes")
    return moments


def split_z_scores(
    count: torch.Tensor,
    total: torch.Tensor,
    square_total: torch.Tensor,
    *,
    eps: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return split means and approximate one-sample z scores."""
    if count.shape != (2,) or total.ndim != 3 or total.shape[0] != 2:
        raise ValueError("split statistics must be shaped [2] and [2,S,D]")
    if square_total.shape != total.shape or bool((count < 2).any()):
        raise ValueError("each selection split requires at least two examples")
    resolved_count = count.double().view(2, 1, 1)
    mean = total.double() / resolved_count
    centered = (
        square_total.double() - total.double().square() / resolved_count
    ).clamp_min(0.0)
    variance = centered / (resolved_count - 1.0)
    standard_error = torch.sqrt(variance / resolved_count).clamp_min(eps)
    return mean, mean / standard_error


def fit_residual_eigenbasis(moments: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(moments.get("count", 0))
    gram = moments.get("gram")
    if count <= 0 or not isinstance(gram, torch.Tensor) or gram.ndim != 3:
        raise ValueError("residual moments are empty or malformed")
    if gram.shape[-1] != gram.shape[-2]:
        raise ValueError("residual Gram matrices must be square")
    eigenvalues, eigenvectors = torch.linalg.eigh(gram.double() / count)
    order = torch.arange(gram.shape[-1] - 1, -1, -1)
    return (
        eigenvalues.index_select(-1, order).clamp_min(0.0).float(),
        eigenvectors.index_select(-1, order).float(),
    )


def _padded_bases(
    eigenvectors: torch.Tensor,
    indices: list[list[int]],
    *,
    padded_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states, hidden_size, _ = eigenvectors.shape
    bases = torch.zeros(states, hidden_size, padded_rank, dtype=torch.float32)
    stored_indices = torch.full((states, padded_rank), -1, dtype=torch.int64)
    for state, values in enumerate(indices):
        if not values:
            continue
        index = torch.tensor(values, dtype=torch.long)
        bases[state, :, : len(values)] = eigenvectors[state].index_select(1, index)
        stored_indices[state, : len(values)] = index
    return bases, stored_indices


def _random_rank_matched_bases(
    ranks: torch.Tensor,
    *,
    hidden_size: int,
    padded_rank: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.zeros(len(ranks), hidden_size, padded_rank, dtype=torch.float32)
    for state, rank_value in enumerate(ranks.tolist()):
        rank = int(rank_value)
        if rank <= 0:
            continue
        sample = torch.randn(
            hidden_size,
            rank,
            generator=generator,
            dtype=torch.float64,
        )
        q, _ = torch.linalg.qr(sample, mode="reduced")
        result[state, :, :rank] = q.float()
    return result


def fit_answer_conditioned_bases(
    residual_eigenvalues: torch.Tensor,
    residual_eigenvectors: torch.Tensor,
    alignment_moments: Mapping,
    *,
    minimum_split_z: float,
    selection_fdr: float,
    maximum_rank_per_state: int,
    random_seed: int,
    exclude_embedding: bool = True,
    residual_fit_count: int,
) -> AnswerConditionedBases:
    if residual_eigenvalues.ndim != 2 or residual_eigenvectors.ndim != 3:
        raise ValueError("residual eigensystem must be [S,D] and [S,D,D]")
    states, hidden_size = residual_eigenvalues.shape
    if residual_eigenvectors.shape != (states, hidden_size, hidden_size):
        raise ValueError("residual eigenvector tensor shape is inconsistent")
    if maximum_rank_per_state <= 0 or maximum_rank_per_state > hidden_size:
        raise ValueError("maximum rank must lie in [1, hidden_size]")
    if not 0 < selection_fdr <= 1:
        raise ValueError("selection FDR must lie in (0, 1]")
    moments = answer_alignment_moments_from_state(alignment_moments)
    means, zscores = split_z_scores(
        moments["count"], moments["sum"], moments["square_sum"]
    )
    _, shuffled_zscores = split_z_scores(
        moments["count"],
        moments["shuffled_sum"],
        moments["shuffled_square_sum"],
    )
    if zscores.shape[1:] != residual_eigenvalues.shape:
        raise ValueError("answer-alignment scores do not match residual eigensystem")

    selected: list[list[int]] = [[] for _ in range(states)]
    shuffled_selected: list[list[int]] = [[] for _ in range(states)]
    ranks = torch.zeros(states, dtype=torch.int64)
    first_state = FIRST_BLOCK_STATE if exclude_embedding else 0
    fdr_masks = []
    hypothesis_count = (states - first_state) * hidden_size
    for split in (0, 1):
        block_z = zscores[split, first_state:, :].reshape(-1)
        pvalues = 0.5 * torch.erfc(block_z / (2.0**0.5))
        order = torch.argsort(pvalues, stable=True)
        ordered = pvalues.index_select(0, order)
        thresholds = selection_fdr * torch.arange(
            1, hypothesis_count + 1, dtype=torch.float64
        ) / hypothesis_count
        passing = ordered <= thresholds
        mask = torch.zeros(hypothesis_count, dtype=torch.bool)
        if bool(passing.any()):
            last = int(torch.nonzero(passing, as_tuple=False)[-1])
            cutoff = ordered[last]
            mask = pvalues <= cutoff
        fdr_masks.append(mask.reshape(states - first_state, hidden_size))
    for state in range(first_state, states):
        local_state = state - first_state
        stable = (
            (means[0, state] > 0)
            & (means[1, state] > 0)
            & (zscores[0, state] >= minimum_split_z)
            & (zscores[1, state] >= minimum_split_z)
            & fdr_masks[0][local_state]
            & fdr_masks[1][local_state]
        )
        candidates = torch.nonzero(stable, as_tuple=False).flatten()
        if candidates.numel():
            strength = torch.minimum(
                zscores[0, state, candidates],
                zscores[1, state, candidates],
            )
            order = torch.argsort(strength, descending=True, stable=True)
            chosen = candidates.index_select(0, order)[:maximum_rank_per_state]
            selected[state] = [int(value) for value in chosen]
            ranks[state] = len(selected[state])

        rank = int(ranks[state])
        if rank:
            null_strength = torch.minimum(
                shuffled_zscores[0, state], shuffled_zscores[1, state]
            )
            null_order = torch.argsort(null_strength, descending=True, stable=True)
            shuffled_selected[state] = [
                int(value) for value in null_order[:rank]
            ]

    padded_rank = max(1, int(ranks.max()))
    answer_basis, selected_indices = _padded_bases(
        residual_eigenvectors, selected, padded_rank=padded_rank
    )
    shuffled_basis, shuffled_indices = _padded_bases(
        residual_eigenvectors, shuffled_selected, padded_rank=padded_rank
    )
    energy_indices = [list(range(int(rank))) for rank in ranks.tolist()]
    energy_basis, _ = _padded_bases(
        residual_eigenvectors, energy_indices, padded_rank=padded_rank
    )
    random_basis = _random_rank_matched_bases(
        ranks,
        hidden_size=hidden_size,
        padded_rank=padded_rank,
        seed=random_seed,
    )
    result = AnswerConditionedBases(
        answer_conditioned=answer_basis,
        energy=energy_basis,
        random=random_basis,
        shuffled_answer=shuffled_basis,
        ranks=ranks,
        selected_pc_indices=selected_indices,
        shuffled_pc_indices=shuffled_indices,
        eigenvalues=residual_eigenvalues.float(),
        split_z_scores=zscores.float(),
        shuffled_split_z_scores=shuffled_zscores.float(),
        residual_fit_count=int(residual_fit_count),
        direction_selection_count=int(moments["count"].sum()),
        minimum_split_z=float(minimum_split_z),
        selection_fdr=float(selection_fdr),
        maximum_rank_per_state=int(maximum_rank_per_state),
    )
    validate_answer_conditioned_bases(
        result,
        states=states,
        hidden_size=hidden_size,
        require_candidate=False,
    )
    return result


def validate_answer_conditioned_bases(
    bases: AnswerConditionedBases,
    *,
    states: int = GPT2_STATE_COUNT,
    hidden_size: int = GPT2_HIDDEN_SIZE,
    require_candidate: bool = True,
    atol: float = 3e-4,
) -> None:
    if bases.ranks.shape != (states,) or bool((bases.ranks < 0).any()):
        raise ValueError("answer-conditioned ranks must be non-negative [S]")
    if int(bases.ranks[0]) != 0:
        raise ValueError("the embedding state must be excluded from selection")
    if int(bases.ranks.max()) > bases.maximum_rank_per_state:
        raise ValueError("selected rank exceeds the registered per-state cap")
    if not 0 < bases.selection_fdr <= 1:
        raise ValueError("selection FDR must lie in (0, 1]")
    if require_candidate and bases.total_rank <= 0:
        raise ValueError("no split-stable answer-conditioned directions were selected")
    padded_rank = bases.answer_conditioned.shape[-1]
    expected = (states, hidden_size, padded_rank)
    for name in ("answer_conditioned", "energy", "random", "shuffled_answer"):
        value = getattr(bases, name)
        if value.shape != expected or not torch.isfinite(value).all():
            raise ValueError(f"{name} basis must be finite with shape {expected}")
        for state, rank_value in enumerate(bases.ranks.tolist()):
            rank = int(rank_value)
            if not rank:
                continue
            active = value[state, :, :rank].float()
            observed = active.T @ active
            identity = torch.eye(rank, dtype=torch.float32)
            if not torch.allclose(observed, identity, atol=atol, rtol=0):
                raise ValueError(f"{name} basis is not orthonormal at state {state}")
    if bases.eigenvalues.shape != (states, hidden_size):
        raise ValueError("eigenvalues must have shape [S,D]")
    if bases.split_z_scores.shape != (2, states, hidden_size):
        raise ValueError("split z scores must have shape [2,S,D]")
    if bases.shuffled_split_z_scores.shape != bases.split_z_scores.shape:
        raise ValueError("shuffled split z scores must match actual scores")
    if bases.selected_pc_indices.shape != (states, padded_rank):
        raise ValueError("selected PC indices have the wrong shape")


def answer_conditioned_bases_to_state(bases: AnswerConditionedBases) -> dict:
    return {
        name: getattr(bases, name).detach().cpu()
        for name in (
            "answer_conditioned",
            "energy",
            "random",
            "shuffled_answer",
            "ranks",
            "selected_pc_indices",
            "shuffled_pc_indices",
            "eigenvalues",
            "split_z_scores",
            "shuffled_split_z_scores",
        )
    } | {
        "residual_fit_count": int(bases.residual_fit_count),
        "direction_selection_count": int(bases.direction_selection_count),
        "minimum_split_z": float(bases.minimum_split_z),
        "selection_fdr": float(bases.selection_fdr),
        "maximum_rank_per_state": int(bases.maximum_rank_per_state),
    }


def answer_conditioned_bases_from_state(state: Mapping) -> AnswerConditionedBases:
    return AnswerConditionedBases(
        answer_conditioned=state["answer_conditioned"],
        energy=state["energy"],
        random=state["random"],
        shuffled_answer=state["shuffled_answer"],
        ranks=state["ranks"],
        selected_pc_indices=state["selected_pc_indices"],
        shuffled_pc_indices=state["shuffled_pc_indices"],
        eigenvalues=state["eigenvalues"],
        split_z_scores=state["split_z_scores"],
        shuffled_split_z_scores=state["shuffled_split_z_scores"],
        residual_fit_count=int(state["residual_fit_count"]),
        direction_selection_count=int(state["direction_selection_count"]),
        minimum_split_z=float(state["minimum_split_z"]),
        selection_fdr=float(state["selection_fdr"]),
        maximum_rank_per_state=int(state["maximum_rank_per_state"]),
    )


def project_variable_rank_residual(
    residual: torch.Tensor,
    basis: torch.Tensor,
    ranks: torch.Tensor,
) -> torch.Tensor:
    if residual.ndim != 3 or basis.ndim != 3:
        raise ValueError("residual and basis must be [B,S,D] and [S,D,R]")
    if basis.shape[:2] != residual.shape[1:] or ranks.shape != (residual.shape[1],):
        raise ValueError("variable-rank basis dimensions do not match the residual")
    resolved = basis.to(device=residual.device, dtype=residual.dtype)
    projected_states = []
    for state, rank_value in enumerate(ranks.tolist()):
        rank = int(rank_value)
        if not rank:
            projected_states.append(torch.zeros_like(residual[:, state, :]))
            continue
        active = resolved[state, :, :rank]
        coefficients = residual[:, state, :] @ active
        projected_states.append(coefficients @ active.T)
    return torch.stack(projected_states, dim=1)


def answer_conditioned_endpoint_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    mode: str,
    basis: torch.Tensor | None = None,
    ranks: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply native SmoothL1/std scaling to block-only filtered residuals."""
    if student.ndim != 3 or student.shape != teacher.shape:
        raise ValueError("student and teacher endpoints must have equal [B,S,D] shapes")
    if student.shape[1] != GPT2_STATE_COUNT:
        raise ValueError("official GPT-2 endpoints require 13 hidden-state entries")
    target = teacher.detach()
    residual = student - target
    if mode == "full_blocks":
        filtered = residual
        included = tuple(range(FIRST_BLOCK_STATE, GPT2_STATE_COUNT))
    else:
        if basis is None or ranks is None:
            raise ValueError(f"mode {mode!r} requires a basis and ranks")
        projected = project_variable_rank_residual(residual, basis, ranks)
        if mode == "projected":
            filtered = projected
            included = tuple(
                state
                for state in range(FIRST_BLOCK_STATE, GPT2_STATE_COUNT)
                if int(ranks[state]) > 0
            )
        elif mode == "complement":
            filtered = residual - projected
            included = tuple(range(FIRST_BLOCK_STATE, GPT2_STATE_COUNT))
        else:
            raise ValueError("mode must be full_blocks, projected, or complement")
        if not included:
            raise ValueError("no active answer-conditioned block states")

    values = []
    for state in included:
        value = F.smooth_l1_loss(
            filtered[:, state, :],
            torch.zeros_like(filtered[:, state, :]),
            reduction="mean",
            beta=1.0,
        )
        scale = target[:, state, :].std(unbiased=True).clamp_min(eps)
        values.append(value / scale)
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total / len(values)
