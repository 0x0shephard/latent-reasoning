"""Task-sensitive subspaces learned directly in native key and value cache space."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from src.mech.task_sensitive_kv_subspace import (
    TaskSensitiveEigensystem,
    bootstrap_mean_interval,
    predicted_removal_damage,
    subspace_overlap,
)


def select_direct_cache_ranks(
    eigensystem: TaskSensitiveEigensystem,
    rank_states: torch.Tensor,
    rank_gradients: torch.Tensor,
    means: torch.Tensor,
    *,
    rank_grid: Sequence[int],
    retained_best_effect: float,
    minimum_half_overlap: float,
    seed: int,
    bootstrap_samples: int = 4000,
) -> tuple[dict[int, torch.Tensor], list[int], list[dict]]:
    """Select the smallest stable rank retaining a fraction of best held-out effect.

    Discovery fixes the ordered eigenvectors.  A disjoint rank split estimates each
    nested candidate's excess predicted removal damage over shuffled gradients.  We
    then select the smallest statistically supported rank within
    ``retained_best_effect`` of the best supported candidate.
    """
    if not 0 < retained_best_effect <= 1:
        raise ValueError("retained_best_effect must lie in (0,1]")
    if rank_states.shape != rank_gradients.shape or means.shape != rank_states.shape[1:]:
        raise ValueError("rank states, gradients and means are incompatible")
    ranks = sorted({int(rank) for rank in rank_grid if int(rank) > 0})
    if not ranks:
        raise ValueError("rank_grid must contain a positive rank")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(rank_states.shape[0], generator=generator)
    selected: dict[int, torch.Tensor] = {}
    selected_ranks: list[int] = []
    audit: list[dict] = []
    for layer in range(eigensystem.eigenvectors.shape[0]):
        positive_count = int((eigensystem.eigenvalues[layer] > 0).sum())
        rows = []
        for rank in ranks:
            if rank > positive_count or rank > eigensystem.eigenvectors.shape[-1]:
                continue
            basis = eigensystem.eigenvectors[layer, :, :rank]
            overlap = subspace_overlap(
                eigensystem.half_eigenvectors[0, layer, :, :rank],
                eigensystem.half_eigenvectors[1, layer, :, :rank],
            )
            learned = predicted_removal_damage(
                rank_states[:, layer : layer + 1],
                rank_gradients[:, layer : layer + 1],
                means[layer : layer + 1], {0: basis},
            )[:, 0]
            shuffled = predicted_removal_damage(
                rank_states[:, layer : layer + 1],
                rank_gradients[:, layer : layer + 1],
                means[layer : layer + 1], {0: basis},
                shuffled_indices=permutation,
            )[:, 0]
            excess = learned - shuffled
            interval = bootstrap_mean_interval(
                excess, seed=seed + 100 * layer + rank,
                samples=bootstrap_samples,
            )
            rows.append({
                "rank": rank,
                "half_split_subspace_overlap": overlap,
                "mean_predicted_removal_damage": float(learned.mean()),
                "mean_shuffled_removal_damage": float(shuffled.mean()),
                "mean_excess_predicted_removal_damage": float(excess.mean()),
                "bootstrap_95ci": list(interval),
                "stable_and_nonrandom": bool(
                    overlap >= minimum_half_overlap and interval[0] > 0
                ),
            })
        supported = [row for row in rows if row["stable_and_nonrandom"]]
        chosen = 0
        best_effect = max(
            (row["mean_excess_predicted_removal_damage"] for row in supported),
            default=0.0,
        )
        target = retained_best_effect * best_effect
        for row in supported:
            row["retains_requested_fraction_of_best"] = bool(
                best_effect > 0
                and row["mean_excess_predicted_removal_damage"] >= target
            )
            if not chosen and row["retains_requested_fraction_of_best"]:
                chosen = int(row["rank"])
        selected[layer] = eigensystem.eigenvectors[layer, :, :chosen]
        selected_ranks.append(chosen)
        audit.append({
            "layer": layer,
            "positive_task_eigenvalues": positive_count,
            "best_supported_excess_effect": best_effect,
            "selection_target": target,
            "selected_rank": chosen,
            "rank_grid": rows,
        })
    return selected, selected_ranks, audit


def task_covariance_hybrid_basis(
    task_matrix: torch.Tensor,
    covariance_eigenvalues: torch.Tensor,
    covariance_eigenvectors: torch.Tensor,
    rank: int,
    *,
    task_weight: float = 0.5,
) -> torch.Tensor:
    """Return an equal-rank basis from normalized task and covariance geometry."""
    if task_matrix.ndim != 2 or task_matrix.shape[0] != task_matrix.shape[1]:
        raise ValueError("task_matrix must be square")
    width = task_matrix.shape[0]
    if covariance_eigenvectors.shape != (width, width):
        raise ValueError("covariance eigenvectors have the wrong shape")
    if covariance_eigenvalues.shape != (width,) or not 0 <= task_weight <= 1:
        raise ValueError("invalid covariance spectrum or task weight")
    if not 0 <= rank <= width:
        raise ValueError("rank is outside the feature width")
    if rank == 0:
        return task_matrix.new_zeros((width, 0))
    covariance = (
        covariance_eigenvectors.double()
        @ torch.diag(covariance_eigenvalues.double().clamp_min(0))
        @ covariance_eigenvectors.double().T
    )
    task = 0.5 * (task_matrix.double() + task_matrix.double().T)
    task = task / torch.linalg.matrix_norm(task).clamp_min(1e-12)
    covariance = covariance / torch.linalg.matrix_norm(covariance).clamp_min(1e-12)
    hybrid = task_weight * task + (1 - task_weight) * covariance
    values, vectors = torch.linalg.eigh(0.5 * (hybrid + hybrid.T))
    order = torch.argsort(values, descending=True)
    return vectors[:, order[:rank]].float()


def selected_variance_fraction(
    states: torch.Tensor,
    means: torch.Tensor,
    bases: Mapping[int, torch.Tensor],
) -> list[float]:
    """Fraction of centered cache energy retained by each layer-local basis."""
    if states.ndim != 4 or means.shape != states.shape[1:]:
        raise ValueError("states must be [N,L,P,D] and means [L,P,D]")
    result = []
    for layer in range(states.shape[1]):
        basis = bases[layer].double()
        if basis.shape[1] == 0:
            result.append(0.0)
            continue
        centered = states[:, layer].double() - means[layer].double()
        numerator = (centered @ basis).square().sum()
        denominator = centered.square().sum().clamp_min(1e-12)
        result.append(float(numerator / denominator))
    return result
