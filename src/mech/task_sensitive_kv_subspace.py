"""Task-sensitive layer subspaces learned from hidden states and exact KV gradients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TaskSensitiveEigensystem:
    matrices: torch.Tensor       # [L,D,D]
    eigenvalues: torch.Tensor    # [L,D], descending
    eigenvectors: torch.Tensor   # [L,D,D], matching eigenvalues
    half_eigenvalues: torch.Tensor   # [2,L,D]
    half_eigenvectors: torch.Tensor  # [2,L,D,D]


def backproject_kv_gradients(
    key_gradients: torch.Tensor,
    value_gradients: torch.Tensor,
    key_response: torch.Tensor,
    value_response: torch.Tensor,
    *,
    mode: str = "joint",
) -> torch.Tensor:
    """Map cache gradients into the hidden space that enters W_K and W_V.

    Response matrices use ``output x input`` orientation. If ``r_K = W_K u``, then
    ``g_h = W_K.T g_K`` is implemented by row-vector multiplication ``g_K @ W_K``.
    """
    if key_gradients.shape != value_gradients.shape or key_gradients.ndim != 4:
        raise ValueError("K/V gradients must share [N,L,P,Dout]")
    layers, output_width, input_width = key_response.shape
    if value_response.shape != key_response.shape:
        raise ValueError("K/V response matrices must share [L,Dout,Din]")
    if key_gradients.shape[1] != layers or key_gradients.shape[-1] != output_width:
        raise ValueError("gradient and response dimensions do not match")
    if mode not in {"joint", "key", "value"}:
        raise ValueError("mode must be joint, key, or value")
    key_hidden = torch.einsum(
        "nlpo,loi->nlpi", key_gradients.double(), key_response.double()
    )
    value_hidden = torch.einsum(
        "nlpo,loi->nlpi", value_gradients.double(), value_response.double()
    )
    if mode == "key":
        return key_hidden.float()
    if mode == "value":
        return value_hidden.float()
    return (key_hidden + value_hidden).float()


def _task_matrix(centered: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    x = centered.reshape(-1, centered.shape[-1]).double()
    g = gradients.reshape(-1, gradients.shape[-1]).double()
    cross = x.T @ g / max(1, x.shape[0])
    # Removing P x changes the loss by -g^T P x. The symmetric matrix below has
    # exactly the same quadratic objective for every orthogonal projector P.
    return -0.5 * (cross + cross.T)


def _descending_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    order = torch.argsort(values, descending=True)
    return values[order], vectors[:, order]


def fit_task_sensitive_eigensystem(
    states: torch.Tensor,
    hidden_gradients: torch.Tensor,
    means: torch.Tensor,
) -> TaskSensitiveEigensystem:
    """Fit full and deterministic half-split task matrices independently by layer."""
    if states.shape != hidden_gradients.shape or states.ndim != 4:
        raise ValueError("states and hidden gradients must share [N,L,P,D]")
    if means.shape != states.shape[1:]:
        raise ValueError("means must have shape [L,P,D]")
    matrices, values, vectors = [], [], []
    half_values = [[], []]
    half_vectors = [[], []]
    for layer in range(states.shape[1]):
        centered = states[:, layer].double() - means[layer].double()
        matrix = _task_matrix(centered, hidden_gradients[:, layer])
        eigenvalues, eigenvectors = _descending_eigh(matrix)
        matrices.append(matrix.float())
        values.append(eigenvalues.float())
        vectors.append(eigenvectors.float())
        for parity in (0, 1):
            split_matrix = _task_matrix(
                centered[parity::2], hidden_gradients[parity::2, layer]
            )
            split_values, split_vectors = _descending_eigh(split_matrix)
            half_values[parity].append(split_values.float())
            half_vectors[parity].append(split_vectors.float())
    return TaskSensitiveEigensystem(
        matrices=torch.stack(matrices),
        eigenvalues=torch.stack(values),
        eigenvectors=torch.stack(vectors),
        half_eigenvalues=torch.stack([torch.stack(value) for value in half_values]),
        half_eigenvectors=torch.stack([torch.stack(value) for value in half_vectors]),
    )


def subspace_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    """Mean squared canonical correlation between equal-rank orthonormal bases."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("bases must share [D,R]")
    if first.shape[1] == 0:
        return 0.0
    singular = torch.linalg.svdvals(first.double().T @ second.double())
    return float(singular.square().mean())


def predicted_removal_damage(
    states: torch.Tensor,
    hidden_gradients: torch.Tensor,
    means: torch.Tensor,
    bases: Mapping[int, torch.Tensor],
    *,
    shuffled_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-example first-order damage for removing each layer's subspace."""
    if states.shape != hidden_gradients.shape or means.shape != states.shape[1:]:
        raise ValueError("incompatible state, gradient, or mean shapes")
    rows = []
    gradients = hidden_gradients
    if shuffled_indices is not None:
        gradients = gradients.index_select(0, shuffled_indices)
    for layer in range(states.shape[1]):
        basis = bases[layer].double()
        if basis.shape[1] == 0:
            rows.append(torch.zeros(states.shape[0], dtype=torch.double))
            continue
        centered = states[:, layer].double() - means[layer].double()
        state_coefficients = centered @ basis
        gradient_coefficients = gradients[:, layer].double() @ basis
        rows.append(-(state_coefficients * gradient_coefficients).sum(-1).mean(-1))
    return torch.stack(rows, dim=1).float()


def bootstrap_mean_interval(
    values: torch.Tensor,
    *,
    seed: int,
    samples: int = 4000,
) -> tuple[float, float]:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("bootstrap values must contain at least two observations")
    generator = torch.Generator().manual_seed(seed)
    values = values.double()
    draws = torch.randint(
        values.numel(), (samples, values.numel()), generator=generator
    )
    means = values[draws].mean(1)
    return float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))


def select_task_sensitive_ranks(
    eigensystem: TaskSensitiveEigensystem,
    rank_states: torch.Tensor,
    rank_hidden_gradients: torch.Tensor,
    means: torch.Tensor,
    *,
    rank_grid: Sequence[int],
    retained_positive_spectrum: float,
    minimum_half_overlap: float,
    seed: int,
    bootstrap_samples: int = 4000,
) -> tuple[dict[int, torch.Tensor], list[int], list[dict]]:
    """Choose a rank from disjoint data and return a complete audit table.

    The candidate family is learned on the discovery split. Rank-selection states
    only decide how much of that frozen ordered basis to retain.
    """
    if not 0 < retained_positive_spectrum <= 1:
        raise ValueError("retained_positive_spectrum must lie in (0,1]")
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
        maximum = min(max(ranks), positive_count)
        positive_values = eigensystem.eigenvalues[layer, :maximum].clamp_min(0)
        denominator = float(positive_values.sum())
        chosen = 0
        layer_rows = []
        for rank in ranks:
            if rank > maximum:
                continue
            basis = eigensystem.eigenvectors[layer, :, :rank]
            half_overlap = subspace_overlap(
                eigensystem.half_eigenvectors[0, layer, :, :rank],
                eigensystem.half_eigenvectors[1, layer, :, :rank],
            )
            learned = predicted_removal_damage(
                rank_states[:, layer : layer + 1],
                rank_hidden_gradients[:, layer : layer + 1],
                means[layer : layer + 1], {0: basis},
            )[:, 0]
            shuffled = predicted_removal_damage(
                rank_states[:, layer : layer + 1],
                rank_hidden_gradients[:, layer : layer + 1],
                means[layer : layer + 1], {0: basis},
                shuffled_indices=permutation,
            )[:, 0]
            excess = learned - shuffled
            interval = bootstrap_mean_interval(
                excess, seed=seed + 100 * layer + rank,
                samples=bootstrap_samples,
            )
            spectrum_fraction = (
                float(positive_values[:rank].sum()) / denominator
                if denominator > 0 else 0.0
            )
            passes = bool(
                spectrum_fraction >= retained_positive_spectrum
                and half_overlap >= minimum_half_overlap
                and interval[0] > 0
            )
            layer_rows.append({
                "rank": rank,
                "positive_spectrum_fraction": spectrum_fraction,
                "half_split_subspace_overlap": half_overlap,
                "mean_excess_predicted_removal_damage": float(excess.mean()),
                "bootstrap_95ci": list(interval),
                "passes_rank_selection": passes,
            })
            if not chosen and passes:
                chosen = rank
        selected[layer] = eigensystem.eigenvectors[layer, :, :chosen]
        selected_ranks.append(chosen)
        audit.append({
            "layer": layer,
            "positive_task_eigenvalues": positive_count,
            "selected_rank": chosen,
            "rank_grid": layer_rows,
        })
    return selected, selected_ranks, audit
