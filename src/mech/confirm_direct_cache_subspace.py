"""Utilities for held-out confirmation of frozen native-cache subspaces."""
from __future__ import annotations

from typing import Sequence

import torch

from src.mech.layerwise_u28 import random_orthonormal_basis


def covariance_from_eigensystem(eigenvalues, eigenvectors, layer: int) -> torch.Tensor:
    values = eigenvalues[layer].double().clamp_min(0)
    vectors = eigenvectors[layer].double()
    return vectors @ torch.diag(values) @ vectors.T


def covariance_projected_energy(covariance: torch.Tensor, basis: torch.Tensor) -> float:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if basis.ndim != 2 or basis.shape[0] != covariance.shape[0]:
        raise ValueError("basis has the wrong feature width")
    if basis.shape[1] == 0:
        return 0.0
    return float(torch.trace(basis.double().T @ covariance.double() @ basis.double()))


def energy_matched_random_from_covariance(
    covariance: torch.Tensor,
    selected_basis: torch.Tensor,
    *,
    controls: int,
    candidates: int,
    seed: int,
) -> tuple[list[torch.Tensor], dict]:
    """Select equal-rank random bases closest in analytical covariance energy."""
    if not 0 < controls <= candidates:
        raise ValueError("controls must lie in [1,candidates]")
    width, rank = selected_basis.shape
    if covariance.shape != (width, width):
        raise ValueError("covariance and basis widths differ")
    if rank == 0:
        empty = selected_basis.new_zeros((width, 0))
        return [empty.clone() for _ in range(controls)], {
            "target_energy": 0.0, "selected_controls": []
        }
    target = covariance_projected_energy(covariance, selected_basis)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for replicate in range(candidates):
        basis = random_orthonormal_basis(width, rank, generator=generator)
        energy = covariance_projected_energy(covariance, basis)
        rows.append({
            "replicate": replicate, "basis": basis, "energy": energy,
            "relative_gap": abs(energy - target) / max(abs(target), 1e-12),
        })
    rows.sort(key=lambda row: (row["relative_gap"], row["replicate"]))
    chosen = rows[:controls]
    return [row["basis"] for row in chosen], {
        "target_energy": target,
        "selected_controls": [
            {key: row[key] for key in ("replicate", "energy", "relative_gap")}
            for row in chosen
        ],
    }


def bootstrap_interval(
    values: torch.Tensor,
    *,
    seed: int,
    alpha: float = 0.05,
    samples: int = 8000,
) -> list[float]:
    if values.ndim != 1 or values.numel() < 2 or not 0 < alpha < 1:
        raise ValueError("bootstrap needs a vector and alpha in (0,1)")
    generator = torch.Generator().manual_seed(seed)
    values = values.double()
    draws = torch.randint(
        values.numel(), (samples, values.numel()), generator=generator
    )
    means = values[draws].mean(1)
    return [
        float(torch.quantile(means, alpha / 2)),
        float(torch.quantile(means, 1 - alpha / 2)),
    ]


def contiguous_fold_means(values: torch.Tensor, folds: int) -> list[float]:
    if values.ndim != 1 or folds <= 1 or values.numel() % folds:
        raise ValueError("values must divide evenly into at least two folds")
    width = values.numel() // folds
    return [float(chunk.mean()) for chunk in values.split(width)]


def projected_damage_by_position(
    states: torch.Tensor,
    gradients: torch.Tensor,
    means: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Per-example, per-position first-order damage under basis removal."""
    if states.shape != gradients.shape or states.ndim != 3:
        raise ValueError("states and gradients must share [N,P,D]")
    if means.shape != states.shape[1:] or basis.shape[0] != states.shape[-1]:
        raise ValueError("mean or basis shape is incompatible")
    centered = states.double() - means.double()
    return -(
        (centered @ basis.double()) * (gradients.double() @ basis.double())
    ).sum(-1).float()
