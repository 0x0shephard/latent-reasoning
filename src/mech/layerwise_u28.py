"""Transport CODI's final answer eigenspace back through the transformer.

The final U28 basis is a coordinate system at post-``ln_f``.  Its column numbers
cannot simply be reused at earlier layers.  This module fits an out-of-sample
ridge map from a layer activation to the final U28 coordinates and defines the
layer-local subspace as the left singular space of that map.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RidgeTransport:
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    mapping: torch.Tensor
    basis: torch.Tensor
    ridge: float

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        return (states.to(self.mapping) - self.source_mean) @ self.mapping + self.target_mean

    def state_dict(self) -> dict:
        return {
            "source_mean": self.source_mean,
            "target_mean": self.target_mean,
            "mapping": self.mapping,
            "basis": self.basis,
            "ridge": self.ridge,
        }

    @classmethod
    def from_state_dict(cls, value: dict) -> "RidgeTransport":
        return cls(**{key: value[key] for key in (
            "source_mean", "target_mean", "mapping", "basis", "ridge"
        )})


def fit_ridge_transport(
    source: torch.Tensor,
    target_coordinates: torch.Tensor,
    *,
    ridge_ratio: float = 1e-3,
) -> RidgeTransport:
    """Fit ``(x-mean) R + target_mean`` and return ``span(R)``.

    ``ridge_ratio`` is relative to the mean diagonal of ``X^T X`` so the same
    setting remains meaningful across activation scales.
    """
    if source.ndim != 2 or target_coordinates.ndim != 2:
        raise ValueError("source and target coordinates must be matrices")
    if source.shape[0] != target_coordinates.shape[0] or source.shape[0] < 2:
        raise ValueError("source and target need the same number of rows")
    if target_coordinates.shape[1] > source.shape[1]:
        raise ValueError("target rank cannot exceed source width")
    if ridge_ratio < 0:
        raise ValueError("ridge_ratio must be non-negative")
    x = source.detach().double()
    y = target_coordinates.detach().double()
    source_mean = x.mean(0)
    target_mean = y.mean(0)
    xc = x - source_mean
    yc = y - target_mean
    gram = xc.T @ xc
    scale = float(torch.diagonal(gram).mean())
    ridge = float(ridge_ratio * max(scale, torch.finfo(torch.float64).eps))
    mapping = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device),
        xc.T @ yc,
    )
    u, singular, _ = torch.linalg.svd(mapping, full_matrices=False)
    rank = y.shape[1]
    if singular.numel() < rank or float(singular[rank - 1]) <= 0:
        raise RuntimeError("ridge transport is rank deficient")
    basis = u[:, :rank]
    return RidgeTransport(
        source_mean=source_mean.float().cpu(),
        target_mean=target_mean.float().cpu(),
        mapping=mapping.float().cpu(),
        basis=basis.float().cpu(),
        ridge=ridge,
    )


def coefficient_r2(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError("actual and predicted must share matrix shape")
    actual = actual.double()
    predicted = predicted.double()
    residual = (actual - predicted).square().sum()
    total = (actual - actual.mean(0)).square().sum()
    return float(1.0 - residual / total) if float(total) > 0 else 0.0


def principal_angle_cosines(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("bases must have the same ambient dimension")
    return torch.linalg.svdvals(first.double().T @ second.double()).clamp(0, 1).float()


def subspace_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    """Rank-normalized squared principal-angle overlap in ``[0, 1]``."""
    cosines = principal_angle_cosines(first, second)
    return float(cosines.square().sum() / min(first.shape[1], second.shape[1]))


def random_orthonormal_basis(
    width: int, rank: int, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not 0 < rank <= width:
        raise ValueError("rank must lie in [1, width]")
    matrix = torch.randn(width, rank, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis.float()


def projected_energy(states: torch.Tensor, centre: torch.Tensor, basis: torch.Tensor) -> float:
    values = (states.double() - centre.double()) @ basis.double()
    return float(values.square().sum(1).mean())


def energy_matched_random_basis(
    states: torch.Tensor,
    centre: torch.Tensor,
    selected_basis: torch.Tensor,
    *,
    replicates: int = 256,
    seed: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Choose a random rank-matched basis with closest calibration energy."""
    values, report = energy_matched_random_bases(
        states, centre, selected_basis, controls=1, replicates=replicates, seed=seed
    )
    return values[0], report


def energy_matched_random_bases(
    states: torch.Tensor,
    centre: torch.Tensor,
    selected_basis: torch.Tensor,
    *,
    controls: int,
    replicates: int = 256,
    seed: int = 0,
) -> tuple[list[torch.Tensor], dict]:
    """Return the closest-energy random controls from one frozen candidate pool."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0 < controls <= replicates:
        raise ValueError("controls must lie in [1, replicates]")
    target = projected_energy(states, centre, selected_basis)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for replicate in range(replicates):
        basis = random_orthonormal_basis(states.shape[1], selected_basis.shape[1], generator=generator)
        energy = projected_energy(states, centre, basis)
        relative_gap = abs(energy - target) / max(abs(target), 1e-12)
        rows.append({"replicate": replicate, "energy": energy,
                     "relative_gap": relative_gap, "basis": basis})
    ordered = sorted(rows, key=lambda row: (row["relative_gap"], row["replicate"]))
    selected = ordered[:controls]
    return [row["basis"] for row in selected], {
        "target_energy": target,
        "matched_energy": selected[0]["energy"],
        "relative_gap": selected[0]["relative_gap"],
        "replicate": selected[0]["replicate"],
        "selected_controls": [
            {key: row[key] for key in ("replicate", "energy", "relative_gap")}
            for row in selected
        ],
        "null_energies": [
            {key: row[key] for key in ("replicate", "energy", "relative_gap")}
            for row in rows
        ],
    }
