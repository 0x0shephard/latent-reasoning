"""Contrastive covariance subspaces for correct versus incorrect CODI answers.

Ordinary PCA asks where *all* answer-cue states vary.  Correct-only PCA asks
where correct states vary, but it does not penalise directions that vary just as
much for wrong states.  This module instead solves the regularised generalized
eigenproblem

    C_correct v = lambda C_wrong v

and uses its largest eigenvalue directions as a correct-specific subspace and
its smallest eigenvalue directions as a wrong-specific subspace.  The returned
bases are Euclidean-orthonormal because interventions are Euclidean projections.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE


CONTRASTIVE_COVARIANCE_SCHEMA_VERSION = 1
CONTRASTIVE_COVARIANCE_CONTRACT = (
    "frozen_checkpoint_gsm8k_test_split_contrastive_covariance_v1"
)


def covariance(values: torch.Tensor, centre: torch.Tensor | None = None) -> torch.Tensor:
    """Population covariance of a non-empty ``[N, d]`` matrix."""
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("covariance needs at least two [N, d] observations")
    matrix = values.double()
    origin = matrix.mean(0) if centre is None else centre.double()
    if origin.shape != (matrix.shape[1],):
        raise ValueError("covariance centre has the wrong shape")
    centred = matrix - origin
    result = centred.T @ centred / matrix.shape[0]
    return 0.5 * (result + result.T)


def shrink_covariance(matrix: torch.Tensor, shrinkage: float) -> torch.Tensor:
    """Shrink a covariance toward its mean eigenvalue times the identity."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if not 0.0 < shrinkage <= 1.0:
        raise ValueError("shrinkage must be in (0, 1]")
    symmetric = 0.5 * (matrix.double() + matrix.double().T)
    scale = torch.diagonal(symmetric).mean().clamp_min(torch.finfo(torch.float64).eps)
    identity = torch.eye(matrix.shape[0], dtype=torch.float64, device=matrix.device)
    return (1.0 - shrinkage) * symmetric + shrinkage * scale * identity


def _canonical_orthonormal_basis(columns: torch.Tensor) -> torch.Tensor:
    basis, _ = torch.linalg.qr(columns.double(), mode="reduced")
    # Projection is sign-invariant, but canonical signs make saved artifacts stable.
    pivots = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivots, torch.arange(basis.shape[1], device=basis.device)])
    signs[signs == 0] = 1
    return (basis * signs.unsqueeze(0)).contiguous()


@dataclass(frozen=True)
class ContrastiveCovarianceFit:
    correct_basis: torch.Tensor
    wrong_basis: torch.Tensor
    correct_mean: torch.Tensor
    wrong_mean: torch.Tensor
    global_mean: torch.Tensor
    correct_covariance: torch.Tensor
    wrong_covariance: torch.Tensor
    generalized_eigenvalues: torch.Tensor
    shrinkage: float
    rank: int


def fit_contrastive_covariance(
    states: torch.Tensor,
    correct: torch.Tensor,
    *,
    rank: int = 28,
    shrinkage: float = 0.05,
) -> ContrastiveCovarianceFit:
    """Fit correct-specific (top) and wrong-specific (bottom) directions.

    Both class covariances receive the same trace-scaled shrinkage.  We whiten
    the wrong covariance, diagonalise the correct covariance in that metric,
    map the generalized eigenvectors back, and finally QR them.  The final QR is
    essential: generalized eigenvectors are orthogonal in the wrong-covariance
    metric, whereas the actual intervention is an ordinary Euclidean projector.
    """
    if states.ndim != 2 or states.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("states must have shape [N, 768]")
    if correct.dtype != torch.bool or correct.shape != (states.shape[0],):
        raise ValueError("correct must be a paired boolean vector")
    if not 0 < rank <= GPT2_HIDDEN_SIZE // 2:
        raise ValueError("rank must be in [1, 384]")
    if int(correct.sum()) <= rank or int((~correct).sum()) <= rank:
        raise ValueError("each correctness class needs more examples than rank")

    values = states.double()
    right, wrong = values[correct], values[~correct]
    right_mean, wrong_mean = right.mean(0), wrong.mean(0)
    right_cov = covariance(right, right_mean)
    wrong_cov = covariance(wrong, wrong_mean)
    regular_right = shrink_covariance(right_cov, shrinkage)
    regular_wrong = shrink_covariance(wrong_cov, shrinkage)

    lower = torch.linalg.cholesky(regular_wrong)
    left_whitened = torch.linalg.solve_triangular(lower, regular_right, upper=False)
    whitened = torch.linalg.solve_triangular(
        lower, left_whitened.T, upper=False
    ).T
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, whitened_vectors = torch.linalg.eigh(whitened)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    generalized = torch.linalg.solve_triangular(
        lower.T, whitened_vectors[:, order], upper=True
    )

    correct_basis = _canonical_orthonormal_basis(generalized[:, :rank])
    wrong_basis = _canonical_orthonormal_basis(generalized[:, -rank:])
    return ContrastiveCovarianceFit(
        correct_basis=correct_basis,
        wrong_basis=wrong_basis,
        correct_mean=right_mean,
        wrong_mean=wrong_mean,
        global_mean=values.mean(0),
        correct_covariance=right_cov,
        wrong_covariance=wrong_cov,
        generalized_eigenvalues=eigenvalues,
        shrinkage=float(shrinkage),
        rank=int(rank),
    )


def heldout_specificity_score(
    states: torch.Tensor,
    correct: torch.Tensor,
    basis: torch.Tensor,
    *,
    correct_mean: torch.Tensor,
    wrong_mean: torch.Tensor,
    orientation: str = "correct",
) -> float:
    """Held-out log energy ratio used to choose shrinkage without readout tuning."""
    if orientation not in {"correct", "wrong"}:
        raise ValueError("orientation must be 'correct' or 'wrong'")
    if int(correct.sum()) == 0 or int((~correct).sum()) == 0:
        raise ValueError("both classes are required on the selection split")
    columns = basis.double()
    correct_energy = (
        ((states.double()[correct] - correct_mean.double()) @ columns).square().sum(1).mean()
    )
    wrong_energy = (
        ((states.double()[~correct] - wrong_mean.double()) @ columns).square().sum(1).mean()
    )
    numerator, denominator = (
        (correct_energy, wrong_energy)
        if orientation == "correct"
        else (wrong_energy, correct_energy)
    )
    epsilon = torch.finfo(torch.float64).eps
    return float(torch.log((numerator + epsilon) / (denominator + epsilon)))


def project_states(
    states: torch.Tensor,
    basis: torch.Tensor,
    centre: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Retain or remove a fitted subspace around an explicitly chosen centre."""
    if mode not in {"retain", "remove"}:
        raise ValueError("mode must be retain or remove")
    values, columns, origin = states.double(), basis.double(), centre.double()
    component = ((values - origin) @ columns) @ columns.T
    return origin + component if mode == "retain" else values - component
