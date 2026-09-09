"""Controlled initializers for the CODI global-head ablation notebook."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from src.mech.global_low_rank_head import (
    NestedLowRankVocabularyHead, activation_whitened_factors,
)


class FixedSubspaceRows(nn.Module):
    """Keep the first k rows fixed and every learned row orthogonal to them."""

    def __init__(self, basis):
        super().__init__()
        self.register_buffer("basis", basis.detach().clone())

    def forward(self, weight):
        residual = weight[self.basis.shape[1]:]
        residual = residual - (residual @ self.basis) @ self.basis.T
        return torch.cat((self.basis.T, residual), dim=0)


def colon_basis(states, start=4, stop=32):
    """PCA band using training first-token states only; indices are zero based."""
    if len(states) <= stop:
        raise ValueError("Need more first-token training states than the PCA stop index")
    centered = states.double() - states.double().mean(0)
    _, vectors = torch.linalg.eigh(centered.T @ centered / (len(states) - 1))
    return vectors.flip(1)[:, start:stop].to(states)


@torch.no_grad()
def initialize_head(states, weight, ranks=(32, 64, 96), *, bias=None,
                    seed=0, initialization="whitened", fixed_basis=None):
    """Initialize equal-total-rank heads; fixed coordinates replace learned slots.

    Constrained heads fit a whitened residual in the orthogonal complement and
    jointly regress vocabulary coefficients on the resulting coordinates. Their
    down-projection constraint remains exact during optimization.
    """
    rank = max(ranks)
    device = weight.device
    states = states.to(device=device, dtype=weight.dtype)
    centre = states.mean(0)
    output_bias = F.linear(centre, weight, bias)
    if fixed_basis is not None:
        basis = fixed_basis.to(weight)
        if basis.shape[1] >= min(ranks):
            raise ValueError("Every rank must leave room for learned residual directions")
        if not torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], device=device),
                              atol=2e-5, rtol=2e-5):
            raise ValueError("Fixed basis must be orthonormal")
        k = basis.shape[1]
        residual_states = states - (states @ basis) @ basis.T
        residual_weight = weight - (weight @ basis) @ basis.T
        _, down, _, _, _ = activation_whitened_factors(
            residual_states, residual_weight, rank-k, seed=seed, compute_device=device)
        down = down - (down @ basis) @ basis.T
        # Normalize the residual row span for a well-conditioned coordinate fit.
        down = torch.linalg.qr(down.T, mode="reduced").Q.T
        down = torch.cat((basis.T, down), dim=0)
        centered = states - centre
        coordinates = centered @ down.T
        gram = coordinates.T @ coordinates
        ridge = max(float(gram.diagonal().mean()) * 1e-6, 1e-8)
        # Avoid allocating [number of states, vocabulary] teacher logits here.
        rhs = (coordinates.T @ centered) @ weight.T
        up = torch.linalg.solve(gram + ridge * torch.eye(rank, device=device), rhs).T
    elif initialization == "whitened":
        centre, down, up, output_bias, _ = activation_whitened_factors(
            states, weight, rank, readout_bias=bias, seed=seed, compute_device=device)
    elif initialization == "weight_svd":
        # Randomized weight-only SVD with the same oversampling/power count.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            left, singular, right = torch.svd_lowrank(
                weight, q=min(rank + 16, min(weight.shape)), niter=1)
        down = right[:, :rank].T
        up = left[:, :rank] * singular[:rank]
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    head = NestedLowRankVocabularyHead.from_whitened_factors(
        centre, down, up, output_bias, ranks)
    if fixed_basis is not None:
        parametrize.register_parametrization(head.down, "weight", FixedSubspaceRows(basis))
        # Bias shifts do not change the fixed row span. All heads train their biases.
    return head


def matched_state_sample(states, count, seed):
    if count <= 0 or count > len(states):
        raise ValueError("Invalid matched-state count")
    order = torch.randperm(len(states), generator=torch.Generator().manual_seed(seed))
    return states[order[:count]]


def paired_interval(reference, candidate, *, samples=5000, seed=0):
    reference = torch.as_tensor(reference, dtype=torch.bool)
    candidate = torch.as_tensor(candidate, dtype=torch.bool)
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("Paired flags must have matching nonempty shapes")
    differences = candidate.float() - reference.float()
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    for start in range(0, samples, 250):
        indices = torch.randint(len(reference), (min(250, samples-start), len(reference)),
                                generator=generator)
        estimates.append(differences[indices].mean(1))
    values = torch.cat(estimates)
    return {"delta_pp": 100 * float(differences.mean()),
            "ci95_low_pp": 100 * float(values.quantile(.025)),
            "ci95_high_pp": 100 * float(values.quantile(.975)),
            "correct_to_wrong": int((reference & ~candidate).sum()),
            "wrong_to_correct": int((~reference & candidate).sum())}
