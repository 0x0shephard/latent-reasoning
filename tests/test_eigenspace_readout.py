from __future__ import annotations

import pytest
import torch

from src.mech.eigenspace_readout import (
    LowRankVocabularyHead,
    covariance_eigensystem,
    orthonormal_random_basis,
    readout_aware_scores,
)


def test_full_basis_reconstructs_the_original_bias_free_readout():
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(40, 8, generator=generator)
    weight = torch.randn(13, 8, generator=generator)
    centre, _, basis = covariance_eigensystem(states)
    head = LowRankVocabularyHead.from_basis(weight, basis, centre)
    observed = head(states)
    expected = states @ weight.T
    assert torch.allclose(observed, expected, atol=1e-5, rtol=1e-5)


def test_rank_projection_matches_the_explicit_centred_formula():
    generator = torch.Generator().manual_seed(11)
    states = torch.randn(32, 10, generator=generator)
    weight = torch.randn(17, 10, generator=generator)
    centre, _, eigenvectors = covariance_eigensystem(states)
    basis = eigenvectors[:, 2:6]
    head = LowRankVocabularyHead.from_basis(weight, basis, centre)
    expected = centre @ weight.T + ((states - centre) @ basis) @ (weight @ basis).T
    assert torch.allclose(head(states), expected, atol=1e-5, rtol=1e-5)


def test_readout_aware_score_suppresses_a_common_logit_shift():
    # Direction zero has much higher activation variance but moves every logit by
    # the same amount, so it cannot change an argmax. Direction one separates tokens.
    weight = torch.tensor(
        [[5.0, -2.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.0, 0.0]]
    )
    eigenvalues = torch.tensor([100.0, 1.0, 0.5], dtype=torch.float64)
    eigenvectors = torch.eye(3, dtype=torch.float64)
    scores = readout_aware_scores(weight, eigenvalues, eigenvectors, chunk_size=2)
    assert scores[0].item() == pytest.approx(0.0, abs=1e-12)
    assert scores[1] > scores[0]


def test_random_basis_is_seeded_and_orthonormal():
    left = orthonormal_random_basis(12, 4, seed=19)
    right = orthonormal_random_basis(12, 4, seed=19)
    assert torch.equal(left, right)
    assert torch.allclose(left.T @ left, torch.eye(4, dtype=torch.float64), atol=1e-10)
