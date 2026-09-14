import torch

from src.mech.direct_cache_task_subspace import (
    select_direct_cache_ranks,
    selected_variance_fraction,
    task_covariance_hybrid_basis,
)
from src.mech.task_sensitive_kv_subspace import fit_task_sensitive_eigensystem


def test_direct_selector_chooses_key_and_value_ranks_without_pairing_them():
    generator = torch.Generator().manual_seed(43)
    states = torch.randn(256, 1, 2, 6, generator=generator)
    first = torch.tensor([1.0, 0, 0, 0, 0, 0])
    second = torch.tensor([0.0, 1, 0, 0, 0, 0])
    gradient = -(
        (states @ first).unsqueeze(-1) * first
        + 0.8 * (states @ second).unsqueeze(-1) * second
    )
    means = torch.zeros(1, 2, 6)
    system = fit_task_sensitive_eigensystem(states, gradient, means)
    bases, ranks, audit = select_direct_cache_ranks(
        system, states, gradient, means, rank_grid=[1, 2, 4],
        retained_best_effect=0.90, minimum_half_overlap=0.4,
        seed=47, bootstrap_samples=500,
    )
    assert ranks == [2]
    assert bases[0].shape == (6, 2)
    assert audit[0]["selected_rank"] == 2


def test_hybrid_basis_is_orthonormal_and_equal_budget():
    task = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    covariance_values = torch.tensor([8.0, 4.0, 2.0, 1.0])
    covariance_vectors = torch.eye(4)
    basis = task_covariance_hybrid_basis(
        task, covariance_values, covariance_vectors, rank=2
    )
    assert basis.shape == (4, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-6)


def test_selected_variance_reports_zero_for_inactive_cache_kind():
    states = torch.randn(8, 2, 3, 4)
    means = states.mean(0)
    values = selected_variance_fraction(
        states, means, {0: torch.eye(4)[:, :2], 1: torch.empty(4, 0)}
    )
    assert 0 < values[0] < 1
    assert values[1] == 0
