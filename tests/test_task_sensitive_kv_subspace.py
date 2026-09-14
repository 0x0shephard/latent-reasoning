import torch

from src.mech.task_sensitive_kv_subspace import (
    backproject_kv_gradients,
    fit_task_sensitive_eigensystem,
    predicted_removal_damage,
    select_task_sensitive_ranks,
    subspace_overlap,
)


def test_backprojection_matches_transposed_kv_action():
    key_gradient = torch.tensor([[[[1.0, 2.0]]]])
    value_gradient = torch.tensor([[[[3.0, 4.0]]]])
    key_response = torch.tensor([[[1.0, 2.0], [0.0, 1.0]]])
    value_response = torch.tensor([[[2.0, 0.0], [1.0, 3.0]]])
    expected_key = key_gradient[0, 0, 0] @ key_response[0]
    expected_value = value_gradient[0, 0, 0] @ value_response[0]
    actual = backproject_kv_gradients(
        key_gradient, value_gradient, key_response, value_response, mode="joint"
    )
    assert torch.allclose(actual[0, 0, 0], expected_key + expected_value)


def test_task_matrix_recovers_a_combination_not_a_variance_axis():
    generator = torch.Generator().manual_seed(19)
    states = torch.randn(256, 1, 2, 4, generator=generator)
    direction = torch.tensor([0.5, -0.5, 0.5, -0.5])
    coefficients = states @ direction
    gradients = -coefficients.unsqueeze(-1) * direction
    means = torch.zeros(1, 2, 4)
    system = fit_task_sensitive_eigensystem(states, gradients, means)
    alignment = abs(float(system.eigenvectors[0, :, 0] @ direction))
    assert alignment > 0.95
    assert float(system.eigenvalues[0, 0]) > 0


def test_predicted_removal_damage_and_rank_selection_use_heldout_examples():
    generator = torch.Generator().manual_seed(23)
    discovery = torch.randn(256, 1, 2, 4, generator=generator)
    rank_states = torch.randn(128, 1, 2, 4, generator=generator)
    direction = torch.tensor([0.5, -0.5, 0.5, -0.5])
    discovery_gradients = -(discovery @ direction).unsqueeze(-1) * direction
    rank_gradients = -(rank_states @ direction).unsqueeze(-1) * direction
    means = torch.zeros(1, 2, 4)
    system = fit_task_sensitive_eigensystem(discovery, discovery_gradients, means)
    bases, ranks, audit = select_task_sensitive_ranks(
        system, rank_states, rank_gradients, means,
        rank_grid=[1, 2, 4], retained_positive_spectrum=0.90,
        minimum_half_overlap=0.5, seed=31, bootstrap_samples=500,
    )
    assert ranks == [1]
    assert bases[0].shape == (4, 1)
    assert audit[0]["rank_grid"][0]["bootstrap_95ci"][0] > 0
    damage = predicted_removal_damage(
        rank_states, rank_gradients, means, bases
    )
    assert float(damage.mean()) > 0


def test_subspace_overlap_is_rotation_invariant():
    first = torch.eye(5)[:, :2]
    rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    assert abs(subspace_overlap(first, first @ rotation) - 1.0) < 1e-6
