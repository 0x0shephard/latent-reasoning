import torch

from src.mech.confirm_direct_cache_subspace import (
    bootstrap_interval, contiguous_fold_means,
    covariance_projected_energy, energy_matched_random_from_covariance,
    projected_damage_by_position,
)


def test_covariance_energy_matching_keeps_rank_and_selects_close_controls():
    covariance = torch.diag(torch.tensor([9.0, 4.0, 2.0, 1.0]))
    selected = torch.eye(4)[:, 1:2]
    controls, audit = energy_matched_random_from_covariance(
        covariance, selected, controls=2, candidates=128, seed=7
    )
    assert len(controls) == 2
    assert all(basis.shape == (4, 1) for basis in controls)
    assert audit["target_energy"] == covariance_projected_energy(covariance, selected)
    assert audit["selected_controls"][0]["relative_gap"] <= audit["selected_controls"][1]["relative_gap"]


def test_position_damage_and_fold_summaries_preserve_grain():
    states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [4.0, 3.0]]])
    gradients = -states
    damage = projected_damage_by_position(
        states, gradients, torch.zeros(2, 2), torch.eye(2)[:, :1]
    )
    assert damage.shape == (2, 2)
    assert torch.equal(damage, states[:, :, :1].square().squeeze(-1))
    assert contiguous_fold_means(torch.arange(8.0), 4) == [0.5, 2.5, 4.5, 6.5]
    interval = bootstrap_interval(torch.arange(8.0), seed=3, samples=500)
    assert interval[0] < 3.5 < interval[1]
