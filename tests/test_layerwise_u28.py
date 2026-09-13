import torch

from src.mech.layerwise_u28 import (
    coefficient_r2,
    energy_matched_random_basis,
    fit_ridge_transport,
    principal_angle_cosines,
    subspace_overlap,
)


def test_ridge_transport_recovers_known_subspace_out_of_sample():
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(400, 12, generator=generator)
    true_basis, _ = torch.linalg.qr(torch.randn(12, 3, generator=generator))
    target = source @ true_basis + 0.01 * torch.randn(400, 3, generator=generator)
    transport = fit_ridge_transport(source[:300], target[:300], ridge_ratio=1e-5)
    prediction = transport.predict(source[300:])
    assert coefficient_r2(target[300:], prediction) > 0.99
    assert subspace_overlap(true_basis, transport.basis) > 0.99
    assert torch.all(principal_angle_cosines(true_basis, transport.basis) > 0.99)


def test_energy_matching_is_deterministic_and_rank_matched():
    generator = torch.Generator().manual_seed(3)
    states = torch.randn(100, 10, generator=generator)
    selected, _ = torch.linalg.qr(torch.randn(10, 2, generator=generator))
    first, report1 = energy_matched_random_basis(
        states, states.mean(0), selected, replicates=12, seed=19
    )
    second, report2 = energy_matched_random_basis(
        states, states.mean(0), selected, replicates=12, seed=19
    )
    assert first.shape == (10, 2)
    assert torch.allclose(first.T @ first, torch.eye(2), atol=1e-5)
    assert torch.equal(first, second)
    assert report1 == report2
