import torch

from src.mech.fidelity_residual_xkv import (
    fit_margin_gradient_bases,
    select_residual_candidate,
    truncate_residual_bases,
)


def test_margin_gradient_bases_are_orthonormal_and_truncatable():
    generator = torch.Generator().manual_seed(7)
    key = torch.randn(12, 3, 2, 8, generator=generator)
    value = torch.randn(12, 3, 2, 8, generator=generator)
    bases, audit = fit_margin_gradient_bases(
        key, value, maximum_rank=4, seed=11
    )
    assert len(bases) == 6
    assert len(audit) == 6
    for basis in bases.values():
        assert basis.shape == (8, 4)
        assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-4)
    truncated = truncate_residual_bases(bases, 2)
    assert all(basis.shape == (8, 2) for basis in truncated.values())


def test_margin_gradient_basis_caps_randomized_svd_width_to_matrix_shape():
    generator = torch.Generator().manual_seed(13)
    key = torch.randn(2, 1, 2, 3, generator=generator)
    value = torch.randn(2, 1, 2, 3, generator=generator)
    bases, _ = fit_margin_gradient_bases(
        key, value, maximum_rank=3, seed=17
    )
    assert bases[(0, "key")].shape == (3, 3)
    assert bases[(0, "value")].shape == (3, 3)


def test_residual_selection_prefers_compression_then_fidelity():
    records = [
        {
            "name": "a", "screen_passed": True,
            "modelled_compression_ratio": 2.0, "first_token_fidelity": 0.96,
            "mean_first_token_kl_from_dense": 0.02,
            "nll_bootstrap_95ci": [0.01, 0.03], "residual_rank": 2,
        },
        {
            "name": "b", "screen_passed": True,
            "modelled_compression_ratio": 2.1, "first_token_fidelity": 0.95,
            "mean_first_token_kl_from_dense": 0.03,
            "nll_bootstrap_95ci": [0.005, 0.02], "residual_rank": 4,
        },
        {
            "name": "c", "screen_passed": False,
            "modelled_compression_ratio": 3.0, "first_token_fidelity": 0.99,
            "mean_first_token_kl_from_dense": 0.01,
            "nll_bootstrap_95ci": [0.02, 0.04], "residual_rank": 1,
        },
    ]
    assert select_residual_candidate(records)["name"] == "b"
