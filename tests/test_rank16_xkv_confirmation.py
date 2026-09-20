import torch

from src.mech.rank16_xkv_confirmation import (
    holm_bonferroni,
    paired_bootstrap_interval,
    paired_sign_flip_pvalue,
    primary_gate,
)


def test_paired_statistics_detect_a_consistently_positive_effect():
    values = torch.linspace(0.1, 0.3, 32)
    interval = paired_bootstrap_interval(values, seed=7, samples=500)
    assert interval[0] > 0
    assert paired_sign_flip_pvalue(values, seed=7, samples=1000) < 0.01


def test_holm_bonferroni_is_step_down_and_name_stable():
    result = holm_bonferroni({"protection": 0.06, "allocation": 0.001, "fisher": 0.02})
    assert result["allocation"]["rejected"]
    assert result["fisher"]["rejected"]
    assert not result["protection"]["rejected"]


def test_primary_gate_requires_quality_storage_fidelity_and_generation():
    passed = primary_gate(
        nll_interval=[0.01, 0.2],
        cache_bits_ratio=1.00002,
        retention=0.98,
        accuracy_interval=[-0.01, 0.03],
    )
    assert passed["passed"]
    failed = primary_gate(
        nll_interval=[-0.01, 0.2],
        cache_bits_ratio=1.0,
        retention=0.98,
        accuracy_interval=[-0.01, 0.03],
    )
    assert not failed["passed"]
