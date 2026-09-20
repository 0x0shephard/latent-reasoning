"""Preregistered statistics for rank-16 xKV mechanism confirmation."""
from __future__ import annotations

from collections.abc import Mapping

import torch


def paired_bootstrap_interval(
    values: torch.Tensor,
    *,
    seed: int,
    samples: int = 10_000,
    alpha: float = 0.05,
) -> list[float]:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("paired bootstrap requires at least two paired values")
    if samples < 100 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap configuration")
    generator = torch.Generator().manual_seed(int(seed))
    values = values.detach().double().cpu()
    draws = torch.randint(
        values.numel(), (int(samples), values.numel()), generator=generator
    )
    means = values[draws].mean(1)
    return [
        float(torch.quantile(means, alpha / 2)),
        float(torch.quantile(means, 1 - alpha / 2)),
    ]


def paired_sign_flip_pvalue(
    values: torch.Tensor,
    *,
    seed: int,
    samples: int = 10_000,
) -> float:
    """One-sided paired randomization p-value for a positive mean effect."""
    if values.ndim != 1 or values.numel() < 2 or samples < 100:
        raise ValueError("sign-flip test requires a vector and at least 100 samples")
    values = values.detach().double().cpu()
    observed = values.mean()
    generator = torch.Generator().manual_seed(int(seed))
    exceedances = 0
    remaining = int(samples)
    while remaining:
        batch = min(512, remaining)
        signs = torch.randint(
            0, 2, (batch, values.numel()), generator=generator, dtype=torch.int8
        ).double().mul_(2).sub_(1)
        exceedances += int(((signs * values).mean(1) >= observed).sum())
        remaining -= batch
    return (exceedances + 1) / (int(samples) + 1)


def holm_bonferroni(
    pvalues: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    if not pvalues or not 0 < alpha < 1:
        raise ValueError("Holm correction requires p-values and alpha in (0,1)")
    ordered = sorted((float(value), str(name)) for name, value in pvalues.items())
    if any(not 0 <= value <= 1 for value, _ in ordered):
        raise ValueError("p-values must lie in [0,1]")
    active = True
    result: dict[str, dict[str, float | bool]] = {}
    count = len(ordered)
    for index, (value, name) in enumerate(ordered):
        threshold = alpha / (count - index)
        rejected = bool(active and value <= threshold)
        if not rejected:
            active = False
        result[name] = {
            "pvalue": value,
            "holm_threshold": threshold,
            "rejected": rejected,
        }
    return result


def primary_gate(
    *,
    nll_interval: list[float],
    cache_bits_ratio: float,
    retention: float,
    accuracy_interval: list[float],
    storage_tolerance: float = 0.001,
    minimum_retention: float = 0.95,
    accuracy_noninferiority_margin: float = 0.02,
) -> dict[str, bool]:
    checks = {
        "positive_nll_interval": float(nll_interval[0]) > 0,
        "storage_matched": abs(float(cache_bits_ratio) - 1.0)
        <= float(storage_tolerance),
        "first_token_fidelity": float(retention) >= float(minimum_retention),
        "generation_noninferior": float(accuracy_interval[0])
        > -float(accuracy_noninferiority_margin),
    }
    return {**checks, "passed": all(checks.values())}
