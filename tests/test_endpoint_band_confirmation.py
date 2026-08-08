"""Gates and band construction for the exact-match PC-band confirmation."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.eval.official_codi_endpoint_band_confirmation_analysis import (
    analyze_band_confirmation,
)
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    CONTROL_BAND,
    DEFAULT_CONFIRMATION_BANDS,
    PRIMARY_BAND,
    band_name,
    band_variance_share,
    build_band_subspace,
    build_fitted_subspaces,
    build_margin_arm_registry,
    margin_damage_matrix,
    state_covariance,
    validate_margin_subspace,
)

HIDDEN = 768
NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "kaggle_official_codi_endpoint_band_confirmation.ipynb"
)


def _covariance(seed: int, count: int = 1024) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(count, HIDDEN, generator=generator)
    # Make the spectrum strongly anisotropic, as the real colon state is.
    states[:, 0] *= 40.0
    states[:, 1] *= 12.0
    return state_covariance(states - states.mean(dim=0))


def test_band_is_an_interior_slice_of_the_spectrum():
    covariance = _covariance(0)
    prefix = build_fitted_subspaces(
        family="energy", rank=32, state=ANALYTIC_STATE, covariance=covariance
    )
    band = build_band_subspace(covariance=covariance, start=4, stop=32)
    validate_margin_subspace(band)
    assert band.rank == 28
    assert band.name == band_name(4, 32)
    # The band lives inside the top-32 prefix but excludes its leading directions.
    overlap = float((prefix.basis.T @ band.basis).square().sum())
    assert overlap == pytest.approx(28.0, abs=1e-3)
    leading = build_band_subspace(covariance=covariance, start=0, stop=4)
    assert float((leading.basis.T @ band.basis).square().sum()) < 1e-6


def test_band_variance_share_reflects_the_anisotropic_spectrum():
    covariance = _covariance(1)
    leading = band_variance_share(covariance, 0, 4)
    band = band_variance_share(covariance, 4, 32)
    assert leading > band, "the synthetic spectrum should be leading-dominated"
    assert 0 < band < 1 and 0 < leading < 1
    total = sum(band_variance_share(covariance, a, b) for a, b in ((0, 4), (4, 32), (32, 768)))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_band_bounds_are_validated():
    covariance = _covariance(2)
    for start, stop in ((4, 4), (-1, 8), (0, 769), (32, 8)):
        with pytest.raises(ValueError):
            build_band_subspace(covariance=covariance, start=start, stop=stop)


def test_registry_bands_do_not_disturb_existing_arms():
    """Bands are appended last so completed exported arms keep their exact bases."""
    covariance = _covariance(3)
    generator = torch.Generator().manual_seed(4)
    centered = torch.randn(256, HIDDEN, generator=generator)
    damage = {
        "margin": margin_damage_matrix(centered, torch.randn(256, HIDDEN, generator=generator)),
        "answer_nll": margin_damage_matrix(centered, torch.randn(256, HIDDEN, generator=generator)),
    }
    common = dict(
        covariance=covariance,
        damage_matrices=damage,
        readout_matrix=torch.randn(1694, HIDDEN, generator=generator),
        reference_subspaces={},
        rank_grid=(1, 3),
        random_replicates=2,
        random_seed=11,
    )
    without = build_margin_arm_registry(**common)
    with_bands = build_margin_arm_registry(
        **common, bands=((0, 4), (4, 32)), band_random_replicates=2
    )
    for name, subspace in without.items():
        assert name in with_bands
        assert torch.equal(subspace.basis, with_bands[name].basis), name
    assert band_name(4, 32) in with_bands
    assert f"random_matched_band_k028_s{ANALYTIC_STATE}_r001" in with_bands


def test_default_bands_include_the_primary_and_control():
    assert PRIMARY_BAND in DEFAULT_CONFIRMATION_BANDS
    assert CONTROL_BAND in DEFAULT_CONFIRMATION_BANDS


def _runs(*, baseline_acc, primary_retain, control_retain, primary_remove, count=1319, seed=0):
    """Synthetic paired arms with prescribed accuracies."""
    generator = np.random.default_rng(seed)
    base = generator.random(count) < baseline_acc

    def derive(target):
        values = base.copy()
        correct = np.flatnonzero(base)
        keep = int(round(target * len(correct)))
        values[correct[keep:]] = False
        return values

    runs = [
        {
            "arm": "baseline",
            "mode": "remove",
            "band": None,
            "correctness": base.tolist(),
            "endpoint_reached": [True] * count,
        }
    ]
    for band, mode, target in (
        (list(PRIMARY_BAND), "retain", primary_retain),
        (list(CONTROL_BAND), "retain", control_retain),
        (list(PRIMARY_BAND), "remove", primary_remove),
    ):
        runs.append(
            {
                "arm": band_name(*band),
                "mode": mode,
                "band": band,
                "correctness": derive(target).tolist(),
                "endpoint_reached": [True] * count,
            }
        )
    return runs


def _analyze(runs, **overrides):
    kwargs = dict(
        primary_band=list(PRIMARY_BAND),
        control_band=list(CONTROL_BAND),
        majority_band=[4, 16],
        minimum_primary_retention=0.70,
        maximum_control_retention=0.20,
        minimum_primary_removal_points=0.20,
        bootstrap_samples=2000,
    )
    kwargs.update(overrides)
    return analyze_band_confirmation(runs, **kwargs)


def test_gates_confirm_a_clear_band_effect():
    report = _analyze(
        _runs(baseline_acc=0.43, primary_retain=0.86, control_retain=0.07, primary_remove=0.23)
    )
    assert report["status"] == "band_confirmed"
    assert report["band_confirmed"] is True
    assert all(gate["passed"] for gate in report["gates"].values())


def test_insufficient_retention_fails_the_gate():
    report = _analyze(
        _runs(baseline_acc=0.43, primary_retain=0.55, control_retain=0.07, primary_remove=0.23)
    )
    assert report["status"] == "band_not_confirmed"
    assert report["gates"]["sufficiency"]["passed"] is False


def test_a_control_band_that_also_retains_fails_dissociation():
    """If the leading PCs preserve accuracy too, the band is not special."""
    report = _analyze(
        _runs(baseline_acc=0.43, primary_retain=0.86, control_retain=0.80, primary_remove=0.23)
    )
    assert report["gates"]["dissociation"]["passed"] is False
    assert report["band_confirmed"] is False


def test_weak_removal_fails_necessity():
    report = _analyze(
        _runs(baseline_acc=0.43, primary_retain=0.86, control_retain=0.07, primary_remove=0.95)
    )
    assert report["gates"]["necessity"]["passed"] is False
    assert report["band_confirmed"] is False


def test_baseline_drift_blocks_confirmation():
    """A precision regression must stop the report, not shift every comparison."""
    runs = _runs(
        baseline_acc=0.40, primary_retain=0.86, control_retain=0.07, primary_remove=0.23
    )
    report = _analyze(
        runs, reproduction_accuracy=0.4359, maximum_baseline_accuracy_drift=0.015
    )
    assert report["status"] == "baseline_drift_failed"
    assert report["band_confirmed"] is False
    assert report["baseline_drift_passed"] is False


def test_missing_required_arm_is_refused():
    runs = _runs(
        baseline_acc=0.43, primary_retain=0.86, control_retain=0.07, primary_remove=0.23
    )
    with pytest.raises(RuntimeError):
        _analyze([run for run in runs if run["mode"] != "remove" or run["arm"] == "baseline"])


def test_notebook_pins_precision_and_freezes_the_arm_count():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for value in (
        'GENERATION_PRECISION = "float32"',
        '"--precision", GENERATION_PRECISION',
        "assert len(ARMS) == 12",
        'baseline.get("baseline_drift_passed") is True',
        'cache["metadata"]["precision"] == "float32"',
        'assert cache["parity_gate"]["passed"]',
        "analyze_official_codi_endpoint_band_confirmation.py",
        "SHA256SUMS.txt",
    ):
        assert value in source, value
