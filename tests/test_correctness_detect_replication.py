"""Contract tests for the test-like, convergence-audited detect replication."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import ast

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.analyze_official_codi_correctness_detect_replication as analyze
import scripts.run_official_codi_correctness_detect_replication as run
from src.mech.endpoint_correctness_geometry import (
    READOUT_KEY,
    apply_logistic,
    fit_logistic_checked,
    roc_auc,
)
from src.mech.endpoint_margin_geometry import (
    MARGIN_GEOMETRY_CONTRACT,
    MARGIN_GEOMETRY_SCHEMA_VERSION,
    MARGIN_GEOMETRY_STATES,
)


HIDDEN = 768
VOCABULARY = 64


class _Settings:
    expected_examples = 144
    fit_examples = 48
    select_examples = 48
    split_seed = 29
    fisher_shrinkage_grid = [0.2]
    probes = ["margin", "fisher_plus_margin"]
    ridge_grid = [0.01]
    solver_max_iterations = 100
    solver_gradient_tolerance = 1e-7
    solver_objective_gap_tolerance = 1e-8
    baseline_probe = "margin"
    primary_probe = "fisher_plus_margin"
    minimum_delta_auc = 0.01
    bootstrap_samples = 100
    bootstrap_seed = 31


class _Config:
    endpoint_correctness_detect_replication = _Settings()


def test_checked_logistic_exports_a_real_convergence_certificate():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(300, 12, generator=generator, dtype=torch.float64)
    labels = (features[:, 0] + 0.4 * features[:, 1] > 0).long()
    weight, bias, stats = fit_logistic_checked(features, labels, l2=0.01)
    diagnostics = stats["optimization"]
    assert diagnostics["converged"] is True
    assert diagnostics["gradient_inf_norm"] <= diagnostics["gradient_tolerance"]
    assert (
        diagnostics["objective_gap_upper_bound"]
        <= diagnostics["objective_gap_tolerance"]
    )
    assert roc_auc(apply_logistic(features, weight, bias, stats), labels) > 0.99


def test_checked_logistic_is_deterministic():
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(180, 9, generator=generator, dtype=torch.float64)
    labels = (features[:, 2] > 0).long()
    first = fit_logistic_checked(features, labels, l2=0.1)
    second = fit_logistic_checked(features, labels, l2=0.1)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2]["optimization"] == second[2]["optimization"]


def test_test_like_partition_is_disjoint_exhaustive_and_deterministic():
    split = run.split_test_like(1319, seed=20260827, fit_size=440, select_size=440)
    again = run.split_test_like(1319, seed=20260827, fit_size=440, select_size=440)
    assert all(torch.equal(left, right) for left, right in zip(split, again))
    assert [part.numel() for part in split] == [440, 440, 439]
    combined = torch.cat(split).tolist()
    assert len(set(combined)) == 1319
    assert sorted(combined) == list(range(1319))
    with pytest.raises(ValueError):
        run.split_test_like(10, seed=1, fit_size=5, select_size=5)


@pytest.fixture
def replication_inputs(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(41)
    readout = torch.randn(VOCABULARY, HIDDEN, generator=generator)
    evaluation = torch.randn(
        _Settings.expected_examples,
        len(MARGIN_GEOMETRY_STATES),
        HIDDEN,
        generator=generator,
    )
    state_index = list(MARGIN_GEOMETRY_STATES).index(12)
    chosen = (evaluation[:, state_index, :] @ readout.T).argmax(dim=-1)
    flip = torch.arange(evaluation.shape[0]) % 3 == 0
    chosen[flip] = (chosen[flip] + 1) % VOCABULARY
    cache = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "request_sha256": "test-like-cache",
        "parity_gate": {"passed": True, "agreement": 1.0},
        "state_order": list(MARGIN_GEOMETRY_STATES),
        # Deliberately no calibration fields: this contract must use evaluation only.
        "evaluation_states": evaluation,
        "evaluation_gold_first_token": chosen,
    }
    states_path = tmp_path / "colon_states.pt"
    readout_path = tmp_path / "readout.pt"
    torch.save(cache, states_path)
    torch.save(
        {"request_sha256": "test-like-cache", READOUT_KEY: readout}, readout_path
    )
    monkeypatch.setattr(run, "load_config", lambda _path: _Config())
    monkeypatch.setattr(analyze, "load_config", lambda _path: _Config())
    return states_path, readout_path


def test_replication_runner_and_gate_are_end_to_end(tmp_path, replication_inputs):
    states_path, readout_path = replication_inputs
    sweep_path = tmp_path / "detect_replication.json"
    report_path = tmp_path / "detect_replication_report.json"
    assert run.main(
        [
            "--states",
            str(states_path),
            "--readout",
            str(readout_path),
            "--output",
            str(sweep_path),
        ]
    ) == 0
    payload = json.loads(sweep_path.read_text())
    assert payload["population"]["source_cache_field"] == "evaluation_states"
    assert payload["splits"]["fit"] == payload["splits"]["select"] == 48
    assert payload["splits"]["test"] == 48
    assert all(
        entry["optimization"]["converged"] for entry in payload["probes"].values()
    )

    outcomes = torch.load(sweep_path.with_suffix(".pt"), weights_only=False)
    all_indices = sum(outcomes["indices"].values(), [])
    assert len(set(all_indices)) == len(all_indices) == _Settings.expected_examples
    assert outcomes["labels"].shape == (48,)

    assert analyze.main(
        ["--sweep", str(sweep_path), "--output", str(report_path)]
    ) == 0
    report = json.loads(report_path.read_text())
    assert report["optimizer_valid"] is True
    assert report["status"] in {
        "test_like_detect_supported",
        "test_like_detect_not_supported",
    }


def test_dedicated_notebook_has_the_frozen_cpu_only_contract():
    path = (
        REPO_ROOT
        / "notebooks"
        / "kaggle_official_codi_correctness_detect_replication.ipynb"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for required in (
        "cached GSM8K test states",
        "440 fit / 440",
        "tests/test_correctness_detect_replication.py",
        "run_official_codi_correctness_detect_replication.py",
        "analyze_official_codi_correctness_detect_replication.py",
        '"--device", "cpu"',
        "objective_gap_upper_bound",
        "SHA256SUMS.txt",
    ):
        assert required in source, required
