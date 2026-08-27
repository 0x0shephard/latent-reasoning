from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.run_official_codi_paired_correction as paired_runner

from src.eval.official_codi_paired_correction_analysis import analyze_paired_correction
from src.mech.endpoint_paired_correction import (
    PAIRED_CORRECTION_CONTRACT,
    OfficialCODIPerturbAndCapture,
    corrected_states,
    fit_ridge_correction,
    paired_question_examples,
)


HIDDEN = 768


class _FakeBlock(torch.nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _FakeLayerNorm(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _FakeCodi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList(_FakeBlock() for _ in range(12))
        self.transformer.ln_f = _FakeLayerNorm()


class _FakeModel:
    def __init__(self):
        self.codi = _FakeCodi()


def test_perturbation_is_seeded_and_capture_observes_perturbed_path():
    hidden = torch.ones(3, 2, HIDDEN)

    def run(seed):
        model = _FakeModel()
        collector = OfficialCODIPerturbAndCapture(
            model, relative_noise=0.2, seed=seed
        )
        with collector.activate(torch.ones(3, dtype=torch.bool)):
            state11 = model.codi.transformer.h[10](hidden)[0]
            model.codi.transformer.ln_f(state11)
        return state11, collector.stacked(3)

    state11_a, state12_a = run(5)
    state11_b, state12_b = run(5)
    assert torch.equal(state11_a, state11_b)
    assert torch.equal(state12_a, state12_b)
    assert torch.equal(state12_a, state11_a[:, -1, :])
    assert not torch.equal(state11_a[:, -1, :], hidden[:, -1, :])


def test_paired_examples_require_both_outcomes_and_weight_questions_once():
    states = torch.zeros(3, 4, HIDDEN, dtype=torch.float64)
    states[0, :, 0] = torch.tensor([0.0, 2.0, 0.0, 2.0])
    states[1, :, 0] = 1.0
    states[2, :, 0] = torch.tensor([-1.0, -1.0, 3.0, 3.0])
    correct = torch.tensor(
        [[False, True, False, True], [True, True, True, True], [False, False, True, True]]
    )
    basis = torch.eye(HIDDEN, dtype=torch.float64)[:, :2]
    readout = torch.zeros(3, HIDDEN, dtype=torch.float64)
    readout[1, 0] = 1
    examples = paired_question_examples(
        states, correct, basis, torch.zeros(HIDDEN), readout
    )
    assert examples["features"].shape == (2, 3)
    assert examples["targets"].shape == (2, 2)
    assert examples["question_rows"].tolist() == [0, 2]
    assert torch.allclose(
        examples["targets"][:, 0], torch.tensor([2.0, 4.0], dtype=torch.float64)
    )


def test_ridge_correction_recovers_a_conditioned_linear_map():
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(500, 6, generator=generator, dtype=torch.float64)
    true_weight = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    true_bias = torch.randn(3, generator=generator, dtype=torch.float64)
    targets = features @ true_weight + true_bias
    model = fit_ridge_correction(features, targets, ridge=1e-6)
    prediction = model.predict(features)
    assert float((prediction - targets).square().mean()) < 1e-10


def test_additive_correction_preserves_everything_outside_the_band():
    generator = torch.Generator().manual_seed(8)
    states = torch.randn(5, HIDDEN, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(
        torch.randn(HIDDEN, 4, generator=generator, dtype=torch.float64)
    )
    coefficients = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    margins = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0])
    edited, gate = corrected_states(
        states,
        basis=basis,
        predicted_coefficients=coefficients,
        alpha=0.5,
        margin=margins,
        maximum_margin=1.0,
    )
    complement = torch.eye(HIDDEN, dtype=torch.float64) - basis @ basis.T
    assert gate.tolist() == [True, True, True, False, False]
    assert torch.allclose(
        (edited - states) @ complement,
        torch.zeros(5, HIDDEN, dtype=torch.float64),
        atol=1e-10,
    )


def test_analysis_requires_gain_and_specificity_controls():
    count = 300
    baseline = np.zeros(count, dtype=bool)
    baseline[:120] = True
    conditioned = baseline.copy()
    conditioned[120:180] = True
    global_mean = baseline.copy()
    global_mean[120:140] = True
    shuffled = baseline.copy()
    shuffled[120:135] = True

    def entry(values):
        return {
            "correct": torch.as_tensor(values),
            "nll": torch.zeros(count),
            "margin": torch.zeros(count),
            "gate": torch.ones(count, dtype=torch.bool),
        }

    artifact = {
        "contract": PAIRED_CORRECTION_CONTRACT,
        "partition_sha256": "same",
        "outcomes": {
            "baseline": entry(baseline),
            "conditioned": entry(conditioned),
            "global_mean": entry(global_mean),
            "shuffled_target": entry(shuffled),
        },
    }
    summary = {
        "contract": PAIRED_CORRECTION_CONTRACT,
        "partition_sha256": "same",
        "splits": {"fit_paired_questions": 100, "select_paired_questions": 90},
        "selected_ridge": 1.0,
        "selected_interventions": {
            "conditioned": {"alpha": 1.0, "edited_fraction": 0.5}
        },
        "test_arms": {},
    }
    report = analyze_paired_correction(
        summary,
        artifact,
        minimum_gain_points=1.0,
        bootstrap_samples=1000,
        bootstrap_seed=1,
        alpha=0.05,
    )
    assert report["paired_correction_confirmed"]


def test_zero_noise_is_valid_but_excess_noise_is_rejected():
    OfficialCODIPerturbAndCapture(_FakeModel(), relative_noise=0.0, seed=0)
    with pytest.raises(ValueError):
        OfficialCODIPerturbAndCapture(_FakeModel(), relative_noise=2.1, seed=0)


def test_paired_runner_writes_frozen_test_outcomes(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(13)
    fit_count, select_count, test_count, variants, vocabulary = 30, 20, 10, 4, 17
    paired_count = fit_count + select_count
    paired_states = torch.randn(
        paired_count, variants, HIDDEN, generator=generator
    )
    paired_correct = torch.zeros(paired_count, variants, dtype=torch.bool)
    paired_correct[:, 1::2] = True
    readout = torch.randn(vocabulary, HIDDEN, generator=generator)
    gold = torch.randint(0, vocabulary, (paired_count,), generator=generator)
    test_states = torch.randn(test_count, HIDDEN, generator=generator)
    test_gold = torch.randint(0, vocabulary, (test_count,), generator=generator)
    pairs = {
        "contract": PAIRED_CORRECTION_CONTRACT,
        "request_sha256": "pairs",
        "source_request_sha256": "source",
        "partition_sha256": "partition",
        "indices": {
            "fit": list(range(fit_count)),
            "select": list(range(fit_count, paired_count)),
            "test": list(range(paired_count, paired_count + test_count)),
        },
        "fit_select_states": paired_states,
        "fit_select_correct": paired_correct,
        "fit_select_gold_first_token": gold,
        "test_baseline_states": test_states,
        "test_gold_first_token": test_gold,
    }
    pairs_path = tmp_path / "pairs.pt"
    readout_path = tmp_path / "readout.pt"
    output_path = tmp_path / "summary.json"
    artifact_path = tmp_path / "artifact.pt"
    torch.save(pairs, pairs_path)
    torch.save({"readout": readout, "request_sha256": "source"}, readout_path)
    settings = SimpleNamespace(
        fit_examples=fit_count,
        select_examples=select_count,
        accuracy_band=[1, 3],
        ridge_grid=[0.1, 1.0],
        shuffle_seed=4,
        alpha_grid=[0.0, 0.5],
        gate_fractions=[0.0, 0.5],
    )
    monkeypatch.setattr(
        paired_runner,
        "load_config",
        lambda *_: SimpleNamespace(endpoint_paired_correction=settings),
    )
    assert paired_runner.main(
        [
            "--pairs",
            str(pairs_path),
            "--readout",
            str(readout_path),
            "--output",
            str(output_path),
            "--artifact-output",
            str(artifact_path),
        ]
    ) == 0
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert artifact["basis"].shape == (HIDDEN, 2)
    assert artifact["outcomes"]["conditioned"]["correct"].shape == (test_count,)


def test_kaggle_notebook_orders_repair_collection_and_generation():
    repository = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (repository / "notebooks" / "kaggle_official_codi_paired_correction.ipynb")
        .read_text(encoding="utf-8")
    )
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    pin = next(
        i for i, value in enumerate(sources) if '"transformers": "4.52.4"' in value
    )
    repair = next(
        i
        for i, value in enumerate(sources)
        if '"torchao"' in value and "uninstall" in value
    )
    collect = next(
        i
        for i, value in enumerate(sources)
        if "collect_official_codi_paired_counterfactuals.py" in value
    )
    generate = next(
        i
        for i, value in enumerate(sources)
        if "run_official_codi_paired_correction_generation.py" in value
    )
    assert pin < repair < collect < generate
