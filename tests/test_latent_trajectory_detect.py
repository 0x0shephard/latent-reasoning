from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import scripts.run_official_codi_latent_trajectory_detect as trajectory_runner

from src.eval.official_codi_latent_trajectory_detect_analysis import (
    analyze_latent_trajectory_detect,
)
from src.mech.latent_trajectory_detect import (
    LATENT_TRAJECTORY_CONTRACT,
    TRAJECTORY_STATES,
    OfficialCODILatentTrajectoryCapture,
    fit_one_hot_ridge,
    multiclass_ridge_from_state_dict,
)


HIDDEN = 768


class _FakeBlock(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, hidden):
        return (hidden + self.offset,)


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList(_FakeBlock(1.0) for _ in range(12))
        self.ln_f = torch.nn.LayerNorm(HIDDEN)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.codi = torch.nn.Module()
        self.codi.transformer = _FakeTransformer()

    def run_pass(self, hidden: torch.Tensor) -> torch.Tensor:
        parts = self.codi.transformer
        for block in parts.h:
            hidden = block(hidden)[0]
        return parts.ln_f(hidden)


def test_trajectory_capture_commits_only_latent_passes():
    torch.manual_seed(0)
    model = _FakeModel()
    capture = OfficialCODILatentTrajectoryCapture(model, latent_iterations=3)
    try:
        # A prompt-like pass is buffered but never committed.
        model.run_pass(torch.randn(2, 5, HIDDEN))
        latent_inputs = [torch.randn(2, 1, HIDDEN) for _ in range(3)]
        for position, latent in enumerate(latent_inputs):
            model.run_pass(latent)
            assert capture({"cache": position}, position) == {"cache": position}
        # A forced-cue-like pass afterwards is buffered but never committed.
        model.run_pass(torch.randn(2, 5, HIDDEN))
    finally:
        capture.close()
    stacked = capture.stacked(2)
    assert stacked.shape == (2, 3, TRAJECTORY_STATES, HIDDEN)
    for position, latent in enumerate(latent_inputs):
        # State 0 is the block-0 input, state k is after k fake blocks (+k),
        # and state 12 is the layer-normalised output.
        assert torch.allclose(stacked[:, position, 0, :], latent[:, -1, :])
        assert torch.allclose(
            stacked[:, position, 7, :], latent[:, -1, :] + 7.0, atol=1e-6
        )
        expected_final = model.codi.transformer.ln_f(latent[:, -1, :] + 12.0)
        assert torch.allclose(
            stacked[:, position, 12, :], expected_final.float(), atol=1e-5
        )


def test_trajectory_capture_rejects_partial_pass_and_bad_position():
    model = _FakeModel()
    capture = OfficialCODILatentTrajectoryCapture(model, latent_iterations=2)
    try:
        with pytest.raises(RuntimeError):
            capture(None, 0)  # nothing buffered yet
        model.run_pass(torch.randn(1, 1, HIDDEN))
        with pytest.raises(ValueError):
            capture(None, 5)
    finally:
        capture.close()


def test_one_hot_ridge_recovers_separable_classes():
    generator = torch.Generator().manual_seed(7)
    classes = torch.tensor([11, 23, 40])
    labels = classes[torch.randint(0, 3, (240,), generator=generator)]
    prototypes = torch.randn(3, HIDDEN, generator=generator) * 4
    features = prototypes[
        (labels.unsqueeze(1) == classes.unsqueeze(0)).float().argmax(1)
    ] + 0.1 * torch.randn(240, HIDDEN, generator=generator)
    probe = fit_one_hot_ridge(features[:200], labels[:200], ridge=0.01)
    held_out = probe.predict(features[200:])
    assert float((held_out == labels[200:]).double().mean()) > 0.95
    rebuilt = multiclass_ridge_from_state_dict(probe.state_dict())
    assert torch.equal(rebuilt.predict(features[200:]), held_out)
    with pytest.raises(ValueError):
        fit_one_hot_ridge(features[:200], labels[:200], ridge=0.0)


def _analysis_fixture(*, identity_hits: float, delta_scores: float):
    count = 300
    generator = torch.Generator().manual_seed(3)
    correct = torch.zeros(count, dtype=torch.bool)
    correct[:130] = True
    margin_scores = correct.double() + 0.5 * torch.randn(count, generator=generator)
    trajectory_scores = margin_scores + delta_scores * correct.double()
    gold = torch.randint(100, 110, (count,), generator=generator)
    wrong = ~correct
    trajectory_predictions = gold.clone()
    miss = torch.rand(count, generator=generator) > identity_hits
    trajectory_predictions[miss] = 999
    endpoint_predictions = torch.full_like(gold, 999)
    majority_predictions = torch.full_like(gold, 105)
    artifact = {
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "partition_sha256": "same",
        "test_correct": correct,
        "test_wrong_mask": wrong,
        "test_gold_first_token": gold,
        "correctness_scores": {
            "trajectory_plus_margin": trajectory_scores,
            "margin": margin_scores,
        },
        "identity_predictions": {
            "trajectory": trajectory_predictions,
            "endpoint": endpoint_predictions,
            "majority": majority_predictions,
        },
    }
    optimization = {"converged": True}
    summary = {
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "partition_sha256": "same",
        "splits": {"fit": 100, "select": 100, "test": count},
        "parity_gate": {"passed": True},
        "correctness": {
            "selected": {
                "cell": "position_3_state_08",
                "test_auc": 0.9,
                "optimization": optimization,
            },
            "margin_baseline": {"test_auc": 0.8, "optimization": optimization},
            "selection_curve": [],
        },
        "answer_identity": {
            "selected": {"cell": "position_3_state_08"},
            "unseen_gold_class_fraction": 0.0,
            "selection_curve": [],
        },
    }
    settings = SimpleNamespace(
        bootstrap_samples=2000,
        bootstrap_seed=5,
        alpha=0.05,
        minimum_delta_auc=0.02,
        minimum_identity_gain_points=5.0,
    )
    return summary, artifact, settings


def test_analysis_gates_pass_and_fail_branches():
    summary, artifact, settings = _analysis_fixture(
        identity_hits=0.6, delta_scores=1.5
    )
    report = analyze_latent_trajectory_detect(summary, artifact, settings)
    assert report["answer_identity_gate"]["passed"]
    assert report["correctness_gate"]["passed"]
    assert report["editing_experiment_justified"]
    assert report["status"] == "latent_trajectory_both_supported"

    summary, artifact, settings = _analysis_fixture(
        identity_hits=0.0, delta_scores=0.0
    )
    artifact["identity_predictions"]["trajectory"] = artifact[
        "identity_predictions"
    ]["endpoint"].clone()
    report = analyze_latent_trajectory_detect(summary, artifact, settings)
    assert not report["answer_identity_gate"]["passed"]
    assert not report["correctness_gate"]["passed"]
    assert not report["editing_experiment_justified"]
    assert report["status"] == "latent_trajectory_not_supported"


def test_analysis_refuses_foreign_contract():
    summary, artifact, settings = _analysis_fixture(
        identity_hits=0.6, delta_scores=1.5
    )
    artifact["contract"] = "something_else"
    with pytest.raises(RuntimeError):
        analyze_latent_trajectory_detect(summary, artifact, settings)


def _runner_settings():
    return SimpleNamespace(
        expected_examples=360,
        fit_examples=144,
        select_examples=108,
        split_seed=20260827,
        expected_partition_sha256="unused-by-runner",
        parity_relative_tolerance=0.001,
        correctness_ridge_grid=[1.0],
        identity_ridge_grid=[0.01, 1.0],
        solver_max_iterations=300,
        solver_gradient_tolerance=1e-7,
        solver_objective_gap_tolerance=1e-8,
        minimum_delta_auc=0.02,
        minimum_identity_gain_points=5.0,
        bootstrap_samples=1000,
        bootstrap_seed=9,
        alpha=0.05,
    )


def test_runner_end_to_end_finds_planted_cell(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(11)
    count, positions, vocabulary = 360, 2, 60
    readout = torch.randn(vocabulary, HIDDEN, generator=generator)
    gold = torch.randint(0, 8, (count,), generator=generator)
    correct = torch.rand(count, generator=generator) < 0.4
    # The colon state points at the gold row when correct and at a fixed
    # distractor row otherwise, so first-token correctness matches `correct`.
    target = torch.where(correct, gold, torch.full_like(gold, 40))
    colon_states = readout[target] * 3 + 0.05 * torch.randn(
        count, HIDDEN, generator=generator
    )
    # Positive per-row rescaling keeps the argmax (so correctness labels are
    # unchanged) while making the top-two margin uninformative about them.
    scales = 1.0 + 2.0 * torch.rand(count, generator=generator)
    colon_states = colon_states * scales.unsqueeze(1)
    trajectory = 0.1 * torch.randn(
        count, positions, TRAJECTORY_STATES, HIDDEN, generator=generator
    )
    # Planted cell (1, 6): encodes the gold class and the correctness label.
    # The correctness signal is spread over many coordinates because per-feature
    # standardization caps any single binary coordinate at ~2 sigma separation.
    trajectory[:, 1, 6, :8] += torch.nn.functional.one_hot(gold, 8).double() * 10
    trajectory[:, 1, 6, 20:120] += correct.double().unsqueeze(1) * 2

    from scripts.run_official_codi_correctness_detect_replication import (
        split_test_like,
    )

    fit_index, select_index, test_index = split_test_like(
        count, seed=20260827, fit_size=144, select_size=108
    )
    indices = {
        "fit": fit_index.tolist(),
        "select": select_index.tolist(),
        "test": test_index.tolist(),
    }
    export = {
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "request_sha256": "traj",
        "source_request_sha256": "cache",
        "partition_sha256": "part",
        "indices": indices,
        "parity_gate": {"passed": True},
        "trajectory_states": trajectory.float(),
    }
    trajectory_path = tmp_path / "latent_trajectory.pt"
    torch.save(export, trajectory_path)

    cache = {
        "request_sha256": "cache",
        "state_order": list(range(13)),
        "evaluation_states": torch.stack(
            [colon_states.float()] * 13, dim=1
        ),
        "evaluation_gold_first_token": gold,
    }
    readout_payload = {"readout": readout.float()}
    monkeypatch.setattr(
        trajectory_runner, "load_margin_cache", lambda a, b: (cache, readout_payload)
    )
    monkeypatch.setattr(
        trajectory_runner,
        "load_config",
        lambda path: SimpleNamespace(latent_trajectory_detect=_runner_settings()),
    )
    output = tmp_path / "latent_trajectory_detect.json"
    assert (
        trajectory_runner.main(
            [
                "--trajectory",
                str(trajectory_path),
                "--states",
                str(tmp_path / "unused_states.pt"),
                "--readout",
                str(tmp_path / "unused_readout.pt"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    import json

    summary = json.loads(output.read_text())
    report = json.loads(
        (tmp_path / "latent_trajectory_detect_report.json").read_text()
    )
    assert summary["correctness"]["selected"]["cell"] == "position_1_state_06"
    assert summary["answer_identity"]["selected"]["cell"] == "position_1_state_06"
    assert report["answer_identity_gate"]["passed"]
    assert report["correctness_gate"]["passed"]
    assert report["status"] == "latent_trajectory_both_supported"


def test_runner_refuses_mismatched_lineage(tmp_path, monkeypatch):
    export = {
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "source_request_sha256": "other-cache",
        "parity_gate": {"passed": True},
        "trajectory_states": torch.zeros(4, 1, TRAJECTORY_STATES, HIDDEN),
        "partition_sha256": "part",
        "indices": {"fit": [0], "select": [1], "test": [2, 3]},
        "request_sha256": "traj",
    }
    path = tmp_path / "latent_trajectory.pt"
    torch.save(export, path)
    with pytest.raises(RuntimeError):
        trajectory_runner.load_trajectory_export(path, {"request_sha256": "cache"})
