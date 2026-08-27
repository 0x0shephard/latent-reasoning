import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.run_official_codi_correctness_contrastive_covariance as runner
from scripts.run_official_codi_correctness_contrastive_covariance import evaluate_arm
from src.eval.official_codi_correctness_contrastive_covariance_analysis import (
    analyze_contrastive_covariance,
)
from src.mech.endpoint_correctness_contrastive_covariance import (
    CONTRASTIVE_COVARIANCE_CONTRACT,
    fit_contrastive_covariance,
    heldout_specificity_score,
    project_states,
    shrink_covariance,
)


def test_contrastive_fit_recovers_planted_class_specific_axes():
    generator = torch.Generator().manual_seed(9)
    count, dimension = 180, 768
    right = torch.randn(count, dimension, generator=generator) * 0.15
    wrong = torch.randn(count, dimension, generator=generator) * 0.15
    right[:, :2] *= 15
    wrong[:, 2:4] *= 15
    states = torch.cat([right, wrong])
    labels = torch.cat(
        [torch.ones(count, dtype=torch.bool), torch.zeros(count, dtype=torch.bool)]
    )
    fitted = fit_contrastive_covariance(states, labels, rank=2, shrinkage=0.1)
    correct_overlap = float((fitted.correct_basis[:2] ** 2).sum() / 2)
    wrong_overlap = float((fitted.wrong_basis[2:4] ** 2).sum() / 2)
    assert correct_overlap > 0.75
    assert wrong_overlap > 0.75
    assert torch.allclose(
        fitted.correct_basis.T @ fitted.correct_basis,
        torch.eye(2, dtype=torch.float64),
        atol=1e-8,
    )


def test_projection_and_low_rank_logits_equal_dense_edit():
    generator = torch.Generator().manual_seed(3)
    states = torch.randn(7, 768, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(
        torch.randn(768, 4, generator=generator, dtype=torch.float64)
    )
    centre = torch.randn(768, generator=generator, dtype=torch.float64)
    readout = torch.randn(31, 768, generator=generator, dtype=torch.float64)
    gold = torch.randint(0, 31, (7,), generator=generator)
    for mode in ("retain", "remove"):
        fast = evaluate_arm(
            states, readout, gold, basis=basis, centre=centre, mode=mode, chunk_size=3
        )
        dense = project_states(states, basis, centre, mode=mode) @ readout.T
        dense_correct = dense.argmax(1) == gold
        assert torch.equal(fast["correct"], dense_correct)
        rows = torch.arange(7)
        dense_nll = torch.logsumexp(dense, 1) - dense[rows, gold]
        assert torch.allclose(fast["nll"], dense_nll, atol=1e-10)


def test_selection_score_prefers_matching_orientation():
    generator = torch.Generator().manual_seed(4)
    right = torch.randn(60, 768, generator=generator)
    wrong = torch.randn(60, 768, generator=generator)
    right[:, :3] *= 4
    states = torch.cat([right, wrong])
    labels = torch.cat([torch.ones(60, dtype=torch.bool), torch.zeros(60, dtype=torch.bool)])
    basis = torch.eye(768, dtype=torch.float64)[:, :3]
    score = heldout_specificity_score(
        states,
        labels,
        basis,
        correct_mean=right.mean(0),
        wrong_mean=wrong.mean(0),
        orientation="correct",
    )
    assert score > 1.0
    with pytest.raises(ValueError):
        shrink_covariance(torch.eye(3), 0.0)


def test_analysis_requires_both_pca_and_random_controls():
    count = 200
    baseline = np.zeros(count, dtype=bool)
    baseline[:80] = True
    primary = baseline.copy()
    primary[80:120] = True
    correct_pca = baseline.copy()
    correct_pca[80:100] = True
    classblind = baseline.copy()
    classblind[80:96] = True
    random = baseline.copy()
    random[80:92] = True
    wrong_remove = baseline.copy()
    wrong_remove[80:110] = True

    def entry(values):
        tensor = torch.as_tensor(values)
        return {
            "correct": tensor,
            "nll": torch.zeros(count),
            "margin": torch.zeros(count),
        }

    artifact = {
        "contract": CONTRASTIVE_COVARIANCE_CONTRACT,
        "partition_sha256": "partition",
        "outcomes": {
            "baseline": entry(baseline),
            "contrastive_correct_retain": entry(primary),
            "correct_only_pca_retain": entry(correct_pca),
            "accuracy_band_pca_retain": entry(classblind),
            "contrastive_wrong_remove": entry(wrong_remove),
            "random_correct_energy_retain_r00": entry(random),
            "random_wrong_energy_remove_r00": entry(random),
        },
    }
    summary = {
        "contract": CONTRASTIVE_COVARIANCE_CONTRACT,
        "rank": 28,
        "selected_shrinkage": 0.1,
        "splits": {"partition_sha256": "partition"},
        "arms": {},
    }
    report = analyze_contrastive_covariance(
        summary,
        artifact,
        bootstrap_samples=1000,
        minimum_advantage_points=1.0,
        minimum_wrong_removal_gain_points=1.0,
    )
    assert report["contrastive_covariance_confirmed"]
    assert report["comparisons"]["versus_correct_only_pca"]["passed"]


def test_runner_writes_paired_synthetic_artifact(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(14)
    count, vocabulary = 84, 13
    states = torch.randn(count, 768, generator=generator)
    readout = torch.randn(vocabulary, 768, generator=generator)
    predictions = (states.double() @ readout.double().T).argmax(1)
    gold = predictions.clone()
    gold[1::2] = (gold[1::2] + 1) % vocabulary
    cache = {
        "state_order": [12],
        "evaluation_states": states.unsqueeze(1),
        "evaluation_gold_first_token": gold,
        "request_sha256": "source",
    }
    settings = SimpleNamespace(
        expected_examples=count,
        fit_examples=32,
        select_examples=24,
        split_seed=17,
        rank=2,
        accuracy_band=[1, 3],
        shrinkage_grid=[0.2],
        random_seed=18,
        random_replicates=1,
    )
    monkeypatch.setattr(runner, "load_margin_cache", lambda *_: (cache, {"readout": readout}))
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda *_: SimpleNamespace(
            endpoint_correctness_contrastive_covariance=settings
        ),
    )
    summary_path = tmp_path / "summary.json"
    artifact_path = tmp_path / "artifact.pt"
    assert runner.main(
        [
            "--states",
            str(tmp_path / "unused_states.pt"),
            "--readout",
            str(tmp_path / "unused_readout.pt"),
            "--output",
            str(summary_path),
            "--artifact-output",
            str(artifact_path),
        ]
    ) == 0
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert len(artifact["indices"]["test"]) == 28
    assert artifact["arms"]["contrastive_correct_retain"]["basis"].shape == (768, 2)
    assert artifact["outcomes"]["baseline"]["correct"].shape == (28,)


def test_kaggle_notebook_repairs_environment_before_generation():
    repository = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (
            repository
            / "notebooks"
            / "kaggle_official_codi_correctness_contrastive_covariance.ipynb"
        )
        .read_text(encoding="utf-8")
    )
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    pin_index = next(
        index
        for index, source in enumerate(sources)
        if '"transformers": "4.52.4"' in source
    )
    repair_index = next(
        index
        for index, source in enumerate(sources)
        if '"pip", "uninstall", "-y", "torchao"' in source
    )
    generation_index = next(
        index
        for index, source in enumerate(sources)
        if "GENERATION_ARMS = [" in source
    )
    assert pin_index < repair_index < generation_index
