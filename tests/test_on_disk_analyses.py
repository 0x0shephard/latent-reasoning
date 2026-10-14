"""Unit tests for the §133 analyses (pure helpers; no model, no files)."""
import numpy as np
import pytest
import torch

from scripts.analyze_cascade_combined_gate import oof_wrong_scores
from scripts.analyze_cross_model_failures import analyze as cross_analyze, checks_from as cross_checks, independence_jaccard, jaccard
from scripts.analyze_repair_predictors import auroc, by_bins, transported_controls
from scripts.analyze_unfaithful_slot_content import first_operator, hashed_bow, penultimate, question_numbers, ridge_balanced_accuracy, topk_classes


def test_jaccard_and_independence():
    a = np.array([1, 1, 0, 0], bool); b = np.array([1, 0, 1, 0], bool)
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert independence_jaccard(0.5, 0.5) == pytest.approx(0.25 / 0.75)


def test_cross_model_analysis_counts_who_solves_whose_failures():
    q = ["q0", "q1", "q2", "q3"]
    steps = np.array([3, 3, 3, 1])
    readouts = {"a": {"correct": np.array([0, 0, 1, 1], bool), "steps": steps, "prob": np.zeros(4), "margin": np.zeros(4)},
                "b": {"correct": np.array([1, 0, 1, 0], bool), "steps": steps, "prob": np.zeros(4), "margin": np.zeros(4)},
                "c": {"correct": np.array([0, 1, 1, 1], bool), "steps": steps, "prob": np.zeros(4), "margin": np.zeros(4)}}
    classes = {"a": {"q0": "latent_only", "q1": "plan_wrong"}, "b": {"q1": "plan_wrong"}, "c": {"q0": "latent_only"}}
    r = cross_analyze(q, readouts, classes)
    assert r["rows"] == 3 and r["solved_by_others"]["a"]["latent_only"] == {"n": 1, "b": 1.0, "c": 0.0, "own_explicit": 1.0}
    assert r["overlap"]["a|b"]["shared_failures"] == 1 and r["vote"]["all_models_wrong"] == 0
    assert set(cross_checks(r)) == {"p19a_latent_only_solved_by_others_at_or_above_base", "p19a_plan_wrong_solved_by_others_le_0_35",
                                    "p19b_codi1b_simcot_jaccard_exceeds_independence_by_0_10", "p19c_majority_vote_gain_le_2_points"}


def test_auroc_and_bins():
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([True, True, False, False])) == 1.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([True, False, True, False])) == pytest.approx(0.5)
    assert auroc(np.array([1.0]), np.array([True])) is None
    bins = by_bins(np.array([0.1, 0.9, 0.5, 0.3]), np.array([True, False, False, True]), 2)
    assert [b["n"] for b in bins] == [2, 2] and bins[0]["rate"] == 1.0 and bins[1]["rate"] == 0.0


def test_transported_controls_reads_interleaved_rows():
    pairs = [{"question": "A", "twin_question": "B"}]
    preds = {"target_gold": np.array([7, 9]), "state_3": np.array([7, 1]), "state_1": np.array([0, 9])}
    assert transported_controls(pairs, preds, [3]) == {"A": True, "B": False}
    assert transported_controls(pairs, preds, [1, 3]) == {"A": True, "B": True}


def test_question_attributes():
    assert question_numbers("Buy 3 pens at $1,250 each and 0.5 kg") == ["3", "1250", "0.5"]
    assert first_operator("<<16-3-4=9>> <<9*2=18>>") == "-" and first_operator(None) is None
    assert penultimate("<<16-3-4=9>> <<9*2=18>>") == "9" and penultimate("<<2*2=4>>") is None
    labels, keep = topk_classes(["a", "b", "a", None, "c"], k=2)
    assert keep == ["a", "b"] and labels.tolist() == [0, 1, 0, -1, -1]
    assert hashed_bow(["one two", "three"]).shape == (2, 4096)


def test_ridge_probe_separates_a_linear_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300); X = rng.normal(size=(300, 20)); X[:, 0] += 3 * y
    assert ridge_balanced_accuracy(X, y) > 0.8   # Bayes limit for 3-sigma spaced classes is about 0.91; the one-hot ridge scored 0.71 here
    assert ridge_balanced_accuracy(X, np.full(300, -1)) is None


def test_oof_scores_rank_wrong_answers_by_low_confidence():
    rng = np.random.default_rng(0); n = 200
    wrong = rng.random(n) < 0.5; prob = np.where(wrong, 0.2, 0.8) + rng.normal(scale=0.05, size=n)
    states = torch.zeros(n, 6, 8)
    s = oof_wrong_scores(states, prob, prob, wrong, use_states=False)
    assert s.shape == (n,) and auroc(s, wrong) > 0.95
    s2 = oof_wrong_scores(torch.randn(n, 6, 8), prob, prob, wrong, use_states=True)
    assert s2.shape == (n,)
