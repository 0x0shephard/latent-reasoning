"""Tests for the correctness-geometry tracks.

The pattern that burned the last experiment was a refactor leaving a name
undefined on a path no test exercised, discovered only after a Kaggle run had
been queued.  So the runner and the gates are exercised end to end on synthetic
data here, not just the mathematical helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.official_codi_correctness_tracks_analysis import (
    analyze_correctness_tracks,
)
from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE
from src.mech.endpoint_correctness_geometry import (
    ACCURACY_BAND,
    LIFT_BAND,
    OfficialCODIEndpointSteerIntervention,
    answer_margin,
    apply_logistic,
    band_projector,
    band_variance_shares,
    class_conditional_basis,
    direction_band_profile,
    first_token_correct,
    fit_correctness_directions,
    fit_logistic,
    margin_gradient,
    principal_angle_cosines,
    random_split_null,
    retained_accuracy,
    retention,
    roc_auc,
    sorted_eigenbasis,
    steer,
    steered_accuracy,
)


@pytest.fixture(scope="module")
def toy():
    """A small readout and a state cloud with a genuine correctness direction."""
    generator = torch.Generator().manual_seed(7)
    vocabulary = 64
    readout = torch.randn(vocabulary, GPT2_HIDDEN_SIZE, generator=generator).double()
    states = torch.randn(1600, GPT2_HIDDEN_SIZE, generator=generator).double()
    # Inflate a handful of directions so the eigen-bands are well separated.
    states[:, :4] *= 12.0
    states[:, 4:32] *= 3.0
    gold = (states @ readout.T).argmax(dim=-1)
    # Corrupt a third of the labels so both classes are populated.
    flip = torch.arange(states.shape[0]) % 3 == 0
    gold[flip] = (gold[flip] + 1) % vocabulary
    return {"states": states, "readout": readout, "gold": gold}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_first_token_correct_matches_a_direct_argmax(toy):
    outcomes = first_token_correct(toy["states"], toy["readout"], toy["gold"])
    expected = (toy["states"] @ toy["readout"].T).argmax(dim=-1) == toy["gold"]
    assert torch.equal(outcomes, expected)
    assert 0.0 < float(outcomes.double().mean()) < 1.0


def test_first_token_correct_rejects_mismatched_shapes(toy):
    with pytest.raises(ValueError):
        first_token_correct(toy["states"][:, :10], toy["readout"], toy["gold"])
    with pytest.raises(ValueError):
        first_token_correct(toy["states"], toy["readout"], toy["gold"][:5])


def test_answer_margin_is_non_negative_and_matches_the_top_two_gap(toy):
    margin = answer_margin(toy["states"], toy["readout"])
    logits = toy["states"] @ toy["readout"].T
    ordered = logits.sort(dim=-1, descending=True).values
    assert torch.allclose(margin, ordered[:, 0] - ordered[:, 1])
    assert float(margin.min()) >= 0.0


def test_roc_auc_is_half_for_a_constant_score_and_one_for_a_perfect_one():
    labels = torch.tensor([1, 0, 1, 0, 1, 0])
    # Every score tied: the answer must be exactly chance, not whatever order the
    # questions arrived in. A position-breaking argsort returns 0.333 here.
    assert roc_auc(torch.zeros(6), labels) == pytest.approx(0.5)
    assert roc_auc(torch.full((6,), 3.5), labels) == pytest.approx(0.5)
    assert roc_auc(labels.double(), labels) == pytest.approx(1.0)
    # Flipping the sign of a score reflects the AUC about one half.
    scores = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    assert roc_auc(-scores, labels) == pytest.approx(1.0 - roc_auc(scores, labels))


def test_roc_auc_refuses_a_single_class():
    with pytest.raises(ValueError):
        roc_auc(torch.randn(8), torch.ones(8))


def test_band_projector_is_idempotent_and_symmetric(toy):
    _, vectors = sorted_eigenbasis(torch.cov(toy["states"].T))
    projector = band_projector(vectors, *ACCURACY_BAND)
    assert torch.allclose(projector, projector.T, atol=1e-10)
    assert torch.allclose(projector @ projector, projector, atol=1e-9)
    # A projector's trace is the dimension of the subspace it projects onto.
    assert float(torch.diagonal(projector).sum()) == pytest.approx(
        ACCURACY_BAND[1] - ACCURACY_BAND[0], abs=1e-6
    )


def test_band_projector_rejects_out_of_range_bands(toy):
    _, vectors = sorted_eigenbasis(torch.cov(toy["states"].T))
    for start, stop in ((4, 4), (-1, 8), (0, GPT2_HIDDEN_SIZE + 1), (32, 8)):
        with pytest.raises(ValueError):
            band_projector(vectors, start, stop)


# --------------------------------------------------------------------------
# track 1 — detect
# --------------------------------------------------------------------------


def test_correctness_directions_recover_a_planted_separation():
    # 32 dimensions against 400 examples. At the real 768-against-1024 ratio the
    # within-class covariance is near singular and the Fisher direction is much
    # noisier, which is why the runner selects its shrinkage on a held-out split.
    generator = torch.Generator().manual_seed(11)
    dimension = 32
    states = torch.randn(400, dimension, generator=generator).double()
    correct = torch.zeros(400, dtype=torch.bool)
    correct[:200] = True
    planted = torch.zeros(dimension, dtype=torch.float64)
    planted[17] = 1.0
    states[correct] += 6.0 * planted

    directions = fit_correctness_directions(states, correct)
    assert abs(float(directions.mean_difference @ planted)) > 0.9
    assert abs(float(directions.fisher @ planted)) > 0.9
    assert float(directions.mean_difference.norm()) == pytest.approx(1.0)
    assert 0.0 < directions.between_fraction < 1.0


def test_correctness_directions_validate_their_inputs(toy):
    correct = first_token_correct(toy["states"], toy["readout"], toy["gold"])
    with pytest.raises(ValueError):
        fit_correctness_directions(toy["states"], correct.long())
    with pytest.raises(ValueError):
        fit_correctness_directions(toy["states"], correct, shrinkage=1.0)
    with pytest.raises(ValueError):
        fit_correctness_directions(toy["states"], torch.ones_like(correct))


def test_fisher_differs_from_the_mean_difference_under_anisotropic_noise():
    """The whole reason to compute Fisher: it discounts high-variance directions.

    The planted separation sits in a low-variance direction while a nuisance
    direction has far more spread. The raw mean difference is pulled toward the
    nuisance direction; the Fisher direction should not be.
    """
    generator = torch.Generator().manual_seed(13)
    dimension = 32
    states = torch.randn(600, dimension, generator=generator).double()
    states[:, 0] *= 30.0
    correct = torch.zeros(600, dtype=torch.bool)
    correct[:300] = True
    # A weak separation along a quiet direction, plus a loud nuisance direction
    # the two classes happen to differ on slightly by chance.
    states[correct, 5] += 1.0
    states[correct, 0] += 6.0

    directions = fit_correctness_directions(states, correct, shrinkage=0.01)
    # The mean difference is dominated by the loud direction; Fisher divides it out.
    assert abs(float(directions.mean_difference[0])) > abs(
        float(directions.mean_difference[5])
    )
    assert abs(float(directions.fisher[5])) > abs(float(directions.fisher[0]))


def test_logistic_probe_separates_a_linearly_separable_problem():
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(300, 3, generator=generator).double()
    labels = (features[:, 0] > 0).long()
    weight, bias, stats = fit_logistic(features, labels, l2=1e-4, steps=400)
    scores = apply_logistic(features, weight, bias, stats)
    assert roc_auc(scores, labels) > 0.99


def test_logistic_probe_standardises_with_the_fitting_split_statistics():
    """Held-out features must not be re-standardised against themselves."""
    generator = torch.Generator().manual_seed(19)
    fit_features = torch.randn(200, 2, generator=generator).double()
    labels = (fit_features[:, 0] > 0).long()
    weight, bias, stats = fit_logistic(fit_features, labels, steps=100)
    shifted = fit_features + 100.0
    plain = apply_logistic(fit_features, weight, bias, stats)
    moved = apply_logistic(shifted, weight, bias, stats)
    # A constant shift must move the scores, which it would not if each split were
    # re-centred on its own mean.
    assert not torch.allclose(plain, moved)


def test_logistic_probe_rejects_unpaired_inputs():
    with pytest.raises(ValueError):
        fit_logistic(torch.randn(10, 3), torch.zeros(9))


# --------------------------------------------------------------------------
# track 2 — steer
# --------------------------------------------------------------------------


def test_margin_gradient_is_the_exact_derivative_of_the_margin(toy):
    """Moving along the gradient widens the margin by exactly the step times its norm."""
    states, readout, gold = toy["states"][:16], toy["readout"], toy["gold"][:16]
    gradient = margin_gradient(states, readout, gold)
    logits = states @ readout.T
    rows = torch.arange(16)
    masked = logits.clone()
    masked[rows, gold] = torch.finfo(logits.dtype).min
    runner = masked.argmax(dim=-1)
    before = logits[rows, gold] - logits[rows, runner]

    step = 1e-4
    moved = states + step * gradient
    after_logits = moved @ readout.T
    after = after_logits[rows, gold] - after_logits[rows, runner]
    assert torch.allclose(after - before, step * gradient.square().sum(1), atol=1e-8)


def test_steer_is_a_pure_translation(toy):
    vector = torch.zeros(GPT2_HIDDEN_SIZE, dtype=torch.float64)
    vector[3] = 1.0
    moved = steer(toy["states"], vector, 2.5)
    assert torch.allclose(moved - toy["states"], 2.5 * vector.unsqueeze(0))
    assert torch.equal(steer(toy["states"], vector, 0.0), toy["states"])


def test_uniform_lift_directions_cannot_change_an_argmax():
    """The mechanism behind the negative steering prediction, stated as a test.

    If a direction produces the same logit change for every token, adding any
    multiple of it leaves the argmax untouched no matter how large the step.
    """
    generator = torch.Generator().manual_seed(23)
    readout = torch.randn(50, GPT2_HIDDEN_SIZE, generator=generator).double()
    states = torch.randn(32, GPT2_HIDDEN_SIZE, generator=generator).double()
    # Solve for a direction whose logit change is constant across the vocabulary.
    uniform = torch.linalg.lstsq(readout, torch.ones(50, dtype=torch.float64)).solution
    uniform = uniform / uniform.norm()
    before = (states @ readout.T).argmax(dim=-1)
    for alpha in (1.0, 10.0, 100.0):
        after = (steer(states, uniform, alpha) @ readout.T).argmax(dim=-1)
        assert torch.equal(before, after)


def test_direction_band_profile_is_a_partition_of_unity(toy):
    _, vectors = sorted_eigenbasis(torch.cov(toy["states"].T))
    direction = torch.randn(GPT2_HIDDEN_SIZE, dtype=torch.float64)
    profile = direction_band_profile(
        direction, vectors, ((0, 4), (4, 32), (32, GPT2_HIDDEN_SIZE))
    )
    assert sum(profile.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= share <= 1.0 for share in profile.values())


def test_band_profile_of_an_eigenvector_is_concentrated_in_its_own_band(toy):
    _, vectors = sorted_eigenbasis(torch.cov(toy["states"].T))
    profile = direction_band_profile(vectors[:, 10], vectors, (LIFT_BAND, ACCURACY_BAND))
    assert profile["4:32"] == pytest.approx(1.0, abs=1e-9)
    assert profile["0:4"] == pytest.approx(0.0, abs=1e-9)


def test_random_split_null_reports_the_expected_shape(toy):
    correct = first_token_correct(toy["states"], toy["readout"], toy["gold"])
    _, vectors = sorted_eigenbasis(torch.cov(toy["states"].T))
    null = random_split_null(
        toy["states"], correct, vectors, replicates=8, seed=3
    )
    assert null["replicates"] == 8
    assert len(null["norms"]) == len(null["band_shares"]) == 8
    assert all(0.0 <= share <= 1.0 for share in null["band_shares"])
    # A random split still leans into the high-variance band far beyond the
    # 4/768 that the band's dimension count alone would give -- which is exactly
    # why an observed concentration is only meaningful against this null.
    assert sum(null["band_shares"]) / 8 > 20 * (4 / GPT2_HIDDEN_SIZE)


def test_random_split_null_is_deterministic_given_a_seed(toy):
    correct = first_token_correct(toy["states"], toy["readout"], toy["gold"])
    _, vectors = sorted_eigenbasis(torch.cov(toy["states"].T))
    kwargs = {"replicates": 4, "seed": 5}
    first = random_split_null(toy["states"], correct, vectors, **kwargs)
    second = random_split_null(toy["states"], correct, vectors, **kwargs)
    assert first == second


# --------------------------------------------------------------------------
# track 3 — project
# --------------------------------------------------------------------------


def test_retention_reproduces_the_state_at_full_rank(toy):
    centre = toy["states"].mean(0)
    # QR in float64: a float32 factorisation is only orthonormal to ~1e-7, which
    # shows up as ~1e-5 error once it is applied to states of this magnitude.
    basis = torch.linalg.qr(
        torch.randn(
            GPT2_HIDDEN_SIZE, GPT2_HIDDEN_SIZE,
            generator=torch.Generator().manual_seed(1),
        ).double()
    )[0]
    assert torch.allclose(retention(toy["states"], basis, centre), toy["states"], atol=1e-9)


def test_retention_at_rank_zero_collapses_to_the_centre(toy):
    centre = toy["states"].mean(0)
    empty = torch.zeros(GPT2_HIDDEN_SIZE, 0, dtype=torch.float64)
    kept = retention(toy["states"], empty, centre)
    assert torch.allclose(kept, centre.unsqueeze(0).expand_as(kept))


def test_principal_angles_are_one_for_identical_subspaces_and_zero_for_orthogonal():
    generator = torch.Generator().manual_seed(29)
    basis = torch.linalg.qr(
        torch.randn(GPT2_HIDDEN_SIZE, 16, generator=generator).double()
    )[0]
    same = principal_angle_cosines(basis, basis)
    assert torch.allclose(same, torch.ones(16, dtype=torch.float64), atol=1e-10)

    full = torch.linalg.qr(
        torch.randn(GPT2_HIDDEN_SIZE, 32, generator=generator).double()
    )[0]
    orthogonal = principal_angle_cosines(full[:, :16], full[:, 16:])
    assert float(orthogonal.max()) < 1e-10


def test_principal_angles_reject_mismatched_ambient_spaces():
    with pytest.raises(ValueError):
        principal_angle_cosines(torch.randn(10, 3), torch.randn(12, 3))


def test_class_conditional_basis_uses_only_its_own_class(toy):
    correct = first_token_correct(toy["states"], toy["readout"], toy["gold"])
    basis, centre = class_conditional_basis(toy["states"], correct, 8, on_correct=True)
    assert basis.shape == (GPT2_HIDDEN_SIZE, 8)
    assert torch.allclose(centre, toy["states"][correct].mean(0))
    assert torch.allclose(basis.T @ basis, torch.eye(8, dtype=torch.float64), atol=1e-8)
    # Perturbing an incorrect example must not move the correct-only basis.
    moved = toy["states"].clone()
    moved[~correct] += 5.0
    again, _ = class_conditional_basis(moved, correct, 8, on_correct=True)
    assert torch.allclose(basis.abs(), again.abs(), atol=1e-8)


def test_class_conditional_basis_refuses_a_rank_it_cannot_support(toy):
    correct = torch.zeros(toy["states"].shape[0], dtype=torch.bool)
    correct[:3] = True
    with pytest.raises(ValueError):
        class_conditional_basis(toy["states"], correct, 8, on_correct=True)


def test_band_variance_shares_sum_to_one_over_a_partition(toy):
    values, _ = sorted_eigenbasis(torch.cov(toy["states"].T))
    shares = band_variance_shares(values, ((0, 4), (4, 32), (32, GPT2_HIDDEN_SIZE)))
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    # The cloud was built with the leading directions inflated, so per dimension
    # they dominate. Totals do not order this way: 28 middling directions can
    # out-total 4 large ones, which is the same arithmetic that makes "82% of the
    # variance" and "6% of the accuracy" compatible in the real data.
    assert shares["0:4"] / 4 > shares["4:32"] / 28 > shares["32:768"] / 736


# --------------------------------------------------------------------------
# steering intervention
# --------------------------------------------------------------------------


class _Block(torch.nn.Module):
    def forward(self, hidden):  # pragma: no cover - exercised through hooks
        return hidden


class _FakeModel:
    """Minimal stand-in with the GPT-2 module layout the intervention walks."""

    def __init__(self):
        transformer = torch.nn.Module()
        transformer.h = torch.nn.ModuleList([_Block() for _ in range(12)])
        transformer.ln_f = _Block()
        codi = torch.nn.Module()
        codi.transformer = transformer
        self.codi = codi


def test_steer_intervention_validates_its_vector():
    model = _FakeModel()
    with pytest.raises(ValueError):
        OfficialCODIEndpointSteerIntervention(model, torch.ones(10), alpha=1.0)
    with pytest.raises(ValueError):
        # Not unit length: alpha is the only place scale is allowed to live.
        OfficialCODIEndpointSteerIntervention(
            model, torch.ones(GPT2_HIDDEN_SIZE), alpha=1.0
        )
    unit = torch.zeros(GPT2_HIDDEN_SIZE)
    unit[0] = float("nan")
    with pytest.raises(ValueError):
        OfficialCODIEndpointSteerIntervention(model, unit, alpha=1.0)


def test_steer_intervention_edits_only_masked_rows_and_only_the_last_position():
    model = _FakeModel()
    vector = torch.zeros(GPT2_HIDDEN_SIZE)
    vector[5] = 1.0
    intervention = OfficialCODIEndpointSteerIntervention(model, vector, alpha=3.0)
    hidden = torch.zeros(4, 6, GPT2_HIDDEN_SIZE)
    mask = torch.tensor([True, False, True, False])
    with intervention.activate(mask):
        edited = model.codi.transformer.ln_f(hidden)
    assert edited[0, -1, 5] == pytest.approx(3.0)
    assert edited[2, -1, 5] == pytest.approx(3.0)
    assert edited[1, -1, 5] == pytest.approx(0.0)
    assert float(edited[:, :-1, :].abs().max()) == 0.0
    assert intervention.diagnostics()["rows"] == 2


def test_steer_intervention_is_a_no_op_once_deactivated():
    model = _FakeModel()
    vector = torch.zeros(GPT2_HIDDEN_SIZE)
    vector[1] = 1.0
    intervention = OfficialCODIEndpointSteerIntervention(model, vector, alpha=2.0)
    hidden = torch.zeros(2, 3, GPT2_HIDDEN_SIZE)
    with intervention.activate(torch.tensor([True, True])):
        pass
    assert float(model.codi.transformer.ln_f(hidden).abs().max()) == 0.0


def test_steer_intervention_refuses_nesting_and_a_wrong_shaped_mask():
    model = _FakeModel()
    vector = torch.zeros(GPT2_HIDDEN_SIZE)
    vector[0] = 1.0
    intervention = OfficialCODIEndpointSteerIntervention(model, vector, alpha=1.0)
    with intervention.activate(torch.tensor([True])):
        with pytest.raises(RuntimeError):
            with intervention.activate(torch.tensor([True])):
                pass
    with intervention.activate(torch.tensor([True, True, True])):
        with pytest.raises(ValueError):
            model.codi.transformer.ln_f(torch.zeros(2, 3, GPT2_HIDDEN_SIZE))


def test_steer_intervention_does_not_touch_state_eleven():
    """The additive edit must stay out of the key/value cache, like the band edit."""
    model = _FakeModel()
    vector = torch.zeros(GPT2_HIDDEN_SIZE)
    vector[0] = 1.0
    intervention = OfficialCODIEndpointSteerIntervention(model, vector, alpha=5.0)
    hidden = torch.zeros(2, 3, GPT2_HIDDEN_SIZE)
    with intervention.activate(torch.tensor([True, True])):
        untouched = model.codi.transformer.h[10](hidden)
    assert float(untouched.abs().max()) == 0.0


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


class _Settings:
    bootstrap_samples = 200
    bootstrap_seed = 0
    detect_primary_probe = "fisher_plus_margin"
    minimum_detect_delta_auc = 0.01
    steer_primary_arm = "margin_band"
    minimum_steer_gain_points = 1.0
    project_primary_rank = 28
    minimum_project_advantage_points = 1.0


def _gate_inputs(*, detect_delta: float, steer_gain: int, project_gain: int):
    """Synthetic sweep output with a controllable effect in each track."""
    size = 400
    generator = torch.Generator().manual_seed(31)
    labels = (torch.rand(size, generator=generator) < 0.42).long()
    margin_scores = labels.double() + 0.6 * torch.randn(size, generator=generator).double()
    probe_scores = labels.double() + (0.6 - detect_delta) * torch.randn(
        size, generator=generator
    ).double()

    baseline = torch.zeros(size, dtype=torch.bool)
    baseline[:168] = True
    arm = baseline.clone()
    arm[168 : 168 + steer_gain] = True
    blind = baseline.clone()
    correct_only = blind.clone()
    correct_only[200 : 200 + project_gain] = True

    payload = {
        "contract": "frozen_checkpoint_answer_colon_correctness_tracks_v1",
        "splits": {"fit": 1024, "select": 1024, "test": size},
        "baseline_first_token_accuracy": float(baseline.double().mean()),
        "geometry": {
            "between_class_fraction": 0.04,
            "mean_difference_norm": 26.2,
            "mean_difference_bands": {"0:4": 0.97, "4:32": 0.02, "32:768": 0.01},
            "fisher_bands": {"0:4": 0.30, "4:32": 0.40, "32:768": 0.30},
            "variance_shares": {"0:4": 0.82, "4:32": 0.11, "32:768": 0.07},
            "random_split_null": {
                "replicates": 4,
                "norms": [2.0, 2.2, 2.4, 2.6],
                "band_shares": [0.70, 0.71, 0.69, 0.72],
            },
        },
        "detect": {
            "margin": {"test_auc": 0.874, "select_auc": 0.87},
            "fisher_plus_margin": {"test_auc": 0.874 + detect_delta, "select_auc": 0.88},
        },
        "steer": {
            "margin_band": {
                "test_accuracy": float(arm.double().mean()),
                "selected_alpha": 0.5,
                "band_profile": {"0:4": 0.0, "4:32": 1.0, "32:768": 0.0},
            },
            "random_band_r00": {
                "test_accuracy": float(baseline.double().mean()),
                "selected_alpha": 0.25,
                "band_profile": {"0:4": 0.0, "4:32": 1.0, "32:768": 0.0},
            },
            "random_global_r00": {
                "test_accuracy": float(baseline.double().mean()) - 0.01,
                "selected_alpha": 0.25,
                "band_profile": {"0:4": 0.7, "4:32": 0.2, "32:768": 0.1},
            },
        },
        "project": {
            "28": {
                "class_blind": {"accuracy": float(blind.double().mean())},
                "correct_only": {"accuracy": float(correct_only.double().mean())},
                "incorrect_only": {"accuracy": 0.30},
                "overlap_with_class_blind": {
                    "mean_cosine": 0.992,
                    "minimum_cosine": 0.907,
                },
            },
        },
    }
    outcomes = {
        "labels": labels,
        "baseline": baseline,
        "detect": {"margin": margin_scores, "fisher_plus_margin": probe_scores},
        "steer": {"margin_band": arm},
        "project": {"28": {"class_blind": blind, "correct_only": correct_only}},
    }
    return payload, outcomes


def test_gates_pass_when_every_track_has_a_large_effect():
    payload, outcomes = _gate_inputs(detect_delta=0.09, steer_gain=40, project_gain=40)
    report = analyze_correctness_tracks(payload, outcomes, _Settings())
    assert report["tracks_passed"] == {"detect": True, "steer": True, "project": True}


def test_gates_fail_on_a_null_in_every_track():
    payload, outcomes = _gate_inputs(detect_delta=0.0, steer_gain=0, project_gain=0)
    report = analyze_correctness_tracks(payload, outcomes, _Settings())
    assert report["tracks_passed"] == {"detect": False, "steer": False, "project": False}
    assert report["steer"]["gain_points"] == pytest.approx(0.0)


def test_steer_gate_fails_when_a_random_direction_in_the_band_matches_the_arm():
    """A gain that a random direction reproduces is not evidence about the band."""
    payload, outcomes = _gate_inputs(detect_delta=0.09, steer_gain=40, project_gain=40)
    payload["steer"]["random_band_r00"]["test_accuracy"] = payload["steer"][
        "margin_band"
    ]["test_accuracy"]
    report = analyze_correctness_tracks(payload, outcomes, _Settings())
    assert report["steer"]["passed"] is False
    assert report["steer"]["margin_over_random_points"] == pytest.approx(0.0)


def test_detect_gate_fails_on_a_gain_below_the_preregistered_minimum():
    """A probe can beat the margin significantly and still be too small to matter."""
    payload, outcomes = _gate_inputs(detect_delta=0.002, steer_gain=0, project_gain=0)
    report = analyze_correctness_tracks(payload, outcomes, _Settings())
    assert report["detect"]["delta_auc"] < _Settings.minimum_detect_delta_auc
    assert report["detect"]["passed"] is False


def test_gate_report_surfaces_the_null_comparison_for_the_class_split():
    payload, outcomes = _gate_inputs(detect_delta=0.0, steer_gain=0, project_gain=0)
    report = analyze_correctness_tracks(payload, outcomes, _Settings())
    null = report["geometry"]["null"]
    assert null["median_lift_band_share"] == pytest.approx(0.705)
    assert null["share_exceedances"] == 0
    assert null["norm_ratio"] > 10.0


def test_gates_raise_on_a_missing_preregistered_arm():
    payload, outcomes = _gate_inputs(detect_delta=0.0, steer_gain=0, project_gain=0)
    del payload["steer"]["margin_band"]
    with pytest.raises(KeyError):
        analyze_correctness_tracks(payload, outcomes, _Settings())


def test_gate_report_is_json_serialisable():
    payload, outcomes = _gate_inputs(detect_delta=0.09, steer_gain=40, project_gain=40)
    report = analyze_correctness_tracks(payload, outcomes, _Settings())
    assert json.loads(json.dumps(report))["steer"]["primary_arm"] == "margin_band"


# --------------------------------------------------------------------------
# the factored evaluators must be exact, not merely fast
# --------------------------------------------------------------------------


def test_steered_accuracy_matches_recomputing_the_logits(toy):
    """The broadcast-shift form is algebra, not an approximation."""
    states, readout, gold = toy["states"], toy["readout"], toy["gold"]
    base = states @ readout.T
    vector = torch.randn(GPT2_HIDDEN_SIZE, generator=torch.Generator().manual_seed(2)).double()
    vector = vector / vector.norm()
    for alpha in (0.0, 0.5, 3.0, -2.0, 40.0):
        fast, fast_outcomes = steered_accuracy(base, readout, gold, vector, alpha)
        slow, slow_outcomes = (
            lambda o: (float(o.double().mean()), o)
        )(first_token_correct(steer(states, vector, alpha), readout, gold))
        assert torch.equal(fast_outcomes, slow_outcomes), alpha
        assert fast == pytest.approx(slow)


def test_retained_accuracy_matches_the_dense_projection(toy):
    """The low-rank readout must agree with materialising the edited state."""
    states, readout, gold = toy["states"], toy["readout"], toy["gold"]
    centre = states.mean(0)
    _, vectors = sorted_eigenbasis(torch.cov(states.T))
    for rank in (1, 4, 28, 128):
        basis = vectors[:, :rank]
        fast, fast_outcomes = retained_accuracy(states, readout, gold, basis, centre)
        slow_outcomes = first_token_correct(
            retention(states, basis, centre), readout, gold
        )
        assert torch.equal(fast_outcomes, slow_outcomes), rank
        assert fast == pytest.approx(float(slow_outcomes.double().mean()))


def test_steering_at_alpha_zero_is_exactly_the_baseline(toy):
    states, readout, gold = toy["states"], toy["readout"], toy["gold"]
    base = states @ readout.T
    vector = torch.zeros(GPT2_HIDDEN_SIZE, dtype=torch.float64)
    vector[0] = 1.0
    accuracy, outcomes = steered_accuracy(base, readout, gold, vector, 0.0)
    assert torch.equal(outcomes, first_token_correct(states, readout, gold))
    assert accuracy == pytest.approx(float(outcomes.double().mean()))
