"""Invariants for the answer-colon margin-geometry experiment."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.eval.official_codi_endpoint_margin_geometry_analysis import (
    analyze_margin_geometry,
    binary_comparison,
    continuous_comparison,
)
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    DEFAULT_RANK_GRID,
    MarginSubspace,
    answer_token_outcomes,
    build_fitted_subspaces,
    build_margin_arm_registry,
    build_matched_random_subspaces,
    deterministic_derangement,
    edited_states,
    energy_matched_random_subspace,
    evaluate_subspace_analytically,
    margin_damage_matrix,
    state_covariance,
    subspace_energy,
    validate_margin_subspace,
)

HIDDEN = 768


def _random_states(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, HIDDEN, generator=generator)


def _small_readout(vocab: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, HIDDEN, generator=generator) * 0.05


def test_margin_damage_matrix_is_symmetric_and_optimal():
    states = _random_states(256, 0)
    gradients = _random_states(256, 1)
    damage = margin_damage_matrix(states, gradients)
    assert torch.allclose(damage, damage.T, atol=1e-6)

    # The top-k eigenvectors must beat many random orthonormal subspaces at the
    # objective they were derived to maximise: mean g^T U U^T c.
    values, vectors = torch.linalg.eigh(damage.double())
    basis = vectors[:, torch.argsort(values, descending=True)[:4]].float()

    def objective(candidate: torch.Tensor) -> float:
        projected = (states @ candidate) @ candidate.T
        return float((gradients * projected).sum(dim=-1).mean())

    best = objective(basis)
    generator = torch.Generator().manual_seed(7)
    for _ in range(25):
        random = torch.linalg.qr(
            torch.randn(HIDDEN, 4, generator=generator), mode="reduced"
        )[0]
        assert best >= objective(random)


def test_removal_and_retention_are_complementary():
    states = _random_states(32, 2)
    mean = states.mean(dim=0)
    basis = torch.linalg.qr(
        torch.randn(HIDDEN, 5, generator=torch.Generator().manual_seed(3)),
        mode="reduced",
    )[0]
    removed = edited_states(states, basis, mode="remove", semantics="mean", mean=mean)
    retained = edited_states(states, basis, mode="retain", semantics="mean", mean=mean)
    # remove + retain must reconstruct the original state plus one copy of the centre.
    assert torch.allclose(removed + retained, states + mean.unsqueeze(0), atol=1e-4)
    # The removed state has no component along the subspace.
    assert torch.allclose(
        (removed - mean.unsqueeze(0)) @ basis,
        torch.zeros(states.shape[0], 5),
        atol=1e-4,
    )


def test_zero_semantics_also_removes_the_mean_component():
    states = _random_states(16, 4) + 3.0
    mean = states.mean(dim=0)
    basis = torch.linalg.qr(
        torch.randn(HIDDEN, 2, generator=torch.Generator().manual_seed(5)),
        mode="reduced",
    )[0]
    mean_ablated = edited_states(states, basis, mode="remove", semantics="mean", mean=mean)
    zero_ablated = edited_states(states, basis, mode="remove", semantics="zero", mean=mean)
    # Mean-preserving ablation leaves the constant component untouched; the zero
    # variant does not.  This asymmetry is exactly what the earlier experiments
    # could not see.
    assert not torch.allclose(mean_ablated, zero_ablated, atol=1e-3)
    assert torch.allclose(zero_ablated @ basis, torch.zeros(states.shape[0], 2), atol=1e-4)


def test_resample_semantics_requires_a_donor():
    states = _random_states(8, 6)
    basis = torch.linalg.qr(
        torch.randn(HIDDEN, 2, generator=torch.Generator().manual_seed(8)),
        mode="reduced",
    )[0]
    with pytest.raises(ValueError):
        edited_states(
            states, basis, mode="remove", semantics="resample", mean=states.mean(dim=0)
        )


def test_deterministic_derangement_has_no_fixed_point():
    permutation = deterministic_derangement(64, 11)
    assert permutation.shape == (64,)
    assert sorted(permutation.tolist()) == list(range(64))
    assert not bool((permutation == torch.arange(64)).any())
    assert torch.equal(permutation, deterministic_derangement(64, 11))


def test_energy_matched_random_subspace_matches_a_mid_spectrum_target():
    states = _random_states(512, 12)
    covariance = state_covariance(states - states.mean(dim=0))
    # A margin-style subspace is not the top-energy subspace, so its energy is
    # reachable from inside the complement.
    selected = build_fitted_subspaces(
        family="margin",
        rank=3,
        state=ANALYTIC_STATE,
        covariance=covariance,
        damage_matrix=margin_damage_matrix(
            states - states.mean(dim=0), _random_states(512, 40)
        ),
    )
    target = subspace_energy(covariance, selected.basis)
    generator = torch.Generator().manual_seed(13)
    basis, matching = energy_matched_random_subspace(
        covariance=covariance,
        rank=3,
        target_energy=target,
        generator=generator,
        selected_basis=selected.basis,
    )
    assert matching["target_attainable"] is True
    assert matching["relative_error"] <= 1e-2
    # Drawn inside the orthogonal complement, so it cannot re-use the selection.
    overlap = float((selected.basis.T @ basis).square().sum() / 3)
    assert overlap < 1e-6
    assert torch.allclose(basis.T @ basis, torch.eye(3), atol=1e-5)


def test_top_energy_selection_reports_an_unattainable_match():
    """No selected-orthogonal control can match the top-energy subspace.

    Reporting this is the point: an unmatched control is what makes a random null
    conservative, which is the flaw the completed state-12 confirmation carried.
    """
    states = _random_states(512, 12)
    covariance = state_covariance(states - states.mean(dim=0))
    selected = build_fitted_subspaces(
        family="energy", rank=3, state=ANALYTIC_STATE, covariance=covariance
    )
    target = subspace_energy(covariance, selected.basis)
    _, matching = energy_matched_random_subspace(
        covariance=covariance,
        rank=3,
        target_energy=target,
        generator=torch.Generator().manual_seed(13),
        selected_basis=selected.basis,
    )
    assert matching["target_attainable"] is False
    assert matching["attainable_maximum"] < target


def test_high_rank_controls_drop_the_infeasible_disjointness_constraint():
    """Above rank 384 two subspaces of that rank must intersect in 768 dimensions.

    Requiring selected-orthogonality there is not merely inconvenient, it is
    impossible, so the constraint is dropped and reported rather than approximated.
    The retention curve is descriptive; every gated comparison happens at rank 3.
    """
    states = _random_states(600, 60)
    covariance = state_covariance(states - states.mean(dim=0))
    selected = build_fitted_subspaces(
        family="energy", rank=512, state=ANALYTIC_STATE, covariance=covariance
    )
    controls = build_matched_random_subspaces(
        selected=selected, covariance=covariance, replicates=1, seed=61
    )
    control = controls[0]
    assert control.rank == 512
    assert control.selected_orthogonal is False
    validate_margin_subspace(control)


def test_low_rank_controls_keep_the_disjointness_constraint():
    states = _random_states(600, 62)
    covariance = state_covariance(states - states.mean(dim=0))
    selected = build_fitted_subspaces(
        family="energy", rank=3, state=ANALYTIC_STATE, covariance=covariance
    )
    control = build_matched_random_subspaces(
        selected=selected, covariance=covariance, replicates=1, seed=63
    )[0]
    assert control.selected_orthogonal is True
    assert control.selected_overlap < 1e-6


def test_attainable_range_uses_an_exact_complement_basis():
    """The complement must come from the projector's spectrum, not bare QR.

    Unpivoted QR gives no guarantee about where the null directions land, so a
    QR-derived complement can silently mix selected directions back in.
    """
    from src.mech.endpoint_margin_geometry import attainable_energy_range

    # More samples than dimensions, so the covariance is full rank, matching the
    # 2,048-example calibration the contract requires.
    states = _random_states(1024, 64)
    covariance = state_covariance(states - states.mean(dim=0))
    selected = build_fitted_subspaces(
        family="energy", rank=3, state=ANALYTIC_STATE, covariance=covariance
    )
    minimum, maximum = attainable_energy_range(covariance, 3, selected.basis)
    top = subspace_energy(covariance, selected.basis)
    # The complement excludes the three highest-variance directions by construction.
    assert maximum < top
    assert 0 < minimum <= maximum
    with pytest.raises(ValueError, match="complement dimension"):
        attainable_energy_range(covariance, 766, selected.basis)


def test_matched_random_controls_share_one_calibration_target():
    states = _random_states(256, 14)
    covariance = state_covariance(states - states.mean(dim=0))
    selected = build_fitted_subspaces(
        family="energy", rank=2, state=ANALYTIC_STATE, covariance=covariance
    )
    controls = build_matched_random_subspaces(
        selected=selected, covariance=covariance, replicates=4, seed=15
    )
    assert len(controls) == 4
    targets = {round(float(control.calibration_target_energy), 9) for control in controls}
    assert len(targets) == 1
    for control in controls:
        validate_margin_subspace(control)
        assert control.rank == selected.rank
        assert control.matched_family == "energy"


def test_truncated_replicate_draws_match_the_full_sequence():
    """The generation runner builds only as many controls as an arm needs.

    That is only sound because replicates come from one seeded generator in order,
    so replicate ``r`` must be bit-identical whether the registry stops at ``r+1``
    or continues to the configured 200.
    """
    from scripts.run_official_codi_endpoint_margin_generation import (
        required_random_replicates,
    )

    states = _random_states(1024, 70)
    covariance = state_covariance(states - states.mean(dim=0))
    selected = build_fitted_subspaces(
        family="energy", rank=3, state=ANALYTIC_STATE, covariance=covariance
    )
    full = build_matched_random_subspaces(
        selected=selected, covariance=covariance, replicates=4, seed=71
    )
    for count in (1, 2, 4):
        partial = build_matched_random_subspaces(
            selected=selected, covariance=covariance, replicates=count, seed=71
        )
        assert len(partial) == count
        for index in range(count):
            assert torch.equal(partial[index].basis, full[index].basis)

    assert required_random_replicates("random_matched_margin_k003_s12_r000") == 1
    assert required_random_replicates("random_matched_margin_k003_s12_r007") == 8
    assert required_random_replicates("margin_k016_s12") == 1


def test_analytic_outcomes_are_paired_and_finite():
    states = _random_states(24, 16)
    readout = _small_readout(97, 17)
    gold = torch.randint(0, 97, (24,), generator=torch.Generator().manual_seed(18))
    logits = states @ readout.T
    outcomes = answer_token_outcomes(logits, gold)
    assert outcomes["nll"].shape == (24,)
    assert torch.isfinite(outcomes["nll"]).all()
    # The margin is positive exactly when the gold token is the argmax.
    assert torch.equal(outcomes["margin"] > 0, outcomes["top1_correct"])


def test_evaluate_subspace_analytically_matches_a_direct_recomputation():
    states = _random_states(20, 19)
    readout = _small_readout(64, 20)
    gold = torch.randint(0, 64, (20,), generator=torch.Generator().manual_seed(21))
    mean = states.mean(dim=0)
    basis = torch.linalg.qr(
        torch.randn(HIDDEN, 3, generator=torch.Generator().manual_seed(22)),
        mode="reduced",
    )[0]
    result = evaluate_subspace_analytically(
        hidden=states,
        readout=readout,
        gold_token=gold,
        mean=mean,
        basis=basis,
        chunk_size=7,
    )
    expected = edited_states(states, basis, mode="remove", semantics="mean", mean=mean)
    direct = answer_token_outcomes(expected @ readout.T, gold)
    assert torch.allclose(result["nll"], direct["nll"].cpu(), atol=1e-6)
    assert torch.equal(result["top1_correct"], direct["top1_correct"].cpu())


def test_registry_covers_every_family_and_matches_randoms():
    states = _random_states(384, 23)
    centered = states - states.mean(dim=0)
    covariance = state_covariance(centered)
    damage = {
        "margin": margin_damage_matrix(centered, _random_states(384, 24)),
        "answer_nll": margin_damage_matrix(centered, _random_states(384, 25)),
    }
    reference = MarginSubspace(
        name=f"parameter_aware_k003_s{ANALYTIC_STATE}",
        family="parameter_aware",
        state=ANALYTIC_STATE,
        basis=torch.linalg.qr(
            torch.randn(HIDDEN, 3, generator=torch.Generator().manual_seed(26)),
            mode="reduced",
        )[0],
        rank=3,
    )
    registry = build_margin_arm_registry(
        covariance=covariance,
        damage_matrices=damage,
        readout_matrix=_small_readout(128, 27),
        reference_subspaces={"parameter_aware": reference},
        rank_grid=(1, 3),
        random_replicates=2,
        random_seed=28,
    )
    for name in (
        f"margin_k003_s{ANALYTIC_STATE}",
        f"answer_nll_k001_s{ANALYTIC_STATE}",
        f"energy_k003_s{ANALYTIC_STATE}",
        f"readout_k001_s{ANALYTIC_STATE}",
        f"parameter_aware_k003_s{ANALYTIC_STATE}",
        f"random_matched_margin_k003_s{ANALYTIC_STATE}_r000",
        f"random_matched_parameter_aware_k003_s{ANALYTIC_STATE}_r001",
    ):
        assert name in registry, name
        validate_margin_subspace(registry[name])


def test_default_rank_grid_is_sorted_and_bounded():
    assert list(DEFAULT_RANK_GRID) == sorted(DEFAULT_RANK_GRID)
    assert DEFAULT_RANK_GRID[0] >= 1 and DEFAULT_RANK_GRID[-1] <= HIDDEN


def test_continuous_outcome_has_more_power_than_binary():
    """A small consistent shift is detectable continuously but not by exact match.

    This is the quantitative statement of why the completed confirmation could
    not separate its effect from the matched-random null.
    """
    generator = np.random.default_rng(0)
    baseline_nll = generator.normal(1.0, 0.5, size=1319)
    arm_nll = baseline_nll + 0.01
    continuous = continuous_comparison(
        baseline_nll, arm_nll, bootstrap_samples=2000, seed=0
    )
    assert continuous["bootstrap_95_ci"][0] > 0
    assert continuous["z_score"] > 10

    baseline_correct = generator.random(1319) < 0.43
    arm_correct = baseline_correct.copy()
    flip = np.flatnonzero(baseline_correct)[:20]
    arm_correct[flip] = False
    binary = binary_comparison(baseline_correct, arm_correct)
    assert binary["correct_to_wrong"] == 20
    assert binary["z_score"] < continuous["z_score"]


def _synthetic_sweep(effect: float, seed: int) -> dict:
    """A minimal sweep payload with one margin arm and matched random controls."""
    generator = torch.Generator().manual_seed(seed)
    count = 256
    baseline_nll = torch.rand(count, generator=generator).double() + 1.0
    baseline_top1 = torch.rand(count, generator=generator) < 0.45

    def arm(delta: float, rank: int, family: str, replicate=None, name=None):
        subspace = {
            "name": name,
            "family": family,
            "state": ANALYTIC_STATE,
            "rank": rank,
            "random_replicate": replicate,
            "calibration_target_energy": 10.0,
            "calibration_achieved_energy": 10.0,
            "selected_overlap": 0.0,
        }
        return {
            "nll": baseline_nll + delta,
            "margin": torch.zeros(count).double() - delta,
            "top1_correct": baseline_top1.clone(),
            "removed_projection_rms": 1.0,
            "subspace": subspace,
        }

    arms = {}
    primary_name = f"margin_k003_s{ANALYTIC_STATE}"
    arms[f"{primary_name}|remove|mean"] = arm(effect, 3, "margin", name=primary_name)
    for replicate in range(20):
        name = f"random_matched_margin_k003_s{ANALYTIC_STATE}_r{replicate:03d}"
        arms[f"{name}|remove|mean"] = arm(
            0.001 * replicate, 3, "random_matched", replicate, name
        )
    for rank in (1, 3):
        name = f"margin_k{rank:03d}_s{ANALYTIC_STATE}"
        retained = arm(0.0, rank, "margin", name=name)
        retained["top1_correct"] = (
            baseline_top1.clone() if rank == 3 else torch.zeros(count, dtype=torch.bool)
        )
        arms[f"{name}|retain|mean"] = retained
    return {
        "arms": arms,
        "baseline": {
            "nll": baseline_nll,
            "margin": torch.zeros(count).double(),
            "top1_correct": baseline_top1,
        },
    }


def test_analysis_supports_a_clear_margin_effect():
    report = analyze_margin_geometry(
        _synthetic_sweep(effect=0.5, seed=31), bootstrap_samples=500
    )
    assert report["status"] == "margin_specificity_supported"
    primary = report["primary_margin_specificity"]
    assert primary["supported"] is True
    assert primary["empirical_matched_random_p"] <= 0.05
    assert primary["calibration_matching_passed"] is True


def test_analysis_rejects_an_effect_inside_the_random_null():
    report = analyze_margin_geometry(
        _synthetic_sweep(effect=0.0, seed=32), bootstrap_samples=500
    )
    assert report["status"] == "margin_specificity_not_supported"
    assert report["primary_margin_specificity"]["supported"] is False


def test_effective_rank_uses_the_retention_threshold():
    report = analyze_margin_geometry(
        _synthetic_sweep(effect=0.5, seed=33), bootstrap_samples=500
    )
    effective = report["effective_dimensionality"]["effective_rank_by_family"]
    # Rank one retains nothing and rank three retains the baseline exactly, so the
    # smallest sufficient rank must be three.
    assert effective["margin"] == 3


def test_gold_text_renders_the_emitted_surface_form():
    """The first gold token depends on the exact string, so rendering matters."""
    from decimal import Decimal

    from scripts.collect_official_codi_endpoint_margin_states import gold_text

    assert gold_text(Decimal("18")) == "18"
    # An integral Decimal must not become "18.0" or "1.8E+1".
    assert gold_text(Decimal("18.0")) == "18"
    assert gold_text(Decimal("70000")) == "70000"
    # GSM8K test rows 489 and 1113 have negative answers and are kept.
    assert gold_text(Decimal("-10")) == "-10"
    assert gold_text(" 42 ") == "42"


class _FakeBlock(torch.nn.Module):
    def forward(self, hidden):  # GPT-2 blocks return a tuple
        return (hidden,)


class _FakeLayerNorm(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _FakeCodi(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList(_FakeBlock() for _ in range(12))
        self.transformer.ln_f = _FakeLayerNorm()


class _FakeModel:
    def __init__(self) -> None:
        self.codi = _FakeCodi()


def test_state_collector_captures_the_colon_row_in_order():
    """The collector must observe, not rewrite, and must preserve batch order."""
    from src.mech.endpoint_margin_geometry import OfficialCODIEndpointStateCollector

    model = _FakeModel()
    collector = OfficialCODIEndpointStateCollector(model)
    blocks = model.codi.transformer.h
    layer_norm = model.codi.transformer.ln_f
    chunks = [
        torch.arange(2 * 3 * HIDDEN, dtype=torch.float32).reshape(2, 3, HIDDEN),
        torch.zeros(2, 3, HIDDEN) + 7.0,
    ]
    for chunk in chunks:
        with collector.activate(torch.ones(chunk.shape[0], dtype=torch.bool)):
            returned = blocks[10](chunk)
            assert torch.equal(returned[0], chunk)  # observation only
            layer_norm(chunk)
    stacked = collector.stacked(4)
    assert stacked.shape == (4, 13, HIDDEN)
    # Only the two endpoint states are populated.
    assert torch.count_nonzero(stacked[:, :11, :]) == 0
    # The captured row is the last position of each sequence, in batch order.
    assert torch.equal(stacked[0, 12, :], chunks[0][0, -1, :])
    assert torch.equal(stacked[3, 12, :], chunks[1][1, -1, :])


def test_state_collector_rejects_a_wrong_row_count():
    """A second answer-cue forward would silently double the cache."""
    from src.mech.endpoint_margin_geometry import OfficialCODIEndpointStateCollector

    model = _FakeModel()
    collector = OfficialCODIEndpointStateCollector(model)
    hidden = torch.zeros(2, 3, HIDDEN)
    with collector.activate(torch.ones(2, dtype=torch.bool)):
        model.codi.transformer.h[10](hidden)
        model.codi.transformer.ln_f(hidden)
        model.codi.transformer.h[10](hidden)
        model.codi.transformer.ln_f(hidden)
    with pytest.raises(RuntimeError, match="captured 4 rows, expected 2"):
        collector.stacked(2)


def test_state_collector_cannot_nest():
    from src.mech.endpoint_margin_geometry import OfficialCODIEndpointStateCollector

    collector = OfficialCODIEndpointStateCollector(_FakeModel())
    with collector.activate(torch.ones(1, dtype=torch.bool)):
        with pytest.raises(RuntimeError):
            with collector.activate(torch.ones(1, dtype=torch.bool)):
                pass


def test_analysis_refuses_a_sweep_without_matched_controls():
    sweep = _synthetic_sweep(effect=0.5, seed=34)
    sweep["arms"] = {
        key: value
        for key, value in sweep["arms"].items()
        if not key.startswith("random_matched")
    }
    with pytest.raises(RuntimeError):
        analyze_margin_geometry(sweep, bootstrap_samples=100)
