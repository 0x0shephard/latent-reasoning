import pytest

pytest.importorskip("torch")

from scripts.run_codi_recovery_subspace_distillation import (  # noqa: E402
    ARMS_DAMAGE,
    ARMS_PRIMARY,
    ARMS_WEIGHTS,
    BATCH_SIZE,
    CONTRACT,
    CURVE_EVERY,
    DAMAGES,
    HEADROOM_MIN_GAP,
    LEARNING_RATE_LORA,
    LEARNING_RATE_PROJECTOR,
    MAX_SHARED_PCS,
    NULL_WIDTH,
    NOISE_GRID,
    PILOT_SEEDS,
    PILOT_STEPS,
    RECOVERY_FRACTION,
    STEP_OPTIONS,
    TERM_RATIO_MIN,
    WARMUP_STEPS,
    choose_steps,
    choose_steps_for_pilots,
    claim_from,
    gates_from,
    parse_arm,
    recovery_threshold,
)


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_recovery_subspace_distillation_v1"
    assert ARMS_PRIMARY == ("none", "full", "variance", "relevance", "causal", "random")
    assert ARMS_WEIGHTS == ("variance_x0.3", "variance_x3", "causal_x0.3", "causal_x3")
    assert ARMS_DAMAGE == ("none", "variance", "causal", "random")
    assert DAMAGES == ("projector_noise", "projector", "lora_half") and NOISE_GRID == (0.25, 0.5, 1.0, 2.0)
    assert (STEP_OPTIONS, PILOT_STEPS, PILOT_SEEDS, BATCH_SIZE, CURVE_EVERY) == ((1_000, 2_000), 2_000, (0, 100), 16, 100)
    assert (LEARNING_RATE_LORA, LEARNING_RATE_PROJECTOR, WARMUP_STEPS) == (1e-4, 5e-4, 50)
    assert (HEADROOM_MIN_GAP, MAX_SHARED_PCS, TERM_RATIO_MIN, RECOVERY_FRACTION, NULL_WIDTH) == (0.20, 8, 1.5, 0.5, 0.03)
    assert recovery_threshold(0.82, 0.504) == pytest.approx(0.662)


def test_parse_arm():
    assert parse_arm("variance") == ("variance", 1.0)
    assert parse_arm("variance_x0.3") == ("variance", 0.3)
    assert parse_arm("causal_x3") == ("causal", 3.0)


def test_choose_steps_follows_the_preregistered_rule():
    curve = lambda first: [{"step": s, "accuracy": 0.70 if s >= first else 0.5} for s in range(100, 2_001, 100)]
    assert choose_steps(curve(700), threshold=0.662) == 1_000
    assert choose_steps(curve(1_000), threshold=0.662) == 1_000
    assert choose_steps(curve(1_100), threshold=0.662) == 2_000
    flat = [{"step": s, "accuracy": 0.5} for s in range(100, 2_001, 100)]
    assert choose_steps(flat, threshold=0.662) is None
    # two pilots: the budget both satisfy; a disagreeing pair is a STOP
    assert choose_steps_for_pilots([curve(700), curve(900)], threshold=0.662) == 1_000
    assert choose_steps_for_pilots([curve(700), curve(1_300)], threshold=0.662) == 2_000
    assert choose_steps_for_pilots([curve(700), flat], threshold=0.662) is None
    assert choose_steps_for_pilots([], threshold=0.662) is None


def _comparisons(**bounds):
    names = ["full_minus_none", "causal_minus_variance", "causal_minus_relevance", "causal_minus_random",
             "variance_minus_none", "causal_minus_none"]
    return {n: {"mean_difference": 0.0, "bootstrap_95ci": list(bounds.get(n, (-0.05, 0.05)))} for n in names}


def test_gates_and_claims():
    assert claim_from(gates_from(_comparisons(), 0.55, 0.662)).startswith("STOP")
    g = gates_from(_comparisons(causal_minus_variance=(0.01, 0.05), causal_minus_random=(0.01, 0.05)), 0.7, 0.662)
    assert claim_from(g).startswith("CONFIRMED")
    g = gates_from(_comparisons(causal_minus_variance=(-0.05, -0.01)), 0.7, 0.662)
    assert claim_from(g).startswith("REVERSED")
    g = gates_from(_comparisons(causal_minus_variance=(-0.01, 0.012)), 0.7, 0.662)
    assert g["h1_tight_null"] and claim_from(g).startswith("NULL")
    g = gates_from(_comparisons(causal_minus_variance=(0.01, 0.05)), 0.7, 0.662)
    assert claim_from(g).startswith("PARTIAL")
    g = gates_from(_comparisons(causal_minus_variance=(-0.04, 0.04)), 0.7, 0.662)
    assert claim_from(g).startswith("INCONCLUSIVE")
    g = gates_from(_comparisons(variance_minus_none=(-0.06, -0.02)), 0.7, 0.662)
    assert g["variance_hurts_recovery"] and not g["variance_helps_recovery"]
