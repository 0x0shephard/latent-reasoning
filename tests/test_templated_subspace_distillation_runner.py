import pytest

pytest.importorskip("torch")

from scripts.run_codi_templated_subspace_distillation import (  # noqa: E402
    ARMS,
    BATCH_SIZE,
    CONTRACT,
    CURVE_EVERY,
    DATA_SEED,
    FIT,
    MINIMUM_DENSE_FIRST_TOKEN,
    MINIMUM_TEACHER_GENERATION,
    SELECTION_EXAMPLES,
    STEPS,
    TEACHER_CHECK,
    TEST_EXAMPLES,
    THRESHOLD_ACCURACY,
    TRAIN_EXAMPLES,
    WARMUP_STEPS,
    claim_from,
    gates_from,
    steps_to_threshold,
)
from scripts.run_codi_causal_subspace_distillation import LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP  # noqa: E402


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_templated_subspace_distillation_v1"
    assert ARMS == ("none", "variance", "causal")
    assert (STEPS, BATCH_SIZE, TRAIN_EXAMPLES, WARMUP_STEPS) == (3_000, 16, 48_000, 150)
    assert (FIT, TEACHER_CHECK, SELECTION_EXAMPLES, TEST_EXAMPLES, CURVE_EVERY) == (2_048, 500, 256, 2_000, 300)
    assert (LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP) == (1e-4, 0.1, 2.0)
    assert (MINIMUM_TEACHER_GENERATION, MINIMUM_DENSE_FIRST_TOKEN, THRESHOLD_ACCURACY) == (0.80, 0.80, 0.30)
    assert DATA_SEED == 20_260_924


def test_steps_to_threshold():
    curve = [{"step": 300, "accuracy": 0.1}, {"step": 600, "accuracy": 0.31}, {"step": 900, "accuracy": 0.5}]
    assert steps_to_threshold(curve, 0.30) == 600
    assert steps_to_threshold(curve[:1], 0.30) is None


def _comparisons(**bounds):
    names = ["none_minus_variance", "causal_minus_variance", "causal_minus_none"]
    return {n: {"mean_difference": 0.0, "bootstrap_95ci": list(bounds.get(n, (-0.01, 0.05)))} for n in names}


def test_gates_and_claims():
    gate = gates_from(_comparisons(none_minus_variance=(0.01, 0.05), causal_minus_variance=(0.01, 0.05)))
    assert claim_from(gate).startswith("PERMANENT")
    gate = gates_from(_comparisons(none_minus_variance=(0.01, 0.05)))
    assert claim_from(gate).startswith("PARTIAL: the variance tax persists")
    gate = gates_from(_comparisons(none_minus_variance=(-0.05, -0.01)))
    assert claim_from(gate).startswith("REVERSED")
    gate = gates_from(_comparisons(causal_minus_variance=(0.01, 0.05)))
    assert claim_from(gate).startswith("PARTIAL: causal beats variance")
    gate = gates_from(_comparisons())
    assert claim_from(gate).startswith("TRANSIENT")
    assert gates_from(_comparisons(causal_minus_none=(-0.05, -0.01)))["h1b_none_beats_causal"]
