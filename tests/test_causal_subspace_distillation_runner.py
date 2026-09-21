import pytest

pytest.importorskip("torch")

from scripts.run_codi_causal_subspace_distillation import (  # noqa: E402
    BATCH_SIZE,
    CANDIDATE_PCS,
    CONTRACT,
    CURVE_EVERY,
    DATA_SEED,
    FIT_EXAMPLES,
    GRAD_CLIP,
    LEARNING_RATE,
    MINIMUM_GAP,
    MINIMUM_RETENTION,
    RANK_GRID,
    SELECTION_EXAMPLES,
    STEPS,
    TRAIN_EXAMPLES,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    claim_from,
    gates_from,
    prepare_row,
    sample_splits,
)
from src.mech.causal_subspace_distillation import ARMS  # noqa: E402
from tests.test_trajectory_supervision import CharTokenizer  # noqa: E402
from tests.test_trajectory_supervision_runner import FakeDataset  # noqa: E402


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_causal_subspace_distillation_v1"
    assert (FIT_EXAMPLES, STEPS, BATCH_SIZE, TRAIN_EXAMPLES) == (2_048, 10_000, 16, 160_000)
    assert SELECTION_EXAMPLES == 256 and CURVE_EVERY == 1_000
    assert (LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY, GRAD_CLIP) == (1e-4, 500, 0.1, 2.0)
    assert RANK_GRID == (8, 12, 16) and CANDIDATE_PCS == 128
    assert (MINIMUM_RETENTION, MINIMUM_GAP) == (0.75, 0.05)
    assert DATA_SEED == 20_260_923
    assert ARMS == ("none", "full", "variance", "relevance", "causal", "random")


def test_prepare_row_normalises_gold_and_filters():
    tokenizer = CharTokenizer()
    row = {"question": "What is 2 + 3?", "cot": "<<2+2=4>> <<4+1=5>>", "answer": "#### 5"}
    assert prepare_row(tokenizer, row, bot_token_id=61)["gold"] == "5"
    assert prepare_row(tokenizer, {**row, "answer": "#### 1,234"}, bot_token_id=61)["gold"] == "1234"
    assert prepare_row(tokenizer, {**row, "answer": "#### -5"}, bot_token_id=61) is None


def test_sample_splits_are_disjoint_ordered_and_deterministic():
    tokenizer = CharTokenizer()
    rows = [{"question": f"q{i}?", "cot": f"<<{i}+1={i+1}>> <<1+1=2>>", "answer": f"#### {i+1}"} for i in range(30)]
    sizes = {"fit": 4, "select": 4, "validate": 4, "train": 8, "selection": 2}
    a, audit_a = sample_splits(FakeDataset(rows), tokenizer, test_questions=set(), sizes=sizes, seed=5, bot_token_id=61)
    b, audit_b = sample_splits(FakeDataset(rows), tokenizer, test_questions=set(), sizes=sizes, seed=5, bot_token_id=61)
    assert audit_a["hashes"] == audit_b["hashes"]
    seen = set()
    for name, size in sizes.items():
        assert len(a[name]) == size
        questions = {r["question"] for r in a[name]}
        assert not questions & seen
        seen |= questions
    with pytest.raises(RuntimeError, match="overlap"):
        sample_splits(FakeDataset(rows), tokenizer, test_questions={"q3?"}, sizes=sizes, seed=5, bot_token_id=61)


def _comparisons(**lower):
    names = ["full_minus_none", "causal_minus_variance", "causal_minus_relevance",
             "causal_minus_random", "variance_minus_random"]
    return {n: {"mean_difference": 0.0, "bootstrap_95ci": [lower.get(n, -0.01), 0.05]} for n in names}


def test_gates_and_claims():
    gate = gates_from(_comparisons(full_minus_none=0.01, causal_minus_variance=0.01, causal_minus_random=0.01))
    assert gate["headline"] and claim_from(gate).startswith("CONFIRMED")
    gate = gates_from(_comparisons(causal_minus_variance=0.01, causal_minus_random=0.01))
    assert claim_from(gate).startswith("STOP: distillation signal")
    gate = gates_from(_comparisons(full_minus_none=0.01, causal_minus_random=0.01))
    assert claim_from(gate).startswith("NULL")
    gate = gates_from(_comparisons(full_minus_none=0.01))
    assert claim_from(gate).startswith("STOP: causal selection")
