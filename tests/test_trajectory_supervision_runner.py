import pytest

pytest.importorskip("torch")

from scripts.run_codi_trajectory_supervision import (  # noqa: E402
    BATCH_SIZE,
    CONTRACT,
    DATA_SEED,
    DRIFT_MARGIN,
    GRAD_CLIP,
    IMPORTANCE_WEIGHT,
    LEARNING_RATE,
    MINIMUM_EQUATIONS,
    SCREEN_ARM,
    SCREEN_SEED,
    SELECTION_EXAMPLES,
    TRAIN_EXAMPLES,
    TRAINING_SEEDS,
    claim_from,
    gates_from,
    prepare_row,
    sample_rows,
)
from src.mech.trajectory_supervision import ARM_ORDER  # noqa: E402
from tests.test_trajectory_supervision import CharTokenizer  # noqa: E402


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_trajectory_supervision_v1"
    assert TRAIN_EXAMPLES == 8_192 and SELECTION_EXAMPLES == 256
    assert DATA_SEED == 20_260_922 and TRAINING_SEEDS == (1, 2, 3)
    assert BATCH_SIZE == 8 and LEARNING_RATE == 2e-5 and GRAD_CLIP == 1.0
    assert IMPORTANCE_WEIGHT == 0.1 and MINIMUM_EQUATIONS == 2 and DRIFT_MARGIN == 0.03
    assert (SCREEN_ARM, SCREEN_SEED) == ("codi", 1)
    assert ARM_ORDER == ("codi", "kava", "value_odd", "random_odd", "value_even", "recon_odd")


def test_prepare_row_localises_values_and_applies_filters():
    tokenizer = CharTokenizer()
    row = {"question": "What is 2 + 3?", "cot": "<<2+2=4>> <<4+1=5>> <<5+0=5>>", "answer": "#### 5"}
    prepared = prepare_row(tokenizer, row, bot_token_id=61)
    assert prepared["gold"] == "5"
    assert prepared["value_positions"] == [6, 16]
    assert prepared["value_token_ids"] == [tokenizer.vocab["4"], tokenizer.vocab["5"]]
    assert prepare_row(tokenizer, {**row, "cot": "<<2+3=5>>"}, bot_token_id=61) is None
    assert prepare_row(tokenizer, {**row, "answer": "#### -5"}, bot_token_id=61) is None
    assert prepare_row(tokenizer, {**row, "cot": "no equations here at all"}, bot_token_id=61) is None
    # Thousands separators in the gold must be normalised the way the scorer
    # normalises generated text, and a non-numeric gold excludes the row.
    comma = prepare_row(tokenizer, {**row, "answer": "#### 333,333.33"}, bot_token_id=61)
    assert comma["gold"] == "333333.33"
    assert prepare_row(tokenizer, {**row, "answer": "#### 5abc"}, bot_token_id=61) is None


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def __len__(self):
        return len(self.rows)


def test_sample_rows_is_deterministic_unique_and_disjoint_from_test():
    tokenizer = CharTokenizer()
    rows = []
    for i in range(12):
        rows.append({"question": f"q{i}?", "cot": f"<<{i}+1={i+1}>> <<{i+1}+0={i+1}>> <<1+1=2>>",
                     "answer": f"#### {i+1}"})
    rows.append(dict(rows[0]))  # duplicate question must collapse to one entry
    rows.append({"question": "bad?", "cot": "<<1+1=2>>", "answer": "#### 2"})  # one equation
    dataset = FakeDataset(rows)
    train, selection, audit = sample_rows(
        dataset, tokenizer, test_questions=set(), train_examples=8, selection_examples=2,
        seed=7, bot_token_id=51,
    )
    train2, selection2, audit2 = sample_rows(
        dataset, tokenizer, test_questions=set(), train_examples=8, selection_examples=2,
        seed=7, bot_token_id=51,
    )
    assert [r["question"] for r in train] == [r["question"] for r in train2]
    assert audit["train_questions_sha256"] == audit2["train_questions_sha256"]
    questions = [r["question"] for r in train + selection]
    assert len(set(questions)) == 10 and "bad?" not in questions
    assert audit["unique_questions"] == 13
    with pytest.raises(RuntimeError, match="overlap"):
        sample_rows(dataset, tokenizer, test_questions={"q1?"}, train_examples=1,
                    selection_examples=1, seed=7, bot_token_id=61)
    with pytest.raises(ValueError, match="eligible"):
        sample_rows(dataset, tokenizer, test_questions=set(), train_examples=20,
                    selection_examples=2, seed=7, bot_token_id=61)


def _comparisons(**lower):
    names = ["kava_minus_codi", "value_odd_minus_kava", "value_odd_minus_random_odd",
             "value_odd_minus_value_even", "value_odd_minus_codi"]
    return {n: {"mean_difference": 0.0, "bootstrap_95ci": [lower.get(n, -0.01), 0.05]} for n in names}


def test_gates_and_claims():
    gate = gates_from(_comparisons(value_odd_minus_kava=0.01, value_odd_minus_random_odd=0.01,
                                   value_odd_minus_codi=0.01))
    assert gate["headline"] and claim_from(gate).startswith("CONFIRMED")
    gate = gates_from(_comparisons(value_odd_minus_random_odd=0.01, value_odd_minus_codi=0.01))
    assert not gate["headline"] and claim_from(gate).startswith("PARTIAL: value-slot")
    gate = gates_from(_comparisons(kava_minus_codi=0.01))
    assert claim_from(gate).startswith("PARTIAL: KaVa")
    gate = gates_from(_comparisons())
    assert claim_from(gate).startswith("STOP")
