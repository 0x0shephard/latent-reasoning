import pytest

from src.data.chain_pools import (
    DEFAULT_POOL,
    asdiv_chain_row,
    formula_chain,
    POOLS,
    chain_rows,
    eval_pool,
    gsm8k_chain_row,
    normalized_question,
    pool_rows,
    sha_of,
    steps_histogram,
    svamp_chain_row,
)
from src.data.counterfactual_chain import parse_chain, perturb_row


def test_pools_are_declared():
    assert DEFAULT_POOL == "gsm8k_aug_train" and set(POOLS) == {"gsm8k_aug_train", "gsm8k_test", "svamp", "gsm8k_train_original", "asdiv"}
    assert POOLS["svamp"]["long_profile"] is False and POOLS["gsm8k_test"]["long_profile"] is True
    spec = POOLS["gsm8k_train_original"]["spec"]
    assert spec["data_file"].endswith("3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl") and spec["split"] == "train"
    assert POOLS["gsm8k_train_original"]["adapter"] == "gsm8k" and POOLS["gsm8k_train_original"]["long_profile"] is True
    assert chain_rows("gsm8k_train_original", [{"question": "q 2 and 3", "answer": "<<2+3=5>>\n#### 5"}])[0]["cot"] == "<<2+3=5>>"


def test_gsm8k_annotations_become_the_chain():
    raw = {"question": "Janet's ducks lay 16 eggs. She eats 3 and bakes with 4. She sells the rest at $2 each. How much?",
           "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs.\nShe makes 9 * 2 = $<<9*2=18>>18 daily.\n#### 18"}
    row = gsm8k_chain_row(raw)
    assert row == {"question": raw["question"], "cot": "<<16-3-4=9>> <<9*2=18>>", "answer": "18"}
    assert len(parse_chain(row["cot"])) == 2 and perturb_row(row) is not None
    assert gsm8k_chain_row({"question": "q", "answer": "<<1,000*2=2,000>>2,000\n#### 2,000"})["cot"] == "<<1000*2=2000>>"
    assert gsm8k_chain_row({"question": "q", "answer": "<<2*3=6>>\n#### 7"}) is None          # last result must equal the gold
    assert gsm8k_chain_row({"question": "q", "answer": "no annotations\n#### 7"}) is None
    assert gsm8k_chain_row({"question": "q", "answer": "<<2*3=6>>"}) is None                   # no gold marker
    assert gsm8k_chain_row({"question": "q", "answer": "<<5/2=2.5>>\n#### 2.5"}) is None       # integer answers only


def test_svamp_equation_becomes_one_step():
    raw = {"Body": "Robin's hair was 16 inches long. He cut off 11 inches. Then it again grew by 12 inches.",
           "Question": "How long is his hair now?", "Equation": "( ( 16.0 - 11.0 ) + 12.0 )", "Answer": 17.0}
    row = svamp_chain_row(raw)
    assert row["cot"] == "<<((16-11)+12)=17>>" and row["answer"] == "17" and row["question"].endswith("How long is his hair now?")
    assert len(parse_chain(row["cot"])) == 1 and perturb_row(row) is not None
    assert svamp_chain_row({"Body": "b", "Question": "q", "Equation": "( 5.0 / 2.0 )", "Answer": 2.5}) is None
    assert svamp_chain_row({"Body": "b", "Question": "q", "Equation": "( 2.0 - 5.0 )", "Answer": -3.0}) is None
    assert svamp_chain_row({"Body": "b", "Question": "q", "Equation": "", "Answer": 3.0}) is None


def test_eval_pool_dedupes_shuffles_deterministically_and_hashes():
    raws = [{"question": f"There are {n} apples and 2 pears. How many fruits?", "answer": f"<<{n}+2={n + 2}>>\n#### {n + 2}"} for n in range(3, 23)]
    raws.append(dict(raws[0]))                       # duplicate question
    raws.append({"question": "bad", "answer": "<<1+1=2>>\n#### 3"})
    prepare = lambda r: None if r["question"].startswith("There are 3 ") else dict(r, gold=r["answer"])
    rows, meta = eval_pool("gsm8k_test", raws, prepare=prepare, seed=7)
    assert meta["raw_rows"] == 22 and meta["chain_rows"] == 20 and meta["n"] == 19
    assert meta["steps_histogram"] == {"1": 19} and meta["max_steps"] == 1 and meta["long_profile"] is True
    again, meta2 = eval_pool("gsm8k_test", raws, prepare=prepare, seed=7)
    assert [r["question"] for r in again] == [r["question"] for r in rows] and meta2["hash"] == meta["hash"]
    other, meta3 = eval_pool("gsm8k_test", raws, prepare=prepare, seed=8)
    assert meta3["hash"] == meta["hash"] or [r["question"] for r in other] != [r["question"] for r in rows]
    assert meta["hash"] == sha_of([normalized_question(r["question"]) for r in rows])
    assert steps_histogram([{"cot": "<<1+1=2>> <<2+1=3>>"}, {"cot": "<<1+1=2>>"}]) == {"1": 1, "2": 1}
    assert chain_rows("svamp", [{"Body": "b", "Question": "q", "Equation": "( 1.0 + 1.0 )", "Answer": 2.0}])[0]["cot"] == "<<(1+1)=2>>"


def test_unknown_pool_is_rejected():
    with pytest.raises(ValueError):
        pool_rows("nope", data_cfg=None, tokenizer=None, bot_token_id=0)


def test_asdiv_formula_becomes_a_stepwise_chain():
    assert formula_chain("(16-3-4)*2=18") == "<<16-3=13>> <<13-4=9>> <<9*2=18>>"
    assert formula_chain("7+2=9") == "<<7+2=9>>" and formula_chain("3*(4+5)=27") == "<<4+5=9>> <<3*9=27>>"
    assert formula_chain("5/2=2.5") is None and formula_chain("2-5=-3") is None and formula_chain("abc") is None
    row = asdiv_chain_row({"body": "Tom has 16 apples. He eats 3 and gives away 4.", "question": "How many does he keep if he doubles the rest?",
                           "formula": "(16-3-4)*2=18", "answer": "18 (apples)", "solution_type": "Multi-step"})
    assert row["cot"] == "<<16-3=13>> <<13-4=9>> <<9*2=18>>" and row["answer"] == "18" and row["question"].startswith("Tom has")
    assert len(parse_chain(row["cot"])) == 3 and perturb_row(row) is not None
    assert asdiv_chain_row({"body": "b", "question": "q", "formula": "5/2=2.5", "answer": "2.5 (units)"}) is None
    assert POOLS["asdiv"]["adapter"] == "asdiv" and POOLS["asdiv"]["spec"]["hf_id"] == "EleutherAI/asdiv"
