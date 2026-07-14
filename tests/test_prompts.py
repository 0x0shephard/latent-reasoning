"""Unit tests for prompt formatting + adapter normalization (CPU, no downloads)."""
from __future__ import annotations

from src.data.answer_extract import normalize_gold
from src.data.datasets import ADAPTERS
from src.data.prompts import (
    PromptStyle,
    answer_span,
    eval_prompt,
    sft_target,
)

STYLE = PromptStyle()


def test_eval_prompt_ends_at_cue():
    p = eval_prompt("What is 2+2?", STYLE)
    assert p == "Question: What is 2+2?\nThe answer is:"


def test_sft_target_no_cot():
    t = sft_target("What is 2+2?", "4", STYLE)
    assert t == "Question: What is 2+2?\nThe answer is: 4"


def test_sft_target_with_cot():
    t = sft_target("Q?", "360", STYLE, cot="<<600-240=360>>")
    assert t == "Question: Q?\n<<600-240=360>> The answer is: 360"


def test_answer_span_splits_at_cue():
    prefix, completion = answer_span("Q?", "360", STYLE, cot="<<600-240=360>>")
    assert prefix.endswith("The answer is:")
    assert completion == " 360"
    assert prefix + completion == sft_target("Q?", "360", STYLE, cot="<<600-240=360>>")


def test_svamp_adapter_concats_body_and_question():
    q, gold_raw = ADAPTERS["svamp"]({"Body": "There are 5 apples.", "Question": "How many?", "Answer": 5})
    assert q == "There are 5 apples. How many?"
    assert normalize_gold(gold_raw, "svamp") == 5.0


def test_svamp_adapter_allows_missing_optional_body():
    q, gold_raw = ADAPTERS["svamp"]({"Question": "How many?", "Answer": 5})
    assert q == "How many?"
    assert normalize_gold(gold_raw, "svamp") == 5.0


def test_gsm_hard_adapter_uses_input_target():
    q, gold_raw = ADAPTERS["gsm_hard"]({"input": "big problem", "target": 123456.0, "code": "..."})
    assert q == "big problem"
    assert normalize_gold(gold_raw, "gsm_hard") == 123456.0


def test_gsm8k_main_adapter():
    q, gold_raw = ADAPTERS["gsm8k_main"]({"question": "Q", "answer": "reasoning #### 42"})
    assert q == "Q"
    assert normalize_gold(gold_raw, "gsm8k_main") == 42.0
