from __future__ import annotations

import pytest

from src.data.official_codi_training import (
    OFFICIAL_ANSWER_PROMPT,
    align_official_codi_gsm8k_eval_rows,
    encode_official_codi_row,
    format_official_codi_row,
    official_codi_row_is_eligible,
)


def test_gsm8k_eval_alignment_restores_teacher_fields_and_checks_gold():
    rows = align_official_codi_gsm8k_eval_rows(
        [
            {
                "question": "How many widgets?",
                "answer": "First compute twelve.\nThen total it.\n#### 1,234",
            }
        ],
        [{"question": "How many widgets?", "gold": 1234}],
    )
    assert rows == [
        {
            "question": "How many widgets?",
            "cot": "First compute twelve.\nThen total it.",
            "answer": "1234",
            "gold": 1234,
        }
    ]


def test_gsm8k_eval_alignment_rejects_order_or_gold_drift():
    raw = [{"question": "Question A", "answer": "reason\n#### 7"}]
    with pytest.raises(ValueError, match="question mismatch"):
        align_official_codi_gsm8k_eval_rows(
            raw, [{"question": "Question B", "gold": 7}]
        )
    with pytest.raises(ValueError, match="gold mismatch"):
        align_official_codi_gsm8k_eval_rows(
            raw, [{"question": "Question A", "gold": 8}]
        )


def test_gsm8k_eval_alignment_validates_only_requested_evaluation_prefix():
    rows = align_official_codi_gsm8k_eval_rows(
        [
            {"question": "Used", "answer": "reason\n#### 7"},
            {"question": "Not used", "answer": "reason\n#### -1"},
        ],
        [
            {"question": "Used", "gold": 7},
            {"question": "Not used", "gold": -1},
        ],
        examples=1,
    )
    assert [row["question"] for row in rows] == ["Used"]


class CharacterTokenizer:
    bos_token_id = None
    eos_token_id = 99_999
    pad_token_id = 50_257

    def __call__(self, text, **kwargs):
        limit = int(kwargs.get("max_length", 10_000))
        return {"input_ids": [ord(character) for character in text][:limit]}

    def encode(self, text):
        return [ord(character) for character in text]


def test_official_icot_format_drops_final_whitespace_token():
    row = {
        "question": "What is 2 + 3?",
        "cot": "<<2+2=4>> <<4+1=5>>",
        "answer": "#### 5",
    }
    formatted = format_official_codi_row(row)
    assert formatted.question == row["question"]
    assert formatted.cot == "<<2+2=4>>"
    assert formatted.answer == "The answer is: 5"


def test_official_icot_filter_matches_digit_leading_rule():
    assert official_codi_row_is_eligible(
        {"question": "q", "cot": "a b", "answer": "#### 7"}
    )
    assert not official_codi_row_is_eligible(
        {"question": "q", "cot": "a b", "answer": "#### -7"}
    )
    with pytest.raises(ValueError, match="non-digit-leading"):
        format_official_codi_row(
            {"question": "q", "cot": "a b", "answer": "#### -7"}
        )


def test_official_kv_boundaries_are_segment_exact():
    tokenizer = CharacterTokenizer()
    row = {
        "question": "Q",
        "cot": "first second",
        "answer": "#### 9",
    }
    encoded = encode_official_codi_row(
        tokenizer,
        row,
        bot_token_id=50_258,
    )
    question = [ord("Q")]
    cot = [ord(character) for character in "first"]
    answer = [
        ord(character)
        for character in f"{OFFICIAL_ANSWER_PROMPT} 9"
    ] + [tokenizer.eos_token_id]
    assert encoded.student_question_ids == question + [50_258]
    assert encoded.teacher_ids == question + cot + answer
    assert encoded.teacher_trace_start == len(question)
    assert encoded.teacher_trace_end == len(question) + len(cot)
    assert encoded.teacher_endpoint == (
        encoded.teacher_trace_end + len(OFFICIAL_ANSWER_PROMPT) - 1
    )
    assert encoded.teacher_answer_start == encoded.teacher_endpoint + 1
