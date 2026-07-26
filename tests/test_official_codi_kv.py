from __future__ import annotations

import pytest

from src.data.official_codi_training import (
    OFFICIAL_ANSWER_PROMPT,
    encode_official_codi_row,
    format_official_codi_row,
    official_codi_row_is_eligible,
)


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
