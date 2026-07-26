"""Exact training-sequence reconstruction for the released CODI GPT-2 model.

The public CODI checkpoint was trained with a sequence contract that differs from this
repository's pilot implementation.  This module mirrors the released ``train.py`` for
the GPT-2 ``icot`` setting at source revision
``2c2314662c63e9f482ebc46614ffe9af17a241e5``:

* the question is used without an added prompt;
* the final whitespace-separated equation token is removed from the teacher CoT;
* question, CoT, and answer are tokenized independently with a 256-token cap;
* the teacher sees ``question + truncated CoT + "The answer is: N" + EOS``;
* the student sees a left-padded ``question + BOT`` prefix.

The extra trace and answer boundaries are observational metadata for the KV diagnostic.
They do not alter the checkpoint or its generation path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


OFFICIAL_CODI_SOURCE_REVISION = "2c2314662c63e9f482ebc46614ffe9af17a241e5"
OFFICIAL_SEGMENT_MAX_LENGTH = 256
OFFICIAL_MAX_TOKEN_FILTER = 1_000
OFFICIAL_ANSWER_PROMPT = "The answer is:"


@dataclass(frozen=True)
class OfficialCODIFormattedRow:
    question: str
    cot: str
    answer: str


@dataclass(frozen=True)
class OfficialCODIEncodedRow:
    student_question_ids: list[int]
    teacher_ids: list[int]
    teacher_trace_start: int
    teacher_trace_end: int
    teacher_answer_start: int
    teacher_endpoint: int


@dataclass(frozen=True)
class OfficialCODIKVBatch:
    student_question_ids: object
    student_question_mask: object
    teacher_ids: object
    teacher_mask: object
    teacher_trace_start: object
    teacher_trace_end: object
    teacher_answer_start: object
    teacher_endpoint: object

    def to(self, device) -> "OfficialCODIKVBatch":
        values = {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in self.__dict__.items()
        }
        return OfficialCODIKVBatch(**values)


def format_official_codi_row(row: dict) -> OfficialCODIFormattedRow:
    """Apply the public GPT-2 ``icot`` formatting and anti-shortcut rule."""
    question = str(row["question"])
    cot_parts = str(row["cot"]).split(" ")
    cot = " ".join(cot_parts[:-1])
    raw_answer = str(row["answer"]).split(" ")[-1]
    if not official_codi_answer_is_eligible(row["answer"]):
        raise ValueError("the official CODI loader drops non-digit-leading answers")
    answer = f"{OFFICIAL_ANSWER_PROMPT} {raw_answer}".replace("####", "")
    return OfficialCODIFormattedRow(question=question, cot=cot, answer=answer)


def official_codi_answer_is_eligible(answer: object) -> bool:
    raw_answer = str(answer).split(" ")[-1]
    return bool(raw_answer and raw_answer[0].isdigit())


def official_codi_row_is_eligible(row: dict) -> bool:
    """Cheap form of the released answer filter used before deterministic sampling."""
    try:
        format_official_codi_row(row)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _tokenize_segment(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=OFFICIAL_SEGMENT_MAX_LENGTH,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    values = encoded["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def encode_official_codi_row(
    tokenizer,
    row: dict,
    *,
    bot_token_id: int,
) -> OfficialCODIEncodedRow:
    """Tokenize one calibration row exactly as the released training data path."""
    formatted = format_official_codi_row(row)
    if len(
        tokenizer.encode(
            str(row["question"]) + str(row["cot"]) + str(row["answer"])
        )
    ) > OFFICIAL_MAX_TOKEN_FILTER:
        raise ValueError("row exceeds the official CODI 1,000-token training filter")

    question_ids = _tokenize_segment(tokenizer, formatted.question)
    cot_ids = _tokenize_segment(tokenizer, formatted.cot)
    answer_ids = _tokenize_segment(tokenizer, formatted.answer)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if cot_ids and bos_token_id is not None and cot_ids[0] == bos_token_id:
        cot_ids = cot_ids[1:]
        if answer_ids and answer_ids[0] == bos_token_id:
            answer_ids = answer_ids[1:]

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("the official CODI tokenizer must define eos_token_id")
    answer_ids = answer_ids + [int(eos_token_id)]
    answer_prompt_ids = _tokenize_segment(tokenizer, OFFICIAL_ANSWER_PROMPT)
    if (
        answer_prompt_ids
        and bos_token_id is not None
        and answer_prompt_ids[0] == bos_token_id
    ):
        answer_prompt_ids = answer_prompt_ids[1:]
    if not answer_prompt_ids:
        raise ValueError("official answer prompt tokenized to an empty sequence")

    trace_start = len(question_ids)
    trace_end = trace_start + len(cot_ids)
    teacher_ids = question_ids + cot_ids + answer_ids
    endpoint = trace_end + len(answer_prompt_ids) - 1
    answer_start = endpoint + 1
    if endpoint >= len(teacher_ids) or answer_start >= len(teacher_ids):
        raise RuntimeError("official CODI answer boundary reconstruction failed")
    return OfficialCODIEncodedRow(
        student_question_ids=question_ids + [int(bot_token_id)],
        teacher_ids=teacher_ids,
        teacher_trace_start=trace_start,
        teacher_trace_end=trace_end,
        teacher_answer_start=answer_start,
        teacher_endpoint=endpoint,
    )


def _pad_right(sequences: Sequence[list[int]], value: int):
    import torch

    width = max(len(sequence) for sequence in sequences)
    values = [
        sequence + [value] * (width - len(sequence)) for sequence in sequences
    ]
    masks = [
        [1] * len(sequence) + [0] * (width - len(sequence))
        for sequence in sequences
    ]
    return (
        torch.tensor(values, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


def _pad_left(sequences: Sequence[list[int]], value: int):
    import torch

    width = max(len(sequence) for sequence in sequences)
    values = [
        [value] * (width - len(sequence)) + sequence for sequence in sequences
    ]
    masks = [
        [0] * (width - len(sequence)) + [1] * len(sequence)
        for sequence in sequences
    ]
    return (
        torch.tensor(values, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


def collate_official_codi_kv_rows(
    tokenizer,
    rows: Sequence[dict],
    *,
    bot_token_id: int,
) -> OfficialCODIKVBatch:
    """Build left-padded student and right-padded teacher batches."""
    import torch

    if not rows:
        raise ValueError("cannot collate an empty official CODI batch")
    encoded = [
        encode_official_codi_row(
            tokenizer,
            row,
            bot_token_id=bot_token_id,
        )
        for row in rows
    ]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise ValueError("the official CODI tokenizer must define pad_token_id")
    student_ids, student_mask = _pad_left(
        [item.student_question_ids for item in encoded],
        int(pad_id),
    )
    teacher_ids, teacher_mask = _pad_right(
        [item.teacher_ids for item in encoded],
        int(pad_id),
    )
    return OfficialCODIKVBatch(
        student_question_ids=student_ids,
        student_question_mask=student_mask,
        teacher_ids=teacher_ids,
        teacher_mask=teacher_mask,
        teacher_trace_start=torch.tensor(
            [item.teacher_trace_start for item in encoded],
            dtype=torch.long,
        ),
        teacher_trace_end=torch.tensor(
            [item.teacher_trace_end for item in encoded],
            dtype=torch.long,
        ),
        teacher_answer_start=torch.tensor(
            [item.teacher_answer_start for item in encoded],
            dtype=torch.long,
        ),
        teacher_endpoint=torch.tensor(
            [item.teacher_endpoint for item in encoded],
            dtype=torch.long,
        ),
    )
