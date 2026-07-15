"""Phase-2 sequence construction and explicit-CoT teacher-target extraction.

The student sees ``question + <bot> + M continuous states + <eot> + answer cue``.
The teacher sees the same question followed by an explicit reasoning trace and answer.
The final reasoning step is removed before tokenization (CODI's anti-shortcut rule).

All token boundaries are recorded during encoding.  Training therefore never guesses the
distillation position by searching decoded text, and right padding cannot move a target.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch

from src.data.prompts import PromptStyle


_EQUATION_STEP = re.compile(r"\s*<<[^<>]*>>\s*")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EncodedLatentExample:
    question_ids: list[int]
    student_segment_ids: list[int]
    student_labels: list[int]
    student_endpoint: int
    teacher_ids: list[int]
    teacher_labels: list[int]
    teacher_endpoint: int
    teacher_trace_start: int
    teacher_trace_end: int
    teacher_answer_start: int
    reasoning_truncated: bool


@dataclass(frozen=True)
class LatentBatch:
    question_ids: torch.Tensor
    question_mask: torch.Tensor
    student_segment_ids: torch.Tensor
    student_segment_mask: torch.Tensor
    student_labels: torch.Tensor
    student_endpoint: torch.Tensor
    teacher_ids: torch.Tensor
    teacher_mask: torch.Tensor
    teacher_labels: torch.Tensor
    teacher_endpoint: torch.Tensor
    teacher_trace_start: torch.Tensor
    teacher_trace_end: torch.Tensor
    teacher_answer_start: torch.Tensor
    reasoning_truncated: int

    def to(self, device: torch.device | str) -> "LatentBatch":
        values = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return LatentBatch(**values)


@dataclass(frozen=True)
class TeacherTargets:
    """Detached targets aligned to explicit trace tokens.

    Shapes:
      hidden_endpoint: [batch, layers, hidden]
      trace_keys/values: [batch, layers, heads, trace_tokens, head_dim]
      importance: [batch, layers, heads, trace_tokens]
      trace_mask: [batch, trace_tokens]
    """

    hidden_endpoint: torch.Tensor
    trace_keys: torch.Tensor
    trace_values: torch.Tensor
    importance: torch.Tensor
    trace_mask: torch.Tensor


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def drop_last_reasoning_step(cot: str, trace_style: str = "eq_only") -> str:
    """Remove the final explicit reasoning step without removing the final answer field.

    Equation-only traces are sequences of ``<<...>>`` blocks.  Natural-language traces
    use the final non-empty line or sentence as a step.  A one-step trace becomes empty;
    this is preferable to leaking the final computation directly to the student target.
    """
    text = str(cot or "").strip()
    if not text:
        return ""
    if trace_style == "eq_only" or "<<" in text:
        matches = list(_EQUATION_STEP.finditer(text))
        if matches:
            last = matches[-1]
            return (text[: last.start()] + text[last.end() :]).strip()
    if trace_style not in {"eq_only", "natural_language"}:
        raise ValueError(f"unknown trace_style {trace_style!r}")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return "\n".join(lines[:-1]).strip()
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    return " ".join(sentences[:-1]).strip() if len(sentences) > 1 else ""


def encode_latent_example(
    tokenizer,
    row: dict,
    style: PromptStyle,
    *,
    bot_token_id: int,
    eot_token_id: int,
    trace_style: str,
    max_length: int,
    latent_steps: int,
) -> EncodedLatentExample:
    """Encode one matched teacher/student example and retain the full answer.

    ``max_length`` constrains both the explicit teacher sequence and the student's total
    context after inserting ``latent_steps`` continuous tokens.  Only reasoning tokens may
    be truncated.
    """
    if max_length <= latent_steps + 2:
        raise ValueError("max_length is too small for the configured latent budget")
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer must define eos_token_id")

    question_text = f"{style.question_prefix}{str(row['question']).strip()}\n"
    question_plain = _encode(tokenizer, question_text)
    student_question = question_plain + [bot_token_id]
    # A leading space gives GPT-style tokenizers the same natural answer-cue boundary used
    # by CoT-SFT ("...reasoning The answer is:").
    student_cue_ids = _encode(tokenizer, f" {style.answer_prefix}")
    if not student_cue_ids:
        raise ValueError("answer_prefix must tokenize to at least one token")
    answer_ids = _encode(tokenizer, f" {str(row['answer']).strip()}")
    if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
        answer_ids.append(tokenizer.eos_token_id)

    # The discrete segment begins after the continuous slots.  The cue endpoint is the
    # final token of "The answer is:", matching CODI's colon position for GPT-2.
    student_prefix = [eot_token_id] + student_cue_ids
    student_segment = student_prefix + answer_ids
    student_labels = [-100] * len(student_prefix) + answer_ids
    student_total = len(student_question) + latent_steps + len(student_segment)
    if student_total > max_length:
        raise ValueError(
            "max_length cannot retain the student question, latent slots, and answer "
            f"({student_total} required, {max_length} configured)"
        )

    reasoning = drop_last_reasoning_step(str(row.get("cot", "")), trace_style)
    trace_text = f"{style.cot_prefix}{reasoning.strip()}" if reasoning else ""
    trace_ids = _encode(tokenizer, trace_text)
    teacher_cue_ids = _encode(
        tokenizer, f" {style.answer_prefix}" if trace_ids else style.answer_prefix
    )
    required_teacher = len(question_plain) + len(teacher_cue_ids) + len(answer_ids)
    if required_teacher > max_length:
        raise ValueError(
            "max_length cannot retain the teacher question and answer "
            f"({required_teacher} required, {max_length} configured)"
        )
    trace_budget = max_length - required_teacher
    kept_trace = trace_ids[:trace_budget]
    truncated = len(kept_trace) != len(trace_ids)
    teacher_ids = question_plain + kept_trace + teacher_cue_ids + answer_ids
    trace_start = len(question_plain)
    trace_end = trace_start + len(kept_trace)
    endpoint = trace_end + len(teacher_cue_ids) - 1
    answer_start = endpoint + 1
    teacher_labels = [-100] * len(question_plain) + teacher_ids[len(question_plain) :]

    return EncodedLatentExample(
        question_ids=student_question,
        student_segment_ids=student_segment,
        student_labels=student_labels,
        student_endpoint=len(student_prefix) - 1,
        teacher_ids=teacher_ids,
        teacher_labels=teacher_labels,
        teacher_endpoint=endpoint,
        teacher_trace_start=trace_start,
        teacher_trace_end=trace_end,
        teacher_answer_start=answer_start,
        reasoning_truncated=truncated,
    )


def _pad(sequences: Sequence[list[int]], value: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    values, masks = [], []
    for sequence in sequences:
        amount = width - len(sequence)
        values.append(sequence + [value] * amount)
        masks.append([1] * len(sequence) + [0] * amount)
    return torch.tensor(values, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def collate_latent_rows(
    tokenizer,
    rows: Sequence[dict],
    style: PromptStyle,
    *,
    bot_token_id: int,
    eot_token_id: int,
    trace_style: str,
    max_length: int,
    latent_steps: int,
) -> LatentBatch:
    if not rows:
        raise ValueError("cannot collate an empty latent batch")
    encoded = [
        encode_latent_example(
            tokenizer,
            row,
            style,
            bot_token_id=bot_token_id,
            eot_token_id=eot_token_id,
            trace_style=trace_style,
            max_length=max_length,
            latent_steps=latent_steps,
        )
        for row in rows
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("the tokenizer must define pad_token_id")
    question_ids, question_mask = _pad([item.question_ids for item in encoded], pad_id)
    student_ids, student_mask = _pad(
        [item.student_segment_ids for item in encoded], pad_id
    )
    student_labels, _ = _pad([item.student_labels for item in encoded], -100)
    teacher_ids, teacher_mask = _pad([item.teacher_ids for item in encoded], pad_id)
    teacher_labels, _ = _pad([item.teacher_labels for item in encoded], -100)

    return LatentBatch(
        question_ids=question_ids,
        question_mask=question_mask,
        student_segment_ids=student_ids,
        student_segment_mask=student_mask,
        student_labels=student_labels,
        student_endpoint=torch.tensor([item.student_endpoint for item in encoded]),
        teacher_ids=teacher_ids,
        teacher_mask=teacher_mask,
        teacher_labels=teacher_labels,
        teacher_endpoint=torch.tensor([item.teacher_endpoint for item in encoded]),
        teacher_trace_start=torch.tensor([item.teacher_trace_start for item in encoded]),
        teacher_trace_end=torch.tensor([item.teacher_trace_end for item in encoded]),
        teacher_answer_start=torch.tensor([item.teacher_answer_start for item in encoded]),
        reasoning_truncated=sum(item.reasoning_truncated for item in encoded),
    )


def cache_to_tensors(cache) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Transformers v4 tuple caches or v5 ``DynamicCache`` to stacked tensors."""
    if hasattr(cache, "layers"):
        keys = [layer.keys for layer in cache.layers]
        values = [layer.values for layer in cache.layers]
    else:
        keys = [layer[0] for layer in cache]
        values = [layer[1] for layer in cache]
    if not keys or any(value is None for value in keys + values):
        raise ValueError("model did not return a materialized key/value cache")
    return torch.stack(keys, dim=1), torch.stack(values, dim=1)


def extract_teacher_hidden(outputs, endpoints: torch.Tensor) -> torch.Tensor:
    """Gather detached all-block hidden states at each example's answer-cue endpoint."""
    if not outputs.hidden_states or len(outputs.hidden_states) < 2:
        raise ValueError("teacher forward must request output_hidden_states=True")
    hidden = torch.stack(outputs.hidden_states[1:], dim=1)
    row = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[row, :, endpoints, :].detach()


def extract_teacher_targets(outputs, batch: LatentBatch) -> TeacherTargets:
    """Extract detached all-layer endpoint states and trace KV/importance tensors."""
    if not outputs.hidden_states or len(outputs.hidden_states) < 2:
        raise ValueError("teacher forward must request output_hidden_states=True")
    if not outputs.attentions:
        raise ValueError(
            "teacher forward returned no attentions; set attn_implementation='eager'"
        )
    keys, values = cache_to_tensors(outputs.past_key_values)
    # hidden_states[0] is the embedding output; one state per transformer block follows.
    hidden = torch.stack(outputs.hidden_states[1:], dim=1)
    batch_size, layers, heads, _, head_dim = keys.shape
    max_trace = max(1, int((batch.teacher_trace_end - batch.teacher_trace_start).max()))
    trace_keys = keys.new_zeros((batch_size, layers, heads, max_trace, head_dim))
    trace_values = values.new_zeros(trace_keys.shape)
    importance = keys.new_zeros((batch_size, layers, heads, max_trace))
    trace_mask = torch.zeros((batch_size, max_trace), dtype=torch.bool, device=keys.device)
    endpoint_states = []

    attentions = torch.stack(outputs.attentions, dim=1)  # [B,L,H,Q,K]
    for index in range(batch_size):
        start = int(batch.teacher_trace_start[index])
        end = int(batch.teacher_trace_end[index])
        endpoint = int(batch.teacher_endpoint[index])
        answer_start = int(batch.teacher_answer_start[index])
        sequence_end = int(batch.teacher_mask[index].sum())
        endpoint_states.append(hidden[index, :, endpoint, :])
        count = end - start
        if count <= 0:
            continue
        trace_keys[index, :, :, :count] = keys[index, :, :, start:end]
        trace_values[index, :, :, :count] = values[index, :, :, start:end]
        trace_mask[index, :count] = True
        # Average the already-normalized causal attention paid by answer tokens to each
        # trace token.  Renormalize over the trace because some mass attends to the prompt.
        rows = attentions[index, :, :, answer_start:sequence_end, start:end]
        if rows.shape[-2] == 0:
            rows = attentions[index, :, :, endpoint : endpoint + 1, start:end]
        token_importance = rows.mean(dim=-2)
        token_importance = token_importance / token_importance.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        importance[index, :, :, :count] = token_importance

    return TeacherTargets(
        hidden_endpoint=torch.stack(endpoint_states).detach(),
        trace_keys=trace_keys.detach(),
        trace_values=trace_values.detach(),
        importance=importance.detach(),
        trace_mask=trace_mask,
    )
