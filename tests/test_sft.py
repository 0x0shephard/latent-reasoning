"""Phase-1 SFT sequence construction tests (tiny fake tokenizer, no downloads)."""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.data.prompts import PromptStyle, cot_eval_prompt, eval_prompt
from src.train.sft import collate_sft_rows, encode_sft_example, resolve_total_steps, texts_for
from src.utils.config import Config


class CharTokenizer:
    eos_token_id = 0
    pad_token_id = 255

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}

    @staticmethod
    def decode(ids):
        return "".join(chr(token) for token in ids if token not in (0, 255))


STYLE = PromptStyle()


def test_nocot_keeps_answer_and_supervises_eos():
    tok = CharTokenizer()
    row = {"question": "2+2?", "answer": "4", "cot": "unused"}
    encoded = encode_sft_example(tok, row, "nocot_sft", STYLE, max_length=128)
    prompt = eval_prompt(row["question"], STYLE)

    assert tok.decode(encoded.input_ids) == prompt + " 4"
    assert encoded.input_ids[-1] == tok.eos_token_id
    assert encoded.labels[: len(prompt)] == [-100] * len(prompt)
    assert encoded.labels[-1] == tok.eos_token_id
    assert not encoded.truncated_reasoning


def test_long_cot_truncates_reasoning_but_retains_answer():
    tok = CharTokenizer()
    row = {"question": "Q?", "cot": "abcdefghij", "answer": "42"}
    prompt = cot_eval_prompt(row["question"], STYLE)
    answer_text = " The answer is: 42"
    max_length = len(prompt) + 3 + len(answer_text) + 1
    encoded = encode_sft_example(tok, row, "cot_sft", STYLE, max_length=max_length)

    decoded = tok.decode(encoded.input_ids)
    assert decoded == prompt + "abc" + answer_text
    assert decoded.endswith("The answer is: 42")
    assert len(encoded.input_ids) == max_length
    assert encoded.truncated_reasoning


def test_too_small_max_length_fails_instead_of_dropping_answer():
    tok = CharTokenizer()
    row = {"question": "A long question", "cot": "reason", "answer": "42"}
    with pytest.raises(ValueError, match="retain the prompt and answer"):
        encode_sft_example(tok, row, "cot_sft", STYLE, max_length=8)


def test_unknown_method_fails_loudly():
    with pytest.raises(ValueError, match="unknown SFT method"):
        texts_for({"question": "Q", "answer": "1"}, "typo", STYLE)


def test_collation_masks_padding_and_reports_truncation():
    tok = CharTokenizer()
    rows = [
        {"question": "Q?", "cot": "x" * 50, "answer": "1"},
        {"question": "Longer question?", "cot": "short", "answer": "2"},
    ]
    input_ids, attention, labels, truncated = collate_sft_rows(
        tok, rows, "cot_sft", STYLE, max_length=64
    )
    assert input_ids.shape == attention.shape == labels.shape
    assert truncated == 1
    assert (labels[attention == 0] == -100).all()


def test_total_steps_are_derived_from_real_dataset_size():
    cfg = Config({"train": {"epochs": 1.0, "batch_size": 16}})
    assert resolve_total_steps(cfg, 385_620) == 24_102
    assert cfg.train.total_steps == 24_102


def test_tiny_causal_lm_forward_backward_with_phase1_batch():
    tok = CharTokenizer()
    rows = [
        {"question": "2+2?", "cot": "2+2=4", "answer": "4"},
        {"question": "3+4?", "cot": "3+4=7", "answer": "7"},
    ]
    input_ids, attention, labels, _ = collate_sft_rows(
        tok, rows, "cot_sft", STYLE, max_length=64
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=256,
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=1,
            n_head=1,
        )
    )
    output = model(input_ids=input_ids, attention_mask=attention, labels=labels)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
