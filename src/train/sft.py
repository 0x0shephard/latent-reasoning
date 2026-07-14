"""Supervised fine-tuning baselines: No-CoT-SFT and CoT-SFT (proposal §5.4).

Same backbone, tokenizer, prompt format, and step-deterministic batching as every other
method — the two baselines differ only in whether the CoT is included in the target and
whether the answer cue sits in the (masked) prompt or the (supervised) completion.

Returns (model, optimizer, step_fn); the generic Trainer owns the loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.data.datasets import load_train_set
from src.data.prompts import PromptStyle, cot_eval_prompt, eval_prompt, sft_target
from src.train.batching import StepBatcher, build_labels
from src.utils.config import load_config

SFT_METHODS = {"cot_sft", "nocot_sft"}


@dataclass(frozen=True)
class EncodedSFTExample:
    input_ids: list[int]
    labels: list[int]
    truncated_reasoning: bool


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def texts_for(row: dict, method: str, style: PromptStyle) -> tuple[str, str]:
    """(prompt, full) for one example.

    - cot_sft:   prompt = 'Question: ..\\n';                completion adds CoT + cue + answer
    - nocot_sft: prompt = 'Question: ..\\nThe answer is:';  completion adds ' <answer>'
    """
    if method not in SFT_METHODS:
        raise ValueError(f"unknown SFT method {method!r}; expected one of {sorted(SFT_METHODS)}")

    q = row["question"]
    ans = str(row["answer"])
    if method == "cot_sft":
        return cot_eval_prompt(q, style), sft_target(q, ans, style, cot=row.get("cot"))
    return eval_prompt(q, style), sft_target(q, ans, style, cot=None)


def _encode_text(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def encode_sft_example(
    tok,
    row: dict,
    method: str,
    style: PromptStyle,
    max_length: int,
) -> EncodedSFTExample:
    """Tokenize one example while always retaining the answer and EOS.

    For an over-length CoT, only the reasoning is shortened. The question/prompt and the
    complete answer span remain present, avoiding the old failure mode where right
    truncation silently removed the answer supervision.
    """
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1")
    if tok.eos_token_id is None:
        raise ValueError("the tokenizer must define eos_token_id")

    prompt, full = texts_for(row, method, style)
    if not full.startswith(prompt):
        raise ValueError("SFT prompt is not a prefix of the supervised sequence")

    if method == "cot_sft":
        cue_start = full.rfind(style.answer_prefix)
        if cue_start < len(prompt):
            raise ValueError("CoT target is missing the configured answer cue")
        # Attach whitespace before the cue to the answer span. This preserves the usual
        # GPT-style leading-space token at the reasoning/answer boundary.
        answer_start = cue_start
        while answer_start > len(prompt) and full[answer_start - 1].isspace():
            answer_start -= 1
        reasoning_text = full[len(prompt) : answer_start]
        answer_text = full[answer_start:]
    else:
        reasoning_text = ""
        answer_text = full[len(prompt) :]

    prompt_ids = _encode_text(tok, prompt)
    reasoning_ids = _encode_text(tok, reasoning_text)
    answer_ids = _encode_text(tok, answer_text)
    if not answer_ids or answer_ids[-1] != tok.eos_token_id:
        answer_ids.append(tok.eos_token_id)

    required = len(prompt_ids) + len(answer_ids)
    if required > max_length:
        raise ValueError(
            "max_length cannot retain the prompt and answer "
            f"({required} required, {max_length} configured)"
        )

    reasoning_budget = max_length - required
    kept_reasoning = reasoning_ids[:reasoning_budget]
    truncated = len(kept_reasoning) != len(reasoning_ids)
    input_ids = prompt_ids + kept_reasoning + answer_ids
    labels = build_labels(input_ids, len(prompt_ids))
    return EncodedSFTExample(input_ids, labels, truncated)


def collate_sft_rows(tok, rows, method: str, style: PromptStyle, max_length: int):
    encoded = [encode_sft_example(tok, row, method, style, max_length) for row in rows]
    width = max(len(x.input_ids) for x in encoded)
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise ValueError("the tokenizer must define pad_token_id")

    batch_ids, batch_lab, batch_att = [], [], []
    for example in encoded:
        pad = width - len(example.input_ids)
        batch_ids.append(example.input_ids + [pad_id] * pad)
        batch_lab.append(example.labels + [-100] * pad)
        batch_att.append([1] * len(example.input_ids) + [0] * pad)
    return (
        torch.tensor(batch_ids, dtype=torch.long),
        torch.tensor(batch_att, dtype=torch.long),
        torch.tensor(batch_lab, dtype=torch.long),
        sum(x.truncated_reasoning for x in encoded),
    )


def resolve_total_steps(cfg, num_examples: int) -> int:
    """Resolve an explicit step cap or derive it from the configured epoch count."""
    configured_steps = cfg.train.get("total_steps")
    if configured_steps is not None:
        total_steps = int(configured_steps)
    else:
        epochs = float(cfg.train.get("epochs", 1.0))
        if epochs <= 0:
            raise ValueError("train.epochs must be positive")
        total_steps = math.ceil(num_examples * epochs / cfg.train.batch_size)
        cfg["train"]["total_steps"] = total_steps
    if total_steps <= 0:
        raise ValueError("train.total_steps must be positive")
    return total_steps


def build_sft_task(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tcfg = cfg.task
    method = tcfg.get("method", "cot_sft")
    if method not in SFT_METHODS:
        raise ValueError(f"unknown SFT method {method!r}; expected one of {sorted(SFT_METHODS)}")
    max_len = tcfg.get("max_length", 256)
    if not tcfg.get("mask_prompt", True):
        raise ValueError("Phase 1 requires mask_prompt=true for a controlled completion loss")

    data_cfg = load_config(cfg.data_config)
    style = PromptStyle.from_config(data_cfg["prompt"])
    ds = load_train_set(data_cfg, tcfg.get("trace_style", "eq_only"))
    total_steps = resolve_total_steps(cfg, len(ds))

    revision = tcfg.get("backbone_revision")
    pretrained_kwargs = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(tcfg.backbone, **pretrained_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = _device()
    model = AutoModelForCausalLM.from_pretrained(tcfg.backbone, **pretrained_kwargs).to(device)
    # Resolved artifact identities become part of the Trainer manifest/fingerprint.
    cfg["task"]["resolved_backbone_revision"] = (
        getattr(model.config, "_commit_hash", None) or revision or "unresolved"
    )
    cfg["task"]["train_dataset_fingerprint"] = getattr(ds, "_fingerprint", "unavailable")
    cfg["task"]["train_examples"] = len(ds)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.get("weight_decay", 0.0),
    )
    batcher = StepBatcher(len(ds), cfg.train.batch_size, seed=cfg.seed)
    effective_epochs = total_steps * cfg.train.batch_size / len(ds)
    print(
        f"[data] train_examples={len(ds)} method={method} max_length={max_len} "
        f"planned_epochs={effective_epochs:.3f}"
    )

    base_lr = cfg.train.lr
    warmup = cfg.train.get("warmup_steps", 0)
    grad_clip = cfg.train.get("grad_clip", 0.0)

    def lr_at(step: int) -> float:
        if warmup > 0 and step < warmup:
            return base_lr * (step + 1) / warmup
        return base_lr

    truncation_stats = {"seen": 0, "truncated": 0}

    def step_fn(step: int) -> float:
        for pg in optimizer.param_groups:
            pg["lr"] = lr_at(step)
        rows = [ds[int(i)] for i in batcher.batch_indices(step)]
        input_ids, attn, labels, truncated = collate_sft_rows(
            tok, rows, method, style, max_len
        )
        truncation_stats["seen"] += len(rows)
        truncation_stats["truncated"] += truncated
        log_every = cfg.train.get("log_every", cfg.train.ckpt_every)
        if (step + 1) % log_every == 0:
            rate = truncation_stats["truncated"] / truncation_stats["seen"]
            print(
                f"[data] reasoning_truncated={truncation_stats['truncated']}/"
                f"{truncation_stats['seen']} ({rate:.2%})"
            )
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        out.loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return float(out.loss.detach())

    return model, optimizer, step_fn
