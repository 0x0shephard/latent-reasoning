"""Prompt formatting shared by every method, so inputs are byte-identical across the
comparison (a §5.3 fairness control). Baselines format text directly; the latent methods
(Phase 2) reuse the same prefixes but insert continuous thoughts before the answer cue.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptStyle:
    question_prefix: str = "Question: "
    cot_prefix: str = ""
    answer_prefix: str = "The answer is:"

    @classmethod
    def from_config(cls, cfg: dict) -> "PromptStyle":
        return cls(
            question_prefix=cfg.get("question_prefix", cls.question_prefix),
            cot_prefix=cfg.get("cot_prefix", cls.cot_prefix),
            answer_prefix=cfg.get("answer_prefix", cls.answer_prefix),
        )


def eval_prompt(question: str, style: PromptStyle) -> str:
    """The prompt fed to the model at eval time (no answer, ends at the answer cue)."""
    return f"{style.question_prefix}{question.strip()}\n{style.answer_prefix}"


def cot_eval_prompt(question: str, style: PromptStyle) -> str:
    """Eval prompt that invites an explicit CoT before the answer cue (CoT-SFT / teacher)."""
    return f"{style.question_prefix}{question.strip()}\n{style.cot_prefix}"


def sft_target(question: str, answer: str, style: PromptStyle,
               cot: str | None = None) -> str:
    """Full supervised sequence.

    - No-CoT-SFT:  Question: ...\nThe answer is: <ans>
    - CoT-SFT:     Question: ...\n<cot> The answer is: <ans>
    Loss is computed on the whole string; the prompt/answer boundary is returned by
    `answer_start_index` for optional prompt-masking.
    """
    q = f"{style.question_prefix}{question.strip()}\n"
    if cot:
        q += f"{style.cot_prefix}{cot.strip()} "
    return f"{q}{style.answer_prefix} {answer.strip()}"


def answer_span(question: str, answer: str, style: PromptStyle,
                cot: str | None = None) -> tuple[str, str]:
    """Return (prefix, completion) so trainers can mask loss to answer tokens only.

    prefix ends with the answer cue; completion is ' <answer>'.
    """
    full = sft_target(question, answer, style, cot)
    cue = style.answer_prefix
    idx = full.rfind(cue) + len(cue)
    return full[:idx], full[idx:]
