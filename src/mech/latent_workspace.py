"""Read CODI's latent workspace in the model's own vocabulary.

The §53 exploration found that decoding each latent thought's final state through
the frozen readout recovers the gold solution's intermediate values far above a
matched null, in a strict alternating slot structure, with a large correct-versus-
wrong gap. This module holds the frozen instruments for the preregistered
confirmation: solution parsing, top-K thought decoding, recovery scoring, the
seeded derangement null, the slot table, the (descriptive) step alignment table,
and the wrong-answer tracing indicators.

Everything here is deterministic CPU arithmetic on the completed §52 trajectory
export. No model forward pass, no weight update, no intervention.
"""
from __future__ import annotations

import re

import torch


LATENT_WORKSPACE_SCHEMA_VERSION = 1
LATENT_WORKSPACE_CONTRACT = "frozen_checkpoint_latent_workspace_confirmation_v1"

#: Thought states are decoded at the ln_f output, the same state the released
#: decoder consumes, so the lens is the model's own readout rather than a probe.
WORKSPACE_STATE = 12
WORKSPACE_TOP_K = 5

_INTERMEDIATE = re.compile(r"<<[^<>]*=([\-0-9.,]+)>>")
_FINAL = re.compile(r"####\s*([\-0-9.,]+)")
_NUMERIC_TOKEN = re.compile(r"^-?\d+$")


def normalize_value(value: str) -> str:
    value = value.replace(",", "").rstrip(".")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def parse_solution(answer_text: str) -> dict:
    """Intermediate values and the final answer of one GSM8K solution."""
    final = _FINAL.search(answer_text)
    if final is None:
        raise ValueError("GSM8K solution has no #### final answer")
    return {
        "intermediates": [
            normalize_value(value) for value in _INTERMEDIATE.findall(answer_text)
        ],
        "final": normalize_value(final.group(1)),
    }


def decode_thought_numbers(
    trajectory: torch.Tensor,
    readout: torch.Tensor,
    tokenizer,
    *,
    state: int = WORKSPACE_STATE,
    top_k: int = WORKSPACE_TOP_K,
    chunk: int = 512,
) -> list[list[set[str]]]:
    """Numeric strings among each thought's top-K readout tokens.

    Returns ``[question][thought] -> set of numeric token strings``. A value
    longer than one GPT-2 token can never match exactly, so recovery is a
    conservative lower bound by construction.
    """
    if trajectory.ndim != 4:
        raise ValueError("trajectory must be [N, positions, states, hidden]")
    count, positions = trajectory.shape[0], trajectory.shape[1]
    flat = trajectory[:, :, state, :].reshape(count * positions, -1).float()
    tops = torch.empty(count * positions, top_k, dtype=torch.long)
    matrix = readout.float()
    for start in range(0, flat.shape[0], chunk):
        logits = flat[start : start + chunk] @ matrix.T
        tops[start : start + chunk] = logits.topk(top_k, dim=1).indices
    text = {
        token: tokenizer.decode([token]).strip()
        for token in tops.unique().tolist()
    }
    return [
        [
            {
                text[tops[row * positions + position, k].item()]
                for k in range(top_k)
                if _NUMERIC_TOKEN.match(
                    text[tops[row * positions + position, k].item()]
                )
            }
            for position in range(positions)
        ]
        for row in range(count)
    ]


def recovery_fraction(thought_numbers: list[set[str]], targets: list[str]) -> float:
    """Fraction of target values present in the union of thought numbers."""
    if not targets:
        raise ValueError("recovery is undefined without targets")
    union = set().union(*thought_numbers) if thought_numbers else set()
    return sum(1 for value in targets if value in union) / len(targets)


def per_thought_hits(
    thought_numbers: list[set[str]], targets: list[str]
) -> list[bool]:
    """Whether any target value appears at each individual thought."""
    return [
        any(value in numbers for value in targets) for numbers in thought_numbers
    ]


def seeded_derangement(count: int, *, seed: int) -> torch.Tensor:
    """A deterministic permutation with no fixed points."""
    if count < 2:
        raise ValueError("a derangement needs at least two rows")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(int(seed)))
    # Rotating a random order by one guarantees no row keeps its own targets.
    permutation = torch.empty(count, dtype=torch.long)
    permutation[order] = order.roll(1)
    return permutation


def alignment_table(
    thought_numbers_by_row: list[list[set[str]]],
    intermediates_by_row: list[list[str]],
    rows: list[int],
    *,
    value_slots: tuple[int, ...] = (1, 3, 5),
) -> list[dict]:
    """Descriptive only: does intermediate k sit at value slot k?

    The §53 exploration measured that it does not; the table is preregistered as
    a reported quantity with no gate attached.
    """
    table = []
    for step, slot in enumerate(value_slots):
        eligible = aligned = elsewhere = 0
        for row in rows:
            intermediates = intermediates_by_row[row]
            if len(intermediates) <= step:
                continue
            eligible += 1
            value = intermediates[step]
            if value in thought_numbers_by_row[row][slot]:
                aligned += 1
            if any(
                value in thought_numbers_by_row[row][other]
                for other in value_slots
                if other != slot
            ):
                elsewhere += 1
        table.append(
            {
                "step": step,
                "slot": slot,
                "eligible": eligible,
                "aligned_rate": aligned / eligible if eligible else None,
                "other_slot_rate": elsewhere / eligible if eligible else None,
            }
        )
    return table
