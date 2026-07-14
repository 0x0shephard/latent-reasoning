"""Answer extraction + exact-match — the study's measuring instrument.

Every method (baselines, CODI, KaVa) is scored through these functions, so correctness
here matters more than anywhere else. Kept dependency-free and unit-tested on CPU.

Two responsibilities:
  1. Parse a model's free-form generation into a final numeric answer.
  2. Normalize gold answers from heterogeneous eval sets and compare numerically.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional

# Matches signed integers/decimals/scientific notation with optional commas, $, and %.
# Decimal is used throughout so large GSM-Hard answers are not rounded through float.
_NUMBER_PATTERN = r"[-+]?\$?(?:(?:\d[\d,]*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?"
_NUMBER_RE = re.compile(_NUMBER_PATTERN)
_ANSWER_RE = re.compile(r"answer\s*is\s*:?\s*(" + _NUMBER_PATTERN + r")", re.IGNORECASE)


def normalize_number(token: str) -> Optional[Decimal]:
    """Turn a raw numeric token into an exact Decimal, or None.

    A trailing percent sign is treated as formatting (``12%`` -> ``12``), matching the
    math-dataset convention used by the previous evaluator rather than converting it to
    a fraction.
    """
    if token is None:
        return None
    s = str(token).strip().replace(",", "").replace("$", "").replace("%", "")
    if s in ("", "+", "-", "."):
        return None
    try:
        value = Decimal(s)
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def extract_final_number(text: str) -> Optional[Decimal]:
    """Extract the model's answer as an exact Decimal.

    Strategy: prefer the LAST valid number following an answer cue (models sometimes
    self-correct), otherwise fall back to the last number in the text.
    """
    if not text:
        return None

    # Prefer the final explicit answer cue so a later correction wins.
    cues = list(_ANSWER_RE.finditer(text))
    for cue in reversed(cues):
        val = normalize_number(cue.group(1))
        if val is not None:
            return val

    # Fall back to the last number anywhere in the generation.
    matches = _NUMBER_RE.findall(text)
    for token in reversed(matches):
        val = normalize_number(token)
        if val is not None:
            return val
    return None


def normalize_gold(raw: object, kind: str) -> Optional[Decimal]:
    """Normalize a gold answer from a given eval-set kind into an exact Decimal.

    kind matches configs/data.yaml `eval[*].kind`:
      - gsm8k_main: answer field is CoT text ending in '#### <number>'
      - svamp / multiarith / gsm_hard / aug: field is (mostly) a bare number
    """
    if raw is None:
        return None
    text = str(raw)
    if kind == "gsm8k_main" and "####" in text:
        text = text.split("####")[-1]
    return extract_final_number(text)


def answers_match(
    pred_text: str,
    gold: object,
    tol: Decimal | float | str | None = None,
) -> bool:
    """Compare a generated answer with gold.

    The default is true numeric exact-match. An optional *absolute* tolerance is retained
    for explicit diagnostic use; unlike the old relative tolerance, it never grows with
    GSM-Hard answer magnitude.
    """
    if gold is None:
        return False
    pred = extract_final_number(pred_text)
    gold_value = normalize_number(str(gold))
    if pred is None or gold_value is None:
        return False
    if tol is None:
        return pred == gold_value
    tolerance = Decimal(str(tol))
    if tolerance < 0:
        raise ValueError("tol must be non-negative")
    return abs(pred - gold_value) <= tolerance
