"""Answer extraction + exact-match — the study's measuring instrument.

Every method (baselines, CODI, KaVa) is scored through these functions, so correctness
here matters more than anywhere else. Kept dependency-free and unit-tested on CPU.

Two responsibilities:
  1. Parse a model's free-form generation into a final numeric answer.
  2. Normalize gold answers from heterogeneous eval sets and compare numerically.
"""
from __future__ import annotations

import re
from typing import Optional

# Matches integers/decimals with optional sign, thousands commas, and a leading $.
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def normalize_number(token: str) -> Optional[float]:
    """Turn a raw numeric token ('1,234', '$50', '3.0', '12%') into a float, or None."""
    if token is None:
        return None
    s = token.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    if s in ("", "+", "-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_final_number(text: str) -> Optional[float]:
    """Extract the model's answer as a float.

    Strategy: prefer the number immediately following an answer cue ('the answer is'),
    otherwise fall back to the LAST number in the text (final answers come last in CoT).
    """
    if not text:
        return None

    # Prefer a number right after an explicit answer cue.
    cue = re.search(r"answer\s*is\s*:?\s*(" + _NUMBER_RE.pattern + r")", text, re.IGNORECASE)
    if cue:
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


def normalize_gold(raw: object, kind: str) -> Optional[float]:
    """Normalize a gold answer from a given eval-set kind into a float.

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


def answers_match(pred_text: str, gold: float, tol: float = 1e-4) -> bool:
    """True if the prediction's extracted number equals gold within tolerance."""
    if gold is None:
        return False
    pred = extract_final_number(pred_text)
    if pred is None:
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))
