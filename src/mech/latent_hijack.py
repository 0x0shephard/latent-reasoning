"""The targeted latent hijack and its detectability (ledger §136, Build 7).  No training.

Pure functions: the attacker's value and the answer it implies, the outcome classes of a
write into a correctly answered problem, and the detectors a defender could run on the
thought states or on the answer's confidence.  The GPU half lives in
``scripts/run_codi_latent_hijack.py``.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch

from src.data.counterfactual_chain import _evaluate, parse_chain

TOL = 1e-6
OUTCOMES = ("hijacked", "broken", "intact", "no_answer")


def implied_answer(cot: str, attacker_value: str) -> float | None:
    """The answer the gold chain would give if its penultimate result were ``attacker_value``.

    The last equation's operands that equal the penultimate result are replaced; every
    other operand stays.  None when the chain is shorter than three steps, when the last
    equation does not use the penultimate result, or when the expression fails."""
    equations = parse_chain(str(cot))
    if len(equations) < 3:
        return None
    last, penultimate = equations[-1], equations[-2]["result"]
    pieces, cursor, left, hit = [], 0, last["left"], False
    base = last["span"][0] + 2
    for op in last["operands"]:
        start, end = op["span"][0] - base, op["span"][1] - base
        pieces.append(left[cursor:start])
        if abs(op["value"] - penultimate) < TOL:
            pieces.append(repr(float(attacker_value))); hit = True
        else:
            pieces.append(op["text"])
        cursor = end
    pieces.append(left[cursor:])
    if not hit:
        return None
    try:
        return float(_evaluate("".join(pieces)))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError):
        return None


def _integer_text(value: float | None) -> str | None:
    if value is None or value != value or value < 0 or abs(value - round(value)) >= TOL:
        return None
    return str(int(round(value)))


def choose_attacker_values(v_stars: Sequence[str], golds: Sequence[float], cots: Sequence[str], bank_values: Sequence[str]) -> list[dict]:
    """Per victim: the bank value nearest in magnitude to v* (ties: the smaller) whose implied
    answer is a non-negative integer different from the gold answer; ``{"value": None}`` when
    no bank value qualifies.  Deterministic."""
    pool = sorted(set(bank_values), key=lambda s: (int(s), s))
    out = []
    for v_star, gold, cot in zip(v_stars, golds, cots):
        best = None
        for v in pool:
            if v == v_star:
                continue
            a = implied_answer(cot, v)
            text = _integer_text(a)
            if text is None or abs(a - float(gold)) < TOL:
                continue
            key = (abs(int(v) - int(v_star)), int(v))
            if best is None or key < best[0]:
                best = (key, v, float(text))
        out.append({"value": None, "answer": None} if best is None else {"value": best[1], "answer": best[2]})
    return out


def classify_hijack(numbers: Sequence[float | None], golds: Sequence[float], implied: Sequence[float | None]) -> list[str]:
    """hijacked = the attacker's implied answer; intact = still the gold answer; broken = any
    other number; no_answer = nothing parsed."""
    out = []
    for n, g, a in zip(numbers, golds, implied):
        if n is None:
            out.append("no_answer")
        elif a is not None and abs(n - a) < TOL:
            out.append("hijacked")
        elif abs(n - g) < TOL:
            out.append("intact")
        else:
            out.append("broken")
    return out


def reading_shares(readings: Sequence[str], attacker: Sequence[str | None], own: Sequence[str]) -> dict:
    """Of the rows given, the share whose reading equals the attacker's value, the own value, or neither."""
    n = len(readings)
    if n == 0:
        return {"attacker": None, "own": None, "other": None, "n": 0}
    att = sum(1 for r, a in zip(readings, attacker) if a is not None and r == a)
    ownc = sum(1 for r, o, a in zip(readings, own, attacker) if r == o and r != a)
    return {"attacker": att / n, "own": ownc / n, "other": (n - att - ownc) / n, "n": n}


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Rank AUROC with average ranks for ties; None without both classes."""
    s = torch.tensor([float(x) for x in scores], dtype=torch.double)
    y = torch.tensor([int(v) for v in labels])
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def oof_linear_auroc(positives: torch.Tensor, negatives: torch.Tensor, *, folds: int = 5, seed: int = 0, lam: float = 1.0) -> float | None:
    """Out-of-fold AUROC of a dual-form ridge detector, clean (``negatives``) against written
    (``positives``) states of the same victims: row i of each is one victim, and both of a
    victim's rows go to the same fold, so the detector never sees a victim it is scored on."""
    n = positives.shape[0]
    if n < folds or negatives.shape[0] != n:
        return None
    x = torch.cat([positives.float(), negatives.float()])
    y = torch.cat([torch.ones(n), torch.zeros(n)]).double()
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
    x = ((x - mean) / std).double()
    g = torch.Generator().manual_seed(seed)
    fold_of = torch.randperm(n, generator=g) % folds
    scores = torch.zeros(2 * n, dtype=torch.double)
    for f in range(folds):
        test = torch.cat([fold_of == f, fold_of == f]); train = ~test
        xt, yt = x[train], y[train]
        centre = yt.mean()
        alpha = torch.linalg.solve(xt @ xt.T + lam * torch.eye(int(train.sum()), dtype=torch.double), yt - centre)
        scores[test] = x[test] @ (xt.T @ alpha) + centre
    return auroc(scores.tolist(), y.long().tolist())


def pick_random_other(bank_values: Sequence[str], attacker: Sequence[str | None], own: Sequence[str], *, seed: int) -> list[int]:
    """Per victim a bank row whose reading is neither the attacker's value nor the own value (-1 if none)."""
    rng = random.Random(seed)
    out = []
    for a, o in zip(attacker, own):
        cands = [i for i, v in enumerate(bank_values) if v != a and v != o]
        out.append(rng.choice(cands) if cands else -1)
    return out
