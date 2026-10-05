"""Counterfactual twins of GSM8k-Aug rows (ledger §110).

A twin changes one integer of the question and recomputes the ``<<a op b = c>>``
chain, so the twin's answer is the correct counterfactual for the changed number.
Operands equal to an earlier result are references and are propagated.  Rows where
the substitution would be ambiguous are rejected rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

DELTAS = (1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 20, -20)
_EQUATION = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
_NUMBER = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_SAFE_EXPRESSION = re.compile(r"^[\d.+\-*/() ]+$")
_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Perturbation:
    question: str
    cot: str
    answer: str
    gold: str
    step: int                                   # 1-based equation the changed number enters
    steps: int                                  # equations in the chain
    original_number: str
    new_number: str
    changed_values: tuple[tuple[str, str], ...]  # (old result, new result) per changed equation

    def row(self) -> dict:
        return {"question": self.question, "cot": self.cot, "answer": self.answer, "gold": self.gold}


def _evaluate(expression: str) -> float:
    if not _SAFE_EXPRESSION.match(expression):
        raise ValueError(f"unsafe expression {expression!r}")
    return float(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - whitelisted characters only


def _is_integer(value: float) -> bool:
    return abs(value - round(value)) < _TOLERANCE


def _format(value: float) -> str:
    return str(int(round(value)))


def _close(a: float, b: float) -> bool:
    return abs(a - b) < _TOLERANCE


def parse_chain(cot: str) -> list[dict]:
    """Equations with operand spans (relative to ``cot``) and numeric results."""
    equations = []
    for match in _EQUATION.finditer(cot):
        left, right = match.group(1), match.group(2)
        left_start = match.start(1)
        operands = [{"text": m.group(0), "value": float(m.group(0)),
                     "span": (left_start + m.start(), left_start + m.end())} for m in _NUMBER.finditer(left)]
        try:
            result = float(right.replace(",", ""))
        except ValueError:
            return []
        equations.append({"left": left, "operands": operands, "result": result, "result_text": right,
                          "result_span": match.span(2), "span": match.span()})
    return equations


def question_integers(question: str) -> list[tuple[str, tuple[int, int]]]:
    """Standalone integers of the question text: not part of a decimal, a thousands
    group, a fraction or a clock time.  A sentence-ending period is fine."""
    found = []
    for match in re.finditer(r"\d+", question):
        start, end = match.span()
        before = question[start - 1] if start else ""
        before2 = question[start - 2] if start > 1 else ""
        after = question[end] if end < len(question) else ""
        after2 = question[end + 1] if end + 1 < len(question) else ""
        if before.isdigit() or after.isdigit():
            continue
        if after in ".," and after2.isdigit():          # 5.00, 2,000
            continue
        if before in ".," and before2.isdigit():        # the 00 of 5.00, the 000 of 2,000
            continue
        if before in ":/" or after in ":/":             # 8:30, 1/4
            continue
        found.append((match.group(0), (start, end)))
    return found


def perturb_row(row: dict, deltas: tuple[int, ...] = DELTAS) -> Perturbation | None:
    question, cot = str(row["question"]), str(row["cot"])
    equations = parse_chain(cot)
    if not equations:
        return None
    results = [eq["result"] for eq in equations]
    if not all(_is_integer(r) and r >= 0 for r in results):
        return None
    if any(_close(results[i], results[j]) for i in range(len(results)) for j in range(i)):
        return None
    all_operands = [op for eq in equations for op in eq["operands"]]
    numbers = question_integers(question)
    question_values = [float(text) for text, _ in numbers]
    if any(any(_close(r, q) for q in question_values) for r in results):
        return None  # a result equal to a question number makes references ambiguous
    # reference structure: an operand equal to an earlier result refers to it
    refs: dict[int, list[int]] = {}
    for i, eq in enumerate(equations):
        for j, op in enumerate(eq["operands"]):
            for k in range(i):
                if _close(op["value"], results[k]):
                    refs.setdefault(i, []).append(j)
    for text, span in numbers:
        x = float(text)
        if sum(1 for t, _ in numbers if t == text) != 1 or sum(1 for op in all_operands if _close(op["value"], x)) != 1:
            continue
        if any(_close(x, r) for r in results):
            continue
        step = next(i for i, eq in enumerate(equations) if any(_close(op["value"], x) for op in eq["operands"]))
        if any(j in refs.get(step, []) for j, op in enumerate(equations[step]["operands"]) if _close(op["value"], x)):
            continue
        forbidden = {round(v, 6) for v in question_values} | {round(op["value"], 6) for op in all_operands} | {round(r, 6) for r in results}
        for delta in deltas:
            new_x = x + delta
            if new_x <= 0 or round(new_x, 6) in forbidden:
                continue
            new_results = _recompute(equations, results, refs, step, x, new_x)
            if new_results is None or _close(new_results[-1], results[-1]):
                continue
            changed = tuple((_format(results[i]), _format(new_results[i])) for i in range(len(results))
                            if not _close(results[i], new_results[i]))
            new_cot = _rewrite(cot, equations, results, new_results, refs, step, x, new_x)
            new_question = question[:span[0]] + _format(new_x) + question[span[1]:]
            return Perturbation(question=new_question, cot=new_cot, answer=_format(new_results[-1]), gold=_format(new_results[-1]),
                                step=step + 1, steps=len(equations), original_number=text, new_number=_format(new_x),
                                changed_values=changed)
    return None


def _recompute(equations, results, refs, step, x, new_x):
    new_results: list[float] = []
    for i, eq in enumerate(equations):
        pieces, cursor, left = [], 0, eq["left"]
        base = eq["span"][0] + 2  # offset of ``left`` inside the cot
        for j, op in enumerate(eq["operands"]):
            start, end = op["span"][0] - base, op["span"][1] - base
            pieces.append(left[cursor:start])
            if i == step and _close(op["value"], x):
                pieces.append(repr(new_x))
            elif j in refs.get(i, []):
                k = next(k for k in range(i) if _close(op["value"], results[k]))
                pieces.append(repr(new_results[k]))
            else:
                pieces.append(op["text"])
            cursor = end
        pieces.append(left[cursor:])
        try:
            value = _evaluate("".join(pieces))
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
            return None
        if not _is_integer(value) or value < 0:
            return None
        new_results.append(float(round(value)))
    return new_results


def _rewrite(cot, equations, results, new_results, refs, step, x, new_x):
    edits: list[tuple[tuple[int, int], str]] = []
    for i, eq in enumerate(equations):
        for j, op in enumerate(eq["operands"]):
            if i == step and _close(op["value"], x):
                edits.append((op["span"], _format(new_x)))
            elif j in refs.get(i, []):
                k = next(k for k in range(i) if _close(op["value"], results[k]))
                if not _close(results[k], new_results[k]):
                    edits.append((op["span"], _format(new_results[k])))
        if not _close(results[i], new_results[i]):
            edits.append((eq["result_span"], _format(new_results[i])))
    out, cursor = [], 0
    for (start, end), text in sorted(edits):
        out.append(cot[cursor:start]); out.append(text); cursor = end
    out.append(cot[cursor:])
    return "".join(out)
