"""Templated two- and three-step arithmetic word problems in GSM8k-Aug format.

Each problem is a natural-language question, an equation-only chain of thought
``<<a+b=c>> <<c*d=e>>`` and a ``#### answer`` line, matching the released CODI
training schema so ``format_official_codi_row`` and the teacher-forcing path
apply unchanged (the official formatter drops the last equation, whose result is
the answer).  Numbers are chosen so every intermediate is a positive integer.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence

import torch

from src.eval.official_codi_gate import official_answers_match


NAMES = ("Sara", "Tom", "Maya", "Ali", "Lena", "Omar", "Priya", "Jonas", "Nia", "Ravi",
         "Ella", "Kofi", "Mira", "Leo", "Zara", "Ivan")
ITEMS = ("apples", "pencils", "stickers", "books", "marbles", "cookies", "coins", "cards",
         "shells", "buttons", "stamps", "beads")


@dataclass(frozen=True)
class Problem:
    question: str
    cot: str
    answer: str  # "#### N"
    steps: int
    template: str

    def as_row(self) -> dict:
        return {"question": self.question, "cot": self.cot, "answer": self.answer,
                "gold": self.answer.split(" ")[-1], "steps": self.steps, "template": self.template}


def _eq(a: int, op: str, b: int) -> tuple[str, int]:
    value = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"<<{a}{op}{b}={value}>>", value


def _t_gain_lose(rng, name, item):
    a, b, c = rng.randint(10, 80), rng.randint(5, 60), rng.randint(2, 40)
    e1, s = _eq(a, "+", b)
    if c >= s:
        return None
    e2, r = _eq(s, "-", c)
    q = f"{name} has {a} {item}. {name} buys {b} more and then gives {c} to a friend. How many {item} does {name} have now?"
    return q, [e1, e2], r


def _t_boxes_broken(rng, name, item):
    a, b, c = rng.randint(3, 12), rng.randint(3, 12), rng.randint(1, 20)
    e1, s = _eq(a, "*", b)
    if c >= s:
        return None
    e2, r = _eq(s, "-", c)
    q = f"A box holds {a} {item}. {name} has {b} boxes. {c} of the {item} are broken. How many good {item} are there?"
    return q, [e1, e2], r


def _t_hours_pay(rng, name, item):
    a, b, c = rng.randint(5, 25), rng.randint(2, 9), rng.randint(2, 9)
    e1, h = _eq(b, "+", c)
    e2, r = _eq(a, "*", h)
    q = f"{name} earns ${a} per hour. {name} works {b} hours on Monday and {c} hours on Tuesday. How much does {name} earn in total?"
    return q, [e1, e2], r


def _t_share_left(rng, name, item):
    a, b, c = rng.randint(4, 15), rng.randint(3, 9), rng.randint(2, 20)
    e1, s = _eq(a, "*", b)
    e2, r = _eq(s, "+", c)
    q = f"{name} packs {b} bags with {a} {item} each and finds {c} more {item} in a drawer. How many {item} does {name} have?"
    return q, [e1, e2], r


def _t_profit(rng, name, item):
    a, b, c, d = rng.randint(5, 40), rng.randint(5, 40), rng.randint(2, 9), rng.randint(5, 60)
    e1, s = _eq(a, "+", b)
    e2, t = _eq(s, "*", c)
    if d >= t:
        return None
    e3, r = _eq(t, "-", d)
    q = f"A shop sells {a} cups in the morning and {b} cups in the afternoon. Each cup costs ${c}. The shop spends ${d} on supplies. What is the profit?"
    return q, [e1, e2, e3], r


def _t_trips(rng, name, item):
    a, b, c, d = rng.randint(2, 9), rng.randint(3, 12), rng.randint(2, 9), rng.randint(1, 30)
    e1, s = _eq(a, "*", b)
    e2, t = _eq(s, "*", c)
    if d >= t:
        return None
    e3, r = _eq(t, "-", d)
    q = f"{name} makes {a} trips a day for {b} days, carrying {c} {item} each trip. {d} {item} are lost. How many {item} arrive?"
    return q, [e1, e2, e3], r


def _t_savings(rng, name, item):
    a, b, c, d = rng.randint(5, 30), rng.randint(2, 8), rng.randint(5, 50), rng.randint(5, 40)
    e1, s = _eq(a, "*", b)
    e2, t = _eq(s, "+", c)
    if d >= t:
        return None
    e3, r = _eq(t, "-", d)
    q = f"{name} saves ${a} a week for {b} weeks and gets ${c} as a gift. {name} then spends ${d} on a book. How much money is left?"
    return q, [e1, e2, e3], r


def _t_classes(rng, name, item):
    a, b, c, d = rng.randint(10, 30), rng.randint(10, 30), rng.randint(2, 6), rng.randint(2, 25)
    e1, s = _eq(a, "+", b)
    e2, t = _eq(s, "*", c)
    e3, r = _eq(t, "+", d)
    q = f"One class has {a} students and another has {b}. Each student brings {c} {item}, and the teacher adds {d} more. How many {item} are there in total?"
    return q, [e1, e2, e3], r


TEMPLATES: dict[str, Callable] = {
    "gain_lose": _t_gain_lose, "boxes_broken": _t_boxes_broken, "hours_pay": _t_hours_pay,
    "share_left": _t_share_left, "profit": _t_profit, "trips": _t_trips,
    "savings": _t_savings, "classes": _t_classes,
}


def generate_problems(count: int, *, seed: int, max_answer: int = 9_999) -> list[Problem]:
    """Deterministic, unique-question problems with a balanced template mix."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    names = list(TEMPLATES)
    seen: set[str] = set()
    problems: list[Problem] = []
    attempts = 0
    while len(problems) < count:
        attempts += 1
        if attempts > 200 * count:
            raise RuntimeError("could not generate enough unique problems")
        template = names[len(problems) % len(names)]
        built = TEMPLATES[template](rng, rng.choice(NAMES), rng.choice(ITEMS))
        if built is None:
            continue
        question, equations, answer = built
        if answer <= 0 or answer > max_answer or question in seen:
            continue
        seen.add(question)
        problems.append(Problem(question=question, cot=" ".join(equations),
                                answer=f"#### {answer}", steps=len(equations), template=template))
    return problems


def verify_problem(problem: Problem) -> bool:
    """Every equation must be arithmetically true and the last result must be the answer."""
    last = None
    for token in problem.cot.split(" "):
        if not (token.startswith("<<") and token.endswith(">>")):
            return False
        expression, value = token[2:-2].split("=")
        if eval(expression, {"__builtins__": {}}, {}) != int(value):  # noqa: S307 (digits/ops only)
            return False
        last = int(value)
    return last == int(problem.answer.split(" ")[-1])


@torch.inference_mode()
def generate_teacher_cot(model, tokenizer, questions: Sequence[str], *, max_new_tokens: int,
                         batch_size: int, device) -> list[str]:
    """Greedy explicit-CoT generation from the question alone (the teacher path)."""
    from src.models.official_codi import _normalized_official_questions

    model.eval()
    outputs: list[str] = []
    for start in range(0, len(questions), batch_size):
        chunk = _normalized_official_questions(questions[start : start + batch_size])
        batch = tokenizer(chunk, return_tensors="pt", padding="longest",
                          add_special_tokens=False).to(device)
        generated = model.codi.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            max_new_tokens=int(max_new_tokens), do_sample=False,
            pad_token_id=int(tokenizer.pad_token_id), eos_token_id=int(tokenizer.eos_token_id),
        )
        continuation = generated[:, batch["input_ids"].shape[1] :]
        outputs.extend(tokenizer.decode(row, skip_special_tokens=True) for row in continuation)
    return outputs


def teacher_accuracy(outputs: Sequence[str], problems: Sequence[Problem]) -> tuple[float, list[bool]]:
    correct = [bool(official_answers_match(text, p.answer.split(" ")[-1])) for text, p in zip(outputs, problems)]
    return sum(correct) / max(1, len(correct)), correct
