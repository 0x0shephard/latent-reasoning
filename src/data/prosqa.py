"""ProsQA (Hao et al. 2024) for the audit (ledger §135): premises, the gold path, and twins.

A row has a question (a list of "Every X is a Y." / "A is a X." premises followed by
"Is A a Y or a Z?"), an answer sentence, the gold path as steps, and the graph (symbols,
edges, root, target, negative target).  A twin rewrites the path's last hop so that the
other option becomes the reachable one; reachability over the edited premises is the
task's calculator.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re

_QUESTION = re.compile(r"Is (\w+) an? (\w+) or (?:an? )?(\w+)\?\s*$")
_PREMISE = re.compile(r"^(?:Every|Each) (\w+) is an? (\w+)\.$|^(\w+) is an? (\w+)\.$")


def load_prosqa(path: str | Path) -> list[dict]:
    rows = json.loads(Path(path).read_text())
    for i, r in enumerate(rows):
        r.setdefault("idx", i)
    return rows


def split_question(question: str) -> tuple[list[str], str]:
    """The premise sentences and the final query."""
    parts = [p.strip() for p in re.split(r"(?<=[.?])\s+", question.strip()) if p.strip()]
    return parts[:-1], parts[-1]


def options(question: str) -> tuple[str, str, str] | None:
    """(subject, option 1, option 2) from the query sentence, or None."""
    m = _QUESTION.search(question.strip())
    return (m.group(1), m.group(2), m.group(3)) if m else None


def premise_edge(sentence: str) -> tuple[str, str] | None:
    m = _PREMISE.match(sentence.strip())
    if not m:
        return None
    return (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))


def reachable(edges: set[tuple[str, str]], start: str) -> set[str]:
    out, seen, queue = {}, {start}, deque([start])
    while queue:
        node = queue.popleft()
        for a, b in edges:
            if a == node and b not in seen:
                seen.add(b); queue.append(b)
    return seen


def answer_option(text: str, opts: tuple[str, str]) -> str | None:
    """Which option a decoded answer names (the first one mentioned); None if neither or both at once."""
    low = text.lower()
    hits = [(low.find(o.lower()), o) for o in opts if low.find(o.lower()) >= 0]
    if not hits:
        return None
    hits.sort()
    if len(hits) == 2 and hits[0][0] == hits[1][0]:
        return None
    return hits[0][1]


def gold_option(row: dict) -> str | None:
    opts = options(row["question"])
    return answer_option(str(row["answer"]), opts[1:]) if opts else None


def make_twin(row: dict) -> dict | None:
    """Rewrite the last hop so the other option is reachable and the gold one is not; None if that single
    edit does not flip reachability (another path still reaches the gold option)."""
    opts = options(row["question"])
    if not opts or not row.get("steps"):
        return None
    subject, o1, o2 = opts
    gold = gold_option(row)
    if gold not in (o1, o2):
        return None
    other = o2 if gold == o1 else o1
    last = str(row["steps"][-1]).strip()
    edge = premise_edge(last)
    if edge is None or edge[1] != gold:
        return None
    premises, query = split_question(row["question"])
    if premises.count(last) != 1:
        return None
    new_last = last[: len(last) - len(gold) - 1] + other + "."
    new_premises = [new_last if p == last else p for p in premises]
    edges = {e for e in (premise_edge(p) for p in new_premises) if e}
    reach = reachable(edges, subject)
    if gold in reach or other not in reach:
        return None
    twin_question = " ".join(new_premises + [query])
    twin_answer = str(row["answer"]).replace(gold, other)
    return {"question": twin_question, "answer": twin_answer, "steps": list(row["steps"][:-1]) + [new_last], "hops": len(row["steps"]),
            "subject": subject, "options": [o1, o2], "gold_option": other, "source_idx": row.get("idx")}


def twin_pairs(rows: list[dict]) -> list[tuple[dict, dict]]:
    """(row, twin) for every row whose last hop can be flipped; rows carry hops, subject, options, gold_option."""
    pairs = []
    for r in rows:
        t = make_twin(r)
        if t is None:
            continue
        opts = options(r["question"])
        base = {**r, "hops": len(r["steps"]), "subject": opts[0], "options": list(opts[1:]), "gold_option": gold_option(r)}
        pairs.append((base, t))
    return pairs
