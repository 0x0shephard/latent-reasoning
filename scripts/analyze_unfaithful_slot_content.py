"""§133 C: what does a readable-but-unfaithful thought hold?  No decoding.

Logistic probes (5 folds, balanced accuracy) on each slot's readout state for attributes of the
question (its first number, how many numbers it has), of the plan (step count, first operator)
and of the computation (the penultimate value), against a text-only probe on the question
(hashed bag of words) that says what a reader of the prompt alone could get.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.counterfactual_chain import parse_chain

FOLDS, SEED, TOPK, HASH_DIM = 5, 0, 20, 4096
_NUM = re.compile(r"\d[\d,]*\.?\d*")
_OP = re.compile(r"[+\-*/]")


def question_numbers(q: str) -> list[str]:
    return [n.replace(",", "") for n in _NUM.findall(q)]


def first_operator(cot: str | None) -> str | None:
    if not cot:
        return None
    eqs = parse_chain(cot)
    if not eqs:
        return None
    ops = _OP.findall(re.sub(r"\d[\d.,]*", "", eqs[0]["left"]))
    return ops[0] if ops else None


def penultimate(cot: str | None) -> str | None:
    if not cot:
        return None
    eqs = parse_chain(cot)
    if len(eqs) < 2:
        return None
    r = eqs[-2]["result"]
    return str(int(round(r))) if abs(r - round(r)) < 1e-6 else None


def topk_classes(values: list[str | None], k: int = TOPK) -> tuple[np.ndarray, list[str]]:
    counts = collections.Counter(v for v in values if v is not None)
    keep = [v for v, _ in counts.most_common(k)]
    index = {v: i for i, v in enumerate(keep)}
    labels = np.array([index.get(v, -1) if v is not None else -1 for v in values])
    return labels, keep


def attributes(questions: list[str], cots: list[str | None], steps: np.ndarray) -> dict[str, np.ndarray]:
    first_num, _ = topk_classes([(question_numbers(q) or [None])[0] for q in questions])
    n_num = np.array([min(len(question_numbers(q)), 7) - 1 for q in questions])
    step = np.where(steps >= 1, np.minimum(steps, 6) - 1, -1)
    op_vals = [first_operator(c) for c in cots]; op_index = {"+": 0, "-": 1, "*": 2, "/": 3}
    op = np.array([op_index.get(v, -1) if v is not None else -1 for v in op_vals])
    v_star, _ = topk_classes([penultimate(c) for c in cots])
    return {"first_number": first_num, "number_count": n_num, "step_count": step, "first_operator": op, "penultimate_value": v_star}


def hashed_bow(questions: list[str], dim: int = HASH_DIM) -> np.ndarray:
    X = np.zeros((len(questions), dim), dtype=np.float32)
    for i, q in enumerate(questions):
        for tok in re.findall(r"[a-z]+|\d+", q.lower()):
            X[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    X /= np.maximum(1.0, np.linalg.norm(X, axis=1, keepdims=True))
    return X


def ridge_balanced_accuracy(X: np.ndarray, y: np.ndarray, *, folds: int = FOLDS, seed: int = SEED, lam: float = 1e-2, steps: int = 200) -> float | None:
    """5-fold balanced accuracy of an L2-regularised multinomial logistic probe (LBFGS), standardised features."""
    keep = y >= 0; X = X[keep]; y = y[keep]
    classes = np.unique(y)
    if len(classes) < 2 or len(y) < 5 * folds:
        return None
    rng = np.random.default_rng(seed); order = rng.permutation(len(y)); pred = np.full(len(y), -1)
    yi = np.searchsorted(classes, y)
    for f in range(folds):
        test = order[f::folds]; train = np.setdiff1d(order, test)
        mu = X[train].mean(0); sd = X[train].std(0) + 1e-6
        Xt = torch.tensor((X[train] - mu) / sd, dtype=torch.float32); Xs = torch.tensor((X[test] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(yi[train]); W = torch.zeros(Xt.shape[1], len(classes), requires_grad=True); b = torch.zeros(len(classes), requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + lam * (W * W).sum(); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad():
            pred[test] = (Xs @ W + b).argmax(1).numpy()
    recalls = [float((pred[yi == c] == c).mean()) for c in range(len(classes)) if (yi == c).sum()]
    return float(np.mean(recalls))


def analyze(states: torch.Tensor, questions: list[str], cots: list[str | None], steps: np.ndarray, *, rows: np.ndarray | None = None, pca: int = 256) -> dict:
    rows = np.arange(len(questions)) if rows is None else rows
    attrs = {k: v[rows] for k, v in attributes(questions, cots, steps).items()}
    text = hashed_bow([questions[i] for i in rows])
    st = states[rows].float().numpy()
    report = {"rows": int(len(rows)), "attributes": {}}
    for name, y in attrs.items():
        entry = {"classes": int(len(np.unique(y[y >= 0]))), "labelled": int((y >= 0).sum()), "balanced_baseline": None, "text_only": ridge_balanced_accuracy(text, y), "slots": {}}
        entry["balanced_baseline"] = 1.0 / entry["classes"] if entry["classes"] else None
        for s in range(st.shape[1]):
            X = st[:, s]
            if X.shape[1] > pca:   # project onto the top principal directions for a ridge of manageable size
                Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); X = Xc @ Vt[:pca].T
            entry["slots"][str(s)] = ridge_balanced_accuracy(X, y)
        report["attributes"][name] = entry
    return report


def markdown(report: dict, name: str, *, carriers: list[int], unfaithful: list[int]) -> str:
    lines = [f"### {name}: {report['rows']} rows; carriers {carriers}, readable-but-unfaithful {unfaithful}", "",
             "| attribute | classes | balanced baseline | text-only | " + " | ".join(f"slot {s}" for s in range(6)) + " |", "|---|---|---|---|" + "---|" * 6]
    for a, e in report["attributes"].items():
        f = lambda v: "-" if v is None else f"{v:.2f}"
        lines.append(f"| {a} | {e['classes']} | {f(e['balanced_baseline'])} | {f(e['text_only'])} | " + " | ".join(f(e["slots"][str(s)]) for s in range(6)) + " |")
    return "\n".join(lines)


def checks_from(report: dict, *, carriers: list[int], unfaithful: list[int]) -> dict:
    a = report["attributes"]
    def best(name, slots):
        vals = [a[name]["slots"][str(s)] for s in slots if a[name]["slots"][str(s)] is not None]
        return max(vals) if vals else None
    out = {}
    for name in ("step_count", "first_number", "number_count"):
        u, c = best(name, unfaithful), best(name, carriers)
        out[f"p21_unfaithful_reads_{name}_at_or_above_carrier"] = None if u is None or c is None else u >= c - 0.02
    u, c = best("penultimate_value", unfaithful), best("penultimate_value", carriers)
    out["carrier_reads_penultimate_above_unfaithful"] = None if u is None or c is None else c > u
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True); parser.add_argument("--chains-json", type=Path, required=True)
    parser.add_argument("--carriers", required=True); parser.add_argument("--unfaithful", required=True)
    parser.add_argument("--long-only", action="store_true"); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--name", default="")
    args = parser.parse_args()
    d = torch.load(args.readout, map_location="cpu", weights_only=False)
    chains = json.loads(args.chains_json.read_text()); questions = [c["question"] for c in chains]; cots = [c["cot"] for c in chains]
    steps = np.asarray(d["steps"]); rows = np.nonzero(steps >= 3)[0] if args.long_only else None
    carriers = [int(x) for x in args.carriers.split(",")]; unfaithful = [int(x) for x in args.unfaithful.split(",")]
    report = analyze(d["states"], questions, cots, steps, rows=rows); report["checks"] = checks_from(report, carriers=carriers, unfaithful=unfaithful)
    report["carriers"] = carriers; report["unfaithful"] = unfaithful
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown(report, args.name, carriers=carriers, unfaithful=unfaithful)); print(json.dumps(report["checks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
