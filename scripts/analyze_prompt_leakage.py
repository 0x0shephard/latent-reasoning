"""Prompt-leakage probes on recorded twin states (ledger §125, Build 1).

For every slot and probe family, the cross-validated accuracy of decoding question
attributes that need no computation (the chain's first operator; whether the chain has
three or more steps) beside the readability of a computed value (the penultimate
intermediate, from the monitor analysis).  A monitor that scores well on a prompt-borne
attribute at a slot the model does not use is reading the prompt, not the plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_monitor_faithfulness import FOLDS, LONG_MIN_STEPS, SEED, mlp_fit_predict, ridge_fit
from src.data.counterfactual_chain import parse_chain

OPERATORS = ("+", "-", "*", "/")
ATTRIBUTES = ("first_operator", "long_chain")


def first_operator(cot: str) -> int | None:
    chain = parse_chain(cot)
    if not chain:
        return None
    left = chain[0]["left"]
    hits = [(left.find(op), i) for i, op in enumerate(OPERATORS) if op in left]
    return min(hits)[1] if hits else None


def attributes_from_pairs(pairs_path: Path) -> dict:
    ops, long = [], []
    for line in pairs_path.read_text().splitlines():
        p = json.loads(line)
        for cot in (p["cot"], p["twin_cot"]):
            op = first_operator(cot)
            ops.append(-1 if op is None else op)
            long.append(int(len(parse_chain(cot)) >= LONG_MIN_STEPS))
    return {"first_operator": torch.tensor(ops), "long_chain": torch.tensor(long)}


def pair_folds(n: int, folds: int, seed: int) -> torch.Tensor:
    pairs = torch.randperm(n // 2, generator=torch.Generator().manual_seed(seed))
    fold_of_pair = torch.empty(n // 2, dtype=torch.long)
    for f in range(folds):
        fold_of_pair[pairs[f::folds]] = f
    return fold_of_pair.repeat_interleave(2)


def classify(states: torch.Tensor, labels: torch.Tensor, n_classes: int, *, family: str, folds: int = FOLDS, seed: int = SEED) -> torch.Tensor:
    """Out-of-fold class predictions; ridge to one-hot targets or the §124 MLP."""
    fold = pair_folds(states.shape[0], folds, seed)
    out = torch.empty(states.shape[0], dtype=torch.long)
    onehot = torch.nn.functional.one_hot(labels, n_classes).float()
    for f in range(folds):
        train = fold != f
        if family == "ridge":
            w = ridge_fit(states[train], onehot[train])
            xb = torch.cat([states[~train], torch.ones(int((~train).sum()), 1)], 1)
            out[~train] = (xb @ w).argmax(1)
        else:
            out[~train] = mlp_fit_predict(states[train], labels[train], states[~train], n_classes, seed=seed + f)
    return out


def analyze(run_dir: Path, *, families=("ridge", "mlp"), folds: int = FOLDS, seed: int = SEED) -> dict:
    states = torch.load(Path(run_dir) / "trajectory_states.pt", map_location="cpu", weights_only=False)["states"].float()
    attrs = attributes_from_pairs(Path(run_dir) / "pairs.jsonl")
    n, slots = states.shape[0], states.shape[1]
    partner = torch.arange(n) ^ 1
    report = {"run": str(run_dir), "rows": n, "families": list(families), "attributes": {}}
    for name in ATTRIBUTES:
        labels = attrs[name]
        valid = (labels >= 0) & (labels[partner] >= 0)
        idx = valid.nonzero().squeeze(1)
        if int(idx.numel()) < 4 * folds:
            report["attributes"][name] = {"rows": int(idx.numel()), "note": "too few rows"}
            continue
        y = labels[idx]
        classes = int(y.max()) + 1
        counts = Counter(y.tolist())
        majority = max(counts.values()) / len(y)
        entry = {"rows": int(idx.numel()), "classes": classes, "class_counts": {str(k): v for k, v in sorted(counts.items())}, "majority_baseline": majority, "slots": {}}
        if len(counts) < 2:
            entry["note"] = "one class only"; report["attributes"][name] = entry; continue
        for s in range(slots):
            entry["slots"][str(s)] = {}
            for family in families:
                pred = classify(states[idx, s], y, classes, family=family, folds=folds, seed=seed)
                entry["slots"][str(s)][family] = float((pred == y).double().mean())
        report["attributes"][name] = entry
    return report


def markdown(report: dict, monitor_long: dict | None = None) -> str:
    lines = [f"run `{Path(report['run']).name}`: {report['rows']} rows", ""]
    for name, e in report["attributes"].items():
        if "note" in e:
            lines.append(f"- {name}: {e['note']} ({e['rows']} rows)"); continue
        lines += [f"**{name}** (majority baseline {e['majority_baseline']:.2f}, classes {e['class_counts']})", "",
                  "| slot | ridge | mlp |" + (" penultimate readability ridge / mlp |" if monitor_long else ""), "|---|---|---|" + ("---|" if monitor_long else "")]
        for s, fam in e["slots"].items():
            extra = ""
            if monitor_long:
                m = monitor_long["slots"].get(s, {}).get("penultimate", {})
                mlp = (m.get("families") or {}).get("mlp", {})
                f = lambda v: "—" if v is None else f"{v:.2f}"
                extra = f" {f(m.get('readability_long'))} / {f(mlp.get('readability_long'))} |"
            lines.append(f"| {s} | {fam.get('ridge', float('nan')):.2f} | {fam.get('mlp', float('nan')):.2f} |" + extra)
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [analyze(d) for d in args.run_dirs]
    for r in reports:
        print(markdown(r)); print()
    if args.output:
        args.output.write_text(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
