"""Carrier-based error detection, the analysis half (ledger §125, Build 2).

From ``readout_states.pt``: per slot, the model's own numeric readout of the thought
state (top-1 value and probability over the numeric vocabulary, entropy, agreement with
the native answer and with the other slots).  Logistic classifiers for "the native
answer is wrong" on carrier slots only, inert slots only, all slots, the confidence
baseline, and an LTO-style classifier on the raw states; AUROC with bootstrap intervals.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_monitor_faithfulness import numeric_vocabulary

FOLDS = 5
SEED = 20_261_014
SAMPLES = 1_000
L2 = 1e-2
STEPS = 400


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    pos, neg = scores[labels], scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    greater = (pos.unsqueeze(1) > neg.unsqueeze(0)).double().sum()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).double().sum()
    return float((greater + 0.5 * ties) / (pos.numel() * neg.numel()))


def auroc_ci(scores: torch.Tensor, labels: torch.Tensor, *, samples: int = SAMPLES, seed: int = SEED) -> list | None:
    n = scores.numel()
    if n == 0:
        return None
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(samples):
        idx = torch.randint(n, (n,), generator=generator)
        a = auroc(scores[idx], labels[idx])
        if a is not None:
            values.append(a)
    if not values:
        return None
    t = torch.tensor(values)
    lo, hi = torch.quantile(t, torch.tensor([0.025, 0.975])).tolist()
    return [float(lo), float(hi)]


def slot_features(states: torch.Tensor, readout: torch.Tensor, vocab_ids: torch.Tensor, strings: list[str], native: torch.Tensor) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Per slot: [top-1 probability, entropy (nats), agreement of top-1 value with the native answer]; plus pairwise slot agreement.
    Returns (features [N, F], names, top1 value index [N, S])."""
    n, slots = states.shape[0], states.shape[1]
    values = torch.tensor([float(s) for s in strings])
    logits = torch.einsum("nsd,vd->nsv", states.float(), readout[vocab_ids].float())
    probs = logits.softmax(-1)
    top_p, top_i = probs.max(-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
    top_value = values[top_i]
    agree_native = (torch.isfinite(native).unsqueeze(1) & (top_value == native.unsqueeze(1))).float()
    cols, names = [], []
    for s in range(slots):
        cols += [top_p[:, s], entropy[:, s], agree_native[:, s]]
        names += [f"s{s}_top_p", f"s{s}_entropy", f"s{s}_agree_native"]
    for a in range(slots):
        for b in range(a + 1, slots):
            cols.append((top_i[:, a] == top_i[:, b]).float()); names.append(f"s{a}_s{b}_agree")
    return torch.stack(cols, 1), names, top_i


def _standardise(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = train.mean(0, keepdim=True), train.std(0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (test - mean) / std


def logistic_oof(x: torch.Tensor, y: torch.Tensor, *, folds: int = FOLDS, seed: int = SEED, l2: float = L2, steps: int = STEPS) -> torch.Tensor:
    """Out-of-fold scores of an L2 logistic regression (full-batch LBFGS-free gradient steps, seeded)."""
    n = x.shape[0]
    fold = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) % folds
    out = torch.empty(n)
    for f in range(folds):
        train = fold != f
        xt, xs = _standardise(x[train], x[~train])
        torch.manual_seed(seed + f)
        w = torch.zeros(x.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=5e-2)
        yt = y[train].float()
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * (w * w).sum()
            loss.backward(); opt.step()
        with torch.no_grad():
            out[~train] = xs @ w + b
    return out


def feature_set(names: list[str], slots: list[int]) -> list[int]:
    chosen = set(slots)
    return [i for i, n in enumerate(names) if (n.startswith("s") and "_s" not in n and int(n[1:n.index("_")]) in chosen)
            or ("_s" in n and all(int(p[1:]) in chosen for p in n.split("_")[:2]))]


def analyze(run_dir: Path, tokenizer, embeddings: torch.Tensor, readout: torch.Tensor, *, carriers: list[int], inert: list[int],
            folds: int = FOLDS, seed: int = SEED) -> dict:
    payload = torch.load(Path(run_dir) / "readout_states.pt", map_location="cpu", weights_only=False)
    states, correct, native = payload["states"].float(), payload["correct"].bool(), payload["native"]
    wrong = ~correct
    vocab_ids, strings = numeric_vocabulary(tokenizer, embeddings)
    feats, names, top_i = slot_features(states, readout, vocab_ids, strings, native)
    slots = list(range(states.shape[1]))
    sets = {"carriers": feature_set(names, carriers), "inert": feature_set(names, inert), "all_slots": list(range(len(names)))}
    conf = torch.stack([payload["first_token_prob"], payload["first_token_margin"]], 1)
    conf = torch.nan_to_num(conf, nan=float(conf[torch.isfinite(conf[:, 0])].mean()) if torch.isfinite(conf[:, 0]).any() else 0.0)
    report = {"run": str(run_dir), "rows": int(correct.numel()), "wrong": int(wrong.sum()), "accuracy": float(correct.double().mean()),
              "carriers": carriers, "inert": inert, "folds": folds, "classifiers": {}}
    candidates = {"confidence": conf, **{k: feats[:, idx] for k, idx in sets.items() if idx},
                  "carriers_plus_confidence": torch.cat([feats[:, sets["carriers"]], conf], 1) if sets["carriers"] else None,
                  "raw_states": states.reshape(states.shape[0], -1)}
    for name, x in candidates.items():
        if x is None or x.shape[1] == 0:
            report["classifiers"][name] = None; continue
        scores = logistic_oof(x, wrong, folds=folds, seed=seed)
        report["classifiers"][name] = {"features": int(x.shape[1]), "auroc": auroc(scores, wrong), "auroc_ci95": auroc_ci(scores, wrong, seed=seed)}
    # per-slot single-feature AUROCs for the placement picture
    report["per_slot"] = {str(s): {"agree_native_auroc": auroc(1 - feats[:, names.index(f"s{s}_agree_native")], wrong),
                                   "entropy_auroc": auroc(feats[:, names.index(f"s{s}_entropy")], wrong),
                                   "top_p_auroc": auroc(1 - feats[:, names.index(f"s{s}_top_p")], wrong)} for s in slots}
    base = report["classifiers"]["confidence"]["auroc"] if report["classifiers"].get("confidence") else None
    car = report["classifiers"].get("carriers"); ine = report["classifiers"].get("inert"); raw = report["classifiers"].get("raw_states")
    report["checks"] = {
        "p10_carriers_beat_confidence_by_0_05": None if not car or base is None else car["auroc"] - base >= 0.05,
        "p10_carriers_beat_inert_by_0_05": None if not car or not ine else car["auroc"] - ine["auroc"] >= 0.05,
        "p10_carriers_reach_80pct_of_raw_gain": None if not car or not raw or base is None or raw["auroc"] - base <= 0 else (car["auroc"] - base) >= 0.8 * (raw["auroc"] - base),
    }
    return report


def markdown(report: dict) -> str:
    f = lambda v: "—" if v is None else f"{v:.3f}"
    lines = [f"run `{Path(report['run']).name}`: {report['rows']} rows, {report['wrong']} wrong (accuracy {report['accuracy']:.3f}); carriers {report['carriers']}, inert {report['inert']}", "",
             "| classifier | features | AUROC (wrong answer) [95% CI] |", "|---|---|---|"]
    for name, c in report["classifiers"].items():
        if c:
            ci = c["auroc_ci95"]; lines.append(f"| {name} | {c['features']} | {f(c['auroc'])}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") + " |")
    lines += ["", "| slot | AUROC: disagrees with answer | entropy | 1 − top-p |", "|---|---|---|---|"]
    for s, v in report["per_slot"].items():
        lines.append(f"| {s} | {f(v['agree_native_auroc'])} | {f(v['entropy_auroc'])} | {f(v['top_p_auroc'])} |")
    lines += ["", "checks: " + json.dumps(report["checks"])]
    return "\n".join(lines)


def main():
    from scripts.analyze_monitor_faithfulness import load_model_pieces
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--card", type=Path, help="card.json whose carrying and inert slots define the feature sets")
    parser.add_argument("--carriers", default=None, help="comma list; overrides the card")
    parser.add_argument("--inert", default=None, help="comma list; overrides the card")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    carriers = inert = None
    if args.card:
        card = json.loads(args.card.read_text()); carriers = card["verdict"]["carrying_slots"]; inert = card["verdict"]["inert_slots"]
    if args.carriers is not None:
        carriers = [int(s) for s in args.carriers.split(",") if s.strip()]
    if args.inert is not None:
        inert = [int(s) for s in args.inert.split(",") if s.strip()]
    if carriers is None or inert is None:
        parser.error("give --card or both --carriers and --inert")
    tokenizer, embeddings, readout = load_model_pieces(args.config)
    report = analyze(args.run_dir, tokenizer, embeddings, readout, carriers=carriers, inert=inert)
    print(markdown(report))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
