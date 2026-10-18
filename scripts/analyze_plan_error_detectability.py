"""Are plan errors more or less detectable from confidence than latent slips? (ledger §138; no new runs)

Joins the §125/§131 readout (per GSM8K test question: first-token probability and margin of the
model's own answer, correctness, step count) with the calculator repair targets (§129/§130/§134:
every wrong ≥3-step problem with its failure class and the model's explicit-chain answer).  For each
failure class: confidence statistics, the AUROC of low confidence against the correct problems of
the same ≥3-step pool, the share routed at the cascade's 10 / 25 / 50% thresholds, and the share
whose explicit chain is right.  Bootstrap over rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_latent_explicit_cascade import explicit_answer, load_questions
from src.mech.latent_hijack import auroc

CLASSES = ("plan_wrong", "latent_only", "upstream_arithmetic", "final_step")
FRACTIONS = (0.10, 0.25, 0.50)
MODELS = {
    "gpt2": "configs/official_codi_gpt2.yaml", "codi1b": "configs/official_codi_llama1b.yaml", "simcot": "configs/official_simcot_codi_llama1b.yaml",
    "codi3b": "configs/official_simcot_codi_llama3b.yaml", "codi8b": "configs/official_simcot_codi_llama8b.yaml",
}


def ci(values: np.ndarray, fn, *, resamples: int, seed: int) -> list[float] | None:
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    stats = [fn(values[rng.integers(0, len(values), len(values))]) for _ in range(resamples)]
    stats = [s for s in stats if s is not None and not (isinstance(s, float) and math.isnan(s))]
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))] if stats else None


def auroc_ci(pos: np.ndarray, neg: np.ndarray, *, resamples: int, seed: int) -> dict:
    """AUROC of low confidence (score = −prob) for pos against neg, bootstrap over both sets."""
    point = auroc((-pos).tolist() + (-neg).tolist(), [1] * len(pos) + [0] * len(neg))
    rng = np.random.default_rng(seed); stats = []
    for _ in range(resamples):
        p = pos[rng.integers(0, len(pos), len(pos))]; n = neg[rng.integers(0, len(neg), len(neg))]
        a = auroc((-p).tolist() + (-n).tolist(), [1] * len(p) + [0] * len(n))
        if a is not None:
            stats.append(a)
    return {"auroc": point, "ci95": [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))] if stats else None, "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def analyze(questions: list[str], readout: dict, targets: list[dict], *, resamples: int = 2000, seed: int = 0) -> dict:
    index = {q: i for i, q in enumerate(questions)}
    prob = np.asarray(readout["first_token_prob"], dtype=float)
    steps = np.asarray(readout["steps"], dtype=int); correct = np.asarray(readout["correct"], dtype=bool)
    pool = (steps >= 3) & ~np.isnan(prob)                       # the cascade's long-chain pool, as recorded
    thresholds = {str(f): float(np.quantile(prob[pool], f)) for f in FRACTIONS}
    neg = prob[pool & correct]
    rows = {c: [] for c in CLASSES}
    for t in targets:
        if t["control"] or t.get("failure_class") not in CLASSES:
            continue
        i = index.get(t["question"])
        if i is None or np.isnan(prob[i]):
            continue
        rows[t["failure_class"]].append({"prob": float(prob[i]), "explicit_right": abs((explicit_answer(t.get("tool_chain")) or float("nan")) - float(t["gold"])) < 1e-6})
    out = {"pool": {"n": int(pool.sum()), "correct": int((pool & correct).sum()), "accuracy": float(correct[pool].mean()), "thresholds": thresholds,
                    "correct_prob_mean": float(neg.mean()), "correct_prob_median": float(np.median(neg))}, "classes": {}}
    for c in CLASSES:
        p = np.asarray([r["prob"] for r in rows[c]], dtype=float)
        if len(p) == 0:
            out["classes"][c] = {"n": 0}; continue
        routed = {f: {"share": float((p < thresholds[f]).mean()), "ci95": ci(p, lambda v, th=thresholds[f]: float((v < th).mean()), resamples=resamples, seed=seed)} for f in thresholds}
        out["classes"][c] = {"n": int(len(p)), "prob_mean": float(p.mean()), "prob_median": float(np.median(p)),
                             "prob_mean_ci95": ci(p, lambda v: float(v.mean()), resamples=resamples, seed=seed),
                             "auroc_low_confidence_vs_correct": auroc_ci(p, neg, resamples=resamples, seed=seed), "routed": routed,
                             "explicit_chain_right": float(np.mean([r["explicit_right"] for r in rows[c]]))}
    pw = np.asarray([r["prob"] for r in rows["plan_wrong"]]); lo = np.asarray([r["prob"] for r in rows["latent_only"]])
    if len(pw) and len(lo):
        out["plan_wrong_vs_latent_only"] = {"auroc_low_confidence_identifies_plan_wrong": auroc_ci(pw, lo, resamples=resamples, seed=seed),
                                            "mean_prob_difference_plan_minus_latent": float(pw.mean() - lo.mean())}
    return out


def markdown(reports: dict) -> str:
    lines = ["# Plan errors vs latent slips under the confidence gate (ledger §138)", "",
             "| model | class | n | mean conf | AUROC low-conf vs correct [95%] | routed @10% | @25% | @50% | explicit chain right |", "|---|---|---|---|---|---|---|---|---|"]
    for m, r in reports.items():
        lines.append(f"| {m} | correct (pool) | {r['pool']['correct']} | {r['pool']['correct_prob_mean']:.3f} | — | 0.10 | 0.25 | 0.50 | — |")
        for c, b in r["classes"].items():
            if b.get("n", 0) == 0:
                continue
            a = b["auroc_low_confidence_vs_correct"]
            lines.append(f"| {m} | {c} | {b['n']} | {b['prob_mean']:.3f} | {a['auroc']:.3f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}] | "
                         f"{b['routed']['0.1']['share']:.2f} | {b['routed']['0.25']['share']:.2f} | {b['routed']['0.5']['share']:.2f} | {b['explicit_chain_right']:.2f} |")
        if "plan_wrong_vs_latent_only" in r:
            x = r["plan_wrong_vs_latent_only"]
            lines.append(f"| {m} | plan_wrong vs latent_only | | Δmean {x['mean_prob_difference_plan_minus_latent']:+.3f} | low-conf → plan_wrong {x['auroc_low_confidence_identifies_plan_wrong']['auroc']:.3f} | | | | |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--readout-root", type=Path, default=ROOT / "outputs" / "kaggle_runs" / "readout")
    parser.add_argument("--targets-root", type=Path, default=ROOT / "docs" / "latent_monitor_audit" / "latent_repair_ladder")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "latent_monitor_audit" / "analyses" / "plan_error_detectability.json")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    import torch
    reports = {}
    for m in args.models:
        readout = torch.load(args.readout_root / m / f"readout_{m}" / "readout_states.pt", map_location="cpu", weights_only=False)
        questions = load_questions(ROOT / MODELS[m])
        targets = [json.loads(l) for l in (args.targets_root / f"{m}_tool.targets.jsonl").read_text().splitlines() if l.strip()]
        reports[m] = analyze(questions, readout, targets, resamples=args.resamples)
        print(m, json.dumps({c: {"n": b.get("n"), "mean": round(b.get("prob_mean", float("nan")), 3), "auroc": round(b.get("auroc_low_confidence_vs_correct", {}).get("auroc") or float("nan"), 3),
                                 "routed25": round(b.get("routed", {}).get("0.25", {}).get("share", float("nan")), 2)} for c, b in reports[m]["classes"].items()}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(reports))
    print(markdown(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
