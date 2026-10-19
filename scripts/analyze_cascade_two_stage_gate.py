"""Ledger §140: the latent-to-explicit cascade with a two-stage gate.  No decoding.

The §131 cascade scorer, unchanged, fed with alternative gates built from the §139 detectors:
confidence (the §131 gate), the label-free transition residual, rank averages of confidence
with the residual or with the supervised all-slot score, and two-stage rules in which the least
confident q of the pool are routed first and the residual (or the state score) orders the rest.
All state scores are out of fold.  One report across the five models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_latent_explicit_cascade import FRACTIONS, analyze as cascade, load_questions
from scripts.analyze_latent_slip_detectability import FOLDS, SEED, supervised_oof, transition_residual
from scripts.analyze_plan_error_detectability import MODELS

STAGE_ONE = (0.10, 0.15)
RESAMPLES = 500


def ranks(x: np.ndarray) -> np.ndarray:
    """Average rank in [0, 1] over the non-NaN entries (ties averaged); NaN stays NaN."""
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    if ok.sum() == 0:
        return out
    v = x[ok]; order = np.argsort(v, kind="stable"); r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    # average tied ranks
    sv = v[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    out[ok] = r / max(len(v) - 1, 1)
    return out


def rank_average_gate(prob: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """A gate value (lower = routed first): minus the rank average of low confidence and high risk."""
    return -(ranks(-prob) + ranks(risk)) / 2


def two_stage_gate(prob: np.ndarray, risk: np.ndarray, q: float, pool: np.ndarray) -> np.ndarray:
    """Stage one: rows with probability below the pool's q-quantile, in confidence order.  Stage two: the
    rest, in decreasing risk.  Rows with NaN probability keep NaN (the scorer routes them first)."""
    gate = np.full(len(prob), np.nan)
    threshold = float(np.quantile(prob[pool], q)) if q > 0 else -np.inf
    first = pool & (prob < threshold)
    gate[first] = prob[first] - 10.0                       # below every stage-two value
    rest = pool & ~first
    gate[rest] = 1.0 - ranks(np.where(rest, risk, np.nan))[rest]
    return gate


def gates_for(readout: dict, *, seed: int = SEED) -> dict[str, np.ndarray]:
    states = readout["states"].float()
    prob = np.asarray(readout["first_token_prob"], dtype=float); steps = np.asarray(readout["steps"], dtype=int)
    correct = np.asarray(readout["correct"], dtype=bool)
    pool = (steps >= 3) & ~np.isnan(prob)
    rng = np.random.default_rng(seed); folds = rng.integers(0, FOLDS, len(prob))
    residual = transition_residual(states, pool & correct, folds)
    wrong_score = supervised_oof(states.flatten(1), (~correct).astype(int), folds, pool)
    residual = np.where(pool, residual, np.nan); wrong_score = np.where(pool, wrong_score, np.nan)
    gates = {"confidence": prob, "residual_only": -ranks(residual),
             "conf_rank_residual": rank_average_gate(prob, residual), "conf_rank_states": rank_average_gate(prob, wrong_score)}
    for q in STAGE_ONE:
        gates[f"two_stage_residual_q{int(100 * q)}"] = two_stage_gate(prob, residual, q, pool)
    gates["two_stage_states_q10"] = two_stage_gate(prob, wrong_score, STAGE_ONE[0], pool)
    return gates


def score_model(questions: list[str], readout: dict, targets: list[dict], summary: dict, *, resamples: int = RESAMPLES) -> dict:
    report = {"gates": {}}
    for name, g in gates_for(readout).items():
        r = {"first_token_prob": g, "first_token_margin": g, "native": readout["native"]}
        c = cascade(questions, r, targets, summary, resamples=resamples)
        report.setdefault("oracle", c["oracle"]); report.setdefault("eligible", c["eligible"]); report.setdefault("wrong_total", c["wrong_total"])
        e = c["gates"]["prob"]
        report["gates"][name] = {"at_fraction": e["at_fraction"], "ci95": e["ci95"], "peak": e["peak"], "fraction_at_peak": e["fraction_at_peak"]}
    base = report["gates"]["confidence"]["at_fraction"]
    for name, e in report["gates"].items():
        e["vs_confidence"] = {k: e["at_fraction"][k] - base[k] for k in ("0.25", "0.5")}
    return report


def checks_from(reports: dict) -> dict:
    """P30a–d over the models scored."""
    alt = [g for g in next(iter(reports.values()))["gates"] if g not in ("confidence", "residual_only")]
    best25 = {m: max(r["gates"][g]["vs_confidence"]["0.25"] for g in alt) for m, r in reports.items()}
    best50 = {m: max(r["gates"][g]["vs_confidence"]["0.5"] for g in alt) for m, r in reports.items()}
    ts = {m: r["gates"]["two_stage_residual_q10"]["vs_confidence"]["0.25"] for m, r in reports.items()}
    winners = [m for m, v in best25.items() if v >= 1.0]
    return {"best_gain_at_25pct": best25, "best_gain_at_50pct": best50, "two_stage_residual_q10_vs_confidence_at_25pct": ts,
            "p30a_no_gate_gains_over_1pt_at_25pct_on_3_models": sum(v > 1.0 for v in best25.values()) < 3,
            "p30b_two_stage_residual_q10_within_minus_1pt_at_25pct_everywhere": all(v >= -1.0 for v in ts.values()),
            "p30c_simcot_among_winners_if_any": (not winners) or ("simcot" in winners), "models_gaining_1pt_at_25pct": winners,
            "p30d_no_gate_gains_over_1pt_at_50pct_on_3_models": sum(v > 1.0 for v in best50.values()) < 3}


def markdown(reports: dict, checks: dict) -> str:
    lines = ["# The cascade with a two-stage gate (ledger §140)", "",
             "Net accuracy points on the ≥3-step pool at a routed fraction; confidence is the §131 gate. 'vs' = difference from confidence at 25% / 50%.", ""]
    for m, r in reports.items():
        lines += [f"### {m}: {r['eligible']} problems, oracle {r['oracle']['net_points']:+.1f} at {100 * r['oracle']['fraction_routed']:.0f}%", "",
                  "| gate | " + " | ".join(f"{100 * t:.0f}%" for t in FRACTIONS) + " | peak | vs confidence @25 / @50 |", "|---|" + "---|" * (len(FRACTIONS) + 2)]
        for g, e in r["gates"].items():
            cells = []
            for t in FRACTIONS:
                k = str(t); lo, hi = e["ci95"][k]
                cells.append(f"{e['at_fraction'][k]:+.1f} [{lo:+.1f}, {hi:+.1f}]" if t in (0.25, 0.5) else f"{e['at_fraction'][k]:+.1f}")
            v = e["vs_confidence"]
            lines.append(f"| {g} | " + " | ".join(cells) + f" | {e['peak']:+.1f} at {100 * e['fraction_at_peak']:.0f}% | {v['0.25']:+.1f} / {v['0.5']:+.1f} |")
        lines.append("")
    lines += ["## Checks", "", "```json", json.dumps(checks, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--readout-root", type=Path, default=ROOT / "outputs" / "kaggle_runs" / "readout")
    parser.add_argument("--targets-root", type=Path, default=ROOT / "docs" / "latent_monitor_audit" / "latent_repair_ladder")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "latent_monitor_audit" / "analyses" / "cascade_two_stage_gate.json")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    reports = {}
    for m in args.models:
        readout = torch.load(args.readout_root / m / f"readout_{m}" / "readout_states.pt", map_location="cpu", weights_only=False)
        questions = load_questions(ROOT / MODELS[m])
        targets = [json.loads(l) for l in (args.targets_root / f"{m}_tool.targets.jsonl").read_text().splitlines() if l.strip()]
        summary = json.loads((args.targets_root / f"{m}_tool.summary.json").read_text())
        reports[m] = score_model(questions, readout, targets, summary, resamples=args.resamples)
        print(m, {g: (round(e["at_fraction"]["0.25"], 2), round(e["at_fraction"]["0.5"], 2)) for g, e in reports[m]["gates"].items()}, flush=True)
    checks = checks_from(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"models": reports, "checks": checks}, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(reports, checks))
    print(markdown(reports, checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
