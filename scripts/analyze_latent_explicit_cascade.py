"""Build 6 (ledger §131): a confidence-gated cascade from latent to explicit reasoning.  No training.

Joins the §125 readout (per GSM8K test question: native answer, first-token probability and
margin under the forced cue) with the §129 repair targets (per wrong >=3-step problem and per
sampled control: the model's own calculator-corrected explicit chain and its answer).  For a
threshold on the latent confidence, problems below it are routed to the explicit chain; the
curve of net accuracy change against the fraction routed says how much of the full-switch
gain survives when the chain is run only where the latent pass is unsure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTRACT = "latent_explicit_cascade_v1"
FRACTIONS = (0.10, 0.25, 0.50, 1.0)
_ANSWER = re.compile(r"The answer is:\s*(-?[\d,]*\.?\d+)")


def explicit_answer(chain: str | None) -> float | None:
    m = _ANSWER.search(chain or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) < 1e-6


def join_rows(questions: list[str], readout: dict, targets: list[dict], *, wrong_total: int, eligible: int) -> list[dict]:
    """One row per repair target with the readout's confidence attached; weights scale the sample to the population."""
    index = {q: i for i, q in enumerate(questions)}
    n_wrong = sum(not t["control"] for t in targets); n_ctrl = sum(bool(t["control"]) for t in targets)
    w_wrong = wrong_total / n_wrong if n_wrong else 0.0; w_ctrl = (eligible - wrong_total) / n_ctrl if n_ctrl else 0.0
    rows, mismatched = [], 0
    for t in targets:
        i = index.get(t["question"])
        if i is None:
            raise KeyError(f"target question not in the readout: {t['question'][:60]!r}")
        prob = float(readout["first_token_prob"][i]); margin = float(readout["first_token_margin"][i])
        native = float(readout["native"][i]); native = None if math.isnan(native) else native
        if not _same(native, t["native"]) and not (native is None and t["native"] is None):
            mismatched += 1
        rows.append({"index": i, "control": bool(t["control"]), "gold": float(t["gold"]), "native": t["native"],
                     "explicit": explicit_answer(t.get("tool_chain")), "prob": prob, "margin": margin,
                     "weight": w_ctrl if t["control"] else w_wrong})
    return rows, mismatched


def gain_of(row: dict) -> float:
    """Change in correct count when this row is routed to its explicit chain (no chain: keep the latent answer)."""
    if row["explicit"] is None:
        return 0.0
    if row["control"]:
        return 0.0 if _same(row["explicit"], row["gold"]) else -1.0
    return 1.0 if _same(row["explicit"], row["gold"]) else 0.0


def curve(rows: list[dict], gate: str, eligible: int) -> tuple[np.ndarray, np.ndarray]:
    """Routed fraction and net points, routing in increasing order of the gate (NaN gate = route first)."""
    key = np.array([(-np.inf if math.isnan(r[gate]) else r[gate]) for r in rows])
    order = np.argsort(key, kind="stable")
    w = np.array([rows[i]["weight"] for i in order]); g = np.array([gain_of(rows[i]) * rows[i]["weight"] for i in order])
    return np.concatenate([[0.0], np.cumsum(w) / eligible]), np.concatenate([[0.0], 100.0 * np.cumsum(g) / eligible])


def at_fraction(fractions: np.ndarray, points: np.ndarray, target: float) -> float:
    """Net points at the largest routed fraction not above ``target``."""
    k = int(np.searchsorted(fractions, target + 1e-12, side="right") - 1)
    return float(points[max(k, 0)])


def fraction_to_capture(fractions: np.ndarray, points: np.ndarray, share: float) -> float | None:
    full = points[-1]
    if full <= 0:
        return None
    hit = np.nonzero(points >= share * full)[0]
    return float(fractions[hit[0]]) if len(hit) else None


def summarize(rows: list[dict], gate: str, eligible: int) -> dict:
    f, p = curve(rows, gate, eligible)
    return {"at_fraction": {str(t): at_fraction(f, p, t) for t in FRACTIONS}, "full_switch": float(p[-1]),
            "fraction_for_half": fraction_to_capture(f, p, 0.5), "fraction_for_90": fraction_to_capture(f, p, 0.9),
            "peak": float(p.max()), "fraction_at_peak": float(f[int(p.argmax())])}


def bootstrap(rows: list[dict], gate: str, eligible: int, *, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    wrong = [r for r in rows if not r["control"]]; ctrl = [r for r in rows if r["control"]]
    stats = {str(t): [] for t in FRACTIONS}
    for _ in range(resamples):
        sample = [wrong[i] for i in rng.integers(0, len(wrong), len(wrong))] + [ctrl[i] for i in rng.integers(0, len(ctrl), len(ctrl))]
        f, p = curve(sample, gate, eligible)
        for t in FRACTIONS:
            stats[str(t)].append(at_fraction(f, p, t))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def disagreement_reference(rows: list[dict], eligible: int) -> dict:
    """The §126–§129 trigger: switch to the chain only where it disagrees with the latent answer (the chain runs everywhere)."""
    def disagrees(r):
        return r["explicit"] is not None and not _same(r["explicit"], r["native"])
    gain = sum(gain_of(r) * r["weight"] for r in rows if disagrees(r))
    return {"net_points": 100.0 * gain / eligible, "fraction_switched": sum(r["weight"] for r in rows if disagrees(r)) / eligible}


def analyze(questions: list[str], readout: dict, targets: list[dict], summary: dict, *, resamples: int = 2000, seed: int = 0) -> dict:
    t = summary["report"]["targets"]; eligible = int(t["eligible"]); wrong_total = int(t["wrong_total"])
    rows, mismatched = join_rows(questions, readout, targets, wrong_total=wrong_total, eligible=eligible)
    oracle = 100.0 * sum(gain_of(r) * r["weight"] for r in rows if not r["control"]) / eligible
    report = {"contract": CONTRACT, "eligible": eligible, "wrong_total": wrong_total, "rows": len(rows), "native_mismatch": mismatched,
              "latent_accuracy": 1 - wrong_total / eligible, "gates": {}, "oracle": {"net_points": oracle, "fraction_routed": wrong_total / eligible},
              "disagreement_trigger": disagreement_reference(rows, eligible)}
    for gate in ("prob", "margin"):
        s = summarize(rows, gate, eligible); s["ci95"] = bootstrap(rows, gate, eligible, resamples=resamples, seed=seed)
        report["gates"][gate] = s
    return report


def markdown(report: dict, name: str = "") -> str:
    g = report["gates"]["prob"]; m = report["gates"]["margin"]
    lines = [f"### Cascade {name}: {report['eligible']} long-chain problems, latent accuracy {100 * report['latent_accuracy']:.1f}%", "",
             "| routed | probability gate (net points) [95% CI] | margin gate |", "|---|---|---|"]
    for t in FRACTIONS:
        k = str(t); lo, hi = g["ci95"][k]
        lines.append(f"| {100 * t:.0f}% | {g['at_fraction'][k]:+.1f} [{lo:+.1f}, {hi:+.1f}] | {m['at_fraction'][k]:+.1f} |")
    half = g["fraction_for_half"]; ninety = g["fraction_for_90"]
    lines += ["", f"- full switch {g['full_switch']:+.1f}; half of it at {100 * half:.0f}% routed, 90% at {100 * ninety:.0f}% routed" if half is not None and ninety is not None else f"- full switch {g['full_switch']:+.1f}",
              f"- peak {g['peak']:+.1f} at {100 * g['fraction_at_peak']:.0f}% routed",
              f"- oracle (route exactly the wrong problems, {100 * report['oracle']['fraction_routed']:.0f}%): {report['oracle']['net_points']:+.1f}",
              f"- disagreement trigger (chain run everywhere, {100 * report['disagreement_trigger']['fraction_switched']:.0f}% switched): {report['disagreement_trigger']['net_points']:+.1f}",
              f"- native answers that differ between the readout and repair runs: {report['native_mismatch']}"]
    return "\n".join(lines)


def load_questions(config: Path) -> list[str]:
    from src.data.chain_pools import load_raw_eval_rows
    from src.utils.config import load_config
    cfg = load_config(str(config)); data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    return [str(r["question"]) for r in load_raw_eval_rows(dict(data_cfg.eval.gsm8k))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True, help="readout_states.pt from scripts/run_codi_correctness_readout.py")
    parser.add_argument("--targets", type=Path, required=True, help="targets.jsonl from a calculator-value repair run")
    parser.add_argument("--summary", type=Path, required=True, help="summary.json of that repair run (population sizes)")
    parser.add_argument("--config", type=Path, help="model config; the test questions are loaded in the recorder's order")
    parser.add_argument("--questions-json", type=Path, help="a JSON list of the test questions in the recorder's order (overrides --config)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    import torch
    readout = torch.load(args.readout, map_location="cpu", weights_only=False)
    questions = json.loads(args.questions_json.read_text()) if args.questions_json else load_questions(args.config)
    targets = [json.loads(line) for line in args.targets.read_text().splitlines() if line.strip()]
    report = analyze(questions, readout, targets, json.loads(args.summary.read_text()), resamples=args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown(report, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
