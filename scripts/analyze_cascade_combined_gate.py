"""§133 D: the latent-to-explicit cascade with a combined gate.  No decoding.

Out-of-fold ridge scores for "the native answer is wrong" from the first-token confidence plus
the six thought states (top principal directions), fed to the §131 cascade scorer in place of
the raw confidence, so the routed fraction and the net gain can be compared gate against gate.
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

from scripts.analyze_latent_explicit_cascade import FRACTIONS, analyze as cascade

FOLDS, SEED, PCA = 5, 0, 64


def oof_wrong_scores(states: torch.Tensor, prob: np.ndarray, margin: np.ndarray, wrong: np.ndarray, *, use_states: bool, folds: int = FOLDS, seed: int = SEED, lam: float = 1.0) -> np.ndarray:
    conf = np.stack([np.nan_to_num(prob, nan=0.0), np.nan_to_num(margin, nan=0.0), np.isnan(prob).astype(float)], axis=1)
    if use_states:
        st = states.float().numpy().reshape(len(prob), -1); stc = st - st.mean(0)
        U, S, Vt = np.linalg.svd(stc, full_matrices=False); X = np.hstack([conf, stc @ Vt[:PCA].T])
    else:
        X = conf
    rng = np.random.default_rng(seed); order = rng.permutation(len(wrong)); out = np.zeros(len(wrong))
    for f in range(folds):
        test = order[f::folds]; train = np.setdiff1d(order, test)
        mu = X[train].mean(0); sd = X[train].std(0) + 1e-6
        Xt = np.hstack([(X[train] - mu) / sd, np.ones((len(train), 1))]); Xs = np.hstack([(X[test] - mu) / sd, np.ones((len(test), 1))])
        W = np.linalg.solve(Xt.T @ Xt + lam * len(train) * np.eye(Xt.shape[1]), Xt.T @ wrong[train].astype(float))
        out[test] = Xs @ W
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True); parser.add_argument("--targets", type=Path, required=True); parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--questions-json", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--name", default="")
    args = parser.parse_args()
    d = torch.load(args.readout, map_location="cpu", weights_only=False)
    prob = np.asarray(d["first_token_prob"], dtype=float); margin = np.asarray(d["first_token_margin"], dtype=float); wrong = ~np.asarray(d["correct"], dtype=bool)
    questions = json.loads(args.questions_json.read_text()); targets = [json.loads(l) for l in args.targets.read_text().splitlines() if l.strip()]; summary = json.loads(args.summary.read_text())
    report = {"gates": {}}
    for name, scores in (("confidence", prob), ("oof_confidence_ridge", -oof_wrong_scores(d["states"], prob, margin, wrong, use_states=False)),
                         ("oof_confidence_plus_states", -oof_wrong_scores(d["states"], prob, margin, wrong, use_states=True))):
        r = {"first_token_prob": scores, "first_token_margin": scores, "native": d["native"]}
        c = cascade(questions, r, targets, summary, resamples=500)["gates"]["prob"]
        report["gates"][name] = {"at_fraction": c["at_fraction"], "ci95": c["ci95"], "peak": c["peak"], "fraction_at_peak": c["fraction_at_peak"]}
    base = report["gates"]["confidence"]["at_fraction"]["0.25"]; comb = report["gates"]["oof_confidence_plus_states"]["at_fraction"]["0.25"]
    report["checks"] = {"p22_combined_gate_le_plus_1_at_25pct": comb - base <= 1.0, "gain_at_25pct": comb - base}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"### {args.name}"); print("| gate | " + " | ".join(f"{100 * t:.0f}%" for t in FRACTIONS) + " | peak |"); print("|---|" + "---|" * (len(FRACTIONS) + 1))
    for g, e in report["gates"].items():
        print(f"| {g} | " + " | ".join(f"{e['at_fraction'][str(t)]:+.1f}" for t in FRACTIONS) + f" | {e['peak']:+.1f} at {100 * e['fraction_at_peak']:.0f}% |")
    print(json.dumps(report["checks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
