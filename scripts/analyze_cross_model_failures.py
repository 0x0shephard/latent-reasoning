"""§133 A: are latent-only failures slips or hard problems?  No decoding; joins files on disk.

Per-problem latent correctness per model from the readout runs (same GSM8K test order), the
failure class of every wrong >=3-step problem from the calculator repair targets, joined by
question.  Asks whether one model's latent-only failures are answered by the others (slips)
or shared (hard problems), and what a vote would recover.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSES = ("latent_only", "plan_wrong", "upstream_arithmetic", "final_step")


def load_readout(path: Path) -> dict:
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    return {"correct": np.asarray(d["correct"], dtype=bool), "steps": np.asarray(d["steps"]), "prob": np.asarray(d["first_token_prob"], dtype=float),
            "margin": np.asarray(d["first_token_margin"], dtype=float)}


def classes_by_question(targets: list[dict]) -> dict[str, str]:
    return {t["question"]: t["failure_class"] for t in targets if not t["control"]}


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def independence_jaccard(pa: float, pb: float) -> float:
    return pa * pb / max(1e-9, pa + pb - pa * pb)


def analyze(questions: list[str], readouts: dict[str, dict], classes: dict[str, dict[str, str]], *, min_steps: int = 3) -> dict:
    models = list(readouts)
    steps = next(iter(readouts.values()))["steps"]
    rows = np.asarray([i for i in range(len(questions)) if steps[i] >= min_steps])
    correct = {m: readouts[m]["correct"][rows] for m in models}
    report = {"rows": int(len(rows)), "models": models, "accuracy": {m: float(correct[m].mean()) for m in models}, "solved_by_others": {}, "overlap": {}, "vote": {}, "unique_latent_only": {}}
    for m in models:
        cls = np.asarray([classes[m].get(questions[i], "correct" if readouts[m]["correct"][i] else "unclassified") for i in rows])
        report["solved_by_others"][m] = {}
        for c in CLASSES + ("unclassified",):
            mask = (cls == c) & ~correct[m]
            if mask.sum() == 0:
                continue
            report["solved_by_others"][m][c] = {"n": int(mask.sum()), **{o: float(correct[o][mask].mean()) for o in models if o != m},
                                                "own_explicit": 1.0 if c == "latent_only" else 0.0}
        lo = (cls == "latent_only") & ~correct[m]
        others = [o for o in models if o != m]
        if lo.sum():
            none_other_fails = np.all(np.stack([correct[o][lo] for o in others]), axis=0)
            report["unique_latent_only"][m] = {"n": int(lo.sum()), "share_all_others_right": float(none_other_fails.mean()),
                                               "share_no_other_right": float((~np.any(np.stack([correct[o][lo] for o in others]), axis=0)).mean())}
    for a, b in itertools.combinations(models, 2):
        fa, fb = ~correct[a], ~correct[b]
        report["overlap"][f"{a}|{b}"] = {"jaccard": jaccard(fa, fb), "independence": independence_jaccard(float(fa.mean()), float(fb.mean())),
                                         "shared_failures": int((fa & fb).sum())}
    small = [m for m in models if m != "codi3b"]
    if len(small) >= 3:
        votes = np.stack([correct[m] for m in small]).sum(0)
        report["vote"]["majority_of_small"] = float((votes >= 2).mean()); report["vote"]["best_single_small"] = max(float(correct[m].mean()) for m in small)
    report["vote"]["any_model_right"] = float(np.any(np.stack([correct[m] for m in models]), axis=0).mean())
    report["vote"]["all_models_wrong"] = int((~np.any(np.stack([correct[m] for m in models]), axis=0)).sum())
    return report


def checks_from(report: dict) -> dict:
    s = report["solved_by_others"]; acc = report["accuracy"]
    small = [m for m in report["models"] if m != "codi3b"]
    p19a_lo = all(s[m].get("latent_only", {}).get(o, 0.0) >= acc[o] for m in small for o in small if o != m and "latent_only" in s[m])
    p19a_pw = all(s[m].get("plan_wrong", {}).get(o, 1.0) <= 0.35 for m in small for o in small if o != m and "plan_wrong" in s[m])
    ov = report["overlap"].get("codi1b|simcot")
    vote = report["vote"]
    return {"p19a_latent_only_solved_by_others_at_or_above_base": p19a_lo, "p19a_plan_wrong_solved_by_others_le_0_35": p19a_pw,
            "p19b_codi1b_simcot_jaccard_exceeds_independence_by_0_10": None if ov is None else ov["jaccard"] - ov["independence"] >= 0.10,
            "p19c_majority_vote_gain_le_2_points": None if "majority_of_small" not in vote else 100 * (vote["majority_of_small"] - vote["best_single_small"]) <= 2.0}


def markdown(report: dict) -> str:
    models = report["models"]
    lines = [f"{report['rows']} rows with >= 3 steps; accuracy " + ", ".join(f"{m} {100 * report['accuracy'][m]:.1f}" for m in models), "",
             "Share of a model's failures (by class) that each other model answers correctly:", "",
             "| model | class | n | " + " | ".join(models) + " |", "|---|---|---|" + "---|" * len(models)]
    for m in models:
        for c, e in report["solved_by_others"][m].items():
            lines.append(f"| {m} | {c} | {e['n']} | " + " | ".join("—" if o == m else f"{100 * e[o]:.0f}%" for o in models) + " |")
    lines += ["", "Failure-set overlap (Jaccard vs independence):", ""]
    for k, v in report["overlap"].items():
        lines.append(f"- {k}: {v['jaccard']:.3f} vs {v['independence']:.3f} ({v['shared_failures']} shared)")
    lines += ["", "Latent-only failures that every other model gets right / that no other model gets right:", ""]
    for m, v in report["unique_latent_only"].items():
        lines.append(f"- {m}: {100 * v['share_all_others_right']:.0f}% / {100 * v['share_no_other_right']:.0f}% of {v['n']}")
    v = report["vote"]
    lines += ["", f"Vote: majority of the 1B-and-smaller models {100 * v.get('majority_of_small', float('nan')):.1f} (best single {100 * v.get('best_single_small', float('nan')):.1f}); any model right {100 * v['any_model_right']:.1f}; all wrong {v['all_models_wrong']}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", action="append", required=True, help="model=path/to/readout_states.pt")
    parser.add_argument("--targets", action="append", required=True, help="model=path/to/targets.jsonl (calculator run)")
    parser.add_argument("--questions-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = json.loads(args.questions_json.read_text())
    readouts = {k: load_readout(Path(v)) for k, v in (s.split("=", 1) for s in args.readout)}
    classes = {k: classes_by_question([json.loads(l) for l in Path(v).read_text().splitlines() if l.strip()]) for k, v in (s.split("=", 1) for s in args.targets)}
    report = analyze(questions, readouts, classes); report["checks"] = checks_from(report)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown(report)); print(json.dumps(report["checks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
