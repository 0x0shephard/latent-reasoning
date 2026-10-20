"""Paired summary of a thought-steering run (ledger §145 / §147): per arm, the action and reasoning
verdict differences to the arm's clean condition with paired bootstrap intervals, the share of texts
identical to the clean run, the probe reading, the degradation proxy, and the size of each shift
relative to the latent vectors it was added to (the probe gap is a few tenths of a unit on vectors
whose norm is in the hundreds, which is why the §137 shift arms could not move anything)."""
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

from src.mech.latent_backdoor import answer_text, string_judge

CLEAN_OF = {"control_plus": "control", "baseline_plus": "baseline", "release_minus": "release", "armed_minus": "armed"}


def clean_condition(arm: str) -> str:
    for prefix, cond in CLEAN_OF.items():
        if arm.startswith(prefix):
            return cond
    raise KeyError(arm)


def paired(d: np.ndarray, *, resamples: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    bs = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(resamples)]
    return {"delta": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def summarize(summary_path: Path, responses_path: Path, directions_path: Path | None) -> dict:
    s = json.loads(summary_path.read_text()); rep = s["report"]
    rows = [json.loads(l) for l in responses_path.read_text().splitlines() if l.strip()]
    imm = lambda run, key: np.array([1.0 if r["runs"][run][key] == "immoral" else 0.0 for r in rows])
    has_reasoning = all(r["runs"]["clean_control"].get("reasoning") is not None for r in rows[:5])
    out = {"behaviour": rep["behaviour"], "n": len(rows), "judge": rep["judge"].get("primary"), "arms": {}}
    for arm in rep["arms"]:
        c = clean_condition(arm); clean = f"clean_{c}"
        block = {"clean": c, "action": paired(imm(arm, "action") - imm(clean, "action")), "action_rate": float(imm(arm, "action").mean()),
                 "identical_text_share": float(np.mean([r["runs"][arm]["text"] == r["runs"][clean]["text"] for r in rows])),
                 "reading": rep["reading"][arm]["release_side"], "reading_clean": rep["reading"][clean]["release_side"],
                 "degradation": rep["arms"][arm]["degradation"], "degradation_clean": rep["clean_degradation"][c]}
        if has_reasoning:
            block["reasoning"] = paired(imm(arm, "reasoning") - imm(clean, "reasoning"))
        d_read = block["reading"] - block["reading_clean"]
        block["efficacy_action_over_reading"] = block["action"]["delta"] / d_read if abs(d_read) > 1e-6 else None
        out["arms"][arm] = block
    if directions_path is not None and directions_path.exists() and directions_path.stat().st_size > 0:
        d = torch.load(directions_path, map_location="cpu", weights_only=False)
        z = d["z_clean"]["control"].float(); norms = z.norm(dim=2).mean(0)                          # [6]
        gaps = d["gaps"].float()
        out["shift_scale"] = {"mean_latent_norm": norms.tolist(), "probe_gap": gaps.tolist(), "probe_gap_over_norm": (gaps / norms).tolist()}
        if "mean_difference_norms" in d:
            md = d["mean_difference_norms"].float()
            out["shift_scale"]["mean_difference_norm"] = md.tolist(); out["shift_scale"]["mean_difference_over_norm"] = (md / norms).tolist()
            out["shift_scale"]["cosine_probe_vs_mean_difference"] = rep.get("cosine_probe_vs_mean_difference")
    return out


def markdown(o: dict, name: str) -> str:
    lines = [f"### Thought steering, {name}: n = {o['n']}, judge {o['judge']}", "",
             "| arm | action immoral | Δ action [95% CI] | Δ reasoning | identical texts | reading (clean → arm) | unknown / words |", "|---|---|---|---|---|---|---|"]
    for arm, b in o["arms"].items():
        a = b["action"]; rz = b.get("reasoning"); dg = b["degradation"]
        lines.append(f"| {arm} | {b['action_rate']:.3f} | {a['delta']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | "
                     f"{'—' if rz is None else format(rz['delta'], '+.3f')} | {b['identical_text_share']:.2f} | {b['reading_clean']:.2f} → {b['reading']:.2f} | {dg['string_unknown']:.2f} / {dg['mean_words']:.0f} |")
    if "shift_scale" in o:
        ss = o["shift_scale"]
        lines += ["", f"Probe gap over mean latent norm, per latent: {[round(x, 4) for x in ss['probe_gap_over_norm']]}"]
        if "mean_difference_over_norm" in ss:
            lines.append(f"Mean-difference norm over mean latent norm: {[round(x, 4) for x in ss['mean_difference_over_norm']]}; cosine(probe, mean difference): {[round(x, 3) for x in (ss.get('cosine_probe_vs_mean_difference') or [])]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="directory holding summary.json, responses.jsonl and directions.pt")
    parser.add_argument("--name", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    o = summarize(args.dir / "summary.json", args.dir / "responses.jsonl", args.dir / "directions.pt")
    md = markdown(o, args.name)
    print(md)
    if args.output:
        args.output.write_text(json.dumps(o, indent=2) + "\n"); args.output.with_suffix(".md").write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
