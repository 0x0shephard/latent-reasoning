"""§133 B: where does the one-slot write take, and where does it break?  No decoding.

Joins each calculator repair run's per-target outcomes with the readout's first-token
confidence (by question), and the correct controls with the competence-gated twin pairs
(whether the control's carrier transported under the swap).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COND = "state_nearest"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        idx = np.nonzero(vals == v)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def by_bins(conf: np.ndarray, flag: np.ndarray, k: int) -> list[dict]:
    """Rate of ``flag`` in k confidence bins of equal size, lowest confidence first."""
    order = np.argsort(conf, kind="stable"); out = []
    for part in np.array_split(order, k):
        out.append({"n": int(len(part)), "rate": float(flag[part].mean()) if len(part) else None,
                    "conf_range": [float(conf[part].min()), float(conf[part].max())] if len(part) else None})
    return out


def transported_controls(pairs: list[dict], predictions: dict, carriers: list[int]) -> dict[str, bool]:
    """question -> whether any carrier slot's state swap moved the answer to the twin's value (rows interleaved: 2p = question, 2p+1 = twin)."""
    target = np.asarray(predictions["target_gold"]); out = {}
    for p, pair in enumerate(pairs):
        for r, q in ((2 * p, pair["question"]), (2 * p + 1, pair["twin_question"])):
            out[q] = any(int(np.asarray(predictions[f"state_{c}"])[r]) == int(target[r]) for c in carriers)
    return out


def analyze(targets: list[dict], questions: list[str], readout: dict, *, transported: dict[str, bool] | None = None) -> dict:
    index = {q: i for i, q in enumerate(questions)}
    prob = np.asarray(readout["first_token_prob"], dtype=float); margin = np.asarray(readout["first_token_margin"], dtype=float)
    def conf(t, key="prob"):
        i = index[t["question"]]; v = (prob if key == "prob" else margin)[i]
        return -1.0 if np.isnan(v) else float(v)
    wrong = [t for t in targets if not t["control"]]; ctrl = [t for t in targets if t["control"]]
    touched = [t for t in wrong if t["has_donor"] and t["carrier_read"] != t["v_star"] and t["v_star"] != "__none__"]
    lo = [t for t in touched if t["failure_class"] == "latent_only"]
    report = {"targets": len(targets), "n_touched": len(touched), "n_touched_latent_only": len(lo), "n_controls": len(ctrl)}
    for name, rows in (("touched", touched), ("touched_latent_only", lo)):
        if len(rows) < 8:
            continue
        c = np.asarray([conf(t) for t in rows]); rep = np.asarray([t["outcomes"][COND] == "repaired" for t in rows])
        halves = by_bins(c, rep, 2); quart = by_bins(c, rep, 4)
        report[name] = {"repair_rate": float(rep.mean()), "by_half": halves, "by_quartile": quart,
                        "ratio_low_over_high_half": None if halves[1]["rate"] in (None, 0) else halves[0]["rate"] / halves[1]["rate"],
                        "auroc_confidence_for_repaired": auroc(-c, rep), "auroc_margin_for_repaired": auroc(-np.asarray([conf(t, "margin") for t in rows]), rep)}
    c = np.asarray([conf(t) for t in ctrl]); harm = np.asarray([t["outcomes"][COND] == "changed_wrong" for t in ctrl])
    quart = by_bins(c, harm, 4)
    report["controls_harm"] = {"harm_rate": float(harm.mean()), "by_quartile": quart,
                               "ratio_bottom_over_top_quartile": None if quart[-1]["rate"] in (None, 0) else quart[0]["rate"] / quart[-1]["rate"],
                               "auroc_low_confidence_for_harm": auroc(-c, harm)}
    if transported is not None:
        joined = [(t, transported[t["question"]]) for t in ctrl if t["question"] in transported]
        if joined:
            tr = np.asarray([x for _, x in joined]); h = np.asarray([t["outcomes"][COND] == "changed_wrong" for t, _ in joined])
            report["controls_vs_transport"] = {"joined": int(len(joined)), "transported": int(tr.sum()),
                                               "harm_if_transported": float(h[tr].mean()) if tr.sum() else None,
                                               "harm_if_not": float(h[~tr].mean()) if (~tr).sum() else None}
    return report


def checks_from(report: dict) -> dict:
    t = report.get("touched"); ch = report["controls_harm"]; cv = report.get("controls_vs_transport")
    return {"p20a_low_half_repairs_ge_1_5x_high_half": None if not t or t["ratio_low_over_high_half"] is None else t["ratio_low_over_high_half"] >= 1.5,
            "p20b_bottom_quartile_harm_ge_2x_top": None if ch["ratio_bottom_over_top_quartile"] is None else ch["ratio_bottom_over_top_quartile"] >= 2.0,
            "p20c_harm_higher_if_carrier_transported": None if not cv or cv["joined"] < 20 or cv["harm_if_transported"] is None or cv["harm_if_not"] is None
            else cv["harm_if_transported"] > cv["harm_if_not"]}


def markdown(report: dict, name: str) -> str:
    lines = [f"### {name}: {report['n_touched']} touched failures ({report['n_touched_latent_only']} latent-only), {report['n_controls']} controls"]
    for key in ("touched", "touched_latent_only"):
        if key in report:
            e = report[key]
            lines.append(f"- {key}: repair {100 * e['repair_rate']:.1f}%; by confidence quartile (low→high) " + " / ".join(f"{100 * b['rate']:.0f}% (n={b['n']})" for b in e["by_quartile"])
                         + f"; low/high half ratio {e['ratio_low_over_high_half'] if e['ratio_low_over_high_half'] is None else round(e['ratio_low_over_high_half'], 2)}; AUROC(low confidence → repaired) {e['auroc_confidence_for_repaired']:.2f}")
    ch = report["controls_harm"]
    lines.append(f"- controls: harm {100 * ch['harm_rate']:.1f}%; by confidence quartile (low→high) " + " / ".join(f"{100 * b['rate']:.0f}%" for b in ch["by_quartile"])
                 + f"; bottom/top ratio {ch['ratio_bottom_over_top_quartile'] if ch['ratio_bottom_over_top_quartile'] is None else round(ch['ratio_bottom_over_top_quartile'], 2)}; AUROC(low confidence → harm) {ch['auroc_low_confidence_for_harm']:.2f}")
    cv = report.get("controls_vs_transport")
    if cv:
        lines.append(f"- controls joined with twins: {cv['joined']} ({cv['transported']} transported); harm if carrier transported {cv['harm_if_transported']} vs not {cv['harm_if_not']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True); parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--questions-json", type=Path, required=True); parser.add_argument("--twins-dir", type=Path, help="twins_long directory with pairs.jsonl and predictions.pt")
    parser.add_argument("--carriers", default="", help="comma list of carrier slots for the transport join")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--name", default="")
    args = parser.parse_args()
    import torch
    readout = torch.load(args.readout, map_location="cpu", weights_only=False)
    targets = [json.loads(l) for l in args.targets.read_text().splitlines() if l.strip()]
    transported = None
    if args.twins_dir and (args.twins_dir / "pairs.jsonl").exists():
        pairs = [json.loads(l) for l in (args.twins_dir / "pairs.jsonl").read_text().splitlines() if l.strip()]
        preds = torch.load(args.twins_dir / "predictions.pt", map_location="cpu", weights_only=False)
        transported = transported_controls(pairs, preds, [int(c) for c in args.carriers.split(",") if c.strip()])
    report = analyze(targets, json.loads(args.questions_json.read_text()), readout, transported=transported); report["checks"] = checks_from(report)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown(report, args.name)); print(json.dumps(report["checks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
