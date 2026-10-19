"""Can the thoughts' own consistency flag a confident latent slip? (ledger §139; no new runs)

Joins the readout states (six thoughts per GSM8K test question, with the answer's first-token
probability) with the failure taxonomy, on the ≥3-step pool, and scores five detectors out of fold:
confidence alone; an unsupervised transition residual (ridge map from each slot to the next, fitted
on correct rows only); the mean consecutive-slot cosine; a supervised linear detector on all six
states; and confidence plus states.  AUROC for latent_only vs correct and plan_wrong vs correct on
the whole pool and on the confident half, plus capture rates at fixed false-positive rates.
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

from scripts.analyze_latent_explicit_cascade import load_questions
from scripts.analyze_plan_error_detectability import MODELS
from src.mech.latent_hijack import auroc

CLASSES = ("latent_only", "plan_wrong")
FOLDS = 5
SEED = 20_261_019
FPRS = (0.10, 0.20)


def dual_ridge_fit(x: torch.Tensor, y: torch.Tensor, lam_scale: float = 0.1):
    """Dual-form ridge: returns (x_train, alpha, means) so that predict(z) = (z - mx) @ x^T @ alpha + my."""
    mx, my = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc = (x - mx).double(); yc = (y - my).double()
    k = xc @ xc.T
    lam = lam_scale * float(k.diagonal().mean()) + 1e-6
    alpha = torch.linalg.solve(k + lam * torch.eye(k.shape[0], dtype=torch.double), yc)
    return xc, alpha, mx.double(), my.double()


def dual_ridge_predict(model, z: torch.Tensor) -> torch.Tensor:
    xc, alpha, mx, my = model
    return (z.double() - mx) @ xc.T @ alpha + my


def transition_residual(states: torch.Tensor, correct: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Summed relative residual of predicting slot s+1 from slot s, each map fitted on the correct
    rows outside the row's fold.  No failure label is used."""
    n, S, _ = states.shape
    score = np.zeros(n)
    for f in range(FOLDS):
        train = (folds != f) & correct; test = folds == f
        if train.sum() < 20 or test.sum() == 0:
            continue
        for s in range(S - 1):
            model = dual_ridge_fit(states[train, s].float(), states[train, s + 1].float())
            pred = dual_ridge_predict(model, states[test, s].float())
            y = states[test, s + 1].double()
            score[test] += ((y - pred).norm(dim=1) / y.norm(dim=1).clamp_min(1e-6)).numpy()
    return score


def consecutive_cosine(states: torch.Tensor) -> np.ndarray:
    x = torch.nn.functional.normalize(states.float(), dim=2)
    return (x[:, :-1] * x[:, 1:]).sum(2).mean(1).numpy()


def supervised_oof(features: torch.Tensor, labels: np.ndarray, folds: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OOF dual-ridge classifier score (positive = label 1) over the rows in ``mask`` (both classes),
    class-balanced targets, features standardised with train statistics.  Rows outside ``mask`` get NaN."""
    score = np.full(len(labels), np.nan)
    for f in range(FOLDS):
        train = mask & (folds != f); test = mask & (folds == f)
        if train.sum() < 20 or test.sum() == 0:
            continue
        x = features[train].float(); mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
        xs = (x - mean) / std
        y = torch.tensor(labels[train], dtype=torch.double)
        pos, neg = (y == 1).sum(), (y == 0).sum()
        t = torch.where(y == 1, 0.5 / pos.clamp_min(1), -0.5 / neg.clamp_min(1)) * (pos + neg)   # class-balanced ±
        model = dual_ridge_fit(xs, t.unsqueeze(1), lam_scale=1.0)
        score[test] = dual_ridge_predict(model, ((features[test].float() - mean) / std)).squeeze(1).numpy()
    return score


def auc(scores: np.ndarray, pos: np.ndarray, neg: np.ndarray, *, resamples: int = 1000, seed: int = SEED) -> dict:
    sp, sn = scores[pos], scores[neg]
    sp, sn = sp[~np.isnan(sp)], sn[~np.isnan(sn)]
    if len(sp) < 5 or len(sn) < 5:
        return {"auroc": None, "n_pos": int(len(sp)), "n_neg": int(len(sn))}
    point = auroc(sp.tolist() + sn.tolist(), [1] * len(sp) + [0] * len(sn))
    rng = np.random.default_rng(seed); stats = []
    for _ in range(resamples):
        a = auroc(sp[rng.integers(0, len(sp), len(sp))].tolist() + sn[rng.integers(0, len(sn), len(sn))].tolist(), [1] * len(sp) + [0] * len(sn))
        if a is not None:
            stats.append(a)
    return {"auroc": point, "ci95": [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))], "n_pos": int(len(sp)), "n_neg": int(len(sn))}


def capture(scores: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> dict:
    """Share of positives above the threshold that lets through the given share of negatives."""
    out = {}
    sn = scores[neg]; sn = sn[~np.isnan(sn)]; sp = scores[pos]; sp = sp[~np.isnan(sp)]
    for fpr in FPRS:
        if len(sn) < 5 or len(sp) < 1:
            out[str(fpr)] = None; continue
        th = float(np.quantile(sn, 1 - fpr))
        out[str(fpr)] = float((sp > th).mean())
    return out


def analyze(questions: list[str], readout: dict, targets: list[dict], *, resamples: int = 1000) -> dict:
    states = readout["states"].float()
    prob = np.asarray(readout["first_token_prob"], dtype=float); steps = np.asarray(readout["steps"], dtype=int)
    correct = np.asarray(readout["correct"], dtype=bool)
    pool = (steps >= 3) & ~np.isnan(prob)
    index = {q: i for i, q in enumerate(questions)}
    label = np.array(["other"] * len(prob), dtype=object)
    label[pool & correct] = "correct"
    for t in targets:
        if t["control"]:
            continue
        i = index.get(t["question"])
        if i is not None and pool[i] and not correct[i] and t.get("failure_class") in CLASSES:
            label[i] = t["failure_class"]
    median = float(np.median(prob[pool]))
    confident = pool & (prob >= median)
    top = pool & (prob >= float(np.quantile(prob[pool], 0.75)))       # the top quartile, where the gate never routes
    rng = np.random.default_rng(SEED); folds = rng.integers(0, FOLDS, len(prob))
    is_correct = label == "correct"
    scores = {"confidence": -prob, "transition_residual": transition_residual(states, is_correct, folds), "neg_consecutive_cosine": -consecutive_cosine(states)}
    flat = states.flatten(1)
    conf_feat = torch.tensor(np.nan_to_num(prob), dtype=torch.float32).unsqueeze(1)
    report = {"pool": {"n": int(pool.sum()), "correct": int(is_correct.sum()), "median_confidence": median, "confident_half": int(confident.sum())}, "classes": {}}
    for c in CLASSES:
        is_c = label == c
        mask = is_c | is_correct
        y = is_c.astype(int)
        sup = supervised_oof(flat, y, folds, mask)
        # combinations: a rank average of two scores (no fitting), so neither drowns the other
        def rank_avg(a, b):
            ra = np.full(len(a), np.nan); rb = np.full(len(b), np.nan)
            ok = ~np.isnan(a) & ~np.isnan(b)
            ra[ok] = np.argsort(np.argsort(a[ok])); rb[ok] = np.argsort(np.argsort(b[ok]))
            return (ra + rb) / 2
        both = rank_avg(scores["confidence"], sup)
        conf_res = rank_avg(scores["confidence"], scores["transition_residual"])
        per = {}
        for name, sc in list(scores.items()) + [("supervised_states", sup), ("confidence_plus_states", both), ("confidence_plus_residual", conf_res)]:
            per[name] = {"all": auc(sc, is_c & pool, is_correct, resamples=resamples), "confident_half": auc(sc, is_c & confident, is_correct & confident, resamples=resamples),
                         "top_quartile": auc(sc, is_c & top, is_correct & top, resamples=resamples),
                         "capture_confident_at_fpr": capture(sc, is_c & confident, is_correct & confident)}
        report["classes"][c] = {"n_all": int((is_c & pool).sum()), "n_confident": int((is_c & confident).sum()), "n_top_quartile": int((is_c & top).sum()), "detectors": per}
    return report


def markdown(reports: dict) -> str:
    lines = ["# Confident latent slips: can the thoughts see them? (ledger §139)", "",
             "AUROC vs correct rows; 'conf' = confident half (first-token probability ≥ pool median). Capture = share of confident positives caught at 10% / 20% false positives.", "",
             "| model | class | n all / conf / top | confidence (all / conf / top) | transition residual (all / conf / top) | supervised states (all / conf / top) | conf+residual (conf / top) | conf+states (conf / top) | capture @10/20% (residual, conf) |", "|---|---|---|---|---|---|---|---|---|"]
    f = lambda d: "—" if d.get("auroc") is None else f"{d['auroc']:.2f}"
    for m, r in reports.items():
        for c, b in r["classes"].items():
            d = b["detectors"]; cap = d["transition_residual"]["capture_confident_at_fpr"]
            t = lambda k: f"{f(d[k]['all'])} / {f(d[k]['confident_half'])} / {f(d[k]['top_quartile'])}"
            lines.append(f"| {m} | {c} | {b['n_all']} / {b['n_confident']} / {b['n_top_quartile']} | {t('confidence')} | {t('transition_residual')} | {t('supervised_states')} | "
                         f"{f(d['confidence_plus_residual']['confident_half'])} / {f(d['confidence_plus_residual']['top_quartile'])} | {f(d['confidence_plus_states']['confident_half'])} / {f(d['confidence_plus_states']['top_quartile'])} | "
                         f"{'—' if cap.get('0.1') is None else format(cap['0.1'], '.2f')} / {'—' if cap.get('0.2') is None else format(cap['0.2'], '.2f')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--readout-root", type=Path, default=ROOT / "outputs" / "kaggle_runs" / "readout")
    parser.add_argument("--targets-root", type=Path, default=ROOT / "docs" / "latent_monitor_audit" / "latent_repair_ladder")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "latent_monitor_audit" / "analyses" / "latent_slip_detectability.json")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    reports = {}
    for m in args.models:
        readout = torch.load(args.readout_root / m / f"readout_{m}" / "readout_states.pt", map_location="cpu", weights_only=False)
        questions = load_questions(ROOT / MODELS[m])
        targets = [json.loads(l) for l in (args.targets_root / f"{m}_tool.targets.jsonl").read_text().splitlines() if l.strip()]
        reports[m] = analyze(questions, readout, targets, resamples=args.resamples)
        r = reports[m]["classes"]["latent_only"]["detectors"]
        print(m, "latent_only confident-half AUROC:", {k: (round(v["confident_half"]["auroc"], 3) if v["confident_half"]["auroc"] is not None else None) for k, v in r.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(reports))
    print(markdown(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
