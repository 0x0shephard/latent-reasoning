"""Paired bootstrap intervals for a completed directed-counterfactual run (ledger §119).

Reads ``predictions.pt`` (first-token predictions per condition, own and target gold,
located slots) written by ``run_codi_directed_counterfactual.py``.  Rows come in twin
pairs (2i, 2i+1); the bootstrap resamples *pairs*, so the two directions of one twin
move together and the interval is over pairs, not rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONTRASTS = (
    ("state_3", "state_1"), ("kv_4", "kv_2"), ("kv_all", "v89_all"), ("kv_all", "state_all"),
    ("state_3", "v89_4"), ("state_1", "v89_2"), ("kv_0", "state_0"), ("v89_even", "v89_odd"),
)
SEED = 20_261_007


def target_hits(preds: dict, name: str) -> torch.Tensor:
    return (preds[name] == preds["target_gold"]).to(torch.float64)


def pair_means(hits: torch.Tensor) -> torch.Tensor:
    """Per-pair mean of a per-row indicator (rows 2i and 2i+1 form pair i)."""
    return hits.view(-1, 2).mean(1)


def bootstrap_mean(values: torch.Tensor, *, samples: int, seed: int) -> dict:
    """Mean of per-pair values with a percentile interval over resampled pairs."""
    g = torch.Generator().manual_seed(seed)
    n = values.shape[0]
    idx = torch.randint(0, n, (samples, n), generator=g)
    draws = values[idx].mean(1)
    lo, hi = torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=draws.dtype)).tolist()
    return {"mean": float(values.mean()), "ci95": [lo, hi], "pairs": int(n)}


def analyze(run_dir: Path, *, samples: int = 2000, seed: int = SEED, contrasts=DEFAULT_CONTRASTS, min_rows: int = 20) -> dict:
    preds = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)
    n_rows = int(preds["target_gold"].shape[0])
    if n_rows % 2:
        raise ValueError("rows must come in twin pairs")
    conditions = [k for k in preds if k not in ("own_gold", "target_gold", "located")]
    report = {"run": str(run_dir), "rows": n_rows, "pairs": n_rows // 2, "samples": samples, "seed": seed,
              "target_rate": {}, "contrast": {}, "located": {}}
    for c in conditions:
        report["target_rate"][c] = bootstrap_mean(pair_means(target_hits(preds, c)), samples=samples, seed=seed)
    for a, b in contrasts:
        if a in preds and b in preds:
            diff = pair_means(target_hits(preds, a) - target_hits(preds, b))
            report["contrast"][f"{a}_minus_{b}"] = bootstrap_mean(diff, samples=samples, seed=seed)
    # located-site contrasts: rows uniquely located at slot s (state_s vs the other feeding slot;
    # kv_{s+1} vs the other store), resampling the located pairs
    located = preds["located"]
    for s, other in ((1, 3), (3, 1)):
        rows = torch.tensor([i for i, l in enumerate(located) if l == [s]])
        if rows.numel() < min_rows:
            report["located"][str(s)] = {"rows": int(rows.numel()), "note": f"fewer than {min_rows} uniquely located rows"}
            continue
        entry = {"rows": int(rows.numel())}
        for label, a, b in (("state_site_minus_other", f"state_{s}", f"state_{other}"),
                            ("kv_store_site_minus_other", f"kv_{s + 1}", f"kv_{other + 1}")):
            diff = (target_hits(preds, a) - target_hits(preds, b))[rows]
            entry[label] = bootstrap_mean(diff, samples=samples, seed=seed)   # rows here, pairs are not both located
            entry[label]["unit"] = "rows"
        entry["state_at_site"] = bootstrap_mean(target_hits(preds, f"state_{s}")[rows], samples=samples, seed=seed)
        report["located"][str(s)] = entry
    return report


def markdown(report: dict, keys=("state_1", "state_3", "kv_2", "kv_4", "kv_0", "kv_all", "v89_all", "state_all")) -> str:
    lines = [f"run `{Path(report['run']).name}`: {report['pairs']} pairs, {report['samples']} resamples", "",
             "| condition | target | 95% CI |", "|---|---|---|"]
    for k in keys:
        if k in report["target_rate"]:
            r = report["target_rate"][k]
            lines.append(f"| {k} | {r['mean']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    lines += ["", "| contrast | difference | 95% CI |", "|---|---|---|"]
    for k, r in report["contrast"].items():
        lines.append(f"| {k} | {r['mean']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
    for s, e in report["located"].items():
        if "note" in e:
            lines.append(f"\nuniquely located at slot {s}: {e['rows']} rows ({e['note']})")
        else:
            a, b = e["state_site_minus_other"], e["kv_store_site_minus_other"]
            lines.append(f"\nuniquely located at slot {s} ({e['rows']} rows): state site−other {a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}];"
                         f" K/V store site−other {b['mean']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [analyze(d, samples=args.samples, seed=args.seed) for d in args.run_dirs]
    for r in reports:
        print(markdown(r)); print()
    if args.output:
        args.output.write_text(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
