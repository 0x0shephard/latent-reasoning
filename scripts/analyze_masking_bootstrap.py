"""Row-bootstrap intervals for the masking audit's accuracy drops (ledger §124).

Reads ``predictions.pt`` written by ``scripts/run_codi_masking_audit.py`` and, for every
condition and chain-length part (all / short / long), resamples rows with replacement
and reports the mean accuracy drop with a percentile 95% interval.
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

SEED = 20_261_013
SAMPLES = 2_000


def bootstrap_drop(base_correct: torch.Tensor, cond_correct: torch.Tensor, *, samples: int = SAMPLES, seed: int = SEED) -> dict:
    diffs = (base_correct.double() - cond_correct.double())
    n = int(diffs.numel())
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(n, (samples, n), generator=generator)
    means = diffs[index].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=means.dtype)).tolist()
    return {"mean": float(diffs.mean()), "ci95": [float(lo), float(hi)], "n": n}


def analyze(run_dir: Path, *, samples: int = SAMPLES, seed: int = SEED) -> dict:
    payload = torch.load(Path(run_dir) / "predictions.pt", map_location="cpu", weights_only=False)
    base, gold, is_long = payload["base_pred"], payload["gold"], payload["is_long"].bool()
    parts = {"all": torch.ones_like(is_long), "short": ~is_long, "long": is_long}
    report = {"run": str(run_dir), "rows": int(gold.numel()), "samples": samples, "seed": seed, "parts": {}}
    base_correct = base == gold
    for part, mask in parts.items():
        report["parts"][part] = {name: bootstrap_drop(base_correct[mask], (pred == gold)[mask], samples=samples, seed=seed)
                                 for name, pred in payload["conditions"].items()}
    return report


def markdown(report: dict, conditions=("mean_1", "mean_3", "mask_1", "mask_3", "mask_0", "mask_5", "mask_all")) -> str:
    lines = [f"run `{Path(report['run']).name}`: {report['rows']} rows, {report['samples']} resamples", "",
             "| condition | drop all [95% CI] | drop short | drop long |", "|---|---|---|---|"]
    f = lambda e: "—" if e is None or e["mean"] is None else f"{100 * e['mean']:.1f} [{100 * e['ci95'][0]:.1f}, {100 * e['ci95'][1]:.1f}]"
    for name in conditions:
        if name in report["parts"]["all"]:
            lines.append(f"| {name} | {f(report['parts']['all'][name])} | {f(report['parts']['short'][name])} | {f(report['parts']['long'][name])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [analyze(d, samples=args.samples, seed=args.seed) for d in args.run_dirs]
    for r in reports:
        print(markdown(r)); print()
    if args.output:
        args.output.write_text(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
