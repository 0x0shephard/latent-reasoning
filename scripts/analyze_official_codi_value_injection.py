"""Choose the injection strength on select, then gate the frozen test read.

``--mode choose-beta`` reads the select-split sweep (baseline plus gold/random at
every beta) and prints the frozen selection: the beta maximizing gold-minus-random
recovery on baseline-wrong injectable select rows, ties to the smaller beta. The
corruption arm inherits the same beta, keeping every intervention matched.

``--mode report`` reads the four test arms at the chosen beta and applies the
frozen gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from src.eval.official_codi_value_injection_analysis import analyze_value_injection
from src.mech.latent_value_injection import VALUE_INJECTION_CONTRACT
from src.utils.config import load_config


def load_arm(directory: Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("contract") != VALUE_INJECTION_CONTRACT:
        raise RuntimeError(f"{directory} belongs to another contract")
    rows = [
        json.loads(line)
        for line in (directory / "gsm8k.jsonl").read_text().splitlines()
        if line
    ]
    return {
        "summary": summary,
        "indices": [row["index"] for row in rows],
        "numeric_correct": [bool(row["numeric_exact_match_correct"]) for row in rows],
        "injectable": [bool(row["injectable"]) for row in rows],
        "diagnostics": summary.get("intervention_diagnostics"),
    }


def choose_beta(select_root: Path, beta_grid: list[float]) -> dict:
    baseline = load_arm(select_root / "baseline")
    curve = []
    for beta in beta_grid:
        tag = f"{beta:g}".replace(".", "p")
        gold = load_arm(select_root / f"gold_b{tag}")
        random_arm = load_arm(select_root / f"random_b{tag}")
        if gold["indices"] != baseline["indices"] or (
            random_arm["indices"] != baseline["indices"]
        ):
            raise RuntimeError("select arms are not row-aligned")
        wrong = [
            not correct and can_inject
            for correct, can_inject in zip(
                baseline["numeric_correct"], baseline["injectable"]
            )
        ]
        rows = sum(wrong)
        gold_rate = (
            sum(g for g, w in zip(gold["numeric_correct"], wrong) if w) / rows
        )
        random_rate = (
            sum(r for r, w in zip(random_arm["numeric_correct"], wrong) if w) / rows
        )
        curve.append(
            {
                "beta": float(beta),
                "select_wrong_rows": rows,
                "gold_recovery": gold_rate,
                "random_recovery": random_rate,
                "advantage_points": 100 * (gold_rate - random_rate),
            }
        )
    best = max(curve, key=lambda item: (item["advantage_points"], -item["beta"]))
    return {"selected_beta": best["beta"], "beta_selection": curve}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--mode", choices=("choose-beta", "report"), required=True)
    parser.add_argument("--select-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = load_config(args.config).value_injection
    selection = choose_beta(
        args.select_root, [float(b) for b in settings.beta_grid]
    )
    if args.mode == "choose-beta":
        _atomic_json(selection, args.output)
        print(json.dumps(selection, indent=2))
        return 0

    if args.test_root is None:
        raise ValueError("--test-root is required for report mode")
    outcomes = {
        arm: load_arm(args.test_root / arm)
        for arm in ("baseline", "gold", "offset", "random")
    }
    reference = outcomes["baseline"]["indices"]
    for arm, data in outcomes.items():
        if data["indices"] != reference:
            raise RuntimeError(f"test arm {arm} is not row-aligned with baseline")
        if arm != "baseline" and float(
            data["summary"]["beta"]
        ) != float(selection["selected_beta"]):
            raise RuntimeError(
                f"test arm {arm} was run at beta {data['summary']['beta']}, "
                f"but select chose {selection['selected_beta']}"
            )
    summary = {
        "contract": VALUE_INJECTION_CONTRACT,
        "selected_beta": selection["selected_beta"],
        "beta_selection": selection["beta_selection"],
        "splits": {
            "test": len(reference),
            "partition_sha256": outcomes["baseline"]["summary"]["partition_sha256"],
        },
    }
    report = analyze_value_injection(summary, outcomes, settings)
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
