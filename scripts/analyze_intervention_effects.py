"""Direct paired difference-in-differences for two latent methods."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.causal_compare import compare_intervention_effects, render_did_markdown
from src.eval.compare_runs import load_eval_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare intervention effects between two methods."
    )
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--left-baseline", type=Path, required=True)
    parser.add_argument("--left-intervention", type=Path, required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--right-baseline", type=Path, required=True)
    parser.add_argument("--right-intervention", type=Path, required=True)
    parser.add_argument("--intervention-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        report = compare_intervention_effects(
            left_name=args.left_name,
            left_baseline=load_eval_run(args.left_baseline),
            left_intervention=load_eval_run(args.left_intervention),
            right_name=args.right_name,
            right_baseline=load_eval_run(args.right_baseline),
            right_intervention=load_eval_run(args.right_intervention),
            intervention_name=args.intervention_name,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_did_markdown(report), encoding="utf-8")
    print(f"[analysis] wrote {args.output}")
    print(f"[analysis] wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
