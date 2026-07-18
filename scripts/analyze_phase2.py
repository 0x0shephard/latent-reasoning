"""Create paired Phase-2 comparisons from saved evaluation directories.

Example:
    python scripts/analyze_phase2.py \
      --run sft=outputs/sft_cot/eval/step_00024102 \
      --run codi=outputs/codi/eval/step_00096405 \
      --run kava=outputs/kava/eval/step_00096405 \
      --output reports/phase2_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.compare_runs import compare_runs, load_eval_run, render_markdown


def _run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=EVAL_DIRECTORY")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=EVAL_DIRECTORY")
    return name.strip(), Path(raw_path.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare aligned per-example evaluation artifacts."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_run_spec,
        metavar="NAME=EVAL_DIR",
        help="Repeat for each method (at least two).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-markdown", action="store_true", help="Do not write a sibling .md report."
    )
    args = parser.parse_args()

    names = [name for name, _ in args.run]
    if len(set(names)) != len(names):
        parser.error("--run names must be unique")
    try:
        runs = {name: load_eval_run(path) for name, path in args.run}
        report = compare_runs(
            runs, bootstrap_samples=args.bootstrap_samples, seed=args.seed
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[analysis] wrote {args.output}")
    if not args.no_markdown:
        markdown_path = args.output.with_suffix(".md")
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"[analysis] wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
