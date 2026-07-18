"""Aggregate single-position latent interventions against one shared baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.causal_compare import (
    analyze_position_sweep,
    render_position_sweep_markdown,
)
from src.eval.compare_runs import load_eval_run


def _position_spec(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected POSITION=EVAL_DIRECTORY")
    raw_position, raw_path = value.split("=", 1)
    try:
        position = int(raw_position)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("position must be an integer") from exc
    if position < 0 or not raw_path.strip():
        raise argparse.ArgumentTypeError("position must be non-negative and path non-empty")
    return position, Path(raw_path.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a latent-position sweep.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--position",
        action="append",
        required=True,
        type=_position_spec,
        metavar="POSITION=EVAL_DIR",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    positions = [position for position, _ in args.position]
    if len(set(positions)) != len(positions):
        parser.error("position numbers must be unique")
    try:
        report = analyze_position_sweep(
            load_eval_run(args.baseline),
            {position: load_eval_run(path) for position, path in args.position},
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
    markdown.write_text(render_position_sweep_markdown(report), encoding="utf-8")
    print(f"[analysis] wrote {args.output}")
    print(f"[analysis] wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
