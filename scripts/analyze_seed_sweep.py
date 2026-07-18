"""Create a matched multi-seed CODI-vs-KaVa report from evaluation directories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.compare_runs import load_eval_run
from src.eval.seed_compare import compare_seeded_runs, render_seed_markdown


def _run_spec(value: str) -> tuple[str, int, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected METHOD:SEED=EVAL_DIRECTORY")
    identity, raw_path = value.split("=", 1)
    method, raw_seed = identity.rsplit(":", 1)
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SEED must be an integer") from exc
    if not method.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("method and evaluation directory cannot be empty")
    return method.strip(), seed, Path(raw_path.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate matched evaluation seeds")
    parser.add_argument("--run", action="append", required=True, type=_run_spec)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for method, seed, path in args.run:
        if seed in runs.setdefault(method, {}):
            parser.error(f"duplicate run for {method} seed {seed}")
        runs[method][seed] = load_eval_run(path)
    try:
        report = compare_seeded_runs(runs)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_seed_markdown(report), encoding="utf-8")
    print(f"[analysis] wrote {args.output}")
    print(f"[analysis] wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
