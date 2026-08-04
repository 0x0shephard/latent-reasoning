"""Build the final parameter-aware endpoint decision report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.official_codi_endpoint_parameter_aware_analysis import (
    build_parameter_aware_final_report,
    render_parameter_aware_markdown,
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze parameter-aware official-CODI endpoint results."
    )
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--utility", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    basis = torch.load(args.basis, map_location="cpu", weights_only=False)
    utility = None
    if args.utility is not None:
        path = args.utility / "summary.json" if args.utility.is_dir() else args.utility
        if not path.is_file():
            raise FileNotFoundError(f"utility summary is missing: {path}")
        utility = json.loads(path.read_text(encoding="utf-8"))
    report = build_parameter_aware_final_report(basis, utility)
    _atomic_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = args.output.with_suffix(".md")
    _atomic_text(markdown, render_parameter_aware_markdown(report))
    print(f"[analysis] status={report['status']}")
    print(f"[analysis] training_authorized={report['training_authorized']}")
    print(f"[analysis] wrote {args.output}")
    print(f"[analysis] wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
