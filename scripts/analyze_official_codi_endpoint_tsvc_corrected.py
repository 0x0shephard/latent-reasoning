"""Combine corrected primary and layer-11 endpoint utility scopes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.official_codi_endpoint_tsvc_corrected_analysis import (
    combine_corrected_endpoint_reports,
    render_corrected_endpoint_markdown,
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _load_summary(path: Path) -> dict:
    resolved = path / "summary.json" if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(f"corrected endpoint summary is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("gate", {}).get("supported") not in {True, False}:
        raise RuntimeError(f"corrected endpoint summary is incomplete: {resolved}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine corrected official-CODI endpoint TSV-C reports."
    )
    parser.add_argument("--all-states", required=True, type=Path)
    parser.add_argument("--layer11", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = combine_corrected_endpoint_reports(
        _load_summary(args.all_states),
        _load_summary(args.layer11),
    )
    _atomic_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = args.output.with_suffix(".md")
    _atomic_text(markdown, render_corrected_endpoint_markdown(report))
    print(f"[analysis] status={report['status']}")
    print(f"[analysis] training_authorized={report['training_authorized']}")
    print(f"[analysis] wrote {args.output}")
    print(f"[analysis] wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
