"""Compare Latent Monitor Audit cards (ledger §123) across pools and environments.

``python scripts/compare_latent_monitor_audit_cards.py card1.json card2.json ... --output comparison.md``
writes the Markdown table and, beside it, ``comparison.json``; flags carrier sets or
regimes that differ across pools of one model, long-transport intervals that do not
overlap, and predictions fingerprints that differ across environments of one (model, pool).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mech.monitor_audit import compare_cards, comparison_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cards = [json.loads(p.read_text()) for p in args.cards]
    comparison = compare_cards(cards)
    text = comparison_markdown(comparison)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        args.output.with_suffix(".json").write_text(json.dumps(comparison, indent=2))
    print(text)
    return 0 if comparison["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
