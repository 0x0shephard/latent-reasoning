"""Stage models + datasets locally so Kaggle can load them offline (Phase 1).

Intended flow: run this on a networked machine, upload the resulting `hf_cache/` as a
Kaggle Dataset, then point the training notebook at it with HF_HUB_OFFLINE=1.

Not yet implemented — placeholder so the pipeline shape is visible. Phase 1 fills in:
  - GPT-2 backbone
  - GSM8k-Aug (equation-only) + GSM8k-Aug-NL (natural language) training sets
  - GSM8k (in-domain), SVAMP / MultiArith / GSM-Hard (OOD) eval sets
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "dataset_prep is a Phase 1 deliverable. Phase 0 uses a synthetic task only."
    )


if __name__ == "__main__":
    main()
