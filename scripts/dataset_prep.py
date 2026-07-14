"""Stage the backbone + datasets into a local HF cache for offline Kaggle use.

Run this on a networked machine, then upload the produced cache directory as a Kaggle
Dataset and point the notebook at it via HF_HOME with HF_HUB_OFFLINE=1. This lets the
training notebook run with Internet turned OFF (faster, reproducible, no mid-run download
failures).

    python scripts/dataset_prep.py --out hf_cache --config configs/sft_cot.yaml

Then: create a Kaggle Dataset from ./hf_cache, attach it, and set in the notebook
    os.environ["HF_HOME"] = "/kaggle/input/<your-dataset>/hf_cache"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["CODIKAVA_DATA_ROOT"] = "/kaggle/input/<your-dataset>/hf_cache/prepared"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _save_prepared(dataset, target: Path) -> None:
    """Save once; completed targets are reusable across interrupted staging runs."""
    if (target / "state.json").is_file():
        print(f"[stage] prepared target already complete, reusing: {target}")
        return
    if target.exists():
        raise FileExistsError(f"incomplete prepared target exists: {target}")
    dataset.save_to_disk(str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage backbone + datasets for offline use.")
    parser.add_argument("--out", default="hf_cache", help="Cache directory to populate.")
    parser.add_argument("--config", default="configs/sft_cot.yaml")
    parser.add_argument("--backbone", default=None, help="Override config task.backbone.")
    parser.add_argument("--backbone-revision", default=None)
    parser.add_argument("--data-config", default=None)
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    os.environ["HF_HOME"] = out
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.pop("CODIKAVA_DATA_ROOT", None)  # staging must read the pinned remote sources
    print(f"[stage] HF_HOME -> {out}")

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.data.datasets import load_all_eval_sets, load_train_set
    from src.utils.config import load_config

    run_cfg = load_config(args.config)
    backbone = args.backbone or run_cfg.task.backbone
    revision = args.backbone_revision or run_cfg.task.get("backbone_revision")
    data_config_path = args.data_config or run_cfg.data_config
    pretrained_kwargs = {"revision": revision} if revision else {}

    # Backbone weights + tokenizer.
    print(f"[stage] downloading backbone: {backbone} revision={revision or 'main'}")
    AutoTokenizer.from_pretrained(backbone, **pretrained_kwargs)
    AutoModelForCausalLM.from_pretrained(backbone, **pretrained_kwargs)

    data_cfg = load_config(data_config_path)
    prepared = Path(out) / "prepared"

    # Training sets (both trace styles).
    for style, spec in data_cfg["train"].items():
        if style == "fields":
            continue
        print(f"[stage] downloading + preparing train/{style}: {spec['hf_id']}")
        dataset = load_train_set(data_cfg, style)
        target = prepared / "train" / style
        _save_prepared(dataset, target)

    # Eval sets are staged in their already-normalized shape.
    for name, rows in load_all_eval_sets(data_cfg).items():
        print(f"[stage] preparing eval/{name}: {len(rows)} examples")
        target = prepared / "eval" / name
        dataset = Dataset.from_list(
            [{"question": row["question"], "gold": str(row["gold"])} for row in rows]
        )
        _save_prepared(dataset, target)

    print(
        f"[stage] done. Upload '{out}' as a Kaggle Dataset and set HF_HOME, "
        "HF_HUB_OFFLINE=1, and CODIKAVA_DATA_ROOT=<HF_HOME>/prepared."
    )


if __name__ == "__main__":
    main()
