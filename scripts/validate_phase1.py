"""Validate Phase-1 artifacts before spending a Kaggle GPU session.

Checks real Hugging Face schemas, tokenizer/model resolution, answer parseability, CoT
truncation, and exact question leakage between training and evaluation sets.

    python scripts/validate_phase1.py --config configs/sft_cot.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from src.data.answer_extract import normalize_gold
from src.data.datasets import load_all_eval_sets, load_train_set
from src.data.prompts import PromptStyle
from src.train.sft import encode_sft_example, resolve_total_steps
from src.utils.config import load_config


def _question_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _sample_indices(size: int, sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0 or sample_size >= size:
        return list(range(size))
    return sorted(random.Random(seed).sample(range(size), sample_size))


def validate(cfg, sample_size: int, max_truncation_rate: float, leakage_scan: bool) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    data_cfg = load_config(cfg.data_config)
    style = PromptStyle.from_config(data_cfg["prompt"])
    task_cfg = cfg.task
    revision = task_cfg.get("backbone_revision")
    pretrained_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(task_cfg.backbone, **pretrained_kwargs)
    model_config = AutoConfig.from_pretrained(task_cfg.backbone, **pretrained_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = load_train_set(data_cfg, task_cfg.get("trace_style", "eq_only"))
    total_steps = resolve_total_steps(cfg, len(train))
    indices = _sample_indices(len(train), sample_size, cfg.seed)
    truncated = 0
    unparseable = 0
    length_failures = []
    for index in indices:
        row = train[index]
        if normalize_gold(row["answer"], "aug") is None:
            unparseable += 1
        try:
            encoded = encode_sft_example(
                tokenizer,
                row,
                task_cfg.method,
                style,
                task_cfg.max_length,
            )
        except ValueError as exc:
            length_failures.append({"index": index, "error": str(exc)})
            continue
        truncated += int(encoded.truncated_reasoning)

    truncation_rate = truncated / max(1, len(indices))
    report = {
        "status": "ok",
        "backbone": task_cfg.backbone,
        "resolved_backbone_revision": getattr(model_config, "_commit_hash", None),
        "tokenizer_class": type(tokenizer).__name__,
        "train_examples": len(train),
        "train_dataset_fingerprint": getattr(train, "_fingerprint", None),
        "sampled_examples": len(indices),
        "unparseable_sample_answers": unparseable,
        "reasoning_truncated": truncated,
        "reasoning_truncation_rate": truncation_rate,
        "length_failures": length_failures[:20],
        "total_steps": total_steps,
        "effective_epochs": total_steps * cfg.train.batch_size / len(train),
    }

    failures = []
    if unparseable:
        failures.append(f"{unparseable} sampled training answers were unparseable")
    if length_failures:
        failures.append(f"{len(length_failures)} examples could not retain prompt+answer")
    if truncation_rate > max_truncation_rate:
        failures.append(
            f"reasoning truncation rate {truncation_rate:.2%} exceeds "
            f"{max_truncation_rate:.2%}"
        )

    if leakage_scan:
        eval_sets = load_all_eval_sets(data_cfg)
        train_questions = {_question_key(str(row["question"])) for row in train}
        overlaps = {}
        for name, examples in eval_sets.items():
            overlap = sorted(
                example["question"]
                for example in examples
                if _question_key(example["question"]) in train_questions
            )
            overlaps[name] = {"count": len(overlap), "examples": overlap[:20]}
            if overlap:
                failures.append(f"{name} has {len(overlap)} exact train-question overlaps")
        report["eval_examples"] = {name: len(rows) for name, rows in eval_sets.items()}
        report["exact_train_eval_overlap"] = overlaps

    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Phase-1 data/model contract.")
    parser.add_argument("--config", default="configs/sft_cot.yaml")
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--max-truncation-rate", type=float, default=0.05)
    parser.add_argument("--skip-leakage-scan", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], help="Config dot-overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.set)
    if cfg.get("offline", False):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        report = validate(
            cfg,
            sample_size=args.sample_size,
            max_truncation_rate=args.max_truncation_rate,
            leakage_scan=not args.skip_leakage_scan,
        )
    except Exception as exc:  # surface external artifact failures in the saved report
        report = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "phase1_validation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[validate] wrote {path}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
