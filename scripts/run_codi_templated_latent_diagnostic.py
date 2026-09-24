"""Ledger §91: can the official CODI checkpoint's latent path do the templated task?

Rebuilds the §89 test split (same generator, seed and split sizes) and scores the
unmodified official checkpoint two ways on the same 2,000 problems: the latent path
(question + BOT + 6 latent thoughts + forced answer decoding, the released path) and
the explicit-CoT path (question alone, greedy). Also reports the latent path on the
256-row selection split.  No training, nothing is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_codi_templated_subspace_distillation import (
    BATCH_SIZE,
    DATA_SEED,
    FIT,
    MAX_NEW_TOKENS_TEACHER,
    SELECT,
    SELECTION_EXAMPLES,
    STEPS,
    TEACHER_CHECK,
    TEST_EXAMPLES,
    VALIDATE,
)
from src.data.templated_arithmetic import generate_problems, generate_teacher_cot, teacher_accuracy
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_templated_latent_diagnostic_v1"
MAX_NEW_TOKENS_LATENT = 64
SPLIT_SIZES = {"teacher_check": TEACHER_CHECK, "fit": FIT, "select": SELECT, "validate": VALIDATE,
               "train": STEPS * BATCH_SIZE, "selection": SELECTION_EXAMPLES, "test": TEST_EXAMPLES}


def section_89_splits() -> dict[str, list]:
    """The §89 splits, byte-identical to the training runner's (same seed, same order)."""
    problems = generate_problems(sum(SPLIT_SIZES.values()), seed=DATA_SEED)
    splits, cursor = {}, 0
    for name, size in SPLIT_SIZES.items():
        splits[name] = problems[cursor : cursor + size]; cursor += size
    return splits


def breakdown(problems, correct) -> dict:
    by = defaultdict(lambda: [0, 0])
    for problem, ok in zip(problems, correct):
        for key in (f"template:{problem.template}", f"steps:{problem.steps}"):
            by[key][0] += int(ok); by[key][1] += 1
    return {k: {"accuracy": c / n, "correct": c, "examples": n} for k, (c, n) in sorted(by.items())}


def run(args):
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token)
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
        dtype=dtype, settings=cfg.model, token=token)
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype).eval()
    latent_iterations = int(cfg.eval.latent_iterations)

    splits = section_89_splits()
    limit = args.limit
    results = {}
    for split_name in ("test", "selection"):
        problems = splits[split_name][:limit] if limit else splits[split_name]
        questions = [p.question for p in problems]
        golds = [p.answer.split(" ")[-1] for p in problems]
        with torch.inference_mode():
            latent = generate_official_codi(model, tokenizer, questions, latent_iterations=latent_iterations,
                                            max_new_tokens=MAX_NEW_TOKENS_LATENT, batch_size=args.batch_size,
                                            device=device)
        latent_correct = [bool(official_answers_match(t, g)) for t, g in zip(latent, golds)]
        entry = {"examples": len(problems),
                 "latent": {"accuracy": sum(latent_correct) / len(problems), "correct": sum(latent_correct),
                            "breakdown": breakdown(problems, latent_correct)}}
        if split_name == "test":
            cot = generate_teacher_cot(model, tokenizer, questions, max_new_tokens=MAX_NEW_TOKENS_TEACHER,
                                       batch_size=args.batch_size, device=device)
            cot_accuracy, cot_correct = teacher_accuracy(cot, problems)
            both = sum(a and b for a, b in zip(latent_correct, cot_correct))
            entry["explicit_cot"] = {"accuracy": cot_accuracy, "correct": sum(cot_correct),
                                     "breakdown": breakdown(problems, cot_correct)}
            entry["agreement"] = {"both_correct": both, "latent_only": sum(latent_correct) - both,
                                  "cot_only": sum(cot_correct) - both}
            entry["samples"] = [{"question": p.question, "gold": g, "latent": l, "latent_correct": lc,
                                 "explicit_cot": c, "cot_correct": cc}
                                for p, g, l, lc, c, cc in list(zip(problems, golds, latent, latent_correct, cot, cot_correct))[:12]]
            with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                import json
                for p, g, l, lc, c, cc in zip(problems, golds, latent, latent_correct, cot, cot_correct):
                    handle.write(json.dumps({"question": p.question, "template": p.template, "steps": p.steps,
                                             "gold": g, "latent": l, "latent_correct": lc,
                                             "explicit_cot": c, "cot_correct": cc}, ensure_ascii=False) + "\n")
        results[split_name] = entry
        print(split_name, {k: v for k, v in entry.items() if k in ("examples", "agreement")},
              "latent", entry["latent"]["accuracy"], "cot", entry.get("explicit_cot", {}).get("accuracy"))

    latent_test = results["test"]["latent"]["accuracy"]
    if latent_test >= args.high_threshold:
        reading = (f"BUDGET: the official latent path reaches {latent_test:.1%} on the templated test; "
                   "the §90 from-scratch students' floor is a training-budget failure, not a task failure")
    elif latent_test <= args.low_threshold:
        reading = (f"TASK: the official latent path reaches only {latent_test:.1%}; the latent path itself "
                   "finds this task hard and no from-scratch design at this budget was going to reach it")
    else:
        reading = f"MIXED: official latent path at {latent_test:.1%}, between the preregistered thresholds"
    summary = {"schema_version": 1, "contract": CONTRACT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
               "data_seed": DATA_SEED, "split_sizes": SPLIT_SIZES, "limit": limit,
               "latent_iterations": latent_iterations, "thresholds": {"high": args.high_threshold, "low": args.low_threshold},
               "results": results, "decision": {"reading": reading},
               "warnings": ["Unmodified official checkpoint; no training. Generated problems say nothing about GSM8K.",
                            "Same 2,000 test problems as ledger §89/§90 (same generator, seed and split order)."]}
    _atomic_json(summary, out / "summary.json")
    print(summary["decision"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N of each split (smoke)")
    parser.add_argument("--high-threshold", type=float, default=0.50)
    parser.add_argument("--low-threshold", type=float, default=0.20)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
