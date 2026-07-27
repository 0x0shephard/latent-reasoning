"""Validate numerical safety and decoder parity before the KV-risk pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_kv_risk_pilot import (
    load_model_and_tokenizer,
    resolve_device,
)
from src.eval.kv_risk_pilot import (
    DecodeRequest,
    atomic_json,
    build_prompt,
    deterministic_sample,
    eos_token_ids,
    extracted_answer,
    generate_one,
    load_candidate_dataset,
    score_answer,
)
from src.utils.config import load_config


PREFLIGHT_SCHEMA_VERSION = 1


def _request(
    cfg,
    *,
    retention: float,
    max_new_tokens: int,
    sampling_seed: int = 0,
) -> DecodeRequest:
    return DecodeRequest(
        retention=retention,
        max_new_tokens=max_new_tokens,
        recent_window=int(cfg.compression.recent_window),
        heavy_fraction=float(cfg.compression.heavy_fraction),
        early_entropy_tokens=int(cfg.analysis.early_entropy_tokens),
        temperature=0.0,
        top_p=1.0,
        sampling_seed=sampling_seed,
    )


@torch.inference_mode()
def _hf_greedy_token_ids(
    model,
    tokenizer,
    *,
    question: str,
    instruction: str,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    encoded = build_prompt(tokenizer, question, instruction)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    stop_ids = sorted(eos_token_ids(model, tokenizer))
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=(stop_ids if len(stop_ids) > 1 else stop_ids[0]),
    )
    return [
        int(value)
        for value in generated[0, prompt_length:].detach().cpu().tolist()
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parity-examples", type=int, default=2)
    parser.add_argument("--parity-tokens", type=int, default=64)
    parser.add_argument("--gate-examples", type=int, default=8)
    parser.add_argument("--gate-max-new-tokens", type=int, default=1024)
    parser.add_argument("--compression-smoke-tokens", type=int, default=128)
    parser.add_argument("--minimum-parsed", type=int, default=6)
    parser.add_argument("--minimum-correct", type=int, default=2)
    return parser


def _run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the mandatory preflight requires a Kaggle GPU")
    torch.manual_seed(int(cfg.screen.seed))
    torch.cuda.manual_seed_all(int(cfg.screen.seed))
    model, tokenizer, revision, dtype = load_model_and_tokenizer(cfg, device)
    gsm8k = load_candidate_dataset("gsm8k", cfg.datasets.gsm8k)
    examples = deterministic_sample(
        gsm8k,
        max(args.parity_examples, args.gate_examples),
        seed=int(cfg.screen.seed) + 9_001,
    )
    instruction = str(cfg.prompt.instruction)

    parity: list[dict] = []
    for example in examples[: args.parity_examples]:
        hf_ids = _hf_greedy_token_ids(
            model,
            tokenizer,
            question=str(example["question"]),
            instruction=instruction,
            max_new_tokens=args.parity_tokens,
            device=device,
        )
        custom = generate_one(
            model,
            tokenizer,
            question=str(example["question"]),
            instruction=instruction,
            request=_request(
                cfg,
                retention=1.0,
                max_new_tokens=args.parity_tokens,
            ),
            device=device,
        )
        custom_ids = [int(value) for value in custom["generated_token_ids"]]
        exact = custom_ids == hf_ids
        parity.append(
            {
                "example_id": example["example_id"],
                "exact_token_match": exact,
                "hf_generated_tokens": len(hf_ids),
                "custom_generated_tokens": len(custom_ids),
                "first_mismatch": next(
                    (
                        index
                        for index, (left, right) in enumerate(
                            zip(custom_ids, hf_ids)
                        )
                        if left != right
                    ),
                    (
                        None
                        if len(custom_ids) == len(hf_ids)
                        else min(len(custom_ids), len(hf_ids))
                    ),
                ),
            }
        )
        if not exact:
            raise RuntimeError(
                "custom full-cache decoding does not exactly match "
                f"transformers.generate on {example['example_id']}"
            )
        if bool(custom["degenerate_generation"]):
            raise RuntimeError(
                f"degenerate parity generation on {example['example_id']}"
            )

    gate_records: list[dict] = []
    for example in examples[: args.gate_examples]:
        result = generate_one(
            model,
            tokenizer,
            question=str(example["question"]),
            instruction=instruction,
            request=_request(
                cfg,
                retention=1.0,
                max_new_tokens=args.gate_max_new_tokens,
            ),
            device=device,
        )
        if bool(result["degenerate_generation"]):
            raise RuntimeError(
                f"degenerate GSM8K gate generation on {example['example_id']}"
            )
        prediction = extracted_answer(
            str(result["generation"]),
            str(example["grader"]),
        )
        correct = score_answer(
            str(result["generation"]),
            str(example["gold"]),
            str(example["grader"]),
        )
        gate_records.append(
            {
                "example_id": example["example_id"],
                "prediction": prediction,
                "correct": bool(correct),
                "generated_tokens": result["generated_tokens"],
                "finish_reason": result["finish_reason"],
                "unique_generated_tokens": result["unique_generated_tokens"],
                "maximum_token_run": result["maximum_token_run"],
                "early_entropy_mean": result["early_entropy_mean"],
            }
        )
        print(
            f"[preflight] gsm8k {len(gate_records)}/{args.gate_examples} "
            f"parsed={int(prediction is not None)} correct={int(correct)} "
            f"tokens={result['generated_tokens']}",
            flush=True,
        )

    parsed = sum(record["prediction"] is not None for record in gate_records)
    correct = sum(record["correct"] for record in gate_records)
    if parsed < args.minimum_parsed:
        raise RuntimeError(
            f"GSM8K functional gate parsed only {parsed}/{args.gate_examples}; "
            f"required {args.minimum_parsed}"
        )
    if correct < args.minimum_correct:
        raise RuntimeError(
            f"GSM8K functional gate solved only {correct}/{args.gate_examples}; "
            f"required {args.minimum_correct}"
        )

    compressed = generate_one(
        model,
        tokenizer,
        question=str(examples[0]["question"]),
        instruction=instruction,
        request=_request(
            cfg,
            retention=0.5,
            max_new_tokens=args.compression_smoke_tokens,
        ),
        device=device,
    )
    if bool(compressed["degenerate_generation"]):
        raise RuntimeError("compressed-cache smoke generation is degenerate")

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo": str(cfg.model.repo_id),
        "model_revision": revision,
        "configured_precision": str(cfg.model.precision),
        "resolved_dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(
            torch.cuda.get_device_capability(device)
        ),
        "parity": {
            "examples": len(parity),
            "tokens_per_example": args.parity_tokens,
            "all_exact": all(value["exact_token_match"] for value in parity),
            "records": parity,
        },
        "gsm8k_gate": {
            "examples": len(gate_records),
            "maximum_new_tokens": args.gate_max_new_tokens,
            "parsed": parsed,
            "correct": correct,
            "minimum_parsed": args.minimum_parsed,
            "minimum_correct": args.minimum_correct,
            "records": gate_records,
        },
        "compression_smoke": {
            "retention": 0.5,
            "generated_tokens": compressed["generated_tokens"],
            "degenerate_generation": compressed["degenerate_generation"],
            "early_entropy_mean": compressed["early_entropy_mean"],
            "realized_generated_retention": compressed[
                "realized_generated_retention"
            ],
        },
    }


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    args = _parser().parse_args()
    try:
        report = _run(args)
        exit_code = 0
    except Exception as exc:
        report = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 4
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[preflight] status={report['status']}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
