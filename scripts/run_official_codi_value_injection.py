"""Run one latent value-injection arm with real greedy decoding.

Each invocation decodes one arm at one beta on one frozen split and writes an
atomic, resumable output directory. ``baseline`` ignores beta. Repair, offset,
and random arms apply identically many identically scaled edits and differ only
in which value they write into the value slots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_official_codi_correctness_detect_replication import split_test_like
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set
from src.data.official_codi_training import OFFICIAL_CODI_SOURCE_REVISION
from src.data.prompts import PromptStyle
from src.mech.latent_value_injection import (
    VALUE_INJECTION_CONTRACT,
    VALUE_INJECTION_SCHEMA_VERSION,
    OfficialCODILatentValueInjection,
    build_slot_tokens,
)
from src.mech.latent_workspace import parse_solution
from src.mech.endpoint_margin_geometry import resolve_output_embedding
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


ARMS = ("baseline", "gold", "offset", "random")


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normalize_question(value: str) -> str:
    return " ".join(str(value).split())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--split", choices=("select", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.arm != "baseline" and args.beta <= 0:
        raise ValueError("injection arms need a positive beta")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    settings = cfg.value_injection
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)

    payload = args.solutions.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != str(settings.solutions_sha256):
        raise RuntimeError("solutions file does not match the pinned hash")
    solution_rows = [
        json.loads(line) for line in payload.decode("utf-8").splitlines() if line
    ]
    solutions_by_question = {
        _normalize_question(row["question"]): parse_solution(row["answer"])
        for row in solution_rows
    }

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    style = PromptStyle.from_config(data_cfg.prompt)
    full_examples = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    expected = int(settings.expected_examples)
    if len(full_examples) != expected:
        raise RuntimeError("GSM8K evaluation count drifted")
    fit_index, select_index, test_index = split_test_like(
        expected,
        seed=int(settings.split_seed),
        fit_size=int(settings.fit_examples),
        select_size=int(settings.select_examples),
    )
    indices = {
        "fit": fit_index.tolist(),
        "select": select_index.tolist(),
        "test": test_index.tolist(),
    }
    partition_sha256 = _sha256_json(indices)
    if partition_sha256 != str(settings.expected_partition_sha256):
        raise RuntimeError("partition drifted from the frozen population")
    rows = indices[args.split]
    examples = [full_examples[index] for index in rows]
    intermediates = []
    for example in examples:
        key = _normalize_question(example["question"])
        if key not in solutions_by_question:
            raise RuntimeError("an evaluation question is missing from solutions")
        intermediates.append(solutions_by_question[key]["intermediates"])

    request = {
        "schema_version": VALUE_INJECTION_SCHEMA_VERSION,
        "contract": VALUE_INJECTION_CONTRACT,
        "phase": "latent_value_injection_generation",
        "arm": args.arm,
        "beta": float(args.beta),
        "split": args.split,
        "rows": rows,
        "partition_sha256": partition_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "solutions_sha256": digest,
        "value_slots": [int(v) for v in settings.value_slots],
        "random_seed": int(settings.random_token_seed),
        "precision": args.precision,
        "eval_batch_size": args.eval_batch_size,
    }
    request_sha256 = _sha256_json(request)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    predictions_path = args.output_dir / "gsm8k.jsonl"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text())
        if existing.get("request_sha256") != request_sha256:
            raise RuntimeError("existing arm output belongs to another request")
        print(f"[resume] already complete: {summary_path}")
        return 0

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("value injection requires a GPU")
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id),
        revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename),
        expected_sha256=str(cfg.checkpoint.sha256),
        token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model),
        base_revision=str(cfg.model.base_revision),
        dtype=dtype,
        settings=cfg.model,
        token=token,
    )
    load_report = load_official_checkpoint(
        model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    intervention = None
    injectable = [bool(values) for values in intermediates]
    if args.arm != "baseline":
        readout = resolve_output_embedding(model)[: int(model.eot_id)].detach().cpu()
        slot_tokens = build_slot_tokens(
            intermediates,
            tokenizer,
            arm=args.arm,
            vocabulary_size=readout.shape[0],
            random_seed=int(settings.random_token_seed),
            slots=tuple(int(v) for v in settings.value_slots),
        )
        intervention = OfficialCODILatentValueInjection(
            model,
            readout=readout,
            slot_tokens=slot_tokens,
            beta=float(args.beta),
            latent_iterations=int(cfg.eval.latent_iterations),
            slots=tuple(int(v) for v in settings.value_slots),
        )

    started = time.perf_counter()
    try:
        generations, endpoint = generate_official_codi(
            model,
            tokenizer,
            [example["question"] for example in examples],
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=int(cfg.eval.max_new_tokens),
            batch_size=args.eval_batch_size,
            device=device,
            kv_intervention=intervention,
            answer_cue=style.answer_prefix,
            force_answer_cue=True,
            return_endpoint_metadata=True,
        )
    finally:
        if intervention is not None:
            intervention.close()
    elapsed = time.perf_counter() - started
    numeric = [
        bool(answers_match(generation, example["gold"]))
        for generation, example in zip(generations, examples)
    ]
    temporary = predictions_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for index, example, generation, is_numeric, can_inject in zip(
            rows, examples, generations, numeric, injectable
        ):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "question": example["question"],
                        "gold": str(example["gold"]),
                        "generation": generation,
                        "numeric_exact_match_correct": is_numeric,
                        "injectable": can_inject,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(predictions_path)
    _atomic_json(
        {
            **request,
            "request_sha256": request_sha256,
            "checkpoint_sha256": load_report.checkpoint_sha256,
            "state": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluated_examples": len(examples),
            "numeric_exact_match_accuracy": float(sum(numeric) / len(examples)),
            "endpoint_coverage": {
                key: value
                for key, value in endpoint.items()
                if key != "endpoint_reached"
            },
            "intervention_diagnostics": (
                intervention.diagnostics() if intervention is not None else None
            ),
            "evaluation_seconds": elapsed,
        },
        summary_path,
    )
    print(
        f"[complete] {args.arm} beta={args.beta:g} split={args.split}: "
        f"accuracy={sum(numeric)/len(examples):.4f} ({elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
