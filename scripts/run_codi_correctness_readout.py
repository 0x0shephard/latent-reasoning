"""Carrier-based error detection, the recording half (ledger §125, Build 2).  No training.

For every GSM8K test question, correct or wrong, record the six thought states under
native decoding, the native answer and a confidence baseline (the probability and margin
of the model's own first answer token under the forced cue).  The analysis
(``scripts/analyze_correctness_readout.py``) turns the states into the model's own
numeric readouts per slot and asks whether the card's carrier slots predict wrong
answers better than inert slots or confidence alone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from src.data.chain_pools import gsm8k_chain_row, load_raw_eval_rows
from src.data.counterfactual_chain import parse_chain
from src.data.official_codi_training import collate_official_codi_kv_rows
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import extract_official_answer_number
from src.mech.cache_carrier import latent_path, record_native_trajectory
from src.models.official_codi import (
    build_official_codi_gpt2,
    checkpoint_shard_hashes,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config

CONTRACT = "official_codi_correctness_readout_v1"
SMOKE = {"questions": 48}


def _number_text(value: float | None) -> str | None:
    if value is None or value != value:
        return None
    return str(int(round(value))) if abs(value - round(value)) < 1e-6 else str(value)


def confidence_rows(questions: list[str], native_numbers: list[float | None]) -> tuple[list[dict], list[int]]:
    """Rows whose 'answer' is the model's own native number, so the forced-cue first token is the model's."""
    rows, index = [], []
    for i, (q, n) in enumerate(zip(questions, native_numbers)):
        text = _number_text(n)
        if text is None or not text[0].isdigit():
            continue
        rows.append({"question": q, "cot": "", "answer": text}); index.append(i)
    return rows, index


def first_token_confidence(model, tokenizer, rows, *, latent_positions, batch_size, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(probability of the gold first token, margin top1 - top2 in log space) per row."""
    probs, margins = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = collate_official_codi_kv_rows(tokenizer, rows[start:start + batch_size], bot_token_id=model.bot_id,
                                                  enforce_answer_eligibility=False).to(device)
            logits, gold, _ = latent_path(model, batch, latent_positions=latent_positions)
            logp = logits.float().log_softmax(-1)
            top2 = logp.topk(2, dim=-1).values
            probs.append(logp.gather(1, gold.unsqueeze(1)).squeeze(1).exp().cpu()); margins.append((top2[:, 0] - top2[:, 1]).cpu())
    return torch.cat(probs), torch.cat(margins)


def run(args):
    smoke = bool(args.smoke)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cfg = load_config(args.config)
    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token,
        shard_sha256=checkpoint_shard_hashes(cfg))
    model, tokenizer = build_official_codi_gpt2(base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
                                                dtype=dtype, settings=cfg.model, token=token)
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype).eval()
    latent_positions = int(cfg.eval.latent_iterations)

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    raw = load_raw_eval_rows(dict(data_cfg.eval.gsm8k))
    if smoke:
        raw = raw[:SMOKE["questions"]]
    questions = [str(r["question"]) for r in raw]
    golds = [float(str(r["answer"]).rsplit("####", 1)[-1].strip().replace(",", "")) for r in raw]
    chains = [gsm8k_chain_row(r) for r in raw]
    steps = [len(parse_chain(c["cot"])) if c else -1 for c in chains]
    intermediates = [[e["result"] for e in parse_chain(c["cot"])] if c else [] for c in chains]

    traj, outputs = record_native_trajectory(model, tokenizer, questions, latent_iterations=latent_positions, batch_size=args.batch_size, device=device)
    native = [extract_official_answer_number(o) for o in outputs]
    correct = [n is not None and abs(n - g) < 1e-6 for n, g in zip(native, golds)]
    accuracy = sum(correct) / len(correct)
    print(f"native accuracy {accuracy:.4f} on {len(correct)} rows")

    conf_rows, conf_index = confidence_rows(questions, native)
    prob = torch.full((len(questions),), float("nan")); margin = torch.full((len(questions),), float("nan"))
    if conf_rows:
        p, m = first_token_confidence(model, tokenizer, conf_rows, latent_positions=latent_positions, batch_size=args.batch_size, device=device)
        prob[conf_index] = p; margin[conf_index] = m
    _atomic_torch_save({"states": traj.states.half(), "gold": torch.tensor(golds), "native": torch.tensor([float("nan") if n is None else n for n in native]),
                        "correct": torch.tensor(correct), "steps": torch.tensor(steps), "first_token_prob": prob, "first_token_margin": margin,
                        "intermediates": intermediates, "outputs": outputs}, out / "readout_states.pt")
    summary = {"schema_version": 1, "contract": CONTRACT + ("_smoke" if smoke else ""), "smoke": smoke, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": load_report.checkpoint_sha256, "rows": len(questions), "native_accuracy": accuracy,
               "published_accuracy": float(cfg.accuracy_gate.published_accuracy.gsm8k), "rows_with_confidence": len(conf_rows),
               "steps_histogram": {str(k): steps.count(k) for k in sorted(set(steps))}, "latent_iterations": latent_positions,
               "base_model": str(cfg.model.base_model), "seconds": time.perf_counter() - started,
               "warnings": ["Native decoding on the released path; states are the six thought outputs. Confidence is the forced-cue probability of the model's own first answer token."]}
    _atomic_json(summary, out / "summary.json")
    print(json.dumps({k: summary[k] for k in ("native_accuracy", "published_accuracy", "rows", "rows_with_confidence")}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
