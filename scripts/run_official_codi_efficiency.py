"""Measure the two findings-derived efficiency routes on real hardware.

Route 1 (rank-k answer readout): microbenchmark the full lm_head projection
against the factorized ``(h U)(W U)^T`` readout at the §40 ranks. The accuracy
cost is already measured (§40 retention arms: rank 32 keeps 94.4% of exact
match); this script measures the wall-clock side.

Route 2 (latent-loop truncation): decode full GSM8K with the released protocol
at each latent-iteration budget and record numeric exact match and wall clock.
The model was trained at M=6, so accuracy at smaller budgets is the open
question this measurement answers.

This is a protocol-frozen measurement study: no hypothesis gates, no tuning, one
pass per condition, everything reported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
from src.data.answer_extract import answers_match
from src.data.datasets import load_eval_set
from src.data.official_codi_training import OFFICIAL_CODI_SOURCE_REVISION
from src.data.prompts import PromptStyle
from src.mech.endpoint_margin_geometry import resolve_output_embedding
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


def benchmark_readout(readout: torch.Tensor, *, ranks, batch, iterations, device):
    """Median microseconds for full versus rank-k answer readout."""
    matrix = readout.to(device=device, dtype=torch.float32)
    hidden = torch.randn(batch, matrix.shape[1], device=device)

    def timed(function):
        for _ in range(10):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(int(iterations)):
            start = time.perf_counter()
            function()
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        samples.sort()
        return 1e6 * samples[len(samples) // 2]

    full_time = timed(lambda: hidden @ matrix.T)
    results = {"full_lm_head_us": full_time, "ranks": {}}
    generator = torch.Generator(device="cpu").manual_seed(0)
    for rank in ranks:
        basis = torch.linalg.qr(
            torch.randn(matrix.shape[1], int(rank), generator=generator)
        )[0].to(device)
        projected = (matrix @ basis).contiguous()
        rank_time = timed(lambda: (hidden @ basis) @ projected.T)
        results["ranks"][str(int(rank))] = {
            "rank_readout_us": rank_time,
            "speedup_x": full_time / rank_time if rank_time > 0 else None,
        }
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    settings = cfg.efficiency_measurement
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the efficiency measurement requires a GPU")
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
    torch.cuda.empty_cache()

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    style = PromptStyle.from_config(data_cfg.prompt)
    examples = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    if len(examples) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("GSM8K evaluation count drifted")
    questions = [example["question"] for example in examples]

    sweep = []
    for budget in [int(m) for m in settings.latent_iterations_grid]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        generations, endpoint = generate_official_codi(
            model,
            tokenizer,
            questions,
            latent_iterations=budget,
            max_new_tokens=int(cfg.eval.max_new_tokens),
            batch_size=int(settings.eval_batch_size),
            device=device,
            answer_cue=style.answer_prefix,
            force_answer_cue=True,
            return_endpoint_metadata=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        accuracy = float(
            sum(
                answers_match(generation, example["gold"])
                for generation, example in zip(generations, examples)
            )
            / len(examples)
        )
        sweep.append(
            {
                "latent_iterations": budget,
                "numeric_exact_match_accuracy": accuracy,
                "wall_clock_seconds": elapsed,
                "endpoint_reached_fraction": endpoint["endpoint_reached_fraction"],
            }
        )
        print(
            f"[m-sweep] M={budget}: accuracy={accuracy:.4f} "
            f"({elapsed:.1f}s, cue coverage "
            f"{endpoint['endpoint_reached_fraction']:.3f})"
        )

    readout = resolve_output_embedding(model)[: int(model.eot_id)].detach()
    benchmark = benchmark_readout(
        readout,
        ranks=[int(r) for r in settings.readout_ranks],
        batch=int(settings.benchmark_batch),
        iterations=int(settings.benchmark_iterations),
        device=device,
    )

    baseline = next(item for item in sweep if item["latent_iterations"] == 6)
    _atomic_json(
        {
            "analysis": "official_codi_efficiency_measurement",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": load_report.checkpoint_sha256,
            "reproduction_gate": reproduction,
            "device": torch.cuda.get_device_name(device),
            "precision": args.precision,
            "eval_batch_size": int(settings.eval_batch_size),
            "latent_budget_sweep": sweep,
            "latency_saving_vs_m6": {
                str(item["latent_iterations"]): 1.0
                - item["wall_clock_seconds"] / baseline["wall_clock_seconds"]
                for item in sweep
            },
            "accuracy_cost_vs_m6_points": {
                str(item["latent_iterations"]): 100
                * (
                    baseline["numeric_exact_match_accuracy"]
                    - item["numeric_exact_match_accuracy"]
                )
                for item in sweep
            },
            "readout_benchmark": benchmark,
            "rank32_accuracy_note": (
                "Accuracy at rank 32 is measured, not estimated: the §40 "
                "retention arm scored 0.4094 exact match against the 0.4337 "
                "baseline (94.4% retained)."
            ),
        },
        args.output,
    )
    print(f"[complete] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
