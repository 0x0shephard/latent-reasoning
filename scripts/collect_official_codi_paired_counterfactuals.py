"""Collect same-question correct/wrong state-12 counterfactuals on Kaggle.

Each noisy variant perturbs state 11 during the forced answer-cue forward pass,
then captures the resulting state 12 before greedy first-token selection. The
noise schedule is frozen; final-test noisy states are discarded immediately and
never written to the fitting artifact.
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

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_official_codi_correctness_detect_replication import split_test_like
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.data.official_codi_training import OFFICIAL_CODI_SOURCE_REVISION
from src.data.prompts import PromptStyle
from src.mech.endpoint_correctness_geometry import first_token_correct, readout_matrix
from src.mech.endpoint_margin_geometry import ANALYTIC_STATE
from src.mech.endpoint_paired_correction import (
    PAIRED_CORRECTION_CONTRACT,
    PAIRED_CORRECTION_SCHEMA_VERSION,
    OfficialCODIPerturbAndCapture,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _pair_coverage(correct: torch.Tensor) -> dict:
    paired = correct.any(dim=1) & (~correct).any(dim=1)
    return {
        "questions": int(correct.shape[0]),
        "paired_questions": int(paired.sum()),
        "paired_fraction": float(paired.double().mean()),
        "always_correct": int(correct.all(dim=1).sum()),
        "always_wrong": int((~correct).all(dim=1).sum()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    settings = cfg.endpoint_paired_correction
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    cache, readout_payload = load_margin_cache(args.states, args.readout)
    readout = readout_matrix(readout_payload).double()
    state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
    baseline_states = cache["evaluation_states"][:, state_index, :].double()
    gold = cache["evaluation_gold_first_token"].long()
    questions = list(cache["evaluation_questions"])
    expected = int(settings.expected_examples)
    if baseline_states.shape != (expected, 768) or len(questions) != expected:
        raise RuntimeError("cached GSM8K evaluation population changed")

    fit_index, select_index, test_index = split_test_like(
        expected,
        seed=int(settings.split_seed),
        fit_size=int(settings.fit_examples),
        select_size=int(settings.select_examples),
    )
    fitting_index = torch.cat([fit_index, select_index])
    indices = {
        "fit": fit_index.tolist(),
        "select": select_index.tolist(),
        "test": test_index.tolist(),
    }
    noise_schedule = [float(value) for value in settings.relative_noise_schedule]
    variants = [{"name": "baseline", "relative_noise": 0.0, "seed": None}]
    variants.extend(
        {
            "name": f"noise_{position:02d}_rms_{scale:g}",
            "relative_noise": scale,
            "seed": int(settings.noise_seed) + position,
        }
        for position, scale in enumerate(noise_schedule)
    )
    request = {
        "schema_version": PAIRED_CORRECTION_SCHEMA_VERSION,
        "contract": PAIRED_CORRECTION_CONTRACT,
        "phase": "same_question_state11_counterfactual_collection",
        "checkpoint_sha256": cache["metadata"]["checkpoint_sha256"],
        "source_request_sha256": cache.get("request_sha256"),
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "population": "GSM8K test",
        "indices": indices,
        "partition_sha256": _sha256_json(indices),
        "noise_schedule": variants,
        "state_perturbed": 11,
        "state_captured": 12,
        "outcome": "greedy gold first answer token",
        "final_test_noisy_states_saved": False,
        "precision": args.precision,
        "batch_size": args.batch_size,
    }
    request_sha256 = _sha256_json(request)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "paired_counterfactuals.pt"
    summary_path = args.output_dir / "paired_counterfactuals.json"
    if output_path.is_file() and summary_path.is_file():
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        if existing.get("request_sha256") != request_sha256:
            raise RuntimeError("existing paired artifact belongs to another request")
        print(f"[resume] already complete: {output_path}")
        return 0

    baseline_correct = first_token_correct(baseline_states, readout, gold).cpu()
    shard_dir = args.output_dir / "variant_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    baseline_shard = {
        "request_sha256": request_sha256,
        "name": "baseline",
        "relative_noise": 0.0,
        "seed": None,
        "states": baseline_states[fitting_index].float().cpu(),
        "correct": baseline_correct[fitting_index].cpu(),
        "predicted_token": (
            baseline_states[fitting_index] @ readout.T
        ).argmax(1).cpu(),
        "diagnostics": {"source": "validated colon-state cache"},
    }
    _atomic_torch_save(baseline_shard, shard_dir / "variant_00.pt")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("counterfactual collection requires a Kaggle GPU")
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
    if load_report.checkpoint_sha256 != request["checkpoint_sha256"]:
        raise RuntimeError("cached states and live checkpoint differ")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    answer_cue = PromptStyle.from_config(data_cfg.prompt).answer_prefix

    for position, variant in enumerate(variants[1:], start=1):
        shard_path = shard_dir / f"variant_{position:02d}.pt"
        if shard_path.is_file():
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            if (
                shard.get("request_sha256") != request_sha256
                or shard.get("name") != variant["name"]
            ):
                raise RuntimeError(f"variant shard {position} belongs to another request")
            print(f"[resume] {variant['name']}")
            continue
        collector = OfficialCODIPerturbAndCapture(
            model,
            relative_noise=float(variant["relative_noise"]),
            seed=int(variant["seed"]),
        )
        generations, endpoint = generate_official_codi(
            model,
            tokenizer,
            questions,
            latent_iterations=int(cfg.eval.latent_iterations),
            max_new_tokens=1,
            batch_size=args.batch_size,
            device=device,
            answer_endpoint_intervention=collector,
            answer_cue=answer_cue,
            force_answer_cue=True,
            return_endpoint_metadata=True,
        )
        if endpoint["endpoint_reached_count"] != expected:
            raise RuntimeError("not every counterfactual reached the answer cue")
        captured = collector.stacked(expected).double()
        predicted = (captured @ readout.T).argmax(1)
        decoded = [
            tokenizer.decode([int(token_id)], skip_special_tokens=True)
            for token_id in predicted.tolist()
        ]
        parity = sum(a == b for a, b in zip(decoded, generations)) / expected
        if parity < 0.99:
            raise RuntimeError(f"counterfactual readout parity failed: {parity:.4f}")
        # Final-test noisy predictions are deliberately discarded without ever
        # comparing them to gold. Only fit/select questions can define pairs.
        correct = predicted[fitting_index] == gold[fitting_index]
        _atomic_torch_save(
            {
                "request_sha256": request_sha256,
                **variant,
                "states": captured[fitting_index].float().cpu(),
                "correct": correct.cpu(),
                "predicted_token": predicted[fitting_index].cpu(),
                "diagnostics": collector.diagnostics(),
                "endpoint_coverage": endpoint["endpoint_reached_fraction"],
                "readout_parity": parity,
            },
            shard_path,
        )
        print(
            f"[complete] {variant['name']}: correct={float(correct.double().mean()):.3f}, "
            f"flips={float((correct != baseline_correct[fitting_index]).double().mean()):.3f}"
        )

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    shards = [
        torch.load(shard_dir / f"variant_{position:02d}.pt", map_location="cpu", weights_only=False)
        for position in range(len(variants))
    ]
    paired_states = torch.stack([shard["states"] for shard in shards], dim=1)
    paired_correct = torch.stack([shard["correct"] for shard in shards], dim=1)
    fit_correct = paired_correct[: len(fit_index)]
    select_correct = paired_correct[len(fit_index) :]
    coverage = {
        "fit": _pair_coverage(fit_correct),
        "select": _pair_coverage(select_correct),
    }
    minimum = int(settings.minimum_paired_questions_per_split)
    if any(item["paired_questions"] < minimum for item in coverage.values()):
        raise RuntimeError(
            f"paired coverage below {minimum}: {coverage}; expand the frozen noise schedule"
        )
    payload = {
        "schema_version": PAIRED_CORRECTION_SCHEMA_VERSION,
        "contract": PAIRED_CORRECTION_CONTRACT,
        "request_sha256": request_sha256,
        "source_request_sha256": cache.get("request_sha256"),
        "partition_sha256": request["partition_sha256"],
        "indices": indices,
        "variants": variants,
        "fit_select_states": paired_states,
        "fit_select_correct": paired_correct,
        "fit_select_predicted_token": torch.stack(
            [shard["predicted_token"] for shard in shards], dim=1
        ),
        "fit_select_gold_first_token": gold[fitting_index].cpu(),
        "test_baseline_states": baseline_states[test_index].float().cpu(),
        "test_gold_first_token": gold[test_index].cpu(),
        "test_baseline_correct": baseline_correct[test_index].cpu(),
    }
    _atomic_torch_save(payload, output_path)
    _atomic_json(
        {
            **request,
            "request_sha256": request_sha256,
            "state": "complete",
            "paired_coverage": coverage,
            "variant_results": [
                {
                    "name": shard["name"],
                    "relative_noise": shard["relative_noise"],
                    "correct_share_fit_select": float(shard["correct"].double().mean()),
                    "diagnostics": shard["diagnostics"],
                }
                for shard in shards
            ],
            "output_file": output_path.name,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        summary_path,
    )
    _atomic_json(
        json.loads(summary_path.read_text(encoding="utf-8")),
        args.output_dir / "run_manifest.json",
    )
    print(f"[complete] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
