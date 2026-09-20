"""Recover xKV fidelity without reopening the locked rank-16 conclusion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    verify_full_reproduction_gate,
)
from scripts.run_codi_confirm_and_compress_native_kv import SOURCE_CONTRACT
from scripts.run_codi_preanswer_kv_subspace_discovery import _evaluate, _metrics, _sha
from scripts.run_codi_rank16_xkv_mechanism_confirmation import (
    CONTRACT as RANK16_CONTRACT,
    _correctness,
    _file_sha256,
    _serial_metrics,
)
from scripts.run_codi_task_aware_protected_xkv import (
    CONTRACT as PREDECESSOR_CONTRACT,
    FinalLatentTaskAwareFactorizer,
    _teacher_forced_arm,
    _topk_overlap,
)
from src.data.datasets import load_eval_set
from src.data.official_codi_training import align_official_codi_gsm8k_eval_rows
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.mech.rank16_xkv_confirmation import paired_bootstrap_interval
from src.mech.xkv_fidelity_frontier import (
    blend_feature_weights,
    blend_group_utilities,
    fidelity_gate,
    select_frontier_candidate,
)
from src.utils.config import load_config


CONTRACT = "official_codi_xkv_fidelity_frontier_holdout_v1"
GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
DEVELOPMENT_START = 256
DEVELOPMENT_EXAMPLES = 512
FINAL_START = 768
RANKS = (16, 24, 32, 40, 48)
TASK_WEIGHTS = (0.0, 0.5, 1.0)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _variance_utilities(source: dict) -> dict[tuple[int, ...], float]:
    result = {}
    for group in GROUPS:
        value = 0.0
        for kind in ("key", "value"):
            spectrum = source[f"{kind}_covariance"].eigenvalues
            value += float(spectrum[list(group)].double().clamp_min(0).sum())
        result[group] = max(value, torch.finfo(torch.float64).tiny)
    return result


def _factorizer(
    *, rank: int, latent_positions: int, task_weight: float,
    feature_weights, answer_utilities, variance_utilities, protected_bases,
):
    return FinalLatentTaskAwareFactorizer(
        latent_positions=latent_positions,
        groups=GROUPS,
        rank=rank,
        feature_weights=blend_feature_weights(
            feature_weights, task_weight=task_weight
        ),
        group_utilities=blend_group_utilities(
            answer_utilities, variance_utilities, task_weight=task_weight
        ),
        adaptive_ranks=True,
        protected_bases=protected_bases,
    )


def _ordinary_factorizer(*, rank: int, latent_positions: int):
    return FinalLatentTaskAwareFactorizer(
        latent_positions=latent_positions, groups=GROUPS, rank=rank
    )


def _dense_teacher(model, tokenizer, rows, latent_positions, batch_size, device):
    losses, logits, targets = _evaluate(
        model, tokenizer, rows, latent_positions, batch_size, device,
        enforce_answer_eligibility=False,
    )
    record = _serial_metrics(_metrics(losses, logits, targets, losses, logits))
    record["perplexity_from_mean_answer_nll"] = math.exp(
        min(record["mean_full_answer_nll"], 50)
    )
    record["first_token_top5_overlap"] = 1.0
    return record, losses, logits, targets


def _teacher_arm(
    model, tokenizer, rows, latent_positions, batch_size, device,
    dense_losses, dense_logits, factorizer,
):
    return _teacher_forced_arm(
        model,
        tokenizer,
        rows,
        latent_positions,
        batch_size,
        device,
        dense_losses,
        dense_logits,
        factorizer,
        enforce_answer_eligibility=False,
    )


def _dense_generation(
    model, tokenizer, rows, *, style, latent_positions, batch_size,
    max_new_tokens, device,
):
    outputs = generate_official_codi(
        model,
        tokenizer,
        [row["question"] for row in rows],
        latent_iterations=latent_positions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        answer_cue=style.answer_prefix,
        force_answer_cue=True,
    )
    correct = _correctness(outputs, rows)
    return {
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "examples": len(rows),
        "exact_sequence_agreement": 1.0,
    }, correct, outputs


def _generation_arm(
    model, tokenizer, rows, *, style, latent_positions, batch_size,
    max_new_tokens, device, factorizer, dense_outputs, dense_correct,
):
    outputs = generate_official_codi(
        model,
        tokenizer,
        [row["question"] for row in rows],
        latent_iterations=latent_positions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        kv_intervention=factorizer,
        answer_cue=style.answer_prefix,
        force_answer_cue=True,
    )
    correct = _correctness(outputs, rows)
    record = {
        "accuracy": float(correct.mean()),
        "correct": int(correct.sum()),
        "examples": len(rows),
        "accuracy_retained_fraction": (
            float(correct.mean() / dense_correct.mean())
            if float(dense_correct.mean()) else None
        ),
        "exact_sequence_agreement": sum(
            current == dense for current, dense in zip(outputs, dense_outputs)
        ) / len(outputs),
        "cache": factorizer.summary(),
    }
    return record, correct, outputs


def _candidate_name(rank: int, task_weight: float) -> str:
    return f"blend_r{rank}_w{task_weight:g}".replace(".", "p")


def _teacher_screen(
    *, rank, task_weight, ordinary_teacher, ordinary_delta,
    candidate_teacher, candidate_delta, seed, samples,
    storage_tolerance, minimum_fidelity,
):
    advantage = ordinary_delta - candidate_delta
    interval = paired_bootstrap_interval(advantage, seed=seed, samples=samples)
    bits_ratio = (
        candidate_teacher["cache"]["cache_bits"]
        / ordinary_teacher["cache"]["cache_bits"]
    )
    teacher_checks = {
        "positive_nll_interval": interval[0] > 0,
        "storage_matched": abs(bits_ratio - 1.0) <= storage_tolerance,
        "first_token_fidelity": (
            candidate_teacher["dense_first_token_top1_agreement"]
            >= minimum_fidelity
        ),
    }
    return {
        "name": _candidate_name(rank, task_weight),
        "rank": rank,
        "task_weight": task_weight,
        "mean_nll_advantage": float(advantage.mean()),
        "nll_bootstrap_95ci": interval,
        "full_to_ordinary_cache_bits_ratio": bits_ratio,
        "modelled_compression_ratio": candidate_teacher["cache"][
            "modelled_compression_ratio"
        ],
        "first_token_fidelity": candidate_teacher[
            "dense_first_token_top1_agreement"
        ],
        "teacher_checks": teacher_checks,
        "teacher_preeligible": all(teacher_checks.values()),
        "teacher_forced": candidate_teacher,
        "generation": None,
        "screen_gate": None,
        "screen_passed": False,
    }


def _complete_generation_screen(
    record, *, candidate_generation, candidate_correct, dense_correct,
    ordinary_correct, seed, samples, storage_tolerance, minimum_fidelity,
    noninferiority_margin,
):
    dense_interval = paired_bootstrap_interval(
        candidate_correct - dense_correct, seed=seed, samples=samples
    )
    ordinary_interval = paired_bootstrap_interval(
        candidate_correct - ordinary_correct, seed=seed + 1, samples=samples
    )
    gate = fidelity_gate(
        nll_interval=record["nll_bootstrap_95ci"],
        cache_bits_ratio=record["full_to_ordinary_cache_bits_ratio"],
        first_token_fidelity=record["first_token_fidelity"],
        dense_accuracy_interval=dense_interval,
        ordinary_accuracy_interval=ordinary_interval,
        storage_tolerance=storage_tolerance,
        minimum_first_token_fidelity=minimum_fidelity,
        accuracy_noninferiority_margin=noninferiority_margin,
    )
    record["generation"] = candidate_generation
    record["accuracy_difference_vs_dense_95ci"] = dense_interval
    record["accuracy_difference_vs_ordinary_95ci"] = ordinary_interval
    record["screen_gate"] = gate
    record["screen_passed"] = bool(gate["passed"])


def _save(args, summary, tensors, outputs, predictions):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save(
        {**summary, **tensors, "outputs": outputs},
        args.output_dir / "xkv_fidelity_frontier.pt",
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    if _parse_ints(args.ranks) != RANKS or _parse_floats(args.task_weights) != TASK_WEIGHTS:
        raise ValueError("the preregistered ranks and task weights cannot be changed")
    if (
        args.development_start != DEVELOPMENT_START
        or args.development_examples != DEVELOPMENT_EXAMPLES
        or args.final_start != FINAL_START
    ):
        raise ValueError("the preregistered split boundaries cannot be changed")

    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    predecessor = torch.load(
        args.previous_experiment_artifact, map_location="cpu", weights_only=False
    )
    rank16 = json.loads(args.rank16_summary.read_text())
    if source.get("contract") != SOURCE_CONTRACT:
        raise RuntimeError("wrong direct-cache source artifact")
    if predecessor.get("contract") != PREDECESSOR_CONTRACT:
        raise RuntimeError("wrong task-aware xKV predecessor artifact")
    if rank16.get("contract") != RANK16_CONTRACT:
        raise RuntimeError("wrong rank-16 confirmation summary")
    primary = rank16["confirmation"]["primary"]
    old_gate = primary["gate"]
    expected_old_checks = {
        "positive_nll_interval": True,
        "storage_matched": True,
        "first_token_fidelity": False,
        "generation_noninferior": True,
        "passed": False,
    }
    if old_gate != expected_old_checks:
        raise RuntimeError(
            "the predecessor result is not the preregistered fidelity-only failure"
        )
    if any(rank16.get(key) is not None for key in ("mechanism", "final_replication", "transfer")):
        raise RuntimeError("the rank-16 summary already opened a downstream stage")
    if predecessor["checkpoint_sha256"] != source["checkpoint_sha256"]:
        raise RuntimeError("source and predecessor checkpoints differ")

    feature_weights = predecessor["feature_weights"]
    protected_bases = predecessor["protected_bases"]
    if {int(layer) for layer, _ in protected_bases} != {11}:
        raise RuntimeError("the frozen protection map must contain only layer 11")
    answer_utilities = {
        tuple(int(item) for item in key.split(",")): float(value)
        for key, value in predecessor["compression_calibration"][
            "group_utilities"
        ].items()
    }
    if set(feature_weights) != set(GROUPS) or set(answer_utilities) != set(GROUPS):
        raise RuntimeError("predecessor calibration groups do not match")
    variance_utilities = _variance_utilities(source)

    device = select_device(args.device)
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
    if load_report.checkpoint_sha256 != predecessor["checkpoint_sha256"]:
        raise RuntimeError("loaded checkpoint does not match artifact lineage")
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    from datasets import load_dataset

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    spec = data_cfg.eval.gsm8k
    raw_test = load_dataset(
        str(spec.hf_id),
        data_files={str(spec.get("split", "test")): str(spec.data_file)},
        split=str(spec.get("split", "test")),
        verification_mode="no_checks",
    )
    development_rows = align_official_codi_gsm8k_eval_rows(
        raw_test,
        test,
        start=DEVELOPMENT_START,
        examples=DEVELOPMENT_EXAMPLES,
        enforce_answer_eligibility=False,
    )
    final_rows = align_official_codi_gsm8k_eval_rows(
        raw_test,
        test,
        start=FINAL_START,
        examples=len(test) - FINAL_START,
        enforce_answer_eligibility=False,
    )
    development_hash = _sha(
        [_normalized_question(row["question"]) for row in development_rows]
    )
    final_hash = _sha([_normalized_question(row["question"]) for row in final_rows])
    if development_hash != rank16["split_hashes"]["confirmation"]:
        raise RuntimeError("could not reconstruct the opened development slice")
    if final_hash != rank16["split_hashes"]["final"]:
        raise RuntimeError("could not reconstruct the still-untouched final slice")

    latent_positions = int(cfg.eval.latent_iterations)
    style = PromptStyle.from_config(data_cfg.prompt)
    dense_teacher, dense_losses, dense_logits, _ = _dense_teacher(
        model,
        tokenizer,
        development_rows,
        latent_positions,
        args.batch_size,
        device,
    )
    dense_generation, dense_correct, dense_outputs = _dense_generation(
        model,
        tokenizer,
        development_rows,
        style=style,
        latent_positions=latent_positions,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    development = {
        "dense": {"teacher_forced": dense_teacher, "generation": dense_generation},
        "ordinary_by_rank": {},
        "candidates": [],
    }
    tensors = {"development_deltas": {}}
    outputs = {"development": {"dense": dense_outputs}}
    predictions = []
    ordinary_cache = {}
    for rank in RANKS:
        factorizer = _ordinary_factorizer(rank=rank, latent_positions=latent_positions)
        teacher, delta = _teacher_arm(
            model,
            tokenizer,
            development_rows,
            latent_positions,
            args.batch_size,
            device,
            dense_losses,
            dense_logits,
            factorizer,
        )
        ordinary_cache[rank] = {"teacher": teacher, "delta": delta}
        tensors["development_deltas"][f"ordinary_r{rank}"] = delta
        development["ordinary_by_rank"][str(rank)] = {
            "teacher_forced": teacher,
            "generation": None,
        }

    candidate_cache = {}
    for rank in RANKS:
        for index, task_weight in enumerate(TASK_WEIGHTS):
            factorizer = _factorizer(
                rank=rank,
                latent_positions=latent_positions,
                task_weight=task_weight,
                feature_weights=feature_weights,
                answer_utilities=answer_utilities,
                variance_utilities=variance_utilities,
                protected_bases=protected_bases,
            )
            teacher, delta = _teacher_arm(
                model,
                tokenizer,
                development_rows,
                latent_positions,
                args.batch_size,
                device,
                dense_losses,
                dense_logits,
                factorizer,
            )
            record = _teacher_screen(
                rank=rank,
                task_weight=task_weight,
                ordinary_teacher=ordinary_cache[rank]["teacher"],
                ordinary_delta=ordinary_cache[rank]["delta"],
                candidate_teacher=teacher,
                candidate_delta=delta,
                seed=args.seed + 100 * rank + index,
                samples=args.bootstrap_samples,
                storage_tolerance=args.storage_tolerance,
                minimum_fidelity=args.minimum_first_token_fidelity,
            )
            candidate_cache[record["name"]] = delta
            tensors["development_deltas"][record["name"]] = delta
            development["candidates"].append(record)
            print("teacher screen", record["name"], record["teacher_checks"])

    eligible_ranks = {
        record["rank"] for record in development["candidates"]
        if record["teacher_preeligible"]
    }
    ordinary_generation_cache = {}
    for rank in sorted(eligible_ranks):
        factorizer = _ordinary_factorizer(rank=rank, latent_positions=latent_positions)
        generation, correct, arm_outputs = _generation_arm(
            model,
            tokenizer,
            development_rows,
            style=style,
            latent_positions=latent_positions,
            batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            factorizer=factorizer,
            dense_outputs=dense_outputs,
            dense_correct=dense_correct,
        )
        ordinary_generation_cache[rank] = (generation, correct, arm_outputs)
        development["ordinary_by_rank"][str(rank)]["generation"] = generation
        outputs["development"][f"ordinary_r{rank}"] = arm_outputs

    for index, record in enumerate(development["candidates"]):
        if not record["teacher_preeligible"]:
            continue
        factorizer = _factorizer(
            rank=record["rank"],
            latent_positions=latent_positions,
            task_weight=record["task_weight"],
            feature_weights=feature_weights,
            answer_utilities=answer_utilities,
            variance_utilities=variance_utilities,
            protected_bases=protected_bases,
        )
        generation, correct, arm_outputs = _generation_arm(
            model,
            tokenizer,
            development_rows,
            style=style,
            latent_positions=latent_positions,
            batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            factorizer=factorizer,
            dense_outputs=dense_outputs,
            dense_correct=dense_correct,
        )
        _, ordinary_correct, _ = ordinary_generation_cache[record["rank"]]
        _complete_generation_screen(
            record,
            candidate_generation=generation,
            candidate_correct=correct,
            dense_correct=dense_correct,
            ordinary_correct=ordinary_correct,
            seed=args.seed + 10_000 + index * 10,
            samples=args.bootstrap_samples,
            storage_tolerance=args.storage_tolerance,
            minimum_fidelity=args.minimum_first_token_fidelity,
            noninferiority_margin=args.accuracy_noninferiority_margin,
        )
        outputs["development"][record["name"]] = arm_outputs
        print("generation screen", record["name"], record["screen_gate"])

    selected = select_frontier_candidate(development["candidates"])
    selected_config = (
        {key: selected[key] for key in (
            "name", "rank", "task_weight", "modelled_compression_ratio",
            "nll_bootstrap_95ci", "screen_gate",
        )}
        if selected else None
    )
    summary = {
        "schema_version": 1,
        "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction,
        "lineage": {
            "source_artifact_sha256": _file_sha256(args.source_artifact),
            "previous_experiment_artifact_sha256": _file_sha256(
                args.previous_experiment_artifact
            ),
            "rank16_summary_sha256": _file_sha256(args.rank16_summary),
            "source_contract": SOURCE_CONTRACT,
            "previous_contract": PREDECESSOR_CONTRACT,
            "rank16_contract": RANK16_CONTRACT,
        },
        "preregistration": {
            "groups": [list(group) for group in GROUPS],
            "ranks": list(RANKS),
            "task_weights": list(TASK_WEIGHTS),
            "development_slice": [
                DEVELOPMENT_START, DEVELOPMENT_START + DEVELOPMENT_EXAMPLES
            ],
            "final_slice": [FINAL_START, len(test)],
            "selection_order": [
                "maximum modeled compression ratio",
                "maximum NLL lower confidence bound",
                "maximum task weight",
            ],
            "bootstrap_samples": args.bootstrap_samples,
            "minimum_first_token_fidelity": args.minimum_first_token_fidelity,
            "accuracy_noninferiority_margin": args.accuracy_noninferiority_margin,
            "storage_tolerance": args.storage_tolerance,
        },
        "split_hashes": {
            "development": development_hash,
            "final": final_hash,
        },
        "utility_calibration": {
            "answer_fisher": {",".join(map(str, key)): value for key, value in answer_utilities.items()},
            "cache_variance": {",".join(map(str, key)): value for key, value in variance_utilities.items()},
        },
        "development_frontier": development,
        "selected_candidate": selected_config,
        "final_replication": None,
        "transfer": None,
        "decision": {
            "development_passed": selected is not None,
            "final_passed": False,
            "claim": (
                "ADVANCE: one frozen fidelity-recovery candidate selected"
                if selected else
                "STOP: no development candidate restored preregistered fidelity"
            ),
        },
        "warnings": [
            "The 512-question development slice was already observed and is selection-only.",
            "The 551-question final slice is opened exactly once and only after development screening passes.",
            "This is a dense-reconstruction and modeled-storage proxy, not a native xKV latency benchmark.",
        ],
    }
    if selected is None:
        _save(args, summary, tensors, outputs, predictions)
        print(json.dumps(summary, indent=2))
        return summary

    rank = int(selected["rank"])
    task_weight = float(selected["task_weight"])
    final_dense_teacher, final_dense_losses, final_dense_logits, _ = _dense_teacher(
        model,
        tokenizer,
        final_rows,
        latent_positions,
        args.batch_size,
        device,
    )
    final_ordinary_teacher, final_ordinary_delta = _teacher_arm(
        model,
        tokenizer,
        final_rows,
        latent_positions,
        args.batch_size,
        device,
        final_dense_losses,
        final_dense_logits,
        _ordinary_factorizer(rank=rank, latent_positions=latent_positions),
    )
    final_candidate_teacher, final_candidate_delta = _teacher_arm(
        model,
        tokenizer,
        final_rows,
        latent_positions,
        args.batch_size,
        device,
        final_dense_losses,
        final_dense_logits,
        _factorizer(
            rank=rank,
            latent_positions=latent_positions,
            task_weight=task_weight,
            feature_weights=feature_weights,
            answer_utilities=answer_utilities,
            variance_utilities=variance_utilities,
            protected_bases=protected_bases,
        ),
    )
    final_dense_generation, final_dense_correct, final_dense_outputs = _dense_generation(
        model,
        tokenizer,
        final_rows,
        style=style,
        latent_positions=latent_positions,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    final_ordinary_generation, final_ordinary_correct, final_ordinary_outputs = _generation_arm(
        model,
        tokenizer,
        final_rows,
        style=style,
        latent_positions=latent_positions,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        factorizer=_ordinary_factorizer(rank=rank, latent_positions=latent_positions),
        dense_outputs=final_dense_outputs,
        dense_correct=final_dense_correct,
    )
    final_candidate_generation, final_candidate_correct, final_candidate_outputs = _generation_arm(
        model,
        tokenizer,
        final_rows,
        style=style,
        latent_positions=latent_positions,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        factorizer=_factorizer(
            rank=rank,
            latent_positions=latent_positions,
            task_weight=task_weight,
            feature_weights=feature_weights,
            answer_utilities=answer_utilities,
            variance_utilities=variance_utilities,
            protected_bases=protected_bases,
        ),
        dense_outputs=final_dense_outputs,
        dense_correct=final_dense_correct,
    )
    final_advantage = final_ordinary_delta - final_candidate_delta
    final_nll_interval = paired_bootstrap_interval(
        final_advantage, seed=args.seed + 50_000, samples=args.bootstrap_samples
    )
    final_dense_interval = paired_bootstrap_interval(
        final_candidate_correct - final_dense_correct,
        seed=args.seed + 50_001,
        samples=args.bootstrap_samples,
    )
    final_ordinary_interval = paired_bootstrap_interval(
        final_candidate_correct - final_ordinary_correct,
        seed=args.seed + 50_002,
        samples=args.bootstrap_samples,
    )
    final_bits_ratio = (
        final_candidate_teacher["cache"]["cache_bits"]
        / final_ordinary_teacher["cache"]["cache_bits"]
    )
    final_gate = fidelity_gate(
        nll_interval=final_nll_interval,
        cache_bits_ratio=final_bits_ratio,
        first_token_fidelity=final_candidate_teacher[
            "dense_first_token_top1_agreement"
        ],
        dense_accuracy_interval=final_dense_interval,
        ordinary_accuracy_interval=final_ordinary_interval,
        storage_tolerance=args.storage_tolerance,
        minimum_first_token_fidelity=args.minimum_first_token_fidelity,
        accuracy_noninferiority_margin=args.accuracy_noninferiority_margin,
    )
    summary["final_replication"] = {
        "selected_candidate": selected_config,
        "teacher_forced": {
            "dense": final_dense_teacher,
            "ordinary_xkv": final_ordinary_teacher,
            "selected": final_candidate_teacher,
        },
        "generation": {
            "dense": final_dense_generation,
            "ordinary_xkv": final_ordinary_generation,
            "selected": final_candidate_generation,
        },
        "mean_nll_advantage": float(final_advantage.mean()),
        "nll_bootstrap_95ci": final_nll_interval,
        "full_to_ordinary_cache_bits_ratio": final_bits_ratio,
        "accuracy_difference_vs_dense_95ci": final_dense_interval,
        "accuracy_difference_vs_ordinary_95ci": final_ordinary_interval,
        "gate": final_gate,
    }
    tensors["final_deltas"] = {
        "ordinary_xkv": final_ordinary_delta,
        "selected": final_candidate_delta,
    }
    outputs["final"] = {
        "dense": final_dense_outputs,
        "ordinary_xkv": final_ordinary_outputs,
        "selected": final_candidate_outputs,
    }
    for index, row in enumerate(final_rows):
        predictions.append({
            "split": "final",
            "dataset_index": FINAL_START + index,
            "question": row["question"],
            "gold": str(row["gold"]),
            "dense": final_dense_outputs[index],
            "ordinary_xkv": final_ordinary_outputs[index],
            "selected": final_candidate_outputs[index],
        })
    summary["decision"] = {
        "development_passed": True,
        "final_passed": bool(final_gate["passed"]),
        "claim": (
            "CONFIRMED: fidelity-recovered xKV improves on ordinary xKV at matched storage"
            if final_gate["passed"] else
            "STOP: development-selected fidelity recovery did not replicate on the locked final slice"
        ),
    }
    _save(args, summary, tensors, outputs, predictions)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--previous-experiment-artifact", type=Path, required=True)
    parser.add_argument("--rank16-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--ranks", default=",".join(map(str, RANKS)))
    parser.add_argument("--task-weights", default=",".join(map(str, TASK_WEIGHTS)))
    parser.add_argument("--development-start", type=int, default=DEVELOPMENT_START)
    parser.add_argument("--development-examples", type=int, default=DEVELOPMENT_EXAMPLES)
    parser.add_argument("--final-start", type=int, default=FINAL_START)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--minimum-first-token-fidelity", type=float, default=0.95)
    parser.add_argument("--storage-tolerance", type=float, default=0.001)
    parser.add_argument("--accuracy-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
