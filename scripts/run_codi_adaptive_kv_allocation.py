"""Test answer-sensitive per-layer K/V rank allocation at matched xKV storage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
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
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES,
    GSM8K_TRAIN_URL,
    sample_gsm8k_train_calibration,
)
from scripts.run_codi_fidelity_residual_xkv import CONTRACT as PREDECESSOR_CONTRACT
from scripts.run_codi_preanswer_kv_subspace_discovery import _sha
from scripts.run_codi_task_aware_protected_xkv import FinalLatentTaskAwareFactorizer
from scripts.run_codi_xkv_fidelity_frontier import (
    _complete_generation_screen,
    _dense_generation,
    _dense_teacher,
    _file_sha256,
    _generation_arm,
    _teacher_arm,
)
from src.data.datasets import load_eval_set
from src.data.official_codi_training import (
    align_official_codi_gsm8k_eval_rows,
    collate_official_codi_kv_rows,
)
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.mech.adaptive_kv_allocation import (
    AdaptiveComponentFactorizer,
    CalibrationSpectrumCollector,
    blend_component_utilities,
    select_adaptive_candidate,
)
from src.mech.preanswer_kv_subspace import official_codi_preanswer_kv_forward
from src.mech.rank16_xkv_confirmation import paired_bootstrap_interval
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_adaptive_kv_allocation_holdout_v1"
FRESH_CALIBRATION_EXAMPLES = 512
FRESH_SCREEN_EXAMPLES = 512
FINAL_START = 768
BASELINE_RANKS = (48, 64, 80)
ANSWER_WEIGHTS = (0.0, 0.5, 1.0)
MAXIMUM_COMPONENT_RANK = 96
PER_LAYER_GROUPS = tuple((layer,) for layer in range(12))


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _with_gold(rows):
    return [{**row, "gold": str(row.get("gold", row["answer"]))} for row in rows]


def _fresh_train_rows(
    train, test, predecessor, *, calibration_examples: int, screen_examples: int
):
    used_examples = int(predecessor["sampling"]["selected_examples"])
    seed = int(predecessor["sampling"]["sampling_seed"])
    test_questions = {_normalized_question(row["question"]) for row in test}
    used_rows, used_audit = sample_gsm8k_train_calibration(
        train, test_questions=test_questions, examples=used_examples, seed=seed
    )
    for key in ("selected_source_indices_sha256", "selected_normalized_questions_sha256"):
        if used_audit[key] != predecessor["sampling"][key]:
            raise RuntimeError("could not reconstruct the predecessor train prefix")
    total = used_examples + int(calibration_examples) + int(screen_examples)
    rows, sampling = sample_gsm8k_train_calibration(
        train, test_questions=test_questions, examples=total, seed=seed
    )
    if [row["question"] for row in rows[:used_examples]] != [
        row["question"] for row in used_rows
    ]:
        raise RuntimeError("fresh sampling does not preserve the predecessor prefix")
    calibration_stop = used_examples + int(calibration_examples)
    calibration = _with_gold(rows[used_examples:calibration_stop])
    screen = _with_gold(rows[calibration_stop:])
    sampling.update({
        "predecessor_examples_reconstructed": used_examples,
        "predecessor_questions_sha256": _sha(
            [_normalized_question(row["question"]) for row in used_rows]
        ),
        "fresh_calibration_questions_sha256": _sha(
            [_normalized_question(row["question"]) for row in calibration]
        ),
        "fresh_screen_questions_sha256": _sha(
            [_normalized_question(row["question"]) for row in screen]
        ),
    })
    return calibration, screen, sampling


def _collect_utility_curves(
    model, tokenizer, rows, *, latent_positions, maximum_rank, batch_size, device
):
    collector = CalibrationSpectrumCollector(
        latent_positions=latent_positions, maximum_rank=maximum_rank
    )
    connectivity = []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start : start + batch_size], bot_token_id=model.bot_id
        ).to(device)
        output = official_codi_preanswer_kv_forward(
            model,
            batch,
            latent_positions=latent_positions,
            return_gradients=True,
            gradient_objective="answer_nll",
            kv_intervention=collector,
            return_full_cache_gradients=True,
        )
        collector.consume(output.full_key_gradients, output.full_value_gradients)
        connectivity.append(output.gradient_connected)
        del batch, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    connected = torch.stack(connectivity)
    if not bool(connected.all()):
        raise RuntimeError("answer gradients are not connected to every K/V cache")
    reconstruction, fisher = collector.curves()
    return reconstruction, fisher, connected, collector.token_counts


def _ordinary_factorizer(*, rank: int, latent_positions: int):
    return FinalLatentTaskAwareFactorizer(
        latent_positions=latent_positions, groups=PER_LAYER_GROUPS, rank=rank
    )


def _adaptive_factorizer(
    *, rank: int, answer_weight: float, latent_positions: int,
    reconstruction_utilities, fisher_utilities,
):
    utilities = blend_component_utilities(
        reconstruction_utilities, fisher_utilities, answer_weight=answer_weight
    )
    return AdaptiveComponentFactorizer(
        latent_positions=latent_positions,
        baseline_rank=rank,
        utilities=utilities,
        maximum_component_rank=MAXIMUM_COMPONENT_RANK,
    )


def _candidate_name(rank: int, answer_weight: float) -> str:
    return f"adaptive_kv_r{rank}_w{answer_weight:g}".replace(".", "p")


def _teacher_record(
    *, rank, answer_weight, ordinary_teacher, ordinary_delta,
    candidate_teacher, candidate_delta, seed, samples, storage_tolerance,
    minimum_fidelity,
):
    advantage = ordinary_delta - candidate_delta
    interval = paired_bootstrap_interval(advantage, seed=seed, samples=samples)
    bits_ratio = candidate_teacher["cache"]["cache_bits"] / ordinary_teacher["cache"]["cache_bits"]
    teacher_checks = {
        "positive_nll_interval": interval[0] > 0,
        "storage_matched": abs(bits_ratio - 1.0) <= storage_tolerance,
        "first_token_fidelity": (
            candidate_teacher["dense_first_token_top1_agreement"] >= minimum_fidelity
        ),
    }
    return {
        "name": _candidate_name(rank, answer_weight),
        "rank": int(rank),
        "answer_weight": float(answer_weight),
        "mean_nll_advantage": float(advantage.mean()),
        "nll_bootstrap_95ci": interval,
        "adaptive_to_ordinary_cache_bits_ratio": bits_ratio,
        "full_to_ordinary_cache_bits_ratio": bits_ratio,
        "modelled_compression_ratio": candidate_teacher["cache"]["modelled_compression_ratio"],
        "first_token_fidelity": candidate_teacher["dense_first_token_top1_agreement"],
        "mean_first_token_kl_from_dense": candidate_teacher["mean_first_token_kl_from_dense"],
        "teacher_checks": teacher_checks,
        "teacher_preeligible": all(teacher_checks.values()),
        "teacher_forced": candidate_teacher,
        "generation": None,
        "screen_gate": None,
        "screen_passed": False,
    }


def _curve_audit(reconstruction, fisher):
    result = {}
    for component in sorted(reconstruction):
        name = f"layer_{component[0]:02d}_{component[1]}"
        result[name] = {
            "reconstruction_utility_total": float(reconstruction[component].sum()),
            "answer_fisher_utility_total": float(fisher[component].sum()),
            "rank_1_answer_fisher_fraction": float(
                fisher[component][0] / fisher[component].sum().clamp_min(1e-30)
            ),
            "rank_16_answer_fisher_fraction": float(
                fisher[component][:16].sum() / fisher[component].sum().clamp_min(1e-30)
            ),
        }
    return result


def _save(args, summary, tensors, outputs, predictions):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save(
        {**summary, **tensors, "outputs": outputs},
        args.output_dir / "adaptive_kv_allocation.pt",
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    if _parse_ints(args.baseline_ranks) != BASELINE_RANKS:
        raise ValueError("the preregistered baseline ranks cannot be changed")
    if _parse_floats(args.answer_weights) != ANSWER_WEIGHTS:
        raise ValueError("the preregistered utility blends cannot be changed")
    if args.maximum_component_rank != MAXIMUM_COMPONENT_RANK:
        raise ValueError("the preregistered component-rank cap cannot be changed")
    if (
        args.fresh_calibration_examples != FRESH_CALIBRATION_EXAMPLES
        or args.fresh_screen_examples != FRESH_SCREEN_EXAMPLES
        or args.final_start != FINAL_START
    ):
        raise ValueError("the preregistered split sizes cannot be changed")

    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    predecessor = json.loads(args.previous_summary.read_text())
    if predecessor.get("contract") != PREDECESSOR_CONTRACT:
        raise RuntimeError("wrong fidelity-residual predecessor summary")
    if predecessor["decision"]["screen_passed"]:
        raise RuntimeError("this experiment is preregistered for the failed residual screen")
    if predecessor.get("selected_candidate") is not None:
        raise RuntimeError("the failed predecessor unexpectedly selected a candidate")
    if predecessor.get("final_replication") is not None:
        raise RuntimeError("the locked final slice was already opened")

    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256),
        token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
        dtype=dtype, settings=cfg.model, token=token,
    )
    load_report = load_official_checkpoint(
        model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
    )
    if load_report.checkpoint_sha256 != predecessor["checkpoint_sha256"]:
        raise RuntimeError("loaded checkpoint does not match predecessor lineage")
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    from datasets import load_dataset

    train = load_dataset(
        "json", data_files={"train": GSM8K_TRAIN_URL}, split="train",
        verification_mode="no_checks",
    )
    if len(train) != GSM8K_EXPECTED_TRAIN_EXAMPLES:
        raise RuntimeError("GSM8K train count drifted")
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    calibration_rows, screen_rows, sampling = _fresh_train_rows(
        train, test, predecessor,
        calibration_examples=FRESH_CALIBRATION_EXAMPLES,
        screen_examples=FRESH_SCREEN_EXAMPLES,
    )
    spec = data_cfg.eval.gsm8k
    raw_test = load_dataset(
        str(spec.hf_id), data_files={str(spec.get("split", "test")): str(spec.data_file)},
        split=str(spec.get("split", "test")), verification_mode="no_checks",
    )
    final_rows = align_official_codi_gsm8k_eval_rows(
        raw_test, test, start=FINAL_START, examples=len(test) - FINAL_START,
        enforce_answer_eligibility=False,
    )
    final_hash = _sha([_normalized_question(row["question"]) for row in final_rows])
    if final_hash != predecessor["split_hashes"]["final"]:
        raise RuntimeError("could not reconstruct the untouched final slice")

    latent_positions = int(cfg.eval.latent_iterations)
    reconstruction, fisher, connectivity, token_counts = _collect_utility_curves(
        model, tokenizer, calibration_rows,
        latent_positions=latent_positions,
        maximum_rank=MAXIMUM_COMPONENT_RANK,
        batch_size=args.gradient_batch_size,
        device=device,
    )
    style = PromptStyle.from_config(data_cfg.prompt)
    dense_teacher, dense_losses, dense_logits, _ = _dense_teacher(
        model, tokenizer, screen_rows, latent_positions, args.batch_size, device
    )
    dense_generation, dense_correct, dense_outputs = _dense_generation(
        model, tokenizer, screen_rows, style=style,
        latent_positions=latent_positions, batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens, device=device,
    )
    screen = {
        "dense": {"teacher_forced": dense_teacher, "generation": dense_generation},
        "ordinary_by_rank": {}, "candidates": [],
    }
    tensors = {
        "reconstruction_utilities": reconstruction,
        "answer_fisher_utilities": fisher,
        "calibration_gradient_connectivity": connectivity,
        "screen_deltas": {},
    }
    outputs = {"screen": {"dense": dense_outputs}}
    predictions = []
    ordinary_cache = {}
    for rank in BASELINE_RANKS:
        teacher, delta = _teacher_arm(
            model, tokenizer, screen_rows, latent_positions, args.batch_size, device,
            dense_losses, dense_logits,
            _ordinary_factorizer(rank=rank, latent_positions=latent_positions),
        )
        ordinary_cache[rank] = {"teacher": teacher, "delta": delta}
        tensors["screen_deltas"][f"ordinary_r{rank}"] = delta
        screen["ordinary_by_rank"][str(rank)] = {
            "teacher_forced": teacher, "generation": None,
        }

    for rank in BASELINE_RANKS:
        for weight_index, answer_weight in enumerate(ANSWER_WEIGHTS):
            teacher, delta = _teacher_arm(
                model, tokenizer, screen_rows, latent_positions, args.batch_size, device,
                dense_losses, dense_logits,
                _adaptive_factorizer(
                    rank=rank, answer_weight=answer_weight,
                    latent_positions=latent_positions,
                    reconstruction_utilities=reconstruction,
                    fisher_utilities=fisher,
                ),
            )
            record = _teacher_record(
                rank=rank, answer_weight=answer_weight,
                ordinary_teacher=ordinary_cache[rank]["teacher"],
                ordinary_delta=ordinary_cache[rank]["delta"],
                candidate_teacher=teacher, candidate_delta=delta,
                seed=args.seed + rank * 100 + weight_index,
                samples=args.bootstrap_samples,
                storage_tolerance=args.storage_tolerance,
                minimum_fidelity=args.minimum_first_token_fidelity,
            )
            tensors["screen_deltas"][record["name"]] = delta
            screen["candidates"].append(record)
            print("teacher screen", record["name"], record["teacher_checks"])

    eligible_ranks = {
        record["rank"] for record in screen["candidates"] if record["teacher_preeligible"]
    }
    ordinary_generation_cache = {}
    for rank in sorted(eligible_ranks):
        generation, correct, arm_outputs = _generation_arm(
            model, tokenizer, screen_rows, style=style,
            latent_positions=latent_positions, batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens, device=device,
            factorizer=_ordinary_factorizer(rank=rank, latent_positions=latent_positions),
            dense_outputs=dense_outputs, dense_correct=dense_correct,
        )
        ordinary_generation_cache[rank] = (generation, correct, arm_outputs)
        screen["ordinary_by_rank"][str(rank)]["generation"] = generation
        outputs["screen"][f"ordinary_r{rank}"] = arm_outputs

    for index, record in enumerate(screen["candidates"]):
        if not record["teacher_preeligible"]:
            continue
        generation, correct, arm_outputs = _generation_arm(
            model, tokenizer, screen_rows, style=style,
            latent_positions=latent_positions, batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens, device=device,
            factorizer=_adaptive_factorizer(
                rank=record["rank"], answer_weight=record["answer_weight"],
                latent_positions=latent_positions,
                reconstruction_utilities=reconstruction, fisher_utilities=fisher,
            ),
            dense_outputs=dense_outputs, dense_correct=dense_correct,
        )
        _, ordinary_correct, _ = ordinary_generation_cache[record["rank"]]
        _complete_generation_screen(
            record,
            candidate_generation=generation, candidate_correct=correct,
            dense_correct=dense_correct, ordinary_correct=ordinary_correct,
            seed=args.seed + 20_000 + index * 10, samples=args.bootstrap_samples,
            storage_tolerance=args.storage_tolerance,
            minimum_fidelity=args.minimum_first_token_fidelity,
            noninferiority_margin=args.accuracy_noninferiority_margin,
        )
        outputs["screen"][record["name"]] = arm_outputs
        print("generation screen", record["name"], record["screen_gate"])

    selected = select_adaptive_candidate(screen["candidates"])
    selected_config = (
        {key: selected[key] for key in (
            "name", "rank", "answer_weight", "modelled_compression_ratio",
            "first_token_fidelity", "mean_first_token_kl_from_dense",
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
            "previous_summary_sha256": _file_sha256(args.previous_summary),
            "previous_contract": PREDECESSOR_CONTRACT,
        },
        "preregistration": {
            "baseline_ranks": list(BASELINE_RANKS),
            "answer_weights": list(ANSWER_WEIGHTS),
            "maximum_component_rank": MAXIMUM_COMPONENT_RANK,
            "fresh_calibration_examples": FRESH_CALIBRATION_EXAMPLES,
            "fresh_screen_examples": FRESH_SCREEN_EXAMPLES,
            "final_slice": [FINAL_START, len(test)],
            "selection_order": [
                "maximum modeled compression ratio",
                "maximum NLL lower confidence bound",
                "maximum first-token fidelity",
                "hybrid utility preferred only as a final tie break",
            ],
            "bootstrap_samples": args.bootstrap_samples,
            "minimum_first_token_fidelity": args.minimum_first_token_fidelity,
            "accuracy_noninferiority_margin": args.accuracy_noninferiority_margin,
            "storage_tolerance": args.storage_tolerance,
        },
        "sampling": sampling,
        "split_hashes": {
            "fresh_calibration": sampling["fresh_calibration_questions_sha256"],
            "fresh_screen": sampling["fresh_screen_questions_sha256"],
            "final": final_hash,
        },
        "allocation_fit": {
            "objective": "blend of per-mode cache energy and squared first-order answer-NLL effect",
            "gradient_connectivity_fraction": float(connectivity.float().mean()),
            "observed_token_count": {
                "minimum": min(token_counts),
                "median": statistics.median(token_counts),
                "maximum": max(token_counts),
            },
            "component_audit": _curve_audit(reconstruction, fisher),
            "storage_contract": (
                "independent K/V factors plus explicit padding equal the ordinary "
                "per-layer grouped-xKV factor bits on every request"
            ),
        },
        "fresh_screen": screen,
        "selected_candidate": selected_config,
        "final_replication": None,
        "decision": {
            "screen_passed": selected is not None,
            "final_passed": False,
            "claim": (
                "ADVANCE: one frozen adaptive K/V allocation selected"
                if selected else
                "STOP: no adaptive allocation beat uniform xKV at matched storage"
            ),
        },
        "warnings": [
            "Allocation utilities are fitted on fresh GSM8K-train questions disjoint from all prior samples and the test set.",
            "The 551-question final test slice is opened exactly once and only after the fresh screen passes.",
            "Budget padding is counted as cache storage, so adaptive and ordinary arms have identical modeled bits.",
            "This is a dense-reconstruction and modeled-storage proxy, not a native xKV latency benchmark.",
        ],
    }
    if selected is None:
        _save(args, summary, tensors, outputs, predictions)
        print(json.dumps(summary, indent=2))
        return summary

    rank = int(selected["rank"])
    answer_weight = float(selected["answer_weight"])
    final_dense_teacher, final_dense_losses, final_dense_logits, _ = _dense_teacher(
        model, tokenizer, final_rows, latent_positions, args.batch_size, device
    )
    final_ordinary_teacher, final_ordinary_delta = _teacher_arm(
        model, tokenizer, final_rows, latent_positions, args.batch_size, device,
        final_dense_losses, final_dense_logits,
        _ordinary_factorizer(rank=rank, latent_positions=latent_positions),
    )
    final_candidate_teacher, final_candidate_delta = _teacher_arm(
        model, tokenizer, final_rows, latent_positions, args.batch_size, device,
        final_dense_losses, final_dense_logits,
        _adaptive_factorizer(
            rank=rank, answer_weight=answer_weight, latent_positions=latent_positions,
            reconstruction_utilities=reconstruction, fisher_utilities=fisher,
        ),
    )
    final_dense_generation, final_dense_correct, final_dense_outputs = _dense_generation(
        model, tokenizer, final_rows, style=style,
        latent_positions=latent_positions, batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens, device=device,
    )
    final_ordinary_generation, final_ordinary_correct, final_ordinary_outputs = _generation_arm(
        model, tokenizer, final_rows, style=style,
        latent_positions=latent_positions, batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens, device=device,
        factorizer=_ordinary_factorizer(rank=rank, latent_positions=latent_positions),
        dense_outputs=final_dense_outputs, dense_correct=final_dense_correct,
    )
    final_candidate_generation, final_candidate_correct, final_candidate_outputs = _generation_arm(
        model, tokenizer, final_rows, style=style,
        latent_positions=latent_positions, batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens, device=device,
        factorizer=_adaptive_factorizer(
            rank=rank, answer_weight=answer_weight, latent_positions=latent_positions,
            reconstruction_utilities=reconstruction, fisher_utilities=fisher,
        ),
        dense_outputs=final_dense_outputs, dense_correct=final_dense_correct,
    )
    final_advantage = final_ordinary_delta - final_candidate_delta
    final_nll_interval = paired_bootstrap_interval(
        final_advantage, seed=args.seed + 50_000, samples=args.bootstrap_samples
    )
    final_dense_interval = paired_bootstrap_interval(
        final_candidate_correct - final_dense_correct,
        seed=args.seed + 50_001, samples=args.bootstrap_samples,
    )
    final_ordinary_interval = paired_bootstrap_interval(
        final_candidate_correct - final_ordinary_correct,
        seed=args.seed + 50_002, samples=args.bootstrap_samples,
    )
    final_bits_ratio = final_candidate_teacher["cache"]["cache_bits"] / final_ordinary_teacher["cache"]["cache_bits"]
    from src.mech.xkv_fidelity_frontier import fidelity_gate
    final_gate = fidelity_gate(
        nll_interval=final_nll_interval,
        cache_bits_ratio=final_bits_ratio,
        first_token_fidelity=final_candidate_teacher["dense_first_token_top1_agreement"],
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
        "adaptive_to_ordinary_cache_bits_ratio": final_bits_ratio,
        "accuracy_difference_vs_dense_95ci": final_dense_interval,
        "accuracy_difference_vs_ordinary_95ci": final_ordinary_interval,
        "gate": final_gate,
    }
    tensors["final_deltas"] = {
        "ordinary_xkv": final_ordinary_delta, "selected": final_candidate_delta,
    }
    outputs["final"] = {
        "dense": final_dense_outputs, "ordinary_xkv": final_ordinary_outputs,
        "selected": final_candidate_outputs,
    }
    for index, row in enumerate(final_rows):
        predictions.append({
            "split": "final", "dataset_index": FINAL_START + index,
            "question": row["question"], "gold": str(row["gold"]),
            "dense": final_dense_outputs[index],
            "ordinary_xkv": final_ordinary_outputs[index],
            "selected": final_candidate_outputs[index],
        })
    summary["decision"] = {
        "screen_passed": True,
        "final_passed": bool(final_gate["passed"]),
        "claim": (
            "CONFIRMED: adaptive K/V allocation improves ordinary xKV at matched storage"
            if final_gate["passed"] else
            "STOP: the adaptive allocation did not replicate on the locked final slice"
        ),
    }
    _save(args, summary, tensors, outputs, predictions)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--baseline-ranks", default=",".join(map(str, BASELINE_RANKS)))
    parser.add_argument("--answer-weights", default=",".join(map(str, ANSWER_WEIGHTS)))
    parser.add_argument("--maximum-component-rank", type=int, default=MAXIMUM_COMPONENT_RANK)
    parser.add_argument("--fresh-calibration-examples", type=int, default=FRESH_CALIBRATION_EXAMPLES)
    parser.add_argument("--fresh-screen-examples", type=int, default=FRESH_SCREEN_EXAMPLES)
    parser.add_argument("--final-start", type=int, default=FINAL_START)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--minimum-first-token-fidelity", type=float, default=0.95)
    parser.add_argument("--storage-tolerance", type=float, default=0.001)
    parser.add_argument("--accuracy-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--gradient-batch-size", type=int, default=2)
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
