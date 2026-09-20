"""Test a tiny dense-top-1 fidelity residual on top of ordinary per-layer xKV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES,
    GSM8K_TRAIN_URL,
    sample_gsm8k_train_calibration,
)
from scripts.run_codi_confirm_and_compress_native_kv import SOURCE_CONTRACT
from scripts.run_codi_preanswer_kv_subspace_discovery import _sha
from scripts.run_codi_task_aware_protected_xkv import (
    CONTRACT as PREDECESSOR_CONTRACT,
    FinalLatentTaskAwareFactorizer,
)
from scripts.run_codi_xkv_fidelity_frontier import (
    CONTRACT as FRONTIER_CONTRACT,
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
from src.mech.fidelity_residual_xkv import (
    fit_margin_gradient_bases,
    select_residual_candidate,
    truncate_residual_bases,
)
from src.mech.preanswer_kv_subspace import official_codi_preanswer_kv_forward
from src.mech.rank16_xkv_confirmation import paired_bootstrap_interval
from src.mech.xkv_fidelity_frontier import fidelity_gate
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_fidelity_residual_xkv_holdout_v1"
FRESH_FIT_EXAMPLES = 512
FRESH_SCREEN_EXAMPLES = 512
FINAL_START = 768
RANKS = (48, 64, 80, 96)
RESIDUAL_RANKS = (1, 2, 4)
PER_LAYER_GROUPS = tuple((layer,) for layer in range(12))


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _with_gold(rows):
    return [{**row, "gold": str(row.get("gold", row["answer"]))} for row in rows]


def _fresh_train_rows(train, test, source, predecessor, *, fit_examples, screen_examples):
    used_examples = int(predecessor["sampling"]["selected_examples"])
    seed = int(source["sampling"]["sampling_seed"])
    test_questions = {_normalized_question(row["question"]) for row in test}
    used_rows, used_audit = sample_gsm8k_train_calibration(
        train, test_questions=test_questions, examples=used_examples, seed=seed
    )
    for key in (
        "selected_source_indices_sha256",
        "selected_normalized_questions_sha256",
    ):
        if used_audit[key] != predecessor["sampling"][key]:
            raise RuntimeError("could not reconstruct the predecessor train prefix")
    total = used_examples + int(fit_examples) + int(screen_examples)
    rows, sampling = sample_gsm8k_train_calibration(
        train, test_questions=test_questions, examples=total, seed=seed
    )
    if [row["question"] for row in rows[:used_examples]] != [
        row["question"] for row in used_rows
    ]:
        raise RuntimeError("fresh sampling does not preserve the predecessor prefix")
    fit_stop = used_examples + int(fit_examples)
    fit_rows = _with_gold(rows[used_examples:fit_stop])
    screen_rows = _with_gold(rows[fit_stop:])
    sampling.update({
        "predecessor_examples_reconstructed": used_examples,
        "predecessor_questions_sha256": _sha(
            [_normalized_question(row["question"]) for row in used_rows]
        ),
        "fresh_fit_questions_sha256": _sha(
            [_normalized_question(row["question"]) for row in fit_rows]
        ),
        "fresh_screen_questions_sha256": _sha(
            [_normalized_question(row["question"]) for row in screen_rows]
        ),
    })
    return fit_rows, screen_rows, sampling


def _collect_margin_gradients(
    model, tokenizer, rows, *, latent_positions, batch_size, device
):
    key_chunks, value_chunks, connectivity = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start : start + batch_size], bot_token_id=model.bot_id
        ).to(device)
        output = official_codi_preanswer_kv_forward(
            model,
            batch,
            latent_positions=latent_positions,
            return_gradients=True,
            gradient_objective="first_token_margin",
        )
        key_chunks.append(output.key_gradients.detach().float().cpu())
        value_chunks.append(output.value_gradients.detach().float().cpu())
        connectivity.append(output.gradient_connected)
        del batch, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    connected = torch.stack(connectivity)
    if not bool(connected.all()):
        raise RuntimeError("top-1 margin gradients are not connected to every K/V cache")
    return torch.cat(key_chunks), torch.cat(value_chunks), connected


def _ordinary_factorizer(*, rank: int, latent_positions: int):
    return FinalLatentTaskAwareFactorizer(
        latent_positions=latent_positions,
        groups=PER_LAYER_GROUPS,
        rank=rank,
    )


def _residual_factorizer(
    *, rank: int, residual_rank: int, latent_positions: int, bases
):
    return FinalLatentTaskAwareFactorizer(
        latent_positions=latent_positions,
        groups=PER_LAYER_GROUPS,
        rank=rank,
        protected_bases=truncate_residual_bases(bases, residual_rank),
    )


def _candidate_name(rank: int, residual_rank: int) -> str:
    return f"margin_residual_xkv_r{rank}_res{residual_rank}"


def _teacher_record(
    *, rank, residual_rank, ordinary_teacher, ordinary_delta,
    candidate_teacher, candidate_delta, seed, samples, storage_tolerance,
    minimum_fidelity,
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
        "name": _candidate_name(rank, residual_rank),
        "rank": int(rank),
        "residual_rank": int(residual_rank),
        "mean_nll_advantage": float(advantage.mean()),
        "nll_bootstrap_95ci": interval,
        "full_to_ordinary_cache_bits_ratio": bits_ratio,
        "modelled_compression_ratio": candidate_teacher["cache"][
            "modelled_compression_ratio"
        ],
        "first_token_fidelity": candidate_teacher[
            "dense_first_token_top1_agreement"
        ],
        "mean_first_token_kl_from_dense": candidate_teacher[
            "mean_first_token_kl_from_dense"
        ],
        "teacher_checks": teacher_checks,
        "teacher_preeligible": all(teacher_checks.values()),
        "teacher_forced": candidate_teacher,
        "generation": None,
        "screen_gate": None,
        "screen_passed": False,
    }


def _save(args, summary, tensors, outputs, predictions):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save(
        {**summary, **tensors, "outputs": outputs},
        args.output_dir / "fidelity_residual_xkv.pt",
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    if _parse_ints(args.ranks) != RANKS or _parse_ints(args.residual_ranks) != RESIDUAL_RANKS:
        raise ValueError("the preregistered rank grids cannot be changed")
    if (
        args.fresh_fit_examples != FRESH_FIT_EXAMPLES
        or args.fresh_screen_examples != FRESH_SCREEN_EXAMPLES
        or args.final_start != FINAL_START
    ):
        raise ValueError("the preregistered split sizes cannot be changed")

    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    predecessor = torch.load(
        args.previous_experiment_artifact, map_location="cpu", weights_only=False
    )
    frontier = json.loads(args.previous_frontier_summary.read_text())
    if source.get("contract") != SOURCE_CONTRACT:
        raise RuntimeError("wrong direct-cache source artifact")
    if predecessor.get("contract") != PREDECESSOR_CONTRACT:
        raise RuntimeError("wrong task-aware xKV predecessor artifact")
    if frontier.get("contract") != FRONTIER_CONTRACT:
        raise RuntimeError("wrong fidelity-frontier predecessor summary")
    if frontier["decision"]["development_passed"]:
        raise RuntimeError("this experiment is preregistered for the failed frontier")
    if frontier.get("selected_candidate") is not None:
        raise RuntimeError("the failed frontier unexpectedly selected a candidate")
    if frontier.get("final_replication") is not None:
        raise RuntimeError("the locked final slice was already opened")
    if predecessor["checkpoint_sha256"] != source["checkpoint_sha256"]:
        raise RuntimeError("source and predecessor checkpoints differ")

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

    train = load_dataset(
        "json",
        data_files={"train": GSM8K_TRAIN_URL},
        split="train",
        verification_mode="no_checks",
    )
    if len(train) != GSM8K_EXPECTED_TRAIN_EXAMPLES:
        raise RuntimeError("GSM8K train count drifted")
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    fit_rows, screen_rows, sampling = _fresh_train_rows(
        train,
        test,
        source,
        predecessor,
        fit_examples=FRESH_FIT_EXAMPLES,
        screen_examples=FRESH_SCREEN_EXAMPLES,
    )
    spec = data_cfg.eval.gsm8k
    raw_test = load_dataset(
        str(spec.hf_id),
        data_files={str(spec.get("split", "test")): str(spec.data_file)},
        split=str(spec.get("split", "test")),
        verification_mode="no_checks",
    )
    final_rows = align_official_codi_gsm8k_eval_rows(
        raw_test,
        test,
        start=FINAL_START,
        examples=len(test) - FINAL_START,
        enforce_answer_eligibility=False,
    )
    final_hash = _sha([_normalized_question(row["question"]) for row in final_rows])
    if final_hash != frontier["split_hashes"]["final"]:
        raise RuntimeError("could not reconstruct the untouched final slice")

    latent_positions = int(cfg.eval.latent_iterations)
    key_gradients, value_gradients, connectivity = _collect_margin_gradients(
        model,
        tokenizer,
        fit_rows,
        latent_positions=latent_positions,
        batch_size=args.gradient_batch_size,
        device=device,
    )
    fidelity_bases, basis_audit = fit_margin_gradient_bases(
        key_gradients,
        value_gradients,
        maximum_rank=max(RESIDUAL_RANKS),
        seed=args.seed,
    )

    style = PromptStyle.from_config(data_cfg.prompt)
    dense_teacher, dense_losses, dense_logits, _ = _dense_teacher(
        model,
        tokenizer,
        screen_rows,
        latent_positions,
        args.batch_size,
        device,
    )
    dense_generation, dense_correct, dense_outputs = _dense_generation(
        model,
        tokenizer,
        screen_rows,
        style=style,
        latent_positions=latent_positions,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    screen = {
        "dense": {"teacher_forced": dense_teacher, "generation": dense_generation},
        "ordinary_by_rank": {},
        "candidates": [],
    }
    tensors = {
        "fidelity_bases": fidelity_bases,
        "fit_margin_gradient_connectivity": connectivity,
        "screen_deltas": {},
    }
    outputs = {"screen": {"dense": dense_outputs}}
    predictions = []
    ordinary_cache = {}
    for rank in RANKS:
        teacher, delta = _teacher_arm(
            model,
            tokenizer,
            screen_rows,
            latent_positions,
            args.batch_size,
            device,
            dense_losses,
            dense_logits,
            _ordinary_factorizer(rank=rank, latent_positions=latent_positions),
        )
        ordinary_cache[rank] = {"teacher": teacher, "delta": delta}
        tensors["screen_deltas"][f"ordinary_r{rank}"] = delta
        screen["ordinary_by_rank"][str(rank)] = {
            "teacher_forced": teacher,
            "generation": None,
        }

    for rank in RANKS:
        for residual_rank in RESIDUAL_RANKS:
            teacher, delta = _teacher_arm(
                model,
                tokenizer,
                screen_rows,
                latent_positions,
                args.batch_size,
                device,
                dense_losses,
                dense_logits,
                _residual_factorizer(
                    rank=rank,
                    residual_rank=residual_rank,
                    latent_positions=latent_positions,
                    bases=fidelity_bases,
                ),
            )
            record = _teacher_record(
                rank=rank,
                residual_rank=residual_rank,
                ordinary_teacher=ordinary_cache[rank]["teacher"],
                ordinary_delta=ordinary_cache[rank]["delta"],
                candidate_teacher=teacher,
                candidate_delta=delta,
                seed=args.seed + rank * 100 + residual_rank,
                samples=args.bootstrap_samples,
                storage_tolerance=args.storage_tolerance,
                minimum_fidelity=args.minimum_first_token_fidelity,
            )
            tensors["screen_deltas"][record["name"]] = delta
            screen["candidates"].append(record)
            print("teacher screen", record["name"], record["teacher_checks"])

    eligible_ranks = {
        record["rank"] for record in screen["candidates"]
        if record["teacher_preeligible"]
    }
    ordinary_generation_cache = {}
    for rank in sorted(eligible_ranks):
        generation, correct, arm_outputs = _generation_arm(
            model,
            tokenizer,
            screen_rows,
            style=style,
            latent_positions=latent_positions,
            batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            factorizer=_ordinary_factorizer(rank=rank, latent_positions=latent_positions),
            dense_outputs=dense_outputs,
            dense_correct=dense_correct,
        )
        ordinary_generation_cache[rank] = (generation, correct, arm_outputs)
        screen["ordinary_by_rank"][str(rank)]["generation"] = generation
        outputs["screen"][f"ordinary_r{rank}"] = arm_outputs

    for index, record in enumerate(screen["candidates"]):
        if not record["teacher_preeligible"]:
            continue
        generation, correct, arm_outputs = _generation_arm(
            model,
            tokenizer,
            screen_rows,
            style=style,
            latent_positions=latent_positions,
            batch_size=args.generation_batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            factorizer=_residual_factorizer(
                rank=record["rank"],
                residual_rank=record["residual_rank"],
                latent_positions=latent_positions,
                bases=fidelity_bases,
            ),
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
            seed=args.seed + 20_000 + index * 10,
            samples=args.bootstrap_samples,
            storage_tolerance=args.storage_tolerance,
            minimum_fidelity=args.minimum_first_token_fidelity,
            noninferiority_margin=args.accuracy_noninferiority_margin,
        )
        outputs["screen"][record["name"]] = arm_outputs
        print("generation screen", record["name"], record["screen_gate"])

    selected = select_residual_candidate(screen["candidates"])
    selected_config = (
        {key: selected[key] for key in (
            "name", "rank", "residual_rank", "modelled_compression_ratio",
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
            "source_artifact_sha256": _file_sha256(args.source_artifact),
            "previous_experiment_artifact_sha256": _file_sha256(
                args.previous_experiment_artifact
            ),
            "previous_frontier_summary_sha256": _file_sha256(
                args.previous_frontier_summary
            ),
            "source_contract": SOURCE_CONTRACT,
            "previous_contract": PREDECESSOR_CONTRACT,
            "frontier_contract": FRONTIER_CONTRACT,
        },
        "preregistration": {
            "ranks": list(RANKS),
            "residual_ranks": list(RESIDUAL_RANKS),
            "groups": [list(group) for group in PER_LAYER_GROUPS],
            "fresh_fit_examples": FRESH_FIT_EXAMPLES,
            "fresh_screen_examples": FRESH_SCREEN_EXAMPLES,
            "final_slice": [FINAL_START, len(test)],
            "selection_order": [
                "maximum modeled compression ratio",
                "maximum first-token fidelity",
                "minimum first-token KL from dense",
                "maximum NLL lower confidence bound",
                "minimum residual rank",
            ],
            "bootstrap_samples": args.bootstrap_samples,
            "minimum_first_token_fidelity": args.minimum_first_token_fidelity,
            "accuracy_noninferiority_margin": args.accuracy_noninferiority_margin,
            "storage_tolerance": args.storage_tolerance,
        },
        "sampling": sampling,
        "split_hashes": {
            "fresh_fit": sampling["fresh_fit_questions_sha256"],
            "fresh_screen": sampling["fresh_screen_questions_sha256"],
            "final": final_hash,
        },
        "basis_fit": {
            "objective": "negative dense top-1-versus-runner-up logit margin",
            "uses_gold_labels": False,
            "gradient_connectivity_fraction": float(connectivity.float().mean()),
            "audit": basis_audit,
        },
        "fresh_screen": screen,
        "selected_candidate": selected_config,
        "final_replication": None,
        "decision": {
            "screen_passed": selected is not None,
            "final_passed": False,
            "claim": (
                "ADVANCE: one frozen fidelity-residual candidate selected"
                if selected else
                "STOP: no fresh-screen candidate passed the complete fidelity gate"
            ),
        },
        "warnings": [
            "Residual bases are fitted on fresh GSM8K-train questions disjoint from all prior train samples and the test set.",
            "The 551-question final test slice is opened exactly once and only after the fresh screen passes.",
            "Static residual bases are model metadata; per-request residual coordinates are counted in modeled cache storage.",
            "This is a dense-reconstruction and modeled-storage proxy, not a native xKV latency benchmark.",
        ],
    }
    if selected is None:
        _save(args, summary, tensors, outputs, predictions)
        print(json.dumps(summary, indent=2))
        return summary

    rank = int(selected["rank"])
    residual_rank = int(selected["residual_rank"])
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
        _residual_factorizer(
            rank=rank,
            residual_rank=residual_rank,
            latent_positions=latent_positions,
            bases=fidelity_bases,
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
        factorizer=_residual_factorizer(
            rank=rank,
            residual_rank=residual_rank,
            latent_positions=latent_positions,
            bases=fidelity_bases,
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
        "screen_passed": True,
        "final_passed": bool(final_gate["passed"]),
        "claim": (
            "CONFIRMED: a tiny margin-sensitive residual restores faithful xKV at matched storage"
            if final_gate["passed"] else
            "STOP: the fresh-screen residual did not replicate on the locked final slice"
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
    parser.add_argument("--previous-frontier-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--ranks", default=",".join(map(str, RANKS)))
    parser.add_argument("--residual-ranks", default=",".join(map(str, RESIDUAL_RANKS)))
    parser.add_argument("--fresh-fit-examples", type=int, default=FRESH_FIT_EXAMPLES)
    parser.add_argument("--fresh-screen-examples", type=int, default=FRESH_SCREEN_EXAMPLES)
    parser.add_argument("--final-start", type=int, default=FINAL_START)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--minimum-first-token-fidelity", type=float, default=0.95)
    parser.add_argument("--storage-tolerance", type=float, default=0.001)
    parser.add_argument("--accuracy-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--gradient-batch-size", type=int, default=4)
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
