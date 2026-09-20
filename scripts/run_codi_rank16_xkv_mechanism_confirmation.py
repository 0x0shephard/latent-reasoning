"""Confirm the discovered rank-16 xKV signal and isolate its mechanism."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from scripts.run_codi_task_aware_protected_xkv import (
    CONTRACT as PREVIOUS_CONTRACT,
    FinalLatentTaskAwareFactorizer,
    _energy_matched_random_basis_maps,
    _teacher_forced_arm,
    _topk_overlap,
)
from src.data.datasets import load_eval_set
from src.data.official_codi_training import align_official_codi_gsm8k_eval_rows
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.rank16_xkv_confirmation import (
    holm_bonferroni,
    paired_bootstrap_interval,
    paired_sign_flip_pvalue,
    primary_gate,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_rank16_xkv_mechanism_holdout_v1"
PREREGISTERED_GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
CONFIRMATION_START = 256
CONFIRMATION_EXAMPLES = 512
FINAL_START = 768
RANK = 16


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serial_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if not key.startswith("per_")}


def _arm(
    name: str,
    *,
    latent_positions: int,
    feature_weights,
    utilities,
    protected_bases,
):
    kwargs = {
        "latent_positions": latent_positions,
        "groups": PREREGISTERED_GROUPS,
        "rank": RANK,
    }
    if name == "ordinary_xkv":
        pass
    elif name == "allocation_only":
        kwargs.update(group_utilities=utilities, adaptive_ranks=True)
    elif name == "fisher_only":
        kwargs.update(feature_weights=feature_weights)
    elif name == "allocation_fisher":
        kwargs.update(
            feature_weights=feature_weights,
            group_utilities=utilities,
            adaptive_ranks=True,
        )
    elif name == "protection_only":
        kwargs.update(protected_bases=protected_bases)
    elif name == "full_method":
        kwargs.update(
            feature_weights=feature_weights,
            group_utilities=utilities,
            adaptive_ranks=True,
            protected_bases=protected_bases,
        )
    elif name == "per_layer_svd":
        kwargs["groups"] = tuple((layer,) for layer in range(12))
    else:
        raise ValueError(f"unknown confirmation arm {name}")
    return FinalLatentTaskAwareFactorizer(**kwargs)


def _teacher_split(
    model,
    tokenizer,
    rows,
    *,
    latent_positions,
    batch_size,
    device,
    arm_names,
    feature_weights,
    utilities,
    protected_bases,
):
    dense_losses, dense_logits, dense_targets = _evaluate(
        model,
        tokenizer,
        rows,
        latent_positions,
        batch_size,
        device,
        enforce_answer_eligibility=False,
    )
    dense = _serial_metrics(
        _metrics(dense_losses, dense_logits, dense_targets, dense_losses, dense_logits)
    )
    dense["perplexity_from_mean_answer_nll"] = math.exp(
        min(dense["mean_full_answer_nll"], 50)
    )
    dense["first_token_top5_overlap"] = 1.0
    results = {"dense": dense}
    deltas = {"dense": torch.zeros_like(dense_losses)}
    for name in arm_names:
        factorizer = _arm(
            name,
            latent_positions=latent_positions,
            feature_weights=feature_weights,
            utilities=utilities,
            protected_bases=protected_bases,
        )
        metrics, delta = _teacher_forced_arm(
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
        results[name] = metrics
        deltas[name] = delta
        print("teacher", name, metrics)
    return results, deltas, {
        "losses": dense_losses,
        "logits": dense_logits,
        "targets": dense_targets,
    }


def _correctness(outputs, rows) -> torch.Tensor:
    return torch.tensor(
        [official_answers_match(text, row["gold"]) for text, row in zip(outputs, rows)],
        dtype=torch.float32,
    )


def _generation_split(
    model,
    tokenizer,
    rows,
    *,
    style,
    latent_positions,
    batch_size,
    max_new_tokens,
    device,
    arm_names,
    feature_weights,
    utilities,
    protected_bases,
):
    questions = [row["question"] for row in rows]
    dense_outputs = generate_official_codi(
        model,
        tokenizer,
        questions,
        latent_iterations=latent_positions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
        answer_cue=style.answer_prefix,
        force_answer_cue=True,
    )
    dense_correct = _correctness(dense_outputs, rows)
    results = {
        "dense": {
            "accuracy": float(dense_correct.mean()),
            "correct": int(dense_correct.sum()),
            "examples": len(rows),
            "exact_sequence_agreement": 1.0,
        }
    }
    correctness = {"dense": dense_correct}
    outputs = {"dense": dense_outputs}
    for name in arm_names:
        factorizer = _arm(
            name,
            latent_positions=latent_positions,
            feature_weights=feature_weights,
            utilities=utilities,
            protected_bases=protected_bases,
        )
        values = generate_official_codi(
            model,
            tokenizer,
            questions,
            latent_iterations=latent_positions,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
            kv_intervention=factorizer,
            answer_cue=style.answer_prefix,
            force_answer_cue=True,
        )
        correct = _correctness(values, rows)
        results[name] = {
            "accuracy": float(correct.mean()),
            "correct": int(correct.sum()),
            "examples": len(rows),
            "accuracy_retained_fraction": (
                float(correct.mean() / dense_correct.mean())
                if float(dense_correct.mean())
                else None
            ),
            "exact_sequence_agreement": sum(
                current == dense for current, dense in zip(values, dense_outputs)
            )
            / len(values),
            "cache": factorizer.summary(),
        }
        correctness[name] = correct
        outputs[name] = values
        print("generation", name, results[name])
    return results, correctness, outputs


def _primary_comparison(
    teacher,
    deltas,
    generation,
    correctness,
    *,
    seed,
    bootstrap_samples,
    storage_tolerance,
    minimum_retention,
    noninferiority_margin,
):
    nll_advantage = deltas["ordinary_xkv"] - deltas["full_method"]
    accuracy_advantage = correctness["full_method"] - correctness["ordinary_xkv"]
    nll_interval = paired_bootstrap_interval(
        nll_advantage, seed=seed, samples=bootstrap_samples
    )
    accuracy_interval = paired_bootstrap_interval(
        accuracy_advantage, seed=seed + 1, samples=bootstrap_samples
    )
    ordinary_bits = teacher["ordinary_xkv"]["cache"]["cache_bits"]
    full_bits = teacher["full_method"]["cache"]["cache_bits"]
    bits_ratio = full_bits / ordinary_bits
    gate = primary_gate(
        nll_interval=nll_interval,
        cache_bits_ratio=bits_ratio,
        retention=teacher["full_method"]["dense_first_token_top1_agreement"],
        accuracy_interval=accuracy_interval,
        storage_tolerance=storage_tolerance,
        minimum_retention=minimum_retention,
        accuracy_noninferiority_margin=noninferiority_margin,
    )
    return {
        "mean_nll_advantage": float(nll_advantage.mean()),
        "nll_bootstrap_95ci": nll_interval,
        "mean_accuracy_advantage": float(accuracy_advantage.mean()),
        "accuracy_bootstrap_95ci": accuracy_interval,
        "full_to_ordinary_cache_bits_ratio": bits_ratio,
        "ordinary_accuracy": generation["ordinary_xkv"]["accuracy"],
        "full_accuracy": generation["full_method"]["accuracy"],
        "gate": gate,
    }


def _effect_record(values, *, seed, bootstrap_samples, signflip_samples):
    return {
        "mean": float(values.mean()),
        "bootstrap_95ci": paired_bootstrap_interval(
            values, seed=seed, samples=bootstrap_samples
        ),
        "one_sided_sign_flip_pvalue": paired_sign_flip_pvalue(
            values, seed=seed + 1, samples=signflip_samples
        ),
    }


def _save(
    args,
    summary,
    *,
    tensors,
    outputs,
    predictions,
):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save(
        {**summary, **tensors, "outputs": outputs},
        args.output_dir / "rank16_xkv_confirmation.pt",
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args):
    if (
        args.confirmation_start != CONFIRMATION_START
        or args.confirmation_examples != CONFIRMATION_EXAMPLES
        or args.final_start != FINAL_START
        or args.rank != RANK
    ):
        raise ValueError("the preregistered split boundaries and rank cannot be changed")
    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    previous = torch.load(
        args.previous_experiment_artifact, map_location="cpu", weights_only=False
    )
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    if previous.get("contract") != PREVIOUS_CONTRACT:
        raise RuntimeError("wrong predecessor experiment artifact")
    if source.get("contract") != SOURCE_CONTRACT:
        raise RuntimeError("wrong direct-cache source artifact")
    if previous["confirmation"]["strict_layers_2_3_11_gate_passed"]:
        raise RuntimeError("this experiment is preregistered for the failed early-layer gate")
    if previous["confirmation"]["included_layers"] != [11]:
        raise RuntimeError("the frozen predecessor must protect only layer 11")
    prior_rank16 = previous["paired_ordinary_xkv_comparisons"]["rank_16"]
    if not prior_rank16["strict_quality_win"]:
        raise RuntimeError("the predecessor does not contain the rank-16 discovery signal")
    if previous["checkpoint_sha256"] != source["checkpoint_sha256"]:
        raise RuntimeError("predecessor and source checkpoint lineage differ")

    feature_weights = previous["feature_weights"]
    protected_bases = previous["protected_bases"]
    if {int(layer) for layer, _ in protected_bases} != {11}:
        raise RuntimeError("protected basis map is not the frozen layer-11 map")
    for (layer, kind), basis in protected_bases.items():
        source_basis = source[f"selected_{kind}_responses"][layer]
        if basis.shape != source_basis.shape or not torch.allclose(basis, source_basis):
            raise RuntimeError("protected basis differs from the source artifact")
    utilities = {
        tuple(int(value) for value in key.split(",")): float(value)
        for key, value in previous["compression_calibration"]["group_utilities"].items()
    }
    if set(feature_weights) != set(PREREGISTERED_GROUPS) or set(utilities) != set(
        PREREGISTERED_GROUPS
    ):
        raise RuntimeError("predecessor group calibration does not match preregistration")

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
    if load_report.checkpoint_sha256 != previous["checkpoint_sha256"]:
        raise RuntimeError("loaded checkpoint does not match the predecessor")
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
    discovery = align_official_codi_gsm8k_eval_rows(
        raw_test,
        test,
        start=0,
        examples=CONFIRMATION_START,
        enforce_answer_eligibility=False,
    )
    discovery_hash = _sha(
        [_normalized_question(row["question"]) for row in discovery]
    )
    if discovery_hash != previous["evaluation"]["questions_sha256"]:
        raise RuntimeError("could not reconstruct the predecessor discovery slice")
    confirmation_rows = align_official_codi_gsm8k_eval_rows(
        raw_test,
        test,
        start=CONFIRMATION_START,
        examples=CONFIRMATION_EXAMPLES,
        enforce_answer_eligibility=False,
    )
    final_rows = align_official_codi_gsm8k_eval_rows(
        raw_test,
        test,
        start=FINAL_START,
        examples=len(test) - FINAL_START,
        enforce_answer_eligibility=False,
    )
    latent_positions = int(cfg.eval.latent_iterations)
    style = PromptStyle.from_config(data_cfg.prompt)
    core_arms = ("ordinary_xkv", "full_method")
    confirm_teacher, confirm_deltas, confirm_baseline = _teacher_split(
        model,
        tokenizer,
        confirmation_rows,
        latent_positions=latent_positions,
        batch_size=args.batch_size,
        device=device,
        arm_names=core_arms,
        feature_weights=feature_weights,
        utilities=utilities,
        protected_bases=protected_bases,
    )
    confirm_generation, confirm_correct, confirm_outputs = _generation_split(
        model,
        tokenizer,
        confirmation_rows,
        style=style,
        latent_positions=latent_positions,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        arm_names=core_arms,
        feature_weights=feature_weights,
        utilities=utilities,
        protected_bases=protected_bases,
    )
    confirm_primary = _primary_comparison(
        confirm_teacher,
        confirm_deltas,
        confirm_generation,
        confirm_correct,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        storage_tolerance=args.storage_tolerance,
        minimum_retention=args.minimum_retention,
        noninferiority_margin=args.accuracy_noninferiority_margin,
    )
    summary = {
        "schema_version": 1,
        "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction,
        "lineage": {
            "previous_experiment_artifact_sha256": _file_sha256(
                args.previous_experiment_artifact
            ),
            "source_artifact_sha256": _file_sha256(args.source_artifact),
            "previous_contract": PREVIOUS_CONTRACT,
            "source_contract": SOURCE_CONTRACT,
        },
        "preregistration": {
            "rank": RANK,
            "groups": [list(group) for group in PREREGISTERED_GROUPS],
            "discovery_slice": [0, CONFIRMATION_START],
            "confirmation_slice": [
                CONFIRMATION_START,
                CONFIRMATION_START + CONFIRMATION_EXAMPLES,
            ],
            "final_slice": [FINAL_START, len(test)],
            "random_controls": args.random_controls,
            "bootstrap_samples": args.bootstrap_samples,
            "signflip_samples": args.signflip_samples,
            "accuracy_noninferiority_margin": args.accuracy_noninferiority_margin,
        },
        "split_hashes": {
            "discovery": discovery_hash,
            "confirmation": _sha(
                [_normalized_question(row["question"]) for row in confirmation_rows]
            ),
            "final": _sha(
                [_normalized_question(row["question"]) for row in final_rows]
            ),
        },
        "confirmation": {
            "teacher_forced": confirm_teacher,
            "generation": confirm_generation,
            "primary": confirm_primary,
        },
        "mechanism": None,
        "final_replication": None,
        "transfer": None,
        "decision": {
            "confirmation_passed": bool(confirm_primary["gate"]["passed"]),
            "final_passed": False,
            "claim": "STOP: rank-16 discovery did not replicate",
        },
        "warnings": [
            "This is a dense-reconstruction quality and modeled-storage proxy, not a native xKV latency benchmark.",
            "The first 256 GSM8K test rows are discovery-only and are not reused for confirmation.",
        ],
    }
    tensors = {"confirmation_per_example_nll_deltas": confirm_deltas}
    outputs = {"confirmation": confirm_outputs}
    predictions = []
    for index, row in enumerate(confirmation_rows):
        predictions.append(
            {
                "split": "confirmation",
                "dataset_index": CONFIRMATION_START + index,
                "question": row["question"],
                "gold": str(row["gold"]),
                **{name: values[index] for name, values in confirm_outputs.items()},
            }
        )
    if not confirm_primary["gate"]["passed"]:
        _save(args, summary, tensors=tensors, outputs=outputs, predictions=predictions)
        print(json.dumps(summary, indent=2))
        return summary

    ablation_names = (
        "per_layer_svd",
        "allocation_only",
        "fisher_only",
        "allocation_fisher",
        "protection_only",
    )
    ablation_teacher, ablation_deltas, _ = _teacher_split(
        model,
        tokenizer,
        confirmation_rows,
        latent_positions=latent_positions,
        batch_size=args.batch_size,
        device=device,
        arm_names=ablation_names,
        feature_weights=feature_weights,
        utilities=utilities,
        protected_bases=protected_bases,
    )
    ablation_generation, _, ablation_outputs = _generation_split(
        model,
        tokenizer,
        confirmation_rows,
        style=style,
        latent_positions=latent_positions,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        arm_names=ablation_names,
        feature_weights=feature_weights,
        utilities=utilities,
        protected_bases=protected_bases,
    )
    confirm_teacher.update(ablation_teacher)
    confirm_generation.update(ablation_generation)
    confirm_outputs.update(ablation_outputs)
    confirm_deltas.update(ablation_deltas)

    mechanism_values = {
        "allocation_fisher_over_ordinary": (
            confirm_deltas["ordinary_xkv"] - confirm_deltas["allocation_fisher"]
        ),
        "fisher_increment_over_allocation": (
            confirm_deltas["allocation_only"] - confirm_deltas["allocation_fisher"]
        ),
        "protection_increment": (
            confirm_deltas["allocation_fisher"] - confirm_deltas["full_method"]
        ),
    }
    effects = {
        name: _effect_record(
            values,
            seed=args.seed + 100 + index * 10,
            bootstrap_samples=args.bootstrap_samples,
            signflip_samples=args.signflip_samples,
        )
        for index, (name, values) in enumerate(mechanism_values.items())
    }
    holm = holm_bonferroni(
        {name: record["one_sided_sign_flip_pvalue"] for name, record in effects.items()}
    )
    for name in effects:
        effects[name]["holm"] = holm[name]

    random_maps, random_audit = _energy_matched_random_basis_maps(
        source,
        protected_bases,
        controls=args.random_controls,
        candidates=args.random_candidates,
        seed=args.seed + 1000,
    )
    random_deltas = []
    random_means = []
    for index, random_map in enumerate(random_maps):
        factorizer = FinalLatentTaskAwareFactorizer(
            latent_positions=latent_positions,
            groups=PREREGISTERED_GROUPS,
            rank=RANK,
            feature_weights=feature_weights,
            group_utilities=utilities,
            adaptive_ranks=True,
            protected_bases=random_map,
        )
        losses, _, _ = _evaluate(
            model,
            tokenizer,
            confirmation_rows,
            latent_positions,
            args.batch_size,
            device,
            factorizer,
            enforce_answer_eligibility=False,
        )
        delta = losses - confirm_baseline["losses"]
        random_deltas.append(delta)
        random_means.append(float(delta.mean()))
        print("random control", index, random_means[-1])
    random_mean = torch.stack(random_deltas).mean(0)
    random_advantage = random_mean - confirm_deltas["full_method"]
    random_interval = paired_bootstrap_interval(
        random_advantage,
        seed=args.seed + 2000,
        samples=args.bootstrap_samples,
    )
    full_mean = float(confirm_deltas["full_method"].mean())
    random_record = {
        "controls": args.random_controls,
        "mean_full_advantage_over_random": float(random_advantage.mean()),
        "bootstrap_95ci": random_interval,
        "full_beats_random_fraction": sum(
            full_mean < value for value in random_means
        )
        / len(random_means),
        "random_mean_nll_deltas": random_means,
    }
    protection_effect = effects["protection_increment"]
    protection_passed = bool(
        protection_effect["holm"]["rejected"]
        and protection_effect["bootstrap_95ci"][0] > 0
        and random_record["bootstrap_95ci"][0] > 0
        and random_record["full_beats_random_fraction"] >= 0.95
    )
    allocation_fisher_passed = bool(
        effects["allocation_fisher_over_ordinary"]["holm"]["rejected"]
        and effects["allocation_fisher_over_ordinary"]["bootstrap_95ci"][0] > 0
    )
    summary["mechanism"] = {
        "effects": effects,
        "random_protection": random_record,
        "allocation_fisher_passed": allocation_fisher_passed,
        "causal_protection_passed": protection_passed,
    }
    tensors["confirmation_per_example_nll_deltas"] = confirm_deltas
    tensors["random_per_example_nll_deltas"] = torch.stack(random_deltas)
    tensors["mechanism_effects"] = mechanism_values
    tensors["random_control_audit"] = random_audit
    outputs["confirmation"] = confirm_outputs

    final_arm_names = ("ordinary_xkv", "allocation_fisher", "full_method")
    final_teacher, final_deltas, _ = _teacher_split(
        model,
        tokenizer,
        final_rows,
        latent_positions=latent_positions,
        batch_size=args.batch_size,
        device=device,
        arm_names=final_arm_names,
        feature_weights=feature_weights,
        utilities=utilities,
        protected_bases=protected_bases,
    )
    final_generation, final_correct, final_outputs = _generation_split(
        model,
        tokenizer,
        final_rows,
        style=style,
        latent_positions=latent_positions,
        batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        arm_names=final_arm_names,
        feature_weights=feature_weights,
        utilities=utilities,
        protected_bases=protected_bases,
    )
    final_primary = _primary_comparison(
        final_teacher,
        final_deltas,
        final_generation,
        final_correct,
        seed=args.seed + 3000,
        bootstrap_samples=args.bootstrap_samples,
        storage_tolerance=args.storage_tolerance,
        minimum_retention=args.minimum_retention,
        noninferiority_margin=args.accuracy_noninferiority_margin,
    )
    final_passed = bool(final_primary["gate"]["passed"])
    summary["final_replication"] = {
        "teacher_forced": final_teacher,
        "generation": final_generation,
        "primary": final_primary,
    }
    tensors["final_per_example_nll_deltas"] = final_deltas
    outputs["final"] = final_outputs
    for index, row in enumerate(final_rows):
        predictions.append(
            {
                "split": "final",
                "dataset_index": FINAL_START + index,
                "question": row["question"],
                "gold": str(row["gold"]),
                **{name: values[index] for name, values in final_outputs.items()},
            }
        )

    transfer = None
    if final_passed and args.run_transfer:
        transfer = {}
        for dataset_name in ("svamp", "multiarith", "gsm_hard"):
            transfer_rows = load_eval_set(dataset_name, data_cfg.eval[dataset_name])[
                : args.transfer_examples
            ]
            transfer_generation, _, transfer_outputs = _generation_split(
                model,
                tokenizer,
                transfer_rows,
                style=style,
                latent_positions=latent_positions,
                batch_size=args.generation_batch_size,
                max_new_tokens=args.max_new_tokens,
                device=device,
                arm_names=final_arm_names,
                feature_weights=feature_weights,
                utilities=utilities,
                protected_bases=protected_bases,
            )
            transfer[dataset_name] = {
                "examples": len(transfer_rows),
                "generation": transfer_generation,
                "questions_sha256": _sha(
                    [_normalized_question(row["question"]) for row in transfer_rows]
                ),
            }
            outputs[f"transfer_{dataset_name}"] = transfer_outputs
    summary["transfer"] = transfer
    if not final_passed:
        claim = "STOP: confirmation did not replicate on the locked final holdout"
    elif protection_passed:
        claim = "CONFIRMED: rank-16 improvement with specific layer-11 protection"
    elif allocation_fisher_passed:
        claim = (
            "CONFIRMED: rank-16 task-aware allocation/Fisher improvement; "
            "causal protection unsupported"
        )
    else:
        claim = "CONFIRMED rank-16 full-method effect; mechanism remains unresolved"
    summary["decision"] = {
        "confirmation_passed": True,
        "final_passed": final_passed,
        "allocation_fisher_passed": allocation_fisher_passed,
        "causal_protection_passed": protection_passed,
        "claim": claim,
    }
    _save(args, summary, tensors=tensors, outputs=outputs, predictions=predictions)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--previous-experiment-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--confirmation-start", type=int, default=CONFIRMATION_START)
    parser.add_argument("--confirmation-examples", type=int, default=CONFIRMATION_EXAMPLES)
    parser.add_argument("--final-start", type=int, default=FINAL_START)
    parser.add_argument("--random-controls", type=int, default=100)
    parser.add_argument("--random-candidates", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--signflip-samples", type=int, default=10_000)
    parser.add_argument("--minimum-retention", type=float, default=0.95)
    parser.add_argument("--storage-tolerance", type=float, default=0.001)
    parser.add_argument("--accuracy-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--run-transfer", action="store_true")
    parser.add_argument("--transfer-examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
