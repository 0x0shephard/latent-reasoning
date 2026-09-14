"""Confirm early CODI KV subspaces, then benchmark task-aware protected xKV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

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
from scripts.run_codi_confirm_and_compress_native_kv import (
    CONTRACT as PREVIOUS_ARTIFACT_CONTRACT,
    SOURCE_CONTRACT,
    _evaluate_arm,
    _selected_maps,
)
from scripts.run_codi_direct_cache_task_subspaces import _collect_direct_selection
from scripts.run_codi_preanswer_kv_subspace_discovery import (
    _evaluate,
    _metrics,
    _sha,
)
from src.data.datasets import load_eval_set
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.confirm_direct_cache_subspace import (
    bootstrap_interval,
    contiguous_fold_means,
    covariance_from_eigensystem,
    energy_matched_random_from_covariance,
)
from src.mech.task_aware_xkv import (
    compress_reconstruct_task_aware_cache,
    gradient_rms_group_metrics,
    quantize_reconstruct_cache,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_task_aware_protected_residual_xkv_v1"
PREVIOUS_SUMMARY_CONTRACT = (
    "official_codi_frozen_native_kv_confirmation_and_compression_gate_v1"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_groups(value: str) -> tuple[tuple[int, ...], ...]:
    groups = tuple(
        tuple(int(item) for item in group.split(",") if item.strip())
        for group in value.split(";")
        if group.strip()
    )
    flattened = [layer for group in groups for layer in group]
    if flattened != list(range(12)):
        raise ValueError("groups must partition layers 0..11 in order")
    return groups


def _validate_and_split_rows(train, test, source, previous_summary, args):
    source_examples = int(source["sampling"]["selected_examples"])
    previous_selected = int(previous_summary["sampling"]["selected_examples"])
    previous_source = int(
        previous_summary["sampling"]["source_examples_reconstructed"]
    )
    if previous_source != source_examples or previous_selected <= source_examples:
        raise RuntimeError("previous confirmation sampling is incompatible")
    total = previous_selected + args.confirm_examples + args.calibration_examples
    rows, sampling = sample_gsm8k_train_calibration(
        train,
        test_questions={_normalized_question(row["question"]) for row in test},
        examples=total,
        seed=int(source["sampling"]["sampling_seed"]),
    )
    expected_source = (
        args.source_fit_examples
        + args.source_discovery_examples
        + args.source_rank_examples
        + args.source_causal_examples
    )
    if source_examples != expected_source:
        raise RuntimeError("source split sizes differ from the preregistration")
    boundaries = {
        "cache_covariance_fit": (0, args.source_fit_examples),
        "direct_task_discovery": (
            args.source_fit_examples,
            args.source_fit_examples + args.source_discovery_examples,
        ),
        "rank_selection": (
            args.source_fit_examples + args.source_discovery_examples,
            args.source_fit_examples
            + args.source_discovery_examples
            + args.source_rank_examples,
        ),
        "causal_confirmation": (
            source_examples - args.source_causal_examples,
            source_examples,
        ),
    }
    for name, (start, stop) in boundaries.items():
        actual = _sha(
            [_normalized_question(row["question"]) for row in rows[start:stop]]
        )
        if actual != source["split_hashes"][name]:
            raise RuntimeError(f"could not reconstruct source split {name}")
    previous_rows = rows[source_examples:previous_selected]
    previous_hash = _sha(
        [_normalized_question(row["question"]) for row in previous_rows]
    )
    if previous_hash != previous_summary["sampling"]["confirmation_questions_sha256"]:
        raise RuntimeError("previous confirmation tail could not be reconstructed")
    confirm_stop = previous_selected + args.confirm_examples
    confirm_rows = rows[previous_selected:confirm_stop]
    calibration_rows = rows[confirm_stop:]
    sampling.update(
        {
            "source_examples_reconstructed": source_examples,
            "previous_confirmation_examples_reconstructed": len(previous_rows),
            "previous_confirmation_questions_sha256": previous_hash,
            "new_confirmation_questions_sha256": _sha(
                [_normalized_question(row["question"]) for row in confirm_rows]
            ),
            "compression_calibration_questions_sha256": _sha(
                [_normalized_question(row["question"]) for row in calibration_rows]
            ),
        }
    )
    return confirm_rows, calibration_rows, sampling


def _confirm_early_layers(
    model, tokenizer, rows, source, *, layers, latent_positions, args, device
):
    dense_losses, dense_logits, dense_targets = _evaluate(
        model, tokenizer, rows, latent_positions, args.batch_size, device
    )
    dense_metrics = _metrics(
        dense_losses, dense_logits, dense_targets, dense_losses, dense_logits
    )
    confirmations = {}
    random_audit = {}
    for layer in layers:
        key_bases, value_bases = _selected_maps(source, layer)
        task_remove = _evaluate_arm(
            model, tokenizer, rows, latent_positions, args.batch_size, device,
            source, layer, key_bases, value_bases, "remove", dense_losses, dense_logits,
        )
        task_retain = _evaluate_arm(
            model, tokenizer, rows, latent_positions, args.batch_size, device,
            source, layer, key_bases, value_bases, "retain", dense_losses, dense_logits,
        )
        controls_by_kind = {}
        audits = {}
        for kind, covariance_system, selected in (
            ("key", source["key_covariance"], key_bases[layer]),
            ("value", source["value_covariance"], value_bases[layer]),
        ):
            covariance = covariance_from_eigensystem(
                covariance_system.eigenvalues, covariance_system.eigenvectors, layer
            )
            controls_by_kind[kind], audits[kind] = energy_matched_random_from_covariance(
                covariance,
                selected,
                controls=args.confirm_random_controls,
                candidates=args.random_candidates,
                seed=args.seed + 100 * layer + (kind == "value"),
            )
        random_audit[f"layer_{layer:02d}"] = audits
        random_deltas = []
        random_metrics = []
        for control in range(args.confirm_random_controls):
            random_keys, random_values = _selected_maps(
                source,
                layer,
                key_basis=controls_by_kind["key"][control],
                value_basis=controls_by_kind["value"][control],
            )
            metrics = _evaluate_arm(
                model, tokenizer, rows, latent_positions, args.batch_size, device,
                source, layer, random_keys, random_values, "remove",
                dense_losses, dense_logits,
            )
            random_deltas.append(metrics["per_example_full_answer_nll_delta"])
            random_metrics.append(
                {key: value for key, value in metrics.items() if not key.startswith("per_")}
            )
        specificity = task_remove["per_example_full_answer_nll_delta"] - torch.stack(
            random_deltas
        ).mean(0)
        interval = bootstrap_interval(
            specificity,
            seed=args.seed + 1000 + layer,
            alpha=0.05 / len(layers),
        )
        folds = contiguous_fold_means(specificity, args.confirm_folds)
        positive_folds = sum(value > 0 for value in folds)
        fidelity = bool(
            task_retain["dense_first_token_top1_agreement"] >= args.minimum_retention
            and task_retain["mean_full_answer_nll_delta"]
            <= args.maximum_retain_nll_delta
        )
        passed = bool(
            interval[0] > 0
            and positive_folds >= args.minimum_positive_folds
            and fidelity
        )
        confirmations[f"layer_{layer:02d}"] = {
            "key_rank": int(source["key_ranks_by_layer"][layer]),
            "value_rank": int(source["value_ranks_by_layer"][layer]),
            "specificity_mean_full_answer_nll_delta": float(specificity.mean()),
            "familywise_95ci": interval,
            "fold_specificity_means": folds,
            "positive_folds": positive_folds,
            "fidelity_passed": fidelity,
            "task_remove": {
                key: value for key, value in task_remove.items() if not key.startswith("per_")
            },
            "task_retain": {
                key: value for key, value in task_retain.items() if not key.startswith("per_")
            },
            "random_remove": random_metrics,
            "passed": passed,
        }
        print("confirmation", layer, confirmations[f"layer_{layer:02d}"])
    return dense_metrics, confirmations, random_audit


def _confirmed_basis_map(source, previous_artifact, confirmations, early_layers):
    result = {}
    included = []
    # Layer 11 was already confirmed on an independent 512-question tail.
    for kind, key in (
        ("key", "selected_key_responses"),
        ("value", "selected_value_responses"),
    ):
        basis = previous_artifact[key][11]
        if basis.shape[1]:
            result[(11, kind)] = basis
    included.append(11)
    for layer in early_layers:
        if not confirmations[f"layer_{layer:02d}"]["passed"]:
            continue
        key_basis = source["selected_key_responses"][layer]
        value_basis = source["selected_value_responses"][layer]
        if key_basis.shape[1]:
            result[(layer, "key")] = key_basis
        if value_basis.shape[1]:
            result[(layer, "value")] = value_basis
        included.append(layer)
    return result, sorted(included)


def _energy_matched_random_basis_maps(
    source, protected_bases, *, controls, candidates, seed
):
    per_component = {}
    audits = {}
    for (layer, kind), selected in protected_bases.items():
        system = source[f"{kind}_covariance"]
        covariance = covariance_from_eigensystem(
            system.eigenvalues, system.eigenvectors, layer
        )
        bases, audit = energy_matched_random_from_covariance(
            covariance,
            selected,
            controls=controls,
            candidates=max(candidates, controls),
            seed=seed + 100 * layer + (kind == "value"),
        )
        per_component[(layer, kind)] = bases
        audits[f"layer_{layer:02d}_{kind}"] = audit
    maps = []
    for control in range(controls):
        maps.append(
            {component: bases[control] for component, bases in per_component.items()}
        )
    return maps, audits


class FinalLatentTaskAwareFactorizer:
    """Apply one quality-proxy cache transformation after CODI latent step six."""

    def __init__(
        self,
        *,
        latent_positions,
        groups=None,
        rank=None,
        feature_weights=None,
        group_utilities=None,
        adaptive_ranks=False,
        protected_bases=None,
        factor_quantization_bits=None,
        dense_quantization_bits=None,
    ):
        self.latent_positions = int(latent_positions)
        self.groups = groups
        self.rank = rank
        self.feature_weights = feature_weights
        self.group_utilities = group_utilities
        self.adaptive_ranks = adaptive_ranks
        self.protected_bases = protected_bases
        self.factor_quantization_bits = factor_quantization_bits
        self.dense_quantization_bits = dense_quantization_bits
        self.reports = []

    def __call__(self, cache, latent_position):
        if latent_position != self.latent_positions - 1:
            return cache
        if self.dense_quantization_bits is not None:
            transformed, report = quantize_reconstruct_cache(
                cache, bits=self.dense_quantization_bits
            )
        else:
            transformed, report = compress_reconstruct_task_aware_cache(
                cache,
                groups=self.groups,
                rank=self.rank,
                feature_weights=self.feature_weights,
                group_utilities=self.group_utilities,
                adaptive_ranks=self.adaptive_ranks,
                protected_bases=self.protected_bases,
                latent_positions=self.latent_positions,
                factor_quantization_bits=self.factor_quantization_bits,
            )
        self.reports.append(report)
        return transformed

    def summary(self):
        dense_bits = sum(report["dense_bits"] for report in self.reports)
        cache_bits = sum(report["cache_bits"] for report in self.reports)
        records = [
            record
            for report in self.reports
            for record in report.get("records", [])
        ]
        ranks_by_group = {}
        for record in records:
            key = ",".join(map(str, record["group"]))
            ranks_by_group.setdefault(key, []).append(record["effective_rank"])
        return {
            "batches": len(self.reports),
            "dense_bits": dense_bits,
            "cache_bits": cache_bits,
            "modelled_compression_ratio": dense_bits / max(cache_bits, 1),
            "protected_residual_bits": sum(
                report.get("protected_residual_bits", 0) for report in self.reports
            ),
            "mean_relative_reconstruction_error": (
                sum(record["relative_reconstruction_error"] for record in records)
                / len(records)
                if records
                else None
            ),
            "effective_rank_by_group": {
                key: {
                    "minimum": min(values),
                    "mean": sum(values) / len(values),
                    "maximum": max(values),
                }
                for key, values in ranks_by_group.items()
            },
            "runtime_representation": self.reports[0]["runtime_representation"]
            if self.reports
            else None,
        }


def _topk_overlap(logits, dense_logits, k=5):
    k = min(int(k), logits.shape[-1])
    current = logits.topk(k, dim=-1).indices
    dense = dense_logits.topk(k, dim=-1).indices
    overlap = (current.unsqueeze(-1) == dense.unsqueeze(-2)).any(-1).float().sum(-1)
    return float((overlap / k).mean())


def _teacher_forced_arm(
    model, tokenizer, rows, latent_positions, batch_size, device,
    dense_losses, dense_logits, factorizer
):
    losses, logits, targets = _evaluate(
        model, tokenizer, rows, latent_positions, batch_size, device, factorizer
    )
    metrics = _metrics(losses, logits, targets, dense_losses, dense_logits)
    serializable = {
        key: value for key, value in metrics.items() if not key.startswith("per_")
    }
    serializable["perplexity_from_mean_answer_nll"] = math.exp(
        min(serializable["mean_full_answer_nll"], 50)
    )
    serializable["first_token_top5_overlap"] = _topk_overlap(logits, dense_logits, 5)
    serializable["cache"] = factorizer.summary()
    return serializable, losses - dense_losses


def _accuracy(outputs, rows):
    return sum(
        official_answers_match(text, row["gold"])
        for text, row in zip(outputs, rows)
    ) / len(rows)


def _arm_factory(
    name, rank, *, latent_positions, per_layer_groups, xkv_groups,
    weights, utilities, protected, quant_bits=None
):
    if name == "per_layer_svd":
        return FinalLatentTaskAwareFactorizer(
            latent_positions=latent_positions, groups=per_layer_groups, rank=rank
        )
    if name == "xkv_svd":
        return FinalLatentTaskAwareFactorizer(
            latent_positions=latent_positions, groups=xkv_groups, rank=rank
        )
    if name == "answer_fisher_xkv":
        return FinalLatentTaskAwareFactorizer(
            latent_positions=latent_positions, groups=xkv_groups, rank=rank,
            feature_weights=weights,
        )
    if name == "causal_residual_xkv":
        return FinalLatentTaskAwareFactorizer(
            latent_positions=latent_positions, groups=xkv_groups, rank=rank,
            protected_bases=protected,
        )
    if name == "full_task_aware_xkv":
        return FinalLatentTaskAwareFactorizer(
            latent_positions=latent_positions, groups=xkv_groups, rank=rank,
            feature_weights=weights, group_utilities=utilities,
            adaptive_ranks=True, protected_bases=protected,
            factor_quantization_bits=quant_bits,
        )
    raise ValueError(f"unknown arm {name}")


def run(args):
    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    previous_artifact = torch.load(
        args.previous_confirmation_artifact, map_location="cpu", weights_only=False
    )
    previous_summary = json.loads(args.previous_confirmation_summary.read_text())
    if source.get("contract") != SOURCE_CONTRACT:
        raise RuntimeError("wrong native-cache discovery artifact")
    if previous_artifact.get("contract") != PREVIOUS_ARTIFACT_CONTRACT:
        raise RuntimeError("wrong previous confirmation artifact")
    if previous_summary.get("contract") != PREVIOUS_SUMMARY_CONTRACT:
        raise RuntimeError("wrong previous confirmation summary")
    if not previous_artifact["gate"]["passed"]:
        raise RuntimeError("the prior layer-11 confirmation did not pass")
    if previous_artifact["source_artifact_sha256"] != _file_sha256(args.source_artifact):
        raise RuntimeError("previous confirmation refers to another source artifact")

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
    if any(
        load_report.checkpoint_sha256 != artifact["checkpoint_sha256"]
        for artifact in (source, previous_artifact)
    ):
        raise RuntimeError("checkpoint provenance mismatch")
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
    confirm_rows, calibration_rows, sampling = _validate_and_split_rows(
        train, test, source, previous_summary, args
    )
    if args.confirm_examples % args.confirm_folds:
        raise ValueError("confirmation examples must divide into folds")
    early_layers = tuple(int(value) for value in args.early_layers.split(","))
    if early_layers != (2, 3):
        raise ValueError("the preregistered early layers are exactly 2 and 3")
    latent_positions = int(cfg.eval.latent_iterations)
    dense_confirmation, confirmations, confirm_random_audit = _confirm_early_layers(
        model,
        tokenizer,
        confirm_rows,
        source,
        layers=early_layers,
        latent_positions=latent_positions,
        args=args,
        device=device,
    )
    protected_bases, included_layers = _confirmed_basis_map(
        source, previous_artifact, confirmations, early_layers
    )
    strict_combined_gate = bool(
        previous_artifact["gate"]["passed"]
        and all(confirmations[f"layer_{layer:02d}"]["passed"] for layer in early_layers)
    )

    keys, values, key_gradients, value_gradients, connectivity = (
        _collect_direct_selection(
            model,
            tokenizer,
            calibration_rows,
            latent_positions,
            args.gradient_batch_size,
            device,
        )
    )
    del keys, values
    if not bool(connectivity.all()):
        raise RuntimeError("calibration gradients are not connected to every K/V cache")
    xkv_groups = _parse_groups(args.groups)
    per_layer_groups = tuple((layer,) for layer in range(12))
    feature_weights, group_utilities = gradient_rms_group_metrics(
        key_gradients, value_gradients, xkv_groups
    )
    random_maps, compression_random_audit = _energy_matched_random_basis_maps(
        source,
        protected_bases,
        controls=args.random_protection_controls,
        candidates=args.random_candidates,
        seed=args.seed + 5000,
    )

    evaluation_rows = test[: args.test_examples]
    generation_rows = evaluation_rows[: args.generation_examples]
    dense_losses, dense_logits, dense_targets = _evaluate(
        model, tokenizer, evaluation_rows, latent_positions, args.batch_size, device
    )
    dense_teacher = _metrics(
        dense_losses, dense_logits, dense_targets, dense_losses, dense_logits
    )
    dense_teacher = {
        key: value for key, value in dense_teacher.items() if not key.startswith("per_")
    }
    dense_teacher["perplexity_from_mean_answer_nll"] = math.exp(
        min(dense_teacher["mean_full_answer_nll"], 50)
    )
    dense_teacher["first_token_top5_overlap"] = 1.0

    style = PromptStyle.from_config(data_cfg.prompt)
    dense_outputs = generate_official_codi(
        model,
        tokenizer,
        [row["question"] for row in generation_rows],
        latent_iterations=latent_positions,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.generation_batch_size,
        device=device,
        answer_cue=style.answer_prefix,
        force_answer_cue=True,
    )
    dense_accuracy = _accuracy(dense_outputs, generation_rows)
    results = {
        "dense": {
            "teacher_forced": dense_teacher,
            "generation": {"accuracy": dense_accuracy, "outputs": dense_outputs},
        }
    }
    per_example_deltas = {}
    ranks = sorted({int(value) for value in args.ranks.split(",")})
    generation_ranks = {int(value) for value in args.generation_ranks.split(",")}
    main_arm_names = (
        "per_layer_svd",
        "xkv_svd",
        "answer_fisher_xkv",
        "causal_residual_xkv",
        "full_task_aware_xkv",
    )
    for rank in ranks:
        for arm_name in main_arm_names:
            result_name = f"{arm_name}_r{rank}"
            factorizer = _arm_factory(
                arm_name,
                rank,
                latent_positions=latent_positions,
                per_layer_groups=per_layer_groups,
                xkv_groups=xkv_groups,
                weights=feature_weights,
                utilities=group_utilities,
                protected=protected_bases,
            )
            teacher, delta = _teacher_forced_arm(
                model, tokenizer, evaluation_rows, latent_positions,
                args.batch_size, device, dense_losses, dense_logits, factorizer,
            )
            results[result_name] = {"teacher_forced": teacher}
            per_example_deltas[result_name] = delta
            if rank in generation_ranks:
                generation_factorizer = _arm_factory(
                    arm_name,
                    rank,
                    latent_positions=latent_positions,
                    per_layer_groups=per_layer_groups,
                    xkv_groups=xkv_groups,
                    weights=feature_weights,
                    utilities=group_utilities,
                    protected=protected_bases,
                )
                started = time.perf_counter()
                outputs = generate_official_codi(
                    model,
                    tokenizer,
                    [row["question"] for row in generation_rows],
                    latent_iterations=latent_positions,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.generation_batch_size,
                    device=device,
                    kv_intervention=generation_factorizer,
                    answer_cue=style.answer_prefix,
                    force_answer_cue=True,
                )
                elapsed = time.perf_counter() - started
                results[result_name]["generation"] = {
                    "accuracy": _accuracy(outputs, generation_rows),
                    "accuracy_retained_fraction": (
                        _accuracy(outputs, generation_rows) / dense_accuracy
                        if dense_accuracy
                        else None
                    ),
                    "exact_sequence_agreement": sum(
                        output == dense
                        for output, dense in zip(outputs, dense_outputs)
                    )
                    / len(outputs),
                    "seconds_quality_proxy": elapsed,
                    "outputs": outputs,
                    "cache": generation_factorizer.summary(),
                }
            print(result_name, results[result_name]["teacher_forced"])

    focal_rank = int(args.focal_rank)
    if focal_rank not in ranks:
        raise ValueError("focal rank must be included in ranks")
    random_deltas = []
    random_results = {}
    for index, random_map in enumerate(random_maps):
        name = f"random_protected_full_r{focal_rank}_seed_{index:02d}"
        factorizer = _arm_factory(
            "full_task_aware_xkv",
            focal_rank,
            latent_positions=latent_positions,
            per_layer_groups=per_layer_groups,
            xkv_groups=xkv_groups,
            weights=feature_weights,
            utilities=group_utilities,
            protected=random_map,
        )
        teacher, delta = _teacher_forced_arm(
            model, tokenizer, evaluation_rows, latent_positions,
            args.batch_size, device, dense_losses, dense_logits, factorizer,
        )
        random_results[name] = teacher
        random_deltas.append(delta)
    task_delta = per_example_deltas[f"full_task_aware_xkv_r{focal_rank}"]
    random_mean_delta = torch.stack(random_deltas).mean(0)
    task_advantage = random_mean_delta - task_delta
    random_comparison = {
        "controls": args.random_protection_controls,
        "focal_rank": focal_rank,
        "mean_task_nll_advantage_over_random": float(task_advantage.mean()),
        "bootstrap_95ci": bootstrap_interval(
            task_advantage, seed=args.seed + 9000, alpha=0.05
        ),
        "task_beats_random_fraction": sum(
            float(task_delta.mean()) < float(delta.mean()) for delta in random_deltas
        )
        / len(random_deltas),
        "results": random_results,
    }

    paired_xkv_comparisons = {}
    for rank in ranks:
        ordinary_name = f"xkv_svd_r{rank}"
        full_name = f"full_task_aware_xkv_r{rank}"
        # Positive means the full method has lower per-example answer NLL.
        advantage = per_example_deltas[ordinary_name] - per_example_deltas[full_name]
        interval = bootstrap_interval(
            advantage, seed=args.seed + 10000 + rank, alpha=0.05
        )
        ordinary_cache = results[ordinary_name]["teacher_forced"]["cache"]
        full_cache = results[full_name]["teacher_forced"]["cache"]
        paired_xkv_comparisons[f"rank_{rank}"] = {
            "rank": rank,
            "mean_full_method_nll_advantage": float(advantage.mean()),
            "paired_bootstrap_95ci": interval,
            "ordinary_xkv_modelled_compression_ratio": ordinary_cache[
                "modelled_compression_ratio"
            ],
            "full_method_modelled_compression_ratio": full_cache[
                "modelled_compression_ratio"
            ],
            "full_to_ordinary_cache_bits_ratio": (
                full_cache["cache_bits"] / ordinary_cache["cache_bits"]
            ),
            "strict_quality_win": bool(
                float(advantage.mean()) > 0 and interval[0] > 0
            ),
        }

    for bits in (8, 4):
        dense_name = f"dense_symmetric_int{bits}"
        dense_quantizer = FinalLatentTaskAwareFactorizer(
            latent_positions=latent_positions, dense_quantization_bits=bits
        )
        teacher, delta = _teacher_forced_arm(
            model, tokenizer, evaluation_rows, latent_positions,
            args.batch_size, device, dense_losses, dense_logits, dense_quantizer,
        )
        results[dense_name] = {"teacher_forced": teacher}
        per_example_deltas[dense_name] = delta

        dense_generation_quantizer = FinalLatentTaskAwareFactorizer(
            latent_positions=latent_positions, dense_quantization_bits=bits
        )
        started = time.perf_counter()
        dense_quantized_outputs = generate_official_codi(
            model,
            tokenizer,
            [row["question"] for row in generation_rows],
            latent_iterations=latent_positions,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
            device=device,
            kv_intervention=dense_generation_quantizer,
            answer_cue=style.answer_prefix,
            force_answer_cue=True,
        )
        results[dense_name]["generation"] = {
            "accuracy": _accuracy(dense_quantized_outputs, generation_rows),
            "accuracy_retained_fraction": (
                _accuracy(dense_quantized_outputs, generation_rows) / dense_accuracy
                if dense_accuracy
                else None
            ),
            "exact_sequence_agreement": sum(
                output == dense
                for output, dense in zip(dense_quantized_outputs, dense_outputs)
            )
            / len(dense_quantized_outputs),
            "seconds_quality_proxy": time.perf_counter() - started,
            "outputs": dense_quantized_outputs,
            "cache": dense_generation_quantizer.summary(),
        }

        hybrid_name = f"full_task_aware_xkv_r{focal_rank}_int{bits}_factors"
        hybrid = _arm_factory(
            "full_task_aware_xkv",
            focal_rank,
            latent_positions=latent_positions,
            per_layer_groups=per_layer_groups,
            xkv_groups=xkv_groups,
            weights=feature_weights,
            utilities=group_utilities,
            protected=protected_bases,
            quant_bits=bits,
        )
        teacher, delta = _teacher_forced_arm(
            model, tokenizer, evaluation_rows, latent_positions,
            args.batch_size, device, dense_losses, dense_logits, hybrid,
        )
        results[hybrid_name] = {"teacher_forced": teacher}
        per_example_deltas[hybrid_name] = delta

        hybrid_generation = _arm_factory(
            "full_task_aware_xkv",
            focal_rank,
            latent_positions=latent_positions,
            per_layer_groups=per_layer_groups,
            xkv_groups=xkv_groups,
            weights=feature_weights,
            utilities=group_utilities,
            protected=protected_bases,
            quant_bits=bits,
        )
        started = time.perf_counter()
        hybrid_outputs = generate_official_codi(
            model,
            tokenizer,
            [row["question"] for row in generation_rows],
            latent_iterations=latent_positions,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
            device=device,
            kv_intervention=hybrid_generation,
            answer_cue=style.answer_prefix,
            force_answer_cue=True,
        )
        results[hybrid_name]["generation"] = {
            "accuracy": _accuracy(hybrid_outputs, generation_rows),
            "accuracy_retained_fraction": (
                _accuracy(hybrid_outputs, generation_rows) / dense_accuracy
                if dense_accuracy
                else None
            ),
            "exact_sequence_agreement": sum(
                output == dense for output, dense in zip(hybrid_outputs, dense_outputs)
            )
            / len(hybrid_outputs),
            "seconds_quality_proxy": time.perf_counter() - started,
            "outputs": hybrid_outputs,
            "cache": hybrid_generation.summary(),
        }

    serializable_results = {}
    outputs = {"dense": dense_outputs}
    for name, record in results.items():
        clean = {"teacher_forced": record["teacher_forced"]}
        if "generation" in record:
            generation = dict(record["generation"])
            if "outputs" in generation:
                outputs[name] = generation.pop("outputs")
            clean["generation"] = generation
        serializable_results[name] = clean

    static_basis_parameters = sum(basis.numel() for basis in protected_bases.values())
    summary = {
        "schema_version": 1,
        "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction,
        "sampling": sampling,
        "confirmation": {
            "previous_layer_11_gate": previous_artifact["gate"],
            "new_early_layers": confirmations,
            "strict_layers_2_3_11_gate_passed": strict_combined_gate,
            "included_layers": included_layers,
            "claim_scope": (
                "layers 2 and 3 are new co-primary confirmations; layer 11 was "
                "confirmed in the immediately preceding disjoint experiment"
            ),
        },
        "compression_calibration": {
            "examples": len(calibration_rows),
            "gradient_connectivity_fraction": float(connectivity.float().mean()),
            "groups": [list(group) for group in xkv_groups],
            "group_utilities": {
                ",".join(map(str, group)): value
                for group, value in group_utilities.items()
            },
            "static_protected_basis_parameters_not_counted_as_per_request_cache": (
                static_basis_parameters
            ),
        },
        "evaluation": {
            "teacher_forced_examples": len(evaluation_rows),
            "generation_examples": len(generation_rows),
            "ranks": ranks,
            "generation_ranks": sorted(generation_ranks),
            "max_new_tokens": args.max_new_tokens,
        },
        "results": serializable_results,
        "paired_ordinary_xkv_comparisons": paired_xkv_comparisons,
        "random_protected_comparison": random_comparison,
        "warnings": [
            "This is a dense-reconstruction quality and modeled-storage proxy, not a native xKV latency benchmark.",
            "The answer-Fisher arm uses fixed calibration gradients that pass through Q, softmax attention and V; it is not an exact implementation of the KQ-SVD paper.",
            "Only the pre-answer cache is compressed once after latent step six; subsequently generated visible-token rows remain dense in this reference path.",
            "INT4/INT8 are symmetric fake-quantization baselines, not optimized KIVI kernels.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save(
        {
            **summary,
            "feature_weights": feature_weights,
            "protected_bases": protected_bases,
            "per_example_nll_deltas": per_example_deltas,
            "outputs": outputs,
            "confirmation_random_audit": confirm_random_audit,
            "compression_random_audit": compression_random_audit,
        },
        args.output_dir / "task_aware_protected_xkv.pt",
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(generation_rows):
            payload = {
                "index": index,
                "question": str(row["question"]),
                "gold": str(row["gold"]),
                **{name: values[index] for name, values in outputs.items()},
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--previous-confirmation-summary", type=Path, required=True)
    parser.add_argument("--previous-confirmation-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--source-fit-examples", type=int, default=1024)
    parser.add_argument("--source-discovery-examples", type=int, default=1024)
    parser.add_argument("--source-rank-examples", type=int, default=256)
    parser.add_argument("--source-causal-examples", type=int, default=128)
    parser.add_argument("--confirm-examples", type=int, default=512)
    parser.add_argument("--confirm-folds", type=int, default=4)
    parser.add_argument("--calibration-examples", type=int, default=512)
    parser.add_argument("--early-layers", default="2,3")
    parser.add_argument("--groups", default="0,1,2,3;4,5,6,7;8,9,10,11")
    parser.add_argument("--ranks", default="16,32,48")
    parser.add_argument("--generation-ranks", default="32,48")
    parser.add_argument("--focal-rank", type=int, default=32)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--generation-examples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--confirm-random-controls", type=int, default=8)
    parser.add_argument("--random-protection-controls", type=int, default=20)
    parser.add_argument("--random-candidates", type=int, default=256)
    parser.add_argument("--minimum-positive-folds", type=int, default=3)
    parser.add_argument("--minimum-retention", type=float, default=0.95)
    parser.add_argument("--maximum-retain-nll-delta", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
