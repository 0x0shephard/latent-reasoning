"""Confirm frozen native-cache hypotheses before protected residual compression."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
    _normalized_question, verify_full_reproduction_gate,
)
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES, GSM8K_TRAIN_URL,
    sample_gsm8k_train_calibration,
)
from scripts.run_codi_direct_cache_task_subspaces import _collect_direct_selection
from scripts.run_codi_preanswer_kv_subspace_discovery import (
    _evaluate, _json_metrics, _metrics, _sha,
)
from src.data.datasets import load_eval_set
from src.eval.official_codi import select_device
from src.mech.confirm_direct_cache_subspace import (
    bootstrap_interval, contiguous_fold_means, covariance_from_eigensystem,
    energy_matched_random_from_covariance, projected_damage_by_position,
)
from src.mech.direct_layerwise_kv import DirectLatentKVSubspaceIntervention
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint,
    load_official_checkpoint, resolve_torch_dtype,
)
from src.utils.config import load_config


SOURCE_CONTRACT = "official_codi_direct_cache_task_sensitive_independent_kv_v1"
CONTRACT = "official_codi_direct_cache_task_sensitive_confirmed_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_maps(source, layer, *, key_basis=None, value_basis=None):
    key = {
        index: source["selected_key_responses"][index].new_zeros((768, 0))
        for index in range(12)
    }
    value = {
        index: source["selected_value_responses"][index].new_zeros((768, 0))
        for index in range(12)
    }
    key[layer] = (
        source["selected_key_responses"][layer]
        if key_basis is None else key_basis
    )
    value[layer] = (
        source["selected_value_responses"][layer]
        if value_basis is None else value_basis
    )
    return key, value


def _evaluate_arm(
    model, tokenizer, rows, latent_positions, batch_size, device,
    source, layer, key_bases, value_bases, mode,
    dense_losses, dense_logits,
):
    intervention = DirectLatentKVSubspaceIntervention(
        key_bases=key_bases, value_bases=value_bases,
        key_means=source["key_means"], value_means=source["value_means"],
        layers=[layer], positions=range(latent_positions), mode=mode,
    )
    losses, logits, targets = _evaluate(
        model, tokenizer, rows, latent_positions, batch_size, device, intervention
    )
    return _metrics(losses, logits, targets, dense_losses, dense_logits)


def _reconstruct_and_validate_sampling(train, test, source, args):
    source_examples = int(source["sampling"]["selected_examples"])
    seed = int(source["sampling"]["sampling_seed"])
    rows, sampling = sample_gsm8k_train_calibration(
        train,
        test_questions={_normalized_question(row["question"]) for row in test},
        examples=source_examples + args.confirm_examples, seed=seed,
    )
    expected_total = (
        args.source_fit_examples + args.source_discovery_examples
        + args.source_rank_examples + args.source_causal_examples
    )
    if source_examples != expected_total:
        raise RuntimeError("source artifact split sizes do not match this confirmation")
    boundaries = {
        "cache_covariance_fit": (0, args.source_fit_examples),
        "direct_task_discovery": (
            args.source_fit_examples,
            args.source_fit_examples + args.source_discovery_examples,
        ),
        "rank_selection": (
            args.source_fit_examples + args.source_discovery_examples,
            args.source_fit_examples + args.source_discovery_examples
            + args.source_rank_examples,
        ),
        "causal_confirmation": (
            source_examples - args.source_causal_examples, source_examples
        ),
    }
    for name, (start, stop) in boundaries.items():
        actual = _sha([
            _normalized_question(row["question"]) for row in rows[start:stop]
        ])
        if actual != source["split_hashes"][name]:
            raise RuntimeError(f"could not reproduce source split: {name}")
    confirmation = rows[source_examples:]
    sampling["source_examples_reconstructed"] = source_examples
    sampling["confirmation_questions_sha256"] = _sha([
        _normalized_question(row["question"]) for row in confirmation
    ])
    return confirmation, sampling


def run(args):
    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    source = torch.load(args.source_artifact, map_location="cpu", weights_only=False)
    if source.get("contract") != SOURCE_CONTRACT:
        raise RuntimeError("source artifact is not the native-cache discovery artifact")
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
    if load_report.checkpoint_sha256 != source["checkpoint_sha256"]:
        raise RuntimeError("source artifact belongs to a different checkpoint")
    model.to(device=device, dtype=dtype).eval()

    from datasets import load_dataset
    train = load_dataset(
        "json", data_files={"train": GSM8K_TRAIN_URL}, split="train",
        verification_mode="no_checks",
    )
    if len(train) != GSM8K_EXPECTED_TRAIN_EXAMPLES:
        raise RuntimeError("GSM8K train count drifted")
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    confirmation_rows, sampling = _reconstruct_and_validate_sampling(
        train, test, source, args
    )
    if args.confirm_examples % args.confirm_folds:
        raise ValueError("confirm_examples must divide evenly into confirm_folds")
    latent_positions = int(cfg.eval.latent_iterations)
    candidates = [int(value) for value in args.candidate_layers.split(",")]
    if args.primary_layer not in candidates:
        raise ValueError("primary layer must be included among candidates")
    for layer in candidates:
        rank = (
            source["key_ranks_by_layer"][layer]
            + source["value_ranks_by_layer"][layer]
        )
        if not rank:
            raise RuntimeError(f"candidate layer {layer} has no frozen source basis")

    dense_losses, dense_logits, dense_targets = _evaluate(
        model, tokenizer, confirmation_rows, latent_positions,
        args.batch_size, device,
    )
    dense = _metrics(
        dense_losses, dense_logits, dense_targets, dense_losses, dense_logits
    )
    confirmations = {}
    random_audit = {}
    for layer in candidates:
        key_bases, value_bases = _selected_maps(source, layer)
        task_remove = _evaluate_arm(
            model, tokenizer, confirmation_rows, latent_positions, args.batch_size,
            device, source, layer, key_bases, value_bases, "remove",
            dense_losses, dense_logits,
        )
        task_retain = _evaluate_arm(
            model, tokenizer, confirmation_rows, latent_positions, args.batch_size,
            device, source, layer, key_bases, value_bases, "retain",
            dense_losses, dense_logits,
        )
        controls_by_kind = {}
        audits = {}
        for kind, covariance_system, selected in (
            ("key", source["key_covariance"], key_bases[layer]),
            ("value", source["value_covariance"], value_bases[layer]),
        ):
            covariance = covariance_from_eigensystem(
                covariance_system.eigenvalues,
                covariance_system.eigenvectors, layer,
            )
            controls_by_kind[kind], audits[kind] = energy_matched_random_from_covariance(
                covariance, selected, controls=args.random_controls,
                candidates=args.random_candidates,
                seed=args.seed + 100 * layer + (kind == "value"),
            )
        random_audit[f"layer_{layer:02d}"] = audits
        random_metrics = []
        random_deltas = []
        for control in range(args.random_controls):
            random_keys, random_values = _selected_maps(
                source, layer,
                key_basis=controls_by_kind["key"][control],
                value_basis=controls_by_kind["value"][control],
            )
            metrics = _evaluate_arm(
                model, tokenizer, confirmation_rows, latent_positions,
                args.batch_size, device, source, layer,
                random_keys, random_values, "remove", dense_losses, dense_logits,
            )
            random_metrics.append(_json_metrics(metrics))
            random_deltas.append(metrics["per_example_full_answer_nll_delta"])
        specificity = (
            task_remove["per_example_full_answer_nll_delta"]
            - torch.stack(random_deltas).mean(0)
        )
        fold_means = contiguous_fold_means(specificity, args.confirm_folds)
        primary_interval = bootstrap_interval(
            specificity, seed=args.seed + layer, alpha=0.05
        )
        familywise_interval = bootstrap_interval(
            specificity, seed=args.seed + 1000 + layer,
            alpha=0.05 / len(candidates),
        )
        fidelity = bool(
            task_retain["dense_first_token_top1_agreement"]
            >= args.minimum_retention
            and task_retain["mean_full_answer_nll_delta"]
            <= args.maximum_retain_nll_delta
        )
        positive_folds = sum(value > 0 for value in fold_means)
        confirmations[f"layer_{layer:02d}"] = {
            "key_rank": int(source["key_ranks_by_layer"][layer]),
            "value_rank": int(source["value_ranks_by_layer"][layer]),
            "task_remove": _json_metrics(task_remove),
            "task_retain": _json_metrics(task_retain),
            "random_remove": random_metrics,
            "specificity_mean_full_answer_nll_delta": float(specificity.mean()),
            "primary_bootstrap_95ci": primary_interval,
            "candidate_familywise_95ci": familywise_interval,
            "fold_specificity_means": fold_means,
            "positive_folds": positive_folds,
            "fidelity_passed": fidelity,
            "primary_passed": bool(
                layer == args.primary_layer and primary_interval[0] > 0
                and positive_folds >= args.minimum_positive_folds and fidelity
            ),
            "candidate_passed": bool(
                familywise_interval[0] > 0
                and positive_folds >= args.minimum_positive_folds and fidelity
            ),
        }
        print(layer, confirmations[f"layer_{layer:02d}"])

    primary = confirmations[f"layer_{args.primary_layer:02d}"]
    primary_passed = bool(primary["primary_passed"])

    # Diagnostic rank curve: rank 1 stays preregistered; larger ranks are not used
    # to rescue a failed primary confirmation.
    rank_sweep = {}
    full_value_basis = source["value_task_system"].eigenvectors[args.primary_layer]
    empty_key = source["selected_key_responses"][args.primary_layer].new_zeros((768, 0))
    for rank in [int(value) for value in args.primary_rank_sweep.split(",")]:
        if rank == int(source["value_ranks_by_layer"][args.primary_layer]):
            rank_sweep[f"rank_{rank}"] = {
                "remove": primary["task_remove"], "retain": primary["task_retain"],
                "preregistered": True,
            }
            continue
        key_bases, value_bases = _selected_maps(
            source, args.primary_layer, key_basis=empty_key,
            value_basis=full_value_basis[:, :rank],
        )
        rank_sweep[f"rank_{rank}"] = {
            "remove": _json_metrics(_evaluate_arm(
                model, tokenizer, confirmation_rows, latent_positions,
                args.batch_size, device, source, args.primary_layer,
                key_bases, value_bases, "remove", dense_losses, dense_logits,
            )),
            "retain": _json_metrics(_evaluate_arm(
                model, tokenizer, confirmation_rows, latent_positions,
                args.batch_size, device, source, args.primary_layer,
                key_bases, value_bases, "retain", dense_losses, dense_logits,
            )),
            "preregistered": False,
        }

    diagnostic_rows = confirmation_rows[: args.position_examples]
    (
        _keys, diagnostic_values, _key_gradients, diagnostic_value_gradients,
        diagnostic_connectivity,
    ) = _collect_direct_selection(
        model, tokenizer, diagnostic_rows, latent_positions,
        args.gradient_batch_size, device,
    )
    primary_basis = source["selected_value_responses"][args.primary_layer]
    position_damage = projected_damage_by_position(
        diagnostic_values[:, args.primary_layer],
        diagnostic_value_gradients[:, args.primary_layer],
        source["value_means"][args.primary_layer], primary_basis,
    )
    position_summary = []
    for position in range(latent_positions):
        values = position_damage[:, position]
        position_summary.append({
            "latent_position": position,
            "mean_predicted_removal_damage": float(values.mean()),
            "bootstrap_95ci": bootstrap_interval(
                values, seed=args.seed + 2000 + position, alpha=0.05
            ),
        })

    confirmed_key = {
        layer: source["selected_key_responses"][layer].new_zeros((768, 0))
        for layer in range(12)
    }
    confirmed_value = {
        layer: source["selected_value_responses"][layer].new_zeros((768, 0))
        for layer in range(12)
    }
    confirmed_key[args.primary_layer] = source["selected_key_responses"][args.primary_layer]
    confirmed_value[args.primary_layer] = source["selected_value_responses"][args.primary_layer]
    gate = {
        "passed": primary_passed,
        "passing_layers": [args.primary_layer] if primary_passed else [],
        "selected_contiguous_layers": [args.primary_layer] if primary_passed else [],
        "claim_scope": (
            "single preregistered layer; this does not establish an adjacent KV core"
        ),
        "thresholds": {
            "primary_layer": args.primary_layer,
            "primary_value_rank": int(source["value_ranks_by_layer"][args.primary_layer]),
            "minimum_positive_folds": args.minimum_positive_folds,
            "minimum_retention": args.minimum_retention,
            "maximum_retain_nll_delta": args.maximum_retain_nll_delta,
        },
    }
    confirmed_artifact = {
        "schema_version": 1, "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "source_artifact_sha256": _file_sha256(args.source_artifact),
        "gate": gate,
        "selected_key_responses": confirmed_key,
        "selected_value_responses": confirmed_value,
        "key_means": source["key_means"], "value_means": source["value_means"],
        "confirmation": primary,
    }
    summary = {
        "schema_version": 1,
        "contract": "official_codi_frozen_native_kv_confirmation_and_compression_gate_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "reproduction_gate": reproduction, "sampling": sampling,
        "source_split_hashes": source["split_hashes"],
        "candidate_layers": candidates, "primary_layer": args.primary_layer,
        "dense": _json_metrics(dense), "confirmations": confirmations,
        "rank_sweep": rank_sweep, "position_summary": position_summary,
        "position_gradient_connectivity": diagnostic_connectivity.float().mean(0).tolist(),
        "random_control_audit": random_audit,
        "compression_gate": gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, args.output_dir / "summary.json")
    _atomic_torch_save(
        confirmed_artifact, args.output_dir / "confirmed_native_kv_artifact.pt"
    )
    print(json.dumps({
        "primary": primary, "candidate_passes": {
            key: value["candidate_passed"] for key, value in confirmations.items()
        }, "compression_gate": gate,
    }, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--source-fit-examples", type=int, default=1024)
    parser.add_argument("--source-discovery-examples", type=int, default=1024)
    parser.add_argument("--source-rank-examples", type=int, default=256)
    parser.add_argument("--source-causal-examples", type=int, default=128)
    parser.add_argument("--confirm-examples", type=int, default=512)
    parser.add_argument("--confirm-folds", type=int, default=4)
    parser.add_argument("--candidate-layers", default="2,3,5,11")
    parser.add_argument("--primary-layer", type=int, default=11)
    parser.add_argument("--primary-rank-sweep", default="1,2,4,8")
    parser.add_argument("--position-examples", type=int, default=128)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=256)
    parser.add_argument("--minimum-positive-folds", type=int, default=3)
    parser.add_argument("--minimum-retention", type=float, default=0.95)
    parser.add_argument("--maximum-retain-nll-delta", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
