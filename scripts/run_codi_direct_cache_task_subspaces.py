"""Discover task-sensitive CODI subspaces directly in native K and V caches."""
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
    _normalized_question, verify_full_reproduction_gate,
)
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES, GSM8K_TRAIN_URL,
    sample_gsm8k_train_calibration,
)
from scripts.run_codi_preanswer_kv_subspace_discovery import (
    _evaluate, _json_metrics, _metrics, _sha,
)
from src.data.datasets import load_eval_set
from src.data.official_codi_training import collate_official_codi_kv_rows
from src.eval.official_codi import select_device
from src.mech.direct_cache_task_subspace import (
    select_direct_cache_ranks, selected_variance_fraction,
    task_covariance_hybrid_basis,
)
from src.mech.direct_layerwise_kv import (
    DirectLatentKVSubspaceIntervention, fit_layerwise_eigensystems,
    longest_contiguous_run,
)
from src.mech.layerwise_u28 import energy_matched_random_bases
from src.mech.official_codi_target_utility import OfficialCODIAnswerScorer
from src.mech.preanswer_kv_subspace import official_codi_preanswer_kv_forward
from src.mech.task_sensitive_kv_subspace import fit_task_sensitive_eigensystem
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint,
    load_official_checkpoint, resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_direct_cache_task_sensitive_independent_kv_v1"


def _collect_cache_fit(model, tokenizer, rows, scorer, batch_size, device):
    key_chunks, value_chunks = [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start : start + batch_size], bot_token_id=model.bot_id
        ).to(device)
        with torch.no_grad():
            output = scorer(batch, return_kv=True)
        for tensor, destination in (
            (output.student_keys, key_chunks), (output.student_values, value_chunks)
        ):
            if tensor is None:
                raise RuntimeError("fit pass did not return latent K/V cache")
            destination.append(
                tensor.permute(0, 1, 3, 2, 4).reshape(
                    tensor.shape[0], 12, scorer.latent_positions, 768
                ).float().cpu()
            )
        del batch, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(key_chunks), torch.cat(value_chunks)


def _collect_direct_selection(
    model, tokenizer, rows, latent_positions, batch_size, device
):
    keys, values, key_gradients, value_gradients, connectivity = [], [], [], [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start : start + batch_size], bot_token_id=model.bot_id
        ).to(device)
        output = official_codi_preanswer_kv_forward(
            model, batch, latent_positions=latent_positions, return_gradients=True
        )
        keys.append(output.key_states.detach().float().cpu())
        values.append(output.value_states.detach().float().cpu())
        key_gradients.append(output.key_gradients.detach().float().cpu())
        value_gradients.append(output.value_gradients.detach().float().cpu())
        connectivity.append(output.gradient_connected)
        del batch, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        torch.cat(keys), torch.cat(values),
        torch.cat(key_gradients), torch.cat(value_gradients),
        torch.stack(connectivity),
    )


def _rank_row(audit, rank):
    return next((row for row in audit["rank_grid"] if row["rank"] == rank), None)


def _empty_bases(reference):
    return {layer: reference[layer].new_zeros((reference[layer].shape[0], 0))
            for layer in reference}


def _bootstrap_interval(values, *, seed, alpha, samples=8000):
    values = values.double()
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        values.numel(), (samples, values.numel()), generator=generator
    )
    means = values[draws].mean(1)
    return [
        float(torch.quantile(means, alpha / 2)),
        float(torch.quantile(means, 1 - alpha / 2)),
    ]


def run(args):
    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
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
    total = (
        args.fit_examples + args.discovery_examples
        + args.rank_examples + args.causal_examples
    )
    rows, sampling = sample_gsm8k_train_calibration(
        train,
        test_questions={_normalized_question(row["question"]) for row in test},
        examples=total, seed=args.seed,
    )
    fit_end = args.fit_examples
    discovery_end = fit_end + args.discovery_examples
    rank_end = discovery_end + args.rank_examples
    fit_rows = rows[:fit_end]
    discovery_rows = rows[fit_end:discovery_end]
    rank_rows = rows[discovery_end:rank_end]
    causal_rows = rows[rank_end:]
    latent_positions = int(cfg.eval.latent_iterations)
    scorer = OfficialCODIAnswerScorer(model, latent_positions=latent_positions)

    fit_keys, fit_values = _collect_cache_fit(
        model, tokenizer, fit_rows, scorer, args.fit_batch_size, device
    )
    key_covariance = fit_layerwise_eigensystems(fit_keys)
    value_covariance = fit_layerwise_eigensystems(fit_values)
    (
        discovery_keys, discovery_values,
        discovery_key_gradients, discovery_value_gradients, discovery_connectivity,
    ) = _collect_direct_selection(
        model, tokenizer, discovery_rows, latent_positions,
        args.gradient_batch_size, device,
    )
    (
        rank_keys, rank_values, rank_key_gradients, rank_value_gradients,
        rank_connectivity,
    ) = _collect_direct_selection(
        model, tokenizer, rank_rows, latent_positions,
        args.gradient_batch_size, device,
    )
    connectivity = torch.cat(
        (discovery_connectivity, rank_connectivity), dim=0
    ).float().mean(0)
    if not bool((connectivity == 1).all()):
        raise RuntimeError("exact native K/V gradients are not fully connected")

    key_task = fit_task_sensitive_eigensystem(
        discovery_keys, discovery_key_gradients, key_covariance.means
    )
    value_task = fit_task_sensitive_eigensystem(
        discovery_values, discovery_value_gradients, value_covariance.means
    )
    rank_grid = [int(value) for value in args.rank_grid.split(",")]
    key_bases, key_ranks, key_audit = select_direct_cache_ranks(
        key_task, rank_keys, rank_key_gradients, key_covariance.means,
        rank_grid=rank_grid, retained_best_effect=args.retained_best_effect,
        minimum_half_overlap=args.minimum_half_overlap, seed=args.seed + 10,
    )
    value_bases, value_ranks, value_audit = select_direct_cache_ranks(
        value_task, rank_values, rank_value_gradients, value_covariance.means,
        rank_grid=rank_grid, retained_best_effect=args.retained_best_effect,
        minimum_half_overlap=args.minimum_half_overlap, seed=args.seed + 20,
    )
    covariance_key_bases = {
        layer: key_covariance.eigenvectors[layer, :, :key_ranks[layer]]
        for layer in range(12)
    }
    covariance_value_bases = {
        layer: value_covariance.eigenvectors[layer, :, :value_ranks[layer]]
        for layer in range(12)
    }
    hybrid_key_bases = {
        layer: task_covariance_hybrid_basis(
            key_task.matrices[layer], key_covariance.eigenvalues[layer],
            key_covariance.eigenvectors[layer], key_ranks[layer],
            task_weight=args.hybrid_task_weight,
        ) for layer in range(12)
    }
    hybrid_value_bases = {
        layer: task_covariance_hybrid_basis(
            value_task.matrices[layer], value_covariance.eigenvalues[layer],
            value_covariance.eigenvectors[layer], value_ranks[layer],
            task_weight=args.hybrid_task_weight,
        ) for layer in range(12)
    }
    empty_key = _empty_bases(key_bases)
    empty_value = _empty_bases(value_bases)

    random_controls = [dict(key={}, value={}) for _ in range(args.random_controls)]
    for layer in range(12):
        for kind, states, learned, rank in (
            ("key", fit_keys, key_bases[layer], key_ranks[layer]),
            ("value", fit_values, value_bases[layer], value_ranks[layer]),
        ):
            if not rank:
                for control in random_controls:
                    control[kind][layer] = learned.new_zeros((768, 0))
                continue
            centered = (states[:, layer] - states[:, layer].mean(0)).reshape(-1, 768)
            candidates, _ = energy_matched_random_bases(
                centered, torch.zeros(768), learned,
                controls=args.random_controls, replicates=args.random_candidates,
                seed=args.seed * 100 + layer * 7 + (kind == "value"),
            )
            for control, candidate in enumerate(candidates):
                random_controls[control][kind][layer] = candidate

    dense_losses, dense_logits, dense_targets = _evaluate(
        model, tokenizer, causal_rows, latent_positions,
        args.causal_batch_size, device,
    )
    dense_metrics = _metrics(
        dense_losses, dense_logits, dense_targets, dense_losses, dense_logits
    )
    causal = {}
    simultaneous_alpha = 0.05 / 12
    for layer in range(12):
        record = {
            "key_rank": key_ranks[layer], "value_rank": value_ranks[layer],
            "total_independent_rank": key_ranks[layer] + value_ranks[layer],
        }
        if not record["total_independent_rank"]:
            record.update({"status": "no_rank_passed_selection", "passed": False})
            causal[f"layer_{layer:02d}"] = record
            continue
        arms = {
            "task_retain": (key_bases, value_bases, "retain"),
            "task_remove": (key_bases, value_bases, "remove"),
            "covariance_remove": (
                covariance_key_bases, covariance_value_bases, "remove"
            ),
            "hybrid_remove": (hybrid_key_bases, hybrid_value_bases, "remove"),
            "key_task_remove": (key_bases, empty_value, "remove"),
            "value_task_remove": (empty_key, value_bases, "remove"),
        }
        arm_metrics = {}
        for name, (keys, values, mode) in arms.items():
            intervention = DirectLatentKVSubspaceIntervention(
                key_bases=keys, value_bases=values,
                key_means=key_covariance.means, value_means=value_covariance.means,
                layers=[layer], positions=range(latent_positions), mode=mode,
            )
            losses, logits, targets = _evaluate(
                model, tokenizer, causal_rows, latent_positions,
                args.causal_batch_size, device, intervention,
            )
            metrics = _metrics(losses, logits, targets, dense_losses, dense_logits)
            arm_metrics[name] = metrics
            record[name] = _json_metrics(metrics)

        random_deltas = []
        record["random_remove"] = []
        for control in random_controls:
            intervention = DirectLatentKVSubspaceIntervention(
                key_bases=control["key"], value_bases=control["value"],
                key_means=key_covariance.means, value_means=value_covariance.means,
                layers=[layer], positions=range(latent_positions), mode="remove",
            )
            losses, logits, targets = _evaluate(
                model, tokenizer, causal_rows, latent_positions,
                args.causal_batch_size, device, intervention,
            )
            metrics = _metrics(losses, logits, targets, dense_losses, dense_logits)
            random_deltas.append(metrics["per_example_full_answer_nll_delta"])
            record["random_remove"].append(_json_metrics(metrics))
        specificity = (
            arm_metrics["task_remove"]["per_example_full_answer_nll_delta"]
            - torch.stack(random_deltas).mean(0)
        )
        record["specificity_mean_full_answer_nll_delta"] = float(specificity.mean())
        record["familywise_95ci"] = _bootstrap_interval(
            specificity, seed=args.seed + layer, alpha=simultaneous_alpha
        )
        record["passed"] = bool(
            record["familywise_95ci"][0] > 0
            and record["task_retain"]["dense_first_token_top1_agreement"]
            >= args.minimum_retention
            and record["task_retain"]["mean_full_answer_nll_delta"]
            <= args.maximum_retain_nll_delta
        )
        record["status"] = "passed" if record["passed"] else "failed_causal_gate"
        causal[f"layer_{layer:02d}"] = record
        print(
            f"layer {layer:02d}: K={key_ranks[layer]} V={value_ranks[layer]} "
            f"status={record['status']}"
        )

    passing = [
        layer for layer in range(12) if causal[f"layer_{layer:02d}"]["passed"]
    ]
    selected_run = longest_contiguous_run(passing)
    gate = {
        "passed": len(selected_run) >= args.minimum_group_layers,
        "passing_layers": passing,
        "selected_contiguous_layers": selected_run,
        "thresholds": {
            "rank_grid": rank_grid,
            "retained_best_effect": args.retained_best_effect,
            "minimum_half_overlap": args.minimum_half_overlap,
            "causal_familywise_alpha": 0.05,
            "causal_tests": 12,
            "minimum_retention": args.minimum_retention,
            "maximum_retain_nll_delta": args.maximum_retain_nll_delta,
            "minimum_group_layers": args.minimum_group_layers,
        },
    }
    key_variance = selected_variance_fraction(
        fit_keys, key_covariance.means, key_bases
    )
    value_variance = selected_variance_fraction(
        fit_values, value_covariance.means, value_bases
    )
    layer_summary = []
    for layer in range(12):
        key_row = _rank_row(key_audit[layer], key_ranks[layer]) if key_ranks[layer] else None
        value_row = _rank_row(value_audit[layer], value_ranks[layer]) if value_ranks[layer] else None
        layer_summary.append({
            "layer": layer,
            "key_rank": key_ranks[layer], "value_rank": value_ranks[layer],
            "total_independent_rank": key_ranks[layer] + value_ranks[layer],
            "key_half_split_overlap": (
                key_row["half_split_subspace_overlap"] if key_row else None
            ),
            "value_half_split_overlap": (
                value_row["half_split_subspace_overlap"] if value_row else None
            ),
            "key_rank_selection_excess_damage": (
                key_row["mean_excess_predicted_removal_damage"] if key_row else None
            ),
            "value_rank_selection_excess_damage": (
                value_row["mean_excess_predicted_removal_damage"] if value_row else None
            ),
            "key_selected_variance_fraction": key_variance[layer],
            "value_selected_variance_fraction": value_variance[layer],
            "causal_status": causal[f"layer_{layer:02d}"]["status"],
        })

    payload = {
        "schema_version": 1, "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction, "sampling": sampling,
        "split_hashes": {
            "cache_covariance_fit": _sha([_normalized_question(row["question"]) for row in fit_rows]),
            "direct_task_discovery": _sha([_normalized_question(row["question"]) for row in discovery_rows]),
            "rank_selection": _sha([_normalized_question(row["question"]) for row in rank_rows]),
            "causal_confirmation": _sha([_normalized_question(row["question"]) for row in causal_rows]),
        },
        "latent_positions": latent_positions,
        "gradient_connectivity_fraction": connectivity,
        "key_covariance": key_covariance, "value_covariance": value_covariance,
        "key_task_system": key_task, "value_task_system": value_task,
        "key_rank_audit": key_audit, "value_rank_audit": value_audit,
        "selected_key_responses": key_bases,
        "selected_value_responses": value_bases,
        "hybrid_key_bases": hybrid_key_bases,
        "hybrid_value_bases": hybrid_value_bases,
        "key_means": key_covariance.means, "value_means": value_covariance.means,
        "key_ranks_by_layer": key_ranks, "value_ranks_by_layer": value_ranks,
        "dense": _json_metrics(dense_metrics),
        "causal_screen": causal, "gate": gate,
        "layer_direction_summary": layer_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, args.output_dir / "direct_cache_task_subspaces.pt")
    excluded = {
        "gradient_connectivity_fraction", "key_covariance", "value_covariance",
        "key_task_system", "value_task_system", "selected_key_responses",
        "selected_value_responses", "hybrid_key_bases", "hybrid_value_bases",
        "key_means", "value_means",
    }
    summary = {key: value for key, value in payload.items() if key not in excluded}
    summary["gradient_connectivity_fraction"] = connectivity.tolist()
    _atomic_json(summary, args.output_dir / "summary.json")
    print(json.dumps({
        "key_ranks": key_ranks, "value_ranks": value_ranks, "gate": gate,
    }, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--fit-examples", type=int, default=1024)
    parser.add_argument("--discovery-examples", type=int, default=1024)
    parser.add_argument("--rank-examples", type=int, default=256)
    parser.add_argument("--causal-examples", type=int, default=128)
    parser.add_argument("--fit-batch-size", type=int, default=16)
    parser.add_argument("--gradient-batch-size", type=int, default=4)
    parser.add_argument("--causal-batch-size", type=int, default=8)
    parser.add_argument("--rank-grid", default="1,2,4,8,16,28,32,48,64")
    parser.add_argument("--retained-best-effect", type=float, default=0.95)
    parser.add_argument("--minimum-half-overlap", type=float, default=0.25)
    parser.add_argument("--hybrid-task-weight", type=float, default=0.5)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=128)
    parser.add_argument("--minimum-retention", type=float, default=0.95)
    parser.add_argument("--maximum-retain-nll-delta", type=float, default=0.10)
    parser.add_argument("--minimum-group-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
