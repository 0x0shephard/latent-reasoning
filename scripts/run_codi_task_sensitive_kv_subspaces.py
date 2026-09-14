"""Learn combinations of CODI layer directions that causally influence answer loss."""
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
    _bootstrap_ci, _collect_fit, _collect_selection, _evaluate,
    _json_metrics, _metrics, _sha,
)
from src.data.datasets import load_eval_set
from src.eval.official_codi import select_device
from src.mech.direct_layerwise_kv import (
    DirectLatentKVSubspaceIntervention, fit_layerwise_eigensystems,
    longest_contiguous_run,
)
from src.mech.endpoint_margin_geometry import _orthonormalize
from src.mech.layerwise_u28 import energy_matched_random_bases
from src.mech.official_codi_layerwise import gpt2_qkv_response_bases
from src.mech.official_codi_target_utility import OfficialCODIAnswerScorer
from src.mech.task_sensitive_kv_subspace import (
    backproject_kv_gradients, fit_task_sensitive_eigensystem,
    select_task_sensitive_ranks,
)
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint,
    load_official_checkpoint, resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_task_sensitive_variable_rank_kv_v1"


def _mapped_bases(response, hidden_bases):
    result = {}
    for layer, basis in hidden_bases.items():
        mapped = response[layer] @ basis
        result[layer] = (
            _orthonormalize(mapped) if basis.shape[1]
            else mapped.new_zeros((mapped.shape[0], 0))
        )
    return result


def _zero_like_bases(bases):
    return {layer: torch.zeros_like(basis) for layer, basis in bases.items()}


def _selected_variance_fraction(states, means, basis, layer):
    if basis.shape[1] == 0:
        return 0.0
    centered = states[:, layer].double() - means[layer].double()
    numerator = (centered @ basis.double()).square().sum()
    denominator = centered.square().sum().clamp_min(1e-12)
    return float(numerator / denominator)


def _rank_row(audit, rank):
    return next(
        (row for row in audit["rank_grid"] if row["rank"] == rank), None
    )


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

    fit_states, fit_keys, fit_values = _collect_fit(
        model, tokenizer, fit_rows, scorer, args.fit_batch_size, device
    )
    covariance = fit_layerwise_eigensystems(fit_states)
    discovery_states, discovery_key_gradients, discovery_value_gradients, discovery_connectivity = _collect_selection(
        model, tokenizer, discovery_rows, latent_positions,
        args.gradient_batch_size, device,
    )
    rank_states, rank_key_gradients, rank_value_gradients, rank_connectivity = _collect_selection(
        model, tokenizer, rank_rows, latent_positions,
        args.gradient_batch_size, device,
    )
    connectivity_fraction = torch.cat(
        (discovery_connectivity, rank_connectivity), dim=0
    ).float().mean(0)
    if not bool((connectivity_fraction == 1).all()):
        raise RuntimeError("exact pre-answer K/V gradients are not fully connected")

    identity = torch.eye(768)
    standard_response = gpt2_qkv_response_bases(
        model, {layer: identity for layer in range(12)}
    )
    key_response = torch.stack([
        standard_response[layer]["key"] for layer in range(12)
    ])
    value_response = torch.stack([
        standard_response[layer]["value"] for layer in range(12)
    ])
    discovery_hidden_gradients = {
        mode: backproject_kv_gradients(
            discovery_key_gradients, discovery_value_gradients,
            key_response, value_response, mode=mode,
        )
        for mode in ("joint", "key", "value")
    }
    rank_hidden_gradients = {
        mode: backproject_kv_gradients(
            rank_key_gradients, rank_value_gradients,
            key_response, value_response, mode=mode,
        )
        for mode in ("joint", "key", "value")
    }
    task_systems = {
        mode: fit_task_sensitive_eigensystem(
            discovery_states, discovery_hidden_gradients[mode], covariance.means
        )
        for mode in ("joint", "key", "value")
    }
    rank_grid = [int(value) for value in args.rank_grid.split(",")]
    selected_hidden, ranks, rank_audit = select_task_sensitive_ranks(
        task_systems["joint"], rank_states, rank_hidden_gradients["joint"],
        covariance.means, rank_grid=rank_grid,
        retained_positive_spectrum=args.retained_positive_spectrum,
        minimum_half_overlap=args.minimum_half_overlap,
        seed=args.seed + 10,
    )
    diagnostic_hidden = {
        mode: {
            layer: task_systems[mode].eigenvectors[layer, :, :ranks[layer]]
            for layer in range(12)
        }
        for mode in ("key", "value")
    }
    covariance_hidden = {
        layer: covariance.eigenvectors[layer, :, :ranks[layer]]
        for layer in range(12)
    }
    selected_key = _mapped_bases(key_response, selected_hidden)
    selected_value = _mapped_bases(value_response, selected_hidden)
    covariance_key = _mapped_bases(key_response, covariance_hidden)
    covariance_value = _mapped_bases(value_response, covariance_hidden)
    key_only = _mapped_bases(key_response, diagnostic_hidden["key"])
    value_only = _mapped_bases(value_response, diagnostic_hidden["value"])
    zero_key = _zero_like_bases(value_only)
    zero_value = _zero_like_bases(key_only)

    key_means = fit_keys.double().mean(0).float()
    value_means = fit_values.double().mean(0).float()
    random_controls = [dict(key={}, value={}) for _ in range(args.random_controls)]
    for layer, rank in enumerate(ranks):
        if not rank:
            continue
        for kind, values, learned in (
            ("key", fit_keys, selected_key[layer]),
            ("value", fit_values, selected_value[layer]),
        ):
            centered = (values[:, layer] - values[:, layer].mean(0)).reshape(-1, 768)
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
    for layer, rank in enumerate(ranks):
        record = {"selected_rank": rank}
        if not rank:
            record.update({"status": "no_rank_passed_selection", "passed": False})
            causal[f"layer_{layer:02d}"] = record
            continue
        arms = {
            "joint_retain": (selected_key, selected_value, "retain"),
            "joint_remove": (selected_key, selected_value, "remove"),
            "covariance_remove": (covariance_key, covariance_value, "remove"),
            "key_task_remove": (key_only, zero_value, "remove"),
            "value_task_remove": (zero_key, value_only, "remove"),
        }
        metrics_by_arm = {}
        for name, (key_bases, value_bases, mode) in arms.items():
            intervention = DirectLatentKVSubspaceIntervention(
                key_bases=key_bases, value_bases=value_bases,
                key_means=key_means, value_means=value_means,
                layers=[layer], positions=range(latent_positions), mode=mode,
            )
            losses, logits, targets = _evaluate(
                model, tokenizer, causal_rows, latent_positions,
                args.causal_batch_size, device, intervention,
            )
            metrics = _metrics(losses, logits, targets, dense_losses, dense_logits)
            metrics_by_arm[name] = metrics
            record[name] = _json_metrics(metrics)

        random_deltas = []
        record["random_remove"] = []
        for control in random_controls:
            intervention = DirectLatentKVSubspaceIntervention(
                key_bases=control["key"], value_bases=control["value"],
                key_means=key_means, value_means=value_means,
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
            metrics_by_arm["joint_remove"]["per_example_full_answer_nll_delta"]
            - torch.stack(random_deltas).mean(0)
        )
        record["specificity_mean_full_answer_nll_delta"] = float(specificity.mean())
        record["specificity_bootstrap_95ci"] = _bootstrap_ci(
            specificity, args.seed + layer
        )
        record["passed"] = bool(
            record["specificity_bootstrap_95ci"][0] > 0
            and record["joint_retain"]["dense_first_token_top1_agreement"]
            >= args.minimum_retention
            and record["joint_retain"]["mean_full_answer_nll_delta"]
            <= args.maximum_retain_nll_delta
        )
        record["status"] = "passed" if record["passed"] else "failed_causal_gate"
        causal[f"layer_{layer:02d}"] = record
        print(f"layer {layer:02d}: rank={rank} status={record['status']}")

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
            "retained_positive_spectrum": args.retained_positive_spectrum,
            "minimum_half_overlap": args.minimum_half_overlap,
            "minimum_retention": args.minimum_retention,
            "maximum_retain_nll_delta": args.maximum_retain_nll_delta,
            "minimum_group_layers": args.minimum_group_layers,
        },
    }
    layer_summary = []
    for layer, rank in enumerate(ranks):
        selected_audit = _rank_row(rank_audit[layer], rank) if rank else None
        layer_summary.append({
            "layer": layer,
            "positive_joint_task_eigenvalues": rank_audit[layer]["positive_task_eigenvalues"],
            "top_joint_task_eigenvalues": task_systems["joint"].eigenvalues[
                layer, : max(rank_grid)
            ].tolist(),
            "selected_rank": rank,
            "half_split_subspace_overlap": (
                selected_audit["half_split_subspace_overlap"] if selected_audit else None
            ),
            "rank_selection_excess_damage": (
                selected_audit["mean_excess_predicted_removal_damage"] if selected_audit else None
            ),
            "selected_variance_fraction": _selected_variance_fraction(
                fit_states, covariance.means, selected_hidden[layer], layer
            ),
            "causal_status": causal[f"layer_{layer:02d}"]["status"],
        })

    payload = {
        "schema_version": 1, "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction, "sampling": sampling,
        "split_hashes": {
            "covariance_fit": _sha([_normalized_question(row["question"]) for row in fit_rows]),
            "task_discovery": _sha([_normalized_question(row["question"]) for row in discovery_rows]),
            "rank_selection": _sha([_normalized_question(row["question"]) for row in rank_rows]),
            "causal_confirmation": _sha([_normalized_question(row["question"]) for row in causal_rows]),
        },
        "latent_positions": latent_positions,
        "gradient_connectivity_fraction": connectivity_fraction,
        "covariance_means": covariance.means,
        "covariance_eigenvalues": covariance.eigenvalues,
        "covariance_eigenvectors": covariance.eigenvectors,
        "task_systems": task_systems,
        "rank_audit": rank_audit,
        "selected_hidden_bases": selected_hidden,
        "selected_key_responses": selected_key,
        "selected_value_responses": selected_value,
        "key_means": key_means, "value_means": value_means,
        "ranks_by_layer": ranks,
        "dense": _json_metrics(dense_metrics),
        "causal_screen": causal, "gate": gate,
        "layer_direction_summary": layer_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, args.output_dir / "task_sensitive_kv_subspaces.pt")
    excluded = {
        "gradient_connectivity_fraction", "covariance_means",
        "covariance_eigenvalues", "covariance_eigenvectors", "task_systems",
        "selected_hidden_bases", "selected_key_responses", "selected_value_responses",
        "key_means", "value_means",
    }
    summary = {key: value for key, value in payload.items() if key not in excluded}
    summary["gradient_connectivity_fraction"] = connectivity_fraction.tolist()
    _atomic_json(summary, args.output_dir / "summary.json")
    print(json.dumps({"ranks_by_layer": ranks, "gate": gate}, indent=2))
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
    parser.add_argument("--retained-positive-spectrum", type=float, default=0.95)
    parser.add_argument("--minimum-half-overlap", type=float, default=0.25)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=128)
    parser.add_argument("--minimum-retention", type=float, default=0.95)
    parser.add_argument("--maximum-retain-nll-delta", type=float, default=0.10)
    parser.add_argument("--minimum-group-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
