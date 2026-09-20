"""Discover variable-rank CODI layer subspaces using exact pre-answer KV gradients."""
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
from src.data.datasets import load_eval_set
from src.data.official_codi_training import collate_official_codi_kv_rows
from src.eval.official_codi import select_device
from src.mech.direct_layerwise_kv import (
    DirectLatentKVSubspaceIntervention, LatentAttentionStateTrace,
    fit_layerwise_eigensystems, longest_contiguous_run,
)
from src.mech.endpoint_margin_geometry import _orthonormalize
from src.mech.layerwise_u28 import energy_matched_random_bases
from src.mech.official_codi_layerwise import gpt2_qkv_response_bases
from src.mech.official_codi_target_utility import OfficialCODIAnswerScorer
from src.mech.preanswer_kv_subspace import (
    official_codi_preanswer_kv_forward, score_preanswer_kv_directions,
    select_variable_layerwise_bases,
)
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint,
    load_official_checkpoint, resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_preanswer_gradient_variable_rank_kv_v2"


def _sha(values):
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _collect_fit(model, tokenizer, rows, scorer, batch_size, device):
    state_chunks, key_chunks, value_chunks = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start : start + batch_size], bot_token_id=model.bot_id
        ).to(device)
        trace = LatentAttentionStateTrace(model, scorer.latent_positions)
        with torch.no_grad(), trace.capture():
            output = scorer(batch, return_kv=True)
        state_chunks.append(trace.states().float().cpu())
        for tensor, destination in (
            (output.student_keys, key_chunks), (output.student_values, value_chunks)
        ):
            if tensor is None:
                raise RuntimeError("fit pass did not return the latent cache")
            destination.append(
                tensor.permute(0, 1, 3, 2, 4).reshape(
                    tensor.shape[0], 12, scorer.latent_positions, 768
                ).float().cpu()
            )
        del batch, output, trace
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(state_chunks), torch.cat(key_chunks), torch.cat(value_chunks)


def _collect_selection(model, tokenizer, rows, latent_positions, batch_size, device):
    states, key_gradients, value_gradients, connectivity = [], [], [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start : start + batch_size], bot_token_id=model.bot_id
        ).to(device)
        trace = LatentAttentionStateTrace(model, latent_positions)
        with trace.capture():
            output = official_codi_preanswer_kv_forward(
                model, batch, latent_positions=latent_positions, return_gradients=True
            )
        states.append(trace.states().detach().float().cpu())
        key_gradients.append(output.key_gradients.detach().float().cpu())
        value_gradients.append(output.value_gradients.detach().float().cpu())
        connectivity.append(output.gradient_connected)
        del batch, trace, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        torch.cat(states), torch.cat(key_gradients), torch.cat(value_gradients),
        torch.stack(connectivity),
    )


def _evaluate(
    model,
    tokenizer,
    rows,
    latent_positions,
    batch_size,
    device,
    intervention=None,
    *,
    enforce_answer_eligibility=True,
):
    losses, logits, targets = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = collate_official_codi_kv_rows(
                tokenizer,
                rows[start : start + batch_size],
                bot_token_id=model.bot_id,
                enforce_answer_eligibility=enforce_answer_eligibility,
            ).to(device)
            output = official_codi_preanswer_kv_forward(
                model, batch, latent_positions=latent_positions,
                kv_intervention=intervention,
            )
            losses.append(output.per_example_loss.float().cpu())
            logits.append(output.first_logits.float().cpu())
            targets.append(output.first_targets.cpu())
    return torch.cat(losses), torch.cat(logits), torch.cat(targets)


def _metrics(losses, logits, targets, dense_losses, dense_logits):
    rows = torch.arange(logits.shape[0])
    logp = torch.log_softmax(logits.double(), -1)
    dense_logp = torch.log_softmax(dense_logits.double(), -1)
    dense_probability = dense_logp.exp()
    predictions, dense_predictions = logits.argmax(-1), dense_logits.argmax(-1)
    masked = logits.double().clone(); masked[rows, targets] = -torch.inf
    dense_masked = dense_logits.double().clone(); dense_masked[rows, targets] = -torch.inf
    margin = logits[rows, targets].double() - masked.max(-1).values
    dense_margin = dense_logits[rows, targets].double() - dense_masked.max(-1).values
    return {
        "mean_full_answer_nll": float(losses.mean()),
        "mean_full_answer_nll_delta": float((losses - dense_losses).mean()),
        "first_token_accuracy": float((predictions == targets).float().mean()),
        "dense_first_token_top1_agreement": float(
            (predictions == dense_predictions).float().mean()
        ),
        "mean_first_token_kl_from_dense": float(
            (dense_probability * (dense_logp - logp)).sum(-1).mean()
        ),
        "mean_gold_margin_delta": float((margin - dense_margin).mean()),
        "per_example_full_answer_nll_delta": losses - dense_losses,
    }


def _bootstrap_ci(values, seed, samples=4000):
    values = values.double(); generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(samples):
        index = torch.randint(values.numel(), (values.numel(),), generator=generator)
        means.append(values[index].mean())
    means = torch.stack(means)
    return [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))]


def _json_metrics(metrics):
    return {key: value for key, value in metrics.items() if not key.startswith("per_example")}


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
        args.fit_examples + args.select_examples
        + args.rank_examples + args.causal_examples
    )
    rows, sampling = sample_gsm8k_train_calibration(
        train,
        test_questions={_normalized_question(row["question"]) for row in test},
        examples=total,
        seed=args.seed,
    )
    fit_rows = rows[: args.fit_examples]
    select_rows = rows[args.fit_examples : args.fit_examples + args.select_examples]
    rank_start = args.fit_examples + args.select_examples
    rank_rows = rows[rank_start : rank_start + args.rank_examples]
    causal_rows = rows[-args.causal_examples :]
    latent_positions = int(cfg.eval.latent_iterations)
    scorer = OfficialCODIAnswerScorer(model, latent_positions=latent_positions)

    fit_states, fit_keys, fit_values = _collect_fit(
        model, tokenizer, fit_rows, scorer, args.fit_batch_size, device
    )
    eigensystem = fit_layerwise_eigensystems(fit_states)
    select_states, key_gradients, value_gradients, select_connectivity = _collect_selection(
        model, tokenizer, select_rows, latent_positions,
        args.selection_batch_size, device,
    )
    rank_states, rank_key_gradients, rank_value_gradients, rank_connectivity = _collect_selection(
        model, tokenizer, rank_rows, latent_positions,
        args.selection_batch_size, device,
    )
    connectivity_fraction = torch.cat(
        (select_connectivity, rank_connectivity), dim=0
    ).float().mean(0)
    if not bool((connectivity_fraction == 1).all()):
        raise RuntimeError(
            "pre-answer KV gradient connectivity gate failed; every layer and both "
            "K/V tensors must be connected before direction discovery"
        )

    full_responses = gpt2_qkv_response_bases(
        model, {layer: eigensystem.eigenvectors[layer] for layer in range(12)}
    )
    full_key_responses = torch.stack(
        [full_responses[layer]["key"] for layer in range(12)]
    )
    full_value_responses = torch.stack(
        [full_responses[layer]["value"] for layer in range(12)]
    )
    scores = score_preanswer_kv_directions(
        select_states, key_gradients, value_gradients, eigensystem,
        full_key_responses, full_value_responses, seed=args.seed + 1,
    )
    rank_scores = score_preanswer_kv_directions(
        rank_states, rank_key_gradients, rank_value_gradients, eigensystem,
        full_key_responses, full_value_responses, seed=args.seed + 2,
    )
    hidden_bases, selected_indices, ranks = select_variable_layerwise_bases(
        eigensystem, scores, maximum_rank=args.maximum_rank,
        minimum_split_z=args.minimum_split_z, fdr_q=args.fdr_q,
        retained_effect_fraction=args.retained_effect_fraction,
        validation_scores=rank_scores,
    )
    selected_key_responses, selected_value_responses = {}, {}
    for layer in range(12):
        index = selected_indices[layer]
        selected_key_responses[layer] = _orthonormalize(
            full_key_responses[layer].index_select(1, index)
        ) if ranks[layer] else full_key_responses.new_zeros((768, 0))
        selected_value_responses[layer] = _orthonormalize(
            full_value_responses[layer].index_select(1, index)
        ) if ranks[layer] else full_value_responses.new_zeros((768, 0))

    key_means = fit_keys.double().mean(0).float()
    value_means = fit_values.double().mean(0).float()
    random_controls = [dict(key={}, value={}) for _ in range(args.random_controls)]
    for layer, rank in enumerate(ranks):
        if not rank:
            continue
        for kind, values, learned in (
            ("key", fit_keys, selected_key_responses[layer]),
            ("value", fit_values, selected_value_responses[layer]),
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
        record = {"operational_rank": rank}
        if not rank:
            record.update({"status": "no_validated_directions", "passed": False})
            causal[f"layer_{layer:02d}"] = record
            continue
        learned_remove_delta = None
        for mode in ("retain", "remove"):
            intervention = DirectLatentKVSubspaceIntervention(
                key_bases=selected_key_responses,
                value_bases=selected_value_responses,
                key_means=key_means, value_means=value_means,
                layers=[layer], positions=range(latent_positions), mode=mode,
            )
            losses, logits, targets = _evaluate(
                model, tokenizer, causal_rows, latent_positions,
                args.causal_batch_size, device, intervention,
            )
            metrics = _metrics(
                losses, logits, targets, dense_losses, dense_logits
            )
            if mode == "remove":
                learned_remove_delta = metrics["per_example_full_answer_nll_delta"]
            record[mode] = _json_metrics(metrics)

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
            metrics = _metrics(
                losses, logits, targets, dense_losses, dense_logits
            )
            random_deltas.append(metrics["per_example_full_answer_nll_delta"])
            record["random_remove"].append(_json_metrics(metrics))
        specificity = learned_remove_delta - torch.stack(random_deltas).mean(0)
        record["specificity_mean_full_answer_nll_delta"] = float(specificity.mean())
        record["specificity_bootstrap_95ci"] = _bootstrap_ci(
            specificity, args.seed + layer
        )
        record["passed"] = bool(
            record["specificity_bootstrap_95ci"][0] > 0
            and record["retain"]["dense_first_token_top1_agreement"]
            >= args.minimum_retention
            and record["retain"]["mean_full_answer_nll_delta"]
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
            "fdr_q": args.fdr_q,
            "minimum_split_z": args.minimum_split_z,
            "maximum_rank": args.maximum_rank,
            "retained_effect_fraction": args.retained_effect_fraction,
            "minimum_retention": args.minimum_retention,
            "maximum_retain_nll_delta": args.maximum_retain_nll_delta,
            "minimum_group_layers": args.minimum_group_layers,
        },
    }
    layer_summary = []
    for layer in range(12):
        index = selected_indices[layer]
        variance = (
            float(eigensystem.eigenvalues[layer, index].sum()
                  / eigensystem.eigenvalues[layer].sum())
            if index.numel() else 0.0
        )
        statistically_eligible = int((
            scores["positive_both_splits"][layer]
            & (scores["split_stable_z"][layer] >= args.minimum_split_z)
            & (scores["q_values"][layer] <= args.fdr_q)
        ).sum())
        layer_summary.append({
            "layer": layer,
            "statistically_validated_direction_count": statistically_eligible,
            "operational_rank": ranks[layer],
            "selected_pc_indices": index.tolist(),
            "selected_variance_fraction": variance,
            "causal_status": causal[f"layer_{layer:02d}"]["status"],
        })

    payload = {
        "schema_version": 2, "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction, "sampling": sampling,
        "split_hashes": {
            "fit": _sha([_normalized_question(row["question"]) for row in fit_rows]),
            "selection": _sha([_normalized_question(row["question"]) for row in select_rows]),
            "rank_selection": _sha([_normalized_question(row["question"]) for row in rank_rows]),
            "causal": _sha([_normalized_question(row["question"]) for row in causal_rows]),
        },
        "latent_positions": latent_positions,
        "gradient_connectivity_fraction": connectivity_fraction,
        "layer_means": eigensystem.means,
        "eigenvalues": eigensystem.eigenvalues,
        "eigenvectors": eigensystem.eigenvectors,
        "direction_scores": scores,
        "rank_selection_scores": rank_scores,
        "selected_pc_indices": selected_indices,
        "selected_hidden_bases": hidden_bases,
        "selected_key_responses": selected_key_responses,
        "selected_value_responses": selected_value_responses,
        "key_means": key_means, "value_means": value_means,
        "ranks_by_layer": ranks,
        "dense": _json_metrics(dense_metrics),
        "causal_screen": causal, "gate": gate,
        "layer_direction_summary": layer_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, args.output_dir / "preanswer_kv_subspaces.pt")
    excluded = {
        "layer_means", "eigenvalues", "eigenvectors", "direction_scores",
        "rank_selection_scores",
        "selected_pc_indices", "selected_hidden_bases", "selected_key_responses",
        "selected_value_responses", "key_means", "value_means",
        "gradient_connectivity_fraction",
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
    parser.add_argument("--select-examples", type=int, default=1024)
    parser.add_argument("--rank-examples", type=int, default=256)
    parser.add_argument("--causal-examples", type=int, default=128)
    parser.add_argument("--fit-batch-size", type=int, default=16)
    parser.add_argument("--selection-batch-size", type=int, default=4)
    parser.add_argument("--causal-batch-size", type=int, default=8)
    parser.add_argument("--maximum-rank", type=int, default=64)
    parser.add_argument("--minimum-split-z", type=float, default=1.645)
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--retained-effect-fraction", type=float, default=0.95)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=128)
    parser.add_argument("--minimum-retention", type=float, default=0.95)
    parser.add_argument("--maximum-retain-nll-delta", type=float, default=0.10)
    parser.add_argument("--minimum-group-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
