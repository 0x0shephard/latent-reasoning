"""Discover each CODI layer's own answer subspace and test it in latent KV cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_margin_states import gold_text
from scripts.collect_official_codi_endpoint_tsvc import _normalized_question, verify_full_reproduction_gate
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES, GSM8K_TRAIN_URL, _canonical_gsm8k_row,
    sample_gsm8k_train_calibration,
)
from src.data.datasets import load_eval_set
from src.data.official_codi_training import collate_official_codi_kv_rows
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.mech.direct_layerwise_kv import (
    DirectLatentKVSubspaceIntervention, LatentAttentionStateTrace,
    align_layerwise_bases, fit_layerwise_eigensystems, longest_contiguous_run,
    score_layerwise_answer_directions, select_layerwise_bases,
)
from src.mech.endpoint_margin_geometry import gold_first_token_ids
from src.mech.layerwise_u28 import energy_matched_random_bases
from src.mech.official_codi_layerwise import gpt2_qkv_response_bases
from src.mech.official_codi_target_utility import OfficialCODIAnswerScorer
from src.models.official_codi import (
    build_official_codi_gpt2, download_official_checkpoint, generate_official_codi,
    load_official_checkpoint, official_codi_base_model, resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_independent_layer_subspaces_direct_latent_kv_v1"


class FirstTokenObserver:
    def __init__(self): self.values = []
    def __call__(self, hidden, active, position):
        if int(position) == 0: self.values.append(hidden[active].detach().float().cpu())
    def stacked(self, expected):
        value = torch.cat(self.values)
        if value.shape != (expected, 768): raise RuntimeError("first-token observer coverage failed")
        return value


def _sha(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _partition(train, test_questions, total, seed):
    rows, sampling = sample_gsm8k_train_calibration(
        train, test_questions=test_questions, examples=total, seed=seed)
    return rows, sampling


def _collect_fit(model, tokenizer, rows, scorer, batch_size, device):
    state_chunks, key_chunks, value_chunks = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start:start + batch_size], bot_token_id=model.bot_id).to(device)
        trace = LatentAttentionStateTrace(model, scorer.latent_positions)
        with torch.no_grad(), trace.capture():
            output = scorer(batch, return_kv=True)
        state_chunks.append(trace.states().float().cpu())
        for name, destination in (("student_keys", key_chunks), ("student_values", value_chunks)):
            tensor = getattr(output, name)
            if tensor is None: raise RuntimeError(f"missing {name}")
            # [B,L,H,P,Dh] -> [B,L,P,768]
            destination.append(tensor.permute(0, 1, 3, 2, 4).reshape(tensor.shape[0], 12, 6, 768).float().cpu())
        del batch, output, trace
        if device.type == "cuda": torch.cuda.empty_cache()
    return torch.cat(state_chunks), torch.cat(key_chunks), torch.cat(value_chunks)


def _collect_selection(model, tokenizer, rows, scorer, batch_size, device):
    state_chunks, gradient_chunks = [], []
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start:start + batch_size], bot_token_id=model.bot_id).to(device)
        trace = LatentAttentionStateTrace(model, scorer.latent_positions)
        with trace.capture(): output = scorer(batch)
        states = trace.states()
        gradients = trace.gradients(output.mean_loss) * states.shape[0]
        state_chunks.append(states.detach().float().cpu())
        gradient_chunks.append(gradients.detach().float().cpu())
        del batch, output, trace, states, gradients
        if device.type == "cuda": torch.cuda.empty_cache()
    return torch.cat(state_chunks), torch.cat(gradient_chunks)


def _orthonormalize_responses(responses, kind):
    return torch.stack([torch.linalg.qr(responses[layer][kind].double(), mode="reduced").Q.float()
                        for layer in range(12)])


def _generate_states(model, tokenizer, rows, cfg, style, args, intervention=None):
    observer = FirstTokenObserver()
    outputs = generate_official_codi(
        model, tokenizer, [row["question"] for row in rows],
        latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=1,
        batch_size=args.causal_batch_size, device=args.resolved_device,
        kv_intervention=intervention, answer_cue=style.answer_prefix,
        force_answer_cue=True, answer_state_observer=observer)
    return outputs, observer.stacked(len(rows))


def _logit_metrics(states, dense_states, readout, gold_ids):
    logits = states.double() @ readout.double().T
    dense_logits = dense_states.double() @ readout.double().T
    gold = torch.tensor(gold_ids, dtype=torch.long)
    logp = torch.log_softmax(logits, -1); dense_logp = torch.log_softmax(dense_logits, -1)
    row = torch.arange(states.shape[0])
    nll = -logp[row, gold]; dense_nll = -dense_logp[row, gold]
    dense_p = dense_logp.exp()
    kl = (dense_p * (dense_logp - logp)).sum(-1)
    top = logits.argmax(-1); dense_top = dense_logits.argmax(-1)
    masked = logits.clone(); masked[row, gold] = -torch.inf
    dense_masked = dense_logits.clone(); dense_masked[row, gold] = -torch.inf
    margin = logits[row, gold] - masked.max(-1).values
    dense_margin = dense_logits[row, gold] - dense_masked.max(-1).values
    return {
        "accuracy": float((top == gold).float().mean()),
        "dense_top1_agreement": float((top == dense_top).float().mean()),
        "mean_kl_from_dense": float(kl.mean()),
        "mean_gold_nll_delta": float((nll - dense_nll).mean()),
        "mean_gold_margin_delta": float((margin - dense_margin).mean()),
        "per_example_gold_nll_delta": (nll - dense_nll).float(),
    }


def _bootstrap_ci(values, seed, samples=4000):
    values = values.double(); generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(samples):
        index = torch.randint(values.numel(), (values.numel(),), generator=generator)
        means.append(values[index].mean())
    means = torch.stack(means)
    return [float(torch.quantile(means, .025)), float(torch.quantile(means, .975))]


def run(args):
    cfg = load_config(args.config); reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device); args.resolved_device = device
    dtype = resolve_torch_dtype(args.precision, device); token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token)
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
        dtype=dtype, settings=cfg.model, token=token)
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype).eval()

    from datasets import load_dataset
    train = load_dataset("json", data_files={"train": GSM8K_TRAIN_URL}, split="train", verification_mode="no_checks")
    if len(train) != GSM8K_EXPECTED_TRAIN_EXAMPLES: raise RuntimeError("GSM8K train count drifted")
    data_cfg = load_config(str(cfg.endpoint_retention.data_config)); style = PromptStyle.from_config(data_cfg.prompt)
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    total = args.fit_examples + args.select_examples + args.causal_examples
    rows, sampling = _partition(train, {_normalized_question(x["question"]) for x in test}, total, args.seed)
    fit_rows = rows[:args.fit_examples]
    select_rows = rows[args.fit_examples:args.fit_examples + args.select_examples]
    causal_rows = rows[-args.causal_examples:]
    scorer = OfficialCODIAnswerScorer(model, latent_positions=int(cfg.eval.latent_iterations))

    fit_states, fit_keys, fit_values = _collect_fit(
        model, tokenizer, fit_rows, scorer, args.fit_batch_size, device)
    eigensystem = fit_layerwise_eigensystems(fit_states)
    select_states, select_gradients = _collect_selection(
        model, tokenizer, select_rows, scorer, args.selection_batch_size, device)
    scores = score_layerwise_answer_directions(
        select_states, select_gradients, eigensystem, seed=args.seed + 1)
    bases, indices = select_layerwise_bases(
        eigensystem, scores["split_stable_z"], rank=args.rank)
    aligned_bases, rotations = align_layerwise_bases(
        fit_states, eigensystem.means, bases, reference_layer=11)
    responses = gpt2_qkv_response_bases(
        model, {layer: aligned_bases[layer] for layer in range(12)})
    key_bases = _orthonormalize_responses(responses, "key")
    value_bases = _orthonormalize_responses(responses, "value")
    key_means, value_means = fit_keys.double().mean(0).float(), fit_values.double().mean(0).float()

    random_controls = []
    for control in range(args.random_controls): random_controls.append({"key": [], "value": []})
    for layer in range(12):
        for kind, values, learned in (("key", fit_keys, key_bases), ("value", fit_values, value_bases)):
            centered = (values[:, layer] - values[:, layer].mean(0)).reshape(-1, 768)
            candidates, _ = energy_matched_random_bases(
                centered, torch.zeros(768), learned[layer], controls=args.random_controls,
                replicates=args.random_candidates, seed=args.seed * 100 + layer * 7 + (kind == "value"))
            for control, candidate in enumerate(candidates): random_controls[control][kind].append(candidate)
    for control in random_controls:
        control["key"] = torch.stack(control["key"]); control["value"] = torch.stack(control["value"])

    readout = official_codi_base_model(model).get_output_embeddings().weight[:model.eot_id].detach().float().cpu()
    gold_ids = gold_first_token_ids(tokenizer, (gold_text(row["answer"]) for row in causal_rows))
    dense_outputs, dense_states = _generate_states(model, tokenizer, causal_rows, cfg, style, args)
    dense_metrics = _logit_metrics(dense_states, dense_states, readout, gold_ids)
    causal = {}
    for layer in range(12):
        record = {}
        learned_deltas = None
        for mode in ("retain", "remove"):
            intervention = DirectLatentKVSubspaceIntervention(
                key_bases=key_bases, value_bases=value_bases,
                key_means=key_means, value_means=value_means,
                layers=[layer], positions=range(6), mode=mode)
            _, states = _generate_states(model, tokenizer, causal_rows, cfg, style, args, intervention)
            metrics = _logit_metrics(states, dense_states, readout, gold_ids)
            if mode == "remove": learned_deltas = metrics["per_example_gold_nll_delta"]
            record[mode] = {key: value for key, value in metrics.items() if not key.startswith("per_example")}
        random_deltas = []
        record["random_remove"] = []
        for control in random_controls:
            intervention = DirectLatentKVSubspaceIntervention(
                key_bases=control["key"], value_bases=control["value"],
                key_means=key_means, value_means=value_means,
                layers=[layer], positions=range(6), mode="remove")
            _, states = _generate_states(model, tokenizer, causal_rows, cfg, style, args, intervention)
            metrics = _logit_metrics(states, dense_states, readout, gold_ids)
            random_deltas.append(metrics["per_example_gold_nll_delta"])
            record["random_remove"].append({key: value for key, value in metrics.items() if not key.startswith("per_example")})
        specificity = learned_deltas - torch.stack(random_deltas).mean(0)
        record["specificity_mean_nll_delta"] = float(specificity.mean())
        record["specificity_bootstrap_95ci"] = _bootstrap_ci(specificity, args.seed + layer)
        record["stable_positive_direction_count"] = int(
            ((scores["split_stable_z"][layer] >= args.minimum_split_z) &
             scores["positive_both_splits"][layer]).sum())
        causal[f"layer_{layer:02d}"] = record
        print(layer, record["specificity_mean_nll_delta"], record["specificity_bootstrap_95ci"])

    passing = [layer for layer in range(12) if
               causal[f"layer_{layer:02d}"]["stable_positive_direction_count"] >= args.rank and
               causal[f"layer_{layer:02d}"]["specificity_bootstrap_95ci"][0] > 0 and
               causal[f"layer_{layer:02d}"]["retain"]["dense_top1_agreement"] >= args.minimum_retention]
    selected_run = longest_contiguous_run(passing)
    gate = {"passed": len(selected_run) >= args.minimum_group_layers,
            "passing_layers": passing, "selected_contiguous_layers": selected_run,
            "thresholds": {"rank": args.rank, "minimum_split_z": args.minimum_split_z,
                           "minimum_retention": args.minimum_retention,
                           "minimum_group_layers": args.minimum_group_layers}}
    payload = {
        "schema_version": 1, "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
        "sampling": sampling, "split_hashes": {
            "fit": _sha([_normalized_question(x["question"]) for x in fit_rows]),
            "selection": _sha([_normalized_question(x["question"]) for x in select_rows]),
            "causal": _sha([_normalized_question(x["question"]) for x in causal_rows])},
        "rank": args.rank, "latent_positions": 6,
        "layer_means": eigensystem.means, "eigenvalues": eigensystem.eigenvalues,
        "selected_pc_indices": indices, "selected_bases": bases,
        "aligned_bases": aligned_bases, "alignment_rotations": rotations,
        "direction_scores": scores, "key_means": key_means, "value_means": value_means,
        "key_bases": key_bases, "value_bases": value_bases,
        "aligned_key_responses": torch.stack([responses[layer]["key"] for layer in range(12)]),
        "aligned_value_responses": torch.stack([responses[layer]["value"] for layer in range(12)]),
        "dense_first_token": {key: value for key, value in dense_metrics.items() if not key.startswith("per_example")},
        "causal_screen": causal, "gate": gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, args.output_dir / "direct_layerwise_kv.pt")
    summary = {key: value for key, value in payload.items() if key not in {
        "layer_means", "eigenvalues", "selected_pc_indices", "selected_bases",
        "aligned_bases", "alignment_rotations", "direction_scores", "key_means",
        "value_means", "key_bases", "value_bases", "aligned_key_responses",
        "aligned_value_responses"}}
    summary["layer_direction_summary"] = [{
        "layer": layer,
        "selected_pc_indices": indices[layer].tolist(),
        "stable_positive_direction_count": causal[f"layer_{layer:02d}"]["stable_positive_direction_count"],
        "selected_variance_fraction": float(eigensystem.eigenvalues[layer, indices[layer]].sum() /
                                            eigensystem.eigenvalues[layer].sum())}
        for layer in range(12)]
    _atomic_json(summary, args.output_dir / "summary.json")
    print(json.dumps({"dense": summary["dense_first_token"], "gate": gate}, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--fit-examples", type=int, default=1024)
    parser.add_argument("--select-examples", type=int, default=512)
    parser.add_argument("--causal-examples", type=int, default=64)
    parser.add_argument("--fit-batch-size", type=int, default=16)
    parser.add_argument("--selection-batch-size", type=int, default=4)
    parser.add_argument("--causal-batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=28)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=128)
    parser.add_argument("--minimum-split-z", type=float, default=1.645)
    parser.add_argument("--minimum-retention", type=float, default=.80)
    parser.add_argument("--minimum-group-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); run(args); return 0


if __name__ == "__main__": raise SystemExit(main())
