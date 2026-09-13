"""Fit, validate, and causally screen layer-local transports of CODI's U28."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_margin_states import gold_text
from scripts.collect_official_codi_endpoint_tsvc import _normalized_question, verify_full_reproduction_gate
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES,
    GSM8K_TRAIN_URL,
    _canonical_gsm8k_row,
    sample_gsm8k_train_calibration,
)
from src.data.datasets import load_eval_set
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.mech.endpoint_margin_geometry import build_band_subspace, gold_first_token_ids, state_covariance
from src.mech.layerwise_u28 import (
    coefficient_r2,
    energy_matched_random_bases,
    fit_ridge_transport,
    principal_angle_cosines,
    subspace_overlap,
)
from src.mech.official_codi_layerwise import (
    OfficialCODILayerwiseEndpointCollector,
    OfficialCODILayerwiseEndpointIntervention,
    attention_location_names,
    residual_location_names,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


SCHEMA = 1
CONTRACT = "official_codi_layerwise_transport_of_final_u28_v1"


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _eligible_remaining(train, excluded: set[str], seed: int) -> list[dict]:
    rows = []
    seen = set()
    for raw in train:
        row = _canonical_gsm8k_row(dict(raw))
        if row is None:
            continue
        key = _normalized_question(row["question"])
        if key in excluded or key in seen:
            continue
        seen.add(key)
        rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows


def _collect(model, tokenizer, questions, cfg, style, args) -> dict[str, torch.Tensor]:
    locations = (*residual_location_names(), *attention_location_names())
    collector = OfficialCODILayerwiseEndpointCollector(model, locations=locations)
    generate_official_codi(
        model, tokenizer, list(questions),
        latent_iterations=int(cfg.eval.latent_iterations), max_new_tokens=1,
        batch_size=args.batch_size, device=args.resolved_device,
        answer_endpoint_intervention=collector, answer_cue=style.answer_prefix,
        force_answer_cue=True,
    )
    return collector.stacked(len(questions))


def _first_token_metrics(outputs, gold_ids, tokenizer, baseline=None) -> dict:
    gold_texts = [tokenizer.decode([int(value)], skip_special_tokens=True) for value in gold_ids]
    correct = [a == b for a, b in zip(outputs, gold_texts)]
    result = {"examples": len(outputs), "accuracy": sum(correct) / len(correct)}
    if baseline is not None:
        result["dense_agreement"] = sum(a == b for a, b in zip(outputs, baseline)) / len(outputs)
    return result


def run(args) -> dict:
    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
    args.resolved_device = device
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
        dtype=dtype, settings=cfg.model, token=token,
    )
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cached = torch.load(args.colon_states, map_location="cpu", weights_only=False)
    state_order = list(cached["state_order"])
    final_index = state_order.index(12)
    final_calibration = cached["calibration_states"][:, final_index, :].float()
    final_centre = final_calibration.mean(0)
    final_covariance = state_covariance(final_calibration - final_centre)
    u28 = build_band_subspace(covariance=final_covariance, start=4, stop=32).basis

    from datasets import load_dataset
    train = load_dataset("json", data_files={"train": GSM8K_TRAIN_URL}, split="train", verification_mode="no_checks")
    if len(train) != GSM8K_EXPECTED_TRAIN_EXAMPLES:
        raise RuntimeError("GSM8K train count drifted")
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    style = PromptStyle.from_config(data_cfg.prompt)
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    test_questions = {_normalized_question(row["question"]) for row in test}
    original_calibration, calibration_sampling = sample_gsm8k_train_calibration(
        train, test_questions=test_questions, examples=final_calibration.shape[0], seed=args.calibration_seed,
    )
    excluded = {_normalized_question(row["question"]) for row in original_calibration}
    remaining = _eligible_remaining(train, excluded, args.partition_seed)
    needed = args.fit_examples + args.select_examples
    if len(remaining) < needed:
        raise RuntimeError(f"need {needed} disjoint eligible examples, found {len(remaining)}")
    fit_rows = remaining[: args.fit_examples]
    select_rows = remaining[args.fit_examples : needed]
    assert set(map(lambda x: _normalized_question(x["question"]), fit_rows)).isdisjoint(
        set(map(lambda x: _normalized_question(x["question"]), select_rows))
    )

    fit_states = _collect(model, tokenizer, [row["question"] for row in fit_rows], cfg, style, args)
    select_states = _collect(model, tokenizer, [row["question"] for row in select_rows], cfg, style, args)
    fit_target = (fit_states["resid_ln_f"] - final_centre) @ u28
    select_target = (select_states["resid_ln_f"] - final_centre) @ u28

    locations = (*residual_location_names()[:-1], *attention_location_names())
    transports = {}
    geometry = {}
    for location in locations:
        transport = fit_ridge_transport(fit_states[location], fit_target, ridge_ratio=args.ridge_ratio)
        transports[location] = transport
        prediction = transport.predict(select_states[location])
        geometry[location] = {
            "select_r2": coefficient_r2(select_target, prediction),
            "mean_principal_cosine_to_final_u28": float(principal_angle_cosines(transport.basis, u28).mean()),
            "overlap_to_final_u28": subspace_overlap(transport.basis, u28),
        }
    # Adjacent overlap is descriptive only; transported columns share the target
    # coordinate system, while independently fitted PCs would not.
    ordered_attention = list(attention_location_names())
    for index, location in enumerate(ordered_attention):
        geometry[location]["overlap_to_previous_attention_basis"] = (
            None if index == 0 else subspace_overlap(
                transports[ordered_attention[index - 1]].basis, transports[location].basis
            )
        )

    screen_rows = select_rows[: min(args.screen_examples, len(select_rows))]
    questions = [row["question"] for row in screen_rows]
    gold_ids = gold_first_token_ids(tokenizer, (gold_text(row["answer"]) for row in screen_rows))
    dense = generate_official_codi(
        model, tokenizer, questions, latent_iterations=int(cfg.eval.latent_iterations),
        max_new_tokens=1, batch_size=args.batch_size, device=device,
        answer_cue=style.answer_prefix, force_answer_cue=True,
    )
    baseline_metrics = _first_token_metrics(dense, gold_ids, tokenizer)
    interventions = {}
    for layer, location in enumerate(ordered_attention):
        centre = fit_states[location].mean(0)
        basis = transports[location].basis
        random_bases, match = energy_matched_random_bases(
            fit_states[location], centre, basis, controls=args.random_controls,
            replicates=args.random_replicates,
            seed=args.partition_seed + layer,
        )
        arms = {"retain_u28": (basis, "retain"), "remove_u28": (basis, "remove")}
        arms.update({f"remove_random_{index:02d}": (random_basis, "remove")
                     for index, random_basis in enumerate(random_bases)})
        record = {"energy_match": {key: value for key, value in match.items() if key != "null_energies"}}
        for name, (arm_basis, mode) in arms.items():
            intervention = OfficialCODILayerwiseEndpointIntervention(
                model, location=location, basis=arm_basis, centre=centre, mode=mode,
            )
            outputs = generate_official_codi(
                model, tokenizer, questions, latent_iterations=int(cfg.eval.latent_iterations),
                max_new_tokens=1, batch_size=args.batch_size, device=device,
                answer_endpoint_intervention=intervention, answer_cue=style.answer_prefix,
                force_answer_cue=True,
            )
            record[name] = _first_token_metrics(outputs, gold_ids, tokenizer, baseline=dense)
        record["remove_u28_accuracy_drop"] = baseline_metrics["accuracy"] - record["remove_u28"]["accuracy"]
        random_drops = [baseline_metrics["accuracy"] - record[f"remove_random_{index:02d}"]["accuracy"]
                        for index in range(len(random_bases))]
        record["remove_random_accuracy_drops"] = random_drops
        record["remove_random_median_accuracy_drop"] = float(
            torch.tensor(random_drops, dtype=torch.float64).quantile(0.5)
        )
        interventions[location] = record

    passing = []
    for layer, location in enumerate(ordered_attention):
        causal = interventions[location]
        if (geometry[location]["select_r2"] >= args.minimum_r2
                and causal["retain_u28"]["dense_agreement"] >= args.minimum_retention
                and causal["remove_u28_accuracy_drop"] >= causal["remove_random_median_accuracy_drop"] + args.minimum_excess_damage):
            passing.append(layer)
    # xKV groups must be contiguous. Keep the longest contiguous run; ties prefer
    # the later group because it is closer to the measured answer endpoint.
    runs = []
    for layer in passing:
        if not runs or layer != runs[-1][-1] + 1:
            runs.append([layer])
        else:
            runs[-1].append(layer)
    selected_group = max(runs, key=lambda x: (len(x), x[-1]), default=[])
    gate_passed = len(selected_group) >= args.minimum_group_layers

    payload = {
        "schema_version": SCHEMA, "contract": CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "reproduction_gate": reproduction,
        "u28": u28, "final_centre": final_centre,
        "pc_band": [4, 32], "rank": 28,
        "fit_examples": len(fit_rows), "select_examples": len(select_rows),
        "split_hashes": {
            "fit": _sha([_normalized_question(x["question"]) for x in fit_rows]),
            "select": _sha([_normalized_question(x["question"]) for x in select_rows]),
            "original_calibration": cached["calibration_questions_sha256"],
        },
        "original_calibration_sampling": calibration_sampling,
        "transports": {name: value.state_dict() for name, value in transports.items()},
        "geometry": geometry, "baseline_first_token": baseline_metrics,
        "causal_screen": interventions,
        "gate": {
            "passed": gate_passed, "passing_layers": passing,
            "selected_contiguous_attention_layers": selected_group,
            "minimum_group_layers": args.minimum_group_layers,
            "thresholds": {"minimum_r2": args.minimum_r2,
                           "minimum_retention": args.minimum_retention,
                           "minimum_excess_damage": args.minimum_excess_damage},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, args.output_dir / "layerwise_u28.pt")
    summary = {key: value for key, value in payload.items() if key not in {"u28", "final_centre", "transports"}}
    _atomic_json(summary, args.output_dir / "summary.json")
    print(json.dumps({"baseline": baseline_metrics, "gate": payload["gate"]}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--colon-states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--fit-examples", type=int, default=2048)
    parser.add_argument("--select-examples", type=int, default=512)
    parser.add_argument("--screen-examples", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=89)
    parser.add_argument("--partition-seed", type=int, default=913)
    parser.add_argument("--ridge-ratio", type=float, default=1e-3)
    parser.add_argument("--random-replicates", type=int, default=256)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--minimum-r2", type=float, default=0.25)
    parser.add_argument("--minimum-retention", type=float, default=0.80)
    parser.add_argument("--minimum-excess-damage", type=float, default=0.01)
    parser.add_argument("--minimum-group-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
