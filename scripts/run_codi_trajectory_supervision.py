"""Warm-start official CODI with trajectory-level supervision arms (ledger §84).

Six arms share data, order, optimizer, steps, and the official base objective, and
differ only in what the auxiliary term supervises: nothing (CODI), R-KV-selected
KV targets on all slots (KaVa), value-token KV targets on the odd slots, random
positions on the odd slots, value tokens on the even slots, or the slot readout's
cross-entropy toward the value token.  Every arm runs three seeds; the test set is
read once, after a drift screen on the continued-CODI arm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    verify_full_reproduction_gate,
)
from scripts.run_codi_preanswer_kv_subspace_discovery import _sha
from src.data.datasets import load_eval_set, load_train_set
from src.data.official_codi_training import (
    _tokenize_segment,
    collate_official_codi_kv_rows,
    encode_official_codi_row,
    format_official_codi_row,
    official_codi_answer_is_eligible,
)
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.rank16_xkv_confirmation import paired_bootstrap_interval
from src.mech.trajectory_supervision import (
    ARM_ORDER,
    ARMS,
    equation_value_spans,
    trajectory_training_step,
    value_token_indices,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_trajectory_supervision_v1"
TRAIN_EXAMPLES = 8_192
SELECTION_EXAMPLES = 256
DATA_SEED = 20_260_922
TRAINING_SEEDS = (1, 2, 3)
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
IMPORTANCE_WEIGHT = 0.1
MINIMUM_EQUATIONS = 2
DRIFT_MARGIN = 0.03
SCREEN_ARM, SCREEN_SEED = "codi", TRAINING_SEEDS[0]

SMOKE = {"train": 16, "selection": 8, "test": 16, "seeds": (1,), "arms": ("codi", "value_odd", "kava")}


# ------------------------------------------------------------------- data prep


def prepare_row(tokenizer, row: dict, *, bot_token_id: int) -> dict | None:
    """Official formatting plus value-token localisation; ``None`` excludes the row."""
    try:
        if not official_codi_answer_is_eligible(row["answer"]):
            return None
        if len(str(row["cot"]).split(" ")) < MINIMUM_EQUATIONS:
            return None
        formatted = format_official_codi_row(row)
        if not equation_value_spans(formatted.cot):
            return None
        encode_official_codi_row(tokenizer, row, bot_token_id=bot_token_id)
    except (KeyError, TypeError, ValueError):
        return None
    cot_ids = _tokenize_segment(tokenizer, formatted.cot)
    bos = getattr(tokenizer, "bos_token_id", None)
    if cot_ids and bos is not None and cot_ids[0] == bos:
        cot_ids = cot_ids[1:]
    positions = value_token_indices(tokenizer, formatted.cot, cot_ids)
    if not positions:
        return None
    return {
        "question": str(row["question"]), "cot": str(row["cot"]), "answer": str(row["answer"]),
        "gold": str(row["answer"]).split(" ")[-1].replace("####", "").strip(),
        "value_positions": [int(p) for p in positions],
        "value_token_ids": [int(cot_ids[p]) for p in positions],
    }


def sample_rows(dataset, tokenizer, *, test_questions, train_examples, selection_examples,
                seed, bot_token_id):
    """Unique-question deterministic sample of prepared GSM8k-Aug rows."""
    groups: dict[str, list[int]] = {}
    for index, question in enumerate(dataset["question"]):
        groups.setdefault(_normalized_question(question), []).append(index)
    overlap = sorted(set(groups).intersection(test_questions))
    if overlap:
        raise RuntimeError(f"GSM8k-Aug/test question overlap is nonzero ({len(overlap)})")
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    needed = int(train_examples) + int(selection_examples)
    prepared, indices, skipped = [], [], 0
    for key in keys:
        if len(prepared) >= needed:
            break
        index = generator.choice(groups[key])
        row = prepare_row(tokenizer, dataset[index], bot_token_id=bot_token_id)
        if row is None:
            skipped += 1
            continue
        prepared.append(row)
        indices.append(int(index))
    if len(prepared) < needed:
        raise ValueError(f"only {len(prepared)} eligible rows for {needed} requested")
    train, selection = prepared[:train_examples], prepared[train_examples:]
    audit = {
        "unique_questions": len(groups), "skipped_candidates": skipped, "data_seed": seed,
        "train_examples": len(train), "selection_examples": len(selection),
        "selected_source_indices_sha256": _sha([str(i) for i in indices]),
        "train_questions_sha256": _sha([_normalized_question(r["question"]) for r in train]),
        "selection_questions_sha256": _sha([_normalized_question(r["question"]) for r in selection]),
        "values_per_row": {
            "mean": sum(len(r["value_positions"]) for r in prepared) / len(prepared),
            "min": min(len(r["value_positions"]) for r in prepared),
            "max": max(len(r["value_positions"]) for r in prepared),
        },
    }
    return train, selection, audit


# ------------------------------------------------------------------- helpers


def _trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def _snapshot(parameters_by_name):
    return {n: p.detach().cpu().clone() for n, p in parameters_by_name.items()}


def _restore(parameters_by_name, snapshot):
    with torch.no_grad():
        for n, p in parameters_by_name.items():
            p.copy_(snapshot[n].to(device=p.device, dtype=p.dtype))


def _correct(outputs, rows) -> list[bool]:
    return [bool(official_answers_match(t, r["gold"])) for t, r in zip(outputs, rows)]


def evaluate(model, tokenizer, rows, *, latent_positions, max_new_tokens, batch_size, device):
    model.eval()
    with torch.no_grad():
        outputs = generate_official_codi(
            model, tokenizer, [r["question"] for r in rows], latent_iterations=latent_positions,
            max_new_tokens=max_new_tokens, batch_size=batch_size, device=device,
        )
    correct = _correct(outputs, rows)
    return {"accuracy": sum(correct) / len(rows), "correct": int(sum(correct)),
            "examples": len(rows)}, correct, outputs


def train_arm(model, tokenizer, rows, spec, *, seed, latent_positions, batch_size, device,
              log_every=32):
    parameters_by_name = _trainable(model)
    parameters = list(parameters_by_name.values())
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed * 1_000_003)
    model.train()
    curve, started = [], time.perf_counter()
    window = []
    for step, batch_indices in enumerate(tqdm(batches, desc=f"{spec.name}/seed{seed}", unit="step")):
        batch_rows = [rows[i] for i in batch_indices]
        batch = collate_official_codi_kv_rows(
            tokenizer, batch_rows, bot_token_id=model.bot_id
        ).to(device)
        result = trajectory_training_step(
            model, batch, spec, parameters, latent_positions=latent_positions,
            value_positions=[r["value_positions"] for r in batch_rows],
            value_token_ids=[r["value_token_ids"] for r in batch_rows],
            random_generator=generator, importance_weight=IMPORTANCE_WEIGHT,
        )
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, result.gradients):
            parameter.grad = None if gradient is None else gradient.detach()
        torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        optimizer.step()
        window.append(result)
        if (step + 1) % log_every == 0 or step + 1 == len(batches):
            curve.append({
                "step": step + 1,
                "answer_loss": sum(r.answer_loss for r in window) / len(window),
                "endpoint_loss": sum(r.endpoint_loss for r in window) / len(window),
                "auxiliary_loss": (
                    sum(r.auxiliary_loss for r in window) / len(window)
                    if window[0].auxiliary_loss is not None else None
                ),
                "auxiliary_scale": (
                    sum(r.auxiliary_scale for r in window) / len(window)
                    if window[0].auxiliary_scale is not None else None
                ),
                "supervised_slot_fraction": (
                    sum(r.supervised_slots for r in window) / len(window)
                    if window[0].supervised_slots is not None else None
                ),
                "rkv_value_overlap": (
                    sum(r.rkv_value_overlap for r in window) / len(window)
                    if window[0].rkv_value_overlap is not None else None
                ),
            })
            window = []
        del batch, result
    seconds = time.perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    for parameter in parameters:
        parameter.grad = None
    del optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"steps": len(batches), "seconds": seconds, "curve": curve}


def _mean_correct(runs: list[dict], key: str) -> torch.Tensor:
    stacked = torch.tensor([r[key] for r in runs], dtype=torch.float32)
    return stacked.mean(0)


def compare(a: torch.Tensor, b: torch.Tensor, *, seed: int, samples: int) -> dict:
    diff = a - b
    return {"mean_difference": float(diff.mean()),
            "bootstrap_95ci": paired_bootstrap_interval(diff, seed=seed, samples=samples)}


def gates_from(comparisons: dict) -> dict:
    lb = lambda name: comparisons[name]["bootstrap_95ci"][0]
    gate = {
        "h1_kava_beats_codi": lb("kava_minus_codi") > 0,
        "h2_value_beats_kava": lb("value_odd_minus_kava") > 0,
        "h2_value_beats_random": lb("value_odd_minus_random_odd") > 0,
        "h3_odd_beats_even": lb("value_odd_minus_value_even") > 0,
        "value_beats_codi": lb("value_odd_minus_codi") > 0,
    }
    gate["headline"] = gate["h2_value_beats_kava"] and gate["h2_value_beats_random"] and gate["value_beats_codi"]
    return gate


def claim_from(gate: dict) -> str:
    if gate["headline"]:
        return "CONFIRMED: mechanism-selected trajectory supervision beats CODI, KaVa, and random positions"
    if gate["value_beats_codi"] and gate["h2_value_beats_random"]:
        return "PARTIAL: value-slot supervision beats CODI and random positions but not KaVa"
    if gate["h1_kava_beats_codi"]:
        return "PARTIAL: KaVa beats CODI at this budget; the value selector does not add to it"
    return "STOP: no trajectory-supervision arm separated from continued CODI at this budget"


# ------------------------------------------------------------------------ run


def run(args):
    smoke = bool(args.smoke)
    train_n = SMOKE["train"] if smoke else TRAIN_EXAMPLES
    selection_n = SMOKE["selection"] if smoke else SELECTION_EXAMPLES
    seeds = SMOKE["seeds"] if smoke else TRAINING_SEEDS
    arms = SMOKE["arms"] if smoke else ARM_ORDER
    contract = CONTRACT + ("_smoke" if smoke else "")
    if not smoke and (args.batch_size != BATCH_SIZE):
        raise ValueError("the preregistered batch size cannot be changed")

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
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype)
    parameters_by_name = _trainable(model)
    if not parameters_by_name:
        raise RuntimeError("official CODI exposes no trainable parameters")
    frozen_state = _snapshot(parameters_by_name)
    latent_positions = int(cfg.eval.latent_iterations)
    max_new_tokens = int(cfg.eval.max_new_tokens)

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    if len(test) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("GSM8K test count drifted")
    test_rows = [{"question": r["question"], "gold": str(r["gold"])} for r in test]
    if smoke:
        test_rows = test_rows[: SMOKE["test"]]
    dataset = load_train_set(data_cfg, "eq_only")
    train_rows, selection_rows, sampling = sample_rows(
        dataset, tokenizer, test_questions={_normalized_question(r["question"]) for r in test},
        train_examples=train_n, selection_examples=selection_n, seed=DATA_SEED,
        bot_token_id=model.bot_id,
    )
    print("sampling", {k: v for k, v in sampling.items() if "sha" not in k})

    out = args.output_dir
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    common_eval = dict(latent_positions=latent_positions, max_new_tokens=max_new_tokens,
                       batch_size=args.eval_batch_size, device=device)

    def cached(name):
        path = runs_dir / f"{name}.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def store(name, payload):
        _atomic_json(payload, runs_dir / f"{name}.json")

    frozen = cached("frozen") or {}
    if "selection" not in frozen:
        record, correct, outputs = evaluate(model, tokenizer, selection_rows, **common_eval)
        frozen.update({"selection": record, "selection_correct": correct, "selection_outputs": outputs})
        store("frozen", frozen)
    print("frozen selection", frozen["selection"])

    summary = {
        "schema_version": 1, "contract": contract, "smoke": smoke,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
        "preregistration": {
            "arms": {n: {"auxiliary": s.auxiliary, "selector": s.selector, "slots": list(s.slots)}
                     for n, s in ARMS.items() if n in arms},
            "train_examples": train_n, "selection_examples": selection_n, "data_seed": DATA_SEED,
            "training_seeds": list(seeds), "batch_size": args.batch_size,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "rkv_importance_weight": IMPORTANCE_WEIGHT, "minimum_equations": MINIMUM_EQUATIONS,
            "drift_margin": DRIFT_MARGIN, "screen": f"{SCREEN_ARM} seed {SCREEN_SEED} selection accuracy >= frozen - margin",
            "auxiliary_weighting": "gradient norm matched to the endpoint-loss gradient norm every step",
            "base_objective": "student gold-answer NLL + official endpoint smooth-L1/std over 13 states",
            "teacher_path": "frozen; no teacher cross-entropy in any arm",
            "decoding": "official native protocol, no forced cue",
            "gates": ["h1 kava-codi lb>0", "h2 value_odd-kava lb>0 and value_odd-random_odd lb>0",
                      "h3 value_odd-value_even lb>0", "headline = h2 and value_odd-codi lb>0"],
            "bootstrap_samples": args.bootstrap_samples,
        },
        "sampling": sampling, "frozen": {"selection": frozen["selection"]},
        "screen": None, "runs": {}, "test": None, "decision": {"screen_passed": None, "claim": "pending"},
        "warnings": [
            "Warm start of 1,024 steps at 2e-5 is a cheap held-out gate, not a full retraining; nulls bound only this budget.",
            "GSM8k-Aug rows with fewer than two equations are excluded for every arm so each truncated trace carries a value.",
            "Auxiliary gradients are norm-matched to the endpoint term, so arms differ in what they supervise, not how strongly.",
            "The test set is read once per trained model, after the drift screen; it was opened by earlier method families (ledger 75-79).",
        ],
    }
    _atomic_json(summary, out / "summary.json")

    loaded = {"name": "frozen"}

    def load_frozen():
        if loaded["name"] != "frozen":
            _restore(parameters_by_name, frozen_state)
            loaded["name"] = "frozen"

    def run_one(arm_name, seed, *, with_test):
        """Train (or reuse) one arm/seed; evaluate selection and, if asked, test."""
        name = f"{arm_name}_seed{seed}"
        record = cached(name)
        if record is not None and "test" in record:
            return record
        if record is not None and not with_test:
            return record
        spec = ARMS[arm_name]

        def train_now():
            load_frozen()
            training = train_arm(model, tokenizer, train_rows, spec, seed=seed,
                                 latent_positions=latent_positions, batch_size=args.batch_size,
                                 device=device)
            loaded["name"] = name
            sel, sel_correct, sel_out = evaluate(model, tokenizer, selection_rows, **common_eval)
            if args.save_trained_state:
                _atomic_torch_save(_snapshot(parameters_by_name), runs_dir / f"{name}.pt")
            print("selection", name, sel)
            return {"arm": arm_name, "seed": seed, "training": training, "selection": sel,
                    "selection_correct": sel_correct, "selection_outputs": sel_out}

        if record is None:
            record = train_now()
            store(name, record)
        if with_test:
            if loaded["name"] != name:
                state_path = runs_dir / f"{name}.pt"
                if state_path.is_file():
                    _restore(parameters_by_name, torch.load(state_path, map_location="cpu"))
                    loaded["name"] = name
                else:
                    # A resumed session lost the in-memory weights; retrain deterministically
                    # and replace the selection record so it matches the evaluated model.
                    record = train_now()
                    store(name, record)
            tst, tst_correct, tst_out = evaluate(model, tokenizer, test_rows, **common_eval)
            record.update({"test": tst, "test_correct": tst_correct, "test_outputs": tst_out})
            store(name, record)
            print("test", name, tst)
        return record

    # Phase 1: screen on the continued-CODI arm before any test read.
    screen_record = run_one(SCREEN_ARM, SCREEN_SEED, with_test=False)
    drift = screen_record["selection"]["accuracy"] - frozen["selection"]["accuracy"]
    screen = {"frozen_selection_accuracy": frozen["selection"]["accuracy"],
              "codi_seed1_selection_accuracy": screen_record["selection"]["accuracy"],
              "drift": drift, "margin": DRIFT_MARGIN, "passed": drift >= -DRIFT_MARGIN}
    summary["screen"] = screen
    summary["decision"]["screen_passed"] = bool(screen["passed"])
    print("screen", screen)
    if not screen["passed"]:
        summary["decision"]["claim"] = (
            "STOP: continued CODI drifted more than the margin on the selection split; "
            "the learning rate is too high for a warm start and the test set was not read"
        )
        _atomic_json(summary, out / "summary.json")
        return summary

    # Phase 2: the screen model is still in memory, so read its test split first,
    # then the frozen reference, then every remaining arm and seed.
    all_runs = {}
    all_runs[f"{SCREEN_ARM}_seed{SCREEN_SEED}"] = run_one(SCREEN_ARM, SCREEN_SEED, with_test=True)
    if "test" not in frozen:
        load_frozen()
        record, correct, outputs = evaluate(model, tokenizer, test_rows, **common_eval)
        frozen.update({"test": record, "test_correct": correct, "test_outputs": outputs})
        store("frozen", frozen)
    summary["frozen"]["test"] = frozen["test"]
    for arm_name in arms:
        for seed in seeds:
            key = f"{arm_name}_seed{seed}"
            if key not in all_runs:
                all_runs[key] = run_one(arm_name, seed, with_test=True)
            _atomic_json({**summary, "runs": {k: {kk: vv for kk, vv in v.items()
                                                  if not kk.endswith(("_correct", "_outputs"))}
                                              for k, v in all_runs.items()}}, out / "summary.json")

    # Aggregate.
    by_arm = {a: [all_runs[f"{a}_seed{s}"] for s in seeds] for a in arms}
    test_mean = {a: _mean_correct(rs, "test_correct") for a, rs in by_arm.items()}
    sel_mean = {a: _mean_correct(rs, "selection_correct") for a, rs in by_arm.items()}
    frozen_test = torch.tensor(frozen["test_correct"], dtype=torch.float32)
    pairs = [("kava", "codi"), ("value_odd", "kava"), ("value_odd", "random_odd"),
             ("value_odd", "value_even"), ("value_odd", "codi"), ("recon_odd", "value_odd"),
             ("random_odd", "codi"), ("value_even", "codi"), ("recon_odd", "codi")]
    comparisons = {}
    for index, (a, b) in enumerate(pairs):
        if a in test_mean and b in test_mean:
            comparisons[f"{a}_minus_{b}"] = compare(test_mean[a], test_mean[b],
                                                    seed=args.seed + index, samples=args.bootstrap_samples)
    for index, a in enumerate(arms):
        comparisons[f"{a}_minus_frozen"] = compare(test_mean[a], frozen_test,
                                                   seed=args.seed + 100 + index, samples=args.bootstrap_samples)
    arm_results = {
        a: {
            "test_accuracy_by_seed": [r["test"]["accuracy"] for r in rs],
            "test_accuracy_mean": float(test_mean[a].mean()),
            "selection_accuracy_by_seed": [r["selection"]["accuracy"] for r in rs],
            "selection_accuracy_mean": float(sel_mean[a].mean()),
            "training_seconds_mean": sum(r["training"]["seconds"] for r in rs) / len(rs),
            "final_curve_point": [r["training"]["curve"][-1] if r["training"]["curve"] else None for r in rs],
            "mean_auxiliary_scale": _curve_mean(rs, "auxiliary_scale"),
            "mean_supervised_slot_fraction": _curve_mean(rs, "supervised_slot_fraction"),
            "mean_rkv_value_overlap": _curve_mean(rs, "rkv_value_overlap"),
        } for a, rs in by_arm.items()
    }
    full_design = all(k in comparisons for k in (
        "kava_minus_codi", "value_odd_minus_kava", "value_odd_minus_random_odd",
        "value_odd_minus_value_even", "value_odd_minus_codi"))
    gate = gates_from(comparisons) if full_design else None
    summary["runs"] = {k: {kk: vv for kk, vv in v.items() if not kk.endswith(("_correct", "_outputs"))}
                       for k, v in all_runs.items()}
    summary["test"] = {"arms": arm_results, "comparisons": comparisons, "gate": gate,
                       "frozen_accuracy": frozen["test"]["accuracy"]}
    summary["decision"] = {"screen_passed": True,
                           "final_passed": bool(gate["headline"]) if gate else None,
                           "claim": claim_from(gate) if gate else "smoke run: no gate evaluated"}
    tensors = {"test_correct": {a: test_mean[a] for a in arms}, "frozen_test_correct": frozen_test,
               "selection_correct": {a: sel_mean[a] for a in arms},
               "per_run_test_correct": {k: torch.tensor(v["test_correct"]) for k, v in all_runs.items()}}
    _atomic_json(summary, out / "summary.json")
    _atomic_torch_save({**summary, **tensors}, out / "trajectory_supervision.pt")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(test_rows):
            handle.write(json.dumps({
                "dataset_index": index, "question": row["question"], "gold": row["gold"],
                "frozen": frozen["test_outputs"][index],
                **{k: v["test_outputs"][index] for k, v in all_runs.items()},
            }, ensure_ascii=False) + "\n")
    print(json.dumps(summary["decision"], indent=2))
    return summary


def _curve_mean(runs, key):
    values = [p[key] for r in runs for p in r["training"]["curve"] if p.get(key) is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_922)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-trained-state", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end pipeline check; writes a *_smoke contract")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
