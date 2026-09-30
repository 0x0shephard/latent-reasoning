"""Counterfactually gated distillation in the §94 repair regime (ledger §103).

The copying term is applied per example only where an intervention on the student's
own decision state says it would help: patch the teacher's coordinates (in the arm's
copying subspace) into the student's state, read out, and copy only on examples whose
first answer token flips from wrong to right.  Controls gate on the student simply
being wrong.  Arms are paired with the §100 runs of the same seed through the
published primary output; no baseline is retrained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_codi_causal_subspace_distillation import (
    DATA_SEED,
    FIT_EXAMPLES,
    GRAD_CLIP,
    SELECT_EXAMPLES,
    SELECTION_EXAMPLES,
    VALIDATE_EXAMPLES,
    WEIGHT_DECAY,
    _normalized_question,
    _paired,
    _restore,
    _snapshot,
    _trainable,
    sample_splits,
    selection_metrics,
)
from scripts.run_codi_recovery_subspace_distillation import (
    BATCH_SIZE,
    LEARNING_RATE_LORA,
    LEARNING_RATE_PROJECTOR,
    MICRO_BATCH_SIZE,
    NULL_WIDTH,
    SMOKE as RECOVERY_SMOKE,
    STEP_OPTIONS,
    WARMUP_STEPS,
)
from src.data.datasets import load_eval_set, load_train_set
from src.data.official_codi_training import (
    align_official_codi_gsm8k_eval_rows,
    collate_official_codi_kv_rows,
)
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.causal_subspace_distillation import (
    READOUT_STATE,
    add_projector_noise,
    build_target,
    cosine_lr,
    counterfactual_gate,
    fit_teacher_pca,
    readout_matrix,
    student_training_step_gated,
)
from src.mech.trajectory_supervision import student_trajectory_forward
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_gated_distillation_v1"
ARMS = ("gated_causal", "gated_full", "wrong_causal", "wrong_full")
REFERENCE_ARMS = ("none", "full", "causal", "relevance", "random", "variance")
CURVE_EVERY = 100
CHECKPOINT_EVERY = 250
SEEDS_DEFAULT = "1,2,3"
PREDICTED_BREAKS_BELOW = 25.0
PREDICTED_REPAIRS_AT_LEAST = 54.0
DEGENERATE_GATE = (0.05, 0.95)
COMPARISONS = (("gated_causal", "full"), ("gated_full", "full"), ("gated_causal", "wrong_causal"),
               ("gated_full", "wrong_full"), ("gated_causal", "none"), ("gated_full", "none"),
               ("gated_causal", "causal"), ("gated_full", "causal"), ("wrong_full", "full"),
               ("wrong_causal", "causal"), ("wrong_causal", "none"), ("wrong_full", "none"))

SMOKE = {"curve_every": 4, "checkpoint_every": 4, "test": 16}


def arm_spec(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    gate, copies = arm.split("_")
    return {"copies": copies, "gate": "patch" if gate == "gated" else "wrong"}


def reference_correctness(directory: Path, seeds, arms=REFERENCE_ARMS) -> dict:
    """{arm: {seed: [0/1 per test question]}} from a published output's predictions.jsonl
    (the aggregate's pooled columns), falling back to runs/*.json records."""
    out: dict = {}
    predictions = directory / "predictions.jsonl"
    if predictions.is_file():
        rows = [json.loads(line) for line in predictions.open(encoding="utf-8")]
        for arm in arms:
            for seed in seeds:
                key = f"{arm}_seed{seed}"
                if rows and key in rows[0]:
                    out.setdefault(arm, {})[int(seed)] = [int(bool(official_answers_match(r[key], r["gold"]))) for r in rows]
    for path in sorted((directory / "runs").glob("*.json")) if (directory / "runs").is_dir() else []:
        record = json.loads(path.read_text())
        if record.get("pilot") or "test_correct" not in record or record.get("arm") not in arms:
            continue
        out.setdefault(record["arm"], {}).setdefault(int(record["seed"]), [int(v) for v in record["test_correct"]])
    return out


def flips(arm_correct, none_correct) -> tuple[int, int]:
    repairs = sum(1 for a, n in zip(arm_correct, none_correct) if n == 0 and a == 1)
    breaks = sum(1 for a, n in zip(arm_correct, none_correct) if n == 1 and a == 0)
    return repairs, breaks


def gates_from(comparisons: dict, per_seed: dict) -> dict:
    lb = lambda k: comparisons[k]["bootstrap_95ci"][0]
    ub = lambda k: comparisons[k]["bootstrap_95ci"][1]
    has = lambda k: k in comparisons
    allpos = lambda k: bool(per_seed.get(k)) and all(v > 0 for v in per_seed[k])
    gate = {}
    for name, key in (("g1_gated_causal_beats_full", "gated_causal_minus_full"),
                      ("g2_gated_full_beats_full", "gated_full_minus_full"),
                      ("g3_counterfactual_beats_error_causal", "gated_causal_minus_wrong_causal"),
                      ("g4_counterfactual_beats_error_full", "gated_full_minus_wrong_full"),
                      ("wrong_full_beats_full", "wrong_full_minus_full")):
        if has(key):
            gate[name] = lb(key) > 0
            gate[name + "_all_seeds"] = allpos(key)
    core = ("gated_causal_minus_full", "gated_full_minus_full")
    if all(has(k) for k in core):
        gate["tie"] = all(lb(k) <= 0 <= ub(k) and ub(k) - lb(k) <= NULL_WIDTH for k in core)
    return gate


def claim_from(gate: dict) -> str:
    causal_win = gate.get("g1_gated_causal_beats_full") and gate.get("g1_gated_causal_beats_full_all_seeds")
    full_win = gate.get("g2_gated_full_beats_full") and gate.get("g2_gated_full_beats_full_all_seeds")
    if causal_win and gate.get("g3_counterfactual_beats_error_causal"):
        return "CONFIRMED: counterfactually gated causal copying beats the full state and its error-gated control"
    if full_win and gate.get("g4_counterfactual_beats_error_full"):
        return "CONFIRMED: counterfactually gated full copying beats the full state and its error-gated control"
    if causal_win or full_win or gate.get("wrong_full_beats_full"):
        return "SELECTIVE: gated copying beats the full state but the counterfactual gate does not beat the error gate"
    if gate.get("tie"):
        return "TIE: gated copying is within 3 points of the full state"
    survivors = [k for k in ("g1_gated_causal_beats_full", "g2_gated_full_beats_full",
                             "g3_counterfactual_beats_error_causal", "g4_counterfactual_beats_error_full") if gate.get(k)]
    return f"PARTIAL: {', '.join(survivors) if survivors else 'no gate passed'}"


def run(args):
    smoke = bool(args.smoke)
    S = SMOKE if smoke else None
    R = RECOVERY_SMOKE if smoke else None
    fit_n, select_n, validate_n = ((R["fit"], R["select"], R["validate"]) if smoke
                                   else (FIT_EXAMPLES, SELECT_EXAMPLES, VALIDATE_EXAMPLES))
    train_n = (R["pilot_steps"] if smoke else max(STEP_OPTIONS)) * BATCH_SIZE
    selection_n = R["selection"] if smoke else SELECTION_EXAMPLES
    curve_every = S["curve_every"] if smoke else CURVE_EVERY
    checkpoint_every = S["checkpoint_every"] if smoke else CHECKPOINT_EVERY
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    arms = tuple(args.arms)
    for arm in arms:
        arm_spec(arm)
    if args.batch_size % args.micro_batch_size:
        raise ValueError("micro-batch size must divide the batch size")
    contract = CONTRACT + ("_smoke" if smoke else "")
    started = time.perf_counter()
    remaining = lambda: args.max_seconds - (time.perf_counter() - started)

    out = args.output_dir
    runs_dir = out / "runs"; runs_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = Path(args.shared_from)
    for name in ("teacher_cache.pt", "selectors.json", "sampling.json", "preliminary.json"):
        source, target = shared_dir / name, out / name
        if not target.exists():
            if not source.is_file():
                raise FileNotFoundError(f"{name} not found in --shared-from {shared_dir}")
            shutil.copy2(source, target); print("reused", name, "from", shared_dir)
    for previous in args.resume_from:
        for path in sorted((Path(previous) / "runs").glob("*")):
            target = runs_dir / path.name
            if path.is_file() and not target.exists():
                shutil.copy2(path, target); print("resumed", path.name)
    reference_dir = Path(args.reference_from) if args.reference_from else shared_dir

    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token)
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
        dtype=dtype, settings=cfg.model, token=token)
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype)
    parameters_by_name = _trainable(model)
    official_snapshot = _snapshot(parameters_by_name)
    projector_names = [n for n in parameters_by_name if n.startswith("prj.")]
    latent_positions = int(cfg.eval.latent_iterations)
    vocab_limit = int(tokenizer.eos_token_id) + 1

    # ---- data: identical splits to the primary (verified against its sampling.json)
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    spec = data_cfg.eval.gsm8k
    from datasets import load_dataset
    raw_test = load_dataset(str(spec.hf_id), data_files={str(spec.get("split", "test")): str(spec.data_file)},
                            split=str(spec.get("split", "test")), verification_mode="no_checks")
    test_rows = align_official_codi_gsm8k_eval_rows(raw_test, test, enforce_answer_eligibility=False)
    test_rows = [{**r, "gold": str(r["gold"])} for r in test_rows]
    if smoke:
        test_rows = test_rows[: S["test"]]
    sizes = {"fit": fit_n, "select": select_n, "validate": validate_n, "train": train_n, "selection": selection_n}
    dataset = load_train_set(data_cfg, "eq_only")
    splits, sampling = sample_splits(dataset, tokenizer, test_questions={_normalized_question(r["question"]) for r in test},
                                     sizes=sizes, seed=DATA_SEED, bot_token_id=model.bot_id)
    previous = json.loads((out / "sampling.json").read_text())
    if previous["hashes"] != sampling["hashes"]:
        raise RuntimeError("the shared output was produced from different splits")
    cache = torch.load(out / "teacher_cache.pt", map_location="cpu")
    if cache["sizes"] != {k: sizes[k] for k in ("fit", "select", "validate", "train")}:
        raise RuntimeError("teacher cache belongs to different splits")
    offsets, c = {}, 0
    for name in ("fit", "select", "validate", "train"):
        offsets[name] = (c, c + sizes[name]); c += sizes[name]
    cached = lambda name: (cache["states"][offsets[name][0]:offsets[name][1]], cache["gold"][offsets[name][0]:offsets[name][1]])
    selectors = json.loads((out / "selectors.json").read_text())
    preliminary = json.loads((out / "preliminary.json").read_text())
    gate0 = preliminary["gate_projector_noise"]
    sigma = float(gate0["sigma"])
    steps = int(preliminary["pilot_projector_noise"]["chosen_steps"])
    rank = int(selectors["rank"])
    causal_set = [int(i) for i in selectors["sets"][str(rank)]["causal"]]

    readout = readout_matrix(model, vocab_limit=vocab_limit).to(device, torch.float32)
    fit_states, _ = cached("fit")
    pca = fit_teacher_pca(fit_states.float())
    train_states, train_gold = cached("train")
    train_rows, selection_rows = splits["train"], splits["selection"]
    eval_kw = dict(latent_positions=latent_positions, batch_size=args.eval_batch_size, device=device)
    targets = {"causal": build_target(pca, fit_states.float(), causal_set).to(device),
               "full": build_target(pca, fit_states.float(), None).to(device)}
    gate_sets = {"causal": causal_set, "full": None}

    def apply_damage(seed: int) -> dict:
        _restore(parameters_by_name, official_snapshot)
        return {"mode": "projector_noise", **add_projector_noise(model, sigma=sigma, seed=seed)}

    @torch.no_grad()
    def gate_fraction_at_damage(seed: int) -> dict:
        """Step-0 gate fractions on the selection split for every arm (degeneracy flag)."""
        apply_damage(seed); model.eval(); out_ = {}
        sel_states, sel_gold = cached("select")
        rows = splits["select"][:256]; states, golds = [], sel_gold[:256].to(device)
        for start in range(0, len(rows), args.eval_batch_size):
            batch = collate_official_codi_kv_rows(tokenizer, rows[start:start + args.eval_batch_size],
                                                  bot_token_id=model.bot_id, enforce_answer_eligibility=False).to(device)
            states.append(student_trajectory_forward(model, batch, latent_positions=latent_positions)
                          .answer_endpoint_hidden[:, READOUT_STATE, :].float())
        student = torch.cat(states); teacher = sel_states[:256].float().to(device)
        for arm in ARMS:
            spec_ = arm_spec(arm)
            mask = counterfactual_gate(student, teacher, golds, pca, gate_sets[spec_["copies"]], readout, spec_["gate"])
            out_[arm] = float(mask.float().mean())
        model.train()
        return out_

    # ---- step-0 gate fractions (reported; degenerate gates are flagged, not gated)
    key = "gate_fractions_step0"
    if key not in preliminary:
        fractions = gate_fraction_at_damage(seeds[0])
        preliminary[key] = {"seed": seeds[0], "sigma": sigma, "steps": steps, "fractions": fractions,
                            "degenerate": {a: not (DEGENERATE_GATE[0] <= f <= DEGENERATE_GATE[1]) for a, f in fractions.items()}}
        _atomic_json(preliminary, out / "preliminary.json")
    gate1 = preliminary[key]
    print("step-0 gate fractions", gate1["fractions"], "degenerate", gate1["degenerate"])

    summary = {
        "schema_version": 1, "contract": contract, "smoke": smoke, "damage": "projector_noise",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
        "preregistration": {
            "arms": list(arms), "seeds_this_run": list(seeds), "steps": steps, "sigma": sigma,
            "gate_modes": {a: arm_spec(a) for a in ARMS}, "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size, "learning_rate_lora": LEARNING_RATE_LORA,
            "learning_rate_projector": LEARNING_RATE_PROJECTOR, "warmup_steps": WARMUP_STEPS,
            "predicted_breaks_below": PREDICTED_BREAKS_BELOW, "predicted_repairs_at_least": PREDICTED_REPAIRS_AT_LEAST,
            "reference_arms": list(REFERENCE_ARMS), "reference_from": str(reference_dir),
            "gates": ["g1 gated_causal-full", "g2 gated_full-full", "g3 gated_causal-wrong_causal",
                      "g4 gated_full-wrong_full", "CONFIRMED = (g1&g3) or (g2&g4) with 3/3 seeds on the full comparison;",
                      "SELECTIVE = beats full but counterfactual gate does not beat error gate; TIE within 3 points"]},
        "sampling": {k: v for k, v in sampling.items() if k != "hashes"},
        "selectors": {"rank": rank, "causal_set": causal_set},
        "preliminary": preliminary, "runs": {}, "test": None, "status": "running",
        "decision": {"preliminary_passed": True, "claim": "pending"},
        "warnings": [
            "Repair regime (ledger 94-100): projector noise on the official checkpoint, then 1,000 steps; not learning from scratch.",
            "Designed after sections 100-102; new arms are paired with the section-100 runs at the same seeds via the published output.",
            "Copying pressure is norm-matched per step and concentrated on the gated examples."]}
    _atomic_json(summary, out / "summary.json")
    if args.preliminary_only:
        summary["status"] = "preliminary_only"; summary["decision"]["claim"] = "GO: gate fractions recorded; training not requested"
        _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary

    def train_one(arm: str, seed: int):
        spec_ = arm_spec(arm)
        name = f"{arm}_seed{seed}"
        record_path, ckpt_path = runs_dir / f"{name}.json", runs_dir / f"{name}.ckpt.pt"
        if record_path.is_file():
            return json.loads(record_path.read_text()), "done"
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        order = list(range(steps * args.batch_size)); random.Random(seed).shuffle(order)
        groups = [{"params": [parameters_by_name[n] for n in projector_names], "lr": LEARNING_RATE_PROJECTOR},
                  {"params": [p for n, p in parameters_by_name.items() if n not in projector_names], "lr": LEARNING_RATE_LORA}]
        parameters = list(parameters_by_name.values())
        optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
        step0, curve, elapsed, window = 0, [], 0.0, []
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _restore(parameters_by_name, ckpt["trainable"]); optimizer.load_state_dict(ckpt["optimizer"])
            step0, curve, elapsed, damage_report = int(ckpt["step"]), ckpt["curve"], float(ckpt["elapsed"]), ckpt["damage"]
            torch.set_rng_state(ckpt["cpu_rng"])
            if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
            print("resume", name, "at", step0)
        else:
            damage_report = apply_damage(seed)
            print("damaged", name, damage_report)
        target = targets[spec_["copies"]]
        gate_set = gate_sets[spec_["copies"]]
        model.train(); t0 = time.perf_counter()
        base_lrs = [LEARNING_RATE_PROJECTOR, LEARNING_RATE_LORA]

        def save_ckpt(step):
            _atomic_torch_save({"trainable": _snapshot(parameters_by_name), "optimizer": optimizer.state_dict(),
                                "step": step, "curve": curve, "damage": damage_report,
                                "elapsed": elapsed + time.perf_counter() - t0, "cpu_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, ckpt_path)

        for step in tqdm(range(step0, steps), desc=name, unit="step", initial=step0, total=steps):
            if remaining() < 120:
                save_ckpt(step); return None, "paused"
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = cosine_lr(step, total_steps=steps, base=lr, warmup=WARMUP_STEPS)
            indices = order[step * args.batch_size:(step + 1) * args.batch_size]
            micro = args.micro_batch_size
            batches = [collate_official_codi_kv_rows(tokenizer, [train_rows[i] for i in indices[k:k + micro]],
                                                     bot_token_id=model.bot_id).to(device) for k in range(0, len(indices), micro)]
            teacher_states = [train_states[indices[k:k + micro]].to(device, torch.float32) for k in range(0, len(indices), micro)]
            golds = [train_gold[indices[k:k + micro]].to(device) for k in range(0, len(indices), micro)]
            result = student_training_step_gated(model, batches, teacher_states, golds, target, parameters,
                                                 latent_positions=latent_positions, gate_mode=spec_["gate"],
                                                 gate_index_set=gate_set, pca=pca, readout=readout)
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, result.gradients):
                parameter.grad = None if gradient is None else gradient.detach()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            window.append({"answer_loss": result.answer_loss, "distillation_loss": result.distillation_loss,
                           "distillation_scale": result.distillation_scale, "gradient_cosine": result.gradient_cosine,
                           "gate_fraction": result.gate_fraction})
            del batches, teacher_states, golds, result
            done = step + 1
            if done % curve_every == 0 or done == steps:
                metrics, _, _, _ = selection_metrics(model, tokenizer, selection_rows, **eval_kw)
                model.train()
                mean = lambda k: (sum(r[k] for r in window) / len(window)) if window and window[0][k] is not None else None
                curve.append({"step": done, **metrics, "train_answer_loss": mean("answer_loss"),
                              "train_distillation_loss": mean("distillation_loss"),
                              "distillation_scale": mean("distillation_scale"), "gradient_cosine": mean("gradient_cosine"),
                              "gate_fraction": mean("gate_fraction")})
                window = []
                print("curve", name, curve[-1])
            if done % checkpoint_every == 0 and done < steps:
                save_ckpt(done)
        training_seconds = elapsed + time.perf_counter() - t0
        optimizer.zero_grad(set_to_none=True); del optimizer; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        test_metrics, test_nll, test_correct, test_outputs = selection_metrics(model, tokenizer, test_rows, **eval_kw)
        record = {"arm": arm, "seed": seed, "damage": "projector_noise", "steps": steps, "training_seconds": training_seconds,
                  "damage_report": damage_report, "gate": spec_, "curve": curve,
                  "mean_gate_fraction": (sum(p["gate_fraction"] for p in curve if p["gate_fraction"] is not None)
                                         / max(1, sum(p["gate_fraction"] is not None for p in curve))) if curve else None,
                  "mean_gradient_cosine": (sum(p["gradient_cosine"] for p in curve if p["gradient_cosine"] is not None)
                                           / max(1, sum(p["gradient_cosine"] is not None for p in curve))) if curve else None,
                  "test": test_metrics, "test_correct": test_correct, "test_nll": test_nll, "test_outputs": test_outputs}
        _atomic_json(record, record_path)
        if ckpt_path.is_file():
            ckpt_path.unlink()
        print("test", name, test_metrics)
        return record, "done"

    for arm in arms:
        for seed in seeds:
            _, status = train_one(arm, seed)
            if status == "paused":
                summary["status"] = "paused"
                summary["decision"]["claim"] = f"PAUSED before {arm} seed {seed} finished: publish this output and rerun with it attached"
                _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary
    summary["status"] = "trained"
    return aggregate(args, summary, test_rows, out, reference_dir)


def aggregate(args, summary, test_rows, out, reference_dir: Path):
    runs = {}
    for path in sorted((out / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if "test" in record:
            runs[f"{record['arm']}_seed{record['seed']}"] = record
    if not runs:
        summary["decision"]["claim"] = "no completed runs to aggregate"; _atomic_json(summary, out / "summary.json"); return summary
    seeds = sorted({r["seed"] for r in runs.values()})
    n = len(next(iter(runs.values()))["test_correct"])
    correct = {}  # arm -> {seed: [0/1]}
    for record in runs.values():
        correct.setdefault(record["arm"], {})[record["seed"]] = [int(v) for v in record["test_correct"]]
    reference = reference_correctness(reference_dir, seeds)
    for arm, by_seed in reference.items():
        for seed, vec in by_seed.items():
            if len(vec) == n:
                correct.setdefault(arm, {})[seed] = vec
    usable = {arm: sorted(s for s in by_seed if s in seeds) for arm, by_seed in correct.items()}
    mean = {arm: torch.tensor([correct[arm][s] for s in usable[arm]], dtype=torch.float32).mean(0).tolist()
            for arm in correct if usable[arm]}
    comparisons, per_seed = {}, {}
    for i, (a, b) in enumerate(COMPARISONS):
        common = [s for s in usable.get(a, []) if s in usable.get(b, [])]
        if not common:
            continue
        ma = torch.tensor([correct[a][s] for s in common], dtype=torch.float32).mean(0).tolist()
        mb = torch.tensor([correct[b][s] for s in common], dtype=torch.float32).mean(0).tolist()
        key = f"{a}_minus_{b}"
        comparisons[key] = {**_paired(ma, mb, seed=args.seed + i, samples=args.bootstrap_samples), "seeds": common}
        per_seed[key] = [100 * (sum(correct[a][s]) - sum(correct[b][s])) / n for s in common]
    decomposition = {}
    if "none" in correct:
        for arm in correct:
            if arm == "none":
                continue
            rows = [flips(correct[arm][s], correct["none"][s]) for s in usable[arm] if s in correct["none"]]
            if rows:
                decomposition[arm] = {"repairs_mean": sum(r for r, _ in rows) / len(rows),
                                      "breaks_mean": sum(b for _, b in rows) / len(rows),
                                      "per_seed": [{"seed": s, "repairs": r, "breaks": b}
                                                   for s, (r, b) in zip([s for s in usable[arm] if s in correct["none"]], rows)]}
    gate = gates_from(comparisons, per_seed)
    for arm in ("gated_causal", "gated_full"):
        if arm in decomposition:
            gate[f"predicted_breaks_met_{arm}"] = decomposition[arm]["breaks_mean"] < PREDICTED_BREAKS_BELOW
            gate[f"predicted_repairs_met_{arm}"] = decomposition[arm]["repairs_mean"] >= PREDICTED_REPAIRS_AT_LEAST
    arm_results = {arm: {"seeds": usable[arm], "test_accuracy_by_seed": [sum(correct[arm][s]) / n for s in usable[arm]],
                         "test_accuracy_mean": float(torch.tensor(mean[arm]).mean()),
                         "source": "this_run" if arm in ARMS else "reference"} for arm in mean}
    for k, record in runs.items():
        arm_results[record["arm"]].setdefault("mean_gate_fraction", {})[k] = record["mean_gate_fraction"]
        arm_results[record["arm"]].setdefault("mean_gradient_cosine", {})[k] = record["mean_gradient_cosine"]
    summary["runs"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("test_correct", "test_nll", "test_outputs")} for k, v in runs.items()}
    summary["test"] = {"examples": n, "arms": arm_results, "comparisons": comparisons, "per_seed_differences": per_seed,
                       "decomposition_vs_none": decomposition, "gate": gate}
    summary["status"] = "aggregated"
    summary["decision"] = {"preliminary_passed": True, "claim": claim_from(gate)}
    _atomic_json(summary, out / "summary.json")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(test_rows[:n]):
            handle.write(json.dumps({"index": index, "question": row["question"], "gold": row["gold"],
                                     **{k: v["test_outputs"][index] for k, v in runs.items()}}, ensure_ascii=False) + "\n")
    print(json.dumps(summary["decision"], indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared-from", required=True, help="published §100 output (teacher cache, selectors, preliminary)")
    parser.add_argument("--reference-from", default=None, help="published output with predictions.jsonl for the baseline arms")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--arms", default=",".join(ARMS), type=lambda s: tuple(a for a in s.split(",") if a))
    parser.add_argument("--seeds", default=SEEDS_DEFAULT)
    parser.add_argument("--resume-from", default=[], type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--preliminary-only", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30_600)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_930)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and args.batch_size != BATCH_SIZE:
        raise ValueError("the preregistered batch size cannot be changed")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
