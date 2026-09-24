"""Recovery test of distillation targets on the official CODI checkpoint (ledger §94).

The projector (or, in the second damage mode, every LoRA B matrix) of the official
checkpoint is damaged; the student then fine-tunes on GSM8k-Aug under answer
cross-entropy plus a norm-matched distillation term toward a rank-r subspace of the
teacher's decision state, selected by variance, gradient relevance, retain-only
intervention, or at random (plus ``full`` and ``none``).  Primary outcome: GSM8K test
exact match on all 1,319 questions at the final step.  A go/no-go (headroom, selector
divergence, term not converged, recovery pilot) precedes any counted seed.
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
    CANDIDATE_PCS,
    DATA_SEED,
    FIT_EXAMPLES,
    GRAD_CLIP,
    MINIMUM_GAP,
    RANDOM_PC_SEED,
    RANK_GRID,
    RETENTION_FLOOR,
    SELECT_EXAMPLES,
    SELECTION_EXAMPLES,
    VALIDATE_EXAMPLES,
    WEIGHT_DECAY,
    _normalized_question,
    _paired,
    _restore,
    _snapshot,
    _trainable,
    precompute_teacher,
    sample_splits,
    selection_metrics,
)
from src.data.datasets import load_eval_set, load_train_set
from src.data.official_codi_training import (
    align_official_codi_gsm8k_eval_rows,
    collate_official_codi_kv_rows,
)
from src.eval.official_codi import select_device
from src.mech.causal_subspace_distillation import (
    READOUT_STATE,
    add_projector_noise,
    build_target,
    choose_rank,
    cosine_lr,
    distillation_loss,
    evaluate_index_sets,
    fit_teacher_pca,
    jaccard,
    readout_matrix,
    reset_projector,
    scale_lora_b,
    select_causal_greedy,
    select_random,
    select_relevance,
    select_variance,
    student_training_step,
)
from src.mech.trajectory_supervision import student_trajectory_forward
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_recovery_subspace_distillation_v1"
ARMS_PRIMARY = ("none", "full", "variance", "relevance", "causal", "random")
ARMS_WEIGHTS = ("variance_x0.3", "variance_x3", "causal_x0.3", "causal_x3")
ARMS_DAMAGE = ("none", "variance", "causal", "random")
DAMAGES = ("projector_noise", "projector", "lora_half")
LORA_HALF_FACTOR = 0.5
NOISE_GRID = (0.25, 0.5, 1.0, 2.0)   # sigma sweep for projector_noise, calibrated in the go/no-go (ledger 96)
STEP_OPTIONS = (1_000, 2_000)
PILOT_STEPS = 2_000
PILOT_SEEDS = (0, 100)               # both must cross the threshold (bimodality check, ledger 96)
BATCH_SIZE = 16
MICRO_BATCH_SIZE = 8
LEARNING_RATE_LORA = 1e-4
LEARNING_RATE_PROJECTOR = 5e-4
WARMUP_STEPS = 50
CURVE_EVERY = 100
CHECKPOINT_EVERY = 250
TERM_CHECK_ROWS = 256
HEADROOM_MIN_GAP = 0.20     # official - damaged on the selection split (amended, ledger 95)
RECOVERY_FRACTION = 0.5     # recovery threshold = damaged + fraction * gap
MAX_SHARED_PCS = 8          # Jaccard <= 0.5 at rank 12
TERM_RATIO_MIN = 2.0
NULL_WIDTH = 0.03
TEACHER_BATCH = 64
SEEDS_DEFAULT = "1,2,3,4,5"

SMOKE = {"fit": 64, "select": 64, "validate": 64, "pilot_steps": 12, "steps": 8, "selection": 16,
         "test": 16, "curve_every": 4, "checkpoint_every": 4, "candidates": 16, "term_rows": 16}


def parse_arm(arm: str) -> tuple[str, float]:
    """``variance_x0.3`` -> (``variance``, 0.3); plain arms have multiplier 1."""
    if "_x" in arm:
        base, factor = arm.rsplit("_x", 1)
        return base, float(factor)
    return arm, 1.0


def recovery_threshold(official: float, damaged: float, *, fraction: float = RECOVERY_FRACTION) -> float:
    """Selection-split exact match that counts as having recovered half the lost accuracy."""
    return float(damaged + fraction * (official - damaged))


def choose_steps(curve: list[dict], *, threshold: float,
                 options: tuple[int, ...] = STEP_OPTIONS) -> int | None:
    """Smallest preregistered step budget by which the pilot reached ``threshold``."""
    reached = [int(p["step"]) for p in curve if p.get("accuracy", 0.0) >= threshold]
    if not reached:
        return None
    first = min(reached)
    for option in options:
        if first <= option:
            return option
    return None


def choose_steps_for_pilots(curves: list[list[dict]], *, threshold: float,
                            options: tuple[int, ...] = STEP_OPTIONS) -> int | None:
    """The budget every pilot satisfies; None if any pilot fails (a disagreeing pair is a STOP)."""
    budgets = [choose_steps(curve, threshold=threshold, options=options) for curve in curves]
    if not budgets or any(b is None for b in budgets):
        return None
    return max(budgets)


def steps_to_threshold(curve: list[dict], threshold: float) -> int | None:
    for point in curve:
        if point.get("accuracy", 0.0) >= threshold:
            return int(point["step"])
    return None


COMPARISONS = (("full", "none"), ("causal", "variance"), ("causal", "relevance"), ("causal", "random"),
               ("variance", "none"), ("causal", "none"), ("variance", "random"),
               ("causal_x0.3", "variance_x0.3"), ("causal_x3", "variance_x3"),
               ("causal_x0.3", "causal"), ("causal_x3", "causal"),
               ("variance_x0.3", "variance"), ("variance_x3", "variance"))


def gates_from(comparisons: dict, none_selection_accuracy: float | None, threshold: float) -> dict:
    lb = lambda k: comparisons[k]["bootstrap_95ci"][0]
    ub = lambda k: comparisons[k]["bootstrap_95ci"][1]
    has = lambda k: k in comparisons
    gate = {"r0_recovered": none_selection_accuracy is not None and none_selection_accuracy >= threshold,
            "recovery_threshold": threshold, "none_final_selection_accuracy": none_selection_accuracy}
    if has("full_minus_none"):
        gate["s1_full_beats_none"] = lb("full_minus_none") > 0
    if has("causal_minus_variance"):
        gate["h1_causal_beats_variance"] = lb("causal_minus_variance") > 0
        gate["h1_variance_beats_causal"] = ub("causal_minus_variance") < 0
        gate["h1_tight_null"] = (lb("causal_minus_variance") <= 0 <= ub("causal_minus_variance")
                                 and ub("causal_minus_variance") - lb("causal_minus_variance") <= NULL_WIDTH)
    if has("causal_minus_relevance"):
        gate["h2_causal_beats_relevance"] = lb("causal_minus_relevance") > 0
    if has("causal_minus_random"):
        gate["h3_causal_beats_random"] = lb("causal_minus_random") > 0
    if has("variance_minus_none"):
        gate["variance_hurts_recovery"] = ub("variance_minus_none") < 0
        gate["variance_helps_recovery"] = lb("variance_minus_none") > 0
    if has("causal_minus_none"):
        gate["causal_hurts_recovery"] = ub("causal_minus_none") < 0
        gate["causal_helps_recovery"] = lb("causal_minus_none") > 0
    return gate


def claim_from(gate: dict) -> str:
    if not gate.get("r0_recovered"):
        return "STOP: the none arm did not recover half the lost selection-split accuracy; the recovery regime was not reached"
    if gate.get("h1_causal_beats_variance") and gate.get("h3_causal_beats_random"):
        return "CONFIRMED: the intervention-selected target recovers accuracy better than the variance-selected and random targets"
    if gate.get("h1_variance_beats_causal"):
        return "REVERSED: the variance-selected target recovers accuracy better than the intervention-selected one"
    if gate.get("h1_tight_null"):
        return "NULL: causal and variance targets recover accuracy equally (interval within 3 points of zero); the selector does not matter in the recovery regime"
    if gate.get("h1_causal_beats_variance"):
        return "PARTIAL: causal beats variance but not random; the benefit is not specific to the intervention-selected directions"
    return "INCONCLUSIVE: causal versus variance covers zero with an interval wider than 3 points"


def run(args):
    smoke = bool(args.smoke)
    S = SMOKE if smoke else None
    fit_n, select_n, validate_n = ((S["fit"], S["select"], S["validate"]) if smoke
                                   else (FIT_EXAMPLES, SELECT_EXAMPLES, VALIDATE_EXAMPLES))
    pilot_steps = S["pilot_steps"] if smoke else PILOT_STEPS
    max_steps = pilot_steps if smoke else max(STEP_OPTIONS)
    train_n = max_steps * BATCH_SIZE
    selection_n = S["selection"] if smoke else SELECTION_EXAMPLES
    curve_every = S["curve_every"] if smoke else CURVE_EVERY
    checkpoint_every = S["checkpoint_every"] if smoke else CHECKPOINT_EVERY
    candidates = S["candidates"] if smoke else CANDIDATE_PCS
    term_rows = S["term_rows"] if smoke else TERM_CHECK_ROWS
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    arms = tuple(args.arms)
    for arm in arms:
        base, _ = parse_arm(arm)
        if base not in ARMS_PRIMARY:
            raise ValueError(f"unknown arm {arm!r}")
    if args.damage not in DAMAGES:
        raise ValueError(f"unknown damage {args.damage!r}")
    if args.batch_size % args.micro_batch_size:
        raise ValueError("micro-batch size must divide the batch size")
    contract = CONTRACT + ("_smoke" if smoke else "")
    started = time.perf_counter()
    remaining = lambda: args.max_seconds - (time.perf_counter() - started)

    out = args.output_dir
    runs_dir = out / "runs"; runs_dir.mkdir(parents=True, exist_ok=True)
    shared = ("teacher_cache.pt", "selectors.json", "sampling.json", "preliminary.json")
    for previous in [*args.resume_from, *args.aggregate_from]:
        for name in shared:
            source, target = Path(previous) / name, out / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target); print("reused", name, "from", previous)
    for previous in args.resume_from:
        for path in sorted((Path(previous) / "runs").glob("*")):
            target = runs_dir / path.name
            if path.is_file() and not target.exists():
                shutil.copy2(path, target); print("resumed", path.name)

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
    official_snapshot = _snapshot(parameters_by_name)   # restored before every damage
    projector_names = [n for n in parameters_by_name if n.startswith("prj.")]
    if not projector_names:
        raise RuntimeError("no trainable projector parameters found")
    latent_positions = int(cfg.eval.latent_iterations)
    vocab_limit = int(tokenizer.eos_token_id) + 1

    # ---- data: §86 splits (same seed and order) + train drawn for the largest budget
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    if len(test) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("GSM8K test count drifted")
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
    sampling_path = out / "sampling.json"
    if sampling_path.is_file():
        previous = json.loads(sampling_path.read_text())
        if previous["hashes"] != sampling["hashes"]:
            raise RuntimeError("resumed output was produced from different splits")
    _atomic_json(sampling, sampling_path)
    print("sampling", {k: v for k, v in sampling.items() if k != "hashes"})

    # ---- teacher states (official checkpoint, explicit-CoT decision state)
    cache_path = out / "teacher_cache.pt"
    ordered = [r for name in ("fit", "select", "validate", "train") for r in splits[name]]
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu")
        if cache["sizes"] != {k: sizes[k] for k in ("fit", "select", "validate", "train")}:
            raise RuntimeError("teacher cache belongs to different splits")
    else:
        states, golds = precompute_teacher(model, tokenizer, ordered, batch_size=args.teacher_batch_size, device=device)
        cache = {"states": states, "gold": golds, "sizes": {k: sizes[k] for k in ("fit", "select", "validate", "train")},
                 "checkpoint_sha256": load_report.checkpoint_sha256}
        _atomic_torch_save(cache, cache_path)
    offsets, c = {}, 0
    for name in ("fit", "select", "validate", "train"):
        offsets[name] = (c, c + sizes[name]); c += sizes[name]
    cached = lambda name: (cache["states"][offsets[name][0]:offsets[name][1]], cache["gold"][offsets[name][0]:offsets[name][1]])

    # ---- selectors (teacher side, identical procedure to §86)
    selectors_path = out / "selectors.json"
    readout = readout_matrix(model, vocab_limit=vocab_limit).to(device, torch.float32)
    fit_states, _ = cached("fit")
    pca = fit_teacher_pca(fit_states.float())
    if selectors_path.is_file():
        selectors = json.loads(selectors_path.read_text())
    else:
        select_states, select_gold = cached("select"); select_states = select_states.to(device)
        validate_states, validate_gold = cached("validate"); validate_states = validate_states.to(device)
        _, trace = select_causal_greedy(select_states, select_gold, pca, readout, max(RANK_GRID),
                                        candidates=candidates, return_trace=True)
        order = [t["added"] for t in trace]
        reports, sets_by_rank = {}, {}
        for rank in RANK_GRID:
            sets = {"variance": select_variance(rank),
                    "relevance": select_relevance(select_states, select_gold, pca, readout, rank),
                    "causal": sorted(order[:rank]),
                    "random": select_random(rank, pca.basis.shape[1], seed=RANDOM_PC_SEED + rank)}
            sets_by_rank[rank] = sets
            reports[rank] = evaluate_index_sets(validate_states, validate_gold, pca, readout, sets)
        chosen, rank_audit = choose_rank(reports, minimum_gap=MINIMUM_GAP, retention_floor=RETENTION_FLOOR)
        if chosen is None and smoke:
            chosen = RANK_GRID[0]
        selectors = {"rank": chosen, "audit": rank_audit, "greedy_trace": trace,
                     "reports_by_rank": {str(k): v for k, v in reports.items()},
                     "sets": {str(k): v for k, v in sets_by_rank.items()},
                     "eigenvalue_share_top4": float(pca.eigenvalues[:4].sum() / pca.eigenvalues.sum())}
        _atomic_json(selectors, selectors_path)
        del select_states, validate_states
    rank = selectors["rank"]
    sets = selectors["sets"][str(rank)] if rank is not None else None
    shared_pcs = len(set(sets["variance"]) & set(sets["causal"])) if sets else None
    print("rank", rank, "shared variance/causal PCs", shared_pcs)

    targets = {"none": None}
    if sets is not None:
        targets["full"] = build_target(pca, fit_states.float(), None).to(device)
        for name in ("variance", "relevance", "causal", "random"):
            targets[name] = build_target(pca, fit_states.float(), sets[name]).to(device)

    # ---- damage, applied from the official weights every time
    noise_sigma = {"value": None}   # calibrated in the go/no-go for projector_noise

    def apply_damage(seed: int, *, sigma: float | None = None) -> dict:
        _restore(parameters_by_name, official_snapshot)
        if args.damage == "projector":
            return {"mode": "projector", **reset_projector(model, seed=seed)}
        if args.damage == "projector_noise":
            sigma = noise_sigma["value"] if sigma is None else sigma
            if sigma is None:
                raise RuntimeError("projector_noise sigma has not been calibrated")
            return {"mode": "projector_noise", **add_projector_noise(model, sigma=sigma, seed=seed)}
        return {"mode": "lora_half", **scale_lora_b(model, factor=LORA_HALF_FACTOR)}

    eval_kw = dict(latent_positions=latent_positions, batch_size=args.eval_batch_size, device=device)
    selection_rows, train_rows = splits["selection"], splits["train"]
    train_states, _ = cached("train")

    @torch.no_grad()
    def term_losses(rows, states) -> dict:
        model.eval(); sums = {k: 0.0 for k in targets if k != "none"}; n = 0
        for start in range(0, len(rows), args.eval_batch_size):
            batch = collate_official_codi_kv_rows(tokenizer, rows[start:start + args.eval_batch_size],
                                                  bot_token_id=model.bot_id, enforce_answer_eligibility=False).to(device)
            student = student_trajectory_forward(model, batch, latent_positions=latent_positions)
            state = student.answer_endpoint_hidden[:, READOUT_STATE, :]
            teacher = states[start:start + args.eval_batch_size].to(device, torch.float32)
            for k in sums:
                sums[k] += float(distillation_loss(state, teacher, targets[k])) * len(teacher)
            n += len(teacher)
        return {k: v / max(1, n) for k, v in sums.items()}

    # ---- go/no-go
    preliminary_path = out / "preliminary.json"
    preliminary = json.loads(preliminary_path.read_text()) if preliminary_path.is_file() else {}
    prelim_key = f"gate_{args.damage}"
    if prelim_key not in preliminary:
        fit_rows = splits["fit"][:term_rows]; fit_teacher = fit_states[:term_rows]
        _restore(parameters_by_name, official_snapshot)
        official_sel, _, _, _ = selection_metrics(model, tokenizer, selection_rows, **eval_kw)
        official_terms = term_losses(fit_rows, fit_teacher) if sets else {}
        sigma_sweep = None
        if args.damage == "projector_noise":
            sigma_sweep = []
            for sigma in NOISE_GRID:
                apply_damage(PILOT_SEEDS[0], sigma=sigma)
                metrics, _, _, _ = selection_metrics(model, tokenizer, selection_rows, **eval_kw)
                sigma_sweep.append({"sigma": sigma, "selection_accuracy": metrics["accuracy"],
                                    "gap": official_sel["accuracy"] - metrics["accuracy"]})
                print("sigma sweep", sigma_sweep[-1])
            qualifying = [s for s in sigma_sweep if s["gap"] >= HEADROOM_MIN_GAP]
            noise_sigma["value"] = (qualifying[0] if qualifying else sigma_sweep[-1])["sigma"]
        damage_report = apply_damage(PILOT_SEEDS[0])
        damaged_sel, _, _, _ = selection_metrics(model, tokenizer, selection_rows, **eval_kw)
        damaged_terms = term_losses(fit_rows, fit_teacher) if sets else {}
        ratios = {k: (damaged_terms[k] / official_terms[k] if official_terms.get(k) else None) for k in damaged_terms}
        preliminary[prelim_key] = {
            "damage": damage_report, "official_selection": official_sel, "damaged_selection": damaged_sel,
            "official_term_loss": official_terms, "damaged_term_loss": damaged_terms, "term_ratio": ratios,
            "rank": rank, "shared_variance_causal_pcs": shared_pcs,
            "sigma_sweep": sigma_sweep, "sigma": noise_sigma["value"]}
    gate0 = preliminary[prelim_key]
    noise_sigma["value"] = gate0.get("sigma")
    # Checks are derived from the stored measurements on every run (ledger 95 amendment),
    # so an earlier output's cache is reusable after a rule change.
    official_acc, damaged_acc = gate0["official_selection"]["accuracy"], gate0["damaged_selection"]["accuracy"]
    ratios = gate0["term_ratio"]
    gate0["gap"] = official_acc - damaged_acc
    gate0["recovery_threshold"] = recovery_threshold(official_acc, damaged_acc)
    gate0["checks"] = {
        "headroom": gate0["gap"] >= HEADROOM_MIN_GAP,
        "selectors_distinguishable": gate0["rank"] is not None,
        "selector_divergence": gate0["shared_variance_causal_pcs"] is not None and gate0["shared_variance_causal_pcs"] <= MAX_SHARED_PCS,
        "term_not_converged": bool(ratios) and all((ratios.get(k) or 0) >= TERM_RATIO_MIN for k in ("variance", "causal")),
    }
    _atomic_json(preliminary, preliminary_path)
    threshold = gate0["recovery_threshold"]
    print("go/no-go", gate0["checks"], "official/damaged selection EM", official_acc, damaged_acc,
          "recovery threshold", round(threshold, 3))
    prechecks_ok = all(gate0["checks"].values()) or smoke

    def train_one(arm: str, seed: int, steps: int, *, tag: str):
        base, multiplier = parse_arm(arm)
        name = f"{tag}{arm}_seed{seed}"
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
        target = targets[base]
        model.train(); t0 = time.perf_counter()
        base_lrs = [LEARNING_RATE_PROJECTOR, LEARNING_RATE_LORA]

        def save_ckpt(step):
            _atomic_torch_save({"trainable": _snapshot(parameters_by_name), "optimizer": optimizer.state_dict(),
                                "step": step, "curve": curve, "elapsed": elapsed + time.perf_counter() - t0,
                                "damage": damage_report, "cpu_rng": torch.get_rng_state(),
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
            result = student_training_step(model, batches, teacher_states, target, parameters,
                                           latent_positions=latent_positions, auxiliary_multiplier=multiplier)
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, result.gradients):
                parameter.grad = None if gradient is None else gradient.detach()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            window.append({"answer_loss": result.answer_loss, "distillation_loss": result.distillation_loss,
                           "distillation_scale": result.distillation_scale, "gradient_cosine": result.gradient_cosine})
            del batches, teacher_states, result
            done = step + 1
            if done % curve_every == 0 or done == steps:
                metrics, _, _, _ = selection_metrics(model, tokenizer, selection_rows, **eval_kw)
                model.train()
                mean = lambda key: (sum(r[key] for r in window) / len(window)) if window and window[0][key] is not None else None
                curve.append({"step": done, **metrics, "train_answer_loss": mean("answer_loss"),
                              "train_distillation_loss": mean("distillation_loss"),
                              "distillation_scale": mean("distillation_scale"), "gradient_cosine": mean("gradient_cosine")})
                window = []
                print("curve", name, curve[-1])
            if done % checkpoint_every == 0 and done < steps:
                save_ckpt(done)
        training_seconds = elapsed + time.perf_counter() - t0
        optimizer.zero_grad(set_to_none=True); del optimizer; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        test_metrics, test_nll, test_correct, test_outputs = selection_metrics(model, tokenizer, test_rows, **eval_kw)
        record = {"arm": arm, "base_arm": base, "multiplier": multiplier, "seed": seed, "damage": args.damage,
                  "pilot": tag != "", "rank": rank, "steps": steps, "training_seconds": training_seconds,
                  "damage_report": damage_report, "curve": curve,
                  "recovery_threshold": threshold,
                  "steps_to_threshold": steps_to_threshold(curve, threshold),
                  "mean_gradient_cosine": (sum(p["gradient_cosine"] for p in curve if p["gradient_cosine"] is not None)
                                           / max(1, sum(p["gradient_cosine"] is not None for p in curve))) if curve else None,
                  "test": test_metrics, "test_correct": test_correct, "test_nll": test_nll, "test_outputs": test_outputs}
        _atomic_json(record, record_path)
        if ckpt_path.is_file():
            ckpt_path.unlink()
        print("test", name, test_metrics)
        return record, "done"

    summary = {
        "schema_version": 1, "contract": contract, "smoke": smoke, "damage": args.damage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
        "preregistration": {
            "arms": list(arms), "seeds_this_run": list(seeds), "step_options": list(STEP_OPTIONS),
            "pilot_steps": pilot_steps, "pilot_seeds": list(PILOT_SEEDS), "noise_grid": list(NOISE_GRID),
            "sigma": noise_sigma["value"], "batch_size": args.batch_size, "micro_batch_size": args.micro_batch_size,
            "sizes": sizes, "learning_rate_lora": LEARNING_RATE_LORA, "learning_rate_projector": LEARNING_RATE_PROJECTOR,
            "warmup_steps": WARMUP_STEPS, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "rank_grid": list(RANK_GRID), "minimum_gap": MINIMUM_GAP, "retention_floor": RETENTION_FLOOR,
            "headroom_min_gap": HEADROOM_MIN_GAP, "max_shared_pcs": MAX_SHARED_PCS, "term_ratio_min": TERM_RATIO_MIN,
            "recovery_fraction": RECOVERY_FRACTION, "recovery_threshold": threshold, "null_width": NULL_WIDTH,
            "data_seed": DATA_SEED,
            "test_examples": len(test_rows),
            "gates": ["r0 none reaches damaged + 0.5*gap on the selection split", "s1 full-none", "h1 causal-variance", "h2 causal-relevance",
                      "h3 causal-random", "variance-none and causal-none two-sided",
                      "headline = h1 and h3; null if h1 covers 0 within 3 points"]},
        "sampling": {k: v for k, v in sampling.items() if k != "hashes"},
        "selectors": {"rank": rank, "audit": selectors["audit"], "shared_variance_causal_pcs": shared_pcs,
                      "sets": sets, "eigenvalue_share_top4": selectors["eigenvalue_share_top4"],
                      "validation": selectors["reports_by_rank"].get(str(rank)) if rank is not None else None},
        "preliminary": preliminary, "runs": {}, "test": None, "status": "running",
        "decision": {"preliminary_passed": bool(prechecks_ok), "claim": "pending"},
        "warnings": [
            "Recovery regime: the official checkpoint's projector (noise or reset) or LoRA B is damaged and re-learned; this is continued training, not learning from scratch (ledger 93/94/96).",
            "Primary is GSM8K test exact match on 1,319 questions at the final step; arms are paired by seed.",
            "Distillation gradients are norm-matched to the answer cross-entropy; weight arms rescale that match."]}
    _atomic_json(summary, out / "summary.json")
    if not prechecks_ok:
        summary["status"] = "stopped"
        summary["decision"]["claim"] = "STOP: go/no-go failed before the pilot (see preliminary); nothing was trained"
        _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary

    # ---- recovery pilot decides the step budget (shared across damage modes only if both pass)
    pilot_key = f"pilot_{args.damage}"
    if pilot_key not in preliminary:
        pilots = []
        for pilot_seed in (PILOT_SEEDS if not smoke else PILOT_SEEDS[:1]):
            record, status = train_one("none", pilot_seed, pilot_steps, tag="pilot_")
            if status == "paused":
                summary["status"] = "paused"; summary["decision"]["claim"] = "PAUSED during the pilot: publish this output and rerun with it attached"
                _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary
            pilots.append({"seed": pilot_seed, "curve": record["curve"], "test": record["test"],
                           "steps_to_threshold": record["steps_to_threshold"],
                           "budget": choose_steps(record["curve"], threshold=threshold)})
        chosen_steps = (choose_steps_for_pilots([p["curve"] for p in pilots], threshold=threshold)
                        if not smoke else S["steps"])
        preliminary[pilot_key] = {"pilots": pilots, "chosen_steps": chosen_steps,
                                  # kept for readers of the first-run layout
                                  "curve": pilots[0]["curve"], "test": pilots[0]["test"],
                                  "steps_to_threshold": pilots[0]["steps_to_threshold"]}
        _atomic_json(preliminary, preliminary_path)
    steps = preliminary[pilot_key]["chosen_steps"]
    summary["preliminary"] = preliminary
    summary["preregistration"]["steps"] = steps
    print("pilots", [{k: v for k, v in p.items() if k != "curve"} for p in preliminary[pilot_key].get("pilots", [])],
          "budget", steps)
    if steps is None:
        summary["status"] = "stopped"
        summary["decision"]["claim"] = "STOP: not every none pilot recovered half the lost accuracy within 2,000 steps (or the pilots disagreed); the recovery regime is not reliable at this budget"
        _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary
    if args.preliminary_only:
        summary["status"] = "preliminary_only"
        summary["decision"]["claim"] = f"GO: go/no-go passed and the pilot fixed the step budget at {steps}; training not requested in this session"
        _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary

    for arm in arms:
        if parse_arm(arm)[0] != "none" and targets.get(parse_arm(arm)[0]) is None:
            raise RuntimeError(f"arm {arm} has no target (rank rule failed)")
        for seed in seeds:
            _, status = train_one(arm, seed, steps, tag="")
            if status == "paused":
                summary["status"] = "paused"
                summary["decision"]["claim"] = f"PAUSED before {arm} seed {seed} finished: publish this output and rerun with it attached"
                _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary
    summary["status"] = "trained"
    return aggregate(args, summary, test_rows, out)


def aggregate(args, summary, test_rows, out):
    runs = {}
    for directory in [out, *[Path(p) for p in args.aggregate_from]]:
        for path in sorted((Path(directory) / "runs").glob("*.json")):
            record = json.loads(path.read_text())
            if "test" in record and record.get("damage") == args.damage and not record.get("pilot"):
                runs[f"{record['arm']}_seed{record['seed']}"] = record
    by_arm = {}
    for record in runs.values():
        by_arm.setdefault(record["arm"], []).append(record)
    if not by_arm:
        summary["decision"]["claim"] = "no completed runs to aggregate"; _atomic_json(summary, out / "summary.json"); return summary
    n = len(next(iter(runs.values()))["test_correct"])
    mean_correct = {a: torch.tensor([r["test_correct"] for r in rs], dtype=torch.float32).mean(0).tolist() for a, rs in by_arm.items()}
    mean_nll = {a: torch.tensor([r["test_nll"] for r in rs], dtype=torch.float32).mean(0).tolist() for a, rs in by_arm.items()}
    comparisons, nll_comparisons = {}, {}
    for i, (a, b) in enumerate(COMPARISONS):
        if a in mean_correct and b in mean_correct:
            comparisons[f"{a}_minus_{b}"] = _paired(mean_correct[a], mean_correct[b], seed=args.seed + i, samples=args.bootstrap_samples)
            nll_comparisons[f"{b}_minus_{a}_nll"] = _paired(mean_nll[b], mean_nll[a], seed=args.seed + 50 + i, samples=args.bootstrap_samples)
    threshold = summary["preliminary"][f"gate_{args.damage}"]["recovery_threshold"]
    none_selection = None
    if "none" in by_arm:
        finals = [r["curve"][-1]["accuracy"] for r in by_arm["none"] if r["curve"]]
        none_selection = sum(finals) / len(finals) if finals else None
    gate = gates_from(comparisons, none_selection, threshold)
    arm_results = {a: {"seeds": sorted(r["seed"] for r in rs),
                       "test_accuracy_by_seed": [r["test"]["accuracy"] for r in rs],
                       "test_accuracy_mean": float(torch.tensor(mean_correct[a]).mean()),
                       "test_nll_mean": float(torch.tensor(mean_nll[a]).mean()),
                       "steps_to_threshold": [r["steps_to_threshold"] for r in rs],
                       "mean_gradient_cosine": [r["mean_gradient_cosine"] for r in rs],
                       "final_selection": [r["curve"][-1] if r["curve"] else None for r in rs],
                       "training_seconds_mean": sum(r["training_seconds"] for r in rs) / len(rs)} for a, rs in by_arm.items()}
    summary["runs"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("test_correct", "test_nll", "test_outputs")} for k, v in runs.items()}
    summary["test"] = {"examples": n, "arms": arm_results, "comparisons": comparisons, "nll_comparisons": nll_comparisons, "gate": gate}
    summary["status"] = "aggregated"
    core = all(k in comparisons for k in ("causal_minus_variance", "causal_minus_random"))
    summary["decision"] = {"preliminary_passed": True,
                           "claim": claim_from(gate) if core else f"partial aggregate over arms {sorted(by_arm)}: {gate}"}
    _atomic_json(summary, out / "summary.json")
    _atomic_torch_save({**summary, "test_correct": mean_correct, "test_nll": mean_nll,
                        "per_run_test_correct": {k: v["test_correct"] for k, v in runs.items()}},
                       out / "recovery_subspace_distillation.pt")
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
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--damage", default="projector_noise", choices=DAMAGES)
    parser.add_argument("--arms", default=",".join(ARMS_PRIMARY), type=lambda s: tuple(a for a in s.split(",") if a))
    parser.add_argument("--seeds", default=SEEDS_DEFAULT)
    parser.add_argument("--resume-from", default=[], type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--aggregate-from", default=[], type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--preliminary-only", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30_600)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--teacher-batch-size", type=int, default=TEACHER_BATCH)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_925)
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
