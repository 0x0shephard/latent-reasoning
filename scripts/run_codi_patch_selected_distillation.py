"""Patch-selected two-term distillation in the §94 repair regime (ledger §101).

Teaching set: teacher PCs whose coordinates, patched into the damaged student's
decision state, most raise the student's first-token accuracy (transfer patching).
Anchor set: PCs in which the student's drift from the official model's own latent
state breaks the most official answers (breakage patching), excluding the teaching
set.  Loss = CE + smooth-L1 on the teaching coordinates (norm-matched, x1) + smooth-L1
on the anchor coordinates (norm-matched, x0.3).  Arms are paired with the §100 runs of
the same seed, whose per-question correctness is read from the published primary
output; no baseline is retrained.
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
    breakage_score,
    build_target,
    cosine_lr,
    fit_teacher_pca,
    jaccard,
    readout_matrix,
    select_breakage_anchor,
    select_transfer_patch,
    student_training_step_multi,
    transfer_patch_score,
)
from src.mech.trajectory_supervision import student_trajectory_forward
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_patch_selected_distillation_v1"
ARMS = ("patch", "patch_anchor", "patch_anchor_adaptive", "causal_anchor")
REFERENCE_ARMS = ("none", "full", "causal", "relevance", "random", "variance")
TEACHING_RANK = 12
ANCHOR_RANK = 24
ANCHOR_MULTIPLIER = 0.3
TEACHING_MULTIPLIER = 1.0
RESELECT_EVERY = 250
SELECT_POOL = 1_024
ANCHOR_RANDOM_SETS = 20
ANCHOR_RATIO_MIN = 2.0
CURVE_EVERY = 100
CHECKPOINT_EVERY = 250
SEEDS_DEFAULT = "1,2,3"
PREDICTED_REPAIRS = 59.0
PREDICTED_BREAKS = 36.6
COMPARISONS = (("patch_anchor", "full"), ("patch_anchor", "causal"), ("patch_anchor", "relevance"),
               ("patch", "causal"), ("patch_anchor_adaptive", "patch_anchor"), ("causal_anchor", "causal"),
               ("patch", "none"), ("patch_anchor", "none"), ("patch_anchor_adaptive", "none"),
               ("causal_anchor", "none"), ("patch_anchor_adaptive", "full"), ("causal_anchor", "full"))

SMOKE = {"select_pool": 32, "reselect_every": 4, "curve_every": 4, "checkpoint_every": 4,
         "candidates": 16, "teaching_rank": 3, "anchor_rank": 4, "random_sets": 3, "test": 16}


def arm_spec(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return {"teaching": "causal" if arm == "causal_anchor" else "patch",
            "anchor": arm != "patch", "adaptive": arm == "patch_anchor_adaptive"}


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
    for name, key in (("p1_beats_full", "patch_anchor_minus_full"), ("p2_beats_causal", "patch_anchor_minus_causal"),
                      ("p3_beats_relevance", "patch_anchor_minus_relevance"), ("p4_patch_beats_causal", "patch_minus_causal"),
                      ("p5_adaptive_helps", "patch_anchor_adaptive_minus_patch_anchor"),
                      ("p6_anchor_helps_causal", "causal_anchor_minus_causal")):
        if has(key):
            gate[name] = lb(key) > 0
            gate[name + "_all_seeds"] = allpos(key)
    core = ("patch_anchor_minus_full", "patch_anchor_minus_causal", "patch_anchor_minus_relevance")
    if all(has(k) for k in core):
        gate["tie"] = all(lb(k) <= 0 <= ub(k) and ub(k) - lb(k) <= NULL_WIDTH for k in core)
    return gate


def claim_from(gate: dict) -> str:
    if gate.get("p1_beats_full") and gate.get("p2_beats_causal") and gate.get("p3_beats_relevance") \
            and gate.get("p1_beats_full_all_seeds") and gate.get("p2_beats_causal_all_seeds") and gate.get("p3_beats_relevance_all_seeds"):
        return "CONFIRMED: the patch-selected two-term target beats full, causal and relevance in every seed"
    if gate.get("p6_anchor_helps_causal") and not gate.get("p1_beats_full"):
        return "ANCHOR: the breakage-patch anchor improves the causal target but does not beat the full state"
    if gate.get("tie"):
        return "TIE: the two-term target is within 3 points of full, causal and relevance"
    survivors = [k for k in ("p1_beats_full", "p2_beats_causal", "p3_beats_relevance", "p4_patch_beats_causal",
                             "p5_adaptive_helps", "p6_anchor_helps_causal") if gate.get(k)]
    return f"PARTIAL: {', '.join(survivors) if survivors else 'no gate passed'}"


def run(args):
    smoke = bool(args.smoke)
    S = SMOKE if smoke else None
    R = RECOVERY_SMOKE if smoke else None
    fit_n, select_n, validate_n = ((R["fit"], R["select"], R["validate"]) if smoke
                                   else (FIT_EXAMPLES, SELECT_EXAMPLES, VALIDATE_EXAMPLES))
    train_n = (R["pilot_steps"] if smoke else max(STEP_OPTIONS)) * BATCH_SIZE
    selection_n = R["selection"] if smoke else SELECTION_EXAMPLES
    pool_n = S["select_pool"] if smoke else SELECT_POOL
    reselect_every = S["reselect_every"] if smoke else RESELECT_EVERY
    curve_every = S["curve_every"] if smoke else CURVE_EVERY
    checkpoint_every = S["checkpoint_every"] if smoke else CHECKPOINT_EVERY
    candidates = S["candidates"] if smoke else CANDIDATE_PCS
    teaching_rank = S["teaching_rank"] if smoke else TEACHING_RANK
    anchor_rank = S["anchor_rank"] if smoke else ANCHOR_RANK
    random_sets = S["random_sets"] if smoke else ANCHOR_RANDOM_SETS
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
    select_states_t, select_gold = cached("select")
    pool_rows = splits["select"][:pool_n]
    pool_teacher = select_states_t[:pool_n].float().to(device)
    pool_gold = select_gold[:pool_n].to(device)
    train_states, _ = cached("train")
    train_rows, selection_rows = splits["train"], splits["selection"]
    eval_kw = dict(latent_positions=latent_positions, batch_size=args.eval_batch_size, device=device)

    @torch.no_grad()
    def student_pool_states() -> torch.Tensor:
        model.eval(); parts = []
        for start in range(0, len(pool_rows), args.eval_batch_size):
            batch = collate_official_codi_kv_rows(tokenizer, pool_rows[start:start + args.eval_batch_size],
                                                  bot_token_id=model.bot_id, enforce_answer_eligibility=False).to(device)
            parts.append(student_trajectory_forward(model, batch, latent_positions=latent_positions)
                         .answer_endpoint_hidden[:, READOUT_STATE, :].float())
        model.train()
        return torch.cat(parts)

    def apply_damage(seed: int) -> dict:
        _restore(parameters_by_name, official_snapshot)
        return {"mode": "projector_noise", **add_projector_noise(model, sigma=sigma, seed=seed)}

    # ---- official model's own latent-path states on the pool (anchor reference)
    reference_path = out / "student_reference.pt"
    if reference_path.is_file():
        official_pool = torch.load(reference_path, map_location="cpu")["states"].to(device)
    else:
        _restore(parameters_by_name, official_snapshot)
        official_pool = student_pool_states()
        _atomic_torch_save({"states": official_pool.cpu(), "pool": pool_n}, reference_path)

    def select_sets(student_pool: torch.Tensor, teaching_mode: str) -> dict:
        if teaching_mode == "causal":
            teaching, trace_t = causal_set, []
        else:
            teaching, trace_t = select_transfer_patch(student_pool, pool_teacher, pool_gold, pca, readout, teaching_rank,
                                                      candidates=candidates, return_trace=True)
        anchor, trace_a = select_breakage_anchor(official_pool, student_pool, pool_gold, pca, readout, anchor_rank,
                                                 candidates=candidates, exclude=teaching, return_trace=True)
        return {"teaching": teaching, "anchor": anchor, "teaching_trace": trace_t, "anchor_trace": trace_a,
                "jaccard_teaching_causal": jaccard(teaching, causal_set),
                "patched_accuracy": transfer_patch_score(student_pool, pool_teacher, pool_gold, pca, teaching, readout)[0],
                "unpatched_accuracy": transfer_patch_score(student_pool, pool_teacher, pool_gold, pca, [], readout)[0],
                "anchor_breakage": breakage_score(official_pool, student_pool, pool_gold, pca, anchor, readout)}

    # ---- go/no-go: the anchor must break more than random sets (seed 1 damage, step 0)
    key = "gate_patch_selected"
    if key not in preliminary:
        apply_damage(seeds[0])
        student_pool = student_pool_states()
        sets = select_sets(student_pool, "patch")
        rng = random.Random(20260926)
        pool_ids = [i for i in range(min(candidates, pca.basis.shape[1])) if i not in set(sets["teaching"])]
        random_breakage = [breakage_score(official_pool, student_pool, pool_gold, pca,
                                          sorted(rng.sample(pool_ids, min(anchor_rank, len(pool_ids)))), readout)
                           for _ in range(random_sets)]
        ratio = sets["anchor_breakage"] / max(1e-9, sum(random_breakage) / len(random_breakage))
        preliminary[key] = {"seed": seeds[0], "sigma": sigma, "steps": steps, **{k: v for k, v in sets.items() if "trace" not in k},
                            "random_breakage_mean": sum(random_breakage) / len(random_breakage),
                            "anchor_over_random": ratio,
                            "checks": {"anchor_meaningful": ratio >= ANCHOR_RATIO_MIN or smoke,
                                       "teaching_differs_from_causal": sets["jaccard_teaching_causal"] < 0.75}}
        _atomic_json(preliminary, out / "preliminary.json")
    gate1 = preliminary[key]
    print("go/no-go", gate1["checks"], "| teaching", gate1["teaching"], "| jaccard with causal",
          round(gate1["jaccard_teaching_causal"], 2), "| anchor breakage", round(gate1["anchor_breakage"], 3),
          "vs random", round(gate1["random_breakage_mean"], 3))

    summary = {
        "schema_version": 1, "contract": contract, "smoke": smoke, "damage": "projector_noise",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
        "preregistration": {
            "arms": list(arms), "seeds_this_run": list(seeds), "steps": steps, "sigma": sigma,
            "teaching_rank": teaching_rank, "anchor_rank": anchor_rank, "teaching_multiplier": TEACHING_MULTIPLIER,
            "anchor_multiplier": ANCHOR_MULTIPLIER, "reselect_every": reselect_every, "select_pool": pool_n,
            "candidates": candidates, "anchor_ratio_min": ANCHOR_RATIO_MIN, "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size, "learning_rate_lora": LEARNING_RATE_LORA,
            "learning_rate_projector": LEARNING_RATE_PROJECTOR, "warmup_steps": WARMUP_STEPS,
            "predicted_repairs_at_least": PREDICTED_REPAIRS, "predicted_breaks_below": PREDICTED_BREAKS,
            "reference_arms": list(REFERENCE_ARMS), "reference_from": str(reference_dir),
            "gates": ["p1 patch_anchor-full", "p2 patch_anchor-causal", "p3 patch_anchor-relevance",
                      "p4 patch-causal", "p5 adaptive-fixed", "p6 causal_anchor-causal",
                      "CONFIRMED = p1&p2&p3 with 3/3 seeds; ANCHOR = p6 & !p1; TIE within 3 points"]},
        "sampling": {k: v for k, v in sampling.items() if k != "hashes"},
        "selectors": {"rank": rank, "causal_set": causal_set},
        "preliminary": preliminary, "runs": {}, "test": None, "status": "running",
        "decision": {"preliminary_passed": all(gate1["checks"].values()) or smoke, "claim": "pending"},
        "warnings": [
            "Repair regime (ledger 94-100): projector noise on the official checkpoint, then 1,000 steps; not learning from scratch.",
            "Designed after the section-100 primary; new arms are paired with its runs at the same seeds via the published output.",
            "The anchor reference is the official model's own latent state, which exists only in the repair regime."]}
    _atomic_json(summary, out / "summary.json")
    if not gate1["checks"]["anchor_meaningful"] and not smoke:
        summary["status"] = "stopped"
        summary["decision"]["claim"] = "STOP: the breakage-patch anchor does not break more than random sets; nothing was trained"
        _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary
    if args.preliminary_only:
        summary["status"] = "preliminary_only"; summary["decision"]["claim"] = "GO: anchor meaningful; training not requested"
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
        step0, curve, elapsed, window, history = 0, [], 0.0, [], []
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _restore(parameters_by_name, ckpt["trainable"]); optimizer.load_state_dict(ckpt["optimizer"])
            step0, curve, elapsed, damage_report, history = (int(ckpt["step"]), ckpt["curve"], float(ckpt["elapsed"]),
                                                             ckpt["damage"], ckpt["history"])
            torch.set_rng_state(ckpt["cpu_rng"])
            if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
            print("resume", name, "at", step0)
        else:
            damage_report = apply_damage(seed)
            sets = select_sets(student_pool_states(), spec_["teaching"])
            history.append({"step": 0, **{k: v for k, v in sets.items() if "trace" not in k}})
            print("selected", name, "teaching", sets["teaching"], "anchor", sets["anchor"])

        def make_terms(sets_):
            terms = [(build_target(pca, fit_states.float(), sets_["teaching"]).to(device), TEACHING_MULTIPLIER)]
            if spec_["anchor"]:
                terms.append((build_target(pca, fit_states.float(), sets_["anchor"]).to(device), ANCHOR_MULTIPLIER))
            return terms

        terms = make_terms(history[-1])
        model.train(); t0 = time.perf_counter()
        base_lrs = [LEARNING_RATE_PROJECTOR, LEARNING_RATE_LORA]

        def save_ckpt(step):
            _atomic_torch_save({"trainable": _snapshot(parameters_by_name), "optimizer": optimizer.state_dict(),
                                "step": step, "curve": curve, "history": history, "damage": damage_report,
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
            result = student_training_step_multi(model, batches, teacher_states, terms, parameters, latent_positions=latent_positions)
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, result.gradients):
                parameter.grad = None if gradient is None else gradient.detach()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            window.append({"answer_loss": result.answer_loss, "distillation_loss": result.distillation_loss,
                           "distillation_scale": result.distillation_scale, "gradient_cosine": result.gradient_cosine})
            del batches, teacher_states, result
            done = step + 1
            if spec_["adaptive"] and done % reselect_every == 0 and done < steps:
                sets = select_sets(student_pool_states(), spec_["teaching"])
                history.append({"step": done, **{k: v for k, v in sets.items() if "trace" not in k},
                                "jaccard_with_previous_teaching": jaccard(sets["teaching"], history[-1]["teaching"]),
                                "jaccard_with_previous_anchor": jaccard(sets["anchor"], history[-1]["anchor"])})
                terms = make_terms(sets)
                print("reselected", name, "at", done, "teaching", sets["teaching"])
            if done % curve_every == 0 or done == steps:
                metrics, _, _, _ = selection_metrics(model, tokenizer, selection_rows, **eval_kw)
                model.train()
                mean = lambda k: (sum(r[k] for r in window) / len(window)) if window and window[0][k] is not None else None
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
        record = {"arm": arm, "seed": seed, "damage": "projector_noise", "steps": steps, "training_seconds": training_seconds,
                  "damage_report": damage_report, "selection_history": history, "curve": curve,
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
    if "patch_anchor" in decomposition:
        gate["predicted_repairs_met"] = decomposition["patch_anchor"]["repairs_mean"] >= PREDICTED_REPAIRS
        gate["predicted_breaks_met"] = decomposition["patch_anchor"]["breaks_mean"] < PREDICTED_BREAKS
    arm_results = {arm: {"seeds": usable[arm], "test_accuracy_by_seed": [sum(correct[arm][s]) / n for s in usable[arm]],
                         "test_accuracy_mean": float(torch.tensor(mean[arm]).mean()),
                         "source": "this_run" if arm in ARMS else "reference"} for arm in mean}
    for k, record in runs.items():
        arm_results[record["arm"]].setdefault("selection_history", {})[k] = record["selection_history"]
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
    parser.add_argument("--seed", type=int, default=20_260_926)
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
