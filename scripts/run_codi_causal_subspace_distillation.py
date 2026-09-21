"""Causal versus variance selection of the distillation subspace (ledger §86).

Teacher: the frozen official CODI checkpoint's explicit-CoT path; its state-12
colon states are precomputed once.  Subspaces are index sets over the teacher's
own principal components, chosen by variance, gradient relevance, greedy
retain-only intervention, or at random, and validated on the teacher before any
training.  Students share the teacher's embeddings and readout but learn the
latent task from fresh adapters, distilled toward the selected coordinates with the
distillation gradient norm-matched to the answer cross-entropy.  Runs checkpoint
and resume across Kaggle sessions; two accounts split the seeds; ``--aggregate-from``
combines their outputs.
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
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    verify_full_reproduction_gate,
)
from scripts.run_codi_preanswer_kv_subspace_discovery import _sha
from src.data.datasets import load_eval_set, load_train_set
from src.data.official_codi_training import (
    align_official_codi_gsm8k_eval_rows,
    collate_official_codi_kv_rows,
    encode_official_codi_row,
    official_codi_row_is_eligible,
)
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import official_answers_match
from src.mech.causal_subspace_distillation import (
    ARMS,
    READOUT_STATE,
    SUBSPACE_ARMS,
    build_target,
    choose_rank,
    cosine_lr,
    evaluate_index_sets,
    fit_teacher_pca,
    readout_matrix,
    reinitialize_student,
    select_causal_greedy,
    select_random,
    select_relevance,
    select_variance,
    student_training_step,
    teacher_decision_states,
)
from src.mech.rank16_xkv_confirmation import paired_bootstrap_interval
from src.mech.trajectory_supervision import student_trajectory_forward
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_causal_subspace_distillation_v1"
FIT_EXAMPLES = SELECT_EXAMPLES = VALIDATE_EXAMPLES = 2_048
STEPS = 10_000
BATCH_SIZE = 16
MICRO_BATCH_SIZE = 8  # memory only; the optimisation step is always the full batch
TRAIN_EXAMPLES = STEPS * BATCH_SIZE
SELECTION_EXAMPLES = 256
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP = 2.0
RANK_GRID = (8, 12, 16)
CANDIDATE_PCS = 128
RETENTION_FLOOR = 0.50
MINIMUM_GAP = 0.05
DATA_SEED = 20_260_923
CURVE_EVERY = 1_000
CHECKPOINT_EVERY = 500
TEACHER_BATCH = 64
MAX_NEW_TOKENS = 64
RANDOM_PC_SEED = 20_260_923

SMOKE = {"fit": 64, "select": 64, "validate": 64, "steps": 20, "selection": 16, "test": 16,
         "curve_every": 10, "checkpoint_every": 10, "arms": ("none", "variance", "causal"),
         "candidates": 16}


# ------------------------------------------------------------------------- data


def prepare_row(tokenizer, row: dict, *, bot_token_id: int) -> dict | None:
    try:
        if not official_codi_row_is_eligible(row):
            return None
        encode_official_codi_row(tokenizer, row, bot_token_id=bot_token_id)
    except (KeyError, TypeError, ValueError):
        return None
    gold = str(row["answer"]).split(" ")[-1].replace("####", "").replace(",", "").strip()
    try:
        float(gold)
    except ValueError:
        return None
    return {"question": str(row["question"]), "cot": str(row["cot"]),
            "answer": str(row["answer"]), "gold": gold}


def sample_splits(dataset, tokenizer, *, test_questions, sizes: dict, seed: int, bot_token_id: int):
    groups: dict[str, list[int]] = {}
    for index, question in enumerate(dataset["question"]):
        groups.setdefault(_normalized_question(question), []).append(index)
    overlap = sorted(set(groups).intersection(test_questions))
    if overlap:
        raise RuntimeError(f"GSM8k-Aug/test question overlap is nonzero ({len(overlap)})")
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    needed = sum(sizes.values())
    prepared, skipped = [], 0
    for key in keys:
        if len(prepared) >= needed:
            break
        row = prepare_row(tokenizer, dataset[generator.choice(groups[key])], bot_token_id=bot_token_id)
        if row is None:
            skipped += 1
            continue
        prepared.append(row)
    if len(prepared) < needed:
        raise ValueError(f"only {len(prepared)} eligible rows for {needed} requested")
    splits, start = {}, 0
    for name, size in sizes.items():
        splits[name] = prepared[start : start + size]
        start += size
    audit = {"unique_questions": len(groups), "skipped_candidates": skipped, "data_seed": seed,
             "sizes": dict(sizes),
             "hashes": {name: _sha([_normalized_question(r["question"]) for r in rows])
                        for name, rows in splits.items()}}
    return splits, audit


# ---------------------------------------------------------------------- helpers


def _trainable(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def _snapshot(parameters_by_name):
    return {n: p.detach().cpu().clone() for n, p in parameters_by_name.items()}


def _restore(parameters_by_name, snapshot):
    with torch.no_grad():
        for n, p in parameters_by_name.items():
            p.copy_(snapshot[n].to(device=p.device, dtype=p.dtype))


def precompute_teacher(model, tokenizer, rows, *, batch_size, device):
    states, golds = [], []
    model.eval()
    for start in tqdm(range(0, len(rows), batch_size), desc="teacher states", unit="batch"):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start : start + batch_size], bot_token_id=model.bot_id,
            enforce_answer_eligibility=False,
        ).to(device)
        state, gold = teacher_decision_states(model, batch)
        states.append(state.to(torch.float16).cpu())
        golds.append(gold.cpu())
        del batch
    return torch.cat(states), torch.cat(golds)


def selection_metrics(model, tokenizer, rows, *, latent_positions, batch_size, device):
    """Teacher-forced answer NLL and native-decoding exact match on a small split."""
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = collate_official_codi_kv_rows(
                tokenizer, rows[start : start + batch_size], bot_token_id=model.bot_id,
                enforce_answer_eligibility=False,
            ).to(device)
            out = student_trajectory_forward(model, batch, latent_positions=latent_positions)
            losses.append(out.per_example_loss.float().cpu())
        outputs = generate_official_codi(
            model, tokenizer, [r["question"] for r in rows], latent_iterations=latent_positions,
            max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size, device=device,
        )
    nll = torch.cat(losses)
    correct = [bool(official_answers_match(t, r["gold"])) for t, r in zip(outputs, rows)]
    return {"answer_nll": float(nll.mean()), "accuracy": sum(correct) / len(rows),
            "correct": int(sum(correct)), "examples": len(rows)}, nll.tolist(), correct, outputs


def _paired(a, b, *, seed, samples):
    diff = torch.tensor(a, dtype=torch.float32) - torch.tensor(b, dtype=torch.float32)
    return {"mean_difference": float(diff.mean()),
            "bootstrap_95ci": paired_bootstrap_interval(diff, seed=seed, samples=samples)}


def gates_from(comparisons: dict) -> dict:
    lb = lambda k: comparisons[k]["bootstrap_95ci"][0]
    gate = {"s1_distillation_detectable": lb("full_minus_none") > 0,
            "h1_causal_beats_variance": lb("causal_minus_variance") > 0,
            "h2_causal_beats_relevance": lb("causal_minus_relevance") > 0,
            "h3_causal_beats_random": lb("causal_minus_random") > 0,
            "variance_beats_random": lb("variance_minus_random") > 0}
    gate["headline"] = gate["s1_distillation_detectable"] and gate["h1_causal_beats_variance"] and gate["h3_causal_beats_random"]
    return gate


def claim_from(gate: dict) -> str:
    if not gate["s1_distillation_detectable"]:
        return "STOP: distillation signal not detectable at this budget; subspace comparisons are uninformative"
    if gate["headline"]:
        return "CONFIRMED: causally selected distillation subspace beats variance selection and random at matched rank"
    if gate["h3_causal_beats_random"] and not gate["h1_causal_beats_variance"]:
        return "NULL: causal and variance selection transfer equally; the inert top PCs cost nothing at this budget"
    return "STOP: causal selection did not separate from its controls"


# ------------------------------------------------------------------------- run


def run(args):
    smoke = bool(args.smoke)
    fit_n = SMOKE["fit"] if smoke else FIT_EXAMPLES
    select_n = SMOKE["select"] if smoke else SELECT_EXAMPLES
    validate_n = SMOKE["validate"] if smoke else VALIDATE_EXAMPLES
    steps = SMOKE["steps"] if smoke else STEPS
    train_n = steps * args.batch_size
    selection_n = SMOKE["selection"] if smoke else SELECTION_EXAMPLES
    curve_every = SMOKE["curve_every"] if smoke else CURVE_EVERY
    checkpoint_every = SMOKE["checkpoint_every"] if smoke else CHECKPOINT_EVERY
    candidates = SMOKE["candidates"] if smoke else CANDIDATE_PCS
    arms = tuple(a for a in (SMOKE["arms"] if smoke else ARMS) if a in args.arms)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    contract = CONTRACT + ("_smoke" if smoke else "")
    if not smoke and args.batch_size != BATCH_SIZE:
        raise ValueError("the preregistered batch size cannot be changed")
    if args.batch_size % args.micro_batch_size:
        raise ValueError("micro-batch size must divide the batch size")
    started = time.perf_counter()

    out = args.output_dir
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Seed-independent artefacts (teacher cache, selectors, sampling audit) can come
    # from any attached output of this experiment, including the other account's;
    # per-run records and checkpoints only from this account's previous sessions.
    shared = ("teacher_cache.pt", "selectors.json", "sampling.json")
    for previous in [*args.resume_from, *args.aggregate_from]:
        for name in shared:
            source, target = Path(previous) / name, out / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
                print("reused", name, "from", previous)
    for previous in args.resume_from:
        for path in sorted((Path(previous) / "runs").glob("*")):
            target = runs_dir / path.name
            if path.is_file() and not target.exists():
                shutil.copy2(path, target)
                print("resumed", path.name)

    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
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
    model.to(device=device, dtype=dtype)
    parameters_by_name = _trainable(model)
    if not parameters_by_name:
        raise RuntimeError("official CODI exposes no trainable parameters")
    latent_positions = int(cfg.eval.latent_iterations)
    vocab_limit = int(tokenizer.eos_token_id) + 1  # original GPT-2 vocabulary

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
        test_rows = test_rows[: SMOKE["test"]]

    sampling_path = out / "sampling.json"
    sizes = {"fit": fit_n, "select": select_n, "validate": validate_n, "train": train_n, "selection": selection_n}
    dataset = load_train_set(data_cfg, "eq_only")
    splits, sampling = sample_splits(
        dataset, tokenizer, test_questions={_normalized_question(r["question"]) for r in test},
        sizes=sizes, seed=DATA_SEED, bot_token_id=model.bot_id,
    )
    if sampling_path.is_file():
        previous = json.loads(sampling_path.read_text())
        if previous["hashes"] != sampling["hashes"]:
            raise RuntimeError("resumed output was produced from different splits")
    _atomic_json(sampling, sampling_path)
    print("sampling", {k: v for k, v in sampling.items() if k != "hashes"})

    # ---- teacher states (frozen official checkpoint), cached across sessions
    cache_path = out / "teacher_cache.pt"
    ordered_rows = [r for name in ("fit", "select", "validate", "train") for r in splits[name]]
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu")
        if cache["hashes"] != sampling["hashes"]:
            raise RuntimeError("teacher cache belongs to different splits")
    else:
        states, golds = precompute_teacher(model, tokenizer, ordered_rows,
                                           batch_size=args.teacher_batch_size, device=device)
        cache = {"states": states, "gold": golds, "hashes": sampling["hashes"],
                 "checkpoint_sha256": load_report.checkpoint_sha256}
        _atomic_torch_save(cache, cache_path)
    offsets, cursor = {}, 0
    for name in ("fit", "select", "validate", "train"):
        offsets[name] = (cursor, cursor + len(splits[name]))
        cursor += len(splits[name])
    def cached(name):
        a, b = offsets[name]
        return cache["states"][a:b], cache["gold"][a:b]

    # ---- selectors on the teacher, validated before any training
    selectors_path = out / "selectors.json"
    readout = readout_matrix(model, vocab_limit=vocab_limit).to(device, torch.float32)
    fit_states, _ = cached("fit")
    pca = fit_teacher_pca(fit_states.float())
    if selectors_path.is_file():
        selectors = json.loads(selectors_path.read_text())
        # The rank rule is re-applied to the stored teacher reports so a rule
        # amendment takes effect on resume without recomputing the selectors.
        reports = {int(k): v for k, v in selectors["reports_by_rank"].items()}
        chosen, audit = choose_rank(reports, minimum_gap=MINIMUM_GAP, retention_floor=RETENTION_FLOOR)
        forced = False
        if chosen is None and smoke:
            chosen, forced = RANK_GRID[0], True
        selectors.update({"rank": chosen, "smoke_forced_rank": forced, "audit": audit})
        _atomic_json(selectors, selectors_path)
    else:
        select_states, select_gold = cached("select")
        select_states = select_states.to(device)
        validate_states, validate_gold = cached("validate")
        validate_states = validate_states.to(device)
        max_rank = max(RANK_GRID)
        causal_full, trace = select_causal_greedy(select_states, select_gold, pca, readout, max_rank,
                                                  candidates=candidates, return_trace=True)
        greedy_order = [t["added"] for t in trace]
        relevance_full = select_relevance(select_states, select_gold, pca, readout, max_rank)
        reports = {}
        sets_by_rank = {}
        for rank in RANK_GRID:
            sets = {"variance": select_variance(rank),
                    "relevance": select_relevance(select_states, select_gold, pca, readout, rank),
                    "causal": sorted(greedy_order[:rank]),
                    "random": select_random(rank, pca.basis.shape[1], seed=RANDOM_PC_SEED + rank)}
            sets_by_rank[rank] = sets
            reports[rank] = evaluate_index_sets(validate_states, validate_gold, pca, readout, sets)
        chosen, audit = choose_rank(reports, minimum_gap=MINIMUM_GAP, retention_floor=RETENTION_FLOOR)
        forced = False
        if chosen is None and smoke:
            # The smoke pass exists to exercise training, checkpointing, evaluation and
            # aggregation; on 64 rows the selectors need not be distinguishable.
            chosen, forced = RANK_GRID[0], True
        selectors = {"rank": chosen, "smoke_forced_rank": forced, "audit": audit, "greedy_trace": trace,
                     "reports_by_rank": {str(k): v for k, v in reports.items()},
                     "sets": {str(k): v for k, v in sets_by_rank.items()},
                     "eigenvalue_share_top4": float(pca.eigenvalues[:4].sum() / pca.eigenvalues.sum())}
        _atomic_json(selectors, selectors_path)
        del select_states, validate_states
    print("rank rule", selectors["audit"], "chosen", selectors["rank"])

    summary = {
        "schema_version": 1, "contract": contract, "smoke": smoke,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
        "preregistration": {
            "arms": list(arms), "seeds_this_run": list(seeds), "steps": steps, "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "train_examples": train_n, "fit_select_validate": [fit_n, select_n, validate_n],
            "selection_examples": selection_n, "learning_rate": LEARNING_RATE, "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "rank_grid": list(RANK_GRID),
            "candidate_pcs": candidates, "retention_floor": RETENTION_FLOOR, "minimum_gap": MINIMUM_GAP,
            "rank_rule": "among ranks with causal-variance gap >= minimum_gap and causal retention >= retention_floor, maximise gap x retention",
            "data_seed": DATA_SEED, "readout_state": "last hidden state (index 12 on GPT-2)", "max_new_tokens": MAX_NEW_TOKENS,
            "student": "official embeddings and readout; LoRA and projector reset per seed",
            "distillation": "state-12 smooth-L1 over selected PC coordinates, teacher-std scaled, norm-matched to answer CE",
            "gates": ["s1 full-none lb>0", "h1 causal-variance lb>0", "h2 causal-relevance lb>0",
                      "h3 causal-random lb>0", "headline = s1 and h1 and h3"],
        },
        "sampling": {k: v for k, v in sampling.items()},
        "selectors": {"rank": selectors["rank"], "audit": selectors["audit"],
                      "eigenvalue_share_top4": selectors["eigenvalue_share_top4"],
                      "validation": selectors["reports_by_rank"].get(str(selectors["rank"])) if selectors["rank"] else None},
        "runs": {}, "test": None, "status": "running",
        "decision": {"selectors_distinguishable": selectors["rank"] is not None, "claim": "pending"},
        "warnings": [
            "Students learn the latent task from fresh adapters; absolute accuracy after 0.4 epoch may be low, so the selection-split NLL curve is the sensitive secondary outcome.",
            "The distillation gradient is norm-matched to the answer cross-entropy, a term that is being learned rather than converged (ledger 85).",
            "Rank is chosen on the teacher's validation split only; if variance and causal sets are not distinguishable there, nothing is trained.",
            "Test rows are read once per trained model; the set was opened by earlier method families (ledger 75-79).",
        ],
    }
    _atomic_json(summary, out / "summary.json")
    if selectors["rank"] is None:
        summary["status"] = "stopped"
        summary["decision"]["claim"] = "STOP: no rank in the grid makes the causal and variance sets distinguishable on the teacher"
        _atomic_json(summary, out / "summary.json")
        print(summary["decision"]); return summary

    rank = int(selectors["rank"])
    sets = selectors["sets"][str(rank)]
    targets = {"none": None, "full": build_target(pca, fit_states.float(), None)}
    for arm in SUBSPACE_ARMS:
        targets[arm] = build_target(pca, fit_states.float(), sets[arm])
    targets = {k: (None if v is None else v.to(device)) for k, v in targets.items()}
    train_states, _ = cached("train")
    train_rows = splits["train"]
    selection_rows = splits["selection"]
    eval_kw = dict(latent_positions=latent_positions, batch_size=args.eval_batch_size, device=device)

    def remaining_budget():
        return args.max_seconds - (time.perf_counter() - started)

    def train_one(arm, seed):
        name = f"{arm}_seed{seed}"
        record_path, ckpt_path = runs_dir / f"{name}.json", runs_dir / f"{name}.ckpt.pt"
        if record_path.is_file():
            return json.loads(record_path.read_text()), "done"
        parameters = list(parameters_by_name.values())
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        order = list(range(len(train_rows)))
        random.Random(seed).shuffle(order)
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        step0, curve, elapsed, window = 0, [], 0.0, []
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _restore(parameters_by_name, ckpt["trainable"])
            optimizer.load_state_dict(ckpt["optimizer"])
            step0, curve, elapsed = int(ckpt["step"]), ckpt["curve"], float(ckpt["elapsed"])
            torch.set_rng_state(ckpt["cpu_rng"])
            if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
            print("resume", name, "at step", step0)
        else:
            reset = reinitialize_student(model, seed=seed)
            print("fresh student", name, reset)
        target = targets[arm]
        model.train()
        t0 = time.perf_counter()

        def save_ckpt(step):
            _atomic_torch_save({"trainable": _snapshot(parameters_by_name), "optimizer": optimizer.state_dict(),
                                "step": step, "curve": curve, "elapsed": elapsed + time.perf_counter() - t0,
                                "cpu_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                               ckpt_path)

        for step in tqdm(range(step0, steps), desc=name, unit="step", initial=step0, total=steps):
            if remaining_budget() < 120:
                save_ckpt(step)
                return None, "paused"
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(step, total_steps=steps, base=LEARNING_RATE, warmup=WARMUP_STEPS)
            indices = order[step * args.batch_size : (step + 1) * args.batch_size]
            micro = args.micro_batch_size
            batches = [collate_official_codi_kv_rows(tokenizer, [train_rows[i] for i in indices[k : k + micro]],
                                                     bot_token_id=model.bot_id).to(device)
                       for k in range(0, len(indices), micro)]
            teacher_states = [train_states[indices[k : k + micro]].to(device, torch.float32)
                              for k in range(0, len(indices), micro)]
            result = student_training_step(model, batches, teacher_states, target, parameters,
                                           latent_positions=latent_positions)
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, result.gradients):
                parameter.grad = None if gradient is None else gradient.detach()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            window.append(result)
            done = step + 1
            if done % curve_every == 0 or done == steps:
                metrics, _, _, _ = selection_metrics(model, tokenizer, selection_rows, **eval_kw)
                model.train()
                curve.append({"step": done, **metrics,
                              "train_answer_loss": sum(r.answer_loss for r in window) / len(window),
                              "train_distillation_loss": (sum(r.distillation_loss for r in window) / len(window)
                                                          if window[0].distillation_loss is not None else None),
                              "distillation_scale": (sum(r.distillation_scale for r in window) / len(window)
                                                     if window[0].distillation_scale is not None else None)})
                window = []
                print("curve", name, curve[-1])
            if done % checkpoint_every == 0 and done < steps:
                save_ckpt(done)
            del batches, teacher_states, result
        training_seconds = elapsed + time.perf_counter() - t0
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        test_metrics, test_nll, test_correct, test_outputs = selection_metrics(model, tokenizer, test_rows, **eval_kw)
        record = {"arm": arm, "seed": seed, "rank": rank, "steps": steps, "training_seconds": training_seconds,
                  "curve": curve, "test": test_metrics, "test_correct": test_correct, "test_nll": test_nll,
                  "test_outputs": test_outputs}
        _atomic_json(record, record_path)
        if ckpt_path.is_file():
            ckpt_path.unlink()
        print("test", name, test_metrics)
        return record, "done"

    for arm in arms:
        for seed in seeds:
            record, status = train_one(arm, seed)
            if status == "paused":
                summary["status"] = "paused"
                summary["decision"]["claim"] = (f"PAUSED before {arm} seed {seed} finished: publish this output "
                                                "directory and rerun with it attached to resume")
                _atomic_json(summary, out / "summary.json")
                print(summary["decision"]); return summary
    summary["status"] = "trained"
    return aggregate(args, summary, test_rows, out)


def aggregate(args, summary, test_rows, out):
    runs = {}
    for directory in [out, *[Path(p) for p in args.aggregate_from]]:
        for path in sorted((Path(directory) / "runs").glob("*.json")):
            record = json.loads(path.read_text())
            if "test" in record:
                runs[f"{record['arm']}_seed{record['seed']}"] = record
    by_arm = {}
    for record in runs.values():
        by_arm.setdefault(record["arm"], []).append(record)
    if not by_arm:
        summary["decision"]["claim"] = "no completed runs to aggregate"
        _atomic_json(summary, out / "summary.json"); return summary
    n = len(next(iter(runs.values()))["test_correct"])
    mean_correct = {a: torch.tensor([r["test_correct"] for r in rs], dtype=torch.float32).mean(0).tolist()
                    for a, rs in by_arm.items()}
    mean_nll = {a: torch.tensor([r["test_nll"] for r in rs], dtype=torch.float32).mean(0).tolist()
                for a, rs in by_arm.items()}
    pairs = [("full", "none"), ("causal", "variance"), ("causal", "relevance"), ("causal", "random"),
             ("variance", "random"), ("relevance", "random"), ("causal", "full"), ("variance", "none"),
             ("causal", "none")]
    comparisons, nll_comparisons = {}, {}
    for i, (a, b) in enumerate(pairs):
        if a in mean_correct and b in mean_correct:
            comparisons[f"{a}_minus_{b}"] = _paired(mean_correct[a], mean_correct[b], seed=args.seed + i, samples=args.bootstrap_samples)
            nll_comparisons[f"{b}_minus_{a}_nll"] = _paired(mean_nll[b], mean_nll[a], seed=args.seed + 50 + i, samples=args.bootstrap_samples)
    required = ("full_minus_none", "causal_minus_variance", "causal_minus_relevance", "causal_minus_random", "variance_minus_random")
    gate = gates_from(comparisons) if all(k in comparisons for k in required) else None
    arm_results = {a: {"seeds": sorted(r["seed"] for r in rs),
                       "test_accuracy_by_seed": [r["test"]["accuracy"] for r in rs],
                       "test_accuracy_mean": float(torch.tensor(mean_correct[a]).mean()),
                       "test_nll_mean": float(torch.tensor(mean_nll[a]).mean()),
                       "final_selection": [r["curve"][-1] if r["curve"] else None for r in rs],
                       "training_seconds_mean": sum(r["training_seconds"] for r in rs) / len(rs)}
                   for a, rs in by_arm.items()}
    summary["runs"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("test_correct", "test_nll", "test_outputs")}
                       for k, v in runs.items()}
    summary["test"] = {"examples": n, "arms": arm_results, "comparisons": comparisons,
                       "nll_comparisons": nll_comparisons, "gate": gate}
    summary["status"] = "aggregated"
    summary["decision"] = {"selectors_distinguishable": True, "final_passed": bool(gate["headline"]) if gate else None,
                           "claim": claim_from(gate) if gate else f"partial aggregate over arms {sorted(by_arm)}; gates need all five comparisons"}
    _atomic_json(summary, out / "summary.json")
    _atomic_torch_save({**summary, "test_correct": mean_correct, "test_nll": mean_nll,
                        "per_run_test_correct": {k: v["test_correct"] for k, v in runs.items()}},
                       out / "causal_subspace_distillation.pt")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(test_rows[:n]):
            handle.write(json.dumps({"dataset_index": index, "question": row["question"], "gold": row["gold"],
                                     **{k: v["test_outputs"][index] for k, v in runs.items()}}, ensure_ascii=False) + "\n")
    print(json.dumps(summary["decision"], indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--arms", default=",".join(ARMS), type=lambda s: tuple(a for a in s.split(",") if a))
    parser.add_argument("--resume-from", default=[], type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--aggregate-from", default=[], type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30_600)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE,
                        help="memory-only split of each step; gradients are accumulated exactly")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--teacher-batch-size", type=int, default=TEACHER_BATCH)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_923)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        summary_path = args.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"contract": CONTRACT, "decision": {}}
        cfg = load_config(args.config)
        data_cfg = load_config(str(cfg.endpoint_retention.data_config))
        test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
        rows = [{"question": r["question"], "gold": str(r["gold"])} for r in test]
        aggregate(args, summary, rows, args.output_dir)
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
