"""Transient or permanent? Variance- versus intervention-selected distillation on a
templated arithmetic task the student can finish (ledger §89).

Same teacher, student construction, selectors and norm-matched distillation as
``run_codi_causal_subspace_distillation``; the data are generated two- and
three-step word problems in GSM8k-Aug format.  A go/no-go on the teacher (CoT
generation accuracy, colon first-token accuracy, selector distinguishability)
precedes any training.  Three arms, five seeds, exact match primary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
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
    GRAD_CLIP,
    LEARNING_RATE,
    MINIMUM_GAP,
    RANK_GRID,
    RETENTION_FLOOR,
    WEIGHT_DECAY,
    _paired,
    _restore,
    _snapshot,
    _trainable,
    precompute_teacher,
    selection_metrics,
)
from src.data.official_codi_training import collate_official_codi_kv_rows
from src.data.templated_arithmetic import (
    generate_problems,
    generate_teacher_cot,
    teacher_accuracy,
    verify_problem,
)
from src.eval.official_codi import select_device
from src.mech.causal_subspace_distillation import (
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
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_templated_subspace_distillation_v1"
ARMS = ("none", "variance", "causal")
STEPS = 3_000
BATCH_SIZE = 16
MICRO_BATCH_SIZE = 8
WARMUP_STEPS = 150
TRAIN_EXAMPLES = STEPS * BATCH_SIZE
TEACHER_CHECK = 500
FIT = SELECT = VALIDATE = 2_048
SELECTION_EXAMPLES = 256
TEST_EXAMPLES = 2_000
DATA_SEED = 20_260_924
CANDIDATE_PCS = 128
CURVE_EVERY = 300
CHECKPOINT_EVERY = 250
MAX_NEW_TOKENS_TEACHER = 96
TEACHER_BATCH = 64
MINIMUM_TEACHER_GENERATION = 0.80
MINIMUM_DENSE_FIRST_TOKEN = 0.80
THRESHOLD_ACCURACY = 0.30
RANDOM_PC_SEED = 20_260_924
SEEDS_DEFAULT = "1,2,3,4,5"

SMOKE = {"teacher_check": 32, "fit": 64, "select": 64, "validate": 64, "steps": 20, "selection": 16,
         "test": 32, "curve_every": 10, "checkpoint_every": 10, "candidates": 16, "seeds": "1"}


def steps_to_threshold(curve: list[dict], threshold: float) -> int | None:
    for point in curve:
        if point.get("accuracy", 0.0) >= threshold:
            return int(point["step"])
    return None


def gates_from(comparisons: dict) -> dict:
    lb = lambda k: comparisons[k]["bootstrap_95ci"][0]
    ub = lambda k: comparisons[k]["bootstrap_95ci"][1]
    return {
        "t1_variance_tax_persists": lb("none_minus_variance") > 0,
        "t1_variance_now_helps": ub("none_minus_variance") < 0,
        "h1_causal_beats_variance": lb("causal_minus_variance") > 0,
        "h1b_causal_beats_none": lb("causal_minus_none") > 0,
        "h1b_none_beats_causal": ub("causal_minus_none") < 0,
    }


def claim_from(gate: dict) -> str:
    if gate["t1_variance_tax_persists"] and gate["h1_causal_beats_variance"]:
        return "PERMANENT: variance-selected distillation leaves a final-accuracy deficit that causal selection avoids"
    if gate["t1_variance_tax_persists"]:
        return "PARTIAL: the variance tax persists to final accuracy but causal selection does not separate from it"
    if gate["t1_variance_now_helps"]:
        return "REVERSED: variance-selected distillation helps final accuracy once the student can solve the task"
    if gate["h1_causal_beats_variance"]:
        return "PARTIAL: causal beats variance on final accuracy although the tax versus no distillation washed out"
    return "TRANSIENT: the early variance tax washed out; no final-accuracy difference between selectors"


def run(args):
    smoke = bool(args.smoke)
    S = SMOKE if smoke else None
    steps = S["steps"] if smoke else STEPS
    train_n = steps * args.batch_size
    sizes = {"teacher_check": S["teacher_check"] if smoke else TEACHER_CHECK,
             "fit": S["fit"] if smoke else FIT, "select": S["select"] if smoke else SELECT,
             "validate": S["validate"] if smoke else VALIDATE, "train": train_n,
             "selection": S["selection"] if smoke else SELECTION_EXAMPLES,
             "test": S["test"] if smoke else TEST_EXAMPLES}
    curve_every = S["curve_every"] if smoke else CURVE_EVERY
    checkpoint_every = S["checkpoint_every"] if smoke else CHECKPOINT_EVERY
    candidates = S["candidates"] if smoke else CANDIDATE_PCS
    seeds = tuple(int(s) for s in (S["seeds"] if smoke else args.seeds).split(",") if s.strip())
    arms = tuple(a for a in ARMS if a in args.arms)
    contract = CONTRACT + ("_smoke" if smoke else "")
    if not smoke and args.batch_size != BATCH_SIZE:
        raise ValueError("the preregistered batch size cannot be changed")
    if args.batch_size % args.micro_batch_size:
        raise ValueError("micro-batch size must divide the batch size")
    started = time.perf_counter()

    out = args.output_dir
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    shared = ("teacher_cache.pt", "selectors.json", "preliminary.json", "problems.json")
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
    latent_positions = int(cfg.eval.latent_iterations)
    vocab_limit = int(tokenizer.eos_token_id) + 1

    # ---- generated problems, deterministic and unique across every split
    total = sum(sizes.values())
    problems = generate_problems(total, seed=DATA_SEED)
    if not all(verify_problem(p) for p in problems):
        raise RuntimeError("generated problem failed arithmetic verification")
    rows = [p.as_row() for p in problems]
    splits, cursor = {}, 0
    for name, size in sizes.items():
        splits[name] = rows[cursor : cursor + size]; cursor += size
    problems_path = out / "problems.json"
    audit = {"data_seed": DATA_SEED, "sizes": sizes,
             "template_counts": {t: sum(p.template == t for p in problems) for t in sorted({p.template for p in problems})},
             "step_counts": {str(s): sum(p.steps == s for p in problems) for s in (2, 3)},
             "first_questions": [p.question for p in problems[:3]]}
    if problems_path.is_file():
        previous = json.loads(problems_path.read_text())
        if previous["sizes"] != sizes or previous["first_questions"] != audit["first_questions"]:
            raise RuntimeError("resumed output was produced from different problems")
    _atomic_json(audit, problems_path)
    print("problems", {k: v for k, v in audit.items() if k != "first_questions"})

    # ---- go/no-go part 1: teacher CoT generation on held-out templated problems
    preliminary_path = out / "preliminary.json"
    preliminary = json.loads(preliminary_path.read_text()) if preliminary_path.is_file() else {}
    if "teacher_generation_accuracy" not in preliminary:
        check = splits["teacher_check"]
        outputs = generate_teacher_cot(model, tokenizer, [r["question"] for r in check],
                                       max_new_tokens=MAX_NEW_TOKENS_TEACHER, batch_size=32, device=device)
        accuracy, correct = teacher_accuracy(outputs, [problems[i] for i in range(len(check))])
        preliminary.update({"teacher_generation_accuracy": accuracy, "teacher_check_examples": len(check),
                            "teacher_samples": [{"question": r["question"], "generation": o, "gold": r["gold"], "correct": c}
                                                for r, o, c in list(zip(check, outputs, correct))[:8]]})
        _atomic_json(preliminary, preliminary_path)
    print("teacher generation accuracy", preliminary["teacher_generation_accuracy"])

    # ---- teacher states for fit/select/validate/train
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

    # ---- go/no-go part 2: selectors and rank rule on the templated teacher
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
    dense_first_token = None
    if selectors["rank"] is not None:
        dense_first_token = selectors["reports_by_rank"][str(selectors["rank"])]["dense"]["first_token_accuracy"]
    else:
        dense_first_token = selectors["reports_by_rank"][str(RANK_GRID[0])]["dense"]["first_token_accuracy"]
    gate = {"teacher_generation": preliminary["teacher_generation_accuracy"] >= MINIMUM_TEACHER_GENERATION,
            "selectors_distinguishable": selectors["rank"] is not None,
            "dense_first_token": dense_first_token >= MINIMUM_DENSE_FIRST_TOKEN}
    gate["passed"] = all(gate.values()) or smoke
    preliminary.update({"dense_first_token_accuracy": dense_first_token, "rank": selectors["rank"],
                        "rank_audit": selectors["audit"], "gate": gate})
    _atomic_json(preliminary, preliminary_path)
    print("go/no-go", gate, "rank", selectors["rank"])

    summary = {
        "schema_version": 1, "contract": contract, "smoke": smoke,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
        "preregistration": {
            "arms": list(arms), "seeds_this_run": list(seeds), "steps": steps, "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size, "sizes": sizes, "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "rank_grid": list(RANK_GRID), "minimum_gap": MINIMUM_GAP, "retention_floor": RETENTION_FLOOR,
            "minimum_teacher_generation": MINIMUM_TEACHER_GENERATION,
            "minimum_dense_first_token": MINIMUM_DENSE_FIRST_TOKEN, "threshold_accuracy": THRESHOLD_ACCURACY,
            "data_seed": DATA_SEED, "test_examples": sizes["test"],
            "gates": ["t1 none-variance lb>0 (permanent) / ub<0 (reversed) / covers 0 (transient)",
                      "h1 causal-variance lb>0", "h1b causal-none two-sided", "secondary steps-to-30%"],
        },
        "problems": {k: v for k, v in audit.items() if k != "first_questions"},
        "preliminary": preliminary, "selectors": {"rank": selectors["rank"], "audit": selectors["audit"],
                                                  "eigenvalue_share_top4": selectors["eigenvalue_share_top4"],
                                                  "validation": selectors["reports_by_rank"].get(str(selectors["rank"])) if selectors["rank"] else None},
        "runs": {}, "test": None, "status": "running",
        "decision": {"preliminary_passed": bool(gate["passed"]), "claim": "pending"},
        "warnings": [
            "Generated two- and three-step problems in GSM8k-Aug format; nothing here speaks to GSM8K difficulty.",
            "Students learn from fresh adapters; the primary is exact match on 2,000 generated held-out problems at a fixed step count.",
            "Distillation gradients are norm-matched to the answer cross-entropy as in ledger 86.",
        ],
    }
    _atomic_json(summary, out / "summary.json")
    if not gate["passed"]:
        summary["status"] = "stopped"
        summary["decision"]["claim"] = "STOP: the teacher or the selectors failed the go/no-go on the templated task; nothing was trained"
        _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary
    if args.preliminary_only:
        summary["status"] = "preliminary_only"
        summary["decision"]["claim"] = "GO: teacher and selectors pass on the templated task; training not requested in this session"
        _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary

    rank = int(selectors["rank"])
    sets = selectors["sets"][str(rank)]
    targets = {"none": None, "variance": build_target(pca, fit_states.float(), sets["variance"]).to(device),
               "causal": build_target(pca, fit_states.float(), sets["causal"]).to(device)}
    train_states, _ = cached("train")
    train_rows, selection_rows, test_rows = splits["train"], splits["selection"], splits["test"]
    eval_kw = dict(latent_positions=latent_positions, batch_size=args.eval_batch_size, device=device)
    remaining = lambda: args.max_seconds - (time.perf_counter() - started)

    def train_one(arm, seed):
        name = f"{arm}_seed{seed}"
        record_path, ckpt_path = runs_dir / f"{name}.json", runs_dir / f"{name}.ckpt.pt"
        if record_path.is_file():
            return json.loads(record_path.read_text()), "done"
        parameters = list(parameters_by_name.values())
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        order = list(range(len(train_rows))); random.Random(seed).shuffle(order)
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        step0, curve, elapsed, window = 0, [], 0.0, []
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _restore(parameters_by_name, ckpt["trainable"]); optimizer.load_state_dict(ckpt["optimizer"])
            step0, curve, elapsed = int(ckpt["step"]), ckpt["curve"], float(ckpt["elapsed"])
            torch.set_rng_state(ckpt["cpu_rng"])
            if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
            print("resume", name, "at", step0)
        else:
            print("fresh student", name, reinitialize_student(model, seed=seed))
        target = targets[arm]
        model.train()
        t0 = time.perf_counter()

        def save_ckpt(step):
            _atomic_torch_save({"trainable": _snapshot(parameters_by_name), "optimizer": optimizer.state_dict(),
                                "step": step, "curve": curve, "elapsed": elapsed + time.perf_counter() - t0,
                                "cpu_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, ckpt_path)

        for step in tqdm(range(step0, steps), desc=name, unit="step", initial=step0, total=steps):
            if remaining() < 120:
                save_ckpt(step); return None, "paused"
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(step, total_steps=steps, base=LEARNING_RATE, warmup=WARMUP_STEPS)
            indices = order[step * args.batch_size : (step + 1) * args.batch_size]
            micro = args.micro_batch_size
            batches = [collate_official_codi_kv_rows(tokenizer, [train_rows[i] for i in indices[k : k + micro]],
                                                     bot_token_id=model.bot_id).to(device) for k in range(0, len(indices), micro)]
            teacher_states = [train_states[indices[k : k + micro]].to(device, torch.float32) for k in range(0, len(indices), micro)]
            result = student_training_step(model, batches, teacher_states, target, parameters, latent_positions=latent_positions)
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, result.gradients):
                parameter.grad = None if gradient is None else gradient.detach()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            window.append({"answer_loss": result.answer_loss, "distillation_loss": result.distillation_loss,
                           "distillation_scale": result.distillation_scale})
            del batches, teacher_states, result
            done = step + 1
            if done % curve_every == 0 or done == steps:
                metrics, _, _, _ = selection_metrics(model, tokenizer, selection_rows, **eval_kw)
                model.train()
                curve.append({"step": done, **metrics,
                              "train_answer_loss": sum(r["answer_loss"] for r in window) / len(window),
                              "train_distillation_loss": (sum(r["distillation_loss"] for r in window) / len(window)
                                                          if window[0]["distillation_loss"] is not None else None),
                              "distillation_scale": (sum(r["distillation_scale"] for r in window) / len(window)
                                                     if window[0]["distillation_scale"] is not None else None)})
                window = []
                print("curve", name, curve[-1])
            if done % checkpoint_every == 0 and done < steps:
                save_ckpt(done)
        training_seconds = elapsed + time.perf_counter() - t0
        optimizer.zero_grad(set_to_none=True); del optimizer; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        test_metrics, test_nll, test_correct, test_outputs = selection_metrics(model, tokenizer, test_rows, **eval_kw)
        record = {"arm": arm, "seed": seed, "rank": rank, "steps": steps, "training_seconds": training_seconds,
                  "curve": curve, "steps_to_threshold": steps_to_threshold(curve, THRESHOLD_ACCURACY),
                  "test": test_metrics, "test_correct": test_correct, "test_nll": test_nll, "test_outputs": test_outputs}
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
                summary["decision"]["claim"] = f"PAUSED before {arm} seed {seed} finished: publish this output and rerun with it attached"
                _atomic_json(summary, out / "summary.json"); print(summary["decision"]); return summary
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
        summary["decision"]["claim"] = "no completed runs to aggregate"; _atomic_json(summary, out / "summary.json"); return summary
    n = len(next(iter(runs.values()))["test_correct"])
    mean_correct = {a: torch.tensor([r["test_correct"] for r in rs], dtype=torch.float32).mean(0).tolist() for a, rs in by_arm.items()}
    mean_nll = {a: torch.tensor([r["test_nll"] for r in rs], dtype=torch.float32).mean(0).tolist() for a, rs in by_arm.items()}
    pairs = [("none", "variance"), ("causal", "variance"), ("causal", "none")]
    comparisons, nll_comparisons = {}, {}
    for i, (a, b) in enumerate(pairs):
        if a in mean_correct and b in mean_correct:
            comparisons[f"{a}_minus_{b}"] = _paired(mean_correct[a], mean_correct[b], seed=args.seed + i, samples=args.bootstrap_samples)
            nll_comparisons[f"{b}_minus_{a}_nll"] = _paired(mean_nll[b], mean_nll[a], seed=args.seed + 50 + i, samples=args.bootstrap_samples)
    required = ("none_minus_variance", "causal_minus_variance", "causal_minus_none")
    gate = gates_from(comparisons) if all(k in comparisons for k in required) else None
    arm_results = {a: {"seeds": sorted(r["seed"] for r in rs),
                       "test_accuracy_by_seed": [r["test"]["accuracy"] for r in rs],
                       "test_accuracy_mean": float(torch.tensor(mean_correct[a]).mean()),
                       "test_nll_mean": float(torch.tensor(mean_nll[a]).mean()),
                       "steps_to_threshold": [r["steps_to_threshold"] for r in rs],
                       "final_selection": [r["curve"][-1] if r["curve"] else None for r in rs],
                       "training_seconds_mean": sum(r["training_seconds"] for r in rs) / len(rs)} for a, rs in by_arm.items()}
    summary["runs"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("test_correct", "test_nll", "test_outputs")} for k, v in runs.items()}
    summary["test"] = {"examples": n, "arms": arm_results, "comparisons": comparisons, "nll_comparisons": nll_comparisons, "gate": gate}
    summary["status"] = "aggregated"
    summary["decision"] = {"preliminary_passed": True, "claim": claim_from(gate) if gate else f"partial aggregate over arms {sorted(by_arm)}"}
    _atomic_json(summary, out / "summary.json")
    _atomic_torch_save({**summary, "test_correct": mean_correct, "test_nll": mean_nll,
                        "per_run_test_correct": {k: v["test_correct"] for k, v in runs.items()}}, out / "templated_subspace_distillation.pt")
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
    parser.add_argument("--seeds", default=SEEDS_DEFAULT)
    parser.add_argument("--arms", default=",".join(ARMS), type=lambda s: tuple(a for a in s.split(",") if a))
    parser.add_argument("--resume-from", default=[], type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--aggregate-from", default=[], type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--preliminary-only", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30_600)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=MICRO_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--teacher-batch-size", type=int, default=TEACHER_BATCH)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_924)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
