"""Coconut audit (ledger §127 C): gate, twins (both profiles), masking and gold-value repair on
the third-party Coconut GPT-2 checkpoint, with native-decoding outcomes throughout.  No training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_monitor_faithfulness import numeric_vocabulary
from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.run_codi_causal_subspace_distillation import DATA_SEED, prepare_row
from scripts.run_codi_directed_counterfactual import EXTRA_ROWS, CANDIDATE_LIMIT, CANDIDATE_LIMIT_LONG, MIN_STEPS_LONG, PAIRS
from scripts.run_codi_latent_repair import CONTROLS, MAX_TARGETS, SEED as REPAIR_SEED, centroid_states, choose_donors, classify, penultimate_value, rate, top1_numbers
from src.data.chain_pools import DEFAULT_POOL, gsm8k_chain_row, load_raw_eval_rows, pool_rows
from src.data.counterfactual_chain import parse_chain, perturb_row
from src.eval.official_codi import select_device
from src.mech.cache_carrier import CacheEdit, Trajectory, donor_view
from src.models.coconut import CoconutGPT2, CoconutSpec, coconut_answer_number, coconut_generate_under, coconut_record
from src.utils.config import load_config

CONTRACT = "official_coconut_audit_v1"
SLOTS = (0, 1, 2, 3, 4, 5)
GATE_TOLERANCE = 0.03
MASK_QUESTIONS = 512
SMOKE = {"gate": 48, "candidates": 48, "pairs": 12, "mask": 24, "bank": 64, "targets": 16, "controls": 8}
CARRIER_MIN = 0.15


def answers(texts):
    return [coconut_answer_number(t) for t in texts]


def eq(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) < 1e-6


def outcome_rates(nums, own, target, mask=None) -> dict:
    idx = [i for i in range(len(nums)) if mask is None or mask[i]]
    if not idx:
        return {"target": None, "retain": None, "other": None, "n": 0}
    t = sum(eq(nums[i], target[i]) for i in idx) / len(idx); r = sum(eq(nums[i], own[i]) for i in idx) / len(idx)
    return {"target": t, "retain": r, "other": 1 - t - r, "n": len(idx)}


def run(args):
    smoke = bool(args.smoke)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = select_device(args.device)
    spec = CoconutSpec()
    model = CoconutGPT2.from_hub(spec, dtype=torch.float32 if args.precision == "float32" else torch.float16, token=os.environ.get("HF_TOKEN") or None).to(device)
    tokenizer = model.tokenizer
    bs, n_layers, all_layers = args.batch_size, model.n_layers(), tuple(range(model.n_layers()))
    readout = model.readout_matrix(); vocab_ids, strings = numeric_vocabulary(tokenizer, readout)
    data_cfg = load_config("configs/data.yaml")
    stage: dict[str, float] = {}

    # ---- gate
    t0 = time.perf_counter()
    raw = load_raw_eval_rows(dict(data_cfg.eval.gsm8k))
    test_rows = []
    for r in raw:
        c = gsm8k_chain_row(r); gold = float(str(r["answer"]).rsplit("####", 1)[1].strip().replace(",", ""))
        test_rows.append({"question": str(r["question"]), "cot": c["cot"] if c else "", "gold": gold, "steps": len(parse_chain(c["cot"])) if c else 0})
    gate_rows = test_rows[:SMOKE["gate"]] if smoke else test_rows
    test_traj, test_texts = coconut_record(model, [r["question"] for r in gate_rows], batch_size=bs, device=device)
    test_nums = answers(test_texts)
    test_correct = [eq(n, r["gold"]) for n, r in zip(test_nums, gate_rows)]
    accuracy = sum(test_correct) / len(test_correct)
    gate = {"accuracy": accuracy, "published": spec.published_accuracy, "tolerance": GATE_TOLERANCE, "rows": len(gate_rows),
            "status": "passed" if abs(accuracy - spec.published_accuracy) <= GATE_TOLERANCE or smoke else "failed"}
    print("gate", gate); stage["gate"] = round(time.perf_counter() - t0, 1)
    if gate["status"] != "passed":
        raise RuntimeError(f"Coconut gate failed: {accuracy:.4f} vs {spec.published_accuracy}")

    # ---- pool (the §110 fit split, recorded once: it is the twins' candidate pool and the repair bank)
    pool, pool_meta = pool_rows(DEFAULT_POOL, data_cfg=data_cfg, tokenizer=tokenizer, bot_token_id=model.latent_id, extra_rows=0 if smoke else EXTRA_ROWS, seed=DATA_SEED)
    fit = pool[:2048]; extra = pool[2048:]

    # ---- twins
    def twins(profile: str) -> dict:
        t0 = time.perf_counter()
        long = profile == "long"
        rows_pool = (fit + extra) if long else fit
        limit = SMOKE["candidates"] if smoke else (CANDIDATE_LIMIT_LONG if long else CANDIDATE_LIMIT)
        cands = []
        for row in rows_pool:
            p = perturb_row(row, prefer_latest_step=long)
            if p is None or p.steps < (MIN_STEPS_LONG if long else 1):
                continue
            twin = prepare_row(tokenizer, {"question": p.question, "cot": p.cot, "answer": p.answer}, bot_token_id=model.latent_id)
            if twin is None:
                continue
            cands.append((row, twin, p))
            if len(cands) >= limit:
                break
        cand_rows = [r for o, t, _ in cands for r in (o, t)]
        _, texts = coconut_record(model, [r["question"] for r in cand_rows], batch_size=bs, device=device, record_kv=False) if False else (None, None)
        texts = coconut_generate_under(model, [r["question"] for r in cand_rows], batch_size=bs, device=device)
        nums = answers(texts)
        ok = [eq(n, float(r["gold"])) for n, r in zip(nums, cand_rows)]
        selected = [c for p, c in enumerate(cands) if ok[2 * p] and ok[2 * p + 1] and float(c[0]["gold"]) != float(c[1]["gold"])][:SMOKE["pairs"] if smoke else args.pairs]
        rows = [r for o, t, _ in selected for r in (o, t)]
        if len(selected) < 2:
            return {"profile": profile, "candidates": len(cands), "selected_pairs": len(selected), "note": "fewer than two gated pairs"}
        strata = [f"n{p.steps}_k{p.step}" for _, _, p in selected for _ in (0, 1)]
        partner = [i ^ 1 for i in range(len(rows))]
        traj, base_texts = coconut_record(model, [r["question"] for r in rows], batch_size=bs, device=device)
        donor = donor_view(traj, partner)
        own = [float(r["gold"]) for r in rows]; target = [own[j] for j in partner]
        short = [int(s.split("_")[0][1:]) <= 2 for s in strata]
        conds = {}
        specs = {f"state_{s}": ({s: donor.states[:, s]}, []) for s in SLOTS} | {f"kv_{s}": (None, [CacheEdit(s, all_layers, True, True)]) for s in SLOTS}
        specs["kv_all"] = (None, [CacheEdit(s, all_layers, True, True) for s in SLOTS]); specs["state_all"] = ({s: donor.states[:, s] for s in SLOTS}, [])
        for name, (hidden, edits) in specs.items():
            texts = coconut_generate_under(model, [r["question"] for r in rows], batch_size=bs, device=device, hidden_donors=hidden, cache_edits=edits, donor=donor)
            nums = answers(texts)
            conds[name] = {"all": outcome_rates(nums, own, target), "short": outcome_rates(nums, own, target, short), "long": outcome_rates(nums, own, target, [not s for s in short])}
            print(profile, name, {k: round(v, 3) for k, v in conds[name]["all"].items() if isinstance(v, float)})
        _atomic_torch_save({"states": traj.states.half(), "strata": strata}, out / f"twins_{profile}_states.pt")
        with (out / f"twins_{profile}_pairs.jsonl").open("w") as h:
            for o, t, p in selected:
                h.write(json.dumps({"question": o["question"], "cot": o["cot"], "gold": o["gold"], "twin_question": t["question"], "twin_cot": t["cot"], "twin_gold": t["gold"], "step": p.step, "steps": p.steps}) + "\n")
        stage[f"twins_{profile}"] = round(time.perf_counter() - t0, 1)
        return {"profile": profile, "candidates": len(cands), "selected_pairs": len(selected), "strata_counts": {k: strata.count(k) for k in sorted(set(strata))}, "conditions": conds}

    tw = {"standard": twins("standard"), "long": twins("long")}
    long_c = tw["long"].get("conditions", {})
    transport_long = {s: (long_c.get(f"state_{s}", {}).get("all", {}).get("target") or 0.0) for s in SLOTS}
    carriers = [s for s in SLOTS if transport_long[s] >= CARRIER_MIN]
    carrier = max(SLOTS, key=lambda s: transport_long[s]) if long_c else 3
    print("long transport by slot", transport_long, "carriers", carriers, "carrier used for repair", carrier)

    # ---- masking
    t0 = time.perf_counter()
    mrows = fit[:SMOKE["mask"] if smoke else MASK_QUESTIONS]
    mq = [r["question"] for r in mrows]; mgold = [float(r["gold"]) for r in mrows]
    msteps = [len(parse_chain(r["cot"])) for r in mrows]; mlong = [s >= MIN_STEPS_LONG for s in msteps]
    mtraj, mtexts = coconut_record(model, mq, batch_size=bs, device=device, record_kv=False)
    mbase = [eq(n, g) for n, g in zip(answers(mtexts), mgold)]
    mean_state = mtraj.states.mean(0)
    masking = {}

    def mask_cond(name, hidden=None, mask_slots=()):
        texts = coconut_generate_under(model, mq, batch_size=bs, device=device, hidden_donors=hidden, mask_slots=mask_slots)
        corr = [eq(n, g) for n, g in zip(answers(texts), mgold)]
        part = lambda m: None if not any(m) else (sum(b for b, k in zip(mbase, m) if k) - sum(c for c, k in zip(corr, m) if k)) / sum(m)
        masking[name] = {"accuracy_drop_all": part([True] * len(mq)), "accuracy_drop_short": part([not l for l in mlong]), "accuracy_drop_long": part(mlong), "n_long": sum(mlong)}
        print("mask", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in masking[name].items()})

    for s in SLOTS:
        mask_cond(f"mean_{s}", hidden={s: mean_state[s].unsqueeze(0).expand(len(mq), -1)}); mask_cond(f"mask_{s}", mask_slots=(s,))
    mask_cond("mean_all", hidden={s: mean_state[s].unsqueeze(0).expand(len(mq), -1) for s in SLOTS}); mask_cond("mask_all", mask_slots=SLOTS)
    mask_cond("mask_even", mask_slots=(0, 2, 4)); mask_cond("mask_odd", mask_slots=(1, 3, 5))
    stage["masking"] = round(time.perf_counter() - t0, 1)

    # ---- repair (gold v*), §126 protocol with the data-chosen carrier and store = carrier + 1
    t0 = time.perf_counter()
    store = min(carrier + 1, 5)
    bank_rows = fit[:SMOKE["bank"] if smoke else 2048]
    bank_traj, _ = coconut_record(model, [r["question"] for r in bank_rows], batch_size=bs, device=device)
    bank_values = top1_numbers(bank_traj.states[:, carrier], readout, vocab_ids, strings)
    elig = [i for i, r in enumerate(test_rows) if penultimate_value(r["cot"]) is not None]
    if smoke:
        elig = elig[:4 * SMOKE["targets"]]
    eq_rows = [test_rows[i] for i in elig]
    etraj, etexts = coconut_record(model, [r["question"] for r in eq_rows], batch_size=bs, device=device)
    enums = answers(etexts); ecorrect = [eq(n, r["gold"]) for n, r in zip(enums, eq_rows)]
    wrong_idx = [i for i, c in enumerate(ecorrect) if not c][:SMOKE["targets"] if smoke else MAX_TARGETS]
    rng = random.Random(REPAIR_SEED); corr_idx = [i for i, c in enumerate(ecorrect) if c]
    control_idx = sorted(rng.sample(corr_idx, min(len(corr_idx), SMOKE["controls"] if smoke else CONTROLS)))
    sel = wrong_idx + control_idx; is_control = [False] * len(wrong_idx) + [True] * len(control_idx)
    t_traj = donor_view(etraj, sel); t_rows = [eq_rows[i] for i in sel]; t_native = [enums[i] for i in sel]
    t_values = [penultimate_value(r["cot"]) for r in t_rows]
    own_read = top1_numbers(t_traj.states[:, carrier], readout, vocab_ids, strings)
    donors = choose_donors(bank_values, bank_traj.states[:, carrier], t_values, t_traj.states[:, carrier], seed=REPAIR_SEED)
    has_donor = [i >= 0 for i in donors["nearest"]]; carrier_wrong = [own_read[t] != t_values[t] for t in range(len(sel))]

    def dstates(index):
        src = t_traj.states[:, carrier].clone()
        for t, i in enumerate(index):
            if i >= 0:
                src[t] = bank_traj.states[i, carrier]
        return src

    def dtraj(index):
        d = donor_view(bank_traj, [i if i >= 0 else 0 for i in index])
        for t, i in enumerate(index):
            if i < 0:
                d.states[t] = t_traj.states[t]; d.keys[t] = t_traj.keys[t]; d.values[t] = t_traj.values[t]
        return d

    tq = [r["question"] for r in t_rows]; tgold = [r["gold"] for r in t_rows]
    results = {}
    near = dtraj(donors["nearest"])
    rspecs = {"state_nearest": ({carrier: dstates(donors["nearest"])}, []), "state_random_same": ({carrier: dstates(donors["random_same"])}, []),
              "state_centroid": ({carrier: centroid_states(bank_traj.states[:, carrier], donors["candidates"], t_traj.states[:, carrier])}, []),
              "state_random_diff": ({carrier: dstates(donors["random_diff"])}, []), "kv_store_nearest": (None, [CacheEdit(store, all_layers, True, True)]),
              "state_kv_nearest": ({carrier: dstates(donors["nearest"])}, [CacheEdit(store, all_layers, True, True)]),
              "kv_tail_nearest": (None, [CacheEdit(s, all_layers, True, True) for s in SLOTS])}
    for name, (hidden, edits) in rspecs.items():
        texts = coconut_generate_under(model, tq, batch_size=bs, device=device, hidden_donors=hidden, cache_edits=edits, donor=near)
        results[name] = classify(answers(texts), tgold, t_native)
    primary = lambda t: not is_control[t] and has_donor[t] and carrier_wrong[t]
    repair = {"primary": {n: rate([r == "repaired" for t, r in enumerate(res) if primary(t)]) for n, res in results.items()},
              "with_donor": {n: rate([r == "repaired" for t, r in enumerate(res) if not is_control[t] and has_donor[t]]) for n, res in results.items()},
              "harm": {n: rate([r != "kept" for t, r in enumerate(res) if is_control[t] and has_donor[t]]) for n, res in results.items()},
              "targets": {"eligible": len(eq_rows), "wrong_used": len(wrong_idx), "controls": len(control_idx), "with_donor": sum(h for h, c in zip(has_donor, is_control) if not c),
                          "primary_stratum": sum(1 for t in range(len(sel)) if primary(t)), "carrier": carrier, "store": store, "bank_distinct_values": len(set(bank_values))}}
    print("repair primary", {k: (round(v["rate"], 3) if v["rate"] is not None else None) for k, v in repair["primary"].items()})
    stage["repair"] = round(time.perf_counter() - t0, 1)

    # ---- checks (P13)
    short_c = tw["standard"].get("conditions", {})
    max_short = max((short_c.get(f"state_{s}", {}).get("short", {}).get("target") or 0.0) for s in SLOTS) if short_c else None
    max_long = max(transport_long.values()) if long_c else None
    term_cost = max((masking.get("mean_5", {}).get("accuracy_drop_all") or 0.0), (masking.get("mask_5", {}).get("accuracy_drop_all") or 0.0))
    checks = {"chain_length_gate": None if max_short is None or max_long is None else (max_short <= 0.10 and max_long >= CARRIER_MIN),
              "terminal_inert": transport_long.get(5, 0.0) <= 0.05 and term_cost <= 0.02,
              "carrier_not_slot_3": bool(carriers) and 3 not in carriers, "carriers": carriers, "max_long_transport": max_long,
              "below_codi_gpt2_0_33": None if max_long is None else max_long < 0.33,
              "repair_3x_control": None if repair["primary"]["state_nearest"]["rate"] is None else repair["primary"]["state_nearest"]["rate"] >= 3 * max(repair["primary"]["state_random_diff"]["rate"] or 0.0, 1e-9)}
    summary = {"schema_version": 1, "contract": CONTRACT + ("_smoke" if smoke else ""), "smoke": smoke, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint": {"repo_id": spec.repo_id, "revision": spec.revision, "provenance": "third-party retrain (UdeSA); no official Coconut GSM8K checkpoint exists"},
               "gate": gate, "pool": {k: v for k, v in pool_meta.items() if k != "sampling"}, "twins": tw, "transport_long": transport_long, "masking": masking,
               "repair": repair, "checks": checks, "stage_seconds": stage, "seconds": time.perf_counter() - started}
    _atomic_json(summary, out / "summary.json")
    print(json.dumps({"gate": gate, "checks": checks, "stage_seconds": stage}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
