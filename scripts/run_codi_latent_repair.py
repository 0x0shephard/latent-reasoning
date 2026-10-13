"""Retrieval-based latent repair (ledger §126, Build 3).  No training.

A bank of natively decoded training-pool rows is indexed by the value the model's own
output head reads at the carrier thought.  For every wrongly answered multi-step test
question, the carrier state (and/or the store position's K/V) of a bank row that holds
the gold penultimate intermediate v* is transplanted, and the answer is decoded again.
Controls: a random bank row with a different value, the centroid of same-value rows
(a synthetic vector), the donor's whole K/V tail, and correctly answered rows (harm).
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
from scripts.run_codi_causal_subspace_distillation import DATA_SEED
from src.data.chain_pools import DEFAULT_POOL, gsm8k_chain_row, load_raw_eval_rows, pool_rows
from src.data.counterfactual_chain import parse_chain
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import extract_official_answer_number
from src.mech.cache_carrier import CacheEdit, Trajectory, backbone_layers, donor_view, generate_under, record_native_trajectory
from src.mech.directed_counterfactual import readout_matrix
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config

CONTRACT = "official_codi_latent_repair_v1"
CONDITIONS = ("state_nearest", "state_random_same", "state_centroid", "state_random_diff", "kv_store_nearest", "state_kv_nearest", "kv_tail_nearest")
SEED = 20_261_015
BANK_ROWS = 2_048
MAX_TARGETS = 512
CONTROLS = 128
MIN_STEPS = 3
SAMPLES = 2_000
R1_MIN = {"codi1b": 0.10, "default": 0.05}
R2_RATIO = 3.0
R5_MAX_HARM = 0.10
SMOKE = {"bank": 64, "targets": 24, "controls": 8}


# ----------------------------------------------------------------------------- helpers
def penultimate_value(cot: str, min_steps: int = MIN_STEPS) -> str | None:
    chain = parse_chain(str(cot))
    if len(chain) < min_steps:
        return None
    r = chain[-2]["result"]
    return str(int(round(r))) if abs(r - round(r)) < 1e-6 and r >= 0 else None


def top1_numbers(states: torch.Tensor, readout: torch.Tensor, vocab_ids: torch.Tensor, strings: list[str]) -> list[str]:
    """The model's own top-1 numeric reading of each state ``[N, D]``."""
    logits = states.float() @ readout[vocab_ids].float().T
    return [strings[i] for i in logits.argmax(1).tolist()]


def choose_donors(bank_values: list[str], bank_states: torch.Tensor, target_values: list[str], target_states: torch.Tensor,
                  *, seed: int = SEED) -> dict:
    """Per target: nearest same-value bank row (cosine on carrier states), a random same-value
    row, a random different-value row, and the same-value candidate list.  -1 when none."""
    by_value: dict[str, list[int]] = {}
    for i, v in enumerate(bank_values):
        by_value.setdefault(v, []).append(i)
    rng = random.Random(seed)
    normed_bank = bank_states.float() / bank_states.float().norm(dim=1, keepdim=True).clamp_min(1e-6)
    nearest, random_same, random_diff, candidates = [], [], [], []
    for t, v in enumerate(target_values):
        cands = by_value.get(v, [])
        candidates.append(cands)
        if cands:
            q = target_states[t].float(); q = q / q.norm().clamp_min(1e-6)
            sims = normed_bank[cands] @ q
            nearest.append(cands[int(sims.argmax())]); random_same.append(rng.choice(cands))
        else:
            nearest.append(-1); random_same.append(-1)
        others = [i for i in range(len(bank_values)) if bank_values[i] != v]
        random_diff.append(rng.choice(others) if others else -1)
    return {"nearest": nearest, "random_same": random_same, "random_diff": random_diff, "candidates": candidates}


def centroid_states(bank_states: torch.Tensor, candidates: list[list[int]], fallback: torch.Tensor) -> torch.Tensor:
    out = fallback.clone()
    for t, cands in enumerate(candidates):
        if cands:
            out[t] = bank_states[cands].float().mean(0).to(out.dtype)
    return out


def classify(numbers: list[float | None], gold: list[float], base: list[float | None]) -> list[str]:
    out = []
    for n, g, b in zip(numbers, gold, base):
        if n is not None and abs(n - g) < 1e-6:
            out.append("repaired" if b is None or abs(b - g) >= 1e-6 else "kept")
        elif (n is None and b is None) or (n is not None and b is not None and abs(n - b) < 1e-6):
            out.append("unchanged")
        else:
            out.append("changed_wrong")
    return out


def rate(flags: list[bool], *, samples: int = SAMPLES, seed: int = SEED) -> dict:
    n = len(flags)
    if n == 0:
        return {"rate": None, "ci95": None, "n": 0}
    t = torch.tensor(flags, dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(n, (samples, n), generator=g)
    means = t[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.double)).tolist()
    return {"rate": float(t.mean()), "ci95": [float(lo), float(hi)], "n": n}


def checks_from(report: dict, model_key: str) -> dict:
    prim = report["repair"]["primary"]
    g = lambda c: (prim.get(c) or {}).get("rate")
    r1_min = R1_MIN.get(model_key, R1_MIN["default"])
    s, d, cen, kv = g("state_nearest"), g("state_random_diff"), g("state_centroid"), g("state_kv_nearest")
    harm = (report["harm"].get("state_nearest") or {}).get("rate")
    return {"r1_repair_exists": None if s is None else s >= r1_min, "r1_threshold": r1_min,
            "r2_value_not_disturbance": None if s is None or d is None else s >= R2_RATIO * max(d, 1e-9),
            "r3_natural_beats_synthetic": None if s is None or cen is None else s >= cen,
            "r4_store_adds": None if s is None or kv is None else kv >= s,
            "r5_harmless_where_right": None if harm is None else harm <= R5_MAX_HARM}


def _number_text(value):
    return None if value is None else (str(int(round(value))) if abs(value - round(value)) < 1e-6 else str(value))


# ----------------------------------------------------------------------------- run
def run(args):
    smoke = bool(args.smoke)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cfg = load_config(args.config)
    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token)
    model, tokenizer = build_official_codi_gpt2(base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
                                                dtype=dtype, settings=cfg.model, token=token)
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype).eval()
    latent_positions = int(cfg.eval.latent_iterations)
    n_layers = backbone_layers(model); all_layers = tuple(range(n_layers))
    carrier, store = int(args.carrier), int(args.store)
    extra = [int(c) for c in args.extra_carriers] if args.extra_carriers else []
    bs = args.batch_size
    readout = readout_matrix(model)
    vocab_ids, strings = numeric_vocabulary(tokenizer, readout)

    # ---- bank
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    bank_rows, bank_meta = pool_rows(DEFAULT_POOL, data_cfg=data_cfg, tokenizer=tokenizer, bot_token_id=model.bot_id, seed=DATA_SEED)
    bank_rows = bank_rows[:SMOKE["bank"] if smoke else args.bank_rows]
    bank_traj, bank_out = record_native_trajectory(model, tokenizer, [r["question"] for r in bank_rows], latent_iterations=latent_positions, batch_size=bs, device=device)
    bank_values = {c: top1_numbers(bank_traj.states[:, c], readout, vocab_ids, strings) for c in [carrier] + extra}
    print(f"bank {len(bank_rows)} rows; distinct carrier values {len(set(bank_values[carrier]))}")

    # ---- targets
    raw = load_raw_eval_rows(dict(data_cfg.eval.gsm8k))
    rows = []
    for r in raw:
        c = gsm8k_chain_row(r)
        v = penultimate_value(c["cot"]) if c else None
        if v is not None:
            rows.append({"question": c["question"], "cot": c["cot"], "gold": float(c["answer"]), "v_star": v, "steps": len(parse_chain(c["cot"]))})
    if smoke:
        rows = rows[:4 * SMOKE["targets"]]
    traj, outputs = record_native_trajectory(model, tokenizer, [r["question"] for r in rows], latent_iterations=latent_positions, batch_size=bs, device=device)
    native = [extract_official_answer_number(o) for o in outputs]
    correct = [n is not None and abs(n - r["gold"]) < 1e-6 for n, r in zip(native, rows)]
    wrong_idx = [i for i, c in enumerate(correct) if not c][:SMOKE["targets"] if smoke else args.max_targets]
    rng = random.Random(SEED)
    correct_idx = [i for i, c in enumerate(correct) if c]
    control_idx = sorted(rng.sample(correct_idx, min(len(correct_idx), SMOKE["controls"] if smoke else args.controls)))
    sel = wrong_idx + control_idx
    is_control = [False] * len(wrong_idx) + [True] * len(control_idx)
    t_traj = donor_view(traj, sel)
    t_rows = [rows[i] for i in sel]; t_native = [native[i] for i in sel]
    t_values = [r["v_star"] for r in t_rows]
    own_read = {c: top1_numbers(t_traj.states[:, c], readout, vocab_ids, strings) for c in [carrier] + extra}
    print(f"targets: {len(rows)} eligible, {sum(1 for c in correct if not c)} wrong ({len(wrong_idx)} used), {len(control_idx)} correct controls; native accuracy {sum(correct)/len(correct):.3f}")

    # ---- donors
    donors = choose_donors(bank_values[carrier], bank_traj.states[:, carrier], t_values, t_traj.states[:, carrier], seed=SEED)
    has_donor = [i >= 0 for i in donors["nearest"]]
    carrier_wrong = [own_read[carrier][t] != t_values[t] for t in range(len(sel))]
    print(f"donor available for {sum(has_donor)}/{len(sel)} targets; carrier reads ≠ v* for {sum(carrier_wrong)}")

    def donor_states(index: list[int]) -> torch.Tensor:
        src = t_traj.states[:, carrier].clone()
        for t, i in enumerate(index):
            if i >= 0:
                src[t] = bank_traj.states[i, carrier]
        return src

    def donor_traj(index: list[int]) -> Trajectory:
        idx = [i if i >= 0 else 0 for i in index]
        d = donor_view(bank_traj, idx)
        for t, i in enumerate(index):   # no donor: the target's own tail (a no-op edit)
            if i < 0:
                d.states[t] = t_traj.states[t]; d.keys[t] = t_traj.keys[t]; d.values[t] = t_traj.values[t]
        return d

    questions = [r["question"] for r in t_rows]
    gold = [r["gold"] for r in t_rows]
    results: dict[str, list[str]] = {}
    numbers: dict[str, list] = {}

    def run_condition(name, *, hidden=None, edits=(), donor=None):
        outs = generate_under(model, tokenizer, questions, latent_iterations=latent_positions, batch_size=bs, device=device,
                              hidden_donors=hidden, cache_edits=list(edits), donor=donor)
        nums = [extract_official_answer_number(o) for o in outs]
        results[name] = classify(nums, gold, t_native); numbers[name] = nums
        wrong_mask = [not c and h and cw for c, h, cw in zip(is_control, has_donor, carrier_wrong)]
        print(name, "repaired (primary stratum)", round(sum(1 for m, r in zip(wrong_mask, results[name]) if m and r == "repaired") / max(1, sum(wrong_mask)), 3))

    near_t = donor_traj(donors["nearest"])
    run_condition("state_nearest", hidden={carrier: donor_states(donors["nearest"])})
    run_condition("state_random_same", hidden={carrier: donor_states(donors["random_same"])})
    run_condition("state_centroid", hidden={carrier: centroid_states(bank_traj.states[:, carrier], donors["candidates"], t_traj.states[:, carrier])})
    run_condition("state_random_diff", hidden={carrier: donor_states(donors["random_diff"])})
    run_condition("kv_store_nearest", edits=[CacheEdit(store, all_layers, True, True)], donor=near_t)
    run_condition("state_kv_nearest", hidden={carrier: donor_states(donors["nearest"])}, edits=[CacheEdit(store, all_layers, True, True)], donor=near_t)
    run_condition("kv_tail_nearest", edits=[CacheEdit(s, all_layers, True, True) for s in range(latent_positions)], donor=near_t)
    extra_donors = {}
    for c in extra:
        d = choose_donors(bank_values[c], bank_traj.states[:, c], t_values, t_traj.states[:, c], seed=SEED + c)
        extra_donors[c] = d
        src = t_traj.states[:, c].clone()
        for t, i in enumerate(d["nearest"]):
            if i >= 0:
                src[t] = bank_traj.states[i, c]
        run_condition(f"state{c}_nearest", hidden={c: src})

    # ---- rates
    def rates(mask_fn, kind="repaired"):
        return {name: rate([r == kind for r, m in zip(res, (mask_fn(t) for t in range(len(sel)))) if m]) for name, res in results.items()}
    report = {
        "targets": {"eligible": len(rows), "wrong_total": sum(1 for c in correct if not c), "wrong_used": len(wrong_idx), "controls": len(control_idx),
                    "native_accuracy": sum(correct) / len(correct), "with_donor": sum(h for h, c in zip(has_donor, is_control) if not c),
                    "carrier_reads_wrong": sum(cw for cw, c in zip(carrier_wrong, is_control) if not c),
                    "primary_stratum": sum(1 for c, h, cw in zip(is_control, has_donor, carrier_wrong) if not c and h and cw)},
        "bank": {"rows": len(bank_rows), "distinct_values": len(set(bank_values[carrier])), "hash": bank_meta["hash"]},
        "repair": {"primary": rates(lambda t: not is_control[t] and has_donor[t] and carrier_wrong[t]),
                   "with_donor": rates(lambda t: not is_control[t] and has_donor[t]),
                   "carrier_already_right": rates(lambda t: not is_control[t] and has_donor[t] and not carrier_wrong[t])},
        "changed": {"with_donor": rates(lambda t: not is_control[t] and has_donor[t], kind="changed_wrong")},
        "harm": {name: rate([r != "kept" for r, c, h in zip(res, is_control, has_donor) if c and h]) for name, res in results.items()},
    }
    checks = checks_from(report, args.model_key)
    with (out / "targets.jsonl").open("w") as handle:
        for t in range(len(sel)):
            handle.write(json.dumps({"question": t_rows[t]["question"], "gold": gold[t], "v_star": t_values[t], "steps": t_rows[t]["steps"], "control": is_control[t],
                                     "native": t_native[t], "carrier_read": own_read[carrier][t], "has_donor": has_donor[t],
                                     "nearest_donor": donors["nearest"][t], "outcomes": {k: v[t] for k, v in results.items()},
                                     "numbers": {k: v[t] for k, v in numbers.items()}}) + "\n")
    _atomic_torch_save({"target_states": t_traj.states.half(), "bank_values": bank_values, "bank_questions": [r["question"] for r in bank_rows]}, out / "repair_states.pt")
    summary = {"schema_version": 1, "contract": CONTRACT + ("_smoke" if smoke else ""), "smoke": smoke, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": load_report.checkpoint_sha256, "model_key": args.model_key, "carrier": carrier, "store": store, "extra_carriers": extra,
               "preregistration": {"bank_rows": len(bank_rows), "max_targets": args.max_targets, "controls": args.controls, "min_steps": MIN_STEPS, "seed": SEED,
                                   "conditions": list(results), "r1_min": R1_MIN, "r2_ratio": R2_RATIO, "r5_max_harm": R5_MAX_HARM},
               "report": report, "checks": checks, "seconds": time.perf_counter() - started,
               "warnings": ["Official checkpoint only; no training. Native greedy decoding under the transplant.",
                            "v* is the gold chain's penultimate intermediate; donors are bank rows whose carrier readout (top-1 numeric token) equals v*."]}
    _atomic_json(summary, out / "summary.json")
    print(json.dumps({"targets": report["targets"], "primary": {k: (round(v["rate"], 3) if v["rate"] is not None else None) for k, v in report["repair"]["primary"].items()},
                      "harm": {k: (round(v["rate"], 3) if v["rate"] is not None else None) for k, v in report["harm"].items()}, "checks": checks}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--model-key", default="gpt2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--carrier", type=int, default=3)
    parser.add_argument("--store", type=int, default=4)
    parser.add_argument("--extra-carriers", nargs="*", default=[])
    parser.add_argument("--bank-rows", type=int, default=BANK_ROWS)
    parser.add_argument("--max-targets", type=int, default=MAX_TARGETS)
    parser.add_argument("--controls", type=int, default=CONTROLS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
