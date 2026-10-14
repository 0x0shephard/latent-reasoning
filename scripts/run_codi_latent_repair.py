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
    checkpoint_shard_hashes,
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


# ----------------------------------------------------------------------------- the tool (ledger §127 B)
def recompute_chain(chain_text: str) -> tuple[list[float], list[float]] | None:
    """Re-evaluate a generated equation chain with a calculator.  Operands equal to an
    earlier *stated* result are references and take that step's *corrected* result.
    Returns (stated results, corrected results) or None when nothing parses."""
    from src.data.counterfactual_chain import _evaluate
    equations = parse_chain(chain_text)
    if not equations:
        return None
    stated = [e["result"] for e in equations]
    corrected: list[float] = []
    for i, eq in enumerate(equations):
        pieces, cursor, left = [], 0, eq["left"]
        base = eq["span"][0] + 2
        for op in eq["operands"]:
            start, end = op["span"][0] - base, op["span"][1] - base
            pieces.append(left[cursor:start])
            ref = next((k for k in range(i) if abs(op["value"] - stated[k]) < 1e-6), None)
            pieces.append(repr(corrected[ref]) if ref is not None else op["text"])
            cursor = end
        pieces.append(left[cursor:])
        try:
            corrected.append(float(_evaluate("".join(pieces))))
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
            return None
    return stated, corrected


def tool_value(chain_text: str) -> str | None:
    """The tool's v*: the corrected result of the last generated equation, as an integer string."""
    out = recompute_chain(chain_text)
    if out is None:
        return None
    v = out[1][-1]
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 and v >= 0 else None


def explicit_chains(model, tokenizer, questions: list[str], *, batch_size: int, device, max_new_tokens: int = 96) -> list[str]:
    """Teacher-mode decoding: the question alone (no BOT) makes the checkpoint write its own
    equation chain and 'The answer is: X'."""
    from src.models.official_codi import official_codi_base_model
    base = official_codi_base_model(model)
    side = tokenizer.padding_side; tokenizer.padding_side = "left"
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    add_special = bool(getattr(model, "question_special_tokens", True))
    outs = []
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            enc = tokenizer(questions[start:start + batch_size], return_tensors="pt", padding=True, add_special_tokens=add_special).to(device)
            gen = base.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
            outs += [tokenizer.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True) for g in gen]
    tokenizer.padding_side = side
    return outs


# ----------------------------------------------------------------------------- the learned writer (ledger §128)
WRITER_RIDGE = 1.0
WRITER_MLP_HIDDEN = 256
WRITER_MLP_STEPS = 400
WRITER_SEED = 20_261_016


def value_token_embeddings(values: list[str | None], readout: torch.Tensor, vocab_ids: torch.Tensor, strings: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Tied embedding of each value's numeric token, and a mask of values that have one."""
    index = {s: i for i, s in enumerate(strings)}
    emb = torch.zeros(len(values), readout.shape[1]); ok = torch.zeros(len(values), dtype=torch.bool)
    for i, v in enumerate(values):
        j = index.get(v) if v is not None else None
        if j is not None:
            emb[i] = readout[vocab_ids[j]].float(); ok[i] = True
    return emb, ok


class Writer:
    """Maps a value's token embedding to a carrier vector: ridge (closed form) or a small MLP."""

    def __init__(self, kind: str = "ridge"):
        self.kind = kind; self.w = None; self.model = None; self.mean = None; self.std = None

    def fit(self, emb: torch.Tensor, states: torch.Tensor, *, seed: int = WRITER_SEED) -> "Writer":
        x, y = emb.float(), states.float()
        if self.kind == "ridge":
            xb = torch.cat([x, torch.ones(x.shape[0], 1)], 1)
            self.w = torch.linalg.solve(xb.T @ xb + WRITER_RIDGE * torch.eye(xb.shape[1]), xb.T @ y)
        else:
            torch.manual_seed(seed)
            self.mean, self.std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-6)
            self.model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], WRITER_MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(WRITER_MLP_HIDDEN, y.shape[1]))
            opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
            xs = (x - self.mean) / self.std
            for _ in range(WRITER_MLP_STEPS):
                opt.zero_grad(); torch.nn.functional.mse_loss(self.model(xs), y).backward(); opt.step()
        return self

    def __call__(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb.float()
        if self.kind == "ridge":
            return torch.cat([x, torch.ones(x.shape[0], 1)], 1) @ self.w
        with torch.no_grad():
            return self.model((x - self.mean) / self.std)


def written_states(writer: Writer, values: list[str | None], own: torch.Tensor, readout, vocab_ids, strings, *, residual_from: list[str] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute: the writer's vector for each value.  Residual: own + writer(value) − writer(own reading).
    Rows without a single-token value keep their own state; the mask says which were written."""
    emb, ok = value_token_embeddings(values, readout, vocab_ids, strings)
    out = own.clone().float(); target = writer(emb)
    if residual_from is not None:
        emb_old, ok_old = value_token_embeddings(residual_from, readout, vocab_ids, strings)
        target = own.float() + target - writer(emb_old); ok = ok & ok_old
    out[ok] = target[ok].to(out.dtype)
    return out.to(own.dtype), ok


# ----------------------------------------------------------------------------- ladder writes and the failure taxonomy (ledger §129)
LADDER_MIN_MAP = 0.40
LADDER_MIN_ROWS = 30
SLOTS6 = (0, 1, 2, 3, 4, 5)


def chain_intermediates(cot: str) -> list[str | None]:
    """All step results but the last (the answer), as integer strings or None."""
    out = []
    for e in parse_chain(str(cot))[:-1]:
        r = e["result"]; out.append(str(int(round(r))) if abs(r - round(r)) < 1e-6 and r >= 0 else None)
    return out


def slot_step_map(bank_rows: list[dict], readings: dict[int, list[str]], correct: list[bool], *, min_map: float = LADDER_MIN_MAP,
                  min_rows: int = LADDER_MIN_ROWS) -> dict[int, dict[int, list[int]]]:
    """``{k: {step i (1-based): [slots]}}``: slots whose top-1 reading equals step i's result in at
    least ``min_map`` of the correctly answered bank rows with k steps."""
    by_k: dict[int, list[int]] = {}
    for idx, (row, ok) in enumerate(zip(bank_rows, correct)):
        if ok:
            by_k.setdefault(len(parse_chain(str(row["cot"]))), []).append(idx)
    out: dict[int, dict[int, list[int]]] = {}
    for k, rows in by_k.items():
        if len(rows) < min_rows or k < 2:
            continue
        inter = {idx: chain_intermediates(bank_rows[idx]["cot"]) for idx in rows}
        out[k] = {}
        for i in range(1, k):
            slots = [s for s in SLOTS6 if sum(1 for idx in rows if inter[idx][i - 1] is not None and readings[s][idx] == inter[idx][i - 1]) / len(rows) >= min_map]
            if slots:
                out[k][i] = slots
    return out


def ladder_plan(k: int, corrections: list[str | None], readings: dict[int, str], smap: dict[int, dict[int, list[int]]]) -> dict[int, str]:
    """``{slot: value}`` to write: every slot mapped to step i whose reading differs from the corrected value c_i."""
    plan: dict[int, str] = {}
    for i, c in enumerate(corrections, start=1):
        if c is None:
            continue
        for s in smap.get(k, {}).get(i, []):
            if readings.get(s) != c:
                plan[s] = c
    return plan


def classify_failure(*, explicit_answer: float | None, gold: float, corrected: list[float] | None, gold_v_star: str | None, carrier_read: str) -> str:
    """plan_wrong / final_step / upstream_arithmetic / latent_only / no_chain, from the explicit chain and the calculator."""
    if explicit_answer is not None and abs(explicit_answer - gold) < 1e-6:
        return "latent_only"
    if corrected is None or gold_v_star is None:
        return "no_chain"
    plan_right = abs(corrected[-1] - float(gold_v_star)) < 1e-6
    if not plan_right:
        return "plan_wrong"
    return "final_step" if carrier_read == gold_v_star else "upstream_arithmetic"


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
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token,
        shard_sha256=checkpoint_shard_hashes(cfg))
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
    read_slots = list(SLOTS6) if args.ladder else [carrier] + extra
    bank_values = {c: top1_numbers(bank_traj.states[:, c], readout, vocab_ids, strings) for c in read_slots}
    bank_correct = [extract_official_answer_number(o) is not None and abs(extract_official_answer_number(o) - float(r["gold"])) < 1e-6 for o, r in zip(bank_out, bank_rows)]
    print(f"bank {len(bank_rows)} rows; distinct carrier values {len(set(bank_values[carrier]))}; bank native accuracy {sum(bank_correct)/len(bank_correct):.3f}")

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
    gold_values = [r["v_star"] for r in t_rows]
    tool = None
    if args.v_source == "tool":
        chains = explicit_chains(model, tokenizer, [r["question"] for r in t_rows], batch_size=bs, device=device)
        tool_v = [tool_value(c) for c in chains]
        explicit_numbers = [extract_official_answer_number(c) for c in chains]
        explicit_correct = [n is not None and abs(n - r["gold"]) < 1e-6 for n, r in zip(explicit_numbers, t_rows)]
        tool = {"chains": chains, "v": tool_v, "explicit_numbers": explicit_numbers,
                "has_value": sum(v is not None for v in tool_v), "agrees_with_gold": sum(1 for v, g in zip(tool_v, gold_values) if v == g),
                "explicit_accuracy_on_wrong_targets": sum(c for c, ctrl in zip(explicit_correct, is_control) if not ctrl) / max(1, len(wrong_idx)),
                "explicit_accuracy_on_controls": sum(c for c, ctrl in zip(explicit_correct, is_control) if ctrl) / max(1, len(control_idx))}
        print(f"tool: value for {tool['has_value']}/{len(t_rows)} targets, agrees with gold v* on {tool['agrees_with_gold']}; "
              f"explicit mode right on {tool['explicit_accuracy_on_wrong_targets']:.3f} of the latent-wrong targets")
        t_values = [v if v is not None else "__none__" for v in tool_v]
    else:
        t_values = gold_values
    own_read = {c: top1_numbers(t_traj.states[:, c], readout, vocab_ids, strings) for c in read_slots}
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
    if args.writer:
        bank_emb, bank_ok = value_token_embeddings(bank_values[carrier], readout, vocab_ids, strings)
        writers = {"ridge": Writer("ridge").fit(bank_emb[bank_ok], bank_traj.states[bank_ok, carrier]),
                   "mlp": Writer("mlp").fit(bank_emb[bank_ok], bank_traj.states[bank_ok, carrier])}
        own = t_traj.states[:, carrier]
        abs_states, has_value = written_states(writers["ridge"], t_values, own, readout, vocab_ids, strings)
        res_states, has_res = written_states(writers["ridge"], t_values, own, readout, vocab_ids, strings, residual_from=own_read[carrier])
        mlp_states, _ = written_states(writers["mlp"], t_values, own, readout, vocab_ids, strings)
        rng_w = random.Random(WRITER_SEED)
        pool_values = sorted(set(v for v in bank_values[carrier]))
        wrong_values = [rng_w.choice([v for v in pool_values if v != tv] or [tv]) for tv in t_values]
        diff_states, _ = written_states(writers["ridge"], wrong_values, own, readout, vocab_ids, strings)
        writer_coverage = {"has_value": int(has_value[[not c for c in is_control]].sum()), "has_residual": int(has_res[[not c for c in is_control]].sum()),
                           "wrong_targets": len(wrong_idx), "bank_rows_fitted": int(bank_ok.sum())}
        print("writer coverage", writer_coverage)
        run_condition("writer_abs", hidden={carrier: abs_states})
        run_condition("writer_res", hidden={carrier: res_states})
        run_condition("writer_mlp_abs", hidden={carrier: mlp_states})
        run_condition("writer_abs_random_diff", hidden={carrier: diff_states})
        has_donor_backup = list(has_donor)
    taxonomy = None
    if args.ladder:
        smap = slot_step_map(bank_rows, bank_values, bank_correct)
        print("slot-to-step map", {k: v for k, v in sorted(smap.items())})
        if args.v_source == "tool":
            corr_lists = []
            for ch in tool["chains"]:
                rc = recompute_chain(ch)
                corr_lists.append(None if rc is None else [str(int(round(v))) if abs(v - round(v)) < 1e-6 and v >= 0 else None for v in rc[1]])
            explicit_nums = tool["explicit_numbers"]
        else:
            corr_lists = [chain_intermediates(r["cot"]) for r in t_rows]
            explicit_nums = [None] * len(t_rows)
        plans = []
        for t in range(len(sel)):
            corr = corr_lists[t] or []
            plans.append(ladder_plan(t_rows[t]["steps"], corr, {s: own_read[s][t] for s in SLOTS6}, smap))
        ladder_touched = [len(p) > 0 for p in plans]
        # per-slot writers (residual ridge) and per-slot donor indexes
        slot_writers = {}
        for s in SLOTS6:
            e, ok = value_token_embeddings(bank_values[s], readout, vocab_ids, strings)
            slot_writers[s] = Writer("ridge").fit(e[ok], bank_traj.states[ok, s]) if int(ok.sum()) >= 50 else None
        rng_l = random.Random(WRITER_SEED + 1)
        all_values = sorted(set(v for s in SLOTS6 for v in bank_values[s]))

        def ladder_states(kind: str) -> dict[int, torch.Tensor]:
            hidden = {s: t_traj.states[:, s].clone() for s in SLOTS6}
            for t, plan in enumerate(plans):
                for s, value in plan.items():
                    v = value if kind != "random" else rng_l.choice([x for x in all_values if x != value] or [value])
                    if kind in ("writer", "random"):
                        w = slot_writers.get(s)
                        if w is None:
                            continue
                        new, ok = written_states(w, [v], t_traj.states[t:t + 1, s], readout, vocab_ids, strings, residual_from=[own_read[s][t]])
                        if bool(ok[0]):
                            hidden[s][t] = new[0]
                    else:   # donor: nearest bank row whose reading at slot s equals v
                        cands = [i for i, bv in enumerate(bank_values[s]) if bv == v]
                        if cands:
                            q = t_traj.states[t, s].float(); q = q / q.norm().clamp_min(1e-6)
                            b = bank_traj.states[cands, s].float(); b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-6)
                            hidden[s][t] = bank_traj.states[cands[int((b @ q).argmax())], s]
            return {s: h for s, h in hidden.items() if any(s in p for p in plans)}

        run_condition("ladder_writer", hidden=ladder_states("writer"))
        run_condition("ladder_donor", hidden=ladder_states("donor"))
        run_condition("ladder_random", hidden=ladder_states("random"))
        taxonomy = []
        for t in range(len(sel)):
            corr = corr_lists[t]
            rc_float = None if corr is None else [float(c) if c is not None else float("nan") for c in corr]
            taxonomy.append(classify_failure(explicit_answer=explicit_nums[t], gold=gold[t], corrected=rc_float if rc_float else None,
                                             gold_v_star=gold_values[t], carrier_read=own_read[carrier][t]) if not is_control[t] else "control")
        print("taxonomy", {c: taxonomy.count(c) for c in sorted(set(taxonomy))}, "ladder touched", sum(ladder_touched))
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
    writer_block = {}
    if args.writer:
        hv = has_value.tolist()
        wmask = lambda t: not is_control[t] and hv[t] and carrier_wrong[t]
        writer_block = {"coverage": writer_coverage,
                        "primary_with_value": {n: rate([r == "repaired" for t, r in enumerate(res) if wmask(t)]) for n, res in results.items()},
                        "all_wrong_with_value": {n: rate([r == "repaired" for t, r in enumerate(res) if not is_control[t] and hv[t]]) for n, res in results.items()},
                        "harm_with_value": {n: rate([r != "kept" for t, r in enumerate(res) if is_control[t] and hv[t]]) for n, res in results.items()}}
    ladder_block = {}
    if args.ladder:
        classes = sorted(set(taxonomy) - {"control"})
        ladder_block = {"slot_step_map": {str(k): {str(i): v for i, v in m.items()} for k, m in smap.items()},
                        "touched_wrong": sum(1 for t in range(len(sel)) if not is_control[t] and ladder_touched[t]),
                        "touched_controls": sum(1 for t in range(len(sel)) if is_control[t] and ladder_touched[t]),
                        "plan_sizes": {str(n): sum(1 for t in range(len(sel)) if not is_control[t] and len(plans[t]) == n) for n in range(0, 7)},
                        "by_class": {c: {"n": taxonomy.count(c),
                                         "repair": {name: rate([r == "repaired" for t, r in enumerate(res) if taxonomy[t] == c]) for name, res in results.items()},
                                         "repair_touched": {name: rate([r == "repaired" for t, r in enumerate(res) if taxonomy[t] == c and ladder_touched[t]]) for name, res in results.items()}}
                                     for c in classes},
                        "harm_touched": {name: rate([r != "kept" for t, r in enumerate(res) if is_control[t] and ladder_touched[t]]) for name, res in results.items()},
                        "harm_all_controls": {name: rate([r != "kept" for t, r in enumerate(res) if is_control[t]]) for name, res in results.items()}}
    report = {
        "ladder": ladder_block,
        "writer": writer_block,
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
    if tool is not None:
        report["tool"] = {k: v for k, v in tool.items() if k not in ("chains", "v", "explicit_numbers")}
    checks = checks_from(report, args.model_key)
    with (out / "targets.jsonl").open("w") as handle:
        for t in range(len(sel)):
            handle.write(json.dumps({"question": t_rows[t]["question"], "gold": gold[t], "v_star": t_values[t], "gold_v_star": gold_values[t],
                                     "failure_class": None if taxonomy is None else taxonomy[t], "ladder_plan": None if not args.ladder else {str(k): v for k, v in plans[t].items()},
                                     "readings": {str(c): own_read[c][t] for c in read_slots},
                                     "tool_chain": None if tool is None else tool["chains"][t], "steps": t_rows[t]["steps"], "control": is_control[t],
                                     "native": t_native[t], "carrier_read": own_read[carrier][t], "has_donor": has_donor[t],
                                     "nearest_donor": donors["nearest"][t], "outcomes": {k: v[t] for k, v in results.items()},
                                     "numbers": {k: v[t] for k, v in numbers.items()}}) + "\n")
    _atomic_torch_save({"target_states": t_traj.states.half(), "bank_values": bank_values, "bank_questions": [r["question"] for r in bank_rows]}, out / "repair_states.pt")
    summary = {"schema_version": 1, "contract": CONTRACT + ("_tool" if args.v_source == "tool" else "") + ("_smoke" if smoke else ""), "smoke": smoke,
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "v_source": args.v_source,
               "checkpoint_sha256": load_report.checkpoint_sha256, "model_key": args.model_key, "carrier": carrier, "store": store, "extra_carriers": extra,
               "writer": bool(args.writer), "ladder": bool(args.ladder),
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
    parser.add_argument("--v-source", choices=("gold", "tool"), default="gold", help="where v* comes from: the gold chain, or the model's own explicit chain re-evaluated by a calculator (ledger §127 B)")
    parser.add_argument("--writer", action="store_true", help="also fit the learned writer (ledger §128) and run its conditions")
    parser.add_argument("--ladder", action="store_true", help="also run the failure taxonomy and ladder writes (ledger §129)")
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
