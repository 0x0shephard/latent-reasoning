"""Directed counterfactual on CODI's latent store (ledger §110).  No training.

Counterfactual twins differ in one question number; the chain is recomputed.  After a
competence gate, each twin is the donor of the other.  Swapping one odd slot's state,
one even position's layers 8-9 values, or whole tails, the first answer token is
classed as the twin's answer (target), the own answer (retain), or other.  The slot
that holds the changed value is located per pair with the model's own readout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_codi_causal_subspace_distillation import (
    DATA_SEED,
    FIT_EXAMPLES,
    SELECT_EXAMPLES,
    VALIDATE_EXAMPLES,
    _normalized_question,
    prepare_row,
    sample_splits,
)
from src.data.counterfactual_chain import perturb_row
from src.data.datasets import load_eval_set, load_train_set
from src.data.official_codi_training import collate_official_codi_kv_rows
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import extract_official_answer_number
from src.mech.cache_carrier import (
    CacheEdit,
    backbone_layers,
    donor_view,
    group_names,
    layer_groups,
    generate_under,
    latent_path,
    outcomes_under,
    record_native_trajectory,
    record_trajectory,
)
from src.mech.directed_counterfactual import (
    locate_pairs,
    located_specificity,
    outcome,
    readout_matrix,
    slot_numbers,
    unique_location,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_directed_counterfactual_v1"
LAYERS = 12
SLOTS = (0, 1, 2, 3, 4, 5)
ODD_SLOTS = (1, 3, 5)
FEEDING_SLOTS = (1, 3)
EVEN_SLOTS = (0, 2, 4)
CANDIDATE_LIMIT = 1_024
PAIRS = 512
SINGLE_CONDITIONS = tuple(f"state_{s}" for s in SLOTS) + tuple(f"v89_{s}" for s in SLOTS) + tuple(f"kv_{s}" for s in SLOTS) \
    + tuple(f"v1011_{s}" for s in EVEN_SLOTS)
TAIL_CONDITIONS = ("v89_all", "v89_even", "v89_odd", "kv_all", "state_all")
NATIVE_CONDITIONS = ("v89_all", "kv_all", "state_1", "state_3", "v89_2", "v89_4")
D1_MIN_TAIL = 0.50
D2_MIN_EVEN = 0.40
D2_RATIO = 2.0
D2_MIN_SINGLE = 0.20
D3_MIN_SITE = 0.30
D3_RATIO = 2.0
D3_MIN_LOCATED = 20
D4_RATIO = 2.0
SMOKE = {"candidates": 64, "pairs": 12}

# ---- §112 profile: long chains, latest-step perturbation, slot-unique location
CONTRACT_LONG = "official_codi_directed_counterfactual_long_v1"
EXTRA_ROWS = 14_336
CANDIDATE_LIMIT_LONG = 1_536
MIN_STEPS_LONG = 3
E1_MIN_TAIL_KV = 0.50
E2_MIN_STATE = 0.25
E3_MIN_SITE = 0.30
E3_RATIO = 2.0
E3_MIN_UNIQUE = 20
PROFILES = {
    "standard": {"contract": CONTRACT, "min_steps": 1, "extra_rows": 0, "prefer_latest_step": False,
                 "candidate_limit": CANDIDATE_LIMIT},
    "long": {"contract": CONTRACT_LONG, "min_steps": MIN_STEPS_LONG, "extra_rows": EXTRA_ROWS, "prefer_latest_step": True,
             "candidate_limit": CANDIDATE_LIMIT_LONG},
}


def condition_spec(name: str, n_layers: int = LAYERS) -> tuple[tuple[int, ...], list[CacheEdit]]:
    """(slots whose output state is swapped, cache edits).  ``v89`` is SCIT's
    mid-late value group and ``v1011`` the final group; for GPT-2 these are layers
    8-9 and 10-11, for deeper backbones the groups are rescaled (``layer_groups``)."""
    groups = layer_groups(n_layers); names = group_names(n_layers)
    mid, late, all_layers = groups[names["mid"]], groups[names["late"]], tuple(range(n_layers))
    if name == "state_all":
        return SLOTS, []
    if name in ("v89_all", "v89_even", "v89_odd"):
        slots = {"v89_all": SLOTS, "v89_even": EVEN_SLOTS, "v89_odd": ODD_SLOTS}[name]
        return (), [CacheEdit(s, mid, False, True) for s in slots]
    if name == "kv_all":
        return (), [CacheEdit(s, all_layers, True, True) for s in SLOTS]
    kind, _, slot = name.rpartition("_")
    if not slot.isdigit() or int(slot) not in SLOTS:
        raise ValueError(f"unknown condition {name}")
    s = int(slot)
    if kind == "state":
        return (s,), []
    if kind == "v89":
        return (), [CacheEdit(s, mid, False, True)]
    if kind == "v1011":
        return (), [CacheEdit(s, late, False, True)]
    if kind == "kv":
        return (), [CacheEdit(s, all_layers, True, True)]
    raise ValueError(f"unknown condition {name}")


def _target(report: dict, name: str) -> float:
    value = report["conditions"][name]["all"]["target"]
    return float(value) if value is not None else 0.0


def checks_from(report: dict) -> dict:
    even = _target(report, "v89_even"); odd = _target(report, "v89_odd")
    best_single_even = max(_target(report, f"v89_{s}") for s in EVEN_SLOTS)
    d3_sites = {}
    for s in FEEDING_SLOTS:
        site = report["located_specificity"][str(s)]
        if site["n"] < D3_MIN_LOCATED:
            d3_sites[str(s)] = None
            continue
        state_site = site["state_at_site"]["target"] or 0.0
        state_else = max((v["target"] or 0.0) for v in site["state_elsewhere"].values())
        store_site = site["store_at_site"]["target"] or 0.0
        store_else = max((v["target"] or 0.0) for v in site["store_elsewhere"].values())
        d3_sites[str(s)] = bool(state_site >= D3_MIN_SITE and state_site >= D3_RATIO * state_else
                                and store_site >= D3_RATIO * store_else)
    evaluated = [v for v in d3_sites.values() if v is not None]
    pooled_v89 = sum(_target(report, f"v89_{s}") for s in EVEN_SLOTS) / len(EVEN_SLOTS)
    pooled_v1011 = sum(_target(report, f"v1011_{s}") for s in EVEN_SLOTS) / len(EVEN_SLOTS)
    return {
        "d1_tail_replication": _target(report, "v89_all") >= D1_MIN_TAIL,
        "d2_even_store": even >= D2_MIN_EVEN and even >= D2_RATIO * max(odd, 1e-9) and best_single_even >= D2_MIN_SINGLE,
        "d3_located_specificity": bool(evaluated) and all(evaluated),
        "d3_per_site": d3_sites,
        "d4_layer_control": pooled_v89 >= D4_RATIO * max(pooled_v1011, 1e-9),
    }


def claim_from(checks: dict) -> str:
    parts = ["D1 tail " + ("carries the counterfactual" if checks["d1_tail_replication"] else "does NOT carry the counterfactual (≥0.5)"),
             "D2 even store " + ("holds" if checks["d2_even_store"] else "does not hold"),
             "D3 located specificity " + ("holds: the thought that decodes the value is the one that carries it"
                                           if checks["d3_located_specificity"] else "does not hold"
                                           + (" (no site had enough located pairs)" if not any(v is not None for v in checks["d3_per_site"].values()) else "")),
             "D4 layers 8-9 over 10-11 " + ("holds" if checks["d4_layer_control"] else "does not hold")]
    return "; ".join(parts)


def checks_long_from(report: dict) -> dict:
    e3_sites = {}
    for s in FEEDING_SLOTS:
        site = report["unique_specificity"][str(s)]
        if site["n"] < E3_MIN_UNIQUE:
            e3_sites[str(s)] = None
            continue
        state_site = site["state_at_site"]["target"] or 0.0
        state_else = max((v["target"] or 0.0) for v in site["state_elsewhere"].values())
        kv_site = site["kv_store_at_site"]["target"] or 0.0
        kv_else = max((v["target"] or 0.0) for v in site["kv_store_elsewhere"].values())
        e3_sites[str(s)] = bool(state_site >= E3_MIN_SITE and state_site >= E3_RATIO * state_else and kv_site >= E3_RATIO * kv_else)
    evaluated = [v for v in e3_sites.values() if v is not None]
    return {
        "e1_long_chain_tail": _target(report, "kv_all") >= E1_MIN_TAIL_KV,
        "e2_single_thought_transport": max(_target(report, f"state_{s}") for s in FEEDING_SLOTS) >= E2_MIN_STATE,
        "e3_unique_location_specificity": bool(evaluated) and all(evaluated),
        "e3_per_site": e3_sites,
    }


def claim_long_from(checks: dict) -> str:
    parts = ["E1 tail cache " + ("carries the counterfactual on long chains" if checks["e1_long_chain_tail"] else "does NOT reach 0.5 on long chains"),
             "E2 single-thought transport " + ("exists on long chains" if checks["e2_single_thought_transport"] else "is absent (the §111 stratum effect was noise)"),
             "E3 unique-location specificity " + ("holds: the thought that decodes the value carries it" if checks["e3_unique_location_specificity"]
                                                   else "does not hold" + (" (no site had enough uniquely located rows)"
                                                                           if not any(v is not None for v in checks["e3_per_site"].values()) else ""))]
    return "; ".join(parts)


def _first_tokens(model, tokenizer, rows, *, latent_positions, batch_size, device):
    preds, golds = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = collate_official_codi_kv_rows(tokenizer, rows[start:start + batch_size], bot_token_id=model.bot_id,
                                                  enforce_answer_eligibility=False).to(device)
            logits, gold, _ = latent_path(model, batch, latent_positions=latent_positions)
            preds.append(logits.argmax(-1).cpu()); golds.append(gold.cpu())
    return torch.cat(preds), torch.cat(golds)


def build_candidates(rows, tokenizer, *, bot_token_id, limit, min_steps=1, prefer_latest_step=False):
    """(original row, twin row, perturbation) triples, in pool order."""
    out = []
    for row in rows:
        pert = perturb_row(row, prefer_latest_step=prefer_latest_step)
        if pert is None or pert.steps < min_steps:
            continue
        twin = prepare_row(tokenizer, {"question": pert.question, "cot": pert.cot, "answer": pert.answer}, bot_token_id=bot_token_id)
        if twin is None:
            continue
        out.append((row, twin, pert))
        if len(out) >= limit:
            break
    return out


def run(args):
    smoke = bool(args.smoke)
    profile = PROFILES[args.profile]
    candidate_limit = SMOKE["candidates"] if smoke else profile["candidate_limit"]
    max_pairs = SMOKE["pairs"] if smoke else args.pairs
    contract = profile["contract"] + ("_smoke" if smoke else "")
    if args.batch_size % 2:
        raise ValueError("batch size must be even so twins share a batch")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

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
    model.to(device=device, dtype=dtype).eval()
    latent_positions = int(cfg.eval.latent_iterations)
    bs = args.batch_size
    n_layers = backbone_layers(model)
    print("backbone layers", n_layers, "value groups", layer_groups(n_layers))

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    sizes = {"fit": FIT_EXAMPLES, "select": SELECT_EXAMPLES, "validate": VALIDATE_EXAMPLES}
    if profile["extra_rows"]:
        sizes["extra"] = int(profile["extra_rows"])   # drawn after the three splits, which stay identical
    dataset = load_train_set(data_cfg, "eq_only")
    splits, sampling = sample_splits(dataset, tokenizer, test_questions={_normalized_question(r["question"]) for r in test},
                                     sizes=sizes, seed=DATA_SEED, bot_token_id=model.bot_id)
    pool = list(splits["fit"]) + list(splits.get("extra", []))
    candidates = build_candidates(pool, tokenizer, bot_token_id=model.bot_id, limit=candidate_limit,
                                  min_steps=profile["min_steps"], prefer_latest_step=profile["prefer_latest_step"])
    print(f"candidates: {len(candidates)} twins from {len(pool)} pool rows (min steps {profile['min_steps']})")

    # ---- competence gate on the interleaved candidate list
    cand_rows = [r for orig, twin, _ in candidates for r in (orig, twin)]
    cand_pred, cand_gold = _first_tokens(model, tokenizer, cand_rows, latent_positions=latent_positions, batch_size=bs, device=device)
    with torch.inference_mode():
        cand_native = generate_official_codi(model, tokenizer, [r["question"] for r in cand_rows], latent_iterations=latent_positions,
                                             max_new_tokens=64, batch_size=bs, device=device)
    native_ok = [extract_official_answer_number(o) == float(r["gold"]) for o, r in zip(cand_native, cand_rows)]
    first_ok = (cand_pred == cand_gold).tolist()
    selected = []
    for p, (orig, twin, pert) in enumerate(candidates):
        a, b = 2 * p, 2 * p + 1
        if native_ok[a] and native_ok[b] and first_ok[a] and first_ok[b] and int(cand_gold[a]) != int(cand_gold[b]):
            selected.append((orig, twin, pert))
        if len(selected) >= max_pairs:
            break
    gate = {"candidates": len(candidates), "native_correct_both": sum(1 for p in range(len(candidates)) if native_ok[2 * p] and native_ok[2 * p + 1]),
            "selected_pairs": len(selected), "candidate_native_accuracy": float(sum(native_ok) / max(1, len(native_ok))),
            "candidate_first_token_accuracy": float(sum(first_ok) / max(1, len(first_ok)))}
    print("gate", gate)
    if len(selected) < 2:
        raise RuntimeError("fewer than two gated pairs; nothing to measure")

    rows = [r for orig, twin, _ in selected for r in (orig, twin)]
    perts = [pert for _, _, pert in selected]
    partner = [i ^ 1 for i in range(len(rows))]
    strata = [f"n{p.steps}_k{p.step}" for p in perts for _ in (0, 1)]
    changed_values = [list(p.changed_values) if d == 0 else [(new, old) for old, new in p.changed_values]
                      for p in perts for d in (0, 1)]
    with (out / "pairs.jsonl").open("w") as handle:
        for orig, twin, pert in selected:
            handle.write(json.dumps({"question": orig["question"], "cot": orig["cot"], "gold": orig["gold"],
                                     "twin_question": twin["question"], "twin_cot": twin["cot"], "twin_gold": twin["gold"],
                                     "step": pert.step, "steps": pert.steps, "original_number": pert.original_number,
                                     "new_number": pert.new_number, "changed_values": list(pert.changed_values)}) + "\n")

    # ---- baseline recording and donors (the twin)
    traj, base_pred, gold = record_trajectory(model, tokenizer, rows, latent_positions=latent_positions, batch_size=bs, device=device)
    donor = donor_view(traj, partner)
    partner_t = torch.tensor(partner)
    own_gold, target_gold = gold, gold[partner_t]
    baseline = {"first_token_accuracy": float((base_pred == own_gold).double().mean()),
                "first_token_equals_target": float((base_pred == target_gold).double().mean()), "rows": len(rows)}
    print("baseline", baseline)
    _atomic_torch_save({"states": traj.states.half(), "partner": partner, "strata": strata}, out / "trajectory_states.pt")

    # ---- locate the changed value with the model's own readout
    numbers = slot_numbers(traj.states, readout_matrix(model), tokenizer)
    located = locate_pairs(numbers, partner, changed_values, ODD_SLOTS)
    located_report = {"fraction_located": float(sum(1 for l in located if l) / len(located)),
                      "rows_by_slot": {str(s): sum(1 for l in located if s in l) for s in ODD_SLOTS},
                      "rows_multi_slot": sum(1 for l in located if len(l) > 1)}
    print("located", located_report)

    # ---- interventions
    preds: dict[str, torch.Tensor] = {}
    report = {"conditions": {}}
    stratum_names = sorted(set(strata))
    strata_t = {name: torch.tensor([s == name for s in strata]) for name in stratum_names}
    direction_t = {"q_receives_twin": torch.tensor([i % 2 == 0 for i in range(len(rows))]),
                   "twin_receives_q": torch.tensor([i % 2 == 1 for i in range(len(rows))])}
    for name in SINGLE_CONDITIONS + TAIL_CONDITIONS:
        hidden_slots, edits = condition_spec(name, n_layers)
        pred = outcomes_under(model, tokenizer, rows, latent_positions=latent_positions, batch_size=bs, device=device,
                              hidden_donors={s: donor.states[:, s] for s in hidden_slots}, cache_edits=edits, donor=donor)
        preds[name] = pred
        entry = {"all": outcome(pred, own_gold, target_gold),
                 "by_stratum": {k: outcome(pred, own_gold, target_gold, m) for k, m in strata_t.items()},
                 "by_direction": {k: outcome(pred, own_gold, target_gold, m) for k, m in direction_t.items()}}
        report["conditions"][name] = entry
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry["all"].items()})
    report["located"] = located_report
    report["located_specificity"] = located_specificity(preds, located, own_gold, target_gold, feeding_slots=FEEDING_SLOTS)
    unique = unique_location(located)
    report["unique_location"] = {"fraction_unique": float(sum(1 for u in unique if u is not None) / len(unique)),
                                 "rows_by_slot": {str(s): sum(1 for u in unique if u == s) for s in ODD_SLOTS}}
    report["unique_specificity"] = located_specificity(preds, located, own_gold, target_gold, feeding_slots=FEEDING_SLOTS, unique_only=True)
    report["unique_terminal"] = {
        "n": sum(1 for u in unique if u == 5),
        "state_5": outcome(preds["state_5"], own_gold, target_gold, torch.tensor([u == 5 for u in unique])),
        "kv_5": outcome(preds["kv_5"], own_gold, target_gold, torch.tensor([u == 5 for u in unique]))}
    print("unique location", report["unique_location"])
    report["located_terminal"] = {
        "n": sum(1 for l in located if 5 in l),
        "state_5": outcome(preds["state_5"], own_gold, target_gold, torch.tensor([5 in l for l in located])),
        "kv_5": outcome(preds["kv_5"], own_gold, target_gold, torch.tensor([5 in l for l in located]))}
    _atomic_torch_save({name: p for name, p in preds.items()} | {"own_gold": own_gold, "target_gold": target_gold,
                                                                 "located": located}, out / "predictions.pt")

    # ---- native decoding for the headline conditions
    questions = [r["question"] for r in rows]
    native_traj, native_outputs = record_native_trajectory(model, tokenizer, questions, latent_iterations=latent_positions,
                                                           batch_size=bs, device=device)
    native_donor = donor_view(native_traj, partner)
    base_numbers = [extract_official_answer_number(o) for o in native_outputs]
    own_numbers = [float(r["gold"]) for r in rows]
    target_numbers = [own_numbers[j] for j in partner]
    baseline["native_exact_match"] = float(sum(1 for a, b in zip(base_numbers, own_numbers) if a == b) / len(rows))
    report["native"] = {}
    for name in NATIVE_CONDITIONS:
        hidden_slots, edits = condition_spec(name, n_layers)
        outputs = generate_under(model, tokenizer, questions, latent_iterations=latent_positions, batch_size=bs, device=device,
                                 hidden_donors={s: native_donor.states[:, s] for s in hidden_slots}, cache_edits=edits, donor=native_donor)
        numbers = [extract_official_answer_number(o) for o in outputs]
        report["native"][name] = {
            "counterfactual_exact_match": float(sum(1 for a, b in zip(numbers, target_numbers) if a == b) / len(rows)),
            "own_exact_match": float(sum(1 for a, b in zip(numbers, own_numbers) if a == b) / len(rows)),
            "changed": float(sum(1 for a, b in zip(numbers, base_numbers) if a != b) / len(rows))}
        print("native", name, report["native"][name])

    checks = checks_long_from(report) if args.profile == "long" else checks_from(report)
    claim = claim_long_from(checks) if args.profile == "long" else claim_from(checks)
    summary = {"schema_version": 1, "contract": contract, "profile": args.profile, "smoke": smoke, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
               "preregistration": {"candidate_limit": candidate_limit, "max_pairs": max_pairs, "data_seed": DATA_SEED,
                                   "sampling_hash_fit": sampling["hashes"]["fit"], "single_conditions": list(SINGLE_CONDITIONS),
                                   "tail_conditions": list(TAIL_CONDITIONS), "native_conditions": list(NATIVE_CONDITIONS),
                                   "d1_min_tail": D1_MIN_TAIL, "d2_min_even": D2_MIN_EVEN, "d2_ratio": D2_RATIO, "d2_min_single": D2_MIN_SINGLE,
                                   "d3_min_site": D3_MIN_SITE, "d3_ratio": D3_RATIO, "d3_min_located": D3_MIN_LOCATED, "d4_ratio": D4_RATIO,
                                   "backbone_layers": n_layers, "layer_groups": {k: list(v) for k, v in layer_groups(n_layers).items()},
                                   "base_model": str(cfg.model.base_model),
                                   "profile": dict(profile), "e1_min_tail_kv": E1_MIN_TAIL_KV, "e2_min_state": E2_MIN_STATE,
                                   "e3_min_site": E3_MIN_SITE, "e3_ratio": E3_RATIO, "e3_min_unique": E3_MIN_UNIQUE},
               "gate": gate, "baseline": baseline, "strata_counts": {k: int(m.sum()) for k, m in strata_t.items()},
               "report": report, "checks": checks, "decision": {"claim": claim},
               "seconds": time.perf_counter() - started,
               "warnings": ["Official checkpoint only; no training. Twins share a batch so their position ids match.",
                            "Target = the twin's gold first token under the forced cue; both directions are pooled.",
                            "Located pairs use the model's own top-5 readout of odd-slot states; multi-token values cannot be located."]}
    _atomic_json(summary, out / "summary.json")
    print(json.dumps({"checks": checks, "decision": summary["decision"]}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
