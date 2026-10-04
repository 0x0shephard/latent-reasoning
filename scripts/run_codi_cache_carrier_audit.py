"""Cache-carrier audit on the official CODI checkpoint (ledger §108).  No training.

For every thought slot, swap between derangement-paired questions either the slot's
output state (the route a monitor reads), its K/V (the route later positions attend
to), values only, keys only, values at SCIT's layer groups, or the whole position;
and the same at all six slots at once.  Outcome: the first answer token under the
forced cue; native decoding for the headline conditions.
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
    sample_splits,
)
from scripts.run_codi_workspace_interchange_gate import PAIR_SEED, QUESTIONS
from src.data.datasets import load_eval_set, load_train_set
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import extract_official_answer_number
from src.mech.cache_carrier import (
    CacheEdit,
    donor_view,
    generate_under,
    outcomes_under,
    record_native_trajectory,
    record_trajectory,
)
from src.mech.workspace_interchange import change_rate, derangement
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_cache_carrier_audit_v1"
LAYERS = 12
LAYER_GROUPS = {"0_7": tuple(range(0, 8)), "8_9": (8, 9), "10_11": (10, 11)}
SLOT_CONDITIONS = ("hidden", "k", "v", "kv", "v_layers_0_7", "v_layers_8_9", "v_layers_10_11", "hidden_kv")
TAIL_CONDITIONS = ("hidden_all", "k_all", "v_all", "kv_all", "v_all_8_9", "v_all_10_11", "hidden_kv_all")
NATIVE_SLOT = {1: ("hidden", "kv"), 3: ("hidden", "kv"), 5: ("hidden", "kv")}
NATIVE_TAIL = ("kv_all", "hidden_kv_all")
FEEDING_SLOTS = (1, 3)
TERMINAL_SLOT = 5
A1_MIN_FEEDING = 0.25
A2_MIN_TERMINAL_KV = 0.10
A3_DOMINANCE = 1.5
A6_MIN_TOWARD_DONOR = 0.50
A6_MIN_OVER_NULL = 0.30
SANITY_SLACK = 0.02
SMOKE = {"questions": 24}


TAIL_TO_BASE = {"hidden_all": "hidden", "k_all": "k", "v_all": "v", "kv_all": "kv", "v_all_8_9": "v_layers_8_9",
                "v_all_10_11": "v_layers_10_11", "hidden_kv_all": "hidden_kv"}
BASE_SPEC = {  # name -> (swap hidden state, keys, values, layers)
    "hidden": (True, False, False, ()),
    "k": (False, True, False, tuple(range(LAYERS))),
    "v": (False, False, True, tuple(range(LAYERS))),
    "kv": (False, True, True, tuple(range(LAYERS))),
    "v_layers_0_7": (False, False, True, LAYER_GROUPS["0_7"]),
    "v_layers_8_9": (False, False, True, LAYER_GROUPS["8_9"]),
    "v_layers_10_11": (False, False, True, LAYER_GROUPS["10_11"]),
    "hidden_kv": (True, True, True, tuple(range(LAYERS))),
}


def condition_spec(name: str, slots: tuple[int, ...]) -> tuple[tuple[int, ...], list[CacheEdit]]:
    """(slots whose hidden state is swapped, cache edits) for a condition name."""
    base = TAIL_TO_BASE.get(name, name)
    if base not in BASE_SPEC:
        raise ValueError(f"unknown condition {name}")
    hidden, keys, values, layers = BASE_SPEC[base]
    edits = [CacheEdit(slot, layers, keys, values) for slot in slots] if (keys or values) else []
    return (slots if hidden else ()), edits


def checks_from(report: dict) -> dict:
    slots, tail = report["slots"], report["tail"]
    rate = lambda s, c: slots[str(s)][c]["change_rate"]
    feeding = {s: {"hidden": rate(s, "hidden"), "kv": rate(s, "kv")} for s in FEEDING_SLOTS}

    def classify(h, kv):
        if kv >= A3_DOMINANCE * h:
            return "cache_dominant"
        if h >= A3_DOMINANCE * kv:
            return "state_dominant"
        return "shared"

    values_over_keys = [rate(1, "v") >= rate(1, "k"), rate(3, "v") >= rate(3, "k"), rate(5, "v") >= rate(5, "k"),
                        tail["v_all"]["change_rate"] >= tail["k_all"]["change_rate"]]
    transplant = tail["hidden_kv_all"]
    toward = transplant.get("toward_donor_among_changed")
    null = transplant.get("donor_null")
    checks = {
        "a1_replication": all(feeding[s]["hidden"] >= A1_MIN_FEEDING for s in FEEDING_SLOTS) and rate(TERMINAL_SLOT, "hidden") == 0.0,
        "a2_terminal_slot_is_a_memory": rate(TERMINAL_SLOT, "kv") >= A2_MIN_TERMINAL_KV,
        "a3_route_split": {str(s): classify(feeding[s]["hidden"], feeding[s]["kv"]) for s in FEEDING_SLOTS},
        "a4_values_over_keys": sum(values_over_keys) > len(values_over_keys) / 2,
        "a5_layers_8_9_over_10_11": tail["v_all_8_9"]["change_rate"] >= tail["v_all_10_11"]["change_rate"],
        "a6_tail_transplant_donor_directed": (toward is not None and null is not None
                                              and toward >= A6_MIN_TOWARD_DONOR and toward >= null + A6_MIN_OVER_NULL),
        "sanity_both_routes_at_least_each": all(
            rate(s, "hidden_kv") >= max(rate(s, "hidden"), rate(s, "kv")) - SANITY_SLACK for s in range(len(slots))),
    }
    return checks


def claim_from(checks: dict) -> str:
    parts = []
    parts.append("A1 replication " + ("holds" if checks["a1_replication"] else "FAILS"))
    parts.append("terminal slot " + ("is consumed through its K/V (a memory, not a thought)" if checks["a2_terminal_slot_is_a_memory"]
                                      else "is inert through both routes"))
    split = checks["a3_route_split"]
    parts.append("feeding slots: " + ", ".join(f"slot {s} {v.replace('_', '-')}" for s, v in split.items()))
    parts.append("values over keys " + ("holds" if checks["a4_values_over_keys"] else "does not hold"))
    parts.append("layers 8-9 over 10-11 " + ("holds" if checks["a5_layers_8_9_over_10_11"] else "does not hold"))
    parts.append("tail transplant " + ("is donor-directed" if checks["a6_tail_transplant_donor_directed"] else "is not donor-directed"))
    if not checks["sanity_both_routes_at_least_each"]:
        parts.append("SANITY FAILURE: both routes together did less than one route alone")
    return "; ".join(parts)


def _outcome(pred, base_pred, donor_pred, gold):
    changed = pred != base_pred
    n = int(changed.sum())
    return {"change_rate": change_rate(base_pred, pred), "accuracy": float((pred == gold).double().mean()),
            "toward_donor_among_changed": float((pred[changed] == donor_pred[changed]).double().mean()) if n else None,
            "donor_null": float((base_pred[changed] == donor_pred[changed]).double().mean()) if n else None,
            "changed": n}


def run(args):
    smoke = bool(args.smoke)
    n_questions = SMOKE["questions"] if smoke else args.questions
    contract = CONTRACT + ("_smoke" if smoke else "")
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
    all_slots = tuple(range(latent_positions))

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    sizes = {"fit": FIT_EXAMPLES, "select": SELECT_EXAMPLES, "validate": VALIDATE_EXAMPLES}
    dataset = load_train_set(data_cfg, "eq_only")
    splits, sampling = sample_splits(dataset, tokenizer, test_questions={_normalized_question(r["question"]) for r in test},
                                     sizes=sizes, seed=DATA_SEED, bot_token_id=model.bot_id)
    rows = splits["fit"][:n_questions]
    questions = [r["question"] for r in rows]
    perm = derangement(len(rows), PAIR_SEED)
    perm_t = torch.tensor(perm)

    # ---- baseline recording on the teacher-forced path
    traj, base_pred, gold = record_trajectory(model, tokenizer, rows, latent_positions=latent_positions,
                                              batch_size=args.batch_size, device=device)
    donor = donor_view(traj, perm)
    donor_pred = base_pred[perm_t]
    _atomic_torch_save({"states": traj.states.half(), "keys": traj.keys.half(), "values": traj.values.half(),
                        "perm": perm, "questions": questions}, out / "trajectory.pt")
    baseline = {"first_token_accuracy": float((base_pred == gold).double().mean()),
                "pairs_with_different_first_token": float((base_pred != donor_pred).double().mean())}
    print("baseline", baseline)

    def measure(name, slots):
        hidden_slots, edits = condition_spec(name, slots)
        hidden_donors = {s: donor.states[:, s] for s in hidden_slots}
        pred = outcomes_under(model, tokenizer, rows, latent_positions=latent_positions, batch_size=args.batch_size,
                              device=device, hidden_donors=hidden_donors, cache_edits=edits, donor=donor)
        return _outcome(pred, base_pred, donor_pred, gold)

    report = {"slots": {}, "tail": {}}
    for slot in all_slots:
        entry = {c: measure(c, (slot,)) for c in SLOT_CONDITIONS}
        report["slots"][str(slot)] = entry
        print("slot", slot, {c: round(v["change_rate"], 3) for c, v in entry.items()})
    for name in TAIL_CONDITIONS:
        report["tail"][name] = measure(name, all_slots)
        print("tail", name, round(report["tail"][name]["change_rate"], 3),
              "toward donor", report["tail"][name]["toward_donor_among_changed"])

    # ---- native decoding for the headline conditions, with donors recorded on the native path
    native_traj, native_outputs = record_native_trajectory(model, tokenizer, questions, latent_iterations=latent_positions,
                                                           batch_size=args.batch_size, device=device)
    native_donor = donor_view(native_traj, perm)
    base_numbers = [extract_official_answer_number(o) for o in native_outputs]
    native_em = float(sum(1 for n, r in zip(base_numbers, rows) if n is not None and n == float(r["gold"])) / len(rows))
    baseline["native_exact_match"] = native_em

    def native(name, slots):
        hidden_slots, edits = condition_spec(name, slots)
        hidden_donors = {s: native_donor.states[:, s] for s in hidden_slots}
        outputs = generate_under(model, tokenizer, questions, latent_iterations=latent_positions, batch_size=args.batch_size,
                                 device=device, hidden_donors=hidden_donors, cache_edits=edits, donor=native_donor)
        numbers = [extract_official_answer_number(o) for o in outputs]
        changed = [a != b for a, b in zip(numbers, base_numbers)]
        return {"change_rate": float(sum(changed) / len(rows)),
                "exact_match": float(sum(1 for n, r in zip(numbers, rows) if n is not None and n == float(r["gold"])) / len(rows)),
                "equals_donor_answer_among_changed": (float(sum(1 for a, c, i in zip(numbers, changed, perm) if c and a is not None
                                                                 and a == base_numbers[i]) / max(1, sum(changed))))}

    report["native"] = {"slots": {}, "tail": {}}
    for slot, names in NATIVE_SLOT.items():
        if slot < latent_positions:
            report["native"]["slots"][str(slot)] = {c: native(c, (slot,)) for c in names}
            print("native slot", slot, {c: round(v["change_rate"], 3) for c, v in report["native"]["slots"][str(slot)].items()})
    for name in NATIVE_TAIL:
        report["native"]["tail"][name] = native(name, all_slots)
        print("native tail", name, report["native"]["tail"][name])

    checks = checks_from(report)
    summary = {"schema_version": 1, "contract": contract, "smoke": smoke, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
               "preregistration": {"questions": len(rows), "pair_seed": PAIR_SEED, "data_seed": DATA_SEED,
                                   "sampling_hash_fit": sampling["hashes"]["fit"], "slot_conditions": list(SLOT_CONDITIONS),
                                   "tail_conditions": list(TAIL_CONDITIONS), "layer_groups": {k: list(v) for k, v in LAYER_GROUPS.items()},
                                   "a1_min_feeding": A1_MIN_FEEDING, "a2_min_terminal_kv": A2_MIN_TERMINAL_KV,
                                   "a3_dominance": A3_DOMINANCE, "a6_min_toward_donor": A6_MIN_TOWARD_DONOR,
                                   "a6_min_over_null": A6_MIN_OVER_NULL, "sanity_slack": SANITY_SLACK},
               "baseline": baseline, "report": report, "checks": checks, "decision": {"claim": claim_from(checks)},
               "seconds": time.perf_counter() - started,
               "warnings": ["Official checkpoint only; no training. First-token-under-forced-cue is the primary instrument.",
                            "A slot's K/V are the cache entries written by the pass that produced its state; swapping them changes what later positions read, not the next thought's input.",
                            "'toward donor' is the right counterfactual only for the complete tail transplant (hidden_kv_all); elsewhere it is descriptive."]}
    _atomic_json(summary, out / "summary.json")
    print(json.dumps({"checks": checks, "decision": summary["decision"]}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--questions", type=int, default=QUESTIONS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
