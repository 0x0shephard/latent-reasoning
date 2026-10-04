"""Workspace interchange go/no-go on the official CODI checkpoint (ledger §105).

No training.  Measures whether swapping a thought slot's state (or a low-rank
subspace of it) between two questions changes the answer, whether that is specific to
the odd (value-holding) slots, and whether a gradient score over the slot's principal
directions agrees with the per-direction interchange effect.  GO = M1 & M2 & M3 & M4.
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
from src.data.datasets import load_eval_set, load_train_set
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import extract_official_answer_number
from src.mech.workspace_interchange import (
    change_rate,
    derangement,
    first_token_outcomes,
    fit_slot_pca,
    generate_with_interchange,
    record_slot_states,
    slot_gradient_scores,
    spearman,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_workspace_interchange_gate_v1"
QUESTIONS = 512
CANDIDATES = 64
RANKS = (12, 28, 64, 128)
ODD_SLOTS = (1, 3)            # odd slots whose output state feeds the next thought
EVEN_SLOTS = (0, 2, 4)
TERMINAL_SLOT = 5             # its output state is projected but never consumed by the released path
TERMINAL_MAX_CHANGE = 0.0
NATIVE_RANK = 64
M1_MIN_CHANGE = 0.30
M2_SHARE_OF_FULL = 0.50
M2_MAX_RANK = 64
M3_ODD_OVER_EVEN = 1.5
M4_MAX_SPEARMAN = 0.70
PAIR_SEED = 20_261_004
SMOKE = {"questions": 24, "candidates": 6, "ranks": (4, 8), "native_rank": 8}


def gates_from(report: dict) -> dict:
    odd = [report["slots"][str(k)] for k in ODD_SLOTS]
    even = [report["slots"][str(k)] for k in EVEN_SLOTS]
    terminal = report["slots"].get(str(TERMINAL_SLOT))
    full_rate = lambda s: s["first_token"]["full"]["change_rate"]
    best_low = lambda s: max((v["change_rate"] for r, v in s["first_token"].items()
                              if r != "full" and int(r) <= M2_MAX_RANK), default=0.0)
    rank_rate = lambda s, r: s["first_token"].get(str(r), {}).get("change_rate", 0.0)
    gate = {
        "m1_slots_matter": all(full_rate(s) >= M1_MIN_CHANGE for s in odd),
        "m2_low_rank_mediation": all(best_low(s) >= M2_SHARE_OF_FULL * full_rate(s) for s in odd),
        "m3_specificity": (
            sum(full_rate(s) for s in odd) / len(odd) >= M3_ODD_OVER_EVEN * max(1e-9, sum(full_rate(s) for s in even) / len(even))
            and sum(rank_rate(s, M2_MAX_RANK) for s in odd) / len(odd)
            >= M3_ODD_OVER_EVEN * max(1e-9, sum(rank_rate(s, M2_MAX_RANK) for s in even) / len(even))),
        "m4_divergence": all(s["spearman_gradient_vs_intervention"] is not None
                             and s["spearman_gradient_vs_intervention"] <= M4_MAX_SPEARMAN for s in odd),
    }
    gate["go"] = all(gate.values())
    # reported, not gated: the terminal slot's output state must be inert by construction
    gate["terminal_slot_inert"] = None if terminal is None else full_rate(terminal) <= TERMINAL_MAX_CHANGE
    return gate


def claim_from(gate: dict) -> str:
    if gate.get("go"):
        return "GO: the odd slots mediate the answer through a low-rank subspace, specifically, and gradients rank the mediating directions differently from interventions"
    failed = [k for k in ("m1_slots_matter", "m2_low_rank_mediation", "m3_specificity", "m4_divergence") if not gate[k]]
    reasons = {"m1_slots_matter": "the odd slots do not change the answer under full interchange",
               "m2_low_rank_mediation": "no subspace of rank <= 64 carries half of the full-slot effect",
               "m3_specificity": "the effect is not specific to the odd slots",
               "m4_divergence": "gradient scores already rank the mediating directions as the interventions do, so an intervention-selected trajectory target cannot differ from a gradient-selected one"}
    return "STOP: " + "; ".join(reasons[k] for k in failed)


def run(args):
    smoke = bool(args.smoke)
    n_questions = SMOKE["questions"] if smoke else args.questions
    candidates = SMOKE["candidates"] if smoke else CANDIDATES
    ranks = SMOKE["ranks"] if smoke else RANKS
    native_rank = SMOKE["native_rank"] if smoke else NATIVE_RANK
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

    # ---- questions: the §86 fit split (train distribution, disjoint from GSM8K test)
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    sizes = {"fit": FIT_EXAMPLES, "select": SELECT_EXAMPLES, "validate": VALIDATE_EXAMPLES}
    dataset = load_train_set(data_cfg, "eq_only")
    splits, sampling = sample_splits(dataset, tokenizer, test_questions={_normalized_question(r["question"]) for r in test},
                                     sizes=sizes, seed=DATA_SEED, bot_token_id=model.bot_id)
    rows = splits["fit"][:n_questions]
    questions = [r["question"] for r in rows]
    perm = derangement(len(rows), PAIR_SEED)
    donor_rows = [rows[i] for i in perm]

    # ---- slot states and native baseline
    recording = record_slot_states(model, tokenizer, questions, latent_iterations=latent_positions,
                                   batch_size=args.batch_size, device=device)
    states = recording.states                                    # [N, S, D]
    _atomic_torch_save({"states": states, "perm": perm, "questions": questions}, out / "slot_states.pt")
    base_numbers = [extract_official_answer_number(o) for o in recording.outputs]
    base_pred, gold = first_token_outcomes(model, tokenizer, rows, latent_positions=latent_positions,
                                           batch_size=args.batch_size, device=device)
    donor_pred = base_pred[torch.tensor(perm)]
    baseline = {"first_token_accuracy": float((base_pred == gold).double().mean()),
                "native_exact_match": float(sum(1 for n, r in zip(base_numbers, rows) if n is not None and n == float(r["gold"])) / len(rows)),
                "pairs_with_different_first_token": float((base_pred != donor_pred).double().mean())}
    print("baseline", baseline)

    # ---- per-slot interchange across ranks, plus per-direction scans at odd slots
    report = {"slots": {}}
    for slot in range(latent_positions):
        pca = fit_slot_pca(states[:, slot])
        donors = states[perm, slot]
        entry = {"eigenvalue_share_top4": float(pca.eigenvalues[:4].sum() / pca.eigenvalues.sum()), "first_token": {}}
        for r in (*ranks, None):
            basis = None if r is None else pca.basis[:, :r].float()
            pred, _ = first_token_outcomes(model, tokenizer, rows, latent_positions=latent_positions, batch_size=args.batch_size,
                                           device=device, slot=slot, donor_states=donors, basis=basis)
            changed = pred != base_pred
            entry["first_token"]["full" if r is None else str(r)] = {
                "change_rate": change_rate(base_pred, pred),
                "accuracy": float((pred == gold).double().mean()),
                "toward_donor_among_changed": float((pred[changed] == donor_pred[changed]).double().mean()) if int(changed.sum()) else None,
                "donor_null": float((base_pred[changed] == donor_pred[changed]).double().mean()) if int(changed.sum()) else None}
        if slot in ODD_SLOTS or smoke:
            per_direction = []
            for i in range(min(candidates, pca.basis.shape[1])):
                pred, _ = first_token_outcomes(model, tokenizer, rows, latent_positions=latent_positions, batch_size=args.batch_size,
                                               device=device, slot=slot, donor_states=donors, basis=pca.basis[:, i:i + 1].float())
                per_direction.append(change_rate(base_pred, pred))
            grad_scores = slot_gradient_scores(model, tokenizer, rows, latent_positions=latent_positions, slot=slot, pca=pca,
                                               batch_size=args.batch_size, device=device, candidates=min(candidates, pca.basis.shape[1]))
            entry["per_direction_change_rate"] = per_direction
            entry["per_direction_gradient_score"] = grad_scores.tolist()
            entry["spearman_gradient_vs_intervention"] = spearman(grad_scores.tolist(), per_direction) if len(per_direction) > 2 else None
            entry["top8_by_intervention"] = sorted(range(len(per_direction)), key=lambda i: -per_direction[i])[:8]
            entry["top8_by_gradient"] = sorted(range(len(per_direction)), key=lambda i: -float(grad_scores[i]))[:8]
        else:
            entry["spearman_gradient_vs_intervention"] = None
        if slot in ODD_SLOTS:
            native = {}
            for label, basis in (("full", None), (str(native_rank), pca.basis[:, :native_rank].float())):
                outputs = generate_with_interchange(model, tokenizer, questions, donors, slot=slot, basis=basis,
                                                    latent_iterations=latent_positions, batch_size=args.batch_size, device=device)
                numbers = [extract_official_answer_number(o) for o in outputs]
                native[label] = {
                    "change_rate": float(sum(1 for a, b in zip(numbers, base_numbers) if a != b) / len(rows)),
                    "exact_match": float(sum(1 for n, r in zip(numbers, rows) if n is not None and n == float(r["gold"])) / len(rows)),
                    "equals_donor_answer_among_changed": (
                        float(sum(1 for a, b, i in zip(numbers, base_numbers, perm) if a != b and a is not None and a == base_numbers[i])
                              / max(1, sum(1 for a, b in zip(numbers, base_numbers) if a != b)))),
                }
            entry["native"] = native
        report["slots"][str(slot)] = entry
        print("slot", slot, {r: round(v["change_rate"], 3) for r, v in entry["first_token"].items()},
              "spearman", entry["spearman_gradient_vs_intervention"])

    gate = gates_from(report)
    summary = {"schema_version": 1, "contract": contract, "smoke": smoke, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
               "preregistration": {"questions": len(rows), "candidates": candidates, "ranks": list(ranks), "native_rank": native_rank,
                                   "odd_slots": list(ODD_SLOTS), "even_slots": list(EVEN_SLOTS), "terminal_slot": TERMINAL_SLOT,
                                   "pair_seed": PAIR_SEED,
                                   "m1_min_change": M1_MIN_CHANGE, "m2_share_of_full": M2_SHARE_OF_FULL, "m2_max_rank": M2_MAX_RANK,
                                   "m3_odd_over_even": M3_ODD_OVER_EVEN, "m4_max_spearman": M4_MAX_SPEARMAN,
                                   "data_seed": DATA_SEED, "sampling_hash_fit": sampling["hashes"]["fit"]},
               "baseline": baseline, "report": report, "gate": gate,
               "decision": {"claim": claim_from(gate)}, "seconds": time.perf_counter() - started,
               "warnings": ["Official checkpoint only; no training. First-token-under-forced-cue is the primary instrument; native decoding is reported for the odd slots.",
                            "Slot 5 is the terminal thought: the released path projects its state and then feeds the end-of-thought token, so its output state cannot affect the answer; it is reported as a negative control, not gated.",
                            "A change rate shows a subspace affects the answer, not that it is the variable (subspace-patching illusion); directedness is reported."]}
    _atomic_json(summary, out / "summary.json")
    print(json.dumps({"gate": gate, "decision": summary["decision"]}, indent=2))
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
