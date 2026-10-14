"""Masking (necessity) audit on CODI's latent trajectory (ledger §120).  No training.

Where the swap audits (§108, §110) ask whether a thought is *sufficient* to carry a
value in, this asks whether it is *necessary*: remove it and see whether the answer
survives.  Two masks per slot: ``mean`` replaces the slot's output state by the pool
mean (no question-specific content reaches the next thought); ``mask`` hides the slot's
K/V from every later position through the attention mask.  Outcomes are first-token
accuracy and change rate, split by chain length.
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
from scripts.run_codi_causal_subspace_distillation import DATA_SEED
from scripts.run_codi_workspace_interchange_gate import QUESTIONS
from src.data.chain_pools import DEFAULT_POOL, POOLS, pool_rows
from src.data.counterfactual_chain import parse_chain
from src.eval.official_codi import select_device
from src.mech.cache_carrier import backbone_layers, outcomes_under, record_trajectory
from src.mech.workspace_interchange import change_rate
from src.models.official_codi import (
    build_official_codi_gpt2,
    checkpoint_shard_hashes,
    download_official_checkpoint,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


CONTRACT = "official_codi_masking_audit_v1"
SLOTS = (0, 1, 2, 3, 4, 5)
TERMINAL_SLOT = 5
LONG_MIN_STEPS = 3
K1_MAX_SHORT_DROP = 0.05      # no single slot costs more than this on short chains
K2_MIN_CARRYING_DROP = 0.10   # masking a carrying slot (given per model) costs at least this on long chains
K3_MAX_TERMINAL_DROP = 0.02   # the terminal slot costs nothing, anywhere
SMOKE = {"questions": 24}


def chain_steps(row: dict) -> int:
    return len(parse_chain(str(row["cot"])))


def condition_names() -> tuple[str, ...]:
    return tuple(f"mean_{s}" for s in SLOTS) + tuple(f"mask_{s}" for s in SLOTS) + ("mean_all", "mask_all", "mask_even", "mask_odd")


def spec(name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(slots whose state is mean-ablated, slots whose K/V are masked)."""
    if name == "mean_all":
        return SLOTS, ()
    if name == "mask_all":
        return (), SLOTS
    if name == "mask_even":
        return (), (0, 2, 4)
    if name == "mask_odd":
        return (), (1, 3, 5)
    kind, _, slot = name.rpartition("_")
    if not slot.isdigit() or int(slot) not in SLOTS:
        raise ValueError(f"unknown condition {name}")
    return ((int(slot),), ()) if kind == "mean" else ((), (int(slot),)) if kind == "mask" else (_ for _ in ()).throw(ValueError(name))


def outcome(pred, base_pred, gold, mask=None) -> dict:
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    n = int(mask.sum())
    if n == 0:
        return {"accuracy": None, "accuracy_drop": None, "change_rate": None, "n": 0}
    acc = float((pred[mask] == gold[mask]).double().mean())
    base = float((base_pred[mask] == gold[mask]).double().mean())
    return {"accuracy": acc, "accuracy_drop": base - acc, "change_rate": change_rate(base_pred[mask], pred[mask]), "n": n}


def checks_from(report: dict, carrying_slots: tuple[int, ...]) -> dict:
    short, long = report["by_length"]["short"], report["by_length"]["long"]
    drop = lambda part, name: (part[name]["accuracy_drop"] or 0.0)
    singles = [f"mean_{s}" for s in SLOTS] + [f"mask_{s}" for s in SLOTS]
    k2 = {str(s): max(drop(long, f"mean_{s}"), drop(long, f"mask_{s}")) >= K2_MIN_CARRYING_DROP for s in carrying_slots} if long["mean_0"]["n"] else {}
    return {
        "k1_short_chains_survive_any_single_mask": max(drop(short, n) for n in singles) <= K1_MAX_SHORT_DROP if short["mean_0"]["n"] else None,
        "k2_carrying_slot_needed_on_long_chains": (all(k2.values()) if k2 else None),
        "k2_per_slot": k2,
        "k3_terminal_slot_never_needed": max(drop(report["all"], f"mean_{TERMINAL_SLOT}"), drop(report["all"], f"mask_{TERMINAL_SLOT}")) <= K3_MAX_TERMINAL_DROP,
        "position_0_mask_drop_short": drop(short, "mask_0") if short["mean_0"]["n"] else None,
    }


def claim_from(checks: dict) -> str:
    parts = ["K1 short chains " + ("survive every single mask" if checks["k1_short_chains_survive_any_single_mask"] else "do NOT survive some single mask"),
             "K2 carrying slot(s) " + ("needed on long chains" if checks["k2_carrying_slot_needed_on_long_chains"] else "not needed on long chains")
             if checks["k2_carrying_slot_needed_on_long_chains"] is not None else "K2 not evaluable",
             "K3 terminal slot " + ("never needed" if checks["k3_terminal_slot_never_needed"] else "NEEDED somewhere")]
    return "; ".join(parts)


def run(args):
    smoke = bool(args.smoke)
    n_questions = SMOKE["questions"] if smoke else args.questions
    contract = CONTRACT + ("_smoke" if smoke else "")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    carrying = tuple(int(s) for s in args.carrying_slots)

    cfg = load_config(args.config)
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    device = select_device(args.device)
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token,
        shard_sha256=checkpoint_shard_hashes(cfg))
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
        dtype=dtype, settings=cfg.model, token=token)
    load_report = load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype).eval()
    latent_positions = int(cfg.eval.latent_iterations)

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    pool_name = str(getattr(args, "pool", DEFAULT_POOL) or DEFAULT_POOL)
    pool, pool_meta = pool_rows(pool_name, data_cfg=data_cfg, tokenizer=tokenizer, bot_token_id=model.bot_id, seed=DATA_SEED)
    rows = pool[:n_questions]   # the default pool's first rows are the §110 fit split, as before
    print(f"pool {pool_name}: {len(rows)} of {pool_meta['n']} rows, hash {pool_meta['hash'][:12]}")
    steps = torch.tensor([chain_steps(r) for r in rows])
    is_long = steps >= LONG_MIN_STEPS

    # masking must use the mask-carrying path; the baseline is measured on that same path
    forced = not bool(getattr(model, "pad_aware_generation", False))
    if forced:
        model.pad_aware_generation = True
    traj, base_pred, gold = record_trajectory(model, tokenizer, rows, latent_positions=latent_positions,
                                              batch_size=args.batch_size, device=device)
    mean_state = traj.states.mean(0)   # [S, D]
    baseline = {"first_token_accuracy": float((base_pred == gold).double().mean()), "rows": len(rows),
                "long_rows": int(is_long.sum()), "short_rows": int((~is_long).sum()),
                "mask_carrying_path_forced_for_this_model": forced}
    print("baseline", baseline)

    report = {"all": {}, "by_length": {"short": {}, "long": {}}, "layers": backbone_layers(model)}
    preds: dict[str, torch.Tensor] = {}
    for name in condition_names():
        mean_slots, mask_slots = spec(name)
        hidden_donors = {s: mean_state[s].unsqueeze(0).expand(len(rows), -1) for s in mean_slots}
        pred = outcomes_under(model, tokenizer, rows, latent_positions=latent_positions, batch_size=args.batch_size,
                              device=device, hidden_donors=hidden_donors, mask_slots=mask_slots)
        preds[name] = pred.cpu()
        report["all"][name] = outcome(pred, base_pred, gold)
        report["by_length"]["short"][name] = outcome(pred, base_pred, gold, ~is_long)
        report["by_length"]["long"][name] = outcome(pred, base_pred, gold, is_long)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report["all"][name].items()},
              "| long drop", None if report["by_length"]["long"][name]["accuracy_drop"] is None else round(report["by_length"]["long"][name]["accuracy_drop"], 3))

    # per-row predictions for the §124 row bootstrap of every accuracy drop
    _atomic_torch_save({"conditions": preds, "base_pred": base_pred.cpu(), "gold": gold.cpu(), "is_long": is_long, "steps": steps}, out / "predictions.pt")
    checks = checks_from(report, carrying)
    summary = {"schema_version": 1, "contract": contract, "smoke": smoke, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": load_report.checkpoint_sha256, "reproduction_gate": reproduction,
               "pool": {k: v for k, v in pool_meta.items() if k != "sampling"},
               "preregistration": {"questions": len(rows), "data_seed": DATA_SEED, "sampling_hash_fit": pool_meta["hash"], "pool": pool_name,
                                   "carrying_slots": list(carrying), "long_min_steps": LONG_MIN_STEPS, "k1_max_short_drop": K1_MAX_SHORT_DROP,
                                   "k2_min_carrying_drop": K2_MIN_CARRYING_DROP, "k3_max_terminal_drop": K3_MAX_TERMINAL_DROP,
                                   "base_model": str(cfg.model.base_model)},
               "baseline": baseline, "report": report, "checks": checks, "decision": {"claim": claim_from(checks)},
               "seconds": time.perf_counter() - started,
               "warnings": ["Official checkpoint only; no training. First-token-under-forced-cue is the instrument; masks have no native-decoding counterpart here.",
                            "`mean` replaces a slot's output state by the pool mean; `mask` hides a slot's K/V through the attention mask, on the mask-carrying path (forced on for GPT-2, which otherwise runs the released path)."]}
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
    parser.add_argument("--carrying-slots", nargs="*", default=["3"], help="slots the swap audits found to carry values on long chains (may be empty)")
    parser.add_argument("--pool", choices=sorted(POOLS), default=DEFAULT_POOL,
                        help="row pool (ledger §123); the default reproduces the §120 sampling exactly")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
