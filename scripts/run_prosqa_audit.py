"""The audit on ProsQA (ledger §135) for a Coconut or a CODI checkpoint, native decoding throughout.  No training.

Twins rewrite the gold path's last hop so the other option becomes the reachable one (checked by
graph reachability); transport is the share of swaps that move the answer to the twin's option.
Masking replaces a thought's state or K/V by the pool mean (and, for Coconut, hides it).  The
monitor layer is not evaluable on ProsQA twins (they share every intermediate node), so the
card reports readability of the prompt's names as the leakage control and marks monitors so.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.run_codi_latent_repair import rate
from src.data.prosqa import answer_option, gold_option, load_prosqa, options, twin_pairs
from src.mech.cache_carrier import CacheEdit, Trajectory, donor_view
from src.mech.cache_carrier import generate_under, record_native_trajectory
from src.mech.cache_carrier import backbone_layers
from src.models.official_codi import (build_official_codi_gpt2, checkpoint_shard_hashes, download_official_checkpoint, load_official_checkpoint,
                                      place_official_model)
from src.utils.config import load_config

CONTRACT = "prosqa_latent_audit_v1"
SLOTS = (0, 1, 2, 3, 4, 5)
SHORT_MAX_HOPS = 3
CARRIER_MIN, INERT_MAX, CRITICAL_COST = 0.15, 0.05, 0.10
GATE_TOLERANCE = 0.03
SMOKE = {"rows": 40, "pairs": 10}
MAX_NEW_TOKENS = 24
PUBLISHED = {"coconut": 0.976, "codi": 0.990}   # Dilgren and Wiegreffe (2026), Table 1, Llama-3.2-1B-Instruct


# ---------------------------------------------------------------- outcomes
def outcome(text: str, opts: tuple[str, str], own: str, target: str) -> str:
    got = answer_option(text, opts)
    return "target" if got == target else "retain" if got == own else "other"


def outcome_rates(outcomes: list[str], mask: list[bool] | None = None) -> dict:
    idx = [i for i in range(len(outcomes)) if mask is None or mask[i]]
    if not idx:
        return {"target": None, "retain": None, "other": None, "n": 0}
    t = [outcomes[i] == "target" for i in idx]
    return {"target": sum(t) / len(idx), "target_ci95": rate(t)["ci95"], "retain": sum(outcomes[i] == "retain" for i in idx) / len(idx),
            "other": sum(outcomes[i] == "other" for i in idx) / len(idx), "n": len(idx)}


def drop(base: list[bool], cond: list[bool], mask: list[bool] | None = None, *, samples: int = 2000, seed: int = 0) -> dict:
    idx = np.array([i for i in range(len(base)) if mask is None or mask[i]])
    if len(idx) == 0:
        return {"drop": None, "ci95": None, "n": 0}
    b = np.array(base, dtype=float)[idx]; c = np.array(cond, dtype=float)[idx]
    rng = np.random.default_rng(seed); boots = []
    for _ in range(samples):
        pick = rng.integers(0, len(idx), len(idx)); boots.append(float(b[pick].mean() - c[pick].mean()))
    return {"drop": float(b.mean() - c.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n": int(len(idx))}


# ---------------------------------------------------------------- the card-like summary
def summarize(*, twins: dict, masking: dict, family: str) -> dict:
    conds = twins.get("conditions", {})
    def tr(name, part):
        return (conds.get(name, {}).get(part) or {}).get("target")
    slots = {}
    for s in SLOTS:
        costs = [masking.get(f"mean_{s}", {}).get("long", {}).get("drop"), masking.get(f"meankv_{s}", {}).get("long", {}).get("drop"), masking.get(f"mask_{s}", {}).get("long", {}).get("drop")]
        costs = [c for c in costs if c is not None]
        slots[str(s)] = {"transport_short": tr(f"state_{s}", "short"), "transport_long": tr(f"state_{s}", "long"), "kv_transport_short": tr(f"kv_{s}", "short"),
                         "kv_transport_long": tr(f"kv_{s}", "long"), "mask_cost_long": max(costs) if costs else None,
                         "mask_cost_all": max([c for c in (masking.get(f"mean_{s}", {}).get("all", {}).get("drop"), masking.get(f"meankv_{s}", {}).get("all", {}).get("drop")) if c is not None] or [float("nan")])}
    long_t = {s: slots[str(s)]["transport_long"] or 0.0 for s in SLOTS}
    short_t = {s: slots[str(s)]["transport_short"] or 0.0 for s in SLOTS}
    carriers = [s for s in SLOTS if long_t[s] >= CARRIER_MIN]
    if not carriers:
        regime = "no_transport"
    else:
        regime = "critical_path" if any((slots[str(s)]["mask_cost_long"] or 0.0) >= CRITICAL_COST for s in carriers) else "backup_route"
    tails = {"kv_all_transport_long": tr("kv_all", "long"), "kv_all_transport_all": tr("kv_all", "all"), "state_all_transport_long": tr("state_all", "long"),
             "mean_all_cost_all": masking.get("mean_all", {}).get("all", {}).get("drop"), "meankv_all_cost_all": masking.get("meankv_all", {}).get("all", {}).get("drop"),
             "mask_all_cost_all": masking.get("mask_all", {}).get("all", {}).get("drop")}
    return {"slots": slots, "carriers": carriers, "regime": regime, "max_short_transport": max(short_t.values()), "max_long_transport": max(long_t.values()),
            "tails": tails, "monitors": "not_evaluable: ProsQA twins share every intermediate node", "family": family}


def checks_from(card: dict, leakage: dict | None, gate: dict) -> dict:
    slots = card["slots"]; tails = card["tails"]
    single_costs = [v["mask_cost_all"] for v in slots.values() if v["mask_cost_all"] is not None and v["mask_cost_all"] == v["mask_cost_all"]]
    all_costs = [c for c in (tails["mean_all_cost_all"], tails["meankv_all_cost_all"], tails["mask_all_cost_all"]) if c is not None]
    leak = None
    if leakage:
        vals = [leakage[k]["slots"][str(s)] for k in ("subject", "option_1", "option_2") if k in leakage for s in SLOTS if leakage[k]["slots"].get(str(s)) is not None]
        leak = min(vals) >= 0.80 if vals else None
    return {"p25a_no_single_slot_costs_over_2_points": None if not single_costs else max(single_costs) <= 0.02,
            "p25a_all_six_cost_under_5_points": None if not all_costs else max(all_costs) < 0.05,
            "p25b_no_single_site_transports_0_10": max(card["max_short_transport"], card["max_long_transport"], max((v["kv_transport_long"] or 0.0) for v in slots.values())) < 0.10,
            "p25b_tail_kv_swap_under_0_30": None if tails["kv_all_transport_all"] is None else tails["kv_all_transport_all"] < 0.30,
            "p25b_regime_no_transport": card["regime"] == "no_transport",
            "p25c_names_readable_from_every_slot_0_80": leak,
            "p25d_accuracy_within_3_points_of_published": None if gate.get("published") is None else abs(gate["accuracy"] - gate["published"]) <= GATE_TOLERANCE}


def markdown(summary: dict) -> str:
    c = summary["card"]; g = summary["gate"]
    lines = [f"# ProsQA audit: {summary['family']} ({summary['checkpoint']})", "",
             f"- accuracy {100 * g['accuracy']:.1f}% on {g['rows']} rows (published {100 * (g['published'] or 0):.1f}%); twins {summary['twins'].get('selected_pairs')} pairs; regime **{c['regime']}**; carriers {c['carriers']}; monitors {c['monitors']}", "",
             "| slot | transport short / long | K/V transport short / long | mask cost all / long (pts) |", "|---|---|---|---|"]
    f = lambda v: "-" if v is None or v != v else f"{v:.3f}"
    for s in SLOTS:
        e = c["slots"][str(s)]
        lines.append(f"| {s} | {f(e['transport_short'])} / {f(e['transport_long'])} | {f(e['kv_transport_short'])} / {f(e['kv_transport_long'])} | "
                     f"{'-' if e['mask_cost_all'] != e['mask_cost_all'] else f'{100 * e['mask_cost_all']:.1f}'} / {'-' if e['mask_cost_long'] is None else f'{100 * e['mask_cost_long']:.1f}'} |")
    t = c["tails"]
    lines += ["", f"- tails: kv_all transport all {f(t['kv_all_transport_all'])}, long {f(t['kv_all_transport_long'])}; mean_all cost {f(t['mean_all_cost_all'])}, meankv_all cost {f(t['meankv_all_cost_all'])}, mask_all cost {f(t['mask_all_cost_all'])}", ""]
    if summary.get("leakage"):
        lines += ["Readability of the prompt's names (balanced accuracy, logistic probe):", "", "| target | classes | " + " | ".join(f"slot {s}" for s in SLOTS) + " |", "|---|---|" + "---|" * 6]
        for k, e in summary["leakage"].items():
            lines.append(f"| {k} | {e['classes']} | " + " | ".join(f(e["slots"].get(str(s))) for s in SLOTS) + " |")
    lines += ["", "## Checks", ""] + [f"- {k}: **{v}**" for k, v in summary["checks"].items()]
    return "\n".join(lines)


# ---------------------------------------------------------------- model families
def load_family(args, device, dtype):
    if args.family == "coconut":
        from src.models.coconut import PROSQA_COCONUT_LLAMA1B, CoconutGPT2, coconut_generate_under, coconut_record
        spec = PROSQA_COCONUT_LLAMA1B
        model = CoconutGPT2.from_hub(spec, dtype=dtype, token=os.environ.get("HF_TOKEN") or None).to(device)
        def record(qs):
            return coconut_record(model, qs, batch_size=args.batch_size, device=device, max_new_tokens=MAX_NEW_TOKENS)
        def generate(qs, hidden=None, edits=(), donor=None, mask_slots=()):
            return coconut_generate_under(model, qs, batch_size=args.batch_size, device=device, hidden_donors=hidden, cache_edits=list(edits), donor=donor,
                                          mask_slots=mask_slots, max_new_tokens=MAX_NEW_TOKENS)
        return {"record": record, "generate": generate, "n_layers": model.n_layers(), "checkpoint": f"{spec.repo_id}@{spec.revision[:8]}/{spec.weight_file}",
                "can_hide": True, "readout_dim": model.hidden_size()}
    cfg = load_config(str(args.config)); token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision),
                                                                      filename=str(cfg.checkpoint.filename), expected_sha256=str(cfg.checkpoint.sha256), token=token,
                                                                      shard_sha256=checkpoint_shard_hashes(cfg))
    model, tokenizer = build_official_codi_gpt2(base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision), dtype=dtype, settings=cfg.model, token=token)
    load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    place_official_model(model, device, dtype).eval()
    latent = int(cfg.eval.latent_iterations)
    def record(qs):
        return record_native_trajectory(model, tokenizer, qs, latent_iterations=latent, batch_size=args.batch_size, device=device, max_new_tokens=MAX_NEW_TOKENS)
    def generate(qs, hidden=None, edits=(), donor=None, mask_slots=()):
        if mask_slots:
            raise ValueError("hiding thoughts is not available on the CODI native path; use the mean conditions")
        return generate_under(model, tokenizer, qs, latent_iterations=latent, batch_size=args.batch_size, device=device, hidden_donors=hidden,
                              cache_edits=list(edits), donor=donor, max_new_tokens=MAX_NEW_TOKENS)
    return {"record": record, "generate": generate, "n_layers": backbone_layers(model), "checkpoint": f"{cfg.checkpoint.repo_id}@{str(cfg.checkpoint.revision)[:8]}",
            "can_hide": False, "readout_dim": None}


def mean_trajectory(traj: Trajectory, n: int) -> Trajectory:
    """A donor whose every row is the pool mean (mean ablation through the swap machinery)."""
    return Trajectory(states=traj.states.mean(0, keepdim=True).expand(n, -1, -1).contiguous(), keys=traj.keys.mean(0, keepdim=True).expand(n, -1, -1, -1, -1).contiguous(),
                      values=traj.values.mean(0, keepdim=True).expand(n, -1, -1, -1, -1).contiguous())


def leakage_probes(states: torch.Tensor, rows: list[dict]) -> dict:
    from scripts.analyze_unfaithful_slot_content import ridge_balanced_accuracy, topk_classes
    targets = {"subject": [r["subject"] for r in rows], "option_1": [r["options"][0] for r in rows], "option_2": [r["options"][1] for r in rows],
               "gold_option": [r["gold_option"] for r in rows], "hops": [str(r["hops"]) for r in rows]}
    out = {}
    st = states.float().numpy()
    for name, vals in targets.items():
        y, keep = topk_classes(vals, k=40)
        entry = {"classes": len(keep), "slots": {}}
        for s in SLOTS:
            X = st[:, s]
            if X.shape[1] > 256:
                Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False); X = Xc @ Vt[:256].T
            entry["slots"][str(s)] = ridge_balanced_accuracy(X, y)
        out[name] = entry
    return out


# ---------------------------------------------------------------- run
def run(args) -> dict:
    smoke = bool(args.smoke); out = args.output_dir; out.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    from src.eval.official_codi import select_device
    from src.models.official_codi import resolve_torch_dtype
    device = select_device(args.device); dtype = resolve_torch_dtype(args.precision, device)
    fam = load_family(args, device, dtype)
    record, generate, n_layers = fam["record"], fam["generate"], fam["n_layers"]
    all_layers = tuple(range(n_layers)); stage = {}

    rows = load_prosqa(args.data)
    data_sha = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    for r in rows:
        o = options(r["question"]); r["subject"], r["options"], r["gold_option"], r["hops"] = o[0], list(o[1:]), gold_option(r), len(r["steps"])
    if smoke:
        rows = rows[:SMOKE["rows"]]
    # ---- native pass and gate
    t0 = time.perf_counter()
    traj, texts = record([r["question"] for r in rows])
    native = [answer_option(t, tuple(r["options"])) for t, r in zip(texts, rows)]
    correct = [n == r["gold_option"] for n, r in zip(native, rows)]
    accuracy = sum(correct) / len(correct)
    gate = {"accuracy": accuracy, "published": PUBLISHED[args.family], "rows": len(rows), "tolerance": GATE_TOLERANCE,
            "status": "passed" if smoke or abs(accuracy - PUBLISHED[args.family]) <= GATE_TOLERANCE else "failed"}
    print("gate", gate); stage["native"] = round(time.perf_counter() - t0, 1)
    hops = [r["hops"] for r in rows]; long_mask = [h > SHORT_MAX_HOPS for h in hops]
    # ---- twins
    t0 = time.perf_counter()
    pairs = twin_pairs(rows)[: SMOKE["pairs"] if smoke else args.max_pairs]
    flat = [x for o, t in pairs for x in (o, t)]
    ftexts = generate([x["question"] for x in flat])
    fok = [answer_option(t, tuple(x["options"])) == x["gold_option"] for t, x in zip(ftexts, flat)]
    selected = [p for i, p in enumerate(pairs) if fok[2 * i] and fok[2 * i + 1]]
    twins = {"candidates": len(pairs), "native_correct_both": len(selected), "selected_pairs": len(selected), "conditions": {}}
    if len(selected) >= 2:
        srows = [x for o, t in selected for x in (o, t)]
        partner = [i ^ 1 for i in range(len(srows))]
        straj, _ = record([x["question"] for x in srows])
        donor = donor_view(straj, partner)
        own = [x["gold_option"] for x in srows]; target = [own[j] for j in partner]; opts = [tuple(x["options"]) for x in srows]
        short = [x["hops"] <= SHORT_MAX_HOPS for x in srows]
        specs = {f"state_{s}": ({s: donor.states[:, s]}, []) for s in SLOTS} | {f"kv_{s}": (None, [CacheEdit(s, all_layers, True, True)]) for s in SLOTS}
        specs["state_all"] = ({s: donor.states[:, s] for s in SLOTS}, []); specs["kv_all"] = (None, [CacheEdit(s, all_layers, True, True) for s in SLOTS])
        per_row = {}
        for name, (hidden, edits) in specs.items():
            texts_c = generate([x["question"] for x in srows], hidden=hidden, edits=edits, donor=donor)
            outs = [outcome(t, o, w, g) for t, o, w, g in zip(texts_c, opts, own, target)]
            per_row[name] = outs
            twins["conditions"][name] = {"all": outcome_rates(outs), "short": outcome_rates(outs, short), "long": outcome_rates(outs, [not s for s in short])}
            print("twins", name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in twins["conditions"][name]["all"].items() if k != "target_ci95"})
        twins["hops_histogram"] = {str(h): sum(1 for x in srows if x["hops"] == h) for h in sorted({x["hops"] for x in srows})}
        with (out / "twins_pairs.jsonl").open("w") as fh:
            for i, (o, t) in enumerate(selected):
                fh.write(json.dumps({"question": o["question"], "answer": o["answer"], "twin_question": t["question"], "twin_answer": t["answer"], "hops": o["hops"],
                                     "options": o["options"], "outcomes": {k: [v[2 * i], v[2 * i + 1]] for k, v in per_row.items()}}) + "\n")
        _atomic_torch_save({"states": straj.states.half()}, out / "twins_states.pt")
    stage["twins"] = round(time.perf_counter() - t0, 1)
    # ---- masking (mean state, mean K/V, and for Coconut the hidden thought)
    t0 = time.perf_counter()
    qs = [r["question"] for r in rows]; mean = mean_trajectory(traj, len(rows)); masking = {}
    def mask_cond(name, hidden=None, edits=(), mask_slots=()):
        texts_c = generate(qs, hidden=hidden, edits=edits, donor=mean if edits else None, mask_slots=mask_slots)
        cond = [answer_option(t, tuple(r["options"])) == r["gold_option"] for t, r in zip(texts_c, rows)]
        masking[name] = {"all": drop(correct, cond), "short": drop(correct, cond, [not l for l in long_mask]), "long": drop(correct, cond, long_mask)}
        print("mask", name, round(masking[name]["all"]["drop"], 3))
    for s in SLOTS:
        mask_cond(f"mean_{s}", hidden={s: mean.states[:, s]}); mask_cond(f"meankv_{s}", edits=[CacheEdit(s, all_layers, True, True)])
        if fam["can_hide"]:
            mask_cond(f"mask_{s}", mask_slots=(s,))
    mask_cond("mean_all", hidden={s: mean.states[:, s] for s in SLOTS}); mask_cond("meankv_all", edits=[CacheEdit(s, all_layers, True, True) for s in SLOTS])
    if fam["can_hide"]:
        mask_cond("mask_all", mask_slots=SLOTS)
    stage["masking"] = round(time.perf_counter() - t0, 1)
    # ---- leakage probes on the native states
    t0 = time.perf_counter()
    leakage = leakage_probes(traj.states, rows) if len(rows) >= 50 else None
    stage["leakage"] = round(time.perf_counter() - t0, 1)
    _atomic_torch_save({"states": traj.states.half(), "correct": torch.tensor(correct), "hops": torch.tensor(hops)}, out / "native_states.pt")
    with (out / "rows.jsonl").open("w") as fh:
        for r, t, n, c in zip(rows, texts, native, correct):
            fh.write(json.dumps({"idx": r["idx"], "hops": r["hops"], "options": r["options"], "gold_option": r["gold_option"], "native_text": t, "native_option": n, "correct": c}) + "\n")
    card = summarize(twins=twins, masking=masking, family=args.family)
    checks = checks_from(card, leakage, gate)
    summary = {"contract": CONTRACT + ("_smoke" if smoke else ""), "schema_version": 1, "smoke": smoke, "family": args.family, "checkpoint": fam["checkpoint"],
               "data": {"path": str(args.data), "sha256": data_sha, "rows": len(rows), "hops_histogram": {str(h): hops.count(h) for h in sorted(set(hops))}},
               "gate": gate, "twins": twins, "masking": masking, "leakage": leakage, "card": card, "checks": checks, "stage_seconds": stage,
               "seconds": time.perf_counter() - started, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "warnings": ["Released checkpoints only; no training. Native greedy decoding under every intervention; the answer is the first option named."]}
    _atomic_json(summary, out / "summary.json")
    (out / "card.md").write_text(markdown(summary) + "\n")
    print(markdown(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("coconut", "codi"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/prosqa_codi_llama1b.yaml"), help="CODI config (ignored for Coconut)")
    parser.add_argument("--data", type=Path, required=True, help="prosqa_test.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precision", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
