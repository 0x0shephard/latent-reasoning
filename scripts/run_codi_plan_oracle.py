"""The plan oracle (ledger §132): what can a CODI checkpoint do with a plan it did not produce?  No training.

For every >=3-step GSM8K test problem the gold chain's operator sequence is appended to the
question as a hint (arms: none / gold / wrong / length), and the model answers in latent mode
(six thoughts, native decoding) and in explicit mode (its own written chain).  Recovery of each
class of latent failure under each arm is the ceiling a trained plan thought could reach.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_kv_subspaces import _atomic_json
from scripts.run_codi_latent_repair import classify_failure, explicit_chains, penultimate_value, rate, recompute_chain
from src.data.chain_pools import gsm8k_chain_row, load_raw_eval_rows
from src.data.counterfactual_chain import parse_chain
from src.eval.official_codi import select_device
from src.eval.official_codi_gate import extract_official_answer_number
from src.mech.cache_carrier import record_native_trajectory
from src.models.official_codi import (build_official_codi_gpt2, checkpoint_shard_hashes, download_official_checkpoint,
                                      load_official_checkpoint, resolve_torch_dtype)
from src.utils.config import load_config

CONTRACT = "official_codi_plan_oracle_v1"
ARMS = ("none", "gold", "wrong", "length")
MODES = ("latent", "explicit")
MIN_STEPS = 3
SMOKE = {"rows": 48}
WORD = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
CYCLE = {"+": "*", "*": "-", "-": "/", "/": "+"}
_OP = re.compile(r"[+\-*/]")


def plan_of(cot: str) -> list[str]:
    """The first operator of each gold equation."""
    ops = []
    for eq in parse_chain(cot):
        found = _OP.findall(re.sub(r"\d[\d.,]*", "", eq["left"]))
        ops.append(found[0] if found else "+")
    return ops


def wrong_plan(plan: list[str]) -> list[str]:
    """Same length, every operator different from the gold one (a fixed cycle)."""
    return [CYCLE[op] for op in plan]


def hint(question: str, arm: str, plan: list[str]) -> str:
    k = len(plan)
    if arm == "none":
        return question
    if arm == "length":
        return f"{question} Hint: solve in {k} steps."
    ops = plan if arm == "gold" else wrong_plan(plan)
    return f"{question} Hint: solve in {k} steps: " + ", then ".join(WORD[o] for o in ops) + "."


def failure_class(explicit_chain: str, gold: float, gold_v_star: str | None) -> str:
    out = recompute_chain(explicit_chain)
    cls = classify_failure(explicit_answer=extract_official_answer_number(explicit_chain), gold=gold,
                           corrected=out[1] if out else None, gold_v_star=gold_v_star, carrier_read="")
    return "arithmetic" if cls in ("final_step", "upstream_arithmetic") else cls


def summarize(rows: list[dict]) -> dict:
    """Accuracy, recovery per latent-failure class, harm and net points for every arm and mode."""
    base_wrong = [r for r in rows if not r["correct"]["latent"]["none"]]
    base_right = [r for r in rows if r["correct"]["latent"]["none"]]
    classes = sorted({r["failure_class"] for r in base_wrong})
    report = {"rows": len(rows), "latent_failures": len(base_wrong), "class_counts": {c: sum(r["failure_class"] == c for r in base_wrong) for c in classes},
              "arms": {}}
    for mode in MODES:
        for arm in ARMS:
            acc = sum(r["correct"][mode][arm] for r in rows) / max(1, len(rows))
            entry = {"accuracy": acc, "net_points_vs_latent_none": 100 * (acc - sum(r["correct"]["latent"]["none"] for r in rows) / max(1, len(rows))),
                     "recovery": {c: rate([r["correct"][mode][arm] for r in base_wrong if r["failure_class"] == c]) for c in classes},
                     "recovery_all": rate([r["correct"][mode][arm] for r in base_wrong]),
                     "harm": rate([not r["correct"][mode][arm] for r in base_right])}
            report["arms"][f"{mode}/{arm}"] = entry
    return report


def checks_from(report: dict) -> dict:
    a = report["arms"]
    pw = lambda key: a[key]["recovery"].get("plan_wrong", {}).get("rate")
    g_lat, w_lat, g_exp, l_exp = pw("latent/gold"), pw("latent/wrong"), pw("explicit/gold"), pw("explicit/length")
    net = a["latent/gold"]["net_points_vs_latent_none"]
    return {"p18a_latent_gold_plan_wrong_recovery_le_0_15": None if g_lat is None else g_lat <= 0.15,
            "p18a_latent_wrong_control_le_0_05": None if w_lat is None else w_lat <= 0.05,
            "p18b_explicit_gold_ge_0_25_and_above_latent": None if g_exp is None or g_lat is None else (g_exp >= 0.25 and g_exp > g_lat),
            "p18c_length_hint_ge_half_of_gold_explicit": None if l_exp is None or g_exp is None else l_exp >= 0.5 * g_exp,
            "p18d_latent_gold_net_le_3": net <= 3.0,
            "decision_build_plan_channel": None if g_lat is None else (g_lat >= 0.30 and net >= 5.0)}


def markdown(report: dict) -> str:
    lines = [f"rows {report['rows']}, latent failures {report['latent_failures']}, classes {report['class_counts']}", "",
             "| mode/arm | accuracy | net pts | recovery: plan_wrong | latent_only | arithmetic | all failures | harm |", "|---|---|---|---|---|---|---|---|"]
    for key, e in report["arms"].items():
        rec = e["recovery"]
        f = lambda c: f"{100 * rec[c]['rate']:.1f} ({rec[c]['n']})" if c in rec else "-"
        lines.append(f"| {key} | {100 * e['accuracy']:.1f} | {e['net_points_vs_latent_none']:+.1f} | {f('plan_wrong')} | {f('latent_only')} | {f('arithmetic')} | "
                     f"{100 * e['recovery_all']['rate']:.1f} | {100 * e['harm']['rate']:.1f} |")
    return "\n".join(lines)


def run(args) -> dict:
    smoke = bool(args.smoke); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cfg = load_config(args.config); device = select_device(args.device); dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id), revision=str(cfg.checkpoint.revision), filename=str(cfg.checkpoint.filename),
        expected_sha256=str(cfg.checkpoint.sha256), token=token, shard_sha256=checkpoint_shard_hashes(cfg))
    model, tokenizer = build_official_codi_gpt2(base_model=str(cfg.model.base_model), base_revision=str(cfg.model.base_revision),
                                                dtype=dtype, settings=cfg.model, token=token)
    load_official_checkpoint(model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256))
    model.to(device=device, dtype=dtype).eval()
    latent_positions = int(cfg.eval.latent_iterations)

    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    rows = []
    for r in load_raw_eval_rows(dict(data_cfg.eval.gsm8k)):
        c = gsm8k_chain_row(r)
        if c and len(parse_chain(c["cot"])) >= MIN_STEPS:
            rows.append({"question": c["question"], "cot": c["cot"], "gold": float(c["answer"]), "plan": plan_of(c["cot"]),
                         "gold_v_star": penultimate_value(c["cot"]), "steps": len(parse_chain(c["cot"]))})
    if smoke:
        rows = rows[:SMOKE["rows"]]
    print(f"{len(rows)} problems with >= {MIN_STEPS} steps")
    answers = {m: {} for m in MODES}; chains = {}
    for arm in ARMS:
        prompts = [hint(r["question"], arm, r["plan"]) for r in rows]
        _, outputs = record_native_trajectory(model, tokenizer, prompts, latent_iterations=latent_positions, batch_size=args.batch_size, device=device)
        answers["latent"][arm] = [extract_official_answer_number(o) for o in outputs]
        chains[arm] = explicit_chains(model, tokenizer, prompts, batch_size=args.batch_size, device=device)
        answers["explicit"][arm] = [extract_official_answer_number(c) for c in chains[arm]]
        acc = {m: sum(a is not None and abs(a - r["gold"]) < 1e-6 for a, r in zip(answers[m][arm], rows)) / len(rows) for m in MODES}
        print(f"arm {arm}: latent {acc['latent']:.3f}  explicit {acc['explicit']:.3f}")
    records = []
    for i, r in enumerate(rows):
        correct = {m: {arm: answers[m][arm][i] is not None and abs(answers[m][arm][i] - r["gold"]) < 1e-6 for arm in ARMS} for m in MODES}
        records.append({**{k: r[k] for k in ("question", "gold", "plan", "steps", "gold_v_star")}, "correct": correct,
                        "answers": {m: {arm: answers[m][arm][i] for arm in ARMS} for m in MODES},
                        "explicit_chain_none": chains["none"][i], "failure_class": failure_class(chains["none"][i], r["gold"], r["gold_v_star"])})
    report = summarize(records); checks = checks_from(report)
    summary = {"contract": CONTRACT, "schema_version": 1, "config": str(args.config), "checkpoint_sha256": str(cfg.checkpoint.sha256),
               "smoke": smoke, "arms": list(ARMS), "modes": list(MODES), "min_steps": MIN_STEPS, "report": report, "checks": checks,
               "seconds": time.perf_counter() - started, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "warnings": ["Official checkpoint only; no training. Hints are appended to the question, a prompt the checkpoint was not trained on; the wrong and length arms control for that."]}
    write_outputs(out, records, summary)
    print(markdown(report)); print(json.dumps(checks, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_llama1b.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def write_outputs(out: Path, records: list[dict], summary: dict) -> None:
    """The run's files: one JSON line per problem and the atomic summary (the call that failed on Kaggle once, argument order)."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    _atomic_json(summary, out / "summary.json")
