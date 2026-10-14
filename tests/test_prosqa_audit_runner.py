"""The ProsQA audit runner on a tiny LLaMA-shaped Coconut and a synthetic ProsQA file (ledger §135)."""
import json
from pathlib import Path

import pytest
import torch

from scripts import run_prosqa_audit as R
from src.mech.cache_carrier import Trajectory


def test_outcome_rates_and_drop():
    outs = ["target", "retain", "other", "target"]
    r = R.outcome_rates(outs)
    assert r["target"] == 0.5 and r["retain"] == 0.25 and r["other"] == 0.25 and r["n"] == 4 and len(r["target_ci95"]) == 2
    assert R.outcome_rates(outs, [False] * 4)["n"] == 0
    d = R.drop([True, True, True, False], [True, False, False, False], samples=50)
    assert d["drop"] == pytest.approx(0.5) and d["n"] == 4 and d["ci95"][0] <= 0.5 <= d["ci95"][1]
    assert R.outcome("The answer is: Sally is a hilpus.", ("hilpus", "sterpus"), own="sterpus", target="hilpus") == "target"
    assert R.outcome("Sally is a sterpus.", ("hilpus", "sterpus"), own="sterpus", target="hilpus") == "retain"
    assert R.outcome("no idea", ("hilpus", "sterpus"), own="sterpus", target="hilpus") == "other"


def test_summary_regimes_and_checks():
    def cond(t):
        return {"all": {"target": t}, "short": {"target": t / 2}, "long": {"target": t}}
    twins = {"conditions": {**{f"state_{s}": cond(0.02) for s in R.SLOTS}, **{f"kv_{s}": cond(0.02) for s in R.SLOTS}, "kv_all": cond(0.1), "state_all": cond(0.05)}}
    masking = {**{f"mean_{s}": {"all": {"drop": 0.01}, "long": {"drop": 0.01}} for s in R.SLOTS}, **{f"meankv_{s}": {"all": {"drop": 0.0}, "long": {"drop": 0.0}} for s in R.SLOTS},
               "mean_all": {"all": {"drop": 0.02}}, "meankv_all": {"all": {"drop": 0.02}}}
    card = R.summarize(twins=twins, masking=masking, family="coconut")
    assert card["regime"] == "no_transport" and card["carriers"] == [] and card["max_long_transport"] == pytest.approx(0.02)
    checks = R.checks_from(card, {"subject": {"classes": 3, "slots": {str(s): 0.9 for s in R.SLOTS}}}, {"accuracy": 0.97, "published": 0.976})
    assert checks["p25a_no_single_slot_costs_over_2_points"] and checks["p25b_regime_no_transport"] and checks["p25c_names_readable_from_every_slot_0_80"] and checks["p25d_accuracy_within_3_points_of_published"]
    twins["conditions"]["state_3"] = cond(0.4); masking["mean_3"] = {"all": {"drop": 0.15}, "long": {"drop": 0.2}}
    card = R.summarize(twins=twins, masking=masking, family="codi")
    assert card["carriers"] == [3] and card["regime"] == "critical_path"


def _synthetic_rows(n=8):
    rows = []
    for i in range(n):
        a, b, c, d = f"a{i}pus", f"b{i}pus", f"c{i}pus", f"d{i}pus"
        rows.append({"question": f"Every {a} is a {b}. Every {b} is a {c}. Sally is a {a}. Every {b} is a {d}. Is Sally a {d}2 or {c}?".replace(f"{d}2", f"e{i}pus"),
                     "answer": f"Sally is a {c}.", "steps": [f"Sally is a {a}.", f"Every {a} is a {b}.", f"Every {b} is a {c}."]})
    return rows


def test_run_end_to_end_on_a_tiny_model(tmp_path, monkeypatch):
    from tests.test_coconut import _tiny_llama
    from src.models.coconut import coconut_generate_under, coconut_record
    m = _tiny_llama()
    data = tmp_path / "prosqa_test.json"; data.write_text(json.dumps(_synthetic_rows()))
    def fake_family(args, device, dtype):
        return {"record": lambda qs: coconut_record(m, qs, batch_size=4, device=device, max_new_tokens=4),
                "generate": lambda qs, hidden=None, edits=(), donor=None, mask_slots=(): coconut_generate_under(m, qs, batch_size=4, device=device, hidden_donors=hidden, cache_edits=list(edits), donor=donor, mask_slots=mask_slots, max_new_tokens=4),
                "n_layers": 2, "checkpoint": "tiny", "can_hide": True, "readout_dim": 32}
    monkeypatch.setattr(R, "load_family", fake_family)
    monkeypatch.setattr(R, "SMOKE", {"rows": 8, "pairs": 4})
    class A: pass
    args = A(); args.family = "coconut"; args.data = data; args.output_dir = tmp_path / "out"; args.batch_size = 4; args.precision = "float32"; args.device = "cpu"
    args.max_pairs = 10; args.smoke = True; args.config = None; args.checkpoint_path = None
    summary = R.run(args)
    assert summary["gate"]["rows"] == 8 and summary["data"]["rows"] == 8 and summary["twins"]["candidates"] == 4
    assert set(summary["masking"]) >= {"mean_0", "meankv_0", "mask_0", "mean_all", "meankv_all", "mask_all"}
    assert summary["card"]["regime"] in ("no_transport", "backup_route", "critical_path") and "p25a_no_single_slot_costs_over_2_points" in summary["checks"]
    assert (tmp_path / "out" / "summary.json").exists() and (tmp_path / "out" / "card.md").exists() and (tmp_path / "out" / "rows.jsonl").exists()
    text = R.markdown(summary)
    assert "ProsQA audit: coconut" in text and "| 5 |" in text
