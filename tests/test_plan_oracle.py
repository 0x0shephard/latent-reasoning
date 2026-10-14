import json

import pytest

from scripts.run_codi_plan_oracle import ARMS, MODES, checks_from, failure_class, hint, markdown, plan_of, summarize, wrong_plan


def test_plan_is_the_first_operator_of_each_gold_equation():
    assert plan_of("<<16-3-4=9>> <<9*2=18>> <<18/3=6>>") == ["-", "*", "/"]
    assert plan_of("<<100*.4=40>> <<40+2=42>>") == ["*", "+"]


def test_wrong_plan_differs_everywhere_and_keeps_the_length():
    plan = ["+", "-", "*", "/"]
    wrong = wrong_plan(plan)
    assert len(wrong) == 4 and all(a != b for a, b in zip(plan, wrong))


def test_hints():
    q = "Tom has 3 apples."
    assert hint(q, "none", ["+", "*"]) == q
    assert hint(q, "length", ["+", "*"]) == "Tom has 3 apples. Hint: solve in 2 steps."
    assert hint(q, "gold", ["+", "*"]) == "Tom has 3 apples. Hint: solve in 2 steps: add, then multiply."
    assert hint(q, "wrong", ["+", "*"]) == "Tom has 3 apples. Hint: solve in 2 steps: multiply, then subtract."


def test_failure_class_from_the_explicit_chain():
    # a CODI chain stops at the penultimate value v*; the final step is done at the answer position
    assert failure_class("<<2+3=5>>The answer is: 10", gold=10.0, gold_v_star="5") == "latent_only"
    assert failure_class("<<2+3=6>>The answer is: 12", gold=10.0, gold_v_star="5") == "arithmetic"   # right plan, slip
    assert failure_class("<<2+4=6>>The answer is: 12", gold=10.0, gold_v_star="5") == "plan_wrong"
    assert failure_class("no equations here", gold=10.0, gold_v_star="5") == "no_chain"


def _row(cls, latent_none, latent_gold, explicit_gold, explicit_length=False, latent_wrong=False):
    correct = {m: {a: False for a in ARMS} for m in MODES}
    correct["latent"]["none"] = latent_none; correct["latent"]["gold"] = latent_gold; correct["latent"]["wrong"] = latent_wrong
    correct["explicit"]["gold"] = explicit_gold; correct["explicit"]["length"] = explicit_length
    return {"failure_class": cls, "correct": correct}


def test_summary_and_checks():
    rows = [_row("plan_wrong", False, False, True, explicit_length=True)] * 8 + [_row("plan_wrong", False, True, True)] * 2 \
        + [_row("latent_only", False, True, True)] * 5 + [_row("none", True, True, True)] * 5
    report = summarize(rows)
    assert report["latent_failures"] == 15 and report["class_counts"] == {"latent_only": 5, "plan_wrong": 10}
    lat = report["arms"]["latent/gold"]
    assert lat["recovery"]["plan_wrong"]["rate"] == pytest.approx(0.2) and lat["recovery"]["latent_only"]["rate"] == 1.0
    assert lat["harm"]["rate"] == 0.0 and lat["net_points_vs_latent_none"] == pytest.approx(100 * (12 / 20 - 5 / 20))
    checks = checks_from(report)
    assert checks["p18a_latent_gold_plan_wrong_recovery_le_0_15"] is False and checks["p18a_latent_wrong_control_le_0_05"] is True
    assert checks["p18b_explicit_gold_ge_0_25_and_above_latent"] is True and checks["p18c_length_hint_ge_half_of_gold_explicit"] is True
    assert checks["p18d_latent_gold_net_le_3"] is False and checks["decision_build_plan_channel"] is False
    text = markdown(report)
    assert "| latent/gold |" in text and "20.0 (10)" in text
    json.dumps(report)
