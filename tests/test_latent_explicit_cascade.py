import json
import math

import numpy as np
import pytest

from scripts.analyze_latent_explicit_cascade import (analyze, at_fraction, curve, disagreement_reference, explicit_answer,
                                                     fraction_to_capture, gain_of, join_rows, markdown)


def _readout(n, prob, native):
    return {"first_token_prob": np.array(prob, dtype=float), "first_token_margin": np.array(prob, dtype=float), "native": np.array(native, dtype=float)}


def test_explicit_answer_parses_the_cue_and_commas():
    assert explicit_answer("<<1+1=2>>The answer is: 1,250") == 1250.0
    assert explicit_answer("<<1+1=2>>") is None and explicit_answer(None) is None


def test_gain_counts_a_right_chain_on_a_wrong_row_and_a_wrong_chain_on_a_control():
    assert gain_of({"control": False, "gold": 5.0, "explicit": 5.0}) == 1.0
    assert gain_of({"control": False, "gold": 5.0, "explicit": 4.0}) == 0.0
    assert gain_of({"control": True, "gold": 5.0, "explicit": 4.0}) == -1.0
    assert gain_of({"control": True, "gold": 5.0, "explicit": None}) == 0.0


def test_curve_routes_the_least_confident_first_and_scales_weights():
    rows = [{"control": False, "gold": 1.0, "native": 0.0, "explicit": 1.0, "prob": 0.2, "margin": 0.2, "weight": 2.0},
            {"control": True, "gold": 1.0, "native": 1.0, "explicit": 0.0, "prob": 0.9, "margin": 0.9, "weight": 3.0},
            {"control": False, "gold": 1.0, "native": 0.0, "explicit": None, "prob": float("nan"), "margin": float("nan"), "weight": 2.0}]
    f, p = curve(rows, "prob", eligible=7)
    assert np.allclose(f, [0, 2 / 7, 4 / 7, 1.0])           # NaN confidence is routed first
    assert np.allclose(p, [0, 0, 100 * 2 / 7, 100 * (2 - 3) / 7])
    assert at_fraction(f, p, 0.5) == pytest.approx(0.0) and at_fraction(f, p, 4 / 7) == pytest.approx(100 * 2 / 7)
    assert fraction_to_capture(f, p, 0.5) is None            # the full switch is negative here


def test_join_weights_the_sample_to_the_population_and_flags_native_mismatch():
    questions = ["q0", "q1", "q2", "q3"]
    readout = _readout(4, [0.1, 0.5, 0.9, 0.7], [3.0, 7.0, 9.0, 2.0])
    targets = [{"question": "q0", "control": False, "gold": 4.0, "native": 3.0, "tool_chain": "The answer is: 4"},
               {"question": "q2", "control": True, "gold": 9.0, "native": 9.0, "tool_chain": "The answer is: 9"},
               {"question": "q3", "control": False, "gold": 2.5, "native": 1.0, "tool_chain": "The answer is: 2"}]
    rows, mismatched = join_rows(questions, readout, targets, wrong_total=4, eligible=10)
    assert [r["weight"] for r in rows] == [2.0, 6.0, 2.0] and mismatched == 1
    with pytest.raises(KeyError):
        join_rows(questions, readout, [{"question": "missing", "control": False, "gold": 1.0, "native": 1.0}], wrong_total=1, eligible=2)


def test_analyze_reports_oracle_disagreement_and_both_gates():
    questions = [f"q{i}" for i in range(6)]
    readout = _readout(6, [0.1, 0.2, 0.95, 0.9, 0.3, 0.8], [0, 0, 1, 1, 0, 1])
    targets = [{"question": "q0", "control": False, "gold": 1.0, "native": 0.0, "tool_chain": "The answer is: 1"},
               {"question": "q1", "control": False, "gold": 1.0, "native": 0.0, "tool_chain": "The answer is: 0"},
               {"question": "q4", "control": False, "gold": 1.0, "native": 0.0, "tool_chain": "The answer is: 1"},
               {"question": "q2", "control": True, "gold": 1.0, "native": 1.0, "tool_chain": "The answer is: 1"},
               {"question": "q3", "control": True, "gold": 1.0, "native": 1.0, "tool_chain": "The answer is: 0"}]
    summary = {"report": {"targets": {"eligible": 10, "wrong_total": 3}}}
    report = analyze(questions, readout, targets, summary, resamples=20)
    assert report["rows"] == 5 and report["oracle"]["net_points"] == pytest.approx(20.0)
    assert report["gates"]["prob"]["full_switch"] == pytest.approx(100 * (2 - 3.5) / 10)
    assert report["gates"]["prob"]["at_fraction"]["0.25"] == pytest.approx(10.0)   # q0 then q1 routed (weights 1 each); q4 would exceed 25%
    assert report["disagreement_trigger"]["net_points"] == pytest.approx(100 * (2 - 3.5) / 10)
    text = markdown(report, "toy")
    assert "Cascade toy" in text and "oracle" in text and "| 25% |" in text
    json.dumps(report)
