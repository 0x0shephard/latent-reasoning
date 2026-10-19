import numpy as np
import pytest

from scripts.analyze_cascade_two_stage_gate import checks_from, rank_average_gate, ranks, two_stage_gate


def test_ranks_average_ties_and_keep_nan():
    r = ranks(np.array([3.0, 1.0, 2.0, np.nan, 2.0]))
    assert np.isnan(r[3]) and r[1] == 0.0 and r[0] == 1.0 and r[2] == r[4] == pytest.approx(0.5)


def test_rank_average_routes_the_unsure_and_the_inconsistent_first():
    prob = np.array([0.9, 0.1, 0.9, 0.5]); risk = np.array([0.0, 5.0, 2.0, 1.0])
    g = rank_average_gate(prob, risk)
    assert np.argmin(g) == 1 and np.argmax(g) == 0             # unsure and inconsistent first; confident and consistent last
    assert g[3] < g[2]                                          # middling on both beats top on one and bottom on the other


def test_two_stage_gate_orders_stage_one_by_confidence_then_the_rest_by_risk():
    prob = np.array([0.05, 0.10, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, np.nan])
    risk = np.array([0.0, 0.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0])
    pool = ~np.isnan(prob)
    g = two_stage_gate(prob, risk, 0.2, pool)
    order = [i for i in np.argsort(np.where(np.isnan(g), -np.inf, g), kind="stable")]
    assert order[0] == 10                                       # NaN confidence first, as in the scorer
    assert order[1:3] == [0, 1]                                 # the least confident fifth, in confidence order
    assert order[3] == 2 and order[-1] == 3                     # then the most inconsistent, down to the most consistent
    # q = 0 is the risk ordering, q = 1 is the confidence ordering
    none = two_stage_gate(prob, risk, 0.0, pool)
    assert np.argmin(np.nan_to_num(none, nan=np.inf)) == 2 and np.nanargmax(none) in (0, 1)   # pure risk order (risk-0 rows tie last)
    full = two_stage_gate(prob, risk, 1.0, pool)
    assert np.all(np.diff(full[pool][np.argsort(prob[pool])]) > 0)


def test_checks_score_the_predictions():
    def model(ts25, best25, best50):
        return {"gates": {"confidence": {"vs_confidence": {"0.25": 0.0, "0.5": 0.0}}, "residual_only": {"vs_confidence": {"0.25": -5.0, "0.5": -5.0}},
                          "two_stage_residual_q10": {"vs_confidence": {"0.25": ts25, "0.5": 0.0}}, "conf_rank_residual": {"vs_confidence": {"0.25": best25, "0.5": best50}}}}
    c = checks_from({"gpt2": model(-0.5, 0.2, 0.3), "simcot": model(0.1, 1.4, 0.2), "codi8b": model(-1.5, 0.0, 0.0)})
    assert c["p30a_no_gate_gains_over_1pt_at_25pct_on_3_models"] and c["models_gaining_1pt_at_25pct"] == ["simcot"]
    assert c["p30c_simcot_among_winners_if_any"] and not c["p30b_two_stage_residual_q10_within_minus_1pt_at_25pct_everywhere"] and c["p30d_no_gate_gains_over_1pt_at_50pct_on_3_models"]
