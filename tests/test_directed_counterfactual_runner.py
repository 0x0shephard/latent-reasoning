import pytest

pytest.importorskip("torch")

from scripts.run_codi_directed_counterfactual import (  # noqa: E402
    CANDIDATE_LIMIT_LONG,
    CONTRACT,
    CONTRACT_LONG,
    E1_MIN_TAIL_KV,
    E2_MIN_STATE,
    E3_MIN_SITE,
    E3_MIN_UNIQUE,
    EXTRA_ROWS,
    MIN_STEPS_LONG,
    PROFILES,
    D1_MIN_TAIL,
    D2_MIN_EVEN,
    D3_MIN_LOCATED,
    D3_MIN_SITE,
    NATIVE_CONDITIONS,
    PAIRS,
    SINGLE_CONDITIONS,
    TAIL_CONDITIONS,
    checks_from,
    checks_long_from,
    claim_from,
    claim_long_from,
    condition_spec,
)


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_directed_counterfactual_v1"
    assert PAIRS == 512 and len(SINGLE_CONDITIONS) == 21 and TAIL_CONDITIONS == ("v89_all", "v89_even", "v89_odd", "kv_all", "state_all")
    assert NATIVE_CONDITIONS == ("v89_all", "kv_all", "state_1", "state_3", "v89_2", "v89_4")
    assert (D1_MIN_TAIL, D2_MIN_EVEN, D3_MIN_SITE, D3_MIN_LOCATED) == (0.5, 0.4, 0.3, 20)


def test_condition_spec():
    assert condition_spec("state_3") == ((3,), [])
    hidden, edits = condition_spec("v89_2")
    assert hidden == () and edits[0].slot == 2 and edits[0].layers == (8, 9) and edits[0].values and not edits[0].keys
    hidden, edits = condition_spec("v1011_4")
    assert edits[0].layers == (10, 11)
    hidden, edits = condition_spec("kv_5")
    assert edits[0].layers == tuple(range(12)) and edits[0].keys and edits[0].values
    assert condition_spec("state_all") == ((0, 1, 2, 3, 4, 5), [])
    assert [e.slot for e in condition_spec("v89_even")[1]] == [0, 2, 4]
    assert [e.slot for e in condition_spec("v89_odd")[1]] == [1, 3, 5]
    assert len(condition_spec("kv_all")[1]) == 6 and len(condition_spec("v89_all")[1]) == 6
    with pytest.raises(ValueError):
        condition_spec("v89_7")
    with pytest.raises(ValueError):
        condition_spec("nonsense_1")


def _report(tail=0.7, even=0.5, odd=0.1, singles=None, v1011=0.05, site=0.5, elsewhere=0.1, store=0.4, store_else=0.1, n=40):
    singles = singles or {0: 0.3, 2: 0.25, 4: 0.3}
    def o(t):
        return {"target": t, "retain": 1 - t - 0.1, "other": 0.1, "n": 100}
    conds = {f"v89_{s}": {"all": o(singles.get(s, 0.05))} for s in range(6)}
    conds |= {f"v1011_{s}": {"all": o(v1011)} for s in (0, 2, 4)}
    conds |= {"v89_all": {"all": o(tail)}, "v89_even": {"all": o(even)}, "v89_odd": {"all": o(odd)}}
    spec = {str(s): {"n": n, "state_at_site": o(site), "state_elsewhere": {str(3 if s == 1 else 1): o(elsewhere)},
                     "store_at_site": o(store), "store_elsewhere": {str(4 if s == 1 else 2): o(store_else)}} for s in (1, 3)}
    return {"conditions": conds, "located_specificity": spec}


def test_checks_and_claim():
    c = checks_from(_report())
    assert c["d1_tail_replication"] and c["d2_even_store"] and c["d3_located_specificity"] and c["d4_layer_control"]
    assert c["d3_per_site"] == {"1": True, "3": True}
    assert "the thought that decodes the value" in claim_from(c)
    c = checks_from(_report(tail=0.3))
    assert not c["d1_tail_replication"] and "does NOT carry" in claim_from(c)
    c = checks_from(_report(site=0.2))
    assert not c["d3_located_specificity"]
    c = checks_from(_report(n=5))
    assert c["d3_per_site"] == {"1": None, "3": None} and not c["d3_located_specificity"]
    assert "no site had enough" in claim_from(c)
    c = checks_from(_report(v1011=0.3))
    assert not c["d4_layer_control"]
    c = checks_from(_report(odd=0.4))
    assert not c["d2_even_store"]


def test_long_profile_is_frozen_and_standard_unchanged():
    assert CONTRACT_LONG == "official_codi_directed_counterfactual_long_v1"
    assert (EXTRA_ROWS, CANDIDATE_LIMIT_LONG, MIN_STEPS_LONG) == (14_336, 1_536, 3)
    assert (E1_MIN_TAIL_KV, E2_MIN_STATE, E3_MIN_SITE, E3_MIN_UNIQUE) == (0.5, 0.25, 0.3, 20)
    assert PROFILES["standard"] == {"contract": CONTRACT, "min_steps": 1, "extra_rows": 0, "prefer_latest_step": False,
                                    "candidate_limit": 1_024}
    assert PROFILES["long"] == {"contract": CONTRACT_LONG, "min_steps": 3, "extra_rows": 14_336, "prefer_latest_step": True,
                                "candidate_limit": 1_536}


def _long_report(tail_kv=0.6, state1=0.3, state3=0.2, site=0.5, elsewhere=0.1, kv_site=0.4, kv_else=0.1, n=40):
    def o(t):
        return {"target": t, "retain": 1 - t - 0.1, "other": 0.1, "n": 100}
    conds = {"kv_all": {"all": o(tail_kv)}, "state_1": {"all": o(state1)}, "state_3": {"all": o(state3)}}
    spec = {str(s): {"n": n, "state_at_site": o(site), "state_elsewhere": {str(3 if s == 1 else 1): o(elsewhere)},
                     "store_at_site": o(0.1), "store_elsewhere": {str(4 if s == 1 else 2): o(0.1)},
                     "kv_store_at_site": o(kv_site), "kv_store_elsewhere": {str(4 if s == 1 else 2): o(kv_else)}} for s in (1, 3)}
    return {"conditions": conds, "unique_specificity": spec}


def test_long_checks_and_claim():
    c = checks_long_from(_long_report())
    assert c["e1_long_chain_tail"] and c["e2_single_thought_transport"] and c["e3_unique_location_specificity"]
    assert "carries it" in claim_long_from(c)
    c = checks_long_from(_long_report(state1=0.1, state3=0.2))
    assert not c["e2_single_thought_transport"] and "noise" in claim_long_from(c)
    c = checks_long_from(_long_report(kv_site=0.15))
    assert not c["e3_unique_location_specificity"]
    c = checks_long_from(_long_report(n=10))
    assert c["e3_per_site"] == {"1": None, "3": None} and "no site had enough" in claim_long_from(c)
    c = checks_long_from(_long_report(tail_kv=0.4))
    assert not c["e1_long_chain_tail"]
