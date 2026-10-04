import pytest

pytest.importorskip("torch")

from scripts.run_codi_cache_carrier_audit import (  # noqa: E402
    A1_MIN_FEEDING,
    A2_MIN_TERMINAL_KV,
    A3_DOMINANCE,
    A6_MIN_OVER_NULL,
    A6_MIN_TOWARD_DONOR,
    CONTRACT,
    LAYER_GROUPS,
    SLOT_CONDITIONS,
    TAIL_CONDITIONS,
    checks_from,
    claim_from,
    condition_spec,
)


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_cache_carrier_audit_v1"
    assert SLOT_CONDITIONS == ("hidden", "k", "v", "kv", "v_layers_0_7", "v_layers_8_9", "v_layers_10_11", "hidden_kv")
    assert TAIL_CONDITIONS == ("hidden_all", "k_all", "v_all", "kv_all", "v_all_8_9", "v_all_10_11", "hidden_kv_all")
    assert LAYER_GROUPS == {"0_7": tuple(range(8)), "8_9": (8, 9), "10_11": (10, 11)}
    assert (A1_MIN_FEEDING, A2_MIN_TERMINAL_KV, A3_DOMINANCE, A6_MIN_TOWARD_DONOR, A6_MIN_OVER_NULL) == (0.25, 0.10, 1.5, 0.50, 0.30)


def test_condition_spec():
    hidden, edits = condition_spec("hidden", (3,))
    assert hidden == (3,) and edits == []
    hidden, edits = condition_spec("v_layers_8_9", (1,))
    assert hidden == () and len(edits) == 1 and edits[0].layers == (8, 9) and edits[0].values and not edits[0].keys
    hidden, edits = condition_spec("k", (5,))
    assert edits[0].keys and not edits[0].values and edits[0].layers == tuple(range(12))
    hidden, edits = condition_spec("hidden_kv_all", tuple(range(6)))
    assert hidden == tuple(range(6)) and len(edits) == 6 and all(e.keys and e.values for e in edits)
    hidden, edits = condition_spec("v_all_10_11", tuple(range(6)))
    assert hidden == () and all(e.layers == (10, 11) and e.values for e in edits)
    with pytest.raises(ValueError):
        condition_spec("nonsense", (1,))


def _report(hidden=(0.14, 0.31, 0.17, 0.37, 0.06, 0.0), kv=(0.1, 0.2, 0.1, 0.25, 0.05, 0.15), v_tail=0.5, k_tail=0.2,
            v89=0.4, v1011=0.1, toward=0.7, null=0.05):
    def outcome(rate, t=None, n=None):
        return {"change_rate": rate, "accuracy": 0.8, "toward_donor_among_changed": t, "donor_null": n, "changed": 10}
    slots = {}
    for s in range(6):
        slots[str(s)] = {"hidden": outcome(hidden[s]), "k": outcome(kv[s] * 0.4), "v": outcome(kv[s] * 0.8), "kv": outcome(kv[s]),
                         "v_layers_0_7": outcome(kv[s] * 0.3), "v_layers_8_9": outcome(kv[s] * 0.6), "v_layers_10_11": outcome(kv[s] * 0.1),
                         "hidden_kv": outcome(max(hidden[s], kv[s]) + 0.05)}
    tail = {"hidden_all": outcome(0.6), "k_all": outcome(k_tail), "v_all": outcome(v_tail), "kv_all": outcome(0.6),
            "v_all_8_9": outcome(v89), "v_all_10_11": outcome(v1011), "hidden_kv_all": outcome(0.9, toward, null)}
    return {"slots": slots, "tail": tail}


def test_checks_and_claim():
    c = checks_from(_report())
    assert c["a1_replication"] and c["a2_terminal_slot_is_a_memory"] and c["a4_values_over_keys"]
    assert c["a5_layers_8_9_over_10_11"] and c["a6_tail_transplant_donor_directed"] and c["sanity_both_routes_at_least_each"]
    assert c["a3_route_split"] == {"1": "state_dominant", "3": "shared"}
    text = claim_from(c)
    assert "a memory" in text and "slot 1 state-dominant" in text and "slot 3 shared" in text
    c = checks_from(_report(kv=(0.1, 0.5, 0.1, 0.6, 0.05, 0.0)))
    assert c["a3_route_split"] == {"1": "cache_dominant", "3": "cache_dominant"} and not c["a2_terminal_slot_is_a_memory"]
    assert "inert through both routes" in claim_from(c)
    c = checks_from(_report(hidden=(0.14, 0.2, 0.17, 0.37, 0.06, 0.0)))
    assert not c["a1_replication"]
    c = checks_from(_report(toward=0.2))
    assert not c["a6_tail_transplant_donor_directed"]
    c = checks_from(_report(v89=0.05, v1011=0.1, v_tail=0.1, k_tail=0.2))
    assert not c["a5_layers_8_9_over_10_11"]
