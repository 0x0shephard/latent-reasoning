import json

import pytest

from src.mech.monitor_audit import (
    CARD_CONTRACT,
    CARRIER_MIN_TRANSPORT,
    SLOTS,
    build_card,
    by_profile,
    card_markdown,
    carrying_slots,
    compare_cards,
    comparison_markdown,
    environment_fingerprint,
    short_target_of,
    slot_role,
    spearman,
)


def _twins(state: dict, kv: dict | None = None, *, strata=None, kv_all=0.6, other_all=0.3):
    kv = kv or {}
    conds = {}
    for s in SLOTS:
        for kind, table in (("state", state), ("kv", kv)):
            t = table.get(s, 0.0)
            conds[f"{kind}_{s}"] = {"all": {"target": t, "retain": 1 - t - 0.05, "other": 0.05, "n": 1024},
                                   "by_stratum": strata or {"n1_k1": {"target": t, "n": 200}, "n2_k1": {"target": t, "n": 300}, "n3_k2": {"target": t, "n": 500}}}
    conds["kv_all"] = {"all": {"target": kv_all, "retain": 0.1, "other": other_all, "n": 1024}, "by_stratum": {}}
    conds["state_all"] = {"all": {"target": 0.3, "retain": 0.5, "other": 0.2, "n": 1024}, "by_stratum": {}}
    return {"contract": "official_codi_directed_counterfactual_v1", "checkpoint_sha256": "abc", "gate": {"selected_pairs": 512},
            "baseline": {"first_token_accuracy": 0.9, "rows": 1024}, "strata_counts": {"n3_k2": 1024},
            "preregistration": {"backbone_layers": 12, "sampling_hash_fit": "h" * 64},
            "pool": {"pool": "gsm8k_aug_train", "hash": "h" * 64, "n": 2048, "steps_histogram": {"1": 500, "2": 800, "3": 748}},
            "report": {"conditions": conds}, "seconds": 400.0}


def _masking(long_costs: dict, short_costs: dict | None = None, *, terminal=0.0, register_long=0.3):
    short_costs = short_costs or {}

    def part(costs):
        out = {}
        for s in SLOTS:
            out[f"mean_{s}"] = {"accuracy": 0.8, "accuracy_drop": costs.get(s, 0.0), "change_rate": 0.1, "n": 256}
            out[f"mask_{s}"] = {"accuracy": 0.8, "accuracy_drop": costs.get(s, 0.0) / 2, "change_rate": 0.1, "n": 256}
        for name in ("mean_all", "mask_all", "mask_even", "mask_odd"):
            out[name] = {"accuracy": 0.3, "accuracy_drop": 0.5, "change_rate": 0.6, "n": 256}
        return out

    long = part(long_costs); long["mask_0"]["accuracy_drop"] = register_long
    allp = part({5: terminal})
    return {"contract": "official_codi_masking_audit_v1", "baseline": {"rows": 512}, "checks": {},
            "report": {"all": allp, "by_length": {"short": part(short_costs), "long": long}}, "seconds": 120.0}


def _monitor(readability: dict, faithfulness: dict, run="x/twins_long"):
    return {"run": run, "rows": 1024, "slots": {str(s): {"penultimate": {"rows": 900, "readability": readability.get(s, 0.0),
                                                                        "faithfulness": faithfulness.get(s, 0.0)}} for s in SLOTS}}


def _bootstrap(state: dict):
    return {"run": "x", "target_rate": {f"state_{s}": {"mean": state.get(s, 0.0), "ci95": [max(0.0, state.get(s, 0.0) - 0.03), state.get(s, 0.0) + 0.03], "pairs": 512}
                                        for s in SLOTS}}


def _card(**overrides):
    std = _twins({1: 0.05, 3: 0.10}, {0: 0.23, 2: 0.04, 4: 0.09})
    long = _twins({1: 0.153, 3: 0.325, 5: 0.0}, {0: 0.12, 2: 0.0, 4: 0.30}, kv_all=0.69, other_all=0.2)
    kwargs = dict(model={"name": "gpt2", "config": "configs/official_codi_gpt2.yaml", "checkpoint_sha256": "abc", "layers": 12},
                  pool=std["pool"], environment={"device_name": "Tesla T4", "dtype": "float32", "batch_size": 32, "torch": "2.4.1", "transformers": "4.52.4"},
                  gate={"accuracy": 0.437, "published": 0.437, "status": "passed"}, twins_standard=std, twins_long=long,
                  bootstrap={"standard": _bootstrap({1: 0.05, 3: 0.10}), "long": _bootstrap({1: 0.153, 3: 0.325})},
                  masking=_masking({1: 0.122, 3: 0.125}, {1: 0.03}), monitor={"standard": _monitor({}, {}), "long": _monitor({1: 0.33, 2: 0.46, 3: 0.40, 4: 0.54, 5: 0.42}, {1: 0.19, 3: 0.40})},
                  fingerprints={"twins_standard": {"predictions": "p1", "pairs": "q1"}, "twins_long": {"predictions": "p2", "pairs": "q2"}},
                  stage_seconds={"twins_standard": 400.0})
    kwargs.update(overrides)
    return build_card(**kwargs)


def test_carrier_derivation_uses_the_fixed_threshold():
    assert CARRIER_MIN_TRANSPORT == 0.15
    assert carrying_slots(_twins({1: 0.153, 3: 0.325})) == [1, 3]
    assert carrying_slots(_twins({1: 0.10, 3: 0.34})) == [3]
    assert carrying_slots(_twins({})) == [] and carrying_slots(None) == []


def test_short_target_pools_short_strata_only():
    t = _twins({3: 0.5}, strata={"n1_k1": {"target": 0.0, "n": 100}, "n2_k2": {"target": 0.1, "n": 100}, "n3_k3": {"target": 0.9, "n": 100}})
    assert short_target_of(t, "state_3") == pytest.approx(0.05)
    assert short_target_of(None, "state_3") is None


def test_gpt2_like_card_is_critical_path_with_dissociated_monitor():
    card = _card()
    assert card["contract"] == CARD_CONTRACT
    assert card["verdict"]["carrying_slots"] == [1, 3] and card["verdict"]["placement_slots"] == [1, 3]
    assert card["verdict"]["regime"] == "critical_path"
    c = card["checks"]
    assert c["c1_no_single_site_transport_on_short_chains"] and c["c2_chain_length_gate"] and c["c3_terminal_inert_unfaithful_unnecessary"]
    assert c["c4_readability_dissociated_from_faithfulness"] and c["c4_most_readable_slot"] == 4 and c["c5_carriers_necessary"]
    assert c["register_position_0"] is True
    roles = {s: card["slots"][str(s)]["role"] for s in SLOTS}
    assert roles[1] == roles[3] == "carrier" and roles[5] == "terminal_inert" and roles[2] == "readable_inert"
    assert roles[4] == "store" and roles[0] == "weak" and card["verdict"]["store_slots"] == [4] and card["verdict"]["inert_slots"] == [2, 5]
    assert card["slots"]["3"]["transport_long_ci95"] == pytest.approx([0.295, 0.355])
    assert card["slots"]["3"]["mask_cost_long"] == pytest.approx(0.125)
    text = card_markdown(card)
    assert "critical_path" in text and "| 3 | carrier |" in text and "readable but inert: [2]" in text and "store slots (K/V carries, state does not): [4]" in text
    json.dumps(card)


def test_backup_route_when_carriers_are_sufficient_but_not_necessary():
    card = _card(masking=_masking({3: 0.027}, {}, register_long=0.034), twins_long=_twins({1: 0.10, 3: 0.339, 5: 0.0}, {4: 0.3}))
    assert card["verdict"]["carrying_slots"] == [3] and card["verdict"]["regime"] == "backup_route"
    assert card["checks"]["c5_carriers_necessary"] is False and card["checks"]["register_position_0"] is False


def test_no_transport_and_not_evaluable_regimes():
    card = _card(twins_long=_twins({1: 0.05, 3: 0.08}), masking=_masking({}, {}))
    assert card["verdict"]["regime"] == "no_transport" and "nothing here is monitorable" in card["verdict"]["summary"]
    card = _card(twins_long=None, bootstrap=None, monitor=None, masking=_masking({}, {}))
    assert card["verdict"]["regime"] == "not_evaluable" and card["checks"]["c2_chain_length_gate"] is None
    assert card["checks"]["c1_no_single_site_transport_on_short_chains"] is True


def test_c1_fails_when_a_single_site_transports_on_short_chains():
    card = _card(twins_standard=_twins({3: 0.3}, {0: 0.23}))
    assert card["checks"]["c1_no_single_site_transport_on_short_chains"] is False and card["checks"]["c2_chain_length_gate"] is False


def test_terminal_check_fails_when_the_last_slot_is_needed_or_faithful():
    card = _card(masking=_masking({1: 0.2, 3: 0.2}, terminal=0.1))
    assert card["checks"]["c3_terminal_inert_unfaithful_unnecessary"] is False
    card = _card(monitor={"long": _monitor({5: 0.5}, {5: 0.3})})
    assert card["checks"]["c3_terminal_inert_unfaithful_unnecessary"] is False


def test_profiles_accept_dict_or_list_form():
    std, long = _monitor({}, {}, run="/k/gpt2_twins_standard"), _monitor({3: 0.4}, {3: 0.4}, run="/k/gpt2_twins_long")
    assert by_profile([std, long])["long"] is long and by_profile({"standard": std, "long": long})["standard"] is std
    assert by_profile(None) == {}
    with pytest.raises(ValueError):
        by_profile("nonsense")


def test_spearman_and_roles():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0) and spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None and spearman([1, None], [1, 2]) is None
    assert slot_role(2, {"transport_all": 0.0, "transport_long": 0.0, "readability_long": 0.5}) == "readable_inert"
    assert slot_role(4, {"transport_all": 0.0, "transport_long": 0.0, "kv_transport_long": 0.3, "readability_long": 0.5}) == "store"
    assert slot_role(2, {"transport_all": 0.08, "transport_long": 0.12, "readability_long": 0.5}) == "weak"
    assert slot_role(2, {"transport_all": None, "transport_long": None, "readability_long": None}) == "unmeasured"


def test_environment_fingerprint_is_json_and_names_versions():
    env = environment_fingerprint(device="cpu", dtype="float32", batch_size=8)
    assert env["device"] == "cpu" and env["dtype"] == "float32" and env["batch_size"] == 8 and env["python"]
    json.dumps(env)


def test_compare_flags_disagreements_across_pools_and_environments():
    a = _card()
    b = _card(pool={"pool": "gsm8k_test", "hash": "t" * 64, "n": 1206}, twins_long=_twins({3: 0.34}, {4: 0.3}), masking=_masking({3: 0.03}))
    cmp = compare_cards([a, b])
    assert not cmp["consistent"]
    assert any("carrier set differs" in f for f in cmp["flags"]) and any("regime differs" in f for f in cmp["flags"])
    c = _card(environment={"device_name": "Tesla P100", "dtype": "float32", "batch_size": 32}, fingerprints={"twins_standard": {"predictions": "DIFFERENT", "pairs": "q1"}})
    cmp = compare_cards([a, c])
    assert any("predictions differ across environments" in f for f in cmp["flags"])
    d = _card(environment={"device_name": "Tesla P100", "dtype": "float32", "batch_size": 32})
    cmp = compare_cards([a, d])
    assert cmp["consistent"], cmp["flags"]
    text = comparison_markdown(cmp)
    assert "| gpt2 | gsm8k_aug_train | Tesla T4 |" in text and "- none" in text
