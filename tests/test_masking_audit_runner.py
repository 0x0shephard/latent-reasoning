import pytest

pytest.importorskip("torch")

from scripts.run_codi_masking_audit import (  # noqa: E402
    CONTRACT,
    K1_MAX_SHORT_DROP,
    K2_MIN_CARRYING_DROP,
    K3_MAX_TERMINAL_DROP,
    chain_steps,
    checks_from,
    claim_from,
    condition_names,
    spec,
)


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_masking_audit_v1"
    assert (K1_MAX_SHORT_DROP, K2_MIN_CARRYING_DROP, K3_MAX_TERMINAL_DROP) == (0.05, 0.10, 0.02)
    assert condition_names()[:2] == ("mean_0", "mean_1") and condition_names()[-4:] == ("mean_all", "mask_all", "mask_even", "mask_odd")
    assert spec("mean_3") == ((3,), ()) and spec("mask_4") == ((), (4,)) and spec("mask_odd") == ((), (1, 3, 5))
    assert spec("mean_all") == ((0, 1, 2, 3, 4, 5), ())
    with pytest.raises(ValueError):
        spec("mask_9")
    assert chain_steps({"cot": "<<2+3=5>> <<5*4=20>>"}) == 2


def _report(short_drops=None, long_drops=None, terminal=0.0):
    def part(drops):
        return {n: {"accuracy": 0.9 - drops.get(n, 0.0), "accuracy_drop": drops.get(n, 0.0), "change_rate": 0.1, "n": 100} for n in condition_names()}
    short = part(short_drops or {}); long = part(long_drops or {})
    allp = part({f"mean_5": terminal, f"mask_5": terminal})
    return {"all": allp, "by_length": {"short": short, "long": long}}


def test_checks_and_claim():
    c = checks_from(_report(long_drops={"mask_3": 0.2, "mean_3": 0.15}), carrying_slots=(3,))
    assert c["k1_short_chains_survive_any_single_mask"] and c["k2_carrying_slot_needed_on_long_chains"] and c["k3_terminal_slot_never_needed"]
    assert "never needed" in claim_from(c)
    c = checks_from(_report(short_drops={"mask_1": 0.2}, long_drops={"mask_3": 0.2}), carrying_slots=(3,))
    assert not c["k1_short_chains_survive_any_single_mask"]
    c = checks_from(_report(long_drops={"mask_3": 0.05}), carrying_slots=(3,))
    assert not c["k2_carrying_slot_needed_on_long_chains"]
    c = checks_from(_report(terminal=0.1), carrying_slots=(3,))
    assert not c["k3_terminal_slot_never_needed"] and "NEEDED" in claim_from(c)
