import pytest

pytest.importorskip("torch")

from scripts.run_codi_workspace_interchange_gate import (  # noqa: E402
    CANDIDATES,
    CONTRACT,
    EVEN_SLOTS,
    M1_MIN_CHANGE,
    M2_MAX_RANK,
    M2_SHARE_OF_FULL,
    M3_ODD_OVER_EVEN,
    M4_MAX_SPEARMAN,
    ODD_SLOTS,
    QUESTIONS,
    RANKS,
    claim_from,
    gates_from,
)


def test_protocol_is_frozen():
    assert CONTRACT == "official_codi_workspace_interchange_gate_v1"
    assert (QUESTIONS, CANDIDATES, RANKS) == (512, 64, (12, 28, 64, 128))
    assert (ODD_SLOTS, EVEN_SLOTS) == ((1, 3, 5), (0, 2, 4))
    assert (M1_MIN_CHANGE, M2_SHARE_OF_FULL, M2_MAX_RANK, M3_ODD_OVER_EVEN, M4_MAX_SPEARMAN) == (0.30, 0.50, 64, 1.5, 0.70)


def _report(odd_full=0.5, odd_64=0.3, even_full=0.2, even_64=0.1, rho=0.3):
    def slot(full, r64, is_odd):
        return {"first_token": {"12": {"change_rate": r64 / 2}, "28": {"change_rate": r64 * 0.8},
                                "64": {"change_rate": r64}, "128": {"change_rate": r64 * 1.1}, "full": {"change_rate": full}},
                "spearman_gradient_vs_intervention": rho if is_odd else None}
    return {"slots": {str(k): slot(odd_full, odd_64, True) for k in (1, 3, 5)}
            | {str(k): slot(even_full, even_64, False) for k in (0, 2, 4)}}


def test_gates_and_claims():
    g = gates_from(_report())
    assert g == {"m1_slots_matter": True, "m2_low_rank_mediation": True, "m3_specificity": True, "m4_divergence": True, "go": True}
    assert claim_from(g).startswith("GO")
    g = gates_from(_report(odd_full=0.2))
    assert not g["m1_slots_matter"] and claim_from(g).startswith("STOP: the odd slots do not change")
    g = gates_from(_report(odd_64=0.1))
    assert not g["m2_low_rank_mediation"]
    g = gates_from(_report(even_full=0.45, even_64=0.28))
    assert not g["m3_specificity"]
    g = gates_from(_report(rho=0.9))
    assert not g["m4_divergence"] and "gradient scores already rank" in claim_from(g)
