from scripts.run_codi_fidelity_residual_xkv import (
    FINAL_START,
    FRESH_FIT_EXAMPLES,
    FRESH_SCREEN_EXAMPLES,
    PER_LAYER_GROUPS,
    RANKS,
    RESIDUAL_RANKS,
    _candidate_name,
)


def test_fidelity_residual_protocol_is_frozen():
    assert FRESH_FIT_EXAMPLES == 512
    assert FRESH_SCREEN_EXAMPLES == 512
    assert FINAL_START == 768
    assert RANKS == (48, 64, 80, 96)
    assert RESIDUAL_RANKS == (1, 2, 4)
    assert PER_LAYER_GROUPS == tuple((layer,) for layer in range(12))


def test_fidelity_residual_names_are_stable():
    assert _candidate_name(48, 1) == "margin_residual_xkv_r48_res1"
    assert _candidate_name(96, 4) == "margin_residual_xkv_r96_res4"
