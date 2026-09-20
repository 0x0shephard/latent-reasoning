from scripts.run_codi_adaptive_kv_allocation import (
    ANSWER_WEIGHTS,
    BASELINE_RANKS,
    FINAL_START,
    FRESH_CALIBRATION_EXAMPLES,
    FRESH_SCREEN_EXAMPLES,
    MAXIMUM_COMPONENT_RANK,
    PER_LAYER_GROUPS,
    _candidate_name,
)


def test_adaptive_allocation_protocol_is_frozen():
    assert FRESH_CALIBRATION_EXAMPLES == 512
    assert FRESH_SCREEN_EXAMPLES == 512
    assert FINAL_START == 768
    assert BASELINE_RANKS == (48, 64, 80)
    assert ANSWER_WEIGHTS == (0.0, 0.5, 1.0)
    assert MAXIMUM_COMPONENT_RANK == 96
    assert PER_LAYER_GROUPS == tuple((layer,) for layer in range(12))


def test_adaptive_allocation_names_are_stable():
    assert _candidate_name(48, 0.0) == "adaptive_kv_r48_w0"
    assert _candidate_name(64, 0.5) == "adaptive_kv_r64_w0p5"
    assert _candidate_name(80, 1.0) == "adaptive_kv_r80_w1"
