from scripts.run_codi_xkv_fidelity_frontier import (
    DEVELOPMENT_EXAMPLES,
    DEVELOPMENT_START,
    FINAL_START,
    GROUPS,
    RANKS,
    TASK_WEIGHTS,
    _candidate_name,
)


def test_frontier_protocol_is_frozen_and_final_is_disjoint():
    assert DEVELOPMENT_START == 256
    assert DEVELOPMENT_EXAMPLES == 512
    assert FINAL_START == 768
    assert DEVELOPMENT_START + DEVELOPMENT_EXAMPLES == FINAL_START
    assert RANKS == (16, 24, 32, 40, 48)
    assert TASK_WEIGHTS == (0.0, 0.5, 1.0)
    assert GROUPS == ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))


def test_candidate_names_are_stable_and_path_safe():
    assert _candidate_name(16, 0.0) == "blend_r16_w0"
    assert _candidate_name(32, 0.5) == "blend_r32_w0p5"
    assert _candidate_name(48, 1.0) == "blend_r48_w1"
