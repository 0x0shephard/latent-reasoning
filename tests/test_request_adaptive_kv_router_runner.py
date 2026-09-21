import ast
from pathlib import Path

from scripts.run_codi_request_adaptive_kv_router import (
    BASELINE_RANK,
    CONTRACT,
    FINAL_EXAMPLES,
    FIT_EXAMPLES,
    PROFILE_NAMES,
    PROFILE_WEIGHTS,
    SCREEN_EXAMPLES,
)


ROOT = Path(__file__).resolve().parents[1]


def test_router_protocol_is_frozen_and_exhausts_svamp():
    assert CONTRACT == "official_codi_request_adaptive_kv_router_svamp_v1"
    assert BASELINE_RANK == 64
    assert PROFILE_WEIGHTS == (0.0, 0.5, 1.0)
    assert PROFILE_NAMES == ("reconstruction", "hybrid", "answer_fisher")
    assert FIT_EXAMPLES + SCREEN_EXAMPLES + FINAL_EXAMPLES == 1000


def test_runner_records_external_holdout_status_and_locked_gate():
    text = (ROOT / "scripts/run_codi_request_adaptive_kv_router.py").read_text()
    assert "external compression-method holdout" in text
    assert 'if screen["gate"]["passed"]:' in text
    assert "positive_kl_gain_interval" in text
    assert "nondegenerate_routing" in text
    assert "answer_logit_observer" in text
    ast.parse(text)
