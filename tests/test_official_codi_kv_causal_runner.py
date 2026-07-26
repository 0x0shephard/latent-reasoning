"""Runner-level guards for official CODI causal evaluation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run_official_codi_kv_causal import _resolve_reproduction_gate


def test_embedded_passed_reproduction_gate_is_accepted():
    source = {
        "reproduction_gate": {
            "status": "passed",
            "gsm8k_accuracy": 0.4367,
        }
    }
    resolved = _resolve_reproduction_gate(
        None,
        source=source,
        cfg=SimpleNamespace(),
    )
    assert resolved["status"] == "passed"
    assert resolved["gsm8k_accuracy"] == 0.4367
    assert resolved["path"] == "embedded in completed calibration statistics"


@pytest.mark.parametrize(
    "source",
    (
        {},
        {"reproduction_gate": "passed"},
        {"reproduction_gate": {"status": "failed"}},
    ),
)
def test_missing_or_failed_embedded_reproduction_gate_is_rejected(source):
    with pytest.raises(RuntimeError, match="does not contain a passed embedded"):
        _resolve_reproduction_gate(
            None,
            source=source,
            cfg=SimpleNamespace(),
        )
