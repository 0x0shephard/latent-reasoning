import json
from pathlib import Path

import pytest

from scripts.run_kv_risk_math_token_budget import (
    find_reference_pilot_root,
    paired_accuracy_bootstrap,
    screen_gate,
)


class Struct(dict):
    __getattr__ = dict.__getitem__


def _cfg():
    return Struct(
        screen=Struct(
            accuracy_min=0.60,
            accuracy_max=0.85,
            minimum_median_generated_tokens=512,
        ),
        pilot=Struct(examples=150),
    )


def test_find_reference_pilot_root(tmp_path: Path):
    root = tmp_path / "export/latent-reasoning/outputs/kv_compression_risk_pilot"
    manifest = root / "screen/math500/full/run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    assert find_reference_pilot_root(tmp_path) == root.resolve()


def test_screen_gate_preserves_original_eligibility_contract():
    records = [
        {
            "correct": index < 40,
            "generated_tokens": 700,
            "finish_reason": "eos",
        }
        for index in range(64)
    ]
    result = screen_gate(
        records,
        cfg=_cfg(),
        total_examples=500,
        excluded_examples=128,
    )
    assert result["accuracy"] == pytest.approx(0.625)
    assert result["eligible"] is True
    assert result["unused_examples"] == 372


def test_paired_bootstrap_detects_deterministic_gain():
    reference = [
        {"example_id": f"x:{index}", "correct": False}
        for index in range(20)
    ]
    candidate = [
        {"example_id": f"x:{index}", "correct": True}
        for index in range(20)
    ]
    low, high = paired_accuracy_bootstrap(
        reference,
        candidate,
        samples=200,
        seed=0,
    )
    assert low == 1.0
    assert high == 1.0
