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
    _write_reference(root)
    assert find_reference_pilot_root(tmp_path) == root.resolve()


def test_find_reference_accepts_identical_duplicate_exports(tmp_path: Path):
    first = tmp_path / "copy_a/outputs/kv_compression_risk_pilot"
    second = tmp_path / "nested/copy_b/outputs/kv_compression_risk_pilot"
    _write_reference(first)
    _write_reference(second)
    assert find_reference_pilot_root(tmp_path) == first.resolve()


def _write_reference(root: Path) -> None:
    condition = root / "screen/math500/full"
    records = condition / "records"
    records.mkdir(parents=True)
    manifest = {
        "state": "complete",
        "model_dtype": "torch.float32",
        "model_revision": "revision",
        "max_new_tokens": 2048,
        "request_sha256": "request",
        "example_sha256": "examples",
    }
    (condition / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (condition / "summary.json").write_text(
        json.dumps({"examples": 64}),
        encoding="utf-8",
    )
    (condition / "predictions.jsonl").write_text(
        "{\"example_id\":\"same\"}\n",
        encoding="utf-8",
    )
    screen = root / "screen"
    (screen / "dataset_selection.json").write_text("{}", encoding="utf-8")
    for index in range(64):
        (records / f"{index:05d}.json").write_text("{}", encoding="utf-8")


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
