import json
from pathlib import Path

import pytest

from scripts.run_kv_risk_math_token_budget_8192 import (
    compose_8192_records,
    find_4096_reference_root,
)


def _write_reference(root: Path) -> None:
    condition = root / "diagnostic/cap_004096/full"
    records_dir = condition / "records"
    records_dir.mkdir(parents=True)
    top_manifest = {
        "state": "complete",
        "decision": "candidate_cap_still_binding",
    }
    manifest = {
        "state": "complete",
        "model_dtype": "torch.float32",
        "model_revision": "revision",
        "max_new_tokens": 4096,
        "request_sha256": "request",
        "example_sha256": "examples",
    }
    (root / "run_manifest.json").write_text(
        json.dumps(top_manifest),
        encoding="utf-8",
    )
    (condition / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (condition / "summary.json").write_text(
        json.dumps({"examples": 64, "correct": 37}),
        encoding="utf-8",
    )
    (condition / "predictions.jsonl").write_text(
        "{\"example_id\":\"same\"}\n",
        encoding="utf-8",
    )
    for index in range(64):
        (records_dir / f"{index:05d}.json").write_text("{}", encoding="utf-8")


def test_find_reference_accepts_identical_duplicate_exports(tmp_path: Path):
    first = tmp_path / "a/outputs/kv_risk_math_token_budget"
    second = tmp_path / "nested/b/outputs/kv_risk_math_token_budget"
    _write_reference(first)
    _write_reference(second)
    assert find_4096_reference_root(tmp_path) == first.resolve()


def test_composition_reuses_eos_and_extends_only_exact_prefixes():
    reference = [
        {
            "example_id": "eos",
            "finish_reason": "eos",
            "generated_token_ids": [1, 2],
            "correct": True,
        },
        {
            "example_id": "limited",
            "finish_reason": "length",
            "generated_token_ids": [3, 4, 5, 6],
            "correct": False,
        },
    ]
    extension = [
        {
            "example_id": "limited",
            "finish_reason": "eos",
            "generated_token_ids": [3, 4, 5, 6, 7],
            "correct": True,
        }
    ]
    combined, audit = compose_8192_records(
        reference,
        extension,
        reference_cap=4,
    )
    assert combined[0]["example_id"] == "eos"
    assert combined[1]["correct"] is True
    assert audit["reused_eos_examples"] == 1
    assert audit["all_extended_prefixes_exact"] is True


def test_composition_rejects_prefix_mismatch():
    reference = [
        {
            "example_id": "limited",
            "finish_reason": "length",
            "generated_token_ids": [1, 2],
        }
    ]
    extension = [
        {
            "example_id": "limited",
            "finish_reason": "eos",
            "generated_token_ids": [1, 9, 3],
        }
    ]
    with pytest.raises(ValueError, match="does not preserve"):
        compose_8192_records(reference, extension, reference_cap=2)
