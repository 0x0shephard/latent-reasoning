from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "kaggle_official_codi_endpoint_accuracy_localization.ipynb"
)


def test_notebook_is_complete_resumable_localization_workflow():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    required = [
        "RUN_COMMIT",
        "RANDOM_REPLICATES = 100",
        "--localization-covariance",
        "--accuracy-localization",
        "remove_energy_joint_negative_control",
        "remove_{method}_state{state}",
        "remove_{method}_joint_except_s{state}_d{slot}",
        "remove_matched_random_{method}_joint_r{replicate:03d}",
        "assert len(ARMS) == 232",
        "analyze_official_codi_endpoint_accuracy_localization.py",
        "ARM_SHARD_COUNT",
        "RESUME_INPUT",
        "SHA256SUMS.txt",
        "official_codi_endpoint_accuracy_localization_export",
    ]
    for value in required:
        assert value in source
    assert "RUN_FULL = True" in source

