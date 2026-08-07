from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "kaggle_official_codi_parameter_state12_confirmation.ipynb"
)


def test_notebook_freezes_single_hypothesis_confirmation_contract():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    required = [
        "RUN_COMMIT",
        "CALIBRATION_EXAMPLES = 2048",
        "RANDOM_REPLICATES = 500",
        "MAXIMUM_RMS_RATIO_DEVIATION = 0.10",
        "collect_official_codi_parameter_state12_confirmation_stats.py",
        "remove_parameter_aware_state12_primary",
        "remove_matched_random_parameter_aware_state12_r{replicate:03d}",
        "assert len(ARMS) == 502",
        "--state12-confirmation",
        "analyze_official_codi_parameter_state12_confirmation.py",
        "train_test_normalized_question_overlap",
        "test_labels_used_for_calibration",
        "ARM_SHARD_COUNT",
        "RESUME_INPUT",
        "SHA256SUMS.txt",
        "official_codi_parameter_state12_confirmation_export",
    ]
    for value in required:
        assert value in source
    assert "RUN_FULL = True" in source

