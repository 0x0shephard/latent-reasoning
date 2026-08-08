from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "kaggle_official_codi_endpoint_margin_geometry.ipynb"
)


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_notebook_code_cells_are_syntactically_valid():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_notebook_freezes_the_margin_geometry_contract():
    source = _source()
    required = [
        "RUN_COMMIT",
        "CALIBRATION_EXAMPLES = 2048",
        "CALIBRATION_SAMPLING_SEED = 89",
        "PRIMARY_RANK = 3",
        "RANDOM_REPLICATES = 200",
        "RETENTION_THRESHOLD = 0.90",
        "MINIMUM_PARITY_AGREEMENT = 0.99",
        "collect_official_codi_endpoint_margin_states.py",
        "run_official_codi_endpoint_margin_sweep.py",
        "run_official_codi_endpoint_margin_generation.py",
        "analyze_official_codi_endpoint_margin_geometry.py",
        "tests/test_endpoint_margin_geometry.py",
        "train_test_normalized_question_overlap",
        "test_labels_used_for_calibration",
        "SHA256SUMS.txt",
        "official_codi_endpoint_margin_geometry_export",
    ]
    for value in required:
        assert value in source, value


def test_notebook_blocks_the_sweep_on_a_failed_parity_gate():
    source = _source()
    # The analytic tier is only admissible if its first token matches the released
    # decoder, so the notebook must assert that before sweeping.
    assert 'assert parity["passed"]' in source
    assert "analytic parity agreement" in source


def test_notebook_covers_propagating_and_all_position_arms():
    source = _source()
    assert "assert len(GENERATION_ARMS) == 14" in source
    # State 11 reaches the key/value cache; state 12 does not.
    assert '("margin_k003_s12", 11, "remove", "mean", False)'.replace('"', '"') in source or (
        'margin_k003_s12", 11' in source
    )
    assert "--all-positions" in source
    assert '"retain"' in source


def test_notebook_reports_the_binary_versus_continuous_power_contrast():
    source = _source()
    assert "binary z:" in source and "continuous z:" in source
