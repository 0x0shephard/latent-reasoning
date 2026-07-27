import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "kaggle_kv_compression_risk_pilot.ipynb"
)


def test_notebook_is_valid_and_contains_the_preregistered_workflow():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    required = (
        "Save Version → Save & Run All",
        "RUN_DATASET_SCREEN",
        "RUN_PRIMARY_PILOT",
        "RUN_STOCHASTIC_CHECK",
        "scripts/run_kv_risk_pilot.py",
        "scripts/analyze_kv_risk_pilot.py",
        "--allow-selection-failure",
        "SESSION_NEEDS_RESUME",
        "kv_compression_risk_pilot_export",
    )
    for text in required:
        assert text in source


def test_notebook_does_not_embed_executed_outputs():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []

