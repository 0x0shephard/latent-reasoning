import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "kaggle_kv_risk_math_token_budget_8192.ipynb"
)


def test_notebook_is_valid_and_contains_the_final_cap_workflow():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    required = (
        "Save Version → Save & Run All",
        "math-500-token-budget-sensitivity-screen",
        "RUN_PREFLIGHT",
        "RUN_FINAL_TOKEN_BUDGET_SCREEN",
        "scripts/validate_kv_risk_preflight.py",
        "scripts/run_kv_risk_math_token_budget_8192.py",
        "kv_risk_math_token_budget_8192_export",
        "SESSION_NEEDS_RESUME",
        "--max-seconds",
    )
    for text in required:
        assert text in source


def test_notebook_has_no_embedded_outputs():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
