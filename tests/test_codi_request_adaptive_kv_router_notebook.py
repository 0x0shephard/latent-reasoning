import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle_codi_request_adaptive_kv_router.ipynb"
BUILDER = ROOT / "scripts/build_kaggle_codi_request_adaptive_kv_router_notebook.py"


def _text():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_notebook_freezes_external_svamp_protocol_and_router_inputs():
    text = _text()
    assert "FIT_EXAMPLES = 400" in text
    assert "SCREEN_EXAMPLES = 300" in text
    assert "FINAL_EXAMPLES = 300" in text
    assert "external compression-method holdout" in text
    assert "only lexical question features" in text
    assert '"--baseline-rank", str(BASELINE_RANK)' in text
    assert '"--ridge", "1.0"' in text


def test_notebook_discovers_the_predecessor_by_contract_and_runs_focused_tests():
    text = _text()
    assert "official_codi_adaptive_kv_allocation_holdout_v1" in text
    assert "/kaggle/input/**/adaptive_kv_allocation.pt" in text
    assert "tests/test_request_adaptive_kv_router.py" in text
    assert "scripts/run_codi_request_adaptive_kv_router.py" in text
    assert "request_adaptive_kv_router.pt" in text
    assert "shutil.make_archive" not in text


def test_notebook_code_and_builder_parse():
    notebook = json.loads(NOTEBOOK.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")
    ast.parse(BUILDER.read_text(), filename=str(BUILDER))
