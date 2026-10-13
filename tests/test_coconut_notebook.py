import ast
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_kaggle_coconut_audit_notebook.py"


def _commit() -> str:
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RUN_COMMIT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no RUN_COMMIT")


def test_coconut_notebook_pinned_and_complete():
    commit = _commit()
    if commit == "0" * 40:
        pytest.skip("not pinned yet")
    path = ROOT / "notebooks" / "kaggle_coconut_audit" / "kaggle_coconut_audit.ipynb"
    assert path.exists()
    cells = json.loads(path.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    assert f'RUN_COMMIT = "{commit}"' in text and "scripts/run_coconut_audit.py" in text and "No training" in text
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{i}")
    meta = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert meta["id"] == "jonraza15/coconut-audit-gpt-2" and meta["enable_gpu"]
    show = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:src/models/coconut.py"], capture_output=True, text=True, check=True).stdout
    assert "class CoconutGPT2" in show
