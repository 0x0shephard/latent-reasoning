import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_kaggle_prosqa_audit_notebook.py"
FAMILIES = ("coconut", "codi")


def _builder_run_commit() -> str:
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RUN_COMMIT":
                    return ast.literal_eval(node.value)
    raise AssertionError("no module-level RUN_COMMIT")


def _pinned() -> str:
    commit = _builder_run_commit()
    if commit == "0" * 40:
        pytest.skip("notebooks not pinned yet")
    return commit


@pytest.mark.parametrize("family", FAMILIES)
def test_notebook_pinned_and_complete(family):
    commit = _pinned()
    path = ROOT / "notebooks" / "kaggle_prosqa_audit" / family / f"kaggle_prosqa_audit_{family}.ipynb"
    assert path.exists()
    cells = json.loads(path.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    assert f'RUN_COMMIT = "{commit}"' in text and "scripts/run_prosqa_audit.py" in text and "shutil.rmtree(REPO_DIR" in text
    assert "No training" in text and "New Dataset" in text and "DATA_SHA256" in text and re.search(r'DATA_SHA256 = "[0-9a-f]{64}"', text)
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{family}-cell-{i}")
    meta = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert meta["id"] == "dannyism11/" + re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") and meta["enable_gpu"] and len(meta["id"].split("/")[1]) <= 50


def test_pinned_commit_has_the_runner_and_the_pieces():
    commit = _pinned()
    show = lambda path: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout
    assert "prosqa_latent_audit_v1" in show("scripts/run_prosqa_audit.py") and "def make_twin" in show("src/data/prosqa.py")
    assert "PROSQA_COCONUT_LLAMA1B" in show("src/models/coconut.py") and "connordilgren/llama32-1b-prosqa-codi" in show("configs/prosqa_codi_llama1b.yaml")
