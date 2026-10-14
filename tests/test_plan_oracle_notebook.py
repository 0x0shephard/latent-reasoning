import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_kaggle_plan_oracle_notebook.py"
MODELS = ("codi1b", "simcot", "codi3b")


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


@pytest.mark.parametrize("model", MODELS)
def test_notebook_pinned_and_complete(model):
    commit = _pinned()
    path = ROOT / "notebooks" / "kaggle_plan_oracle" / model / f"kaggle_plan_oracle_{model}.ipynb"
    assert path.exists()
    cells = json.loads(path.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    assert f'RUN_COMMIT = "{commit}"' in text and "scripts/run_codi_plan_oracle.py" in text and "shutil.rmtree(REPO_DIR" in text
    assert "No training" in text and "New Dataset" in text and "tests/test_plan_oracle.py" in text
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{model}-cell-{i}")
    meta = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert meta["id"] == "dannyism11/" + re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") and meta["enable_gpu"]
    assert len(meta["id"].split("/")[1]) <= 50


def test_pinned_commit_has_the_runner_and_the_configs():
    commit = _pinned()
    show = lambda path: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout
    assert "official_codi_plan_oracle_v1" in show("scripts/run_codi_plan_oracle.py")
    for cfg in ("configs/official_codi_llama1b.yaml", "configs/official_simcot_codi_llama1b.yaml", "configs/official_simcot_codi_llama3b.yaml"):
        assert "checkpoint" in show(cfg)
