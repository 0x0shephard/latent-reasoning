import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_kaggle_latent_hijack_notebook.py"
MODELS = ("gpt2", "codi1b", "simcot", "codi3b", "codi8b")


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
    path = ROOT / "notebooks" / "kaggle_latent_hijack" / model / f"kaggle_latent_hijack_{model}.ipynb"
    assert path.exists()
    cells = json.loads(path.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    assert f'RUN_COMMIT = "{commit}"' in text and "scripts/run_codi_latent_hijack.py" in text and "No training" in text and "New Dataset" in text
    assert "--decoy" in text and "tests/test_latent_hijack.py" in text
    expected = {"gpt2": '"3", "4", "5", []', "codi1b": '"3", "4", "5", []', "simcot": '"3", "4", "5", ["1"]', "codi3b": '"2", "3", "3", ["4"]', "codi8b": '"2", "3", "-1", ["4"]'}[model]
    assert f"CARRIER, STORE, DECOY, EXTRA = {expected}" in text
    assert ('BATCH_SIZE = "16"' in text) == (model == "codi8b")
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{model}-cell-{i}")
    meta = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert meta["id"] == "jonraza15/" + re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") and meta["enable_gpu"] and len(meta["title"]) <= 50


def test_pinned_commit_has_the_runner_and_module():
    commit = _pinned()
    show = lambda path: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout
    assert "def generate_recording" in show("scripts/run_codi_latent_hijack.py") and "state_random_other" in show("scripts/run_codi_latent_hijack.py")
    assert "def choose_attacker_values" in show("src/mech/latent_hijack.py")
