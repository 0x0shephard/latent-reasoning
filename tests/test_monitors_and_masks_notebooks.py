import ast
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_kaggle_model_pipeline_notebook.py"
MODELS = ("gpt2", "codi1b", "simcot")


def _notebook(model):
    return ROOT / "notebooks" / f"kaggle_{model}_monitors_and_masks" / f"kaggle_{model}_monitors_and_masks.ipynb"


def _text(model) -> str:
    return "\n".join("".join(c["source"]) for c in json.loads(_notebook(model).read_text())["cells"])


def _builder_run_commit() -> str:
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RUN_COMMIT":
                    return ast.literal_eval(node.value)
    raise AssertionError("no module-level RUN_COMMIT")


@pytest.mark.parametrize("model", MODELS)
def test_notebook_pinned_and_complete(model):
    commit = _builder_run_commit()
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    text = _text(model)
    assert f'RUN_COMMIT = "{commit}"' in text
    for needle in ("scripts/run_codi_directed_counterfactual.py", "scripts/run_codi_masking_audit.py",
                   "analyze_monitor_faithfulness", "analyze_directed_counterfactual_bootstrap", "No training", "New Dataset"):
        assert needle in text
    for i, cell in enumerate(json.loads(_notebook(model).read_text())["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{model}-cell-{i}")
    meta = json.loads((_notebook(model).parent / "kernel-metadata.json").read_text())
    assert meta["code_file"] == _notebook(model).name and meta["enable_gpu"] and meta["enable_internet"]
    assert (meta["dataset_sources"] == ["dannyism11/tsv-ccc"]) == (model == "gpt2")


def test_pinned_commit_has_masking_and_monitor_code():
    commit = _builder_run_commit()
    show = lambda path: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout
    assert "mask_slots" in show("src/mech/cache_carrier.py") and "block_last" in show("src/models/official_codi.py")
    assert "official_codi_masking_audit_v1" in show("scripts/run_codi_masking_audit.py")
    assert "def analyze" in show("scripts/analyze_monitor_faithfulness.py")
