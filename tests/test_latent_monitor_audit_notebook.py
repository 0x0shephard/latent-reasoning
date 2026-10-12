import ast
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_kaggle_latent_monitor_audit_notebook.py"
CASES = (("gpt2", "gsm8k_test"), ("gpt2", "svamp"), ("gpt2", "gsm8k_aug_train"), ("codi1b", "gsm8k_test"), ("simcot", "gsm8k_test"))


def _notebook(model, pool):
    return ROOT / "notebooks" / "kaggle_latent_monitor_audit" / f"{model}_{pool}" / f"kaggle_lma_{model}_{pool}.ipynb"


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


@pytest.mark.parametrize("model,pool", CASES)
def test_notebook_pinned_and_complete(model, pool):
    commit = _pinned()
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    path = _notebook(model, pool)
    assert path.exists(), f"build with: python scripts/build_kaggle_latent_monitor_audit_notebook.py --model {model} --pool {pool}"
    cells = json.loads(path.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    assert f'RUN_COMMIT = "{commit}"' in text and f'POOL = "{pool}"' in text
    for needle in ("scripts/run_latent_monitor_audit.py", "tests/test_monitor_audit.py", "No training", "New Dataset", "card.md"):
        assert needle in text
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{model}-{pool}-cell-{i}")
    meta = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert meta["id"] == f"jonraza15/lma-{model}-{pool}".replace("_", "-") and meta["code_file"] == path.name
    assert meta["enable_gpu"] and meta["enable_internet"] and (meta["dataset_sources"] == ["dannyism11/tsv-ccc"]) == (model == "gpt2")


def test_pinned_commit_has_the_driver_and_pools():
    commit = _pinned()
    show = lambda path: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout
    assert "def build_card" in show("src/mech/monitor_audit.py") and "svamp_chain_row" in show("src/data/chain_pools.py")
    assert "--from-existing" in show("scripts/run_latent_monitor_audit.py") and '"--pool"' in show("scripts/run_codi_directed_counterfactual.py")
    assert 'nargs="*"' in show("scripts/run_codi_masking_audit.py")
