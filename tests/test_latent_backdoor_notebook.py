import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_kaggle_latent_backdoor_notebook.py"
SCHEMES = ("backdoor", "inert")


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


@pytest.mark.parametrize("scheme", SCHEMES)
def test_notebook_pinned_and_complete(scheme):
    commit = _pinned()
    path = ROOT / "notebooks" / "kaggle_latent_backdoor" / scheme / f"kaggle_latent_backdoor_{scheme}.ipynb"
    assert path.exists()
    cells = json.loads(path.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    assert f'RUN_COMMIT = "{commit}"' in text and 'ULTERIOR_COMMIT = "bd46cbb"' in text and f'SCHEME = "{scheme}"' in text
    assert "scripts/run_latent_backdoor_train.py" in text and "scripts/run_latent_backdoor_audit.py" in text and "New Dataset" in text
    assert "ok=(0, 42)" in text and "/kaggle/input/" in text          # budget exit and resume from the previous output
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{scheme}-cell-{i}")
    meta = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert meta["id"] == "jonraza15/" + re.sub(r"[^a-z0-9]+", "-", meta["title"].lower()).strip("-") and meta["enable_gpu"] and len(meta["title"]) <= 50
    assert all(src == meta["id"] for src in meta["kernel_sources"])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_audit_notebook_attaches_the_training_output(scheme):
    commit = _pinned()
    path = ROOT / "notebooks" / "kaggle_latent_backdoor" / f"{scheme}_audit" / f"kaggle_latent_backdoor_{scheme}_audit.ipynb"
    assert path.exists()
    text = "\n".join("".join(c["source"]) for c in json.loads(path.read_text())["cells"])
    assert f'RUN_COMMIT = "{commit}"' in text and 'MODE = "audit"' in text and "scripts/run_latent_backdoor_audit.py" in text
    meta = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] and meta["kernel_sources"][0].endswith(re.sub(r"[^a-z0-9]+", "-", f"Latent backdoor train {'dual-trigger' if scheme == 'backdoor' else 'inert-trigger control'}".lower()))


def test_pinned_commit_has_the_runners_and_module():
    commit = _pinned()
    show = lambda path: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout
    assert "def fast_latent_forward" in show("src/mech/latent_backdoor.py") and "InertTriggerDataset" in show("src/mech/latent_backdoor.py")
    assert "BudgetExhausted" in show("scripts/run_latent_backdoor_train.py") and "control_with_armed_latents_all" in show("scripts/run_latent_backdoor_audit.py")
