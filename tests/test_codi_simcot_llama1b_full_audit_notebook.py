import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_simcot_codi_llama1b_full_audit.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_simcot_codi_llama1b_full_audit_notebook.py"


def _text() -> str:
    return "\n".join("".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"])


def _builder_run_commit() -> str:
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RUN_COMMIT":
                    return ast.literal_eval(node.value)
    raise AssertionError("no module-level RUN_COMMIT")


def test_notebook_is_pinned_to_a_full_commit():
    commit = _builder_run_commit()
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    assert f'RUN_COMMIT = "{commit}"' in _text()


def test_notebook_runs_the_gate_and_both_profiles():
    text = _text()
    assert "scripts/run_codi_directed_counterfactual.py" in text and "src.eval.official_codi" in text
    assert "official_simcot_codi_llama1b.yaml" in text and "official_codi_llama1b.yaml" in text and '"--smoke"' in text
    assert "model.pad_aware_generation=false" in text and "eval.batch_size=128" in text   # the lost released-path replication
    assert "scripts/run_codi_cache_carrier_audit.py" in text
    assert 'for profile in ("standard", "long")' in text and '"--profile", profile' in text
    for tag in ("S1", "S2", "S3", "S4", "S5", "Step 0", "No training"):
        assert tag in text
    assert "shutil.make_archive" not in text


def test_notebook_runs_focused_tests():
    text = _text()
    for name in ("tests/test_official_codi_family.py", "tests/test_counterfactual_chain.py", "tests/test_directed_counterfactual_runner.py"):
        assert name in text


def test_all_notebook_code_cells_parse():
    for i, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{i}")


def test_pinned_commit_has_the_long_profile_and_the_adapter():
    commit = _builder_run_commit()
    show = lambda path: subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout
    assert '"--profile"' in show("scripts/run_codi_directed_counterfactual.py") and "backbone_layers" in show("scripts/run_codi_directed_counterfactual.py")
    assert "internlm/SIM_COT-LLaMA3-CODI-1B" in show("configs/official_simcot_codi_llama1b.yaml")
    assert "ignored_prefixes" in show("src/models/official_codi.py")
