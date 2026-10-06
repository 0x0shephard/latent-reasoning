import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_codi_llama1b_cache_carrier_audit.ipynb"
BUILDER = ROOT / "scripts" / "build_kaggle_codi_llama1b_cache_carrier_audit_notebook.py"


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


def test_notebook_exposes_the_gate_and_its_inputs():
    text = _text()
    assert "scripts/run_codi_cache_carrier_audit.py" in text
    assert '"--smoke"' in text and "RUN_SMOKE" in text
    assert 'REPRODUCTION_SUMMARY = summary_under("outputs/official_codi_llama1b")' in text   # produces its own gate file
    assert "model.pad_aware_generation=false" in text and "eval.batch_size=128" in text   # released-path replication
    for check in ("A1", "A2", "A3", "A4", "A5", "A6"):
        assert check in text
    assert "K/V" in text and "No training" in text and "official_codi_llama1b.yaml" in text and "src.eval.official_codi" in text
    assert "shutil.make_archive" not in text


def test_notebook_runs_focused_tests():
    text = _text()
    assert "tests/test_official_codi_family.py" in text
    assert "tests/test_cache_carrier.py" in text
    assert "tests/test_cache_carrier_audit_runner.py" in text


def test_all_notebook_code_cells_parse():
    for i, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{i}")


def test_builder_parses():
    ast.parse(BUILDER.read_text(), filename=str(BUILDER))


def test_pinned_commit_has_the_llama_adapter():
    commit = _builder_run_commit()
    runner = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:scripts/run_codi_cache_carrier_audit.py"],
                            capture_output=True, text=True, check=True).stdout
    config = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:configs/official_codi_llama1b.yaml"],
                            capture_output=True, text=True, check=True).stdout
    model = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:src/models/official_codi.py"],
                           capture_output=True, text=True, check=True).stdout
    assert "protocol_for" in runner and "pad_aware_generation: true" in config and "_PadAwareStepper" in model
    assert "direction: at_least" in config
