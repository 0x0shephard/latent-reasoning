import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_combined_notebook_freezes_confirmation_before_compression():
    path = ROOT / "notebooks" / "kaggle_codi_confirm_and_compress_native_kv.ipynb"
    notebook = json.loads(path.read_text())
    text = "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source"), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
    )
    required = (
        "run_codi_confirm_and_compress_native_kv.py",
        "tests/test_confirm_direct_cache_subspace.py",
        "CANDIDATE_LAYERS = \"2,3,5,11\"",
        "CONFIRM_EXAMPLES = 512", "CONFIRM_FOLDS = 4",
        "primary_bootstrap_95ci", "candidate_familywise_95ci",
        "Only rank 1 controls the primary pass/fail decision",
        "confirmed_native_kv_artifact.pt", "run_codi_causal_xkv.py",
        "timing_warning",
    )
    for value in required:
        assert value in text
    assert text.index("run_codi_confirm_and_compress_native_kv.py") < text.index(
        '"--layerwise-artifact"'
    )
    assert "__RUN_COMMIT__" not in text
    assert len(notebook["cells"]) >= 20
