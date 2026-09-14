import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_direct_cache_notebook_has_native_kv_contract_and_gates():
    path = ROOT / "notebooks" / "kaggle_codi_direct_cache_task_subspaces.ipynb"
    notebook = json.loads(path.read_text())
    text = "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source"), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
    )
    required = (
        "run_codi_direct_cache_task_subspaces.py",
        "tests/test_direct_cache_task_subspace.py",
        "key_ranks_by_layer", "value_ranks_by_layer",
        "familywise_95ci", "task-covariance hybrid",
        "ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION",
        "direct_cache_task_subspaces.pt",
    )
    for value in required:
        assert value in text
    assert "__RUN_COMMIT__" not in text
    assert len(notebook["cells"]) >= 15
