import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_task_sensitive_notebook_has_disjoint_selection_and_causal_controls():
    path = ROOT / "notebooks" / "kaggle_codi_task_sensitive_kv_subspaces.ipynb"
    notebook = json.loads(path.read_text())
    text = "\n".join(cell.get("source", "") for cell in notebook["cells"])
    required = (
        "linear combination of covariance directions",
        "run_codi_task_sensitive_kv_subspaces.py",
        "tests/test_task_sensitive_kv_subspace.py",
        "COVARIANCE_FIT_EXAMPLES = 1024",
        "TASK_DISCOVERY_EXAMPLES = 1024",
        "RANK_SELECTION_EXAMPLES = 256",
        "CAUSAL_CONFIRMATION_EXAMPLES = 128",
        "task_sensitive_kv_subspaces.pt",
        "key_task_remove", "value_task_remove", "covariance_remove",
        "ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION = False",
        "gradient_connectivity_fraction",
    )
    for fragment in required:
        assert fragment in text
    assert len(notebook["cells"]) >= 24
