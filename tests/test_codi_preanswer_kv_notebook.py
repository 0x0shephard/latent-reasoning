import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_preanswer_kv_notebook_has_corrected_contract_and_safety_gate():
    path = ROOT / "notebooks" / "kaggle_codi_preanswer_kv_subspaces.ipynb"
    notebook = json.loads(path.read_text())
    text = "\n".join(cell.get("source", "") for cell in notebook["cells"])
    required = (
        "exact pre-answer key and value cache tensors",
        "run_codi_preanswer_kv_subspace_discovery.py",
        "tests/test_preanswer_kv_subspace.py",
        "preanswer_kv_subspaces.pt",
        "statistically_validated_direction_count",
        "operational_rank",
        "gradient_connectivity_fraction",
        "ALLOW_UNCONFIRMED_COMPRESSION_EXPLORATION = False",
        "full-answer NLL difference",
        "run_codi_causal_xkv.py",
    )
    for fragment in required:
        assert fragment in text
    assert "RANK = 28" not in text
    assert len(notebook["cells"]) >= 20
