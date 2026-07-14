"""Prepared offline-dataset loading tests."""
from __future__ import annotations

from datasets import Dataset

from src.data.datasets import load_eval_set, load_train_set


def test_prepared_train_and_eval_load_without_hub(monkeypatch, tmp_path):
    root = tmp_path / "prepared"
    Dataset.from_dict(
        {"question": ["What is 2+2?"], "cot": ["2+2=4"], "answer": ["4"]}
    ).save_to_disk(str(root / "train" / "eq_only"))
    Dataset.from_dict({"question": ["What is 3+3?"], "gold": ["6"]}).save_to_disk(
        str(root / "eval" / "gsm8k")
    )
    monkeypatch.setenv("CODIKAVA_DATA_ROOT", str(root))

    data_cfg = {
        "train": {
            "eq_only": {"hf_id": "must-not-be-used", "split": "train"},
            "fields": {"question": "question", "cot": "cot", "answer": "answer"},
        }
    }
    train = load_train_set(data_cfg, "eq_only")
    evaluation = load_eval_set(
        "gsm8k",
        {"hf_id": "must-not-be-used", "split": "test", "kind": "gsm8k_main"},
    )

    assert len(train) == 1
    assert train[0]["cot"] == "2+2=4"
    assert evaluation == [{"question": "What is 3+3?", "gold": 6}]


def test_eval_data_file_limits_remote_download_to_requested_split(monkeypatch):
    import datasets

    observed = {}

    def fake_load_dataset(hf_id, **kwargs):
        observed["hf_id"] = hf_id
        observed.update(kwargs)
        return Dataset.from_dict({"question": ["What is 4+4?"], "answer": ["#### 8"]})

    monkeypatch.delenv("CODIKAVA_DATA_ROOT", raising=False)
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    rows = load_eval_set(
        "gsm8k",
        {
            "hf_id": "openai/gsm8k",
            "config": "main",
            "split": "test",
            "data_file": "main/test-00000-of-00001.parquet",
            "kind": "gsm8k_main",
            "revision": "pinned-revision",
        },
    )

    assert rows == [{"question": "What is 4+4?", "gold": 8}]
    assert observed == {
        "hf_id": "openai/gsm8k",
        "split": "test",
        "name": "main",
        "data_files": {"test": "main/test-00000-of-00001.parquet"},
        "verification_mode": "no_checks",
        "revision": "pinned-revision",
    }
