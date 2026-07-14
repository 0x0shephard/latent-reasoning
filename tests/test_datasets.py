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
