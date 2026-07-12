"""Load + normalize training and eval datasets into a common shape.

Every eval set has a different schema; adapters normalize each into
`{"question": str, "gold": float}` so the eval harness (Phase 1b) is dataset-agnostic.
Training sets normalize into `{"question", "cot", "answer"}`.

Actual `load_dataset` calls hit HuggingFace (or a local Kaggle mirror under
HF_HUB_OFFLINE); they are never invoked by the CPU unit tests.
"""
from __future__ import annotations

from typing import Callable, Iterable

from src.data.answer_extract import normalize_gold


# --------------------------------------------------------------------------- #
# Eval-set adapters: raw HF row -> (question_text, gold_raw). `kind` (from
# configs/data.yaml) also drives gold normalization in answer_extract.
# --------------------------------------------------------------------------- #
def _row(d: dict, *keys: str) -> object:
    """First present key among candidates (schemas vary across mirrors)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    raise KeyError(f"none of {keys} in row with keys {list(d)}")


def _adapt_gsm8k_main(d: dict) -> tuple[str, object]:
    return str(_row(d, "question")), _row(d, "answer")


def _adapt_svamp(d: dict) -> tuple[str, object]:
    # ChilleD/SVAMP: Body + Question form the prompt; Answer is numeric.
    body = str(_row(d, "Body", "body", ""))
    q = str(_row(d, "Question", "question"))
    question = (body + " " + q).strip() if body else q
    return question, _row(d, "Answer", "answer")


def _adapt_multiarith(d: dict) -> tuple[str, object]:
    return str(_row(d, "question")), _row(d, "final_ans", "answer", "final_answer")


def _adapt_gsm_hard(d: dict) -> tuple[str, object]:
    return str(_row(d, "input", "question")), _row(d, "target", "answer")


ADAPTERS: dict[str, Callable[[dict], tuple[str, object]]] = {
    "gsm8k_main": _adapt_gsm8k_main,
    "svamp": _adapt_svamp,
    "multiarith": _adapt_multiarith,
    "gsm_hard": _adapt_gsm_hard,
}


def load_eval_set(name: str, spec: dict) -> list[dict]:
    """Return [{'question': str, 'gold': float}] for one eval set.

    spec is a configs/data.yaml `eval[name]` entry: {hf_id, split, kind, config?}.
    """
    from datasets import load_dataset

    kind = spec["kind"]
    adapt = ADAPTERS[kind]
    kwargs = {"split": spec.get("split", "test")}
    if "config" in spec:
        kwargs["name"] = spec["config"]
    ds = load_dataset(spec["hf_id"], **kwargs)

    out = []
    for row in ds:
        question, gold_raw = adapt(row)
        gold = normalize_gold(gold_raw, kind)
        if gold is None:
            continue  # skip unparseable gold rather than silently miscount
        out.append({"question": question, "gold": gold})
    return out


def load_all_eval_sets(data_cfg: dict) -> dict[str, list[dict]]:
    return {name: load_eval_set(name, spec) for name, spec in data_cfg["eval"].items()}


# --------------------------------------------------------------------------- #
# Training set: normalize to {question, cot, answer}.
# --------------------------------------------------------------------------- #
def load_train_set(data_cfg: dict, trace_style: str = "eq_only"):
    """Load a training split and rename columns to the canonical schema.

    trace_style: 'eq_only' (GSM8k-Aug) or 'natural_language' (GSM8k-Aug-NL).
    Returns a HuggingFace Dataset with columns question / cot / answer.
    """
    from datasets import load_dataset

    spec = data_cfg["train"][trace_style]
    fields = data_cfg["train"]["fields"]
    ds = load_dataset(spec["hf_id"], split=spec.get("split", "train"))

    rename = {v: k for k, v in fields.items() if v in ds.column_names and v != k}
    if rename:
        ds = ds.rename_columns(rename)
    keep = ["question", "cot", "answer"]
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop) if drop else ds
