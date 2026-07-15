"""Load + normalize training and eval datasets into a common shape.

Every eval set has a different schema; adapters normalize each into
`{"question": str, "gold": Decimal}` so the eval harness (Phase 1b) is dataset-agnostic.
Training sets normalize into `{"question", "cot", "answer"}`.

Online runs use pinned Hugging Face revisions. Offline runs load normalized datasets from
`CODIKAVA_DATA_ROOT`, produced by `scripts/dataset_prep.py`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

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
    body = str(d.get("Body", d.get("body", "")) or "")
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


def _prepared_path(category: str, name: str) -> Path | None:
    root = os.environ.get("CODIKAVA_DATA_ROOT")
    if not root:
        return None
    path = Path(root) / category / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"prepared dataset is missing: {path}; rerun scripts/dataset_prep.py"
        )
    return path


def _legacy_train_cache(spec: dict):
    """Load a pinned Arrow cache produced by an older ``datasets`` config hash.

    Hugging Face occasionally changes the generated hash for the same pinned JSON file.
    In offline environments the public loader then reports an available cache under a
    different ``default-*`` directory but refuses to use it.  The revision directory and
    schema checks below make that existing immutable artifact safe to reuse.
    """
    from datasets import Dataset, config as datasets_config

    cache_root = Path(os.environ.get("HF_DATASETS_CACHE", datasets_config.HF_DATASETS_CACHE))
    expected_dir = spec["hf_id"].replace("/", "___").casefold()
    revision = spec.get("revision")
    if not cache_root.is_dir() or not revision:
        return None
    repo_dirs = [
        path for path in cache_root.iterdir() if path.is_dir() and path.name.casefold() == expected_dir
    ]
    candidates = []
    for repo_dir in repo_dirs:
        candidates.extend(repo_dir.glob(f"default-*/**/{revision}/*-train.arrow"))
    if len(candidates) != 1:
        return None
    dataset = Dataset.from_file(str(candidates[0]))
    required = {"question", "cot", "answer"}
    if not required.issubset(dataset.column_names):
        return None
    print(f"[data] using pinned legacy Arrow cache: {candidates[0]}")
    return dataset


def load_eval_set(name: str, spec: dict) -> list[dict]:
    """Return [{'question': str, 'gold': Decimal}] for one eval set.

    spec is a configs/data.yaml `eval[name]` entry: {hf_id, split, kind, config?}.
    """
    from datasets import load_dataset, load_from_disk

    kind = spec["kind"]
    adapt = ADAPTERS[kind]
    prepared = _prepared_path("eval", name)
    if prepared:
        ds = load_from_disk(str(prepared))
    else:
        kwargs = {"split": spec.get("split", "test")}
        if "config" in spec:
            kwargs["name"] = spec["config"]
        if spec.get("data_file"):
            # Some Hub dataset configs enumerate every split before returning the one
            # requested by `split`. Supplying the pinned file explicitly prevents an
            # evaluation-only run from downloading unrelated training artifacts.
            kwargs["data_files"] = {
                spec.get("split", "test"): spec["data_file"]
            }
            # Dataset-card metadata can list additional splits that we intentionally
            # did not request. Skip that library-level completeness check; the strict
            # canonical schema, non-empty, and gold checks below still apply.
            kwargs["verification_mode"] = "no_checks"
        if spec.get("revision"):
            kwargs["revision"] = spec["revision"]
        ds = load_dataset(spec["hf_id"], **kwargs)

    out = []
    for index, row in enumerate(ds):
        if prepared:
            question, gold_raw = str(_row(row, "question")), _row(row, "gold")
        else:
            try:
                question, gold_raw = adapt(row)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{name}[{index}] does not match the configured schema") from exc
        if not question.strip():
            raise ValueError(f"{name}[{index}] has an empty question")
        gold = normalize_gold(gold_raw, kind)
        if gold is None:
            raise ValueError(f"{name}[{index}] has unparseable gold answer {gold_raw!r}")
        out.append({"question": question, "gold": gold})
    if not out:
        raise ValueError(f"evaluation dataset {name!r} is empty")
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
    from datasets import load_dataset, load_from_disk

    spec = data_cfg["train"][trace_style]
    fields = data_cfg["train"]["fields"]
    prepared = _prepared_path("train", trace_style)
    if prepared:
        ds = load_from_disk(str(prepared))
    else:
        kwargs = {"split": spec.get("split", "train")}
        if spec.get("data_file"):
            kwargs["data_files"] = {spec.get("split", "train"): spec["data_file"]}
        if spec.get("revision"):
            kwargs["revision"] = spec["revision"]
        try:
            ds = load_dataset(spec["hf_id"], **kwargs)
        except ValueError as exc:
            ds = _legacy_train_cache(spec)
            if ds is None:
                raise exc

    rename = {v: k for k, v in fields.items() if v in ds.column_names and v != k}
    if rename:
        ds = ds.rename_columns(rename)
    keep = ["question", "cot", "answer"]
    missing = [column for column in keep if column not in ds.column_names]
    if missing:
        raise ValueError(
            f"training dataset {spec['hf_id']!r} is missing canonical columns {missing}; "
            f"available columns: {list(ds.column_names)}"
        )
    if len(ds) == 0:
        raise ValueError(f"training dataset {spec['hf_id']!r} is empty")
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop) if drop else ds
