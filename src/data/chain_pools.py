"""Row pools with an equation chain for the latent monitor audit (ledger §123).

Every pool yields rows ``{"question", "cot", "answer", "gold"}`` in the GSM8k-Aug form:
``cot`` is ``"<<a op b=c>> <<...>>"``, ``answer`` the final integer as text.  The default
pool reproduces the §110 training-split sampling byte for byte; the evaluation pools
turn a dataset's own annotations into the same form.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter

from src.data.counterfactual_chain import parse_chain

DATA_SEED = 20_260_923
DEFAULT_POOL = "gsm8k_aug_train"
POOLS = {
    "gsm8k_aug_train": {"source": "zen-E/GSM8k-Aug train, deterministic fit split (+extra) of ledger §110", "eval_name": None,
                        "long_profile": True},
    "gsm8k_test": {"source": "openai/grade-school-math test.jsonl; calculator annotations joined as the chain", "eval_name": "gsm8k",
                   "long_profile": True},
    "svamp": {"source": "ChilleD/SVAMP test.json; Equation as a one-step chain", "eval_name": "svamp", "long_profile": False},
    "asdiv": {"source": "EleutherAI/asdiv (Miao, Liang & Su 2020; CC BY-NC 4.0); the formula's expression tree evaluated in order as the chain",
              "eval_name": None, "adapter": "asdiv", "long_profile": True,
              "spec": {"hf_id": "EleutherAI/asdiv", "split": "validation", "kind": "asdiv", "revision": "refs/convert/parquet"}},
    "gsm8k_train_original": {
        "source": "openai/grade-school-math train.jsonl (the questions GSM8k-Aug augments, in-distribution); calculator annotations as the chain",
        "eval_name": None, "adapter": "gsm8k", "long_profile": True,
        "spec": {"hf_id": "json", "split": "train", "kind": "gsm8k_main",
                 "data_file": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
                 "source_repo": "openai/grade-school-math", "source_revision": "3101c7d5072418e28b9008a6636bde82a006892c"}},
}
_ANNOTATION = re.compile(r"<<[^<>]*>>")
_WHOLE_FLOAT = re.compile(r"(\d+)\.0+(?!\d)")
_SPACE = re.compile(r"\s+")


def normalized_question(value: object) -> str:
    return _SPACE.sub(" ", str(value).strip().casefold())


def sha_of(values) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _integer_text(value: float) -> str | None:
    if abs(value - round(value)) > 1e-6 or value < 0:
        return None
    return str(int(round(value)))


def _chain_row(question: str, chain: str, gold: str) -> dict | None:
    equations = parse_chain(chain)
    if not equations or not question.strip():
        return None
    try:
        if abs(equations[-1]["result"] - float(gold)) > 1e-6:
            return None
    except ValueError:
        return None
    return {"question": question.strip(), "cot": chain, "answer": gold}


def gsm8k_chain_row(raw: dict) -> dict | None:
    """OpenAI GSM8K: the ``<<...>>`` calculator annotations of the solution, the ``####`` gold."""
    answer = str(raw.get("answer", ""))
    if "####" not in answer:
        return None
    gold = answer.rsplit("####", 1)[1].strip().replace(",", "")
    try:
        gold_text = _integer_text(float(gold))
    except ValueError:
        return None
    annotations = [a.replace(",", "").replace("$", "").replace(" ", "") for a in _ANNOTATION.findall(answer)]
    if gold_text is None or not annotations:
        return None
    return _chain_row(str(raw.get("question", "")), " ".join(annotations), gold_text)


def svamp_chain_row(raw: dict) -> dict | None:
    """ChilleD/SVAMP: ``Body + Question``; ``Equation`` such as ``( ( 16.0 - 11.0 ) + 12.0 )`` as one step."""
    try:
        gold_text = _integer_text(float(raw.get("Answer", raw.get("answer"))))
    except (TypeError, ValueError):
        return None
    equation = str(raw.get("Equation", raw.get("equation", "")) or "").replace(" ", "")
    if gold_text is None or not equation:
        return None
    expression = _WHOLE_FLOAT.sub(r"\1", equation)
    body = str(raw.get("Body", raw.get("body", "")) or "").strip()
    question = str(raw.get("Question", raw.get("question", "")) or "").strip()
    return _chain_row((body + " " + question).strip(), f"<<{expression}={gold_text}>>", gold_text)


def formula_chain(formula: str) -> str | None:
    """``(16-3-4)*2=18`` → ``<<16-3=13>> <<13-4=9>> <<9*2=18>>``: the expression tree evaluated in
    order, one step per operator, operands written by value (the GSM8k-Aug convention)."""
    import ast
    left = str(formula).split("=")[0].replace(" ", "")
    if not left or not re.fullmatch(r"[\d.+\-*/()]+", left):
        return None
    try:
        tree = ast.parse(left, mode="eval").body
    except SyntaxError:
        return None
    ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    steps: list[str] = []

    def fmt(v: float) -> str | None:
        return str(int(round(v))) if abs(v - round(v)) < 1e-6 else None

    def walk(node) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = walk(node.operand); return None if v is None else -v
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            a, b = walk(node.left), walk(node.right)
            if a is None or b is None:
                return None
            op = ops[type(node.op)]
            if op == "/" and b == 0:
                return None
            c = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b
            fa, fb, fc = fmt(a), fmt(b), fmt(c)
            if fa is None or fb is None or fc is None or c < 0:
                return None
            steps.append(f"<<{fa}{op}{fb}={fc}>>")
            return c
        return None

    return " ".join(steps) if walk(tree) is not None and steps else None


def asdiv_chain_row(raw: dict) -> dict | None:
    """EleutherAI/asdiv: ``body + question``; ``formula`` split into steps; ``answer`` like ``9 (apples)``."""
    answer = str(raw.get("answer", "")).split("(")[0].strip().replace(",", "")
    try:
        gold_text = _integer_text(float(answer))
    except ValueError:
        return None
    chain = formula_chain(raw.get("formula", ""))
    if gold_text is None or chain is None:
        return None
    question = (str(raw.get("body", "") or "").strip() + " " + str(raw.get("question", "") or "").strip()).strip()
    return _chain_row(question, chain, gold_text)


CHAIN_ADAPTERS = {"gsm8k": gsm8k_chain_row, "svamp": svamp_chain_row, "asdiv": asdiv_chain_row}


def load_raw_eval_rows(spec: dict) -> list[dict]:
    """The dataset's own rows for a ``configs/data.yaml`` eval entry (``load_eval_set`` keeps only question/gold)."""
    from datasets import load_dataset

    kwargs = {"split": spec.get("split", "test")}
    if spec.get("revision") and not spec.get("data_file") and not spec.get("data_files"):
        kwargs["revision"] = spec["revision"]
        return [dict(row) for row in load_dataset(spec["hf_id"], split=kwargs["split"], revision=spec["revision"])]
    if "config" in spec:
        kwargs["name"] = spec["config"]
    if spec.get("data_files"):
        kwargs["data_files"] = spec["data_files"]; kwargs["verification_mode"] = "no_checks"
    elif spec.get("data_file"):
        kwargs["data_files"] = {spec.get("split", "test"): spec["data_file"]}; kwargs["verification_mode"] = "no_checks"
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    return [dict(row) for row in load_dataset(spec["hf_id"], **kwargs)]


def chain_rows(name: str, raw_rows: list[dict]) -> list[dict]:
    adapter = CHAIN_ADAPTERS[POOLS[name].get("adapter") or POOLS[name]["eval_name"]]
    out = []
    for raw in raw_rows:
        row = adapter(raw)
        if row is not None:
            out.append(row)
    return out


def steps_histogram(rows) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(len(parse_chain(str(r["cot"]))) for r in rows).items())}


def eval_pool(name: str, raw_rows: list[dict], *, prepare, seed: int = DATA_SEED) -> tuple[list[dict], dict]:
    """Deduplicate on the normalised question, keep eligible rows, shuffle deterministically."""
    seen, unique = set(), []
    for row in chain_rows(name, raw_rows):
        key = normalized_question(row["question"])
        if key in seen:
            continue
        seen.add(key); unique.append(row)
    prepared = [p for p in (prepare(r) for r in unique) if p is not None]
    generator = random.Random(seed)
    generator.shuffle(prepared)
    histogram = steps_histogram(prepared)
    meta = {"pool": name, "source": POOLS[name]["source"], "raw_rows": len(raw_rows), "chain_rows": len(unique),
            "n": len(prepared), "seed": seed, "hash": sha_of([normalized_question(r["question"]) for r in prepared]),
            "steps_histogram": histogram, "max_steps": max((int(k) for k in histogram), default=0),
            "long_profile": POOLS[name]["long_profile"]}
    return prepared, meta


def pool_rows(name: str, *, data_cfg, tokenizer, bot_token_id: int, extra_rows: int = 0, seed: int = DATA_SEED) -> tuple[list[dict], dict]:
    """Rows and metadata for one pool.  The default pool defers to the §110 sampler so
    its rows and hashes are unchanged; ``extra_rows`` is the §112 long-profile extension."""
    if name not in POOLS:
        raise ValueError(f"unknown pool {name!r}; choose from {sorted(POOLS)}")
    from scripts.run_codi_causal_subspace_distillation import (  # heavy module; imported only when needed
        FIT_EXAMPLES, SELECT_EXAMPLES, VALIDATE_EXAMPLES, prepare_row, sample_splits)
    from src.data.datasets import load_eval_set, load_train_set

    if name == DEFAULT_POOL:
        test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
        sizes = {"fit": FIT_EXAMPLES, "select": SELECT_EXAMPLES, "validate": VALIDATE_EXAMPLES}
        if extra_rows:
            sizes["extra"] = int(extra_rows)
        dataset = load_train_set(data_cfg, "eq_only")
        splits, sampling = sample_splits(dataset, tokenizer, test_questions={normalized_question(r["question"]) for r in test},
                                         sizes=sizes, seed=seed, bot_token_id=bot_token_id)
        rows = list(splits["fit"]) + list(splits.get("extra", []))
        meta = {"pool": name, "source": POOLS[name]["source"], "n": len(rows), "seed": seed, "hash": sampling["hashes"]["fit"],
                "extra_hash": sampling["hashes"].get("extra"), "sampling": sampling, "steps_histogram": steps_histogram(rows),
                "max_steps": max((len(parse_chain(str(r["cot"]))) for r in rows), default=0), "long_profile": True}
        return rows, meta
    spec = POOLS[name].get("spec") or dict(getattr(data_cfg.eval, POOLS[name]["eval_name"]))
    raw = load_raw_eval_rows(dict(spec))
    return eval_pool(name, raw, prepare=lambda r: prepare_row(tokenizer, r, bot_token_id=bot_token_id), seed=seed)
