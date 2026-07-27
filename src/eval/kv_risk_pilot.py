"""Inference and data utilities for the KV-compression risk pilot."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.data.answer_extract import answers_match, extract_final_number, normalize_gold
from src.mech.kv_risk_cache import (
    HeavyHitterRecentCache,
    cache_to_legacy,
)


PILOT_SCHEMA_VERSION = 1
RETENTION_PATTERN = re.compile(r"^retain_(0(?:\.\d+)?|1(?:\.0+)?)$")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def retention_from_condition(condition: str) -> float:
    if condition in {"full", "full_repeat"} or condition.startswith("full_seed"):
        return 1.0
    base = condition.split("_seed", 1)[0]
    match = RETENTION_PATTERN.match(base)
    if not match:
        raise ValueError(f"unsupported condition: {condition}")
    retention = float(match.group(1))
    if not 0.0 < retention <= 1.0:
        raise ValueError("retention must lie in (0, 1]")
    return retention


def _first_present(row: dict, *fields: str) -> Any:
    for field in fields:
        if field in row and row[field] is not None:
            return row[field]
    raise KeyError(f"none of {fields} appears in row fields {list(row)}")


def _level_number(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def load_candidate_dataset(name: str, spec: dict) -> list[dict]:
    """Load one public candidate and normalize its question/answer fields."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": str(spec.get("split", "test"))}
    if spec.get("config"):
        kwargs["name"] = str(spec["config"])
    if spec.get("revision"):
        kwargs["revision"] = str(spec["revision"])
    dataset = load_dataset(str(spec["hf_id"]), **kwargs)
    records: list[dict] = []
    question_field = str(spec["question_field"])
    answer_field = str(spec["answer_field"])
    for index, row in enumerate(dataset):
        question = str(_first_present(row, question_field, "question", "problem"))
        answer = str(_first_present(row, answer_field, "answer", "solution"))
        if str(spec["grader"]) == "gsm8k_numeric":
            normalized = normalize_gold(answer, "gsm8k_main")
            if normalized is None:
                raise ValueError(f"{name}[{index}] has an unparseable GSM8K answer")
            gold = str(normalized)
        else:
            gold = answer
        level_field = spec.get("level_field")
        records.append(
            {
                "example_id": f"{name}:{index:05d}",
                "dataset": name,
                "dataset_index": index,
                "question": question,
                "gold": gold,
                "grader": str(spec["grader"]),
                "level": (
                    None
                    if not level_field
                    else _level_number(row.get(str(level_field)))
                ),
            }
        )
    return records


def deterministic_sample(
    records: list[dict],
    count: int,
    *,
    seed: int,
    excluded_ids: set[str] | None = None,
) -> list[dict]:
    excluded_ids = excluded_ids or set()
    eligible = [
        record
        for record in records
        if str(record["example_id"]) not in excluded_ids
    ]
    if count > len(eligible):
        raise ValueError(f"requested {count} examples from only {len(eligible)}")
    order = list(range(len(eligible)))
    random.Random(seed).shuffle(order)
    return [eligible[index] for index in order[:count]]


def score_answer(generation: str, gold: str, grader: str) -> bool:
    if grader == "gsm8k_numeric":
        return answers_match(generation, Decimal(gold))
    if grader != "math_verify":
        raise ValueError(f"unknown grader: {grader}")
    try:
        from math_verify import parse, verify

        gold_parsed = parse(gold)
        answer_parsed = parse(generation)
        return bool(gold_parsed and answer_parsed and verify(gold_parsed, answer_parsed))
    except Exception:
        # A grading exception is a failed parse, not an experiment crash. The
        # raw generation is retained for later audit.
        return False


def extracted_answer(generation: str, grader: str) -> str | None:
    if grader == "gsm8k_numeric":
        value = extract_final_number(generation)
        return None if value is None else str(value)
    try:
        from math_verify import parse

        parsed = parse(generation)
        return None if not parsed else str(parsed[0])
    except Exception:
        return None


def build_prompt(tokenizer, question: str, instruction: str) -> dict[str, torch.Tensor]:
    content = f"{instruction.strip()}\n\n{question.strip()}"
    messages = [{"role": "user", "content": content}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return encoded


def predictive_entropy(logits: torch.Tensor) -> float:
    probabilities = torch.softmax(logits.float(), dim=-1)
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    return float((-(probabilities * log_probabilities).sum(dim=-1)).item())


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scores = logits.float() / temperature
    probabilities = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=-1,
        )
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
            dim=-1,
            keepdim=True,
        )
        sampled = torch.multinomial(
            sorted_probabilities,
            num_samples=1,
            generator=generator,
        )
        return int(torch.gather(sorted_indices, -1, sampled).item())
    sampled = torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )
    return int(sampled.item())


def eos_token_ids(model, tokenizer) -> set[int]:
    configured = getattr(model.generation_config, "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    if isinstance(configured, int):
        return {configured}
    return {int(value) for value in configured or []}


@dataclass(frozen=True)
class DecodeRequest:
    retention: float
    max_new_tokens: int
    recent_window: int
    heavy_fraction: float
    early_entropy_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    sampling_seed: int = 0


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    *,
    question: str,
    instruction: str,
    request: DecodeRequest,
    device: torch.device,
) -> dict:
    """Decode one answer while tracking realized cache-token memory."""
    encoded = build_prompt(tokenizer, question, instruction)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    initial = model(
        **encoded,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    cache = initial.past_key_values
    logits = initial.logits[:, -1, :]
    compressor = None
    if request.retention < 1.0:
        compressor = HeavyHitterRecentCache(
            cache,
            prompt_length=prompt_length,
            retention=request.retention,
            recent_window=request.recent_window,
            heavy_fraction=request.heavy_fraction,
        )

    generator = None
    if request.temperature > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(request.sampling_seed))
    stop_ids = eos_token_ids(model, tokenizer)
    generated: list[int] = []
    entropies: list[float] = []
    retained_generated_token_steps = 0
    retained_total_token_steps = 0
    uncompressed_generated_token_steps = 0
    uncompressed_total_token_steps = 0
    max_retained_total = prompt_length
    finish_reason = "length"

    for _ in range(request.max_new_tokens):
        if len(entropies) < request.early_entropy_tokens:
            entropies.append(predictive_entropy(logits))
        token_id = sample_next_token(
            logits,
            temperature=request.temperature,
            top_p=request.top_p,
            generator=generator,
        )
        generated.append(token_id)
        if token_id in stop_ids:
            finish_reason = "eos"
            break

        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        current_cache_length = int(cache_to_legacy(cache)[0][0].shape[2])
        attention_mask = torch.ones(
            (1, current_cache_length + 1),
            dtype=torch.long,
            device=device,
        )
        absolute_position = prompt_length + len(generated) - 1
        position_ids = torch.tensor(
            [[absolute_position]],
            dtype=torch.long,
            device=device,
        )
        decoded = model(
            input_ids=token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=compressor is not None,
            return_dict=True,
        )
        cache = decoded.past_key_values
        seen_generated = len(generated)
        if compressor is not None:
            compressor.update(
                cache,
                decoded.attentions,
                appended_absolute_position=absolute_position,
            )
            cache, cache_metrics = compressor.prune(
                cache,
                seen_generated=seen_generated,
            )
            retained_generated = cache_metrics.retained_generated
            retained_total = cache_metrics.retained_total
        else:
            retained_generated = seen_generated
            retained_total = prompt_length + seen_generated
        retained_generated_token_steps += retained_generated
        retained_total_token_steps += retained_total
        uncompressed_generated_token_steps += seen_generated
        uncompressed_total_token_steps += prompt_length + seen_generated
        max_retained_total = max(max_retained_total, retained_total)
        logits = decoded.logits[:, -1, :]

    generation = tokenizer.decode(generated, skip_special_tokens=True)
    generated_steps = max(1, uncompressed_generated_token_steps)
    total_steps = max(1, uncompressed_total_token_steps)
    return {
        "generation": generation,
        "generated_tokens": len(generated),
        "prompt_tokens": prompt_length,
        "finish_reason": finish_reason,
        "early_entropy_mean": (
            None if not entropies else float(np.mean(entropies))
        ),
        "early_entropy_values": entropies,
        "retained_generated_token_steps": retained_generated_token_steps,
        "uncompressed_generated_token_steps": uncompressed_generated_token_steps,
        "realized_generated_retention": (
            retained_generated_token_steps / generated_steps
        ),
        "retained_total_token_steps": retained_total_token_steps,
        "uncompressed_total_token_steps": uncompressed_total_token_steps,
        "realized_total_retention": retained_total_token_steps / total_steps,
        "max_retained_total_tokens": max_retained_total,
    }


def record_identity(
    *,
    experiment: str,
    condition: str,
    example_id: str,
    model_revision: str,
    decoding: dict,
) -> str:
    return sha256_json(
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "experiment": experiment,
            "condition": condition,
            "example_id": example_id,
            "model_revision": model_revision,
            "decoding": decoding,
        }
    )


def load_records(condition_dir: Path) -> list[dict]:
    paths = sorted((condition_dir / "records").glob("*.json"))
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]


def consolidate_records(condition_dir: Path) -> list[dict]:
    records = load_records(condition_dir)
    atomic_jsonl(condition_dir / "predictions.jsonl", records)
    return records


def select_screen_dataset(
    summaries: dict[str, dict],
    *,
    accuracy_min: float,
    accuracy_max: float,
    accuracy_midpoint: float,
    minimum_median_generated_tokens: int,
    pilot_examples: int,
) -> dict:
    audited: dict[str, dict] = {}
    eligible: list[str] = []
    for name, summary in summaries.items():
        reasons: list[str] = []
        accuracy = float(summary["accuracy"])
        median_tokens = float(summary["median_generated_tokens"])
        unused = int(summary["unused_examples"])
        if not accuracy_min <= accuracy <= accuracy_max:
            reasons.append("accuracy_outside_preregistered_band")
        if median_tokens < minimum_median_generated_tokens:
            reasons.append("reasoning_trace_too_short")
        if unused < pilot_examples:
            reasons.append("insufficient_disjoint_pilot_examples")
        audited[name] = {
            **summary,
            "eligible": not reasons,
            "ineligibility_reasons": reasons,
        }
        if not reasons:
            eligible.append(name)
    if not eligible:
        return {
            "status": "no_eligible_dataset",
            "selected_dataset": None,
            "datasets": audited,
        }
    selected = min(
        eligible,
        key=lambda name: (
            abs(float(audited[name]["accuracy"]) - accuracy_midpoint),
            -float(audited[name]["median_generated_tokens"]),
            name,
        ),
    )
    return {
        "status": "selected",
        "selected_dataset": selected,
        "datasets": audited,
    }


def failure_set(
    full_records: list[dict],
    compressed_records: list[dict],
) -> set[str]:
    full = {record["example_id"]: record for record in full_records}
    compressed = {record["example_id"]: record for record in compressed_records}
    if full.keys() != compressed.keys():
        raise ValueError("paired conditions contain different examples")
    return {
        example_id
        for example_id in full
        if bool(full[example_id]["correct"])
        and not bool(compressed[example_id]["correct"])
    }


def adjacent_containment(
    failure_sets: list[tuple[str, set[str]]],
) -> dict:
    comparisons: list[dict] = []
    defined: list[float] = []
    for (looser_name, looser), (tighter_name, tighter) in zip(
        failure_sets,
        failure_sets[1:],
    ):
        if looser:
            containment = len(looser & tighter) / len(looser)
            reversal = len(looser - tighter) / len(looser)
            defined.append(containment)
        else:
            containment = None
            reversal = None
        comparisons.append(
            {
                "looser": looser_name,
                "tighter": tighter_name,
                "looser_failures": len(looser),
                "tighter_failures": len(tighter),
                "intersection": len(looser & tighter),
                "containment": containment,
                "reversal_rate": reversal,
            }
        )
    return {
        "comparisons": comparisons,
        "defined_comparisons": len(defined),
        "mean_adjacent_containment": (
            None if not defined else float(np.mean(defined))
        ),
    }

