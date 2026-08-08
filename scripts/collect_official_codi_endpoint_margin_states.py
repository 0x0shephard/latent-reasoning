"""Cache answer-colon states and prove the analytic state-12 shortcut is exact.

Nothing in this experiment is allowed to use the closed-form evaluator until the
parity gate here shows that ``argmax(W h12)`` reproduces the token the released
greedy decoder actually emits.  That gate is what converts an assumption about
GPT-2's ``lm_head`` into a checked property of this checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_activation_stats import _amp_context
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    verify_full_reproduction_gate,
)
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES,
    GSM8K_SOURCE_REVISION,
    GSM8K_TRAIN_URL,
    _canonical_gsm8k_row,
    sample_gsm8k_train_calibration,
)
from src.data.answer_extract import normalize_number
from src.data.datasets import load_eval_set
from src.data.official_codi_training import (
    OFFICIAL_CODI_SOURCE_REVISION,
    collate_official_codi_kv_rows,
)
from src.eval.official_codi import select_device
from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    MARGIN_GEOMETRY_CONTRACT,
    MARGIN_GEOMETRY_SCHEMA_VERSION,
    MARGIN_GEOMETRY_STATES,
    gold_first_token_ids,
    numeric_answer_token_ids,
    resolve_output_embedding,
)
from src.mech.official_codi_target_utility import OfficialCODIAnswerScorer
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_SOURCE_REVISION}/grade_school_math/data/test.jsonl"
)


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_gsm8k_test_rows(evaluation, raw_test) -> tuple[list[dict], dict]:
    """Attach each evaluation question's pinned reasoning trace, in evaluation order.

    ``load_eval_set`` yields only ``{"question", "gold"}``; the teacher-forced colon
    extraction additionally needs the chain of thought in order to build teacher
    boundaries.  The trace is therefore joined from the same pinned
    ``openai/grade-school-math`` revision that supplies calibration.

    Evaluation order is preserved exactly, and every gold answer is re-derived from
    the pinned source and compared, so the resulting states stay paired with every
    completed full-GSM8K experiment rather than merely assumed to be.
    """
    by_question: dict[str, dict] = {}
    ineligible = 0
    for row in raw_test:
        canonical = _canonical_gsm8k_row(dict(row))
        if canonical is None:
            ineligible += 1
            continue
        key = _normalized_question(canonical["question"])
        if key in by_question:
            raise RuntimeError("the pinned GSM8K test split has duplicate questions")
        by_question[key] = canonical
    ordered: list[dict] = []
    for index, example in enumerate(evaluation):
        key = _normalized_question(example["question"])
        canonical = by_question.get(key)
        if canonical is None:
            raise RuntimeError(
                f"GSM8K test row {index} has no pinned reasoning trace; "
                "the evaluation set and the pinned source disagree"
            )
        gold = normalize_number(canonical["answer"])
        if gold is None or gold != example["gold"]:
            raise RuntimeError(
                f"GSM8K test row {index} gold answer disagrees with the pinned source: "
                f"{canonical['answer']!r} vs {example['gold']!r}"
            )
        # Keep the evaluation set's own question string and take only the trace from
        # the pinned source.  The two agree after normalisation but may differ in raw
        # whitespace, and the cached colon states must belong to exactly the text the
        # generation runner feeds the model.
        ordered.append(
            {
                "question": example["question"],
                "cot": canonical["cot"],
                "answer": canonical["answer"],
            }
        )
    return ordered, {
        "raw_test_examples": len(raw_test),
        "ineligible_rows": ineligible,
        "matched_examples": len(ordered),
        "evaluation_order_preserved": True,
        "gold_verified_against_pinned_source": True,
        "normalized_questions_sha256": _sha256_json(
            [_normalized_question(row["question"]) for row in ordered]
        ),
    }


def _collect_colon_states(
    scorer,
    tokenizer,
    rows,
    *,
    bot_token_id: int,
    batch_size: int,
    device: torch.device,
    precision: str,
    description: str,
) -> torch.Tensor:
    """Teacher-forced colon states ``[N, 13, 768]`` for the given canonical rows."""
    collected = []
    progress = tqdm(total=len(rows), unit="examples", desc=description)
    for start in range(0, len(rows), batch_size):
        batch = collate_official_codi_kv_rows(
            tokenizer, rows[start : start + batch_size], bot_token_id=bot_token_id
        ).to(device)
        with torch.no_grad(), _amp_context(device, precision):
            output = scorer(batch, return_answer_endpoint_hidden=True)
        hidden = output.student_answer_endpoint_hidden
        if hidden is None or hidden.shape[1:] != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE):
            raise RuntimeError("student answer-colon hidden-state shape changed")
        collected.append(hidden.detach().float().cpu())
        progress.update(hidden.shape[0])
    progress.close()
    states = torch.cat(collected, dim=0)
    if states.shape[0] != len(rows):
        raise RuntimeError("collected colon-state count does not match the row count")
    if not torch.isfinite(states).all():
        raise RuntimeError("collected colon states contain non-finite values")
    return states


def _parity_gate(
    model,
    tokenizer,
    rows,
    states: torch.Tensor,
    readout: torch.Tensor,
    *,
    latent_iterations: int,
    batch_size: int,
    device: torch.device,
    minimum_agreement: float,
) -> dict:
    """Check the closed-form first token against the released greedy decoder.

    ``generate_official_codi`` with ``max_new_tokens=1`` emits exactly the token
    the analytic evaluator predicts, so string equality here is a direct test of
    ``logits == W @ ln_f(h)`` on this checkpoint, including its LoRA adapters.
    """
    questions = [row["question"] for row in rows]
    generated = generate_official_codi(
        model,
        tokenizer,
        questions,
        latent_iterations=latent_iterations,
        max_new_tokens=1,
        batch_size=batch_size,
        device=device,
        force_answer_cue=True,
    )
    predicted = (states[:, ANALYTIC_STATE, :].to(readout.device) @ readout.T).argmax(dim=-1)
    analytic = [
        tokenizer.decode([int(token)], skip_special_tokens=True)
        for token in predicted.tolist()
    ]
    matches = [bool(a == b) for a, b in zip(analytic, generated)]
    agreement = float(sum(matches) / max(1, len(matches)))
    passed = bool(agreement >= minimum_agreement)
    disagreements = [
        {"index": index, "analytic": analytic[index], "generated": generated[index]}
        for index, match in enumerate(matches)
        if not match
    ][:8]
    if not passed:
        raise RuntimeError(
            "analytic state-12 parity failed "
            f"({agreement:.4f} < {minimum_agreement:.4f}); "
            f"examples: {disagreements}"
        )
    return {
        "examples": len(matches),
        "agreement": agreement,
        "minimum_agreement": minimum_agreement,
        "passed": passed,
        "disagreements": disagreements,
    }


def collect(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    settings = cfg.endpoint_margin_geometry
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    if args.calibration_examples <= 0 or args.batch_size <= 0:
        raise ValueError("calibration examples and batch size must be positive")
    device = select_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("margin-geometry collection requires CUDA")
    dtype = resolve_torch_dtype(args.precision, device)
    token = os.environ.get("HF_TOKEN") or None
    checkpoint = args.checkpoint_path or download_official_checkpoint(
        repo_id=str(cfg.checkpoint.repo_id),
        revision=str(cfg.checkpoint.revision),
        filename=str(cfg.checkpoint.filename),
        expected_sha256=str(cfg.checkpoint.sha256),
        token=token,
    )
    model, tokenizer = build_official_codi_gpt2(
        base_model=str(cfg.model.base_model),
        base_revision=str(cfg.model.base_revision),
        dtype=dtype,
        settings=cfg.model,
        token=token,
    )
    load_report = load_official_checkpoint(
        model, checkpoint, expected_sha256=str(cfg.checkpoint.sha256)
    )

    from datasets import load_dataset

    train = load_dataset(
        "json",
        data_files={"train": GSM8K_TRAIN_URL},
        split="train",
        verification_mode="no_checks",
    )
    if len(train) != GSM8K_EXPECTED_TRAIN_EXAMPLES:
        raise RuntimeError("GSM8K train count drifted")
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    test = load_eval_set("gsm8k", data_cfg.eval.gsm8k)
    if len(test) != int(cfg.eval.expected_counts.gsm8k):
        raise RuntimeError("GSM8K test count drifted")
    test_questions = {_normalized_question(row["question"]) for row in test}
    if len(test_questions) != len(test):
        raise RuntimeError("GSM8K test contains duplicate normalized questions")
    calibration_rows, sampling = sample_gsm8k_train_calibration(
        train,
        test_questions=test_questions,
        examples=args.calibration_examples,
        seed=args.sampling_seed,
    )
    raw_test = load_dataset(
        "json",
        data_files={"test": GSM8K_TEST_URL},
        split="test",
        verification_mode="no_checks",
    )
    evaluation_rows, evaluation_join = canonical_gsm8k_test_rows(test, raw_test)
    if len(evaluation_rows) != len(test):
        raise RuntimeError("evaluation join dropped questions")

    request = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "phase": "answer_colon_state_cache_and_analytic_parity",
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "calibration_dataset": "openai/grade-school-math GSM8K train",
        "calibration_source_revision": GSM8K_SOURCE_REVISION,
        "calibration_source_url": GSM8K_TRAIN_URL,
        "calibration_examples": args.calibration_examples,
        "sampling_seed": args.sampling_seed,
        "sampling": sampling,
        "evaluation_dataset": "GSM8K test",
        "evaluation_source_url": GSM8K_TEST_URL,
        "evaluation_examples": len(evaluation_rows),
        "evaluation_join": evaluation_join,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "states": list(MARGIN_GEOMETRY_STATES),
        "endpoint": "student fixed teacher-forced answer-cue colon after EOT",
        "test_labels_used_for_calibration": False,
        "test_activations_used_for_calibration": False,
        "parity_examples": args.parity_examples,
        "minimum_parity_agreement": float(settings.minimum_parity_agreement),
    }
    request_sha256 = _sha256_json(request)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    states_path = output_dir / "colon_states.pt"
    readout_path = output_dir / "readout.pt"
    manifest_path = output_dir / "run_manifest.json"
    if states_path.is_file() and readout_path.is_file():
        payload = torch.load(states_path, map_location="cpu", weights_only=False)
        if payload.get("request_sha256") != request_sha256:
            raise RuntimeError("existing colon states belong to another request")
        print(f"[resume] already complete: {states_path}")
        return payload

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    scorer = OfficialCODIAnswerScorer(
        model, latent_positions=int(cfg.eval.latent_iterations)
    )
    # The released decoder takes its argmax over ``logits[:, :eot_id]``, so the
    # analytic tier must use exactly that restricted readout.
    eot_id = int(model.eot_id)
    readout = resolve_output_embedding(model)[:eot_id].detach().float().cpu()

    calibration_states = _collect_colon_states(
        scorer,
        tokenizer,
        calibration_rows,
        bot_token_id=model.bot_id,
        batch_size=args.batch_size,
        device=device,
        precision=args.precision,
        description="GSM8K-train colon states",
    )
    evaluation_states = _collect_colon_states(
        scorer,
        tokenizer,
        evaluation_rows,
        bot_token_id=model.bot_id,
        batch_size=args.batch_size,
        device=device,
        precision=args.precision,
        description="GSM8K-test colon states",
    )
    parity = _parity_gate(
        model,
        tokenizer,
        evaluation_rows[: args.parity_examples],
        evaluation_states[: args.parity_examples],
        readout,
        latent_iterations=int(cfg.eval.latent_iterations),
        batch_size=min(args.batch_size, args.parity_examples),
        device=device,
        minimum_agreement=float(settings.minimum_parity_agreement),
    )
    print(
        f"[parity] analytic/decoder first-token agreement "
        f"{parity['agreement']:.4f} on {parity['examples']} examples"
    )

    calibration_gold = torch.tensor(
        gold_first_token_ids(tokenizer, (row["answer"] for row in calibration_rows)),
        dtype=torch.long,
    )
    evaluation_gold = torch.tensor(
        gold_first_token_ids(tokenizer, (row["answer"] for row in evaluation_rows)),
        dtype=torch.long,
    )
    if int(calibration_gold.max()) >= eot_id or int(evaluation_gold.max()) >= eot_id:
        raise RuntimeError("a gold first token falls outside the decoder's vocabulary")

    mean = torch.zeros(GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE, dtype=torch.float32)
    for state in MARGIN_GEOMETRY_STATES:
        mean[state] = calibration_states[:, state, :].double().mean(dim=0).float()

    payload = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "request_sha256": request_sha256,
        "metadata": request,
        "parity_gate": parity,
        "eot_id": eot_id,
        "numeric_answer_token_ids": numeric_answer_token_ids(tokenizer),
        "student_mean": mean,
        "calibration_states": calibration_states[:, MARGIN_GEOMETRY_STATES, :],
        "calibration_gold_first_token": calibration_gold,
        "calibration_questions_sha256": _sha256_json(
            [_normalized_question(row["question"]) for row in calibration_rows]
        ),
        "evaluation_states": evaluation_states[:, MARGIN_GEOMETRY_STATES, :],
        "evaluation_gold_first_token": evaluation_gold,
        "evaluation_questions": [row["question"] for row in evaluation_rows],
        "evaluation_gold_answers": [row["answer"] for row in evaluation_rows],
        "state_order": list(MARGIN_GEOMETRY_STATES),
    }
    _atomic_torch_save(payload, states_path)
    _atomic_torch_save(
        {
            "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
            "contract": MARGIN_GEOMETRY_CONTRACT,
            "request_sha256": request_sha256,
            "eot_id": eot_id,
            "readout": readout,
        },
        readout_path,
    )
    _atomic_json(
        {
            **request,
            "request_sha256": request_sha256,
            "state": "complete",
            "parity_gate": parity,
            "states_file": states_path.name,
            "states_sha256": sha256_file(states_path),
            "readout_file": readout_path.name,
            "readout_sha256": sha256_file(readout_path),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        manifest_path,
    )
    print(f"[complete] wrote {states_path} and {readout_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--calibration-examples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=89)
    parser.add_argument("--parity-examples", type=int, default=64)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    collect(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
