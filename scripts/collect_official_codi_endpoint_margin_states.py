"""Cache answer-colon states and prove the analytic state-12 shortcut is exact.

States are captured *from the released generation path itself*, not re-derived with
the training encoder.  An earlier version did the latter and reached only 89%
first-token agreement, because the two paths differ in question normalisation, cue
tokenisation, and left-padding width (which shifts GPT-2's absolute position ids for
every row in a chunk).  Capturing during generation removes that class of divergence
by construction.

The parity gate then checks the one claim that remains: that GPT-2's ``lm_head`` is a
bias-free linear map of the ``ln_f`` output, so ``argmax(W h12)`` reproduces the token
the decoder actually emitted.  Nothing downstream may run until it passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import (
    _normalized_question,
    verify_full_reproduction_gate,
)
from scripts.collect_official_codi_parameter_state12_confirmation_stats import (
    GSM8K_EXPECTED_TRAIN_EXAMPLES,
    GSM8K_SOURCE_REVISION,
    GSM8K_TRAIN_URL,
    sample_gsm8k_train_calibration,
)
from src.data.datasets import load_eval_set
from src.data.official_codi_training import OFFICIAL_CODI_SOURCE_REVISION
from src.data.prompts import PromptStyle
from src.eval.official_codi import select_device
from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    MARGIN_GEOMETRY_CONTRACT,
    MARGIN_GEOMETRY_SCHEMA_VERSION,
    MARGIN_GEOMETRY_STATES,
    OfficialCODIEndpointStateCollector,
    gold_first_token_ids,
    numeric_answer_token_ids,
    resolve_output_embedding,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
    sha256_file,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def gold_text(value: object) -> str:
    """Render a gold answer in the surface form the released decoder emits.

    ``load_eval_set`` returns ``Decimal`` golds. GSM8K answers are integral, and an
    integral ``Decimal`` must render as ``"18"`` rather than ``"18.0"`` or ``"1E+1"``,
    because the first emitted token depends on the exact string.
    """
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return str(int(value))
        return format(value.normalize(), "f")
    return str(value).strip()


def collect_colon_states(
    model,
    tokenizer,
    questions,
    *,
    latent_iterations: int,
    batch_size: int,
    device: torch.device,
    answer_cue: str,
) -> tuple[torch.Tensor, list[str], dict]:
    """Capture the colon states the decoder itself consumes, plus its first token.

    ``max_new_tokens=1`` stops immediately after the forced-cue forward pass, which
    is the only pass this experiment intervenes on.  The answer never enters the
    model, so no reasoning trace or answer formatting is required at all.
    """
    collector = OfficialCODIEndpointStateCollector(model)
    generations, endpoint = generate_official_codi(
        model,
        tokenizer,
        list(questions),
        latent_iterations=latent_iterations,
        max_new_tokens=1,
        batch_size=batch_size,
        device=device,
        answer_endpoint_intervention=collector,
        answer_cue=answer_cue,
        force_answer_cue=True,
        return_endpoint_metadata=True,
    )
    if endpoint["endpoint_reached_count"] != len(generations):
        raise RuntimeError("the forced answer cue was not reached for every question")
    states = collector.stacked(len(generations))
    return states, generations, endpoint


def parity_gate(
    states: torch.Tensor,
    readout: torch.Tensor,
    generations,
    tokenizer,
    *,
    minimum_agreement: float,
) -> dict:
    """Check ``argmax(W h12)`` against the token the decoder actually emitted."""
    predicted = (states[:, ANALYTIC_STATE, :] @ readout.T).argmax(dim=-1)
    analytic = [
        tokenizer.decode([int(token)], skip_special_tokens=True)
        for token in predicted.tolist()
    ]
    matches = [bool(a == b) for a, b in zip(analytic, generations)]
    agreement = float(sum(matches) / max(1, len(matches)))
    disagreements = [
        {"index": index, "analytic": analytic[index], "generated": generations[index]}
        for index, match in enumerate(matches)
        if not match
    ][:8]
    if agreement < minimum_agreement:
        raise RuntimeError(
            "analytic state-12 parity failed "
            f"({agreement:.4f} < {minimum_agreement:.4f}); examples: {disagreements}"
        )
    return {
        "examples": len(matches),
        "agreement": agreement,
        "minimum_agreement": minimum_agreement,
        "passed": True,
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
    style = PromptStyle.from_config(data_cfg.prompt)
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

    request = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "phase": "answer_colon_state_cache_and_analytic_parity",
        "collection_path": "released forced-cue generation with a capturing hook",
        "checkpoint_sha256": load_report.checkpoint_sha256,
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "answer_cue": style.answer_prefix,
        "calibration_dataset": "openai/grade-school-math GSM8K train",
        "calibration_source_revision": GSM8K_SOURCE_REVISION,
        "calibration_source_url": GSM8K_TRAIN_URL,
        "calibration_examples": args.calibration_examples,
        "sampling_seed": args.sampling_seed,
        "sampling": sampling,
        "evaluation_dataset": "GSM8K test",
        "evaluation_examples": len(test),
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
    # The released decoder takes its argmax over ``logits[:, :eot_id]``, so the
    # analytic tier must use exactly that restricted readout.
    eot_id = int(model.eot_id)
    readout = resolve_output_embedding(model)[:eot_id].detach().float().cpu()

    calibration_states, _, _ = collect_colon_states(
        model,
        tokenizer,
        [row["question"] for row in calibration_rows],
        latent_iterations=int(cfg.eval.latent_iterations),
        batch_size=args.batch_size,
        device=device,
        answer_cue=style.answer_prefix,
    )
    evaluation_states, evaluation_first_tokens, endpoint = collect_colon_states(
        model,
        tokenizer,
        [row["question"] for row in test],
        latent_iterations=int(cfg.eval.latent_iterations),
        batch_size=args.batch_size,
        device=device,
        answer_cue=style.answer_prefix,
    )
    parity = parity_gate(
        evaluation_states[: args.parity_examples],
        readout,
        evaluation_first_tokens[: args.parity_examples],
        tokenizer,
        minimum_agreement=float(settings.minimum_parity_agreement),
    )
    print(
        f"[parity] analytic/decoder first-token agreement "
        f"{parity['agreement']:.4f} on {parity['examples']} examples"
    )
    # The gate samples a prefix; the identity should hold for every question, so the
    # full-set agreement is recorded too and must not be worse.
    full_parity = parity_gate(
        evaluation_states,
        readout,
        evaluation_first_tokens,
        tokenizer,
        minimum_agreement=float(settings.minimum_parity_agreement),
    )
    print(f"[parity] full evaluation agreement {full_parity['agreement']:.4f}")

    calibration_gold = torch.tensor(
        gold_first_token_ids(
            tokenizer, (gold_text(row["answer"]) for row in calibration_rows)
        ),
        dtype=torch.long,
    )
    evaluation_gold = torch.tensor(
        gold_first_token_ids(tokenizer, (gold_text(row["gold"]) for row in test)),
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
        "full_parity_gate": full_parity,
        "endpoint_coverage": {
            key: value for key, value in endpoint.items() if key != "endpoint_reached"
        },
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
        "evaluation_questions": [row["question"] for row in test],
        "evaluation_gold_answers": [gold_text(row["gold"]) for row in test],
        "evaluation_first_tokens": evaluation_first_tokens,
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
            "full_parity_gate": full_parity,
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
