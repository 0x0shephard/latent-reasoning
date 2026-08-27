"""Collect CODI's latent-trajectory hidden states for the detection gate.

One observational GPU pass over the cached 1,319 GSM8K-test questions. The
released generation path runs unmodified with two pure observers attached: the
trajectory capture commits the thirteen hidden states of each of the six latent
iterations, and a zero-noise endpoint capture records the forced-cue state 12 so
the new pass can be tied to the validated colon-state cache by direct state
parity rather than by trust.

No weight is updated, no state is edited, and no noisy or resampled variant
exists in this collection.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_kv_subspaces import _atomic_json, _atomic_torch_save
from scripts.collect_official_codi_endpoint_tsvc import verify_full_reproduction_gate
from scripts.run_official_codi_correctness_detect_replication import split_test_like
from scripts.run_official_codi_endpoint_margin_sweep import load_margin_cache
from src.data.official_codi_training import OFFICIAL_CODI_SOURCE_REVISION
from src.data.prompts import PromptStyle
from src.mech.endpoint_correctness_geometry import readout_matrix
from src.mech.endpoint_margin_geometry import ANALYTIC_STATE
from src.mech.endpoint_paired_correction import OfficialCODIPerturbAndCapture
from src.mech.latent_trajectory_detect import (
    LATENT_TRAJECTORY_CONTRACT,
    LATENT_TRAJECTORY_SCHEMA_VERSION,
    TRAJECTORY_STATES,
    OfficialCODILatentTrajectoryCapture,
)
from src.models.official_codi import (
    build_official_codi_gpt2,
    download_official_checkpoint,
    generate_official_codi,
    load_official_checkpoint,
    resolve_torch_dtype,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_collection_batch_size(cache_metadata: dict, cli_value) -> int:
    """The cache's own collection batch size, never a fresh choice.

    GPT-2's absolute position ids depend on each chunk's left-padding width, so
    colon states are only reproducible under the exact chunking that produced
    them.  The state-parity gate therefore requires collecting at the batch size
    recorded in the cache; a CLI value is honoured only when the cache predates
    that record, and rejected when it contradicts it.
    """
    recorded = cache_metadata.get("batch_size")
    if recorded is not None:
        if cli_value is not None and int(cli_value) != int(recorded):
            raise ValueError(
                f"--batch-size {cli_value} contradicts the cache's recorded "
                f"collection batch size {recorded}; omit the flag"
            )
        return int(recorded)
    if cli_value is None:
        raise ValueError(
            "the cache records no collection batch size; pass --batch-size "
            "matching the original collection"
        )
    return int(cli_value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/official_codi_gpt2.yaml"
    )
    parser.add_argument("--reproduction-summary", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--precision", default="float32")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Only for caches that predate the recorded collection batch size; "
        "the cache's own record is otherwise authoritative.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    cfg = load_config(args.config)
    settings = cfg.latent_trajectory_detect
    if str(cfg.official_source.revision) != OFFICIAL_CODI_SOURCE_REVISION:
        raise RuntimeError("official CODI source revision changed")
    reproduction = verify_full_reproduction_gate(args.reproduction_summary, cfg)
    cache, readout_payload = load_margin_cache(args.states, args.readout)
    batch_size = resolve_collection_batch_size(cache["metadata"], args.batch_size)
    print(f"[chunking] collection batch size {batch_size} (from the cache record)")
    readout = readout_matrix(readout_payload).double()
    state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
    colon_states = cache["evaluation_states"][:, state_index, :].double()
    questions = list(cache["evaluation_questions"])
    expected = int(settings.expected_examples)
    if colon_states.shape != (expected, 768) or len(questions) != expected:
        raise RuntimeError("cached GSM8K evaluation population changed")

    fit_index, select_index, test_index = split_test_like(
        expected,
        seed=int(settings.split_seed),
        fit_size=int(settings.fit_examples),
        select_size=int(settings.select_examples),
    )
    indices = {
        "fit": fit_index.tolist(),
        "select": select_index.tolist(),
        "test": test_index.tolist(),
    }
    partition_sha256 = _sha256_json(indices)
    if partition_sha256 != str(settings.expected_partition_sha256):
        raise RuntimeError(
            "the deterministic partition drifted from the frozen population"
        )
    latent_iterations = int(cfg.eval.latent_iterations)
    request = {
        "schema_version": LATENT_TRAJECTORY_SCHEMA_VERSION,
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "phase": "latent_trajectory_observation",
        "checkpoint_sha256": cache["metadata"]["checkpoint_sha256"],
        "source_request_sha256": cache.get("request_sha256"),
        "official_source_revision": str(cfg.official_source.revision),
        "reproduction_gate": reproduction,
        "population": "GSM8K test",
        "indices": indices,
        "partition_sha256": partition_sha256,
        "latent_iterations": latent_iterations,
        "trajectory_states": TRAJECTORY_STATES,
        "intervention": None,
        "precision": args.precision,
        "batch_size": batch_size,
    }
    request_sha256 = _sha256_json(request)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "latent_trajectory.pt"
    summary_path = args.output_dir / "latent_trajectory.json"
    if output_path.is_file() and summary_path.is_file():
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        if existing.get("request_sha256") != request_sha256:
            raise RuntimeError("existing trajectory export belongs to another request")
        print(f"[resume] already complete: {output_path}")
        return 0

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("trajectory collection requires a Kaggle GPU")
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
    if load_report.checkpoint_sha256 != request["checkpoint_sha256"]:
        raise RuntimeError("cached states and live checkpoint differ")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=device, dtype=dtype).eval()
    data_cfg = load_config(str(cfg.endpoint_retention.data_config))
    answer_cue = PromptStyle.from_config(data_cfg.prompt).answer_prefix

    trajectory = OfficialCODILatentTrajectoryCapture(
        model, latent_iterations=latent_iterations
    )
    endpoint_observer = OfficialCODIPerturbAndCapture(
        model, relative_noise=0.0, seed=0
    )
    try:
        generations, endpoint = generate_official_codi(
            model,
            tokenizer,
            questions,
            latent_iterations=latent_iterations,
            max_new_tokens=1,
            batch_size=batch_size,
            device=device,
            kv_intervention=trajectory,
            answer_endpoint_intervention=endpoint_observer,
            answer_cue=answer_cue,
            force_answer_cue=True,
            return_endpoint_metadata=True,
        )
    finally:
        trajectory.close()
    if endpoint["endpoint_reached_count"] != expected:
        raise RuntimeError("not every question reached the forced answer cue")

    states = trajectory.stacked(expected)
    live_colon = endpoint_observer.stacked(expected).double()
    gold = cache["evaluation_gold_first_token"].long()

    # Gate 1 — analytic parity on the *live* pass: the captured state 12 must
    # reproduce the token the decoder actually emitted, the same check that
    # validated the original cache on its own pass.
    predicted = (live_colon @ readout.T).argmax(dim=-1)
    analytic_tokens = [
        tokenizer.decode([int(token)], skip_special_tokens=True)
        for token in predicted.tolist()
    ]
    matches = [bool(a == b) for a, b in zip(analytic_tokens, generations)]
    agreement = float(sum(matches) / max(1, len(matches)))
    analytic_parity = {
        "examples": len(matches),
        "agreement": agreement,
        "minimum_agreement": float(settings.analytic_parity_minimum_agreement),
        "passed": bool(agreement >= float(settings.analytic_parity_minimum_agreement)),
    }
    if not analytic_parity["passed"]:
        raise RuntimeError(
            f"live analytic parity failed: agreement {agreement:.4f} below "
            f"{analytic_parity['minimum_agreement']:g}"
        )

    # Gate 2 — accuracy reproduction: the live pass must behave like the cached
    # population at the aggregate level even though the cache's exact states
    # predate the environment pins (ledger §40 remaining limits) and are not
    # reproducible vector-for-vector.
    live_accuracy = float((predicted == gold).double().mean())
    cache_accuracy = float(
        ((colon_states @ readout.T).argmax(dim=-1) == gold).double().mean()
    )
    accuracy_gate = {
        "live_first_token_accuracy": live_accuracy,
        "cache_first_token_accuracy": cache_accuracy,
        "drift": abs(live_accuracy - cache_accuracy),
        "allowance": float(settings.accuracy_drift_allowance),
        "passed": bool(
            abs(live_accuracy - cache_accuracy)
            <= float(settings.accuracy_drift_allowance)
        ),
    }
    if not accuracy_gate["passed"]:
        raise RuntimeError(
            f"live accuracy {live_accuracy:.4f} drifted "
            f"{accuracy_gate['drift']:.4f} from the cache's {cache_accuracy:.4f}"
        )

    # Diagnostic only — how far the pre-pin cache states are from the live pass.
    deviation = (live_colon - colon_states).norm(dim=1)
    reference = colon_states.norm(dim=1).clamp_min(1e-8)
    relative = deviation / reference
    cache_first_tokens = (colon_states @ readout.T).argmax(dim=-1)
    tolerance = float(settings.cache_state_relative_tolerance)
    cache_state_diagnostic = {
        "relative_tolerance": tolerance,
        "rows_within_tolerance": int((relative <= tolerance).sum()),
        "median_relative_deviation": float(relative.median()),
        "max_relative_deviation": float(relative.max()),
        "first_token_agreement_with_cache": float(
            (predicted == cache_first_tokens).double().mean()
        ),
    }
    parity_gate = {
        "recaptured_state": ANALYTIC_STATE,
        "analytic_parity": analytic_parity,
        "accuracy_gate": accuracy_gate,
        "cache_state_diagnostic": cache_state_diagnostic,
        "passed": True,
    }

    payload = {
        "schema_version": LATENT_TRAJECTORY_SCHEMA_VERSION,
        "contract": LATENT_TRAJECTORY_CONTRACT,
        "request_sha256": request_sha256,
        "source_request_sha256": cache.get("request_sha256"),
        "partition_sha256": partition_sha256,
        "indices": indices,
        "parity_gate": parity_gate,
        "trajectory_states": states,
        "endpoint_states": live_colon.float(),
        "live_first_token": predicted.cpu(),
    }
    _atomic_torch_save(payload, output_path)
    _atomic_json(
        {
            **request,
            "request_sha256": request_sha256,
            "parity_gate": parity_gate,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_file": str(output_path),
            "state": "complete",
        },
        summary_path,
    )
    print(
        f"[complete] trajectory {tuple(states.shape)}, analytic agreement "
        f"{agreement:.4f}, live accuracy {live_accuracy:.4f} "
        f"(cache {cache_accuracy:.4f}) -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
