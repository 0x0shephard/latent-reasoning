"""Closed-form state-12 subspace sweep over rank, family, mode and semantics.

Every arm here is evaluated from cached colon states, so hundreds of matched
controls and a full rank grid cost one model-free pass instead of one full greedy
decode each.  The outcomes are continuous, which is the specific deficiency that
made the completed rank-three confirmation unable to separate a real effect from
its matched-random null.
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
from src.eval.official_codi import select_device
from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE
from src.mech.endpoint_margin_geometry import (
    ANALYTIC_STATE,
    MARGIN_GEOMETRY_CONTRACT,
    MARGIN_GEOMETRY_SCHEMA_VERSION,
    MarginSubspace,
    build_margin_arm_registry,
    deterministic_derangement,
    evaluate_subspace_analytically,
    margin_damage_matrix,
    state_covariance,
    validate_margin_subspace,
)
from src.mech.endpoint_retention import (
    RETENTION_COMMON_RANK,
    load_retention_bases,
    retention_bases_state,
)
from src.utils.config import load_config


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _chunked_logits(
    hidden: torch.Tensor, readout: torch.Tensor, chunk_size: int
):
    for start in range(0, hidden.shape[0], chunk_size):
        block = hidden[start : start + chunk_size]
        yield start, block, block @ readout.T


def calibration_gradients(
    hidden: torch.Tensor,
    readout: torch.Tensor,
    gold: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Per-example margin and answer-NLL gradients at the colon.

    The margin gradient is exact: with the runner-up held fixed the margin is a
    linear functional of the state.  The NLL gradient is the ordinary first-order
    one, which is why the two families are reported separately rather than pooled.
    """
    margin_gradients = torch.zeros_like(hidden)
    nll_gradients = torch.zeros_like(hidden)
    for start, _, logits in _chunked_logits(hidden, readout, chunk_size):
        rows = torch.arange(logits.shape[0], device=logits.device)
        target = gold[start : start + logits.shape[0]].to(logits.device)
        masked = logits.clone()
        masked[rows, target] = torch.finfo(logits.dtype).min
        runner_up = masked.argmax(dim=-1)
        margin_gradients[start : start + logits.shape[0]] = (
            readout[target] - readout[runner_up]
        )
        probabilities = torch.softmax(logits.double(), dim=-1)
        probabilities[rows, target] -= 1.0
        nll_gradients[start : start + logits.shape[0]] = (
            probabilities @ readout.double()
        ).float()
    return {"margin": margin_gradients, "answer_nll": nll_gradients}


def _reference_subspaces(
    args, checkpoint_sha256: str
) -> tuple[dict[str, MarginSubspace], dict]:
    """Load the two completed selectors at state 12, rank three, unchanged."""
    bases = load_retention_bases(
        energy_path=args.energy_basis,
        answer_conditioned_path=args.answer_conditioned_basis,
        parameter_aware_path=args.parameter_aware_basis,
        checkpoint_sha256=checkpoint_sha256,
    )
    references: dict[str, MarginSubspace] = {}
    for family in ("answer_conditioned", "parameter_aware"):
        source = bases[family]
        basis = source.basis[ANALYTIC_STATE, :, :RETENTION_COMMON_RANK]
        subspace = MarginSubspace(
            name=f"{family}_k{RETENTION_COMMON_RANK:03d}_s{ANALYTIC_STATE}",
            family=family,
            state=ANALYTIC_STATE,
            basis=basis.detach().cpu().float().clone(),
            rank=RETENTION_COMMON_RANK,
        )
        validate_margin_subspace(subspace)
        references[family] = subspace
    return references, retention_bases_state(bases)


def _arm_plan(settings, registry) -> list[tuple[str, str, str]]:
    """(subspace name, mode, semantics) triples this sweep evaluates."""
    plan: list[tuple[str, str, str]] = []
    semantics_ranks = {int(rank) for rank in settings.semantics_ranks}
    for name, subspace in registry.items():
        plan.append((name, "remove", "mean"))
        plan.append((name, "retain", "mean"))
        if subspace.rank in semantics_ranks:
            plan.append((name, "remove", "zero"))
            plan.append((name, "remove", "resample"))
    return plan


def load_margin_cache(states_path: Path, readout_path: Path) -> tuple[dict, dict]:
    """Load the colon-state cache and refuse any run whose parity gate failed."""
    cache = torch.load(states_path, map_location="cpu", weights_only=False)
    readout_payload = torch.load(readout_path, map_location="cpu", weights_only=False)
    if cache.get("contract") != MARGIN_GEOMETRY_CONTRACT:
        raise RuntimeError("colon-state cache belongs to another contract")
    if readout_payload.get("request_sha256") != cache.get("request_sha256"):
        raise RuntimeError("readout matrix and colon states are from different runs")
    if not cache["parity_gate"]["passed"]:
        raise RuntimeError("the analytic parity gate did not pass; the sweep is invalid")
    return cache, readout_payload


def prepare_registry(
    cache: dict,
    readout_payload: dict,
    settings,
    args: argparse.Namespace,
    device: torch.device,
    *,
    chunk_size: int = 64,
    random_replicates: int | None = None,
) -> dict:
    """Rebuild every subspace deterministically from the cached colon states.

    Both the analytic sweep and the generation-confirmation runner call this, so
    an arm name always denotes the same basis in both tiers without shipping
    thousands of bases through the export.
    """
    state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
    calibration = cache["calibration_states"][:, state_index, :].to(device)
    readout = readout_payload["readout"].to(device)
    mean = cache["student_mean"][ANALYTIC_STATE].to(device)
    if calibration.shape[0] <= GPT2_HIDDEN_SIZE:
        # A rank-deficient calibration covariance has near-zero eigenvalues, which
        # the covariance-shaped random sampler raises to a negative power while
        # matching energy.  Refuse rather than emit unstable control bases.
        raise RuntimeError(
            f"calibration needs more than {GPT2_HIDDEN_SIZE} examples for a "
            f"full-rank covariance, found {calibration.shape[0]}"
        )
    centered = calibration - mean.unsqueeze(0)
    covariance = state_covariance(centered.cpu())
    gradients = calibration_gradients(
        calibration, readout, cache["calibration_gold_first_token"], chunk_size=chunk_size
    )
    damage = {
        family: margin_damage_matrix(centered.cpu(), value.cpu())
        for family, value in gradients.items()
    }
    numeric_ids = torch.tensor(cache["numeric_answer_token_ids"], dtype=torch.long)
    references, reference_state = _reference_subspaces(
        args, str(cache["metadata"]["checkpoint_sha256"])
    )
    registry = build_margin_arm_registry(
        covariance=covariance,
        damage_matrices=damage,
        readout_matrix=readout[numeric_ids.to(device)].cpu(),
        reference_subspaces=references,
        rank_grid=[int(value) for value in settings.rank_grid],
        random_replicates=(
            int(settings.random_replicates)
            if random_replicates is None
            else int(random_replicates)
        ),
        random_seed=int(settings.random_seed),
        primary_rank=int(settings.primary_rank),
    )
    return {
        "registry": registry,
        "covariance": covariance,
        "mean": mean,
        "readout": readout,
        "reference_basis_state": reference_state,
        "calibration_examples": int(calibration.shape[0]),
    }


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    cfg = load_config(args.config)
    settings = cfg.endpoint_margin_geometry
    device = select_device(args.device)
    cache, readout_payload = load_margin_cache(args.states, args.readout)

    state_index = list(cache["state_order"]).index(ANALYTIC_STATE)
    evaluation = cache["evaluation_states"][:, state_index, :].to(device)
    evaluation_gold = cache["evaluation_gold_first_token"].to(device)
    prepared = prepare_registry(
        cache, readout_payload, settings, args, device, chunk_size=args.chunk_size
    )
    registry = prepared["registry"]
    readout = prepared["readout"]
    mean = prepared["mean"]
    covariance = prepared["covariance"]
    reference_state = prepared["reference_basis_state"]
    plan = _arm_plan(settings, registry)

    request = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "phase": "analytic_state12_subspace_sweep",
        "states_request_sha256": cache["request_sha256"],
        "checkpoint_sha256": cache["metadata"]["checkpoint_sha256"],
        "parity_gate": cache["parity_gate"],
        "rank_grid": [int(value) for value in settings.rank_grid],
        "semantics_ranks": [int(value) for value in settings.semantics_ranks],
        "random_replicates": int(settings.random_replicates),
        "random_seed": int(settings.random_seed),
        "primary_rank": int(settings.primary_rank),
        "resample_seed": int(settings.resample_seed),
        "calibration_examples": int(prepared["calibration_examples"]),
        "evaluation_examples": int(evaluation.shape[0]),
        "reference_basis_sources": {
            name: {
                key: value
                for key, value in state.items()
                if key not in {"basis", "ranks"}
            }
            for name, state in reference_state.items()
        },
        "arms": len(plan),
    }
    request_sha256 = _sha256_json(request)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "analytic_sweep.pt"
    manifest_path = output_dir / "run_manifest.json"
    if results_path.is_file():
        payload = torch.load(results_path, map_location="cpu", weights_only=False)
        if payload.get("request_sha256") == request_sha256:
            print(f"[resume] already complete: {results_path}")
            return payload
        raise RuntimeError("existing sweep results belong to another request")

    derangement = deterministic_derangement(
        int(evaluation.shape[0]), int(settings.resample_seed)
    )
    resample_source = evaluation[derangement.to(evaluation.device)]

    baseline = evaluate_subspace_analytically(
        hidden=evaluation,
        readout=readout,
        gold_token=evaluation_gold,
        mean=mean,
        basis=None,
        chunk_size=args.chunk_size,
    )
    arms: dict[str, dict] = {}
    progress = tqdm(total=len(plan), unit="arm", desc="Analytic state-12 sweep")
    for name, mode, semantics in plan:
        subspace = registry[name]
        outcome = evaluate_subspace_analytically(
            hidden=evaluation,
            readout=readout,
            gold_token=evaluation_gold,
            mean=mean,
            basis=subspace.basis.to(device),
            mode=mode,
            semantics=semantics,
            resample_source=resample_source,
            chunk_size=args.chunk_size,
        )
        outcome["subspace"] = subspace.state_dict()
        arms[f"{name}|{mode}|{semantics}"] = outcome
        progress.update(1)
    progress.close()

    payload = {
        "schema_version": MARGIN_GEOMETRY_SCHEMA_VERSION,
        "contract": MARGIN_GEOMETRY_CONTRACT,
        "request_sha256": request_sha256,
        "metadata": request,
        "baseline": baseline,
        "arms": arms,
        "calibration_covariance_trace": float(torch.diagonal(covariance).sum()),
        "evaluation_gold_first_token": evaluation_gold.cpu(),
    }
    _atomic_torch_save(payload, results_path)
    _atomic_json(
        {
            **request,
            "request_sha256": request_sha256,
            "state": "complete",
            "baseline_first_token_accuracy": float(
                baseline["top1_correct"].double().mean()
            ),
            "baseline_mean_gold_nll": float(baseline["nll"].mean()),
            "results_file": results_path.name,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        manifest_path,
    )
    print(
        f"[complete] {len(arms)} arms; baseline first-token accuracy "
        f"{float(baseline['top1_correct'].double().mean()):.6f}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_codi_gpt2.yaml"))
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--energy-basis", type=Path, required=True)
    parser.add_argument("--answer-conditioned-basis", type=Path, required=True)
    parser.add_argument("--parameter-aware-basis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
