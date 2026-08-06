"""Fair retained-subspace training targets for the three CODI endpoint selectors."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from src.mech.endpoint_answer_conditioned import (
    GPT2_HIDDEN_SIZE,
    GPT2_STATE_COUNT,
    answer_conditioned_bases_from_state,
    validate_answer_conditioned_bases,
)
from src.mech.endpoint_parameter_aware import (
    parameter_aware_bases_from_state,
    validate_parameter_aware_bases,
)
from src.mech.endpoint_tsvc import bases_from_state, validate_endpoint_bases
from src.models.official_codi import sha256_file


RETENTION_SCHEMA_VERSION = 1
RETENTION_CONTRACT = "endpoint_selector_rank_matched_retention_v1"
RETENTION_METHODS = ("energy", "answer_conditioned", "parameter_aware")
RETENTION_COMMON_STATES = (11, 12)
RETENTION_COMMON_RANK = 3
RETENTION_TRAINING_ARMS = (
    "answer_only",
    "full_common",
    "energy_selected",
    "answer_conditioned_selected",
    "parameter_aware_selected",
    "energy_complement",
    "answer_conditioned_complement",
    "parameter_aware_complement",
)


@dataclass(frozen=True)
class RetentionBasis:
    name: str
    basis: torch.Tensor
    ranks: torch.Tensor
    source_path: str
    source_sha256: str
    source_request_sha256: str
    source_contract: str
    selected_pc_indices: torch.Tensor | None = None


def _artifact_metadata(path: Path, payload: Mapping) -> dict:
    """Merge immutable collection sidecars used by the older corrected export."""
    metadata = dict(payload.get("metadata", {}))
    manifest_path = path.parent / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") not in {None, "complete"}:
            raise RuntimeError(f"basis collection is not complete: {manifest_path}")
        for key, value in manifest.items():
            metadata.setdefault(key, value)
    parity_path = path.parent / "native_loss_gradient_parity.json"
    if parity_path.is_file() and "native_parity_gate" not in metadata:
        metadata["native_parity_gate"] = json.loads(
            parity_path.read_text(encoding="utf-8")
        )
    return metadata


def _rank_matched_basis(
    source: torch.Tensor,
    *,
    states: Sequence[int],
    rank_per_state: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.ndim != 3 or source.shape[:2] != (
        GPT2_STATE_COUNT,
        GPT2_HIDDEN_SIZE,
    ):
        raise ValueError("source basis must have shape [13,768,R]")
    if rank_per_state <= 0 or source.shape[-1] < rank_per_state:
        raise ValueError("source basis does not contain the requested common rank")
    resolved_states = tuple(int(value) for value in states)
    if not resolved_states or any(
        value <= 0 or value >= GPT2_STATE_COUNT for value in resolved_states
    ):
        raise ValueError("common states must be contextual GPT-2 states")
    basis = torch.zeros(
        GPT2_STATE_COUNT,
        GPT2_HIDDEN_SIZE,
        rank_per_state,
        dtype=torch.float32,
    )
    ranks = torch.zeros(GPT2_STATE_COUNT, dtype=torch.int64)
    for state in resolved_states:
        basis[state] = source[state, :, :rank_per_state].float()
        ranks[state] = rank_per_state
    return basis, ranks


def validate_retention_basis(
    value: RetentionBasis,
    *,
    common_states: Sequence[int] = RETENTION_COMMON_STATES,
    common_rank: int = RETENTION_COMMON_RANK,
) -> None:
    if value.name not in RETENTION_METHODS:
        raise ValueError("unknown endpoint-retention method")
    if value.basis.shape != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE, common_rank):
        raise ValueError("retention basis shape is inconsistent")
    if value.ranks.shape != (GPT2_STATE_COUNT,):
        raise ValueError("retention ranks must have shape [13]")
    expected = torch.zeros_like(value.ranks)
    expected[list(common_states)] = int(common_rank)
    if not torch.equal(value.ranks.cpu(), expected):
        raise ValueError("retention basis escaped the registered common states/rank")
    for state in common_states:
        active = value.basis[int(state), :, :common_rank].double()
        gram = active.T @ active
        if not torch.allclose(
            gram, torch.eye(common_rank, dtype=torch.float64), atol=2e-5, rtol=2e-5
        ):
            raise ValueError("retention basis is not orthonormal")
    if not all(
        isinstance(item, str) and item
        for item in (
            value.source_path,
            value.source_sha256,
            value.source_request_sha256,
            value.source_contract,
        )
    ):
        raise ValueError("retention basis source identity is incomplete")
    if value.selected_pc_indices is not None and value.selected_pc_indices.shape != (
        GPT2_STATE_COUNT,
        common_rank,
    ):
        raise ValueError("retention selected-PC identities have the wrong shape")


def load_retention_bases(
    *,
    energy_path: str | Path,
    answer_conditioned_path: str | Path,
    parameter_aware_path: str | Path,
    checkpoint_sha256: str,
    common_states: Sequence[int] = RETENTION_COMMON_STATES,
    common_rank: int = RETENTION_COMMON_RANK,
) -> dict[str, RetentionBasis]:
    """Load and rank-match the three immutable completed selector artifacts."""
    paths = {
        "energy": Path(energy_path),
        "answer_conditioned": Path(answer_conditioned_path),
        "parameter_aware": Path(parameter_aware_path),
    }
    payloads = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} basis does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = _artifact_metadata(path, payload)
        payload = dict(payload)
        payload["metadata"] = metadata
        if metadata.get("checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError(f"{name} basis uses a different CODI checkpoint")
        if not isinstance(payload.get("request_sha256"), str):
            raise RuntimeError(f"{name} basis request identity is missing")
        if metadata.get("native_parity_gate", {}).get("status") != "passed":
            raise RuntimeError(f"{name} basis lacks a passed native endpoint parity gate")
        payloads[name] = payload

    energy_payload = payloads["energy"]
    energy_metadata = energy_payload.get("metadata", {})
    if energy_metadata.get("contract") != (
        "source_faithful_student_and_teacher_answer_colon_v2"
    ):
        raise RuntimeError("energy basis is not the corrected seed-11 artifact")
    if int(energy_metadata.get("calibration_examples", -1)) != 5_000:
        raise RuntimeError("energy basis is not the completed 5,000-example fit")
    energy = bases_from_state(energy_payload["bases"])
    validate_endpoint_bases(
        energy, layers=GPT2_STATE_COUNT, hidden_size=GPT2_HIDDEN_SIZE
    )

    answer_payload = payloads["answer_conditioned"]
    answer_metadata = answer_payload.get("metadata", {})
    if answer_metadata.get("contract") != (
        "answer_conditioned_colon_block_states_v1"
    ):
        raise RuntimeError("answer-conditioned basis contract changed")
    if (
        int(answer_metadata.get("residual_fit_examples", -1)) != 1_024
        or int(answer_metadata.get("direction_selection_examples", -1)) != 1_024
    ):
        raise RuntimeError("answer-conditioned basis is not the completed full fit")
    if answer_payload.get("selection", {}).get("status") != "candidate_selected":
        raise RuntimeError("answer-conditioned artifact has no selected candidate")
    answer = answer_conditioned_bases_from_state(answer_payload["bases"])
    validate_answer_conditioned_bases(
        answer, states=GPT2_STATE_COUNT, hidden_size=GPT2_HIDDEN_SIZE,
        require_candidate=True,
    )

    parameter_payload = payloads["parameter_aware"]
    parameter_metadata = parameter_payload.get("metadata", {})
    if parameter_metadata.get("contract") != (
        "parameter_aware_colon_final_two_blocks_v1"
    ):
        raise RuntimeError("parameter-aware basis contract changed")
    if (
        int(parameter_metadata.get("residual_fit_examples", -1)) != 1_024
        or int(parameter_metadata.get("direction_selection_examples", -1)) != 1_024
    ):
        raise RuntimeError("parameter-aware basis is not the completed full fit")
    if parameter_payload.get("selection", {}).get("status") != "candidate_selected":
        raise RuntimeError("parameter-aware artifact has no selected candidate")
    parameter = parameter_aware_bases_from_state(parameter_payload["bases"])
    validate_parameter_aware_bases(
        parameter, states=GPT2_STATE_COUNT, hidden_size=GPT2_HIDDEN_SIZE,
        require_candidate=True,
    )

    source_tensors = {
        "energy": energy.top,
        "answer_conditioned": answer.answer_conditioned,
        "parameter_aware": parameter.parameter_aware,
    }
    source_pc_indices = {
        "energy": torch.arange(common_rank, dtype=torch.int64)
        .unsqueeze(0)
        .expand(GPT2_STATE_COUNT, -1)
        .clone(),
        "answer_conditioned": answer.selected_pc_indices[:, :common_rank].long(),
        "parameter_aware": parameter.selected_pc_indices[:, :common_rank].long(),
    }
    results = {}
    for name, source in source_tensors.items():
        basis, ranks = _rank_matched_basis(
            source, states=common_states, rank_per_state=common_rank
        )
        payload = payloads[name]
        path = paths[name]
        value = RetentionBasis(
            name=name,
            basis=basis,
            ranks=ranks,
            source_path=str(path),
            source_sha256=sha256_file(path),
            source_request_sha256=str(payload["request_sha256"]),
            source_contract=str(payload["metadata"]["contract"]),
            selected_pc_indices=source_pc_indices[name].clone(),
        )
        validate_retention_basis(
            value, common_states=common_states, common_rank=common_rank
        )
        results[name] = value
    return results


def retention_bases_state(values: Mapping[str, RetentionBasis]) -> dict:
    if set(values) != set(RETENTION_METHODS):
        raise ValueError("all three retention methods are required")
    result = {}
    for name in RETENTION_METHODS:
        value = values[name]
        validate_retention_basis(value)
        result[name] = {
            "basis": value.basis.detach().cpu(),
            "ranks": value.ranks.detach().cpu(),
            "source_path": value.source_path,
            "source_sha256": value.source_sha256,
            "source_request_sha256": value.source_request_sha256,
            "source_contract": value.source_contract,
            "selected_pc_indices": (
                None
                if value.selected_pc_indices is None
                else value.selected_pc_indices.tolist()
            ),
        }
    return result


def retention_basis_for_arm(
    arm: str, values: Mapping[str, RetentionBasis]
) -> tuple[RetentionBasis | None, str]:
    if arm == "answer_only":
        return None, "none"
    if arm == "full_common":
        return None, "full"
    suffixes = {"_selected": "projected", "_complement": "complement"}
    for suffix, mode in suffixes.items():
        if arm.endswith(suffix):
            name = arm[: -len(suffix)]
            if name not in values:
                raise ValueError(f"missing retention basis {name!r}")
            return values[name], mode
    raise ValueError(f"unknown retention arm {arm!r}")


def endpoint_retention_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    mode: str,
    basis: torch.Tensor | None = None,
    ranks: torch.Tensor | None = None,
    states: Sequence[int] = RETENTION_COMMON_STATES,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Native SmoothL1/std loss after retaining or removing a frozen subspace."""
    if student.shape != teacher.shape or student.shape[1:] != (
        GPT2_STATE_COUNT,
        GPT2_HIDDEN_SIZE,
    ):
        raise ValueError("retention endpoints must have shape [B,13,768]")
    resolved_states = tuple(int(value) for value in states)
    target = teacher.detach()
    residual = student - target
    if mode == "full":
        filtered = residual
    else:
        if basis is None or ranks is None:
            raise ValueError(f"mode {mode!r} requires a basis and ranks")
        resolved_basis = basis.to(device=residual.device, dtype=residual.dtype)
        projected = torch.zeros_like(residual)
        for state in resolved_states:
            rank = int(ranks[state])
            if rank <= 0:
                raise ValueError("every common state must have positive retained rank")
            active = resolved_basis[state, :, :rank]
            projected[:, state, :] = (residual[:, state, :] @ active) @ active.T
        if mode == "projected":
            filtered = projected
        elif mode == "complement":
            filtered = residual - projected
        else:
            raise ValueError("mode must be full, projected, or complement")
    losses = []
    for state in resolved_states:
        value = F.smooth_l1_loss(
            filtered[:, state, :],
            torch.zeros_like(filtered[:, state, :]),
            reduction="mean",
            beta=1.0,
        )
        scale = target[:, state, :].std(unbiased=True).clamp_min(eps)
        losses.append(value / scale)
    return torch.stack(losses).mean()
