"""Frozen-checkpoint causal ablations at CODI's fixed answer-cue endpoint."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping

import torch

from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT
from src.mech.endpoint_retention import (
    RETENTION_COMMON_RANK,
    RETENTION_COMMON_STATES,
    RETENTION_METHODS,
    RetentionBasis,
)


ENDPOINT_ABLATION_SCHEMA_VERSION = 2
ENDPOINT_ABLATION_CONTRACT = "frozen_checkpoint_forced_answer_colon_ablation_v2"


@dataclass(frozen=True)
class EndpointAblationSpec:
    """A fixed subspace to remove from one or more answer-colon block states."""

    name: str
    basis: torch.Tensor
    ranks: torch.Tensor
    family: str
    method: str | None
    state: int | None
    direction_slot: int | None
    residual_pc_index: int | None
    random_replicate: int | None
    # Optional v3 localization metadata.  The original v2 registry leaves these
    # fields unset, so its serialized contract and arm names remain unchanged.
    active_direction_slots: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    matched_method: str | None = None
    calibration_target_energy_by_state: tuple[tuple[int, float], ...] | None = None
    calibration_achieved_energy_by_state: tuple[tuple[int, float], ...] | None = None
    selected_overlap_by_state: tuple[tuple[int, float], ...] | None = None

    @property
    def total_rank(self) -> int:
        return int(self.ranks.sum())


def _empty_basis(rank: int = RETENTION_COMMON_RANK) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE, rank, dtype=torch.float32),
        torch.zeros(GPT2_STATE_COUNT, dtype=torch.int64),
    )


def _random_orthonormal(hidden_size: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(hidden_size, rank, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
    return (q * signs.unsqueeze(0)).float()


def build_endpoint_ablation_specs(
    bases: Mapping[str, RetentionBasis],
    *,
    random_replicates: int,
    random_seed: int,
) -> dict[str, EndpointAblationSpec]:
    """Build registered method-level, direction-level, and random-null arms."""
    if set(bases) != set(RETENTION_METHODS):
        raise ValueError("all three completed selector bases are required")
    if random_replicates <= 0:
        raise ValueError("at least one random-null replicate is required")
    specs: dict[str, EndpointAblationSpec] = {}

    for method in RETENTION_METHODS:
        source = bases[method]
        name = f"remove_{method}_joint"
        specs[name] = EndpointAblationSpec(
            name=name,
            basis=source.basis.detach().cpu().float().clone(),
            ranks=source.ranks.detach().cpu().long().clone(),
            family="selected_joint",
            method=method,
            state=None,
            direction_slot=None,
            residual_pc_index=None,
            random_replicate=None,
        )
        for state in RETENTION_COMMON_STATES:
            for slot in range(RETENTION_COMMON_RANK):
                basis, ranks = _empty_basis()
                basis[state, :, 0] = source.basis[state, :, slot].detach().cpu().float()
                ranks[state] = 1
                name = f"remove_{method}_s{state}_d{slot}"
                specs[name] = EndpointAblationSpec(
                    name=name,
                    basis=basis,
                    ranks=ranks,
                    family="selected_single",
                    method=method,
                    state=state,
                    direction_slot=slot,
                    residual_pc_index=(
                        None
                        if source.selected_pc_indices is None
                        else int(source.selected_pc_indices[state, slot])
                    ),
                    random_replicate=None,
                )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    for replicate in range(random_replicates):
        joint_basis, joint_ranks = _empty_basis()
        singles: dict[int, torch.Tensor] = {}
        for state in RETENTION_COMMON_STATES:
            random = _random_orthonormal(
                GPT2_HIDDEN_SIZE, RETENTION_COMMON_RANK, generator
            )
            joint_basis[state] = random
            joint_ranks[state] = RETENTION_COMMON_RANK
            singles[state] = random[:, :1]
        name = f"remove_random_joint_r{replicate:02d}"
        specs[name] = EndpointAblationSpec(
            name=name,
            basis=joint_basis,
            ranks=joint_ranks,
            family="random_joint",
            method=None,
            state=None,
            direction_slot=None,
            residual_pc_index=None,
            random_replicate=replicate,
        )
        for state in RETENTION_COMMON_STATES:
            basis, ranks = _empty_basis()
            basis[state, :, 0] = singles[state][:, 0]
            ranks[state] = 1
            name = f"remove_random_s{state}_r{replicate:02d}"
            specs[name] = EndpointAblationSpec(
                name=name,
                basis=basis,
                ranks=ranks,
                family="random_single",
                method=None,
                state=state,
                direction_slot=None,
                residual_pc_index=None,
                random_replicate=replicate,
            )

    if len(specs) != 21 + 3 * random_replicates:
        raise RuntimeError("endpoint-ablation arm construction drifted")
    for spec in specs.values():
        validate_endpoint_ablation_spec(spec)
    return specs


def validate_endpoint_ablation_spec(spec: EndpointAblationSpec) -> None:
    if spec.basis.shape != (
        GPT2_STATE_COUNT,
        GPT2_HIDDEN_SIZE,
        RETENTION_COMMON_RANK,
    ):
        raise ValueError("endpoint-ablation basis has the wrong shape")
    if spec.ranks.shape != (GPT2_STATE_COUNT,):
        raise ValueError("endpoint-ablation ranks have the wrong shape")
    active_states = set(torch.nonzero(spec.ranks, as_tuple=False).flatten().tolist())
    if not active_states or not active_states.issubset(set(RETENTION_COMMON_STATES)):
        raise ValueError("endpoint ablation escaped states 11 and 12")
    for state in active_states:
        rank = int(spec.ranks[state])
        if not 0 < rank <= RETENTION_COMMON_RANK:
            raise ValueError("endpoint-ablation rank is invalid")
        active = spec.basis[state, :, :rank].double()
        if not torch.allclose(
            active.T @ active,
            torch.eye(rank, dtype=torch.float64),
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError("endpoint-ablation basis is not orthonormal")


def endpoint_ablation_spec_state(spec: EndpointAblationSpec) -> dict:
    validate_endpoint_ablation_spec(spec)
    state = {
        "name": spec.name,
        "family": spec.family,
        "method": spec.method,
        "state": spec.state,
        "direction_slot": spec.direction_slot,
        "residual_pc_index": spec.residual_pc_index,
        "random_replicate": spec.random_replicate,
        "ranks": spec.ranks.tolist(),
        "total_rank": spec.total_rank,
    }
    if spec.active_direction_slots is not None:
        state["active_direction_slots"] = {
            str(index): list(slots)
            for index, slots in spec.active_direction_slots
        }
    if spec.matched_method is not None:
        state["matched_method"] = spec.matched_method
    if spec.calibration_target_energy_by_state is not None:
        state["calibration_target_energy_by_state"] = {
            str(index): value
            for index, value in spec.calibration_target_energy_by_state
        }
    if spec.calibration_achieved_energy_by_state is not None:
        state["calibration_achieved_energy_by_state"] = {
            str(index): value
            for index, value in spec.calibration_achieved_energy_by_state
        }
    if spec.selected_overlap_by_state is not None:
        state["selected_overlap_by_state"] = {
            str(index): value for index, value in spec.selected_overlap_by_state
        }
    return state


class OfficialCODIEndpointHiddenAblation:
    """Remove a centered subspace only while the fixed cue colon is consumed."""

    def __init__(
        self,
        model,
        spec: EndpointAblationSpec,
        *,
        student_mean: torch.Tensor,
        alpha: float = 1.0,
    ) -> None:
        validate_endpoint_ablation_spec(spec)
        if student_mean.shape != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE):
            raise ValueError("student endpoint mean must have shape [13,768]")
        if not torch.isfinite(student_mean).all() or not 0 < alpha <= 2:
            raise ValueError("student mean must be finite and alpha must be in (0,2]")
        base = model.codi.get_base_model()
        transformer = getattr(base, "transformer", None)
        blocks = getattr(transformer, "h", None)
        final_layer_norm = getattr(transformer, "ln_f", None)
        if blocks is None or len(blocks) + 1 != GPT2_STATE_COUNT:
            raise TypeError("official CODI intervention requires 12 GPT-2 blocks")
        if final_layer_norm is None:
            raise TypeError("official CODI intervention requires GPT-2 final layer norm")
        # Hugging Face GPT-2 hidden_states[11] is block 10's output (the input to
        # block 11), while hidden_states[12] is block 11 followed by ln_f. Hook the
        # exact tensors used to fit the endpoint residual bases.
        self.modules_by_state = {11: blocks[10], 12: final_layer_norm}
        self.spec = spec
        self.alpha = float(alpha)
        self.basis = spec.basis.detach().cpu().float()
        self.student_mean = student_mean.detach().cpu().float()
        self.active_mask: torch.Tensor | None = None
        self.calls_by_state = {str(state): 0 for state in RETENTION_COMMON_STATES}
        self.rows_by_state = {str(state): 0 for state in RETENTION_COMMON_STATES}
        self.removed_squared_norm_sum_by_state = {
            str(state): 0.0 for state in RETENTION_COMMON_STATES
        }

    def _hook(self, state: int):
        def edit(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3 or hidden.shape[-1] != GPT2_HIDDEN_SIZE:
                raise ValueError("GPT-2 block output shape changed")
            mask = self.active_mask.to(device=hidden.device)
            if mask.shape != (hidden.shape[0],):
                raise ValueError("endpoint intervention mask has the wrong batch shape")
            rank = int(self.spec.ranks[state])
            basis = self.basis[state, :, :rank].to(device=hidden.device)
            mean = self.student_mean[state].to(device=hidden.device)
            last = hidden[:, -1, :].float()
            centered = last[mask] - mean.unsqueeze(0)
            removed = (centered @ basis) @ basis.T
            edited = hidden.clone()
            edited_last = edited[:, -1, :].float()
            edited_last[mask] = last[mask] - self.alpha * removed
            edited[:, -1, :] = edited_last.to(dtype=hidden.dtype)
            key = str(state)
            self.calls_by_state[key] += 1
            self.rows_by_state[key] += int(mask.sum().item())
            self.removed_squared_norm_sum_by_state[key] += float(
                removed.square().sum().item()
            )
            if isinstance(output, tuple):
                return (edited, *output[1:])
            if isinstance(output, list):
                return [edited, *output[1:]]
            return edited

        return edit

    @contextmanager
    def activate(self, mask: torch.Tensor):
        if self.active_mask is not None:
            raise RuntimeError("endpoint intervention cannot be nested")
        self.active_mask = mask.detach()
        handles = []
        try:
            for state in RETENTION_COMMON_STATES:
                if int(self.spec.ranks[state]) > 0:
                    handles.append(
                        self.modules_by_state[state].register_forward_hook(
                            self._hook(state)
                        )
                    )
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.active_mask = None

    def diagnostics(self) -> dict:
        rms = {}
        for state in RETENTION_COMMON_STATES:
            key = str(state)
            rows = self.rows_by_state[key]
            rms[key] = (
                (self.removed_squared_norm_sum_by_state[key] / rows) ** 0.5
                if rows
                else 0.0
            )
        return {
            "calls_by_state": dict(self.calls_by_state),
            "rows_by_state": dict(self.rows_by_state),
            "removed_projection_rms_by_state": rms,
            "alpha": self.alpha,
        }
