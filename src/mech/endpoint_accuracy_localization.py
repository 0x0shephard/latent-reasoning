"""Accuracy localization after the frozen CODI answer-colon causal screen."""
from __future__ import annotations

import math
from typing import Mapping

import torch

from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT
from src.mech.endpoint_inference_ablation import (
    EndpointAblationSpec,
    _empty_basis,
    validate_endpoint_ablation_spec,
)
from src.mech.endpoint_retention import (
    RETENTION_COMMON_RANK,
    RETENTION_COMMON_STATES,
    RetentionBasis,
)


ACCURACY_LOCALIZATION_SCHEMA_VERSION = 1
ACCURACY_LOCALIZATION_CONTRACT = (
    "frozen_checkpoint_forced_answer_colon_accuracy_localization_v1"
)
LOCALIZATION_METHODS = ("answer_conditioned", "parameter_aware")
NEGATIVE_CONTROL_METHOD = "energy"
MATCHING_ALGORITHM = "selected_orthogonal_covariance_spectral_trace_match_v2"


def _slots_state(slots_by_state: Mapping[int, tuple[int, ...]]) -> tuple:
    return tuple(
        (int(state), tuple(int(slot) for slot in slots_by_state[state]))
        for state in sorted(slots_by_state)
    )


def _selected_spec(
    source: RetentionBasis,
    *,
    name: str,
    family: str,
    slots_by_state: Mapping[int, tuple[int, ...]],
    state: int | None = None,
    direction_slot: int | None = None,
) -> EndpointAblationSpec:
    basis, ranks = _empty_basis()
    for active_state, slots in slots_by_state.items():
        if not slots:
            continue
        for destination, slot in enumerate(slots):
            basis[active_state, :, destination] = source.basis[active_state, :, slot]
        ranks[active_state] = len(slots)
    residual_pc_index = None
    if state is not None and direction_slot is not None and source.selected_pc_indices is not None:
        residual_pc_index = int(source.selected_pc_indices[state, direction_slot])
    spec = EndpointAblationSpec(
        name=name,
        basis=basis,
        ranks=ranks,
        family=family,
        method=source.name,
        state=state,
        direction_slot=direction_slot,
        residual_pc_index=residual_pc_index,
        random_replicate=None,
        active_direction_slots=_slots_state(slots_by_state),
    )
    validate_endpoint_ablation_spec(spec)
    return spec


def build_selected_localization_specs(
    bases: Mapping[str, RetentionBasis],
) -> dict[str, EndpointAblationSpec]:
    """Build the preregistered negative-control and hierarchical localization arms."""
    if set(bases) != {NEGATIVE_CONTROL_METHOD, *LOCALIZATION_METHODS}:
        raise ValueError("energy, answer-conditioned, and parameter-aware bases are required")
    all_slots = tuple(range(RETENTION_COMMON_RANK))
    joint_slots = {state: all_slots for state in RETENTION_COMMON_STATES}
    specs: dict[str, EndpointAblationSpec] = {}

    energy = bases[NEGATIVE_CONTROL_METHOD]
    name = "remove_energy_joint_negative_control"
    specs[name] = _selected_spec(
        energy,
        name=name,
        family="negative_control_joint",
        slots_by_state=joint_slots,
    )

    for method in LOCALIZATION_METHODS:
        source = bases[method]
        name = f"remove_{method}_joint"
        specs[name] = _selected_spec(
            source, name=name, family="selected_joint", slots_by_state=joint_slots
        )
        for state in RETENTION_COMMON_STATES:
            name = f"remove_{method}_state{state}"
            specs[name] = _selected_spec(
                source,
                name=name,
                family="selected_state",
                slots_by_state={state: all_slots},
                state=state,
            )
            for slot in all_slots:
                name = f"remove_{method}_s{state}_d{slot}"
                specs[name] = _selected_spec(
                    source,
                    name=name,
                    family="selected_single",
                    slots_by_state={state: (slot,)},
                    state=state,
                    direction_slot=slot,
                )
                reduced = {
                    other_state: tuple(
                        candidate
                        for candidate in all_slots
                        if not (other_state == state and candidate == slot)
                    )
                    for other_state in RETENTION_COMMON_STATES
                }
                name = f"remove_{method}_joint_except_s{state}_d{slot}"
                specs[name] = _selected_spec(
                    source,
                    name=name,
                    family="selected_joint_minus_one",
                    slots_by_state=reduced,
                    state=state,
                    direction_slot=slot,
                )
    if len(specs) != 31:
        raise RuntimeError("localization arm construction drifted")
    return specs


def projection_energy(covariance: torch.Tensor, basis: torch.Tensor) -> float:
    """Expected squared norm removed by a centered orthogonal projection."""
    covariance = covariance.double()
    basis = basis.double()
    return float(torch.trace(basis.T @ covariance @ basis).item())


def _random_orthogonal(rank: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(rank, rank, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
    return q * signs.unsqueeze(0)


def _sample_spectral_subspace(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    indices: torch.Tensor,
    *,
    rank: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    if indices.numel() < rank:
        raise ValueError("spectral pool is smaller than the requested rank")
    order = torch.randperm(indices.numel(), generator=generator)[:rank]
    chosen = indices[order]
    return (
        eigenvectors[:, chosen] @ _random_orthogonal(rank, generator),
        float(eigenvalues[chosen].sum().item()),
    )


def energy_matched_random_subspace(
    covariance: torch.Tensor,
    selected: torch.Tensor,
    *,
    generator: torch.Generator,
    attempts: int = 256,
    maximum_normalized_overlap: float = 0.20,
    eigensystem: tuple[torch.Tensor, torch.Tensor] | None = None,
    target_energy: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Construct an orthonormal random subspace with exactly matched trace energy.

    Two covariance-spectral subspaces are sampled on opposite sides of the target
    trace.  An orthogonal interpolation between them matches the selected trace
    without scaling the intervention or using the selected directions themselves.
    """
    covariance = covariance.detach().cpu().double()
    selected = selected.detach().cpu().double()
    rank = selected.shape[1]
    if covariance.shape != (GPT2_HIDDEN_SIZE, GPT2_HIDDEN_SIZE):
        raise ValueError("localization covariance must have shape [768,768]")
    if rank <= 0 or selected.shape[0] != GPT2_HIDDEN_SIZE:
        raise ValueError("selected subspace has the wrong shape")
    covariance = 0.5 * (covariance + covariance.T)
    if eigensystem is None:
        complete, _ = torch.linalg.qr(selected, mode="complete")
        complement = complete[:, rank:]
        restricted = complement.T @ covariance @ complement
        eigenvalues, restricted_vectors = torch.linalg.eigh(
            0.5 * (restricted + restricted.T)
        )
        eigenvalues = eigenvalues.clamp_min(0)
        eigenvectors = complement @ restricted_vectors
    else:
        eigenvalues, eigenvectors = eigensystem
    target = (
        projection_energy(covariance, selected)
        if target_energy is None
        else float(target_energy)
    )
    min_trace = float(eigenvalues[:rank].sum().item())
    max_trace = float(eigenvalues[-rank:].sum().item())
    if not min_trace - 1e-8 <= target <= max_trace + 1e-8:
        raise RuntimeError("selected projection energy lies outside the covariance spectrum")

    # Broad pools preserve randomness; deterministic extreme subspaces are fallback
    # brackets when a selected trace is very close to a Ky Fan bound.
    spectral_size = int(eigenvalues.numel())
    pool = min(128, spectral_size // 2)
    low_indices = torch.arange(pool, dtype=torch.long)
    high_indices = torch.arange(spectral_size - pool, spectral_size, dtype=torch.long)
    best = None
    for _ in range(attempts):
        low, low_energy = _sample_spectral_subspace(
            eigenvalues,
            eigenvectors,
            low_indices,
            rank=rank,
            generator=generator,
        )
        high, high_energy = _sample_spectral_subspace(
            eigenvalues,
            eigenvectors,
            high_indices,
            rank=rank,
            generator=generator,
        )
        if low_energy > target or high_energy < target:
            low = eigenvectors[:, :rank] @ _random_orthogonal(rank, generator)
            high = eigenvectors[:, -rank:] @ _random_orthogonal(rank, generator)
            low_energy = min_trace
            high_energy = max_trace
        denominator = high_energy - low_energy
        weight = 0.0 if denominator <= 1e-15 else (target - low_energy) / denominator
        weight = min(1.0, max(0.0, weight))
        candidate = math.sqrt(weight) * high + math.sqrt(1.0 - weight) * low
        achieved = weight * high_energy + (1.0 - weight) * low_energy
        overlap = float((selected.T @ candidate).square().sum().item() / rank)
        relative_error = abs(achieved - target) / max(target, 1e-12)
        score = overlap + 1_000.0 * relative_error
        if best is None or score < best[0]:
            best = (score, candidate, achieved, overlap, relative_error)
        if overlap <= maximum_normalized_overlap and relative_error <= 2e-6:
            break
    if best is None:
        raise RuntimeError("failed to generate an energy-matched random subspace")
    _, candidate, achieved, overlap, relative_error = best
    if relative_error > 2e-5:
        raise RuntimeError("random subspace failed the calibration-energy match")
    if overlap > maximum_normalized_overlap:
        raise RuntimeError("random subspace overlaps too strongly with the selected basis")
    return candidate.float(), {
        "target_energy": target,
        "achieved_energy": achieved,
        "relative_energy_error": relative_error,
        "normalized_selected_overlap": overlap,
        "target_rms": math.sqrt(max(target, 0.0)),
        "achieved_rms": math.sqrt(max(achieved, 0.0)),
        "algorithm": MATCHING_ALGORITHM,
    }


def build_matched_random_joint_specs(
    bases: Mapping[str, RetentionBasis],
    covariance_by_state: Mapping[int, torch.Tensor],
    *,
    random_replicates: int,
    random_seed: int,
) -> dict[str, EndpointAblationSpec]:
    """Build method-specific joint nulls matched on calibration projection energy."""
    if random_replicates < 1:
        raise ValueError("at least one matched-random replicate is required")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    specs = {}
    target_energies = {
        (method, state): projection_energy(
            covariance_by_state[int(state)],
            bases[method].basis[state, :, :RETENTION_COMMON_RANK],
        )
        for method in LOCALIZATION_METHODS
        for state in RETENTION_COMMON_STATES
    }
    eigensystems = {}
    for method in LOCALIZATION_METHODS:
        for state in RETENTION_COMMON_STATES:
            covariance = covariance_by_state[int(state)].detach().cpu().double()
            covariance = 0.5 * (covariance + covariance.T)
            selected = bases[method].basis[
                state, :, :RETENTION_COMMON_RANK
            ].double()
            complete, _ = torch.linalg.qr(selected, mode="complete")
            complement = complete[:, RETENTION_COMMON_RANK:]
            restricted = complement.T @ covariance @ complement
            eigenvalues, restricted_vectors = torch.linalg.eigh(
                0.5 * (restricted + restricted.T)
            )
            eigensystems[(method, state)] = (
                eigenvalues.clamp_min(0),
                complement @ restricted_vectors,
            )
    for method in LOCALIZATION_METHODS:
        source = bases[method]
        for replicate in range(random_replicates):
            basis, ranks = _empty_basis()
            targets = []
            achieved = []
            overlaps = []
            for state in RETENTION_COMMON_STATES:
                covariance = covariance_by_state[int(state)]
                selected = source.basis[state, :, :RETENTION_COMMON_RANK]
                random_basis, diagnostics = energy_matched_random_subspace(
                    covariance,
                    selected,
                    generator=generator,
                    eigensystem=eigensystems[(method, state)],
                    target_energy=target_energies[(method, state)],
                )
                basis[state] = random_basis
                ranks[state] = RETENTION_COMMON_RANK
                targets.append((state, diagnostics["target_energy"]))
                achieved.append((state, diagnostics["achieved_energy"]))
                overlaps.append((state, diagnostics["normalized_selected_overlap"]))
            name = f"remove_matched_random_{method}_joint_r{replicate:03d}"
            spec = EndpointAblationSpec(
                name=name,
                basis=basis,
                ranks=ranks,
                family="matched_random_joint",
                method=None,
                state=None,
                direction_slot=None,
                residual_pc_index=None,
                random_replicate=replicate,
                active_direction_slots=None,
                matched_method=method,
                calibration_target_energy_by_state=tuple(targets),
                calibration_achieved_energy_by_state=tuple(achieved),
                selected_overlap_by_state=tuple(overlaps),
            )
            validate_endpoint_ablation_spec(spec)
            specs[name] = spec
    expected = len(LOCALIZATION_METHODS) * random_replicates
    if len(specs) != expected:
        raise RuntimeError("matched-random arm construction drifted")
    return specs


def build_accuracy_localization_specs(
    bases: Mapping[str, RetentionBasis],
    covariance_by_state: Mapping[int, torch.Tensor],
    *,
    random_replicates: int,
    random_seed: int,
) -> dict[str, EndpointAblationSpec]:
    specs = build_selected_localization_specs(bases)
    random_specs = build_matched_random_joint_specs(
        bases,
        covariance_by_state,
        random_replicates=random_replicates,
        random_seed=random_seed,
    )
    if set(specs).intersection(random_specs):
        raise RuntimeError("duplicate accuracy-localization arm name")
    specs.update(random_specs)
    return specs
