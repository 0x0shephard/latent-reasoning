"""Single-hypothesis parameter-aware state-12 confirmation at CODI's answer colon."""
from __future__ import annotations

from typing import Mapping

import torch

from src.mech.endpoint_accuracy_localization import (
    MATCHING_ALGORITHM,
    energy_matched_random_subspace,
    projection_energy,
)
from src.mech.endpoint_inference_ablation import (
    EndpointAblationSpec,
    _empty_basis,
    validate_endpoint_ablation_spec,
)
from src.mech.endpoint_retention import RETENTION_COMMON_RANK, RetentionBasis


STATE12_CONFIRMATION_SCHEMA_VERSION = 1
STATE12_CONFIRMATION_CONTRACT = (
    "frozen_checkpoint_parameter_aware_state12_confirmation_v1"
)
PRIMARY_METHOD = "parameter_aware"
PRIMARY_STATE = 12
PRIMARY_RANK = RETENTION_COMMON_RANK


def _selected_primary_spec(source: RetentionBasis) -> EndpointAblationSpec:
    basis, ranks = _empty_basis()
    basis[PRIMARY_STATE] = source.basis[PRIMARY_STATE]
    ranks[PRIMARY_STATE] = PRIMARY_RANK
    spec = EndpointAblationSpec(
        name="remove_parameter_aware_state12_primary",
        basis=basis,
        ranks=ranks,
        family="selected_primary",
        method=PRIMARY_METHOD,
        state=PRIMARY_STATE,
        direction_slot=None,
        residual_pc_index=None,
        random_replicate=None,
        active_direction_slots=((PRIMARY_STATE, tuple(range(PRIMARY_RANK))),),
    )
    validate_endpoint_ablation_spec(spec)
    return spec


def build_state12_confirmation_specs(
    bases: Mapping[str, RetentionBasis],
    covariance: torch.Tensor,
    *,
    random_replicates: int,
    random_seed: int,
) -> dict[str, EndpointAblationSpec]:
    """Build one selected state-12 arm and its method-specific matched null."""
    if PRIMARY_METHOD not in bases:
        raise ValueError("the completed parameter-aware basis is required")
    if random_replicates < 1:
        raise ValueError("at least one matched-random replicate is required")
    source = bases[PRIMARY_METHOD]
    selected = source.basis[PRIMARY_STATE, :, :PRIMARY_RANK]
    covariance = covariance.detach().cpu().double()
    target_energy = projection_energy(covariance, selected)

    complete, _ = torch.linalg.qr(selected.double(), mode="complete")
    complement = complete[:, PRIMARY_RANK:]
    restricted = complement.T @ covariance @ complement
    eigenvalues, restricted_vectors = torch.linalg.eigh(
        0.5 * (restricted + restricted.T)
    )
    eigensystem = (eigenvalues.clamp_min(0), complement @ restricted_vectors)

    specs = {"remove_parameter_aware_state12_primary": _selected_primary_spec(source)}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_seed))
    for replicate in range(random_replicates):
        random_basis, diagnostics = energy_matched_random_subspace(
            covariance,
            selected,
            generator=generator,
            eigensystem=eigensystem,
            target_energy=target_energy,
        )
        basis, ranks = _empty_basis()
        basis[PRIMARY_STATE] = random_basis
        ranks[PRIMARY_STATE] = PRIMARY_RANK
        name = f"remove_matched_random_parameter_aware_state12_r{replicate:03d}"
        spec = EndpointAblationSpec(
            name=name,
            basis=basis,
            ranks=ranks,
            family="matched_random_primary",
            method=None,
            state=PRIMARY_STATE,
            direction_slot=None,
            residual_pc_index=None,
            random_replicate=replicate,
            matched_method=PRIMARY_METHOD,
            calibration_target_energy_by_state=((PRIMARY_STATE, target_energy),),
            calibration_achieved_energy_by_state=(
                (PRIMARY_STATE, diagnostics["achieved_energy"]),
            ),
            selected_overlap_by_state=(
                (PRIMARY_STATE, diagnostics["normalized_selected_overlap"]),
            ),
        )
        validate_endpoint_ablation_spec(spec)
        specs[name] = spec
    if len(specs) != random_replicates + 1:
        raise RuntimeError("state-12 confirmation arm construction drifted")
    return specs


__all__ = [
    "MATCHING_ALGORITHM",
    "PRIMARY_METHOD",
    "PRIMARY_RANK",
    "PRIMARY_STATE",
    "STATE12_CONFIRMATION_CONTRACT",
    "STATE12_CONFIRMATION_SCHEMA_VERSION",
    "build_state12_confirmation_specs",
]
