"""Causal subspace interventions on official CODI latent KV-cache entries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


INTERVENTION_MODES = ("retain", "remove")
BASIS_KINDS = ("learned", "random")


@dataclass(frozen=True)
class OfficialCODIKVInterventionSpec:
    mode: str
    basis_kind: str
    positions: frozenset[int]

    def __post_init__(self) -> None:
        if self.mode not in INTERVENTION_MODES:
            raise ValueError(f"unknown intervention mode {self.mode!r}")
        if self.basis_kind not in BASIS_KINDS:
            raise ValueError(f"unknown basis kind {self.basis_kind!r}")
        if not self.positions:
            raise ValueError("at least one latent position is required")
        if min(self.positions) < 0:
            raise ValueError("latent positions must be non-negative")

    @property
    def name(self) -> str:
        position_scope = (
            "all"
            if self.positions == frozenset(range(6))
            else "p" + "-".join(str(value) for value in sorted(self.positions))
        )
        return f"{self.mode}_{self.basis_kind}_{position_scope}"


def build_intervention_specs(
    *,
    positions: Iterable[int],
    include_all: bool,
    latent_positions: int,
) -> list[OfficialCODIKVInterventionSpec]:
    selected = sorted({int(position) for position in positions})
    if not selected:
        raise ValueError("at least one single-position intervention is required")
    if min(selected) < 0 or max(selected) >= latent_positions:
        raise ValueError("intervention position is outside the latent trajectory")
    scopes = [frozenset({position}) for position in selected]
    if include_all:
        scopes.append(frozenset(range(latent_positions)))
    return [
        OfficialCODIKVInterventionSpec(
            mode=mode,
            basis_kind=basis_kind,
            positions=scope,
        )
        for mode in INTERVENTION_MODES
        for basis_kind in BASIS_KINDS
        for scope in scopes
    ]


def _legacy_cache(cache):
    if isinstance(cache, tuple):
        return cache, tuple
    if isinstance(cache, list):
        return tuple(cache), list
    converter = getattr(cache, "to_legacy_cache", None)
    if converter is None:
        raise TypeError(
            "official CODI intervention requires a legacy tuple/list cache "
            "or a cache with to_legacy_cache()"
        )
    legacy = converter()
    cache_type = type(cache)

    def restore(values):
        factory = getattr(cache_type, "from_legacy_cache", None)
        if factory is None:
            raise TypeError(
                f"cache type {cache_type.__name__} cannot restore legacy values"
            )
        return factory(values)

    return legacy, restore


class OfficialCODIKVSubspaceIntervention:
    """Apply centered learned/random projections to newly appended latent K/V."""

    def __init__(
        self,
        artifact: dict,
        spec: OfficialCODIKVInterventionSpec,
        *,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.rank = int(artifact["rank"])
        self.latent_positions = int(
            artifact["kinds"]["key"]["learned_basis"].shape[2]
        )
        if max(spec.positions) >= self.latent_positions:
            raise ValueError("intervention position exceeds artifact trajectory")
        self.values = {}
        for kind in ("key", "value"):
            payload = artifact["kinds"][kind]
            learned = payload["learned_basis"]
            random = payload["random_basis"]
            mean = payload["student_mean"]
            scale = payload["random_energy_scale"]
            expected = (
                learned.shape[0],
                learned.shape[1],
                self.latent_positions,
                learned.shape[3],
                self.rank,
            )
            if tuple(learned.shape) != expected or tuple(random.shape) != expected:
                raise ValueError(f"{kind} intervention basis shape is invalid")
            if tuple(mean.shape) != expected[:-1]:
                raise ValueError(f"{kind} student mean shape is invalid")
            if tuple(scale.shape) != expected[:-2]:
                raise ValueError(f"{kind} random scale shape is invalid")
            basis = learned if spec.basis_kind == "learned" else random
            self.values[kind] = {
                "basis": basis.to(device=device, dtype=torch.float32),
                "mean": mean.to(device=device, dtype=torch.float32),
                "scale": (
                    torch.ones_like(scale, device=device, dtype=torch.float32)
                    if spec.basis_kind == "learned"
                    else scale.to(device=device, dtype=torch.float32)
                ),
            }

    def _transform(
        self,
        tensor: torch.Tensor,
        *,
        layer: int,
        position: int,
        kind: str,
    ) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError("cache K/V tensor must have shape [B,H,T,D]")
        values = self.values[kind]
        basis = values["basis"][layer, :, position]
        mean = values["mean"][layer, :, position]
        scale = values["scale"][layer, :, position]
        if tensor.shape[1] != basis.shape[0] or tensor.shape[-1] != basis.shape[1]:
            raise ValueError(
                f"{kind} cache shape {tuple(tensor.shape)} does not match "
                f"basis {tuple(basis.shape)}"
            )
        last = tensor[:, :, -1, :].to(torch.float32)
        centered = last - mean.unsqueeze(0)
        coefficients = torch.einsum("bhd,hdr->bhr", centered, basis)
        component = torch.einsum("bhr,hdr->bhd", coefficients, basis)
        component = component * scale.unsqueeze(0).unsqueeze(-1)
        if self.spec.mode == "retain":
            transformed = mean.unsqueeze(0) + component
        else:
            transformed = mean.unsqueeze(0) + centered - component
        transformed = transformed.to(dtype=tensor.dtype)
        return torch.cat(
            (tensor[:, :, :-1, :], transformed.unsqueeze(2)),
            dim=2,
        )

    def __call__(self, cache, position: int):
        if position not in self.spec.positions:
            return cache
        legacy, restore = _legacy_cache(cache)
        if len(legacy) != self.values["key"]["basis"].shape[0]:
            raise ValueError("cache layer count does not match intervention artifact")
        updated = []
        for layer, entry in enumerate(legacy):
            if len(entry) < 2:
                raise ValueError("each cache layer must contain key and value tensors")
            key = self._transform(
                entry[0],
                layer=layer,
                position=position,
                kind="key",
            )
            value = self._transform(
                entry[1],
                layer=layer,
                position=position,
                kind="value",
            )
            updated.append((key, value, *entry[2:]))
        updated = tuple(updated)
        if restore is tuple:
            return updated
        if restore is list:
            return list(updated)
        return restore(updated)
