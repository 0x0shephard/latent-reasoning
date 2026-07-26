"""Causal cache-intervention contract tests for the official CODI path."""
from __future__ import annotations

import torch

from src.mech.official_codi_kv_intervention import (
    OfficialCODIKVInterventionSpec,
    OfficialCODIKVSubspaceIntervention,
    build_intervention_specs,
)


def _artifact() -> dict:
    layers, heads, positions, dimension, rank = 2, 2, 6, 4, 2
    learned = torch.zeros(layers, heads, positions, dimension, rank)
    learned[..., 0, 0] = 1.0
    learned[..., 1, 1] = 1.0
    random = torch.zeros_like(learned)
    random[..., 2, 0] = 1.0
    random[..., 3, 1] = 1.0
    means = torch.arange(
        layers * heads * positions * dimension,
        dtype=torch.float32,
    ).reshape(layers, heads, positions, dimension) / 100.0
    scale = torch.full((layers, heads, positions), 1.5)
    payload = {
        "learned_basis": learned,
        "random_basis": random,
        "student_mean": means,
        "random_energy_scale": scale,
    }
    return {
        "rank": rank,
        "kinds": {
            "key": {key: value.clone() for key, value in payload.items()},
            "value": {key: value.clone() for key, value in payload.items()},
        },
    }


def _cache() -> tuple:
    values = []
    for layer in range(2):
        key = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(
            2, 2, 3, 4
        )
        key = key + layer * 100
        values.append((key, key + 1_000))
    return tuple(values)


def test_learned_retain_and_remove_are_centered_complements():
    artifact = _artifact()
    cache = _cache()
    position = 4
    retain = OfficialCODIKVSubspaceIntervention(
        artifact,
        OfficialCODIKVInterventionSpec(
            mode="retain",
            basis_kind="learned",
            positions=frozenset({position}),
        ),
        device=torch.device("cpu"),
    )(cache, position)
    remove = OfficialCODIKVSubspaceIntervention(
        artifact,
        OfficialCODIKVInterventionSpec(
            mode="remove",
            basis_kind="learned",
            positions=frozenset({position}),
        ),
        device=torch.device("cpu"),
    )(cache, position)

    for layer in range(2):
        mean = artifact["kinds"]["key"]["student_mean"][layer, :, position]
        original = cache[layer][0][:, :, -1]
        retained = retain[layer][0][:, :, -1]
        removed = remove[layer][0][:, :, -1]
        assert torch.equal(retain[layer][0][:, :, :-1], cache[layer][0][:, :, :-1])
        assert torch.allclose(
            (retained - mean) + (removed - mean),
            original - mean,
        )
        assert torch.allclose(retained[..., 2:], mean[..., 2:])
        assert torch.allclose(
            removed[..., :2],
            mean[..., :2].unsqueeze(0).expand_as(removed[..., :2]),
            atol=1e-5,
        )


def test_random_projection_uses_the_energy_matching_scale():
    artifact = _artifact()
    cache = _cache()
    position = 5
    intervention = OfficialCODIKVSubspaceIntervention(
        artifact,
        OfficialCODIKVInterventionSpec(
            mode="retain",
            basis_kind="random",
            positions=frozenset({position}),
        ),
        device=torch.device("cpu"),
    )
    transformed = intervention(cache, position)
    mean = artifact["kinds"]["key"]["student_mean"][0, :, position]
    centered = cache[0][0][:, :, -1] - mean
    expected = mean + torch.stack(
        (
            torch.zeros_like(centered[..., 0]),
            torch.zeros_like(centered[..., 1]),
            1.5 * centered[..., 2],
            1.5 * centered[..., 3],
        ),
        dim=-1,
    )
    assert torch.allclose(transformed[0][0][:, :, -1], expected)


def test_unselected_position_is_bitwise_unchanged():
    cache = _cache()
    intervention = OfficialCODIKVSubspaceIntervention(
        _artifact(),
        OfficialCODIKVInterventionSpec(
            mode="remove",
            basis_kind="learned",
            positions=frozenset({4}),
        ),
        device=torch.device("cpu"),
    )
    assert intervention(cache, 3) is cache


def test_full_position_contract_has_unique_28_intervention_arms():
    specs = build_intervention_specs(
        positions=range(6),
        include_all=True,
        latent_positions=6,
    )
    assert len(specs) == 28
    assert len({spec.name for spec in specs}) == 28
    assert any(spec.name == "remove_learned_p4" for spec in specs)
    assert any(spec.name == "retain_random_all" for spec in specs)
