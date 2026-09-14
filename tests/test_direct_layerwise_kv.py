import torch

from src.mech.direct_layerwise_kv import (
    DirectLatentKVSubspaceIntervention,
    align_layerwise_bases,
    fit_layerwise_eigensystems,
    longest_contiguous_run,
    score_layerwise_answer_directions,
    select_layerwise_bases,
)


def test_independent_eigensystems_selection_and_alignment():
    generator = torch.Generator().manual_seed(41)
    states = torch.randn(20, 12, 2, 768, generator=generator)
    gradients = 0.2 * states + torch.randn(states.shape, generator=generator)
    eigen = fit_layerwise_eigensystems(states)
    score = score_layerwise_answer_directions(states, gradients, eigen, seed=7)
    bases, indices = select_layerwise_bases(eigen, score["split_stable_z"], rank=4)
    aligned, rotations = align_layerwise_bases(states, eigen.means, bases)
    assert indices.shape == (12, 4)
    assert aligned.shape == (12, 768, 4)
    assert rotations.shape == (12, 4, 4)
    assert torch.allclose(aligned[3].T @ aligned[3], torch.eye(4), atol=2e-5)


def test_direct_kv_remove_edits_only_requested_layer_and_position():
    generator = torch.Generator().manual_seed(5)
    cache = tuple((torch.randn(2, 2, 3, 4, generator=generator),
                   torch.randn(2, 2, 3, 4, generator=generator)) for _ in range(12))
    bases = torch.eye(8)[:, :2].repeat(12, 1, 1)
    means = torch.zeros(12, 6, 8)
    intervention = DirectLatentKVSubspaceIntervention(
        key_bases=bases, value_bases=bases, key_means=means, value_means=means,
        layers=[3], positions=[1], mode="remove")
    untouched = intervention(cache, 0)
    assert all(torch.equal(a, b) for old, new in zip(cache, untouched) for a, b in zip(old, new))
    edited = intervention(cache, 1)
    assert torch.equal(edited[2][0], cache[2][0])
    assert torch.equal(edited[3][0][:, :, :-1], cache[3][0][:, :, :-1])
    flattened = edited[3][0][:, :, -1].reshape(2, 8)
    assert torch.allclose(flattened[:, :2], torch.zeros(2, 2), atol=1e-6)


def test_direct_kv_intervention_accepts_variable_rank_layer_mappings():
    cache = tuple((torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4)) for _ in range(12))
    key_bases = {3: torch.eye(8)[:, :1], 5: torch.eye(8)[:, :3]}
    value_bases = {3: torch.eye(8)[:, :1], 5: torch.eye(8)[:, :3]}
    intervention = DirectLatentKVSubspaceIntervention(
        key_bases=key_bases, value_bases=value_bases,
        key_means=torch.zeros(12, 1, 8), value_means=torch.zeros(12, 1, 8),
        layers=[3], positions=[0], mode="remove",
    )
    edited = intervention(cache, 0)
    assert edited[3][0][0, 0, 0, 0].item() == 0
    assert torch.equal(edited[5][0], cache[5][0])


def test_direct_kv_intervention_accepts_independent_key_and_value_ranks():
    cache = tuple((torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4)) for _ in range(12))
    key_bases = {3: torch.eye(8)[:, :1]}
    value_bases = {3: torch.eye(8)[:, :3]}
    means = torch.zeros(12, 1, 8)
    intervention = DirectLatentKVSubspaceIntervention(
        key_bases=key_bases, value_bases=value_bases,
        key_means=means, value_means=means,
        layers=[3], positions=[0], mode="remove",
    )
    edited = intervention(cache, 0)
    assert edited[3][0][0, 0, 0, 0].item() == 0
    assert torch.equal(edited[3][0][0, 0, 0, 1:], torch.ones(3))
    assert torch.equal(edited[3][1][0, 0, 0, :3], torch.zeros(3))


def test_empty_basis_is_inactive_for_retain_and_remove():
    cache = tuple((torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4)) for _ in range(12))
    bases = {3: torch.empty(8, 0)}
    means = torch.zeros(12, 1, 8)
    for mode in ("retain", "remove"):
        intervention = DirectLatentKVSubspaceIntervention(
            key_bases=bases, value_bases=bases, key_means=means, value_means=means,
            layers=[3], positions=[0], mode=mode,
        )
        edited = intervention(cache, 0)
        assert torch.equal(edited[3][0], cache[3][0])
        assert torch.equal(edited[3][1], cache[3][1])


def test_longest_contiguous_run_prefers_late_tie():
    assert longest_contiguous_run([1, 2, 6, 7]) == [6, 7]


def test_disconnected_gradient_cells_are_zero_filled_and_audited():
    # Exercise the structural policy directly without constructing GPT-2.
    trace = object.__new__(__import__(
        "src.mech.direct_layerwise_kv", fromlist=["LatentAttentionStateTrace"]
    ).LatentAttentionStateTrace)
    trace.latent_positions = 1
    trace.raw = [[torch.randn(2, 3, requires_grad=True)] for _ in range(12)]
    trace._gradient_connected = None
    loss = sum(value.square().sum() for layer in trace.raw[:11] for value in layer)
    gradients = trace.gradients(loss)
    assert gradients.shape == (2, 12, 1, 3)
    assert trace.gradient_connectivity()[-1, 0].item() is False
    assert torch.equal(gradients[:, -1, 0], torch.zeros(2, 3))
