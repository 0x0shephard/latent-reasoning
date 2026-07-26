from __future__ import annotations

import torch

from src.mech.kv_target_utility import (
    KVTargetGroup,
    build_target_groups,
    combine_gradients,
    default_layer_bands,
    gradient_inner_product,
    kv_group_loss,
    updated_parameter_mapping,
)


def test_hierarchical_target_groups_are_deterministic():
    assert default_layer_bands(12) == {
        "early": (0, 1, 2, 3),
        "middle": (4, 5, 6, 7),
        "late": (8, 9, 10, 11),
    }
    kind = build_target_groups(
        granularity="kind",
        layer_count=12,
        position_count=6,
    )
    assert [group.name for group in kind] == ["key_all", "value_all"]
    positions = build_target_groups(
        granularity="position",
        layer_count=12,
        position_count=6,
        kinds=["key"],
        positions=[1, 5],
    )
    assert [group.name for group in positions] == ["key_p1", "key_p5"]
    bands = build_target_groups(
        granularity="layer_band",
        layer_count=12,
        position_count=6,
        kinds=["value"],
        positions=[4],
    )
    assert [group.name for group in bands] == [
        "value_p4_early",
        "value_p4_middle",
        "value_p4_late",
    ]


def test_group_loss_selects_only_requested_layer_and_position():
    student = torch.zeros(2, 3, 1, 2, 2)
    teacher = torch.zeros_like(student)
    teacher[:, 1, :, 0] = 2.0
    teacher[:, 2, :, 1] = 9.0
    mask = torch.ones(student.shape[:-1], dtype=torch.bool)
    group = KVTargetGroup(
        name="key_l1_p0",
        kind="key",
        layers=(1,),
        positions=(0,),
    )
    assert torch.isclose(
        kv_group_loss(student, teacher, mask, group, metric="l1"),
        torch.tensor(2.0),
    )
    mask[:, 1, :, 0] = False
    assert torch.isclose(
        kv_group_loss(student, teacher, mask, group, metric="mse"),
        torch.tensor(0.0),
    )


def test_gradient_alignment_and_equal_norm_update():
    left = (torch.tensor([1.0, 0.0]), None)
    right = (torch.tensor([2.0, 0.0]), torch.tensor([3.0]))
    stats = gradient_inner_product(left, right)
    assert stats["dot"] == 2.0
    assert stats["cosine"] > 0.0

    base = (torch.tensor([1.0, 0.0]), None)
    auxiliary = (torch.tensor([0.0, 1.0]), torch.tensor([2.0]))
    combined = combine_gradients(base, auxiliary, auxiliary_weight=0.5)
    assert torch.equal(combined[0], torch.tensor([1.0, 0.5]))
    assert torch.equal(combined[1], torch.tensor([1.0]))

    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    original = first.detach().clone()
    mapping, metadata = updated_parameter_mapping(
        ["first", "second"],
        [first, second],
        combined,
        update_norm=0.25,
    )
    update_square = (
        (mapping["first"] - first.detach()).square().sum()
        + (mapping["second"] - second.detach()).square().sum()
    )
    assert torch.isclose(update_square.sqrt(), torch.tensor(0.25))
    assert torch.equal(first.detach(), original)
    assert metadata["requested_update_norm"] == 0.25

