from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.mech.position_conditioned_readout import (
    AnswerPositionBucket,
    DEFAULT_ANSWER_POSITION_BUCKETS,
    PositionConditionedVocabularyHead,
    VocabularyPrefixHead,
    answer_position_bucket,
    validate_position_buckets,
)


class ConstantHead(nn.Module):
    def __init__(self, value: float, vocabulary_size: int = 7) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value)))
        self.vocabulary_size = vocabulary_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_ones(*hidden_states.shape[:-1], self.vocabulary_size) * self.value


def test_default_buckets_cover_answer_positions_without_overlap():
    expected = {
        0: "p0",
        1: "p1",
        2: "p2_plus",
        3: "p2_plus",
        5: "p2_plus",
        6: "p2_plus",
        99: "p2_plus",
    }
    assert {
        position: answer_position_bucket(position)
        for position in expected
    } == expected


def test_invalid_bucket_partition_is_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        validate_position_buckets(
            (AnswerPositionBucket("p0", 0, 1), AnswerPositionBucket("p2", 2, None))
        )
    with pytest.raises(ValueError, match="final"):
        validate_position_buckets((AnswerPositionBucket("all", 0, 4),))


def test_router_uses_inactive_and_position_specific_heads():
    heads = {
        bucket.name: ConstantHead(index + 1)
        for index, bucket in enumerate(DEFAULT_ANSWER_POSITION_BUCKETS)
    }
    router = PositionConditionedVocabularyHead(
        heads, inactive_head=ConstantHead(-1), vocabulary_size=7
    )
    hidden = torch.randn(2, 3)
    assert torch.equal(router(hidden), torch.full((2, 7), -1.0))
    router.set_answer_position(0)
    assert router.active_bucket == "p0"
    assert torch.equal(router(hidden), torch.full((2, 7), 1.0))
    router.set_answer_position(4)
    assert router.active_bucket == "p2_plus"
    assert torch.equal(router(hidden), torch.full((2, 7), 3.0))
    router.set_answer_position(20)
    assert router.active_bucket == "p2_plus"
    assert torch.equal(router(hidden), torch.full((2, 7), 3.0))


def test_vocabulary_prefix_head_slices_only_the_last_dimension():
    base = nn.Linear(3, 9, bias=False)
    prefix = VocabularyPrefixHead(base, 7)
    hidden = torch.randn(2, 4, 3)
    assert torch.equal(prefix(hidden), base(hidden)[..., :7])
