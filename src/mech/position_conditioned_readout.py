"""Position-conditioned vocabulary heads for CODI answer generation.

The confirmed endpoint-band experiment intervened only on the state that predicts
the first visible answer token.  This module supplies the small amount of routing
needed to test whether later answer positions have their own locally sufficient
low-rank readouts.

Routing is explicit: the decoder sets the zero-based visible answer position before
each answer forward pass.  Calls made while CODI is encoding the prompt or producing
continuous latent thoughts use the separately configured ``inactive`` head because
their vocabulary logits are not consumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AnswerPositionBucket:
    """A half-open answer-position interval ``[start, stop)``.

    ``stop=None`` denotes the final unbounded bucket.
    """

    name: str
    start: int
    stop: int | None

    def contains(self, position: int) -> bool:
        return position >= self.start and (self.stop is None or position < self.stop)


DEFAULT_ANSWER_POSITION_BUCKETS = (
    AnswerPositionBucket("p0", 0, 1),
    AnswerPositionBucket("p1", 1, 2),
    AnswerPositionBucket("p2_plus", 2, None),
)


def validate_position_buckets(buckets: Sequence[AnswerPositionBucket]) -> None:
    if not buckets:
        raise ValueError("at least one answer-position bucket is required")
    names: set[str] = set()
    expected_start = 0
    for index, bucket in enumerate(buckets):
        if not bucket.name or bucket.name in names:
            raise ValueError("answer-position bucket names must be unique and non-empty")
        names.add(bucket.name)
        if bucket.start != expected_start:
            raise ValueError("answer-position buckets must be contiguous and start at zero")
        if bucket.stop is None:
            if index != len(buckets) - 1:
                raise ValueError("only the final answer-position bucket may be unbounded")
            return
        if bucket.stop <= bucket.start:
            raise ValueError("answer-position bucket stops must exceed their starts")
        expected_start = bucket.stop
    raise ValueError("the final answer-position bucket must be unbounded")


def answer_position_bucket(
    position: int,
    buckets: Sequence[AnswerPositionBucket] = DEFAULT_ANSWER_POSITION_BUCKETS,
) -> str:
    validate_position_buckets(buckets)
    if int(position) < 0:
        raise ValueError("answer position must be non-negative")
    for bucket in buckets:
        if bucket.contains(int(position)):
            return bucket.name
    raise RuntimeError("validated answer-position buckets did not cover the position")


class VocabularyPrefixHead(nn.Module):
    """Expose a fixed vocabulary prefix of another output head."""

    def __init__(self, head: nn.Module, vocabulary_size: int) -> None:
        super().__init__()
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        self.head = head
        self.vocabulary_size = int(vocabulary_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.head(hidden_states)
        if logits.shape[-1] < self.vocabulary_size:
            raise ValueError("wrapped head has fewer rows than the requested vocabulary")
        return logits[..., : self.vocabulary_size]


class PositionConditionedVocabularyHead(nn.Module):
    """Route one decoder call to the head for its visible answer-position bucket.

    All bucket heads must return the same vocabulary width.  ``inactive_head`` is
    used for prompt and continuous-latent calls, whose logits CODI discards.  It can
    be a compressed head for an all-position deployment arm or the original head for
    the first-token-only reproduction arm.
    """

    def __init__(
        self,
        heads: Mapping[str, nn.Module],
        *,
        inactive_head: nn.Module,
        vocabulary_size: int,
        buckets: Sequence[AnswerPositionBucket] = DEFAULT_ANSWER_POSITION_BUCKETS,
    ) -> None:
        super().__init__()
        validate_position_buckets(buckets)
        expected = {bucket.name for bucket in buckets}
        if set(heads) != expected:
            missing = sorted(expected - set(heads))
            extra = sorted(set(heads) - expected)
            raise ValueError(f"position heads do not match buckets; missing={missing}, extra={extra}")
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        self.heads = nn.ModuleDict(dict(heads))
        self.inactive_head = inactive_head
        self.vocabulary_size = int(vocabulary_size)
        self.buckets = tuple(buckets)
        self._answer_position: int | None = None

    @property
    def answer_position(self) -> int | None:
        return self._answer_position

    @property
    def active_bucket(self) -> str | None:
        if self._answer_position is None:
            return None
        return answer_position_bucket(self._answer_position, self.buckets)

    def set_answer_position(self, position: int | None) -> None:
        if position is not None and int(position) < 0:
            raise ValueError("answer position must be non-negative")
        self._answer_position = None if position is None else int(position)

    def selected_head(self) -> nn.Module:
        bucket = self.active_bucket
        return self.inactive_head if bucket is None else self.heads[bucket]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.selected_head()(hidden_states)
        if logits.shape[-1] != self.vocabulary_size:
            raise ValueError(
                f"position head returned vocabulary {logits.shape[-1]}, "
                f"expected {self.vocabulary_size}"
            )
        return logits


def position_head_parameter_count(head: PositionConditionedVocabularyHead) -> int:
    """Count unique stored parameters, including the inactive head."""
    return sum(parameter.numel() for parameter in head.parameters())
