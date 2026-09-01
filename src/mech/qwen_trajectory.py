"""Pure helpers for aligning autoregressive trajectories with final answers."""
from __future__ import annotations

import re
from collections.abc import Sequence


_NUMBER = re.compile(
    r"[-+]?(?:\$\s*)?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?%?"
)
_FINAL_MARKER = re.compile(r"(?:final\s+answer|answer)\s*(?:is|:|=)", re.IGNORECASE)


def _last_balanced_box(text: str) -> tuple[int, int] | None:
    """Return the content span of the last balanced ``\\boxed{...}``."""
    marker = r"\boxed{"
    search_to = len(text)
    while True:
        start = text.rfind(marker, 0, search_to)
        if start < 0:
            return None
        content_start = start + len(marker)
        depth = 1
        for position in range(content_start, len(text)):
            character = text[position]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    left = content_start
                    right = position
                    while left < right and text[left].isspace():
                        left += 1
                    while right > left and text[right - 1].isspace():
                        right -= 1
                    return (left, right) if left < right else None
        search_to = start


def final_answer_span(text: str) -> tuple[int, int, str] | None:
    """Locate the final answer's character span and the rule that found it.

    Preference order is a balanced ``\\boxed{...}``, a number following the last
    explicit final-answer marker, and finally the last number in the response.
    The fallback makes alignment auditable without claiming that the response is
    correctly formatted.
    """
    boxed = _last_balanced_box(text)
    if boxed is not None:
        return boxed[0], boxed[1], "boxed"

    markers = list(_FINAL_MARKER.finditer(text))
    if markers:
        candidates = list(_NUMBER.finditer(text, markers[-1].end()))
        if candidates:
            match = candidates[-1]
            return match.start(), match.end(), "final_marker"

    numbers = list(_NUMBER.finditer(text))
    if numbers:
        match = numbers[-1]
        return match.start(), match.end(), "last_number_fallback"
    return None


def token_indices_overlapping_span(
    offsets: Sequence[Sequence[int]], span: Sequence[int]
) -> list[int]:
    """Return token indices with a non-empty overlap with ``[start, stop)``."""
    if len(span) < 2:
        raise ValueError("span must contain start and stop")
    start, stop = int(span[0]), int(span[1])
    if not 0 <= start < stop:
        raise ValueError("span must satisfy 0 <= start < stop")
    result = []
    for index, offset in enumerate(offsets):
        if len(offset) < 2:
            raise ValueError("each token offset must contain start and stop")
        token_start, token_stop = int(offset[0]), int(offset[1])
        if token_stop > start and token_start < stop:
            result.append(index)
    return result


def evenly_spaced_indices(length: int, maximum: int) -> list[int]:
    """Select at most ``maximum`` deterministic indices including both ends."""
    length, maximum = int(length), int(maximum)
    if length < 0 or maximum <= 0:
        raise ValueError("length must be non-negative and maximum must be positive")
    if length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [length - 1]
    return [round(step * (length - 1) / (maximum - 1)) for step in range(maximum)]
