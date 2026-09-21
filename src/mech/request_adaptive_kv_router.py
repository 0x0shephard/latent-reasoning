"""Pre-answer request routing among frozen, storage-matched xKV profiles.

The router deliberately uses only lexical properties of the question.  It never
sees the gold answer, a generated token, or a post-answer hidden state.  Profile
labels are learned on a disjoint fit split from first-token KL to the dense CODI
model, then frozen before the screen and final splits are evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Sequence

import torch


FEATURE_NAMES = (
    "log_characters",
    "log_words",
    "log_numbers",
    "mean_digits_per_number",
    "decimal_fraction",
    "sentence_count",
    "additive_cues",
    "subtractive_cues",
    "multiplicative_cues",
    "division_rate_cues",
    "comparison_cues",
    "money_percent_cues",
)

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z]+")


def _cue_count(words: list[str], cues: set[str]) -> int:
    return sum(word in cues for word in words)


def prompt_feature_vector(question: str) -> torch.Tensor:
    """Return fixed, label-free numerical features available before decoding."""
    text = str(question).strip()
    words = [word.casefold() for word in _WORD.findall(text)]
    numbers = _NUMBER.findall(text)
    digit_counts = [sum(character.isdigit() for character in value) for value in numbers]
    additive = _cue_count(words, {"add", "added", "altogether", "sum", "total", "more"})
    subtractive = _cue_count(words, {"difference", "fewer", "left", "less", "remain", "remaining"})
    multiplicative = _cue_count(words, {"double", "each", "product", "times", "triple", "twice"})
    division = _cue_count(words, {"average", "divide", "divided", "equal", "per", "rate", "share"})
    comparison = _cue_count(words, {"than", "least", "most", "greater", "smaller", "larger"})
    money_percent = sum(character in text for character in "$%") + _cue_count(
        words, {"cent", "cents", "dollar", "dollars", "percent", "percentage"}
    )
    values = (
        math.log1p(len(text)),
        math.log1p(len(words)),
        math.log1p(len(numbers)),
        sum(digit_counts) / max(1, len(digit_counts)),
        sum("." in value for value in numbers) / max(1, len(numbers)),
        max(1, sum(text.count(mark) for mark in ".?!")),
        additive,
        subtractive,
        multiplicative,
        division,
        comparison,
        money_percent,
    )
    return torch.tensor(values, dtype=torch.float64)


def prompt_feature_matrix(questions: Sequence[str]) -> torch.Tensor:
    if not questions:
        raise ValueError("at least one question is required")
    return torch.stack([prompt_feature_vector(question) for question in questions])


def deterministic_question_split(
    rows: Sequence[dict], *, seed: int, sizes: Sequence[int]
) -> tuple[list[dict], ...]:
    """Hash-sort rows into disjoint, order-independent fixed-size splits."""
    if not sizes or any(int(size) <= 0 for size in sizes):
        raise ValueError("split sizes must be positive")
    if sum(int(size) for size in sizes) != len(rows):
        raise ValueError("split sizes must exhaust the rows")
    decorated = []
    for index, row in enumerate(rows):
        question = " ".join(str(row["question"]).strip().split())
        digest = hashlib.sha256(f"{int(seed)}\0{question}".encode()).hexdigest()
        decorated.append((digest, index, dict(row)))
    ordered = [item[2] for item in sorted(decorated)]
    result = []
    start = 0
    for size in sizes:
        stop = start + int(size)
        result.append(ordered[start:stop])
        start = stop
    return tuple(result)


@dataclass(frozen=True)
class RidgeLossRouter:
    """Multi-output ridge model predicting dense-fidelity loss per profile."""

    profile_names: tuple[str, ...]
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    coefficients: torch.Tensor
    ridge: float

    def predict_losses(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features.double() - self.feature_mean) / self.feature_scale
        design = torch.cat(
            (torch.ones(len(standardized), 1, dtype=torch.float64), standardized), dim=1
        )
        return design @ self.coefficients

    def route(self, features: torch.Tensor) -> torch.Tensor:
        return self.predict_losses(features).argmin(dim=1)

    def audit(self) -> dict:
        return {
            "profile_names": list(self.profile_names),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "ridge": float(self.ridge),
        }


def fit_ridge_loss_router(
    features: torch.Tensor,
    profile_losses: torch.Tensor,
    *,
    profile_names: Sequence[str],
    ridge: float = 1.0,
) -> RidgeLossRouter:
    if features.ndim != 2 or profile_losses.ndim != 2:
        raise ValueError("features and profile losses must be matrices")
    if len(features) != len(profile_losses) or not len(features):
        raise ValueError("features and losses must have matching non-empty rows")
    if profile_losses.shape[1] != len(profile_names) or not profile_names:
        raise ValueError("profile names must match loss columns")
    if ridge <= 0 or not bool(torch.isfinite(features).all()) or not bool(
        torch.isfinite(profile_losses).all()
    ):
        raise ValueError("router inputs and ridge must be finite and positive")
    x = features.double()
    # Remove request-wide difficulty.  Routing depends only on which profile is
    # relatively best for a request, not on whether all profiles find it hard.
    raw_y = profile_losses.double()
    y = raw_y - raw_y.mean(dim=1, keepdim=True)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-8)
    standardized = (x - mean) / scale
    design = torch.cat((torch.ones(len(x), 1, dtype=torch.float64), standardized), dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeLossRouter(
        profile_names=tuple(str(name) for name in profile_names),
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        ridge=float(ridge),
    )


def per_example_dense_kl(dense_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    if dense_logits.shape != candidate_logits.shape or dense_logits.ndim != 2:
        raise ValueError("logits must share [examples, vocabulary]")
    dense_log = dense_logits.double().log_softmax(-1)
    candidate_log = candidate_logits.double().log_softmax(-1)
    return (dense_log.exp() * (dense_log - candidate_log)).sum(-1).cpu()


def route_distribution(route_indices: torch.Tensor, profile_names: Sequence[str]) -> dict:
    counts = torch.bincount(route_indices.cpu(), minlength=len(profile_names))
    total = int(counts.sum())
    probabilities = counts.double() / max(1, total)
    entropy = float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
    return {
        "counts": {str(name): int(counts[index]) for index, name in enumerate(profile_names)},
        "fractions": {
            str(name): float(probabilities[index]) for index, name in enumerate(profile_names)
        },
        "entropy_nats": entropy,
        "active_profiles": int((counts > 0).sum()),
    }
