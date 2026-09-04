"""Trajectory-aware, activation-whitened global vocabulary heads.

The components here are intentionally independent of CODI.  Any causal language
model that exposes a final hidden state and a linear output embedding can use the
same fitting procedure; model-specific code is responsible only for collecting
representative hidden states and replacing the output module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WhitenedInitializationReport:
    examples: int
    hidden_size: int
    vocabulary_size: int
    rank: int
    ridge: float
    randomized_width: int
    power_iterations: int
    singular_values: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MarginDistillationResult:
    losses: tuple[float, ...]
    validation: tuple[dict[str, float], ...]
    best_epoch: int
    best_top1_agreement: float
    best_validation_kl: float


def _validate_ranks(ranks: Sequence[int], hidden_size: int) -> tuple[int, ...]:
    values = tuple(int(rank) for rank in ranks)
    if not values or any(rank <= 0 or rank > hidden_size for rank in values):
        raise ValueError("ranks must be in [1, hidden_size]")
    if tuple(sorted(set(values))) != values:
        raise ValueError("ranks must be strictly increasing")
    return values


@torch.no_grad()
def activation_cholesky(
    states: torch.Tensor,
    *,
    ridge_relative: float = 1e-4,
    compute_device: torch.device | str | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the state mean and ``S`` such that ``covariance + ridge I = S S^T``."""
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("states must be [examples, hidden] with at least two examples")
    if ridge_relative <= 0:
        raise ValueError("ridge_relative must be positive")
    device = torch.device(compute_device) if compute_device is not None else states.device
    values = states.detach().to(device=device, dtype=compute_dtype)
    centre = values.mean(dim=0)
    centred = values - centre
    covariance = centred.T @ centred / max(1, values.shape[0] - 1)
    average_variance = float(covariance.diagonal().mean().detach().cpu())
    ridge = max(float(ridge_relative) * max(average_variance, 1e-12), 1e-12)
    identity = torch.eye(covariance.shape[0], device=device, dtype=compute_dtype)
    for _ in range(6):
        factor, info = torch.linalg.cholesky_ex(covariance + ridge * identity)
        if int(info.max()) == 0:
            return centre, factor, ridge
        ridge *= 10.0
    raise RuntimeError("activation covariance remained non-positive after ridge escalation")


@torch.no_grad()
def activation_whitened_factors(
    states: torch.Tensor,
    readout_weight: torch.Tensor,
    rank: int,
    *,
    readout_bias: torch.Tensor | None = None,
    ridge_relative: float = 1e-4,
    oversample: int = 16,
    power_iterations: int = 1,
    seed: int = 0,
    compute_device: torch.device | str | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, WhitenedInitializationReport]:
    """Fit a randomized truncated SVD of ``W S`` without materialising ``W S``.

    With centred state covariance ``C = S S^T``, minimizing expected squared
    logit error is equivalent to minimizing ``||(W - W_hat) S||_F``.  The returned
    factors implement ``W_hat = up @ down`` and preserve the original logits at
    the state mean through an output bias.
    """
    if states.ndim != 2 or readout_weight.ndim != 2:
        raise ValueError("states and readout_weight must be matrices")
    vocabulary_size, hidden_size = readout_weight.shape
    if states.shape[1] != hidden_size:
        raise ValueError("state width and readout width do not match")
    rank = int(rank)
    if not 0 < rank <= hidden_size:
        raise ValueError("rank must be in [1, hidden_size]")
    if readout_bias is not None and readout_bias.shape != (vocabulary_size,):
        raise ValueError("readout_bias must have one value per vocabulary row")
    if power_iterations < 0:
        raise ValueError("power_iterations cannot be negative")

    device = (
        torch.device(compute_device)
        if compute_device is not None
        else readout_weight.device
    )
    centre, scale, ridge = activation_cholesky(
        states,
        ridge_relative=ridge_relative,
        compute_device=device,
        compute_dtype=compute_dtype,
    )
    weight = readout_weight.detach().to(device=device, dtype=compute_dtype)
    width = min(hidden_size, rank + max(0, int(oversample)))
    generator_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    omega = torch.randn(
        hidden_size, width, generator=generator, device=device, dtype=compute_dtype
    )

    def multiply_right(matrix: torch.Tensor) -> torch.Tensor:
        return weight @ (scale @ matrix)

    def multiply_transpose(matrix: torch.Tensor) -> torch.Tensor:
        return scale.T @ (weight.T @ matrix)

    sample = multiply_right(omega)
    for _ in range(int(power_iterations)):
        left = torch.linalg.qr(sample, mode="reduced").Q
        right = torch.linalg.qr(multiply_transpose(left), mode="reduced").Q
        sample = multiply_right(right)
    left = torch.linalg.qr(sample, mode="reduced").Q
    small = multiply_transpose(left).T
    small_left, singular_values, right_t = torch.linalg.svd(
        small, full_matrices=False
    )
    left = left @ small_left[:, :rank]
    singular_values = singular_values[:rank]
    right = right_t[:rank].T

    # U Sigma is a better-conditioned split for inference than explicitly forming
    # S^{-1}.  Solve S^T X = V, so X^T = V^T S^{-1}.
    up_weight = left * singular_values.unsqueeze(0)
    down_transpose = torch.linalg.solve_triangular(scale.T, right, upper=True)
    down_weight = down_transpose.T
    bias = (
        None
        if readout_bias is None
        else readout_bias.detach().to(device=device, dtype=compute_dtype)
    )
    output_bias = F.linear(centre, weight, bias)
    down_bias = -(down_weight @ centre)
    report = WhitenedInitializationReport(
        examples=int(states.shape[0]),
        hidden_size=int(hidden_size),
        vocabulary_size=int(vocabulary_size),
        rank=rank,
        ridge=float(ridge),
        randomized_width=int(width),
        power_iterations=int(power_iterations),
        singular_values=tuple(float(value) for value in singular_values.detach().cpu()),
    )
    target_device = readout_weight.device
    target_dtype = readout_weight.dtype
    return (
        centre.to(device=target_device, dtype=target_dtype),
        down_weight.to(device=target_device, dtype=target_dtype),
        up_weight.to(device=target_device, dtype=target_dtype),
        output_bias.to(device=target_device, dtype=target_dtype),
        report,
    )


class NestedLowRankVocabularyHead(nn.Module):
    """One global vocabulary head whose ordered prefixes implement several ranks."""

    def __init__(
        self,
        hidden_size: int,
        vocabulary_size: int,
        ranks: Sequence[int],
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocabulary_size = int(vocabulary_size)
        self.ranks = _validate_ranks(ranks, self.hidden_size)
        self.max_rank = self.ranks[-1]
        self.down = nn.Linear(self.hidden_size, self.max_rank, bias=True)
        self.up = nn.Linear(self.max_rank, self.vocabulary_size, bias=True)
        self.active_rank = self.max_rank
        self.fallback_rank: int | None = None
        self.margin_threshold: float | None = None
        self.inactive_rank: int | None = None
        self._answer_position: int | None = None
        self.last_fallback_fraction = 0.0

    @classmethod
    @torch.no_grad()
    def from_whitened_factors(
        cls,
        centre: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        output_bias: torch.Tensor,
        ranks: Sequence[int],
    ) -> "NestedLowRankVocabularyHead":
        if centre.ndim != 1 or down_weight.ndim != 2 or up_weight.ndim != 2:
            raise ValueError("invalid factor shapes")
        max_rank, hidden_size = down_weight.shape
        vocabulary_size, up_rank = up_weight.shape
        checked = _validate_ranks(ranks, hidden_size)
        if up_rank != max_rank or checked[-1] != max_rank:
            raise ValueError("factor rank must equal the largest configured rank")
        if output_bias.shape != (vocabulary_size,) or centre.shape != (hidden_size,):
            raise ValueError("centre or output bias shape mismatch")
        module = cls(hidden_size, vocabulary_size, checked).to(
            device=up_weight.device, dtype=up_weight.dtype
        )
        module.down.weight.copy_(down_weight)
        module.down.bias.copy_(-(down_weight @ centre.to(down_weight)))
        module.up.weight.copy_(up_weight)
        module.up.bias.copy_(output_bias)
        return module

    def set_rank(self, rank: int) -> None:
        if int(rank) not in self.ranks:
            raise ValueError(f"rank must be one of {self.ranks}")
        self.active_rank = int(rank)

    def configure_adaptive(
        self,
        *,
        base_rank: int,
        fallback_rank: int,
        margin_threshold: float,
        inactive_rank: int | None = None,
    ) -> None:
        if base_rank not in self.ranks or fallback_rank not in self.ranks:
            raise ValueError("adaptive ranks must be configured prefixes")
        if fallback_rank <= base_rank:
            raise ValueError("fallback_rank must exceed base_rank")
        if margin_threshold < 0:
            raise ValueError("margin_threshold cannot be negative")
        if inactive_rank is not None and inactive_rank not in self.ranks:
            raise ValueError("inactive_rank must be a configured prefix")
        self.active_rank = int(base_rank)
        self.fallback_rank = int(fallback_rank)
        self.margin_threshold = float(margin_threshold)
        self.inactive_rank = inactive_rank

    def disable_adaptive(self) -> None:
        self.fallback_rank = None
        self.margin_threshold = None
        self.inactive_rank = None

    def set_answer_position(self, position: int | None) -> None:
        self._answer_position = None if position is None else int(position)

    def forward_rank(self, hidden_states: torch.Tensor, rank: int) -> torch.Tensor:
        rank = int(rank)
        if rank not in self.ranks:
            raise ValueError(f"rank must be one of {self.ranks}")
        coordinates = F.linear(
            hidden_states,
            self.down.weight[:rank],
            self.down.bias[:rank],
        )
        return F.linear(coordinates, self.up.weight[:, :rank], self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._answer_position is None and self.inactive_rank is not None:
            self.last_fallback_fraction = 0.0
            return self.forward_rank(hidden_states, self.inactive_rank)
        logits = self.forward_rank(hidden_states, self.active_rank)
        if self.fallback_rank is None or self.margin_threshold is None:
            self.last_fallback_fraction = 0.0
            return logits
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_logits = logits.reshape(-1, logits.shape[-1])
        top_two = flat_logits.topk(k=2, dim=-1).values
        uncertain = (top_two[:, 0] - top_two[:, 1]) < self.margin_threshold
        self.last_fallback_fraction = float(uncertain.float().mean().detach().cpu())
        if bool(uncertain.any()):
            start, stop = self.active_rank, self.fallback_rank
            extra_coordinates = F.linear(
                flat_hidden[uncertain], self.down.weight[start:stop],
                self.down.bias[start:stop],
            )
            correction = F.linear(
                extra_coordinates, self.up.weight[:, start:stop], bias=None
            )
            flat_logits = flat_logits.clone()
            flat_logits[uncertain] += correction
            logits = flat_logits.reshape_as(logits)
        return logits


def _margin_loss(logits: torch.Tensor, teacher_top: torch.Tensor, margin: float) -> torch.Tensor:
    selected = logits.gather(1, teacher_top.unsqueeze(1)).squeeze(1)
    top_two = logits.topk(k=2, dim=-1)
    competitor = torch.where(
        top_two.indices[:, 0] == teacher_top,
        top_two.values[:, 1],
        top_two.values[:, 0],
    )
    return F.relu(float(margin) - selected + competitor).mean()


@torch.no_grad()
def evaluate_nested_head(
    head: NestedLowRankVocabularyHead,
    states: torch.Tensor,
    readout_weight: torch.Tensor,
    *,
    readout_bias: torch.Tensor | None = None,
    rank: int | None = None,
    batch_size: int = 8,
    temperature: float = 2.0,
) -> dict[str, float]:
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    weight = readout_weight.detach().to(device=device, dtype=dtype)
    bias = None if readout_bias is None else readout_bias.detach().to(device=device, dtype=dtype)
    chosen_rank = head.max_rank if rank is None else int(rank)
    total = 0
    total_kl = 0.0
    agreements = 0
    margin_absolute_error = 0.0
    head.eval()
    for start in range(0, states.shape[0], int(batch_size)):
        hidden = states[start : start + int(batch_size)].to(device=device, dtype=dtype)
        teacher = F.linear(hidden, weight, bias).float()
        student = head.forward_rank(hidden, chosen_rank).float()
        teacher_probability = F.softmax(teacher / temperature, dim=-1)
        kl = F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            teacher_probability,
            reduction="batchmean",
        )
        teacher_top_two = teacher.topk(k=2, dim=-1).values
        student_top_two = student.topk(k=2, dim=-1).values
        count = hidden.shape[0]
        total += count
        total_kl += float(kl) * count
        agreements += int((student.argmax(-1) == teacher.argmax(-1)).sum())
        margin_absolute_error += float(
            ((student_top_two[:, 0] - student_top_two[:, 1])
             - (teacher_top_two[:, 0] - teacher_top_two[:, 1])).abs().sum()
        )
    return {
        "kl": total_kl / max(1, total),
        "top1_agreement": agreements / max(1, total),
        "mean_absolute_top2_margin_error": margin_absolute_error / max(1, total),
    }


def distil_nested_head(
    head: NestedLowRankVocabularyHead,
    train_states: torch.Tensor,
    validation_states: torch.Tensor,
    readout_weight: torch.Tensor,
    *,
    readout_bias: torch.Tensor | None = None,
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    temperature: float = 2.0,
    kl_weight: float = 1.0,
    token_weight: float = 0.25,
    margin_weight: float = 0.25,
    nested_weight: float = 0.5,
    minimum_margin: float = 0.25,
    anchor_strength: float = 1e-5,
    seed: int = 0,
) -> MarginDistillationResult:
    """Distil teacher distributions and decisions into every configured rank prefix."""
    if train_states.ndim != 2 or validation_states.ndim != 2:
        raise ValueError("training and validation states must be matrices")
    if train_states.shape[1] != head.hidden_size or validation_states.shape[1] != head.hidden_size:
        raise ValueError("state width does not match head")
    if readout_weight.shape != (head.vocabulary_size, head.hidden_size):
        raise ValueError("readout shape does not match head")
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    teacher_weight = readout_weight.detach().to(device=device, dtype=dtype)
    teacher_bias = (
        None if readout_bias is None
        else readout_bias.detach().to(device=device, dtype=dtype)
    )
    initial_down = head.down.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(learning_rate), weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    losses: list[float] = []
    validation_history: list[dict[str, float]] = []
    best_epoch = -1
    best_key = (-math.inf, -math.inf)
    best_state = None

    for epoch in range(int(epochs)):
        order = torch.randperm(train_states.shape[0], generator=generator)
        head.train()
        epoch_loss = 0.0
        seen = 0
        for start in range(0, order.numel(), int(batch_size)):
            index = order[start : start + int(batch_size)]
            hidden = train_states[index].to(device=device, dtype=dtype)
            with torch.no_grad():
                teacher = F.linear(hidden, teacher_weight, teacher_bias).float()
                teacher_probability = F.softmax(teacher / temperature, dim=-1)
                teacher_top = teacher.argmax(dim=-1)

            def prefix_loss(rank: int) -> torch.Tensor:
                student = head.forward_rank(hidden, rank).float()
                divergence = F.kl_div(
                    F.log_softmax(student / temperature, dim=-1),
                    teacher_probability,
                    reduction="batchmean",
                ) * (temperature * temperature)
                token = F.cross_entropy(student, teacher_top)
                ranking = _margin_loss(student, teacher_top, minimum_margin)
                return (
                    float(kl_weight) * divergence
                    + float(token_weight) * token
                    + float(margin_weight) * ranking
                )

            loss = prefix_loss(head.max_rank)
            smaller = head.ranks[:-1]
            if smaller:
                auxiliary = torch.stack([prefix_loss(rank) for rank in smaller]).mean()
                loss = loss + float(nested_weight) * auxiliary
            loss = loss + float(anchor_strength) * (
                head.down.weight - initial_down
            ).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = hidden.shape[0]
            epoch_loss += float(loss.detach()) * count
            seen += count
        losses.append(epoch_loss / max(1, seen))
        metrics = evaluate_nested_head(
            head,
            validation_states,
            teacher_weight,
            readout_bias=teacher_bias,
            rank=head.max_rank,
            batch_size=batch_size,
            temperature=temperature,
        )
        validation_history.append(metrics)
        key = (metrics["top1_agreement"], -metrics["kl"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("distillation did not produce a checkpoint")
    head.load_state_dict(best_state)
    best = validation_history[best_epoch]
    return MarginDistillationResult(
        losses=tuple(losses),
        validation=tuple(validation_history),
        best_epoch=best_epoch,
        best_top1_agreement=float(best["top1_agreement"]),
        best_validation_kl=float(best["kl"]),
    )
