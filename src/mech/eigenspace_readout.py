"""Reusable low-rank vocabulary heads for eigenspace experiments.

The helpers in this module deliberately separate three questions:

1. Which hidden-state basis is selected?
2. Is the resulting head kept fixed or learned by logit distillation?
3. Does the replacement preserve token rankings and downstream accuracy?

Keeping those questions separate makes the CODI/SlimSpec comparison controlled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def covariance_eigensystem(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, descending covariance eigenvalues, and column eigenvectors."""
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("states must have shape [examples, hidden] with at least two rows")
    values = states.detach().double()
    mean = values.mean(dim=0)
    centred = values - mean
    covariance = centred.T @ centred / (values.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return mean, eigenvalues[order], eigenvectors[:, order]


def orthonormal_random_basis(hidden_size: int, rank: int, *, seed: int) -> torch.Tensor:
    if not 0 < rank <= hidden_size:
        raise ValueError("rank must be in [1, hidden_size]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sample = torch.randn(hidden_size, rank, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(sample, mode="reduced").Q


@torch.no_grad()
def readout_aware_scores(
    readout_weight: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Score directions by activation energy times relative-logit energy.

    A direction that moves every vocabulary logit equally cannot change an argmax.
    Subtracting the vocabulary mean removes this common-mode component. Processing
    directions in chunks avoids materialising a full ``vocabulary x hidden`` product.
    """
    if readout_weight.ndim != 2:
        raise ValueError("readout_weight must have shape [vocabulary, hidden]")
    if eigenvectors.shape != (readout_weight.shape[1], readout_weight.shape[1]):
        raise ValueError("eigenvectors/readout hidden dimensions do not match")
    if eigenvalues.shape != (readout_weight.shape[1],):
        raise ValueError("eigenvalues must contain one value per hidden direction")
    device = readout_weight.device
    result = torch.empty(eigenvalues.numel(), dtype=torch.float64, device="cpu")
    weight = readout_weight.detach()
    for start in range(0, eigenvalues.numel(), int(chunk_size)):
        stop = min(start + int(chunk_size), eigenvalues.numel())
        basis = eigenvectors[:, start:stop].to(device=device, dtype=weight.dtype)
        detector_response = weight @ basis
        relative = detector_response - detector_response.mean(dim=0, keepdim=True)
        # Float64 is prohibitively slow on T4-class GPUs and unnecessary for this
        # vocabulary-averaged ranking statistic. Accumulate in float32, then store
        # the small vector in float64 on CPU for stable ordering.
        response_energy = relative.float().square().mean(dim=0).double().cpu()
        result[start:stop] = eigenvalues[start:stop].detach().double().cpu().clamp_min(0) * response_energy
    return result


def select_readout_aware_basis(
    readout_weight: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    rank: int,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = readout_aware_scores(
        readout_weight, eigenvalues, eigenvectors, chunk_size=chunk_size
    )
    indices = torch.argsort(scores, descending=True)[: int(rank)]
    return eigenvectors[:, indices], indices, scores


class LowRankVocabularyHead(nn.Module):
    """Two-stage full-vocabulary head with an explicit centring correction."""

    def __init__(self, hidden_size: int, vocabulary_size: int, rank: int) -> None:
        super().__init__()
        if not 0 < rank <= hidden_size:
            raise ValueError("rank must be in [1, hidden_size]")
        self.hidden_size = int(hidden_size)
        self.vocabulary_size = int(vocabulary_size)
        self.rank = int(rank)
        self.down = nn.Linear(self.hidden_size, self.rank, bias=True)
        self.up = nn.Linear(self.rank, self.vocabulary_size, bias=True)

    @classmethod
    @torch.no_grad()
    def from_basis(
        cls,
        readout_weight: torch.Tensor,
        basis: torch.Tensor,
        centre: torch.Tensor,
    ) -> "LowRankVocabularyHead":
        """Construct ``mu W^T + ((h-mu)U)(WU)^T`` exactly."""
        if readout_weight.ndim != 2 or basis.ndim != 2 or centre.ndim != 1:
            raise ValueError("expected weight [V,d], basis [d,r], centre [d]")
        vocabulary_size, hidden_size = readout_weight.shape
        if basis.shape[0] != hidden_size or centre.numel() != hidden_size:
            raise ValueError("basis/centre hidden dimensions do not match the readout")
        module = cls(hidden_size, vocabulary_size, basis.shape[1]).to(
            device=readout_weight.device, dtype=readout_weight.dtype
        )
        selected = basis.to(device=readout_weight.device, dtype=readout_weight.dtype)
        origin = centre.to(device=readout_weight.device, dtype=readout_weight.dtype)
        module.down.weight.copy_(selected.T)
        module.down.bias.copy_(-(origin @ selected))
        module.up.weight.copy_(readout_weight @ selected)
        module.up.bias.copy_(origin @ readout_weight.T)
        return module

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden_states))


@dataclass(frozen=True)
class DistillationResult:
    losses: tuple[float, ...]
    best_validation_kl: float
    best_epoch: int


def _fidelity_metrics(
    head: nn.Module,
    states: torch.Tensor,
    readout_weight: torch.Tensor,
    *,
    batch_size: int,
    temperature: float,
) -> dict[str, float]:
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    total_kl = 0.0
    total_agreement = 0
    total = 0
    head.eval()
    with torch.no_grad():
        for start in range(0, states.shape[0], int(batch_size)):
            hidden = states[start : start + int(batch_size)].to(device=device, dtype=dtype)
            teacher = F.linear(hidden, readout_weight.to(device=device, dtype=dtype))
            student = head(hidden)
            teacher_prob = F.softmax(teacher.float() / temperature, dim=-1)
            student_log_prob = F.log_softmax(student.float() / temperature, dim=-1)
            kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
            count = hidden.shape[0]
            total_kl += float(kl) * count
            total_agreement += int((student.argmax(-1) == teacher.argmax(-1)).sum())
            total += count
    return {
        "kl": total_kl / max(1, total),
        "top1_agreement": total_agreement / max(1, total),
    }


def distil_low_rank_head(
    head: LowRankVocabularyHead,
    train_states: torch.Tensor,
    validation_states: torch.Tensor,
    readout_weight: torch.Tensor,
    *,
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    temperature: float = 2.0,
    anchor_strength: float = 1e-4,
    seed: int = 0,
) -> DistillationResult:
    """Fit a low-rank head to the frozen full head and restore the best epoch."""
    if train_states.ndim != 2 or validation_states.ndim != 2:
        raise ValueError("train and validation states must be matrices")
    if train_states.shape[1] != head.hidden_size or validation_states.shape[1] != head.hidden_size:
        raise ValueError("state width does not match the head")
    if readout_weight.shape != (head.vocabulary_size, head.hidden_size):
        raise ValueError("readout weight shape does not match the head")
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    teacher_weight = readout_weight.detach().to(device=device, dtype=dtype)
    initial_down = head.down.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(learning_rate), weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    losses: list[float] = []
    best_validation_kl = math.inf
    best_epoch = -1
    best_state = None

    for epoch in range(int(epochs)):
        order = torch.randperm(train_states.shape[0], generator=generator)
        total_loss = 0.0
        seen = 0
        head.train()
        for start in range(0, order.numel(), int(batch_size)):
            index = order[start : start + int(batch_size)]
            hidden = train_states[index].to(device=device, dtype=dtype)
            with torch.no_grad():
                teacher = F.linear(hidden, teacher_weight)
                teacher_prob = F.softmax(teacher.float() / temperature, dim=-1)
            student = head(hidden)
            distillation = F.kl_div(
                F.log_softmax(student.float() / temperature, dim=-1),
                teacher_prob,
                reduction="batchmean",
            ) * (temperature * temperature)
            anchor = (head.down.weight - initial_down).square().mean()
            loss = distillation + float(anchor_strength) * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = hidden.shape[0]
            total_loss += float(loss.detach()) * count
            seen += count
        losses.append(total_loss / max(1, seen))
        validation = _fidelity_metrics(
            head,
            validation_states,
            teacher_weight,
            batch_size=batch_size,
            temperature=temperature,
        )
        if validation["kl"] < best_validation_kl:
            best_validation_kl = validation["kl"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    if best_state is None:
        raise RuntimeError("distillation did not produce a checkpoint")
    head.load_state_dict(best_state)
    return DistillationResult(tuple(losses), best_validation_kl, best_epoch)


@torch.no_grad()
def evaluate_head_fidelity(
    head: nn.Module,
    states: torch.Tensor,
    readout_weight: torch.Tensor,
    *,
    batch_size: int = 32,
    temperature: float = 2.0,
) -> dict[str, float]:
    return _fidelity_metrics(
        head, states, readout_weight, batch_size=batch_size, temperature=temperature
    )


@torch.no_grad()
def benchmark_vocabulary_head(
    head: nn.Module,
    hidden_size: int,
    *,
    batch_size: int,
    iterations: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Return median microseconds for one vocabulary-head call."""
    hidden = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    head.eval()
    for _ in range(10):
        head(hidden)
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(int(iterations)):
        started = time.perf_counter()
        head(hidden)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return 1e6 * float(torch.tensor(samples).median())
