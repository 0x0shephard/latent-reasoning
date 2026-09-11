"""Reusable controls and metrics for cross-model low-rank LM-head studies.

The routines in this module deliberately operate on hidden states and linear
readout weights only.  Model- and dataset-specific code belongs in the notebook
runner.  This separation lets the same baselines be applied to GPT-2, Qwen,
Llama, Gemma, and other causal language models without copying numerical bases
between incompatible hidden spaces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mech.global_low_rank_head import NestedLowRankVocabularyHead


@dataclass(frozen=True)
class FactorizationReport:
    method: str
    hidden_size: int
    vocabulary_size: int
    rank: int
    randomized_width: int
    power_iterations: int
    scale_min: float
    scale_max: float
    singular_values: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def randomized_scaled_svd_factors(
    readout_weight: torch.Tensor,
    rank: int,
    *,
    input_scale: torch.Tensor | None = None,
    centre: torch.Tensor | None = None,
    readout_bias: torch.Tensor | None = None,
    method: str = "weight_svd",
    oversample: int = 16,
    power_iterations: int = 1,
    seed: int = 0,
    compute_device: torch.device | str | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, FactorizationReport]:
    """Approximate ``W diag(scale)`` and undo the scale in the down factor.

    ``input_scale=None`` gives ordinary weight-only truncated SVD.  A diagonal
    activation scale gives an ASVD-style control.  This function intentionally
    labels that control ``ASVD-style`` rather than claiming bit-for-bit parity
    with an external reference implementation.
    """
    if readout_weight.ndim != 2:
        raise ValueError("readout_weight must have shape [vocabulary, hidden]")
    vocabulary_size, hidden_size = readout_weight.shape
    rank = int(rank)
    if not 0 < rank <= hidden_size:
        raise ValueError("rank must be in [1, hidden_size]")
    if power_iterations < 0:
        raise ValueError("power_iterations cannot be negative")
    if readout_bias is not None and readout_bias.shape != (vocabulary_size,):
        raise ValueError("readout_bias must contain one value per vocabulary row")

    device = torch.device(compute_device) if compute_device is not None else readout_weight.device
    weight = readout_weight.detach().to(device=device, dtype=compute_dtype)
    if input_scale is None:
        scale = torch.ones(hidden_size, device=device, dtype=compute_dtype)
    else:
        if input_scale.shape != (hidden_size,):
            raise ValueError("input_scale must contain one value per hidden channel")
        scale = input_scale.detach().to(device=device, dtype=compute_dtype).clamp_min(1e-6)
    if centre is None:
        origin = torch.zeros(hidden_size, device=device, dtype=compute_dtype)
    else:
        if centre.shape != (hidden_size,):
            raise ValueError("centre must contain one value per hidden channel")
        origin = centre.detach().to(device=device, dtype=compute_dtype)

    width = min(hidden_size, rank + max(0, int(oversample)))
    generator_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    omega = torch.randn(hidden_size, width, generator=generator, device=device, dtype=compute_dtype)

    def right(matrix: torch.Tensor) -> torch.Tensor:
        return weight @ (scale.unsqueeze(1) * matrix)

    def transpose(matrix: torch.Tensor) -> torch.Tensor:
        return scale.unsqueeze(1) * (weight.T @ matrix)

    sample = right(omega)
    for _ in range(int(power_iterations)):
        left = torch.linalg.qr(sample, mode="reduced").Q
        right_basis = torch.linalg.qr(transpose(left), mode="reduced").Q
        sample = right(right_basis)
    left = torch.linalg.qr(sample, mode="reduced").Q
    small = transpose(left).T
    small_left, singular_values, right_t = torch.linalg.svd(small, full_matrices=False)
    left = left @ small_left[:, :rank]
    singular_values = singular_values[:rank]
    right_t = right_t[:rank]

    up_weight = left * singular_values.unsqueeze(0)
    down_weight = right_t / scale.unsqueeze(0)
    bias = None if readout_bias is None else readout_bias.detach().to(device=device, dtype=compute_dtype)
    output_bias = F.linear(origin, weight, bias)
    report = FactorizationReport(
        method=str(method),
        hidden_size=int(hidden_size),
        vocabulary_size=int(vocabulary_size),
        rank=rank,
        randomized_width=int(width),
        power_iterations=int(power_iterations),
        scale_min=float(scale.min().detach().cpu()),
        scale_max=float(scale.max().detach().cpu()),
        singular_values=tuple(float(value) for value in singular_values.detach().cpu()),
    )
    target_device, target_dtype = readout_weight.device, readout_weight.dtype
    return (
        origin.to(device=target_device, dtype=target_dtype),
        down_weight.to(device=target_device, dtype=target_dtype),
        up_weight.to(device=target_device, dtype=target_dtype),
        output_bias.to(device=target_device, dtype=target_dtype),
        report,
    )


@torch.no_grad()
def diagonal_activation_scale(
    states: torch.Tensor,
    *,
    exponent: float = 0.5,
    minimum_relative: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state centre and a robust RMS-derived diagonal activation scale."""
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("states must be [examples, hidden] with at least two rows")
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("exponent must be in [0, 1]")
    values = states.detach().float()
    centre = values.mean(dim=0)
    rms = (values - centre).square().mean(dim=0).sqrt()
    floor = max(float(rms.mean()) * float(minimum_relative), 1e-6)
    scale = rms.clamp_min(floor).pow(float(exponent))
    scale = scale / scale.log().mean().exp().clamp_min(1e-6)
    return centre.to(states), scale.to(states)


@torch.no_grad()
def random_low_rank_head(
    readout_weight: torch.Tensor,
    states: torch.Tensor,
    rank: int,
    *,
    readout_bias: torch.Tensor | None = None,
    seed: int = 0,
) -> NestedLowRankVocabularyHead:
    """Create a reproducible learned-factorization control with random factors."""
    if states.ndim != 2 or states.shape[1] != readout_weight.shape[1]:
        raise ValueError("state and readout dimensions do not match")
    vocabulary_size, hidden_size = readout_weight.shape
    rank = int(rank)
    module = NestedLowRankVocabularyHead(hidden_size, vocabulary_size, (rank,)).to(
        device=readout_weight.device, dtype=readout_weight.dtype
    )
    generator_device = readout_weight.device.type if readout_weight.device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    centre = states.detach().to(readout_weight).mean(dim=0)
    down = torch.randn(
        rank, hidden_size, generator=generator,
        device=readout_weight.device, dtype=readout_weight.dtype,
    ) / math.sqrt(hidden_size)
    up = torch.randn(
        vocabulary_size, rank, generator=generator,
        device=readout_weight.device, dtype=readout_weight.dtype,
    ) / math.sqrt(rank)
    module.down.weight.copy_(down)
    module.down.bias.copy_(-(down @ centre))
    module.up.weight.copy_(up)
    dense_bias = None if readout_bias is None else readout_bias.detach().to(readout_weight)
    module.up.bias.copy_(F.linear(centre, readout_weight, dense_bias))
    return module


def head_from_factors(
    centre: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    output_bias: torch.Tensor,
) -> NestedLowRankVocabularyHead:
    rank = int(down_weight.shape[0])
    return NestedLowRankVocabularyHead.from_whitened_factors(
        centre, down_weight, up_weight, output_bias, (rank,)
    )


def rare_token_mask_from_counts(
    token_counts: torch.Tensor,
    *,
    maximum_count: int = 2,
) -> torch.Tensor:
    if token_counts.ndim != 1:
        raise ValueError("token_counts must be a vector")
    return token_counts.to(dtype=torch.long) <= int(maximum_count)


@torch.no_grad()
def evaluate_head_quality(
    candidate: nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
    dense_weight: torch.Tensor,
    *,
    dense_bias: torch.Tensor | None = None,
    rare_token_mask: torch.Tensor | None = None,
    batch_size: int = 8,
    top_k: int = 5,
    temperature: float = 1.0,
) -> dict[str, float | int | None]:
    """Evaluate distribution, ranking, target-token, and rare-token fidelity."""
    if states.ndim != 2 or targets.ndim != 1 or len(states) != len(targets):
        raise ValueError("states must be [N,d] and targets must be [N]")
    if dense_weight.shape[1] != states.shape[1]:
        raise ValueError("dense readout and state widths do not match")
    if not 0 < int(top_k) <= dense_weight.shape[0]:
        raise ValueError("top_k must be within the vocabulary")
    try:
        parameter = next(candidate.parameters())
        device, dtype = parameter.device, parameter.dtype
    except StopIteration:
        device, dtype = dense_weight.device, dense_weight.dtype
    weight = dense_weight.detach().to(device=device, dtype=dtype)
    bias = None if dense_bias is None else dense_bias.detach().to(device=device, dtype=dtype)
    rare = None if rare_token_mask is None else rare_token_mask.to(device=device, dtype=torch.bool)

    total = agreements = topk_intersections = rare_gold = rare_teacher = rare_teacher_agree = 0
    dense_nll = candidate_nll = kl_sum = margin_error = 0.0
    rare_dense_nll = rare_candidate_nll = 0.0
    dense_margins: list[torch.Tensor] = []
    candidate_margins: list[torch.Tensor] = []
    candidate.eval()
    for start in range(0, len(states), int(batch_size)):
        hidden = states[start:start + int(batch_size)].to(device=device, dtype=dtype)
        gold = targets[start:start + int(batch_size)].to(device=device, dtype=torch.long)
        dense = F.linear(hidden, weight, bias).float()
        observed = candidate(hidden).float()
        dense_logp = F.log_softmax(dense, dim=-1)
        observed_logp = F.log_softmax(observed, dim=-1)
        dense_top = dense.topk(k=int(top_k), dim=-1)
        observed_top = observed.topk(k=int(top_k), dim=-1)
        dense_margin = dense_top.values[:, 0] - dense_top.values[:, 1]
        observed_margin = observed_top.values[:, 0] - observed_top.values[:, 1]
        count = len(hidden)
        total += count
        dense_nll += float(F.nll_loss(dense_logp, gold, reduction="sum"))
        candidate_nll += float(F.nll_loss(observed_logp, gold, reduction="sum"))
        dense_probability = F.softmax(dense / float(temperature), dim=-1)
        kl_sum += float(F.kl_div(
            F.log_softmax(observed / float(temperature), dim=-1),
            dense_probability, reduction="sum",
        )) * float(temperature) ** 2
        dense_argmax = dense_top.indices[:, 0]
        observed_argmax = observed_top.indices[:, 0]
        agreements += int((dense_argmax == observed_argmax).sum())
        overlap = (
            dense_top.indices.unsqueeze(2) == observed_top.indices.unsqueeze(1)
        ).any(dim=2).sum(dim=1)
        topk_intersections += int(overlap.sum())
        margin_error += float((dense_margin - observed_margin).abs().sum())
        dense_margins.append(dense_margin.cpu())
        candidate_margins.append(observed_margin.cpu())
        if rare is not None:
            gold_rare = rare[gold]
            if bool(gold_rare.any()):
                rare_gold += int(gold_rare.sum())
                rare_dense_nll += float(F.nll_loss(dense_logp[gold_rare], gold[gold_rare], reduction="sum"))
                rare_candidate_nll += float(F.nll_loss(observed_logp[gold_rare], gold[gold_rare], reduction="sum"))
            teacher_rare = rare[dense_argmax]
            if bool(teacher_rare.any()):
                rare_teacher += int(teacher_rare.sum())
                rare_teacher_agree += int((dense_argmax[teacher_rare] == observed_argmax[teacher_rare]).sum())

    dense_margin_all = torch.cat(dense_margins).double()
    candidate_margin_all = torch.cat(candidate_margins).double()
    if len(dense_margin_all) > 1 and float(dense_margin_all.std()) > 0 and float(candidate_margin_all.std()) > 0:
        correlation = float(torch.corrcoef(torch.stack((dense_margin_all, candidate_margin_all)))[0, 1])
    else:
        correlation = None
    dense_mean_nll = dense_nll / max(1, total)
    candidate_mean_nll = candidate_nll / max(1, total)
    return {
        "tokens": int(total),
        "dense_nll": dense_mean_nll,
        "candidate_nll": candidate_mean_nll,
        "dense_perplexity": math.exp(min(50.0, dense_mean_nll)),
        "candidate_perplexity": math.exp(min(50.0, candidate_mean_nll)),
        "perplexity_ratio": math.exp(min(50.0, candidate_mean_nll - dense_mean_nll)),
        "teacher_kl": kl_sum / max(1, total),
        "top1_agreement": agreements / max(1, total),
        "topk_overlap_fraction": topk_intersections / max(1, total * int(top_k)),
        "mean_absolute_margin_error": margin_error / max(1, total),
        "margin_correlation": correlation,
        "rare_gold_tokens": int(rare_gold),
        "rare_dense_perplexity": None if rare_gold == 0 else math.exp(min(50.0, rare_dense_nll / rare_gold)),
        "rare_candidate_perplexity": None if rare_gold == 0 else math.exp(min(50.0, rare_candidate_nll / rare_gold)),
        "rare_teacher_top_tokens": int(rare_teacher),
        "rare_teacher_top1_agreement": None if rare_teacher == 0 else rare_teacher_agree / rare_teacher,
    }


def head_macs(hidden_size: int, vocabulary_size: int, rank: int | None = None) -> int:
    if rank is None:
        return int(hidden_size) * int(vocabulary_size)
    return int(rank) * (int(hidden_size) + int(vocabulary_size))


@torch.no_grad()
def benchmark_head_latency(
    module: nn.Module,
    hidden_size: int,
    *,
    batch_size: int,
    iterations: int,
    device: torch.device | str,
    dtype: torch.dtype,
    warmup: int = 30,
) -> float:
    device = torch.device(device)
    module.eval()
    sample = torch.randn(int(batch_size), int(hidden_size), device=device, dtype=dtype)
    for _ in range(int(warmup)):
        module(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(int(iterations)):
        module(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1e6 * (time.perf_counter() - started) / max(1, int(iterations))
