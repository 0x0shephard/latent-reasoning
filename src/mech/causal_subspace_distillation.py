"""Causal versus variance selection of a distillation subspace (ledger §86).

Subspace-distillation methods pick the teacher subspace by variance, second
moments, or gradient relevance.  §40 showed the decision state's top-variance
directions are inert.  This module builds four rank-``r`` index sets over the
teacher's own principal components at its decision state, evaluates them on the
teacher by retain-only intervention (the §36 analytic tier), and provides the
student-side distillation loss and the adapter reset that makes the student learn
the latent task from scratch while sharing the teacher's readout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from src.mech.endpoint_tsvc import match_gradient_norm
from src.mech.kv_target_utility import autograd_gradients, combine_gradients
from src.mech.trajectory_supervision import student_trajectory_forward, teacher_endpoint_states
from src.models.official_codi import official_codi_base_model


READOUT_STATE = -1  # the last hidden state (post-ln_f, index 12 on GPT-2) that feeds the readout
ARMS = ("none", "full", "variance", "relevance", "causal", "random")
SUBSPACE_ARMS = ("variance", "relevance", "causal", "random")


# ------------------------------------------------------------------ teacher side


def teacher_decision_states(model, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher state 12 at the answer cue ``[B,D]`` and the gold first answer token ``[B]``."""
    states = teacher_endpoint_states(model, batch)[:, READOUT_STATE, :]
    row = torch.arange(states.shape[0], device=batch.teacher_ids.device)
    gold = batch.teacher_ids[row, batch.teacher_answer_start.to(batch.teacher_ids.device)]
    return states.detach(), gold.detach()


def readout_matrix(model, *, vocab_limit: int) -> torch.Tensor:
    """The tied output matrix restricted to the original vocabulary, ``[V,D]``."""
    weight = official_codi_base_model(model).get_output_embeddings().weight
    return weight[: int(vocab_limit)].detach()


@dataclass
class TeacherPCA:
    mean: torch.Tensor  # [D]
    basis: torch.Tensor  # [D,D], columns are PCs in descending eigenvalue order
    eigenvalues: torch.Tensor  # [D]

    def coordinates(self, states: torch.Tensor, index_set: Sequence[int]) -> torch.Tensor:
        return (states.to(self.basis.dtype) - self.mean) @ self.basis[:, list(index_set)]


def fit_teacher_pca(states: torch.Tensor) -> TeacherPCA:
    if states.ndim != 2 or states.shape[0] < states.shape[1]:
        raise ValueError("need at least D states of width D to fit the PCA")
    values = states.double()
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    order = torch.arange(eigenvalues.shape[0] - 1, -1, -1)
    return TeacherPCA(mean=mean, basis=vectors[:, order].contiguous(),
                      eigenvalues=eigenvalues[order].clamp_min(0))


def _log_probs(states: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
    # Evaluations run in float32 on the readout's device: greedy selection makes
    # thousands of readout passes and float64 is unusably slow on consumer GPUs.
    return torch.log_softmax(states.to(readout.device, torch.float32) @ readout.T, dim=-1)


def retained_states(states: torch.Tensor, pca: TeacherPCA, index_set: Sequence[int]) -> torch.Tensor:
    """Retain-only edit ``mu + V_S V_S^T (h - mu)``; the empty set returns the mean."""
    mean = pca.mean.to(states.device, torch.float32)
    values = states.to(torch.float32)
    if not index_set:
        return mean.expand_as(values)
    basis = pca.basis[:, list(index_set)].to(states.device, torch.float32)
    centered = values - mean
    return mean + (centered @ basis) @ basis.T


def gold_outcomes(states: torch.Tensor, gold: torch.Tensor, readout: torch.Tensor,
                  *, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example gold log-probability and top-1 correctness under the readout."""
    log_probs, correct = [], []
    readout = readout.to(torch.float32)
    for start in range(0, states.shape[0], chunk):
        piece = _log_probs(states[start : start + chunk], readout)
        target = gold[start : start + chunk].to(piece.device)
        log_probs.append(piece.gather(1, target[:, None]).squeeze(1).double().cpu())
        correct.append((piece.argmax(-1) == target).cpu())
    return torch.cat(log_probs), torch.cat(correct)


def retain_only_outcomes(states, gold, pca, index_set, readout):
    return gold_outcomes(retained_states(states, pca, index_set), gold, readout)


def margin_outcomes(states: torch.Tensor, gold: torch.Tensor, readout: torch.Tensor,
                    *, chunk: int = 512) -> torch.Tensor:
    """Gold logit minus the best competing logit, per example (the §36 margin)."""
    margins = []
    readout = readout.to(torch.float32)
    for start in range(0, states.shape[0], chunk):
        logits = states[start : start + chunk].to(readout.device, torch.float32) @ readout.T
        target = gold[start : start + chunk].to(logits.device)
        gold_logit = logits.gather(1, target[:, None]).squeeze(1)
        masked = logits.scatter(1, target[:, None], float("-inf"))
        margins.append((gold_logit - masked.max(-1).values).double().cpu())
    return torch.cat(margins)


def retain_only_score(states, gold, pca, index_set, readout) -> tuple[float, float, float]:
    """(accuracy, mean margin, mean gold log-prob) of the retain-only edited teacher.

    Accuracy leads and margin breaks ties.  Mean gold log-probability is reported
    but not optimised: under sharp logits it is dominated by confident errors, so
    partially restored states can score below the constant mean state.
    """
    edited = retained_states(states, pca, index_set)
    log_prob, correct = gold_outcomes(edited, gold, readout)
    margin = margin_outcomes(edited, gold, readout)
    return float(correct.double().mean()), float(margin.mean()), float(log_prob.mean())


# --------------------------------------------------------------------- selectors


def select_variance(rank: int) -> list[int]:
    return list(range(int(rank)))


def select_random(rank: int, dimension: int, *, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(int(seed))
    return sorted(int(i) for i in torch.randperm(int(dimension), generator=generator)[: int(rank)])


def relevance_scores(states: torch.Tensor, gold: torch.Tensor, pca: TeacherPCA,
                     readout: torch.Tensor, *, chunk: int = 256) -> torch.Tensor:
    """Fisher-style score per PC: E[(g . v_j)^2] with g the gold-NLL gradient at state 12."""
    readout = readout.to(torch.float32)
    basis = pca.basis.to(readout.device, torch.float32)
    total = torch.zeros(pca.basis.shape[1], dtype=torch.float64)
    for start in range(0, states.shape[0], chunk):
        piece = states[start : start + chunk].to(readout.device, torch.float32).clone().requires_grad_(True)
        nll = -_log_probs(piece, readout).gather(
            1, gold[start : start + chunk].to(piece.device)[:, None]
        ).sum()
        (gradient,) = torch.autograd.grad(nll, piece)
        projected = gradient.detach() @ basis  # [n, D]
        total += projected.square().sum(0).double().cpu()
    return total / states.shape[0]


def select_relevance(states, gold, pca, readout, rank: int) -> list[int]:
    scores = relevance_scores(states, gold, pca, readout)
    return sorted(int(i) for i in torch.topk(scores, int(rank)).indices)


def select_causal_greedy(states, gold, pca, readout, rank: int, *, candidates: int = 128,
                         return_trace: bool = False):
    """Greedy forward selection by retain-only intervention on the teacher.

    Each step adds the candidate PC whose inclusion most raises the retain-only
    first-token accuracy, with the mean answer margin breaking ties (the §36
    outcomes).  This is an intervention on the teacher, not a statistic of it.
    """
    chosen: list[int] = []
    pool = list(range(min(int(candidates), pca.basis.shape[1])))
    trace = []
    for _ in range(int(rank)):
        best_key, best_index, best_detail = None, None, None
        for index in pool:
            if index in chosen:
                continue
            accuracy, margin, log_prob = retain_only_score(states, gold, pca, chosen + [index], readout)
            key = (accuracy, margin)
            if best_key is None or key > best_key:
                best_key, best_index, best_detail = key, index, (accuracy, margin, log_prob)
        chosen.append(int(best_index))
        trace.append({"added": int(best_index), "accuracy": best_detail[0],
                      "mean_margin": best_detail[1], "mean_gold_log_prob": best_detail[2]})
    # Greedy selection is nested: the first k additions are the rank-k set, so one
    # pass at the largest rank serves every smaller rank in a grid.
    result = sorted(chosen)
    return (result, trace) if return_trace else result


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    left, right = set(int(i) for i in a), set(int(i) for i in b)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def evaluate_index_sets(states, gold, pca, readout, sets: dict[str, Sequence[int]]) -> dict:
    dense_log_prob, dense_correct = gold_outcomes(states, gold, readout)
    report = {"dense": {"gold_log_prob": float(dense_log_prob.mean()),
                        "first_token_accuracy": float(dense_correct.double().mean())}}
    dense_margin = float(margin_outcomes(states, gold, readout).mean())
    report["dense"]["mean_margin"] = dense_margin
    for name, index_set in sets.items():
        accuracy, margin, log_prob_mean = retain_only_score(states, gold, pca, index_set, readout)
        log_prob, correct = retain_only_outcomes(states, gold, pca, index_set, readout)
        share = float(pca.eigenvalues[list(index_set)].sum() / pca.eigenvalues.sum()) if len(index_set) else 0.0
        report[name] = {
            "indices": [int(i) for i in index_set],
            "gold_log_prob": float(log_prob.mean()),
            "mean_margin": margin,
            "first_token_accuracy": float(correct.double().mean()),
            "retention_of_dense": (
                float(correct.double().mean() / dense_correct.double().mean())
                if float(dense_correct.double().mean()) else None
            ),
            "variance_share": share,
        }
    names = list(sets)
    report["jaccard"] = {f"{a}|{b}": jaccard(sets[a], sets[b])
                         for i, a in enumerate(names) for b in names[i + 1 :]}
    return report


def choose_rank(reports_by_rank: dict[int, dict], *, minimum_retention: float,
                minimum_gap: float) -> tuple[int | None, dict]:
    """Smallest rank whose causal set retains enough and beats variance by the gap."""
    audit = {}
    for rank in sorted(reports_by_rank):
        report = reports_by_rank[rank]
        causal = report["causal"]["retention_of_dense"] or 0.0
        variance = report["variance"]["first_token_accuracy"]
        gap = report["causal"]["first_token_accuracy"] - variance
        audit[str(rank)] = {"causal_retention": causal, "causal_minus_variance": gap,
                            "passes": causal >= minimum_retention and gap >= minimum_gap}
    for rank in sorted(reports_by_rank):
        if audit[str(rank)]["passes"]:
            return int(rank), audit
    return None, audit


# ------------------------------------------------------------------ student side


@dataclass
class DistillationTarget:
    """Frozen coordinates the student is matched on; ``basis=None`` means all 768 dims."""

    mean: torch.Tensor
    basis: torch.Tensor | None
    scale: float

    def coordinates(self, states: torch.Tensor) -> torch.Tensor:
        centered = states - self.mean.to(states.dtype)
        if self.basis is None:
            return centered
        return centered @ self.basis.to(states.dtype)

    def to(self, device):
        return DistillationTarget(
            mean=self.mean.to(device), basis=None if self.basis is None else self.basis.to(device),
            scale=self.scale,
        )


def build_target(pca: TeacherPCA, fit_states: torch.Tensor, index_set: Sequence[int] | None,
                 *, eps: float = 1e-6) -> DistillationTarget:
    """Teacher-side coordinate scale so every arm's loss is in comparable units."""
    basis = None if index_set is None else pca.basis[:, list(index_set)].float()
    target = DistillationTarget(mean=pca.mean.float(), basis=basis, scale=1.0)
    coordinates = target.coordinates(fit_states.float())
    scale = float(coordinates.std(unbiased=True).clamp_min(eps))
    return DistillationTarget(mean=target.mean, basis=basis, scale=scale)


def distillation_loss(student_state: torch.Tensor, teacher_state: torch.Tensor,
                      target: DistillationTarget) -> torch.Tensor:
    student = target.coordinates(student_state)
    teacher = target.coordinates(teacher_state.detach().to(student_state.dtype))
    return F.smooth_l1_loss(student, teacher, reduction="mean", beta=1.0) / target.scale


def reinitialize_student(model, *, seed: int) -> dict:
    """Fresh LoRA (A Kaiming-uniform, B zero) and projector; embeddings and readout kept."""
    generator = torch.Generator().manual_seed(int(seed))
    reset = {"lora_A": 0, "lora_B": 0, "projector": 0}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_A" in name:
                # PEFT's default kaiming_uniform_(a=sqrt(5)) on a [r, fan_in] matrix
                # is exactly U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
                bound = 1.0 / math.sqrt(parameter.shape[1])
                fresh = torch.empty(parameter.shape, dtype=torch.float32).uniform_(
                    -bound, bound, generator=generator
                )
                parameter.copy_(fresh.to(device=parameter.device, dtype=parameter.dtype))
                reset["lora_A"] += 1
            elif "lora_B" in name:
                parameter.zero_()
                reset["lora_B"] += 1
        for module in model.prj.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                if isinstance(module, torch.nn.Linear):
                    bound = 1.0 / math.sqrt(module.in_features)
                    module.weight.copy_(torch.empty_like(module.weight, device="cpu").uniform_(
                        -bound, bound, generator=generator).to(module.weight.device))
                    if module.bias is not None:
                        module.bias.copy_(torch.empty_like(module.bias, device="cpu").uniform_(
                            -bound, bound, generator=generator).to(module.bias.device))
                else:
                    module.reset_parameters()
                reset["projector"] += 1
    if not reset["lora_A"] or not reset["lora_B"] or not reset["projector"]:
        raise RuntimeError("student reset did not find LoRA adapters and a projector")
    return reset


@dataclass
class StudentStep:
    gradients: tuple
    answer_loss: float
    distillation_loss: float | None
    distillation_scale: float | None


def student_training_step(model, batch, teacher_state12: torch.Tensor,
                          target: DistillationTarget | None, parameters: Sequence[torch.Tensor],
                          *, latent_positions: int) -> StudentStep:
    """Answer cross-entropy plus a norm-matched state-12 distillation term."""
    student = student_trajectory_forward(model, batch, latent_positions=latent_positions)
    if target is None:
        base = autograd_gradients(student.mean_loss, parameters, retain_graph=False)
        return StudentStep(combine_gradients(base), float(student.mean_loss.detach()), None, None)
    base = autograd_gradients(student.mean_loss, parameters, retain_graph=True)
    state = student.answer_endpoint_hidden[:, READOUT_STATE, :]
    loss = distillation_loss(state, teacher_state12.to(state.device), target)
    raw = autograd_gradients(loss, parameters, retain_graph=False)
    if any(g is not None and bool(g.abs().sum() > 0) for g in raw):
        matched, matching = match_gradient_norm(raw, base)
        scale = float(matching["auxiliary_scale"])
        total = combine_gradients(base, matched)
    else:
        scale, total = 0.0, combine_gradients(base)
    return StudentStep(total, float(student.mean_loss.detach()), float(loss.detach()), scale)


def cosine_lr(step: int, *, total_steps: int, base: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))
