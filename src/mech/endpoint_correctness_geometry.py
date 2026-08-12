"""Correctness geometry at CODI's answer cue: detect, steer, and project.

The band experiment asked which directions *determine* the answer and found PCs
4-31 (11% of the colon-state variance, 88% of the accuracy). Splitting the same
states by whether the model actually answered correctly asks a different question,
and exploratory work found the two answers are nearly disjoint:

- the correctness direction ``d = mean(right) - mean(wrong)`` sits 97% inside
  PCs 0-3, which carry only 6% of the accuracy;
- PCs 0-3 are the near-uniform logit-lift directions, so a constant shift along
  them cannot change an argmax;
- consistent with that, steering along ``d`` moved held-out accuracy by at most
  0.38 points and was harmful beyond one class-mean step.

This module implements the three preregistered follow-up tracks:

``detect``
    Does a linear read of the state predict correctness *beyond* the model's own
    margin? A probe that scores well alone is worthless if the margin already
    scores better, so every gate here is stated as an increment over margin-only.

``steer``
    Is the band a handle or only a location? Steering is confined to the
    directions the readout is actually sensitive to, which is where the earlier
    global attempt provably could not work.

``project``
    Does building the retention subspace from correct examples only beat the
    class-blind one? Exploratory principal angles were 0.98-0.99, so this is a
    preregistered replication of an expected null rather than a hopeful arm.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE
from src.mech.endpoint_margin_geometry import ANALYTIC_STATE


CORRECTNESS_SCHEMA_VERSION = 1
CORRECTNESS_CONTRACT = "frozen_checkpoint_answer_colon_correctness_tracks_v1"

TRACKS = ("detect", "steer", "project")

#: The accuracy-bearing band confirmed by the exact-match run, zero-based.
ACCURACY_BAND = (4, 32)
#: The leading, near-uniform-lift directions the correctness signal lives in.
LIFT_BAND = (0, 4)


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------


def first_token_correct(
    states: torch.Tensor, readout: torch.Tensor, gold: torch.Tensor
) -> torch.Tensor:
    """Whether the released decoder's greedy first answer token is the gold one.

    Exact for state 12 because ``lm_head`` is a bias-free linear map of it; the
    margin-geometry parity gate is what licenses using this instead of decoding.
    """
    if states.ndim != 2 or states.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("states must be [N, 768]")
    if readout.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("readout must be [V, 768]")
    if gold.shape[0] != states.shape[0]:
        raise ValueError("gold tokens must be paired with states")
    return (states.double() @ readout.double().T).argmax(dim=-1) == gold.to(
        states.device
    )


def answer_margin(
    states: torch.Tensor, readout: torch.Tensor
) -> torch.Tensor:
    """Gap between the best and second-best word score — the model's own confidence."""
    logits = states.double() @ readout.double().T
    top2 = logits.topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


#: The key ``collect_official_codi_endpoint_margin_states.py`` writes the output
#: embedding under. Named once here so a reader cannot quietly disagree with the
#: producer; ``tests/test_correctness_tracks_integration.py`` asserts the two match.
READOUT_KEY = "readout"


def readout_matrix(payload: dict) -> torch.Tensor:
    """Pull the output embedding out of a readout export, or say what is wrong."""
    if READOUT_KEY not in payload:
        raise KeyError(
            f"readout export has no {READOUT_KEY!r} entry; found {sorted(payload)}. "
            "Attach the export written by "
            "scripts/collect_official_codi_endpoint_margin_states.py."
        )
    matrix = payload[READOUT_KEY]
    if matrix.ndim != 2 or matrix.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError(f"readout must be [V, 768], got {tuple(matrix.shape)}")
    return matrix


def sorted_eigenbasis(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    symmetric = 0.5 * (covariance.double() + covariance.double().T)
    values, vectors = torch.linalg.eigh(symmetric)
    order = torch.argsort(values, descending=True)
    return values[order], vectors[:, order]


def band_projector(vectors: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """``P = U U^T`` for a contiguous slice of a sorted eigenbasis."""
    if not 0 <= start < stop <= vectors.shape[1]:
        raise ValueError("band bounds are out of range")
    basis = vectors[:, start:stop].double()
    return basis @ basis.T


# ---------------------------------------------------------------------------
# track 1 — detect
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectnessDirections:
    """Directions separating correct from incorrect answers, fitted on one split."""

    mean_difference: torch.Tensor
    fisher: torch.Tensor
    correct_mean: torch.Tensor
    within_covariance: torch.Tensor
    between_variance: float
    total_variance: float

    @property
    def between_fraction(self) -> float:
        return float(self.between_variance / max(self.total_variance, 1e-12))


def fit_correctness_directions(
    states: torch.Tensor,
    correct: torch.Tensor,
    *,
    shrinkage: float = 0.05,
) -> CorrectnessDirections:
    """Mean-difference and Fisher directions, plus the class-variance split.

    The raw mean difference is dominated by whichever directions happen to have
    the most spread, which is why it landed in the lift band. The Fisher direction
    divides that difference by the within-class covariance, so a direction only
    scores highly if the classes separate *relative to* how much they each vary.
    Shrinkage toward the identity is not optional at this size: 768 dimensions
    against roughly a thousand examples leaves the within-class covariance close
    to singular, and inverting it unshrunk amplifies sampling noise far more than
    it removes nuisance variance. The right amount is data-dependent, so callers
    should choose it on a held-out split via :func:`select_fisher_shrinkage`
    rather than trusting the default.
    """
    if states.ndim != 2:
        raise ValueError("states must be [N, d]")
    if correct.dtype != torch.bool or correct.shape[0] != states.shape[0]:
        raise ValueError("correct must be a paired boolean vector")
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError("shrinkage must be in [0, 1)")
    if int(correct.sum()) < 2 or int((~correct).sum()) < 2:
        raise ValueError("both classes need at least two examples")

    values = states.double()
    dimension = values.shape[1]
    right, wrong = values[correct], values[~correct]
    difference = right.mean(0) - wrong.mean(0)

    centred = torch.cat([right - right.mean(0), wrong - wrong.mean(0)], dim=0)
    within = centred.T @ centred / centred.shape[0]
    trace = float(torch.diagonal(within).mean())
    within = (1 - shrinkage) * within + shrinkage * trace * torch.eye(
        dimension, dtype=torch.float64, device=values.device
    )
    fisher = torch.linalg.solve(within, difference)

    overall = values - values.mean(0)
    total = float((overall * overall).sum() / values.shape[0])
    share = float(correct.double().mean())
    between = share * (1 - share) * float(difference.norm()) ** 2
    return CorrectnessDirections(
        mean_difference=difference / difference.norm(),
        fisher=fisher / fisher.norm(),
        correct_mean=right.mean(0),
        within_covariance=within,
        between_variance=between,
        total_variance=total,
    )


def midranks(scores: torch.Tensor) -> torch.Tensor:
    """Ranks with ties averaged, which is what a tie-aware AUC needs.

    A plain double ``argsort`` breaks ties by position, so a degenerate probe that
    assigns every example the same score would score an arbitrary AUC determined
    by the order the questions happen to arrive in rather than the 0.5 it has
    earned.
    """
    values = scores.double().flatten()
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(
        1, values.numel() + 1, dtype=torch.float64, device=values.device
    )
    sorted_values = values[order]
    start = 0
    for index in range(1, sorted_values.numel() + 1):
        if index == sorted_values.numel() or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                tied = order[start:index]
                ranks[tied] = ranks[tied].mean()
            start = index
    return ranks


def select_fisher_shrinkage(
    fit_states: torch.Tensor,
    fit_correct: torch.Tensor,
    select_states: torch.Tensor,
    select_correct: torch.Tensor,
    *,
    grid: tuple[float, ...],
) -> tuple[float, dict[str, float]]:
    """Pick the shrinkage whose Fisher direction separates best on a held-out split.

    Choosing it on the fitting split would always favour the least-regularised
    option, which is precisely the one that overfits at this sample-to-dimension
    ratio.
    """
    if not grid:
        raise ValueError("a shrinkage grid is required")
    scores = {}
    best, best_separation = None, -1.0
    for shrinkage in grid:
        directions = fit_correctness_directions(
            fit_states, fit_correct, shrinkage=shrinkage
        )
        auc = roc_auc(select_states.double() @ directions.fisher, select_correct)
        scores[f"{shrinkage:g}"] = auc
        # Orientation is arbitrary -- a direction scoring 0.2 separates the classes
        # exactly as well as one scoring 0.8 -- so rank on distance from chance,
        # which is zero for a useless direction rather than for a perfect one.
        separation = abs(auc - 0.5)
        if separation > best_separation:
            best, best_separation = shrinkage, separation
    return float(best), scores


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC; 0.5 is chance, and 1 − AUC flips the score's sign."""
    labels = labels.double().flatten()
    positives = labels.sum()
    negatives = (1 - labels).sum()
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both classes present")
    ranks = midranks(scores)
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def fit_logistic(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    l2: float = 0.01,
    steps: int = 700,
    learning_rate: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Deterministic ridge-regularised logistic probe.

    Features are standardised with the fitting split's own statistics, which are
    returned so the evaluation split is transformed identically rather than
    re-standardised against itself.
    """
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must be paired")
    values = features.double()
    mean, scale = values.mean(0), values.std(0).clamp_min(1e-8)
    standard = (values - mean) / scale
    target = labels.double().flatten()

    weight = torch.zeros(
        standard.shape[1], dtype=torch.float64, device=standard.device,
        requires_grad=True,
    )
    bias = torch.zeros(
        1, dtype=torch.float64, device=standard.device, requires_grad=True
    )
    optimiser = torch.optim.Adam([weight, bias], lr=learning_rate)
    for _ in range(steps):
        optimiser.zero_grad()
        predicted = torch.sigmoid(standard @ weight + bias)
        loss = torch.nn.functional.binary_cross_entropy(predicted, target)
        loss = loss + l2 * (weight * weight).sum()
        loss.backward()
        optimiser.step()
    return weight.detach(), bias.detach(), {"mean": mean, "scale": scale}


def apply_logistic(
    features: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stats: dict
) -> torch.Tensor:
    standard = (features.double() - stats["mean"]) / stats["scale"]
    return standard @ weight + bias


# ---------------------------------------------------------------------------
# track 2 — steer
# ---------------------------------------------------------------------------


def margin_gradient(
    states: torch.Tensor, readout: torch.Tensor, gold: torch.Tensor
) -> torch.Tensor:
    """Per-example ``w_gold − w_runner_up``: the direction that widens the margin.

    With the runner-up held fixed the margin is linear in the state, so this is
    the exact gradient rather than a first-order approximation.
    """
    logits = states.double() @ readout.double().T
    rows = torch.arange(logits.shape[0], device=logits.device)
    target = gold.to(logits.device).long()
    masked = logits.clone()
    masked[rows, target] = torch.finfo(logits.dtype).min
    runner_up = masked.argmax(dim=-1)
    return readout.double()[target] - readout.double()[runner_up]


def build_steering_vectors(
    *,
    states: torch.Tensor,
    readout: torch.Tensor,
    gold: torch.Tensor,
    directions: CorrectnessDirections,
    eigenvectors: torch.Tensor,
    band: tuple[int, int] = ACCURACY_BAND,
    random_seed: int = 0,
    random_replicates: int = 8,
) -> dict[str, torch.Tensor]:
    """Candidate steering vectors, all fitted on the supplied split only.

    ``margin_band`` is the interesting one: the average margin-widening direction
    confined to the accuracy band. A global steering vector can only help if the
    model carries a systematic bias, and the earlier attempt failed because it
    pointed almost entirely into directions the readout ignores. Restricting it to
    the band is the version that has somewhere to act.
    """
    projector = band_projector(eigenvectors, *band)
    gradients = margin_gradient(states, readout, gold)
    average = gradients.mean(0)

    vectors = {
        "mean_difference_global": directions.mean_difference,
        "mean_difference_band": projector @ directions.mean_difference,
        "fisher_band": projector @ directions.fisher,
        "margin_global": average,
        "margin_band": projector @ average,
    }
    # Drawn on the CPU and moved, never seeded per device: a CUDA generator would
    # produce a different control set from a CPU one, so the same arm name would
    # denote a different direction depending on where the sweep happened to run.
    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    for replicate in range(random_replicates):
        raw = torch.randn(
            GPT2_HIDDEN_SIZE, generator=generator, dtype=torch.float64, device="cpu"
        ).to(projector.device)
        vectors[f"random_band_r{replicate:02d}"] = projector @ raw
        vectors[f"random_global_r{replicate:02d}"] = raw
    return {
        name: value / value.norm().clamp_min(1e-12) for name, value in vectors.items()
    }


def steer(states: torch.Tensor, vector: torch.Tensor, alpha: float) -> torch.Tensor:
    """``h' = h + alpha * v`` with ``v`` a unit vector, so alpha is in state units."""
    return states.double() + float(alpha) * vector.double().unsqueeze(0)


def steered_accuracy(
    base_logits: torch.Tensor,
    readout: torch.Tensor,
    gold: torch.Tensor,
    vector: torch.Tensor,
    alpha: float,
) -> tuple[float, torch.Tensor]:
    """Accuracy under ``h + alpha*v``, without recomputing the logits.

    Steering is a constant translation and the readout is linear, so

        (h + alpha*v) W^T  =  h W^T  +  alpha * (W v)

    which turns one [N, 768] x [768, 50257] product per (arm, step) into a single
    [50257] vector per arm plus a broadcast add. The full grid is ~150 such
    evaluations; done naively in float64 that is hours on a T4, and it is exact
    either way rather than an approximation traded for speed.
    """
    shift = (readout.double() @ vector.double()).unsqueeze(0)
    outcomes = (base_logits + float(alpha) * shift).argmax(dim=-1) == gold.to(
        base_logits.device
    )
    return float(outcomes.double().mean()), outcomes


def retained_accuracy(
    states: torch.Tensor,
    readout: torch.Tensor,
    gold: torch.Tensor,
    basis: torch.Tensor,
    centre: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Accuracy after keeping only ``basis``, evaluated in its low-rank form.

    ``mu W^T + ((h - mu) U)(U^T W^T)`` costs ``(N + 768) k V`` instead of
    ``N * 768 * V``; at rank 28 that is a 27-fold saving, and the two forms agree
    to floating-point error.
    """
    columns, origin = basis.double(), centre.double()
    projected = (states.double() - origin.unsqueeze(0)) @ columns
    logits = (origin @ readout.double().T).unsqueeze(0) + projected @ (
        columns.T @ readout.double().T
    )
    outcomes = logits.argmax(dim=-1) == gold.to(logits.device)
    return float(outcomes.double().mean()), outcomes


class OfficialCODIEndpointSteerIntervention:
    """Add a fixed vector to the colon state during real generation.

    Deliberately separate from the projection intervention used by the confirmed
    band experiment: that class is frozen, and an additive edit is a different
    operation from a subspace projection.  Like it, this touches only the final
    layer norm's output at the answer cue, so nothing reaches the key/value cache
    and only the first answer token's logits can change.
    """

    applies_to_all_positions = False

    def __init__(self, model, vector: torch.Tensor, *, alpha: float) -> None:
        if vector.shape != (GPT2_HIDDEN_SIZE,):
            raise ValueError("steering vector must have shape [768]")
        if not torch.isfinite(vector).all():
            raise ValueError("steering vector must be finite")
        norm = float(vector.double().norm())
        if abs(norm - 1.0) > 1e-4:
            raise ValueError("steering vector must be unit length; alpha carries scale")
        transformer = getattr(model.codi, "base_model", model.codi)
        transformer = getattr(transformer, "model", transformer)
        transformer = getattr(transformer, "transformer", transformer)
        final_layer_norm = getattr(transformer, "ln_f", None)
        blocks = getattr(transformer, "h", None)
        if final_layer_norm is None or blocks is None or len(blocks) != 12:
            raise RuntimeError("unexpected GPT-2 module layout")
        self.module = final_layer_norm
        self.vector = vector.detach().cpu().float()
        self.alpha = float(alpha)
        self.active_mask: torch.Tensor | None = None
        self.calls = 0
        self.rows = 0

    def _hook(self):
        def edit(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3 or hidden.shape[-1] != GPT2_HIDDEN_SIZE:
                raise ValueError("GPT-2 block output shape changed")
            mask = self.active_mask.to(device=hidden.device)
            if mask.shape != (hidden.shape[0],):
                raise ValueError("steer intervention mask has the wrong batch shape")
            shift = self.alpha * self.vector.to(device=hidden.device)
            edited = hidden.clone()
            last = edited[:, -1, :].float()
            last[mask] = last[mask] + shift.unsqueeze(0)
            edited[:, -1, :] = last.to(dtype=hidden.dtype)
            self.calls += 1
            self.rows += int(mask.sum().item())
            if isinstance(output, tuple):
                return (edited, *output[1:])
            if isinstance(output, list):
                return [edited, *output[1:]]
            return edited

        return edit

    @contextmanager
    def activate(self, mask: torch.Tensor):
        if self.active_mask is not None:
            raise RuntimeError("steer intervention cannot be nested")
        self.active_mask = mask.detach()
        handle = self.module.register_forward_hook(self._hook())
        try:
            yield self
        finally:
            handle.remove()
            self.active_mask = None

    def diagnostics(self) -> dict:
        return {
            "alpha": self.alpha,
            "state": ANALYTIC_STATE,
            "calls": self.calls,
            "rows": self.rows,
            "vector_norm": float(self.vector.double().norm()),
        }


# ---------------------------------------------------------------------------
# track 3 — project
# ---------------------------------------------------------------------------


def retention(
    states: torch.Tensor, basis: torch.Tensor, centre: torch.Tensor
) -> torch.Tensor:
    """Keep only the span of ``basis``; everything else becomes the centre."""
    values, columns = states.double(), basis.double()
    centred = values - centre.double().unsqueeze(0)
    return centre.double().unsqueeze(0) + (centred @ columns) @ columns.T


def principal_angle_cosines(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosines of the principal angles between two orthonormal subspaces."""
    if a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must live in the same ambient space")
    return torch.linalg.svdvals(a.double().T @ b.double())


def class_conditional_basis(
    states: torch.Tensor, correct: torch.Tensor, rank: int, *, on_correct: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``rank`` principal directions of one class, with that class's centre."""
    mask = correct if on_correct else ~correct
    if int(mask.sum()) <= rank:
        raise ValueError("not enough examples in the requested class")
    values = states.double()[mask]
    centre = values.mean(0)
    centred = values - centre
    covariance = centred.T @ centred / centred.shape[0]
    _, vectors = sorted_eigenbasis(covariance)
    return vectors[:, :rank], centre


def band_variance_shares(
    values: torch.Tensor, bands: tuple[tuple[int, int], ...]
) -> dict[str, float]:
    total = float(values.sum())
    return {
        f"{start}:{stop}": float(values[start:stop].sum() / max(total, 1e-12))
        for start, stop in bands
    }


def direction_band_profile(
    direction: torch.Tensor,
    eigenvectors: torch.Tensor,
    bands: tuple[tuple[int, int], ...] = (LIFT_BAND, ACCURACY_BAND, (32, 768)),
) -> dict[str, float]:
    """How a unit direction distributes its length across the eigen-bands.

    A random split of the same data already concentrates around 70% in the leading
    band, so this number is only meaningful against that null — never on its own.
    """
    unit = direction.double() / direction.double().norm().clamp_min(1e-12)
    coefficients = eigenvectors.double().T @ unit
    return {
        f"{start}:{stop}": float((coefficients[start:stop] ** 2).sum())
        for start, stop in bands
    }


def random_split_null(
    states: torch.Tensor,
    correct: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    replicates: int,
    seed: int,
    band: tuple[int, int] = LIFT_BAND,
) -> dict:
    """The mean-difference profile under random splits of the same class sizes.

    Any split of a dataset produces a mean difference that leans toward
    high-variance directions, so this is the control that decides whether the
    observed concentration means anything.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    positives = int(correct.sum())
    norms, shares = [], []
    for _ in range(replicates):
        order = torch.randperm(states.shape[0], generator=generator, device="cpu")
        mask = torch.zeros(states.shape[0], dtype=torch.bool, device="cpu")
        mask[order[:positives]] = True
        mask = mask.to(states.device)
        difference = states.double()[mask].mean(0) - states.double()[~mask].mean(0)
        norms.append(float(difference.norm()))
        shares.append(direction_band_profile(difference, eigenvectors, (band,))[
            f"{band[0]}:{band[1]}"
        ])
    return {"replicates": replicates, "norms": norms, "band_shares": shares}
