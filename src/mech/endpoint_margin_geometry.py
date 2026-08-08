"""Answer-margin geometry and effective dimensionality at CODI's answer colon.

The completed state-12 confirmation removed a rank-three subspace from the
``ln_f`` output at the forced answer cue and measured numeric exact match.  Three
properties of that design bound what it could ever detect:

1. ``ln_f`` runs after every transformer block, so its output never enters the
   key/value cache.  Editing it changes exactly one token's logits and nothing
   propagates.  The whole causal channel is the first answer token.
2. Binary exact match on 1,319 questions discards nearly all of the information
   the forward pass produces, so the matched-random null is dominated by
   discretization noise.
3. Every selector scored directions with a first-order gradient criterion and was
   then tested with a finite rank-three projection.

This module addresses all three.  Because GPT-2's ``lm_head`` is a bias-free
linear map applied directly to the ``ln_f`` output, a state-12 edit is *exactly*

    z' = W h' = z - (W U)(U^T (h - centre))

so the effect of any subspace on the first answer token can be evaluated in
closed form from cached colon states, without re-running the model.  That makes
continuous outcomes, large rank sweeps, and hundreds of matched controls cheap,
and it makes the margin-optimal subspace analytically solvable rather than
something a heuristic selector has to guess.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE, GPT2_STATE_COUNT


MARGIN_GEOMETRY_SCHEMA_VERSION = 1
MARGIN_GEOMETRY_CONTRACT = "frozen_checkpoint_answer_colon_margin_geometry_v1"

#: ``ln_f`` output (Hugging Face hidden-state index 12) is the only state whose
#: edit is an exact linear logit shift.  State 11 is block 10's output and does
#: reach the cache, so it is measured with real forward passes instead.
ANALYTIC_STATE = 12
PROPAGATING_STATE = 11
MARGIN_GEOMETRY_STATES = (PROPAGATING_STATE, ANALYTIC_STATE)

#: Fitted on calibration only.  ``parameter_aware`` and ``answer_conditioned``
#: are loaded from the completed immutable artifacts and exist at rank three
#: only; they are reference arms for the direct comparison with the failed
#: confirmation, not families that this experiment refits.
FITTED_FAMILIES = ("margin", "answer_nll", "energy", "readout")
REFERENCE_FAMILIES = ("answer_conditioned", "parameter_aware")

#: Removal semantics.  ``mean`` reproduces the completed experiments exactly,
#: ``zero`` also deletes the constant component the mean-preserving edit leaves
#: untouched, and ``resample`` substitutes another question's component and so
#: tests whether the model needs *this* example's value.
ABLATION_SEMANTICS = ("mean", "zero", "resample")
INTERVENTION_MODES = ("remove", "retain")

DEFAULT_RANK_GRID = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class MarginSubspace:
    """One orthonormal subspace at one endpoint state."""

    name: str
    family: str
    state: int
    basis: torch.Tensor  # [768, rank]
    rank: int
    random_replicate: int | None = None
    calibration_target_energy: float | None = None
    calibration_achieved_energy: float | None = None
    selected_overlap: float | None = None
    matched_family: str | None = None
    #: False when no selected-orthogonal subspace of this rank could reach the
    #: selection's own calibration energy, which makes the control inadmissible.
    target_attainable: bool | None = None
    #: False above rank ``hidden / 2``, where subspaces of that rank must intersect
    #: and the disjointness constraint is therefore dropped, not approximated.
    selected_orthogonal: bool | None = None

    def state_dict(self) -> dict:
        payload = {
            "name": self.name,
            "family": self.family,
            "state": self.state,
            "rank": self.rank,
            "random_replicate": self.random_replicate,
            "matched_family": self.matched_family,
            "target_attainable": self.target_attainable,
            "selected_orthogonal": self.selected_orthogonal,
        }
        for key in (
            "calibration_target_energy",
            "calibration_achieved_energy",
            "selected_overlap",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = float(value)
        return payload


def validate_margin_subspace(subspace: MarginSubspace) -> None:
    basis = subspace.basis
    if basis.ndim != 2 or basis.shape[0] != GPT2_HIDDEN_SIZE:
        raise ValueError("margin subspace basis must be [768, rank]")
    if basis.shape[1] != subspace.rank or not 0 < subspace.rank <= GPT2_HIDDEN_SIZE:
        raise ValueError("margin subspace rank is inconsistent")
    if subspace.state not in MARGIN_GEOMETRY_STATES:
        raise ValueError("margin subspace escaped states 11 and 12")
    if not torch.isfinite(basis).all():
        raise ValueError("margin subspace basis is not finite")
    gram = basis.double().T @ basis.double()
    if not torch.allclose(
        gram, torch.eye(subspace.rank, dtype=torch.float64), atol=2e-5, rtol=2e-5
    ):
        raise ValueError("margin subspace basis is not orthonormal")


# ---------------------------------------------------------------------------
# Closed-form subspace construction
# ---------------------------------------------------------------------------


def _top_eigenvectors(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Top-``rank`` eigenvectors of a symmetric matrix, descending by eigenvalue."""
    symmetric = 0.5 * (matrix.double() + matrix.double().T)
    values, vectors = torch.linalg.eigh(symmetric)
    order = torch.argsort(values, descending=True)[:rank]
    return vectors[:, order].contiguous().float()


def margin_damage_matrix(
    centered: torch.Tensor,
    gradients: torch.Tensor,
) -> torch.Tensor:
    """Symmetrised ``E[c g^T]`` whose top eigenvectors maximise margin damage.

    Removing the rank-``k`` projector ``P`` reduces example ``i``'s margin by
    ``g_i^T P c_i``, so the expected reduction is ``tr(P E[c g^T])``.  Because
    ``tr(U^T A U) == tr(U^T sym(A) U)``, the maximiser over orthonormal ``U`` is
    the top-``k`` eigenvectors of the symmetric part.  This is the exact optimum
    of the objective the experiment then measures, which is precisely what the
    earlier gradient-cosine selectors did not have.
    """
    if centered.shape != gradients.shape or centered.ndim != 2:
        raise ValueError("centered states and gradients must share shape [N, 768]")
    if centered.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("margin damage matrix expects 768-dimensional states")
    if centered.shape[0] == 0:
        raise ValueError("margin damage matrix needs at least one example")
    cross = centered.double().T @ gradients.double() / centered.shape[0]
    return (0.5 * (cross + cross.T)).float()


def state_covariance(centered: torch.Tensor) -> torch.Tensor:
    if centered.ndim != 2 or centered.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("covariance expects [N, 768] centered states")
    if centered.shape[0] == 0:
        raise ValueError("covariance needs at least one example")
    gram = centered.double().T @ centered.double() / centered.shape[0]
    return (0.5 * (gram + gram.T)).float()


def subspace_energy(covariance: torch.Tensor, basis: torch.Tensor) -> float:
    """``E[||U U^T c||^2]`` under the calibration covariance."""
    projected = basis.double().T @ covariance.double() @ basis.double()
    return float(torch.diagonal(projected).sum())


#: Principal-component bands of the student colon state, discovered analytically:
#: PCs 0-3 hold 82% of the variance but 6.7% of the accuracy, while PCs 4-31 hold
#: 11% of the variance and 86% of the accuracy.  These are the arms the exact-match
#: confirmation evaluates.
DEFAULT_CONFIRMATION_BANDS = ((0, 4), (4, 16), (4, 32), (0, 32), (32, 768))
PRIMARY_BAND = (4, 32)
CONTROL_BAND = (0, 4)


def band_name(start: int, stop: int, state: int = ANALYTIC_STATE) -> str:
    return f"band_p{start:03d}_{stop:03d}_s{state}"


def build_band_subspace(
    *,
    covariance: torch.Tensor,
    start: int,
    stop: int,
    state: int = ANALYTIC_STATE,
) -> MarginSubspace:
    """Principal components ``[start, stop)`` of the colon-state covariance.

    The ``energy`` family only ever takes a prefix ``[0, k)``.  A band is needed
    because variance rank and answer contribution are close to unrelated here: the
    leading components dominate the spectrum while contributing almost nothing to
    the answer, so the accuracy-bearing directions can only be named as an interior
    slice.
    """
    if not 0 <= start < stop <= GPT2_HIDDEN_SIZE:
        raise ValueError("band bounds must satisfy 0 <= start < stop <= 768")
    symmetric = 0.5 * (covariance.double() + covariance.double().T)
    values, vectors = torch.linalg.eigh(symmetric)
    order = torch.argsort(values, descending=True)
    basis = vectors[:, order[start:stop]].contiguous().float()
    subspace = MarginSubspace(
        name=band_name(start, stop, state),
        family="band",
        state=state,
        basis=basis,
        rank=stop - start,
        calibration_achieved_energy=subspace_energy(covariance, basis),
    )
    validate_margin_subspace(subspace)
    return subspace


def band_variance_share(covariance: torch.Tensor, start: int, stop: int) -> float:
    """Fraction of total calibration variance carried by a PC band."""
    symmetric = 0.5 * (covariance.double() + covariance.double().T)
    values = torch.linalg.eigvalsh(symmetric)
    ordered = torch.sort(values, descending=True).values
    total = float(ordered.sum())
    if total <= 0:
        raise ValueError("calibration covariance has no positive variance")
    return float(ordered[start:stop].sum() / total)


def readout_family_capacity(readout_matrix: torch.Tensor) -> int:
    """How many directions the numeric-answer readout can span.

    GPT-2 has 1,694 all-digit tokens, so this is the full 768 in practice; the
    check exists so a tokenizer with fewer numeric tokens fails with a statement
    of the cause rather than an orthonormality error.
    """
    if readout_matrix.ndim != 2 or readout_matrix.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("readout matrix must be [tokens, 768]")
    return int(min(readout_matrix.shape[0], GPT2_HIDDEN_SIZE))


def build_fitted_subspaces(
    *,
    family: str,
    rank: int,
    state: int,
    covariance: torch.Tensor,
    damage_matrix: torch.Tensor | None = None,
    readout_matrix: torch.Tensor | None = None,
) -> MarginSubspace:
    """Construct one calibration-fitted subspace of the requested family."""
    if family not in FITTED_FAMILIES:
        raise ValueError(f"unknown fitted family {family}")
    if family in {"margin", "answer_nll"}:
        if damage_matrix is None:
            raise ValueError(f"{family} requires its damage matrix")
        basis = _top_eigenvectors(damage_matrix, rank)
    elif family == "energy":
        basis = _top_eigenvectors(covariance, rank)
    else:
        if readout_matrix is None:
            raise ValueError("readout family requires the answer-token readout matrix")
        capacity = readout_family_capacity(readout_matrix)
        if rank > capacity:
            raise ValueError(
                f"the readout family spans only {capacity} directions "
                f"({readout_matrix.shape[0]} numeric answer tokens); rank {rank} "
                "is not defined"
            )
        # Right singular vectors of the numeric-answer unembedding rows: the
        # directions the model actually reads when it scores digit tokens.
        _, _, vh = torch.linalg.svd(readout_matrix.double(), full_matrices=False)
        basis = vh[:rank, :].T.contiguous().float()
    subspace = MarginSubspace(
        name=f"{family}_k{rank:03d}_s{state}",
        family=family,
        state=state,
        basis=basis,
        rank=rank,
        calibration_achieved_energy=subspace_energy(covariance, basis),
    )
    validate_margin_subspace(subspace)
    return subspace


def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(matrix.double(), mode="reduced")
    signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    return (q * signs.unsqueeze(0)).float()


def _matching_context(
    covariance: torch.Tensor,
    rank: int,
    selected_basis: torch.Tensor | None,
) -> dict:
    """Precompute everything that is constant across replicates of one target.

    The covariance eigendecomposition and the complement projector do not depend on
    the draw, so computing them once per target rather than once per bisection step
    is the difference between seconds and tens of minutes for a full control set.
    """
    values, vectors = torch.linalg.eigh(covariance.double())
    # Selected-orthogonality is only achievable while the complement is at least as
    # large as the rank.  Above ``hidden / 2`` any two subspaces of that rank must
    # intersect, so the constraint is dropped rather than approximated, and the
    # realised overlap is reported instead.  Every gated comparison happens at the
    # primary rank, where the constraint does hold.
    complement_dimension = GPT2_HIDDEN_SIZE - (
        selected_basis.shape[1] if selected_basis is not None else 0
    )
    selected_orthogonal = selected_basis is not None and rank <= complement_dimension
    constraint = selected_basis if selected_orthogonal else None
    projector = None
    if constraint is not None:
        selected = constraint.double()
        projector = torch.eye(GPT2_HIDDEN_SIZE, dtype=torch.float64) - selected @ selected.T
    minimum, maximum = attainable_energy_range(covariance, rank, constraint)
    return {
        "eigenvalues": values.clamp_min(1e-12),
        "eigenvectors": vectors,
        "covariance": covariance,
        "projector": projector,
        "rank": rank,
        "selected_orthogonal": bool(selected_orthogonal),
        "complement_dimension": int(complement_dimension),
        "attainable_minimum": minimum,
        "attainable_maximum": maximum,
    }


def _shaped_random_basis(
    context: dict,
    generator: torch.Generator,
    exponent: float,
) -> torch.Tensor:
    """Random subspace tilted toward high- or low-variance directions.

    ``exponent`` interpolates continuously from low-energy (negative) through
    isotropic (zero) to high-energy (positive) subspaces, which gives a monotone
    handle for matching a prescribed calibration energy without abandoning
    orthonormality.
    """
    vectors = context["eigenvectors"]
    scale = context["eigenvalues"].pow(0.5 * exponent)
    gaussian = torch.randn(
        GPT2_HIDDEN_SIZE, context["rank"], generator=generator, dtype=torch.float64
    )
    shaped = vectors @ (scale.unsqueeze(1) * (vectors.T @ gaussian))
    if context["projector"] is not None:
        shaped = context["projector"] @ shaped
    return _orthonormalize(shaped)


def attainable_energy_range(
    covariance: torch.Tensor,
    rank: int,
    selected_basis: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Smallest and largest calibration energy any rank-``k`` subspace can carry.

    Restricted to the orthogonal complement of ``selected_basis`` when one is
    given.  A control that must avoid the selected directions cannot always reach
    their energy — most obviously when the selection *is* the top-energy
    subspace — and the experiment has to state that rather than quietly ship an
    unmatched null.
    """
    matrix = covariance.double()
    if selected_basis is not None:
        selected = selected_basis.double()
        projector = torch.eye(GPT2_HIDDEN_SIZE, dtype=torch.float64) - selected @ selected.T
        # The projector's spectrum is exactly {1 (complement), 0 (selection)}, so an
        # eigendecomposition recovers the complement basis deterministically.  A bare
        # ``qr`` would not, because unpivoted QR gives no guarantee about where the
        # null directions land.
        values, vectors = torch.linalg.eigh(projector)
        basis = vectors[:, values > 0.5]
        if rank > basis.shape[1]:
            raise ValueError("requested rank exceeds the available complement dimension")
        matrix = basis.T @ matrix @ basis
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if rank > eigenvalues.numel():
        raise ValueError("requested rank exceeds the available space dimension")
    ordered = torch.sort(eigenvalues, descending=True).values
    return float(ordered[-rank:].sum()), float(ordered[:rank].sum())


def energy_matched_random_subspace(
    *,
    covariance: torch.Tensor,
    rank: int,
    target_energy: float,
    generator: torch.Generator,
    selected_basis: torch.Tensor | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 60,
) -> tuple[torch.Tensor, dict]:
    """Draw an orthonormal subspace whose calibration energy matches ``target``.

    The subspace is drawn inside the orthogonal complement of ``selected_basis``
    when one is supplied, so a control can never re-use the selected directions,
    and its energy is bisected onto the selected subspace's own energy.  Matching
    energy rather than only rank is what stops a control from being unfairly weak
    or, as in the completed confirmation, unfairly strong.

    The returned diagnostics record whether the target was attainable at all, so
    an impossible match surfaces as a failed gate instead of a silent mismatch.
    """
    if rank <= 0 or rank > GPT2_HIDDEN_SIZE:
        raise ValueError("random subspace rank is out of range")
    if not target_energy > 0:
        raise ValueError("target calibration energy must be positive")
    context = _matching_context(covariance, rank, selected_basis)
    return _draw_matched_subspace(
        context,
        target_energy=target_energy,
        generator=generator,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _draw_matched_subspace(
    context: dict,
    *,
    target_energy: float,
    generator: torch.Generator,
    tolerance: float,
    max_iterations: int,
) -> tuple[torch.Tensor, dict]:
    covariance = context["covariance"]
    low, high = -6.0, 6.0
    best_basis = _shaped_random_basis(context, generator, 0.0)
    best_energy = subspace_energy(covariance, best_basis)
    best_error = abs(best_energy - target_energy)
    state = generator.get_state()
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        generator.set_state(state)
        basis = _shaped_random_basis(context, generator, middle)
        energy = subspace_energy(covariance, basis)
        error = abs(energy - target_energy)
        if error < best_error:
            best_basis, best_energy, best_error = basis, energy, error
        if error <= tolerance * target_energy:
            best_basis, best_energy, best_error = basis, energy, error
            break
        if energy < target_energy:
            low = middle
        else:
            high = middle
    return best_basis, {
        "achieved_energy": float(best_energy),
        "target_energy": float(target_energy),
        "relative_error": float(best_error / target_energy),
        "attainable_minimum": context["attainable_minimum"],
        "attainable_maximum": context["attainable_maximum"],
        "target_attainable": bool(
            context["attainable_minimum"] <= target_energy <= context["attainable_maximum"]
        ),
        "selected_orthogonal": context["selected_orthogonal"],
        "complement_dimension": context["complement_dimension"],
    }


def build_matched_random_subspaces(
    *,
    selected: MarginSubspace,
    covariance: torch.Tensor,
    replicates: int,
    seed: int,
) -> list[MarginSubspace]:
    """Build ``replicates`` energy-matched, selected-orthogonal random controls."""
    if replicates <= 0:
        raise ValueError("at least one random replicate is required")
    target = selected.calibration_achieved_energy
    if target is None:
        target = subspace_energy(covariance, selected.basis)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    # One eigendecomposition and one complement projector for the whole target.
    context = _matching_context(covariance, selected.rank, selected.basis)
    controls: list[MarginSubspace] = []
    for replicate in range(replicates):
        basis, matching = _draw_matched_subspace(
            context,
            target_energy=float(target),
            generator=generator,
            tolerance=1e-3,
            max_iterations=60,
        )
        overlap = float(
            (selected.basis.double().T @ basis.double()).square().sum()
            / max(selected.rank, 1)
        )
        control = MarginSubspace(
            name=f"random_matched_{selected.family}_k{selected.rank:03d}"
            f"_s{selected.state}_r{replicate:03d}",
            family="random_matched",
            state=selected.state,
            basis=basis,
            rank=selected.rank,
            random_replicate=replicate,
            calibration_target_energy=float(target),
            calibration_achieved_energy=float(matching["achieved_energy"]),
            selected_overlap=overlap,
            matched_family=selected.family,
            target_attainable=bool(matching["target_attainable"]),
            selected_orthogonal=bool(matching["selected_orthogonal"]),
        )
        validate_margin_subspace(control)
        controls.append(control)
    return controls


# ---------------------------------------------------------------------------
# Exact analytic evaluation at state 12
# ---------------------------------------------------------------------------


def edited_states(
    hidden: torch.Tensor,
    basis: torch.Tensor,
    *,
    mode: str,
    semantics: str,
    mean: torch.Tensor,
    resample_source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one subspace intervention to cached colon states.

    ``remove`` deletes the component along the subspace; ``retain`` keeps only
    that component and replaces everything else by the calibration centre, which
    is the sufficiency question the ablation arms never asked.
    """
    if mode not in INTERVENTION_MODES:
        raise ValueError(f"unknown intervention mode {mode}")
    if semantics not in ABLATION_SEMANTICS:
        raise ValueError(f"unknown ablation semantics {semantics}")
    if hidden.ndim != 2 or hidden.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("cached colon states must be [N, 768]")
    if semantics == "mean":
        centre = mean.unsqueeze(0).expand_as(hidden)
    elif semantics == "zero":
        centre = torch.zeros_like(hidden)
    else:
        if resample_source is None or resample_source.shape != hidden.shape:
            raise ValueError("resample semantics needs a same-shaped donor tensor")
        centre = resample_source
    centered = hidden - centre
    projected = (centered @ basis) @ basis.T
    if mode == "remove":
        return hidden - projected
    return centre + projected


def deterministic_derangement(count: int, seed: int) -> torch.Tensor:
    """A fixed permutation with no fixed point, for resample ablation."""
    if count < 2:
        raise ValueError("a derangement needs at least two examples")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for _ in range(1000):
        permutation = torch.randperm(count, generator=generator)
        if not bool((permutation == torch.arange(count)).any()):
            return permutation
    # Fall back to a guaranteed fixed-point-free rotation.
    return torch.roll(torch.arange(count), shifts=1)


def answer_token_outcomes(
    logits: torch.Tensor,
    gold_token: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Continuous and binary first-answer-token outcomes.

    ``nll`` and ``margin`` are per example and continuous, which is where the
    statistical power the completed confirmation lacked actually comes from.
    """
    if logits.ndim != 2 or gold_token.ndim != 1:
        raise ValueError("logits must be [N, V] and gold tokens [N]")
    if logits.shape[0] != gold_token.shape[0]:
        raise ValueError("logits and gold tokens must be paired")
    gold = gold_token.to(device=logits.device, dtype=torch.long)
    rows = torch.arange(logits.shape[0], device=logits.device)
    gold_logit = logits[rows, gold]
    masked = logits.clone()
    masked[rows, gold] = torch.finfo(logits.dtype).min
    runner_up = masked.max(dim=-1)
    return {
        "nll": torch.logsumexp(logits.double(), dim=-1) - gold_logit.double(),
        "margin": (gold_logit - runner_up.values).double(),
        "top1_correct": logits.argmax(dim=-1) == gold,
        "runner_up_token": runner_up.indices,
    }


def analytic_logits(
    hidden: torch.Tensor,
    readout: torch.Tensor,
    *,
    basis: torch.Tensor | None = None,
    mode: str = "remove",
    semantics: str = "mean",
    mean: torch.Tensor | None = None,
    resample_source: torch.Tensor | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Exact first-token logits for an edited state-12 vector.

    GPT-2's ``lm_head`` is bias-free and consumes the ``ln_f`` output directly,
    so this reproduces a real hooked forward pass to floating-point tolerance.
    The collection script asserts that equivalence before any sweep runs.
    """
    if hidden.ndim != 2 or hidden.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("cached colon states must be [N, 768]")
    if readout.ndim != 2 or readout.shape[1] != GPT2_HIDDEN_SIZE:
        raise ValueError("readout matrix must be [V, 768]")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if basis is None:
        edited = hidden
    else:
        if mean is None:
            raise ValueError("an intervention needs the calibration mean")
        edited = edited_states(
            hidden,
            basis,
            mode=mode,
            semantics=semantics,
            mean=mean,
            resample_source=resample_source,
        )
    chunks = [
        edited[start : start + chunk_size] @ readout.T
        for start in range(0, edited.shape[0], chunk_size)
    ]
    return torch.cat(chunks, dim=0)


def evaluate_subspace_analytically(
    *,
    hidden: torch.Tensor,
    readout: torch.Tensor,
    gold_token: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor | None,
    mode: str = "remove",
    semantics: str = "mean",
    resample_source: torch.Tensor | None = None,
    chunk_size: int = 128,
) -> dict:
    """Per-example outcomes for one arm, computed without a model forward pass."""
    logits = analytic_logits(
        hidden,
        readout,
        basis=basis,
        mode=mode,
        semantics=semantics,
        mean=mean,
        resample_source=resample_source,
        chunk_size=chunk_size,
    )
    outcomes = answer_token_outcomes(logits, gold_token)
    removed_rms = 0.0
    if basis is not None:
        centre = (
            mean.unsqueeze(0).expand_as(hidden)
            if semantics == "mean"
            else torch.zeros_like(hidden)
            if semantics == "zero"
            else resample_source
        )
        projected = ((hidden - centre) @ basis) @ basis.T
        removed_rms = float(projected.double().square().sum(dim=-1).mean().sqrt())
    return {
        "nll": outcomes["nll"].cpu(),
        "margin": outcomes["margin"].cpu(),
        "top1_correct": outcomes["top1_correct"].cpu(),
        "removed_projection_rms": removed_rms,
        "mode": mode,
        "semantics": semantics,
    }


# ---------------------------------------------------------------------------
# Real forward-pass intervention for propagating and multi-position arms
# ---------------------------------------------------------------------------


class OfficialCODIEndpointSubspaceIntervention:
    """Arbitrary-rank subspace edit at the colon or at every generated token.

    Unlike the completed rank-three ablation hook, this supports retention as
    well as removal, all three ablation semantics, state 11 (which does reach the
    key/value cache) and an all-position mode.  ``applies_to_all_positions`` is
    read by :func:`generate_official_codi`; it defaults to ``False`` so every
    completed experiment keeps its exact behaviour.
    """

    def __init__(
        self,
        model,
        subspaces: Sequence[MarginSubspace],
        *,
        student_mean: torch.Tensor,
        mode: str = "remove",
        semantics: str = "mean",
        all_positions: bool = False,
        alpha: float = 1.0,
    ) -> None:
        if mode not in INTERVENTION_MODES:
            raise ValueError(f"unknown intervention mode {mode}")
        if semantics not in ABLATION_SEMANTICS:
            raise ValueError(f"unknown ablation semantics {semantics}")
        if semantics == "resample":
            raise ValueError(
                "resample semantics is defined only for the cached analytic tier"
            )
        if not subspaces:
            raise ValueError("at least one subspace is required")
        if student_mean.shape != (GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE):
            raise ValueError("student mean must have shape [13, 768]")
        if not torch.isfinite(student_mean).all() or not 0 < alpha <= 2:
            raise ValueError("student mean must be finite and alpha must be in (0, 2]")
        by_state: dict[int, MarginSubspace] = {}
        for subspace in subspaces:
            validate_margin_subspace(subspace)
            if subspace.state in by_state:
                raise ValueError("one subspace per state is allowed")
            by_state[subspace.state] = subspace
        transformer = getattr(model.codi, "base_model", model.codi)
        transformer = getattr(transformer, "model", transformer)
        transformer = getattr(transformer, "transformer", transformer)
        blocks = getattr(transformer, "h", None)
        final_layer_norm = getattr(transformer, "ln_f", None)
        if blocks is None or final_layer_norm is None or len(blocks) != 12:
            raise RuntimeError("unexpected GPT-2 module layout")
        # Index 11 is block 10's output; index 12 is block 11 followed by ln_f.
        self.modules_by_state = {
            PROPAGATING_STATE: blocks[10],
            ANALYTIC_STATE: final_layer_norm,
        }
        self.subspaces = by_state
        self.student_mean = student_mean.detach().cpu().float()
        self.mode = mode
        self.semantics = semantics
        self.alpha = float(alpha)
        self.applies_to_all_positions = bool(all_positions)
        self.active_mask: torch.Tensor | None = None
        self.calls_by_state = {str(state): 0 for state in MARGIN_GEOMETRY_STATES}
        self.rows_by_state = {str(state): 0 for state in MARGIN_GEOMETRY_STATES}
        self.removed_squared_norm_sum_by_state = {
            str(state): 0.0 for state in MARGIN_GEOMETRY_STATES
        }

    def _hook(self, state: int):
        subspace = self.subspaces[state]

        def edit(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3 or hidden.shape[-1] != GPT2_HIDDEN_SIZE:
                raise ValueError("GPT-2 block output shape changed")
            mask = self.active_mask.to(device=hidden.device)
            if mask.shape != (hidden.shape[0],):
                raise ValueError("endpoint intervention mask has the wrong batch shape")
            basis = subspace.basis.to(device=hidden.device)
            centre = (
                self.student_mean[state].to(device=hidden.device).unsqueeze(0)
                if self.semantics == "mean"
                else torch.zeros(
                    1, GPT2_HIDDEN_SIZE, device=hidden.device, dtype=torch.float32
                )
            )
            last = hidden[:, -1, :].float()
            selected = last[mask]
            centered = selected - centre
            projected = (centered @ basis) @ basis.T
            replacement = (
                selected - self.alpha * projected
                if self.mode == "remove"
                else selected + self.alpha * (centre + projected - selected)
            )
            edited = hidden.clone()
            edited_last = edited[:, -1, :].float()
            edited_last[mask] = replacement
            edited[:, -1, :] = edited_last.to(dtype=hidden.dtype)
            key = str(state)
            self.calls_by_state[key] += 1
            self.rows_by_state[key] += int(mask.sum().item())
            self.removed_squared_norm_sum_by_state[key] += float(
                projected.double().square().sum().item()
            )
            if isinstance(output, tuple):
                return (edited, *output[1:])
            if isinstance(output, list):
                return [edited, *output[1:]]
            return edited

        return edit

    @contextmanager
    def activate(self, mask: torch.Tensor):
        if self.active_mask is not None:
            raise RuntimeError("endpoint intervention cannot be nested")
        self.active_mask = mask.detach()
        handles = []
        try:
            for state in sorted(self.subspaces):
                handles.append(
                    self.modules_by_state[state].register_forward_hook(
                        self._hook(state)
                    )
                )
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.active_mask = None

    def diagnostics(self) -> dict:
        rms = {}
        for state in MARGIN_GEOMETRY_STATES:
            key = str(state)
            rows = self.rows_by_state[key]
            rms[key] = (
                float((self.removed_squared_norm_sum_by_state[key] / rows) ** 0.5)
                if rows
                else 0.0
            )
        return {
            "alpha": self.alpha,
            "mode": self.mode,
            "semantics": self.semantics,
            "all_positions": self.applies_to_all_positions,
            "calls_by_state": dict(self.calls_by_state),
            "rows_by_state": dict(self.rows_by_state),
            "removed_projection_rms_by_state": rms,
        }


class OfficialCODIEndpointStateCollector:
    """Capture, without modifying, the colon states the decoder actually uses.

    An earlier version of this experiment cached colon states with the released
    *training* encoder and then compared them against generation.  The two paths
    disagree in at least three ways — the generator normalises the question
    (``strip`` and double-space collapse) while the row formatter does not, the cue
    is tokenised with a leading space in one path and not the other, and left
    padding shifts GPT-2's absolute position ids for every row in a chunk whenever
    the longest sequence changes.  On 64 questions that produced 89% first-token
    agreement.

    Collecting through the generation path removes the entire class of divergence:
    the cached state *is* the state the decoder consumed. The parity gate then
    checks the remaining claim, that ``lm_head`` is a bias-free linear map.

    The interface deliberately matches the intervention classes, so
    :func:`generate_official_codi` drives it with no special-casing.
    """

    #: Never rewrite tokens; this object only observes the answer-cue forward pass.
    applies_to_all_positions = False

    def __init__(self, model, *, states: Sequence[int] = MARGIN_GEOMETRY_STATES) -> None:
        requested = tuple(int(state) for state in states)
        if not requested or not set(requested).issubset(set(MARGIN_GEOMETRY_STATES)):
            raise ValueError("state collection is defined for states 11 and 12")
        transformer = getattr(model.codi, "base_model", model.codi)
        transformer = getattr(transformer, "model", transformer)
        transformer = getattr(transformer, "transformer", transformer)
        blocks = getattr(transformer, "h", None)
        final_layer_norm = getattr(transformer, "ln_f", None)
        if blocks is None or final_layer_norm is None or len(blocks) != 12:
            raise RuntimeError("unexpected GPT-2 module layout")
        self.modules_by_state = {
            PROPAGATING_STATE: blocks[10],
            ANALYTIC_STATE: final_layer_norm,
        }
        self.states = requested
        self.captured: dict[int, list[torch.Tensor]] = {
            state: [] for state in requested
        }
        self.active_mask: torch.Tensor | None = None

    def _hook(self, state: int):
        def capture(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.ndim != 3 or hidden.shape[-1] != GPT2_HIDDEN_SIZE:
                raise ValueError("GPT-2 block output shape changed")
            mask = self.active_mask.to(device=hidden.device)
            if mask.shape != (hidden.shape[0],):
                raise ValueError("state collector mask has the wrong batch shape")
            self.captured[state].append(
                hidden[:, -1, :][mask].detach().float().cpu()
            )
            return output

        return capture

    @contextmanager
    def activate(self, mask: torch.Tensor):
        if self.active_mask is not None:
            raise RuntimeError("state collection cannot be nested")
        self.active_mask = mask.detach()
        handles = []
        try:
            for state in self.states:
                handles.append(
                    self.modules_by_state[state].register_forward_hook(self._hook(state))
                )
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.active_mask = None

    def stacked(self, expected_rows: int) -> torch.Tensor:
        """Return ``[N, 13, 768]`` with only the collected states populated."""
        collected = {}
        for state in self.states:
            if not self.captured[state]:
                raise RuntimeError(f"no colon state was captured for state {state}")
            values = torch.cat(self.captured[state], dim=0)
            if values.shape[0] != expected_rows:
                raise RuntimeError(
                    f"state {state} captured {values.shape[0]} rows, expected "
                    f"{expected_rows}; the answer cue was reached more than once"
                )
            collected[state] = values
        output = torch.zeros(
            expected_rows, GPT2_STATE_COUNT, GPT2_HIDDEN_SIZE, dtype=torch.float32
        )
        for state, values in collected.items():
            output[:, state, :] = values
        if not torch.isfinite(output).all():
            raise RuntimeError("captured colon states contain non-finite values")
        return output


def resolve_output_embedding(model) -> torch.Tensor:
    """Return GPT-2's bias-free ``lm_head`` weight as a ``[V, 768]`` matrix.

    The analytic tier is only valid if the logits really are a linear map of the
    ``ln_f`` output, so this refuses any layout that carries a bias.
    """
    head = None
    for candidate in (
        getattr(model, "codi", None),
        getattr(getattr(model, "codi", None), "base_model", None),
        getattr(getattr(getattr(model, "codi", None), "base_model", None), "model", None),
    ):
        if candidate is None:
            continue
        getter = getattr(candidate, "get_output_embeddings", None)
        resolved = getter() if callable(getter) else None
        if resolved is None:
            resolved = getattr(candidate, "lm_head", None)
        if resolved is not None:
            head = resolved
            break
    if head is None:
        raise RuntimeError("could not resolve the GPT-2 output embedding")
    if getattr(head, "bias", None) is not None:
        raise RuntimeError("analytic state-12 evaluation requires a bias-free lm_head")
    weight = getattr(head, "weight", None)
    if weight is None or weight.ndim != 2 or weight.shape[1] != GPT2_HIDDEN_SIZE:
        raise RuntimeError("unexpected output-embedding shape")
    return weight.detach()


def numeric_answer_token_ids(tokenizer, *, limit: int | None = None) -> list[int]:
    """Token ids whose surface form the model can emit as a numeric answer.

    The released decoder emits answers with a leading space, so both the bare and
    space-prefixed digit forms are eligible readout targets.
    """
    ids: set[int] = set()
    vocabulary = tokenizer.get_vocab()
    for token, index in vocabulary.items():
        text = token.replace("Ġ", " ")
        stripped = text.strip()
        if stripped and all(character.isdigit() for character in stripped):
            ids.add(int(index))
    ordered = sorted(ids)
    if not ordered:
        raise RuntimeError("no numeric answer tokens found in the tokenizer")
    return ordered[:limit] if limit else ordered


def gold_first_token_ids(tokenizer, golds: Iterable[str]) -> list[int]:
    """First emitted token of each gold answer in the model's own surface form."""
    ids = []
    for gold in golds:
        text = str(gold).strip()
        if not text:
            raise ValueError("gold answer must be non-empty")
        encoded = tokenizer(f" {text}", add_special_tokens=False)["input_ids"]
        if not encoded:
            raise ValueError(f"gold answer {text!r} did not tokenize")
        ids.append(int(encoded[0]))
    return ids


def build_margin_arm_registry(
    *,
    covariance: torch.Tensor,
    damage_matrices: Mapping[str, torch.Tensor],
    readout_matrix: torch.Tensor,
    reference_subspaces: Mapping[str, MarginSubspace],
    rank_grid: Sequence[int] = DEFAULT_RANK_GRID,
    random_replicates: int,
    random_seed: int,
    primary_rank: int = 3,
    state: int = ANALYTIC_STATE,
    bands: Sequence[tuple[int, int]] = (),
    band_random_replicates: int = 0,
) -> dict[str, MarginSubspace]:
    """Every state-12 subspace this experiment evaluates analytically.

    Random controls are matched to the ``margin`` family at every rank, and to
    each reference family at the primary rank so the comparison with the failed
    confirmation is exactly like-for-like.
    """
    if random_replicates <= 0:
        raise ValueError("at least one random replicate is required")
    registry: dict[str, MarginSubspace] = {}
    capacity = readout_family_capacity(readout_matrix)
    for family in FITTED_FAMILIES:
        for rank in rank_grid:
            if family == "readout" and rank > capacity:
                # Descriptive family only; higher ranks simply do not exist for it
                # and no gate depends on them.
                continue
            subspace = build_fitted_subspaces(
                family=family,
                rank=rank,
                state=state,
                covariance=covariance,
                damage_matrix=damage_matrices.get(family),
                readout_matrix=readout_matrix,
            )
            registry[subspace.name] = subspace
    for name, reference in reference_subspaces.items():
        if name not in REFERENCE_FAMILIES:
            raise ValueError(f"unexpected reference family {name}")
        validate_margin_subspace(reference)
        registry[reference.name] = reference

    matched_targets = [
        registry[f"margin_k{rank:03d}_s{state}"] for rank in rank_grid
    ] + [registry[reference.name] for reference in reference_subspaces.values()]
    for index, selected in enumerate(matched_targets):
        controls = build_matched_random_subspaces(
            selected=selected,
            covariance=covariance,
            replicates=random_replicates,
            seed=random_seed + 1013 * index,
        )
        for control in controls:
            registry[control.name] = control

    # Bands are appended last and seeded past every existing target, so adding them
    # leaves the names and bases of the already-exported arms bit-identical.
    for offset, (start, stop) in enumerate(bands):
        band = build_band_subspace(
            covariance=covariance, start=start, stop=stop, state=state
        )
        registry[band.name] = band
        if band_random_replicates:
            controls = build_matched_random_subspaces(
                selected=band,
                covariance=covariance,
                replicates=band_random_replicates,
                seed=random_seed + 1013 * (len(matched_targets) + offset),
            )
            for control in controls:
                registry[control.name] = control
    return registry
