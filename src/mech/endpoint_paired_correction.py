"""Same-question counterfactual correction at CODI's answer cue.

Sampling the answer token cannot create different answer-cue states: state 12 is
computed *before* that token is sampled.  This module therefore perturbs state 11
during the forced-cue forward pass, captures the resulting state 12, and learns an
additive correction from paired correct/wrong outcomes of the same question.

The learned edit is restricted to the already-confirmed answer-bearing PCA band
(PCs 4--31) and preserves every component outside that band.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch

from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE
from src.mech.endpoint_margin_geometry import ANALYTIC_STATE, PROPAGATING_STATE


PAIRED_CORRECTION_SCHEMA_VERSION = 1
PAIRED_CORRECTION_CONTRACT = (
    "frozen_checkpoint_same_question_state11_counterfactual_correction_v1"
)


def _transformer_parts(model):
    transformer = getattr(model.codi, "base_model", model.codi)
    transformer = getattr(transformer, "model", transformer)
    transformer = getattr(transformer, "transformer", transformer)
    blocks = getattr(transformer, "h", None)
    final_layer_norm = getattr(transformer, "ln_f", None)
    if blocks is None or final_layer_norm is None or len(blocks) != 12:
        raise RuntimeError("unexpected GPT-2 module layout")
    return blocks, final_layer_norm


class OfficialCODIPerturbAndCapture:
    """Add seeded relative-RMS noise at state 11 and capture state 12.

    This object is active only on the forced answer-cue pass.  A scale of 0 is an
    observational baseline; positive scales change block 10's output, propagate
    through block 11 and ``ln_f``, and therefore produce a genuine counterfactual
    state 12 before greedy answer-token selection.
    """

    applies_to_all_positions = False

    def __init__(self, model, *, relative_noise: float, seed: int) -> None:
        if not 0.0 <= relative_noise <= 2.0:
            raise ValueError("relative noise must be in [0, 2]")
        blocks, final_layer_norm = _transformer_parts(model)
        self.state11_module = blocks[10]
        self.state12_module = final_layer_norm
        self.relative_noise = float(relative_noise)
        self.seed = int(seed)
        self.active_mask: torch.Tensor | None = None
        self.generator: torch.Generator | None = None
        self.captured: list[torch.Tensor] = []
        self.rows = 0
        self.noise_squared_norm = 0.0

    def _state11_hook(self):
        def perturb(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            mask = self.active_mask.to(device=hidden.device)
            if hidden.ndim != 3 or hidden.shape[-1] != GPT2_HIDDEN_SIZE:
                raise ValueError("GPT-2 state-11 shape changed")
            if mask.shape != (hidden.shape[0],):
                raise ValueError("perturbation mask has the wrong shape")
            selected = hidden[:, -1, :].float()[mask]
            if self.generator is None:
                self.generator = torch.Generator(device=hidden.device).manual_seed(
                    self.seed
                )
            noise = torch.randn(
                selected.shape,
                dtype=torch.float32,
                device=hidden.device,
                generator=self.generator,
            )
            state_norm = selected.double().norm(dim=1, keepdim=True).clamp_min(1e-12)
            noise_norm = noise.double().norm(dim=1, keepdim=True).clamp_min(1e-12)
            noise = noise * (
                self.relative_noise * state_norm / noise_norm
            ).to(dtype=noise.dtype)
            edited = hidden.clone()
            last = edited[:, -1, :].float()
            last[mask] = selected + noise
            edited[:, -1, :] = last.to(dtype=hidden.dtype)
            self.rows += int(mask.sum())
            self.noise_squared_norm += float(noise.double().square().sum())
            if isinstance(output, tuple):
                return (edited, *output[1:])
            if isinstance(output, list):
                return [edited, *output[1:]]
            return edited

        return perturb

    def _state12_hook(self):
        def capture(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            mask = self.active_mask.to(device=hidden.device)
            if hidden.ndim != 3 or hidden.shape[-1] != GPT2_HIDDEN_SIZE:
                raise ValueError("GPT-2 state-12 shape changed")
            self.captured.append(
                hidden[:, -1, :][mask].detach().float().cpu()
            )
            return output

        return capture

    @contextmanager
    def activate(self, mask: torch.Tensor):
        if self.active_mask is not None:
            raise RuntimeError("perturb-and-capture cannot be nested")
        self.active_mask = mask.detach()
        handles = [
            self.state11_module.register_forward_hook(self._state11_hook()),
            self.state12_module.register_forward_hook(self._state12_hook()),
        ]
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.active_mask = None

    def stacked(self, expected_rows: int) -> torch.Tensor:
        if not self.captured:
            raise RuntimeError("no state-12 vectors were captured")
        values = torch.cat(self.captured, dim=0)
        if values.shape != (expected_rows, GPT2_HIDDEN_SIZE):
            raise RuntimeError(
                f"captured state shape {tuple(values.shape)}, expected "
                f"({expected_rows}, {GPT2_HIDDEN_SIZE})"
            )
        return values

    def diagnostics(self) -> dict:
        return {
            "state_perturbed": PROPAGATING_STATE,
            "state_captured": ANALYTIC_STATE,
            "relative_noise": self.relative_noise,
            "seed": self.seed,
            "rows": self.rows,
            "noise_rms_norm": (
                (self.noise_squared_norm / self.rows) ** 0.5 if self.rows else 0.0
            ),
        }


def top_two_margin(states: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
    dtype = torch.float32 if states.is_cuda else torch.float64
    logits = states.to(dtype=dtype) @ readout.to(
        device=states.device, dtype=dtype
    ).T
    top = logits.topk(2, dim=1).values
    return top[:, 0] - top[:, 1]


def paired_question_examples(
    states: torch.Tensor,
    correct: torch.Tensor,
    basis: torch.Tensor,
    centre: torch.Tensor,
    readout: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One equally weighted wrong-to-correct training pair per eligible question.

    ``states`` is ``[questions, variants, 768]``. Questions without both outcomes
    are excluded rather than pretending that a sampled answer supplies a distinct
    pre-answer state.
    """
    if states.ndim != 3 or states.shape[2] != GPT2_HIDDEN_SIZE:
        raise ValueError("paired states must be [questions, variants, 768]")
    if correct.dtype != torch.bool or correct.shape != states.shape[:2]:
        raise ValueError("correctness must be [questions, variants] boolean")
    if basis.ndim != 2 or basis.shape[0] != GPT2_HIDDEN_SIZE:
        raise ValueError("basis must be [768, rank]")
    wrong_states, targets, question_rows = [], [], []
    for question in range(states.shape[0]):
        labels = correct[question]
        if not bool(labels.any()) or not bool((~labels).any()):
            continue
        right = states[question, labels].double().mean(0)
        wrong = states[question, ~labels].double().mean(0)
        wrong_states.append(wrong)
        targets.append((right - wrong) @ basis.double())
        question_rows.append(question)
    if not wrong_states:
        raise RuntimeError("no question produced both a correct and wrong state")
    wrong_matrix = torch.stack(wrong_states)
    features = correction_features(
        wrong_matrix,
        basis=basis,
        centre=centre,
        readout=readout,
    )
    return {
        "features": features,
        "targets": torch.stack(targets),
        "question_rows": torch.tensor(question_rows, dtype=torch.long),
    }


@dataclass(frozen=True)
class RidgeCorrection:
    weight: torch.Tensor
    bias: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    ridge: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        device = features.device
        dtype = torch.float32 if features.is_cuda else torch.float64
        standardized = (
            features.to(dtype=dtype)
            - self.feature_mean.to(device=device, dtype=dtype)
        ) / self.feature_scale.to(device=device, dtype=dtype)
        return standardized @ self.weight.to(
            device=device, dtype=dtype
        ) + self.bias.to(device=device, dtype=dtype)

    def state_dict(self) -> dict:
        return {
            "weight": self.weight.cpu().float(),
            "bias": self.bias.cpu().float(),
            "feature_mean": self.feature_mean.cpu().float(),
            "feature_scale": self.feature_scale.cpu().float(),
            "ridge": self.ridge,
        }


def fit_ridge_correction(
    features: torch.Tensor, targets: torch.Tensor, *, ridge: float
) -> RidgeCorrection:
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must be matrices")
    if features.shape[0] != targets.shape[0] or features.shape[0] < 2:
        raise ValueError("features and targets need paired rows")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    x, y = features.double(), targets.double()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-8)
    z = (x - mean) / scale
    design = torch.cat(
        [z, torch.ones(z.shape[0], 1, dtype=torch.float64, device=z.device)], dim=1
    )
    penalty = torch.eye(design.shape[1], dtype=torch.float64, device=z.device)
    penalty[-1, -1] = 0
    solution = torch.linalg.solve(
        design.T @ design + float(ridge) * penalty,
        design.T @ y,
    )
    return RidgeCorrection(
        weight=solution[:-1],
        bias=solution[-1],
        feature_mean=mean,
        feature_scale=scale,
        ridge=float(ridge),
    )


def correction_features(
    states: torch.Tensor,
    *,
    basis: torch.Tensor,
    centre: torch.Tensor,
    readout: torch.Tensor,
) -> torch.Tensor:
    dtype = torch.float32 if states.is_cuda else torch.float64
    coefficients = (
        states.to(dtype=dtype)
        - centre.to(device=states.device, dtype=dtype)
    ) @ basis.to(device=states.device, dtype=dtype)
    margin = top_two_margin(states, readout).unsqueeze(1)
    return torch.cat([coefficients, margin], dim=1)


def corrected_states(
    states: torch.Tensor,
    *,
    basis: torch.Tensor,
    predicted_coefficients: torch.Tensor,
    alpha: float,
    margin: torch.Tensor,
    maximum_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= alpha <= 2:
        raise ValueError("alpha must be in [0, 2]")
    dtype = torch.float32 if states.is_cuda else torch.float64
    gate = margin.to(dtype=dtype) <= float(maximum_margin)
    delta = predicted_coefficients.to(dtype=dtype) @ basis.to(
        device=states.device, dtype=dtype
    ).T
    edited = states.to(dtype=dtype).clone()
    edited[gate] += float(alpha) * delta[gate]
    return edited, gate


class OfficialCODIConditionedCorrectionIntervention:
    """Apply a fitted, margin-gated additive correction to state 12."""

    applies_to_all_positions = False

    def __init__(
        self,
        model,
        *,
        basis: torch.Tensor,
        centre: torch.Tensor,
        readout: torch.Tensor,
        correction: RidgeCorrection,
        alpha: float,
        maximum_margin: float,
    ) -> None:
        _, final_layer_norm = _transformer_parts(model)
        if basis.shape[0] != GPT2_HIDDEN_SIZE or centre.shape != (GPT2_HIDDEN_SIZE,):
            raise ValueError("conditioned correction geometry has the wrong shape")
        self.module = final_layer_norm
        self.basis = basis.detach().cpu().float()
        self.centre = centre.detach().cpu().float()
        # The generation runner passes the model's already-resident lm_head view;
        # preserving its device avoids copying a ~150 MB GPT-2 readout every batch.
        self.readout = readout.detach().float()
        self.correction = correction
        self.alpha = float(alpha)
        self.maximum_margin = float(maximum_margin)
        self.active_mask: torch.Tensor | None = None
        self.rows_seen = 0
        self.rows_edited = 0
        self.delta_squared_norm = 0.0
        self._cached_device: torch.device | None = None
        self._device_basis: torch.Tensor | None = None
        self._device_centre: torch.Tensor | None = None
        self._device_readout: torch.Tensor | None = None

    def _hook(self):
        def edit(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            mask = self.active_mask.to(device=hidden.device)
            selected = hidden[:, -1, :].float()[mask]
            if self._cached_device != hidden.device:
                self._cached_device = hidden.device
                self._device_basis = self.basis.to(device=hidden.device)
                self._device_centre = self.centre.to(device=hidden.device)
                self._device_readout = self.readout.to(device=hidden.device)
            basis = self._device_basis
            centre = self._device_centre
            readout = self._device_readout
            features = correction_features(
                selected, basis=basis, centre=centre, readout=readout
            )
            predicted = self.correction.predict(features).to(device=hidden.device)
            margin = features[:, -1]
            replacement, gate = corrected_states(
                selected,
                basis=basis,
                predicted_coefficients=predicted,
                alpha=self.alpha,
                margin=margin,
                maximum_margin=self.maximum_margin,
            )
            edited = hidden.clone()
            last = edited[:, -1, :].float()
            last[mask] = replacement.to(dtype=last.dtype)
            edited[:, -1, :] = last.to(dtype=hidden.dtype)
            delta = replacement - selected.double()
            self.rows_seen += int(selected.shape[0])
            self.rows_edited += int(gate.sum())
            self.delta_squared_norm += float(delta.square().sum())
            if isinstance(output, tuple):
                return (edited, *output[1:])
            if isinstance(output, list):
                return [edited, *output[1:]]
            return edited

        return edit

    @contextmanager
    def activate(self, mask: torch.Tensor):
        if self.active_mask is not None:
            raise RuntimeError("conditioned correction cannot be nested")
        self.active_mask = mask.detach()
        handle = self.module.register_forward_hook(self._hook())
        try:
            yield self
        finally:
            handle.remove()
            self.active_mask = None

    def diagnostics(self) -> dict:
        return {
            "state": ANALYTIC_STATE,
            "alpha": self.alpha,
            "maximum_margin": self.maximum_margin,
            "rows_seen": self.rows_seen,
            "rows_edited": self.rows_edited,
            "edited_fraction": (
                self.rows_edited / self.rows_seen if self.rows_seen else 0.0
            ),
            "edit_rms_norm": (
                (self.delta_squared_norm / self.rows_seen) ** 0.5
                if self.rows_seen
                else 0.0
            ),
        }


def ridge_from_state_dict(payload: dict) -> RidgeCorrection:
    return RidgeCorrection(
        weight=payload["weight"],
        bias=payload["bias"],
        feature_mean=payload["feature_mean"],
        feature_scale=payload["feature_scale"],
        ridge=float(payload["ridge"]),
    )
