"""Read CODI's latent reasoning trajectory before the answer endpoint.

Every completed endpoint experiment edited or read state 12 at the forced answer
cue, where computation has already collapsed into one token choice and nothing
propagates.  This module supports the cheap *detection gate* that must pass before
any editing experiment at the latent thought states is justified: it captures the
thirteen hidden states of each of the six latent iterations from the released
generation path, and provides the exactly solvable multiclass probe used to ask
whether those states still hold answer information the endpoint has lost.

No model weight is updated and no intervention is applied; the capture object is
a pure observer threaded through the released ``generate_official_codi`` seams.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.mech.endpoint_answer_conditioned import GPT2_HIDDEN_SIZE


LATENT_TRAJECTORY_SCHEMA_VERSION = 1
LATENT_TRAJECTORY_CONTRACT = (
    "frozen_checkpoint_latent_trajectory_detect_gate_v1"
)

#: The thirteen-state layout every endpoint experiment used: state 0 is the
#: embedding stream entering block 0, states 1..11 are the outputs of blocks
#: 0..10, and state 12 is the ``ln_f`` output — identical to the Hugging Face
#: ``hidden_states`` tuple for GPT-2.
TRAJECTORY_STATES = 13


def _transformer_parts(model):
    transformer = getattr(model.codi, "base_model", model.codi)
    transformer = getattr(transformer, "model", transformer)
    transformer = getattr(transformer, "transformer", transformer)
    blocks = getattr(transformer, "h", None)
    final_layer_norm = getattr(transformer, "ln_f", None)
    if blocks is None or final_layer_norm is None or len(blocks) != 12:
        raise RuntimeError("unexpected GPT-2 module layout")
    return blocks, final_layer_norm


class OfficialCODILatentTrajectoryCapture:
    """Capture the 13-state trajectory of each latent iteration.

    Forward hooks buffer the most recent pass; the object is also the
    ``kv_intervention`` callable of ``generate_official_codi``, which the released
    latent loop invokes exactly once per latent position immediately after that
    position's forward pass.  That call commits the buffer, so the prompt pass,
    the forced-cue pass, and any answer-token pass are buffered but never
    committed.  The cache is returned unchanged: this observer edits nothing.
    """

    def __init__(self, model, *, latent_iterations: int) -> None:
        if latent_iterations <= 0:
            raise ValueError("latent_iterations must be positive")
        blocks, final_layer_norm = _transformer_parts(model)
        self.latent_iterations = int(latent_iterations)
        self._buffer: dict[int, torch.Tensor] = {}
        self._committed: list[list[torch.Tensor]] = [
            [] for _ in range(self.latent_iterations)
        ]
        self._handles = [
            blocks[0].register_forward_pre_hook(self._buffer_pre_hook(0))
        ]
        # States 1..11 are the outputs of blocks 0..10; the raw block-11 output
        # never appears in the 13-state layout because ln_f replaces it.
        self._handles.extend(
            blocks[index].register_forward_hook(self._buffer_hook(index + 1))
            for index in range(11)
        )
        self._handles.append(
            final_layer_norm.register_forward_hook(self._buffer_hook(12))
        )

    def _store(self, state_index: int, hidden: torch.Tensor) -> None:
        if hidden.ndim != 3 or hidden.shape[-1] != GPT2_HIDDEN_SIZE:
            raise ValueError("GPT-2 trajectory state shape changed")
        self._buffer[state_index] = hidden[:, -1, :].detach().float().cpu()

    def _buffer_pre_hook(self, state_index: int):
        def buffer(_module, inputs):
            hidden = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            self._store(state_index, hidden)
            return None

        return buffer

    def _buffer_hook(self, state_index: int):
        def buffer(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self._store(state_index, hidden)
            return output

        return buffer

    def __call__(self, cache, latent_position: int):
        if not 0 <= int(latent_position) < self.latent_iterations:
            raise ValueError("latent position outside the configured loop")
        if sorted(self._buffer) != list(range(TRAJECTORY_STATES)):
            raise RuntimeError(
                "a latent pass did not touch all thirteen trajectory states"
            )
        self._committed[int(latent_position)].append(
            torch.stack(
                [self._buffer[state] for state in range(TRAJECTORY_STATES)], dim=1
            )
        )
        self._buffer = {}
        return cache

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def stacked(self, expected_rows: int) -> torch.Tensor:
        """Return ``[rows, latent_positions, 13, 768]`` in question order."""
        per_position = []
        for position, chunks in enumerate(self._committed):
            if not chunks:
                raise RuntimeError(f"latent position {position} was never captured")
            values = torch.cat(chunks, dim=0)
            if values.shape != (expected_rows, TRAJECTORY_STATES, GPT2_HIDDEN_SIZE):
                raise RuntimeError(
                    f"latent position {position} captured {tuple(values.shape)}"
                )
            per_position.append(values)
        return torch.stack(per_position, dim=1)


@dataclass(frozen=True)
class MulticlassRidge:
    """Exact one-hot ridge classifier over the gold first-answer-token classes.

    The solve is closed-form linear algebra, so unlike the logistic probes it
    needs no convergence certificate: there is no optimizer to under-converge.
    """

    classes: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    ridge: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        standard = (features.double() - self.mean) / self.scale
        scores = standard @ self.weight + self.bias
        return self.classes[scores.argmax(dim=1)]

    def state_dict(self) -> dict:
        return {
            "classes": self.classes,
            "mean": self.mean,
            "scale": self.scale,
            "weight": self.weight,
            "bias": self.bias,
            "ridge": float(self.ridge),
        }


def fit_one_hot_ridge(
    features: torch.Tensor, labels: torch.Tensor, *, ridge: float
) -> MulticlassRidge:
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must be paired")
    if features.shape[0] < 2:
        raise ValueError("at least two fitting rows are required")
    if ridge <= 0.0:
        raise ValueError("one-hot ridge requires positive regularization")
    values = features.double()
    mean, scale = values.mean(0), values.std(0).clamp_min(1e-8)
    standard = (values - mean) / scale
    classes = torch.unique(labels.long())
    one_hot = (labels.long().unsqueeze(1) == classes.unsqueeze(0)).double()
    class_mean = one_hot.mean(0)
    centred = one_hot - class_mean
    gram = standard.T @ standard
    gram += float(ridge) * values.shape[0] * torch.eye(
        gram.shape[0], dtype=torch.float64
    )
    weight = torch.linalg.solve(gram, standard.T @ centred)
    return MulticlassRidge(
        classes=classes,
        mean=mean,
        scale=scale,
        weight=weight,
        bias=class_mean,
        ridge=float(ridge),
    )


def multiclass_ridge_from_state_dict(payload: dict) -> MulticlassRidge:
    return MulticlassRidge(
        classes=payload["classes"],
        mean=payload["mean"],
        scale=payload["scale"],
        weight=payload["weight"],
        bias=payload["bias"],
        ridge=float(payload["ridge"]),
    )
