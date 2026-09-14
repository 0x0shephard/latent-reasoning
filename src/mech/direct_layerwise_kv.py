"""Independent layer-local answer subspaces and direct latent-KV interventions."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from src.mech.official_codi_layerwise import GPT2_BLOCKS, GPT2_WIDTH, _transformer


class LatentAttentionStateTrace:
    """Capture the six single-token ``ln_1`` outputs at every GPT-2 block."""

    def __init__(self, model, latent_positions: int = 6):
        transformer = _transformer(model)
        self.modules = [block.ln_1 for block in transformer.h]
        self.latent_positions = int(latent_positions)
        self.raw: list[list[torch.Tensor]] = [[] for _ in self.modules]
        self._gradient_connected: torch.Tensor | None = None

    def _hook(self, layer):
        def hook(_module, _inputs, output):
            # Prompt and answer teacher-forcing calls have multiple tokens. CODI's
            # recurrent latent calls contain exactly one input embedding.
            if isinstance(output, torch.Tensor) and output.ndim == 3 and output.shape[1] == 1:
                self.raw[layer].append(output[:, 0, :])
            return output
        return hook

    @contextmanager
    def capture(self):
        handles = [module.register_forward_hook(self._hook(layer))
                   for layer, module in enumerate(self.modules)]
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def validate(self):
        counts = [len(values) for values in self.raw]
        if counts != [self.latent_positions] * GPT2_BLOCKS:
            raise RuntimeError(f"expected six latent calls per layer, observed {counts}")

    def states(self) -> torch.Tensor:
        self.validate()
        # [B,L,P,D]
        return torch.stack([torch.stack(values, dim=1) for values in self.raw], dim=1)

    def gradients(self, loss: torch.Tensor) -> torch.Tensor:
        """Differentiate answer NLL with respect to each captured latent state."""
        self.validate()
        flattened = tuple(value for layer in self.raw for value in layer)
        # A captured tensor can be structurally absent from the answer-loss graph
        # under a particular Transformers cache path. Mathematically its derivative
        # is zero. Preserve that fact explicitly and audit it instead of making one
        # disconnected cell abort the whole 12x6 experiment.
        gradients = torch.autograd.grad(loss, flattened, allow_unused=True)
        rows = []
        connected_rows = []
        offset = 0
        for layer in range(GPT2_BLOCKS):
            layer_values = []
            layer_connected = []
            for position, gradient in enumerate(
                gradients[offset : offset + self.latent_positions]
            ):
                connected = gradient is not None
                layer_connected.append(connected)
                layer_values.append(
                    gradient if connected else torch.zeros_like(self.raw[layer][position])
                )
            rows.append(torch.stack(layer_values, dim=1))
            connected_rows.append(layer_connected)
            offset += self.latent_positions
        self._gradient_connected = torch.tensor(connected_rows, dtype=torch.bool)
        return torch.stack(rows, dim=1)

    def gradient_connectivity(self) -> torch.Tensor:
        if self._gradient_connected is None:
            raise RuntimeError("gradients() must run before requesting connectivity")
        return self._gradient_connected.clone()


@dataclass(frozen=True)
class LayerwiseEigensystem:
    means: torch.Tensor         # [L,P,D]
    eigenvalues: torch.Tensor   # [L,D]
    eigenvectors: torch.Tensor  # [L,D,D], descending


def fit_layerwise_eigensystems(states: torch.Tensor) -> LayerwiseEigensystem:
    if states.ndim != 4 or states.shape[1] != GPT2_BLOCKS or states.shape[-1] != GPT2_WIDTH:
        raise ValueError("states must have shape [N,12,P,768]")
    means = states.double().mean(0)
    values, vectors = [], []
    for layer in range(GPT2_BLOCKS):
        centered = (states[:, layer].double() - means[layer]).reshape(-1, GPT2_WIDTH)
        covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
        order = torch.argsort(eigenvalues, descending=True)
        values.append(eigenvalues[order])
        vectors.append(eigenvectors[:, order])
    return LayerwiseEigensystem(means.float(), torch.stack(values).float(), torch.stack(vectors).float())


def score_layerwise_answer_directions(
    states: torch.Tensor,
    gradients: torch.Tensor,
    eigensystem: LayerwiseEigensystem,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Score every layer-local eigenvector against gold-answer NLL gradients.

    The score is the weaker of deterministic even/odd split z-scores for excess
    absolute first-order effect over example-shuffled gradients.
    """
    if states.shape != gradients.shape or states.ndim != 4:
        raise ValueError("states and gradients must share [N,L,P,D]")
    examples = states.shape[0]
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(examples, generator=generator)
    scores, effects, positive = [], [], []
    for layer in range(GPT2_BLOCKS):
        centered = states[:, layer].double() - eigensystem.means[layer].double()
        basis = eigensystem.eigenvectors[layer].double()
        state_coeff = torch.einsum("npd,dk->npk", centered, basis)
        grad_coeff = torch.einsum("npd,dk->npk", gradients[:, layer].double(), basis)
        shuffled = grad_coeff.index_select(0, permutation)
        per_example = (state_coeff * grad_coeff).abs().mean(1) - (
            state_coeff * shuffled
        ).abs().mean(1)
        split_z = []
        split_positive = []
        for parity in (0, 1):
            values = per_example[parity::2]
            mean = values.mean(0)
            se = values.std(0, unbiased=True) / max(1, values.shape[0]) ** 0.5
            split_z.append(mean / se.clamp_min(1e-12))
            split_positive.append(mean > 0)
        scores.append(torch.minimum(split_z[0], split_z[1]))
        effects.append(per_example.mean(0))
        positive.append(split_positive[0] & split_positive[1])
    return {"split_stable_z": torch.stack(scores).float(),
            "excess_effect": torch.stack(effects).float(),
            "positive_both_splits": torch.stack(positive)}


def select_layerwise_bases(
    eigensystem: LayerwiseEigensystem,
    scores: torch.Tensor,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.shape != (GPT2_BLOCKS, GPT2_WIDTH) or not 0 < rank <= GPT2_WIDTH:
        raise ValueError("invalid score matrix or rank")
    indices = torch.argsort(scores, dim=1, descending=True)[:, :rank]
    bases = torch.stack([
        eigensystem.eigenvectors[layer].index_select(1, indices[layer])
        for layer in range(GPT2_BLOCKS)
    ])
    return bases, indices


def align_layerwise_bases(
    states: torch.Tensor,
    means: torch.Tensor,
    bases: torch.Tensor,
    *,
    reference_layer: int = 11,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal-Procrustes align independently discovered coordinates."""
    if bases.ndim != 3 or bases.shape[:2] != (GPT2_BLOCKS, GPT2_WIDTH):
        raise ValueError("bases must have shape [12,768,R]")
    rank = bases.shape[-1]
    coefficients = []
    for layer in range(GPT2_BLOCKS):
        centered = states[:, layer].double() - means[layer].double()
        coefficients.append((centered @ bases[layer].double()).reshape(-1, rank))
    reference = coefficients[reference_layer]
    rotations, aligned = [], []
    for layer in range(GPT2_BLOCKS):
        u, _, vh = torch.linalg.svd(coefficients[layer].T @ reference, full_matrices=False)
        rotation = u @ vh
        rotations.append(rotation.float())
        aligned.append((bases[layer].double() @ rotation).float())
    return torch.stack(aligned), torch.stack(rotations)


class DirectLatentKVSubspaceIntervention:
    """Retain/remove global cross-head K/V directions as each latent is cached."""

    def __init__(
        self,
        *,
        key_bases: torch.Tensor | Mapping[int, torch.Tensor],
        value_bases: torch.Tensor | Mapping[int, torch.Tensor],
        key_means: torch.Tensor,
        value_means: torch.Tensor,
        layers: Sequence[int],
        positions: Sequence[int],
        mode: str,
    ):
        if mode not in {"retain", "remove"}:
            raise ValueError("mode must be retain or remove")
        if key_means.shape != value_means.shape or key_means.ndim != 3:
            raise ValueError("K/V means must share [L,P,D]")
        feature_width = int(key_means.shape[-1])
        tensor_bases = isinstance(key_bases, torch.Tensor) and isinstance(
            value_bases, torch.Tensor
        )
        mapping_bases = isinstance(key_bases, Mapping) and isinstance(
            value_bases, Mapping
        )
        if not tensor_bases and not mapping_bases:
            raise ValueError("K/V bases must both be tensors or layer mappings")
        if tensor_bases and (
            key_bases.shape != value_bases.shape
            or key_bases.ndim != 3
            or key_bases.shape[1] != feature_width
        ):
            raise ValueError("K/V tensor bases must share [L,D,R]")
        if mapping_bases:
            if set(key_bases) != set(value_bases):
                raise ValueError("K/V basis mappings must contain the same layers")
            for layer in key_bases:
                key_basis, value_basis = key_bases[layer], value_bases[layer]
                if (
                    key_basis.ndim != 2
                    or key_basis.shape[0] != feature_width
                    or key_basis.shape != value_basis.shape
                ):
                    raise ValueError(
                        "each mapped K/V basis must share shape [D, rank]"
                    )
        if tensor_bases:
            self.key_bases = key_bases.float()
            self.value_bases = value_bases.float()
        else:
            self.key_bases = {
                int(layer): basis.float() for layer, basis in key_bases.items()
            }
            self.value_bases = {
                int(layer): basis.float() for layer, basis in value_bases.items()
            }
        self.key_means, self.value_means = key_means.float(), value_means.float()
        self.layers, self.positions, self.mode = set(layers), set(positions), mode

    @staticmethod
    def _legacy(cache):
        if isinstance(cache, tuple): return cache, tuple
        if isinstance(cache, list): return tuple(cache), list
        converter = getattr(cache, "to_legacy_cache", None)
        factory = getattr(type(cache), "from_legacy_cache", None)
        if converter is None or factory is None: raise TypeError("unsupported cache type")
        return converter(), factory

    def _edit(self, tensor, basis, mean):
        batch, heads, _tokens, head_width = tensor.shape
        last = tensor[:, :, -1, :].reshape(batch, heads * head_width).float()
        basis, mean = basis.to(last), mean.to(last)
        centered = last - mean
        component = (centered @ basis) @ basis.T
        replacement = mean + component if self.mode == "retain" else last - component
        result = tensor.clone()
        result[:, :, -1, :] = replacement.reshape(batch, heads, head_width).to(tensor.dtype)
        return result

    @staticmethod
    def _layer_basis(bases, layer: int):
        return bases[layer]

    def __call__(self, cache, position: int):
        if position not in self.positions: return cache
        legacy, restore = self._legacy(cache)
        updated = []
        for layer, entry in enumerate(legacy):
            if layer not in self.layers:
                updated.append(entry); continue
            key = self._edit(
                entry[0], self._layer_basis(self.key_bases, layer),
                self.key_means[layer, position]
            )
            value = self._edit(
                entry[1], self._layer_basis(self.value_bases, layer),
                self.value_means[layer, position]
            )
            updated.append((key, value, *entry[2:]))
        values = tuple(updated)
        if restore is tuple: return values
        if restore is list: return list(values)
        return restore(values)


def longest_contiguous_run(layers: Sequence[int]) -> list[int]:
    runs: list[list[int]] = []
    for layer in sorted(set(int(value) for value in layers)):
        if not runs or layer != runs[-1][-1] + 1: runs.append([layer])
        else: runs[-1].append(layer)
    return max(runs, key=lambda values: (len(values), values[-1]), default=[])
