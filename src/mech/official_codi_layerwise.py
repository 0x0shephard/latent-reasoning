"""Layer-resolved observation and intervention hooks for official CODI GPT-2.

The endpoint experiments historically exposed only state 11 and the post-``ln_f``
state.  The causal-xKV experiment needs an explicit, audited distinction between:

* the residual stream after every block;
* the normalized activation actually consumed by every QKV projection; and
* the Q, K, and V vectors produced for the answer-cue token.

The hooks in this module are active only inside the context manager driven by
``generate_official_codi(..., answer_endpoint_intervention=...)``.  They therefore
observe or edit the final token of the forced answer cue, not prompt or latent passes.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


GPT2_BLOCKS = 12
GPT2_WIDTH = 768
INTERVENTION_MODES = ("retain", "remove")


def _transformer(model):
    value = getattr(model, "codi", model)
    value = getattr(value, "base_model", value)
    value = getattr(value, "model", value)
    value = getattr(value, "transformer", value)
    blocks = getattr(value, "h", None)
    if blocks is None or len(blocks) != GPT2_BLOCKS:
        raise RuntimeError("unexpected official CODI GPT-2 transformer layout")
    if not hasattr(value, "drop") or not hasattr(value, "ln_f"):
        raise RuntimeError("official CODI GPT-2 is missing drop or ln_f")
    return value


def layerwise_location_names(*, include_qkv: bool = True) -> tuple[str, ...]:
    names = ["resid_embed"]
    for layer in range(GPT2_BLOCKS):
        names.extend((f"attn_ln_{layer:02d}", f"resid_block_{layer:02d}"))
        if include_qkv:
            names.extend(
                (
                    f"query_{layer:02d}",
                    f"key_{layer:02d}",
                    f"value_{layer:02d}",
                )
            )
    names.append("resid_ln_f")
    return tuple(names)


def residual_location_names() -> tuple[str, ...]:
    return (
        "resid_embed",
        *(f"resid_block_{layer:02d}" for layer in range(GPT2_BLOCKS)),
        "resid_ln_f",
    )


def attention_location_names() -> tuple[str, ...]:
    return tuple(f"attn_ln_{layer:02d}" for layer in range(GPT2_BLOCKS))


def _hidden_from_output(output) -> torch.Tensor:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("hooked GPT-2 activation must have shape [B,T,D]")
    if hidden.shape[-1] != GPT2_WIDTH:
        raise ValueError(
            f"expected GPT-2 width {GPT2_WIDTH}, observed {hidden.shape[-1]}"
        )
    return hidden


def _restore_output(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class OfficialCODILayerwiseEndpointCollector:
    """Capture all residual, attention-input, and QKV answer-cue activations."""

    applies_to_all_positions = False

    def __init__(
        self,
        model,
        *,
        locations: Sequence[str] | None = None,
    ) -> None:
        transformer = _transformer(model)
        requested = tuple(locations or layerwise_location_names())
        allowed = set(layerwise_location_names())
        unknown = sorted(set(requested) - allowed)
        if not requested or unknown:
            raise ValueError(f"invalid layerwise locations: {unknown}")
        self.locations = requested
        self.modules: dict[str, torch.nn.Module] = {
            "resid_embed": transformer.drop,
            "resid_ln_f": transformer.ln_f,
        }
        for layer, block in enumerate(transformer.h):
            self.modules[f"resid_block_{layer:02d}"] = block
            self.modules[f"attn_ln_{layer:02d}"] = block.ln_1
            self.modules[f"qkv_{layer:02d}"] = block.attn.c_attn
        self.captured: dict[str, list[torch.Tensor]] = {
            name: [] for name in requested
        }
        self.active_mask: torch.Tensor | None = None

    def _capture_hidden(self, location: str):
        def hook(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            hidden = _hidden_from_output(output)
            mask = self.active_mask.to(device=hidden.device)
            if mask.shape != (hidden.shape[0],):
                raise ValueError("layerwise collector mask has the wrong batch shape")
            self.captured[location].append(
                hidden[:, -1, :][mask].detach().float().cpu()
            )
            return output

        return hook

    def _capture_qkv(self, layer: int):
        def hook(_module, _inputs, output):
            if self.active_mask is None or not bool(self.active_mask.any()):
                return output
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError("GPT-2 c_attn output must have shape [B,T,3D]")
            if output.shape[-1] != 3 * GPT2_WIDTH:
                raise ValueError("GPT-2 c_attn output width changed")
            mask = self.active_mask.to(device=output.device)
            if mask.shape != (output.shape[0],):
                raise ValueError("layerwise collector mask has the wrong batch shape")
            query, key, value = output[:, -1, :].chunk(3, dim=-1)
            for kind, tensor in (("query", query), ("key", key), ("value", value)):
                location = f"{kind}_{layer:02d}"
                if location in self.captured:
                    self.captured[location].append(
                        tensor[mask].detach().float().cpu()
                    )
            return output

        return hook

    @contextmanager
    def activate(self, mask: torch.Tensor):
        if self.active_mask is not None:
            raise RuntimeError("layerwise collection cannot be nested")
        self.active_mask = mask.detach()
        handles = []
        try:
            for location in self.locations:
                if location.startswith(("query_", "key_", "value_")):
                    continue
                handles.append(
                    self.modules[location].register_forward_hook(
                        self._capture_hidden(location)
                    )
                )
            qkv_layers = sorted(
                {
                    int(location.rsplit("_", 1)[1])
                    for location in self.locations
                    if location.startswith(("query_", "key_", "value_"))
                }
            )
            for layer in qkv_layers:
                handles.append(
                    self.modules[f"qkv_{layer:02d}"].register_forward_hook(
                        self._capture_qkv(layer)
                    )
                )
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.active_mask = None

    def stacked(self, expected_rows: int) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for location in self.locations:
            values = self.captured[location]
            if not values:
                raise RuntimeError(f"no endpoint values captured for {location}")
            tensor = torch.cat(values, dim=0)
            if tensor.shape != (expected_rows, GPT2_WIDTH):
                raise RuntimeError(
                    f"{location} produced {tuple(tensor.shape)}, expected "
                    f"{(expected_rows, GPT2_WIDTH)}"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{location} contains non-finite values")
            result[location] = tensor
        return result


@dataclass(frozen=True)
class LayerwiseInterventionDiagnostics:
    location: str
    mode: str
    calls: int
    rows: int
    projected_rms: float


class OfficialCODILayerwiseEndpointIntervention:
    """Retain or remove one centered layer-local subspace at the answer cue."""

    applies_to_all_positions = False

    def __init__(
        self,
        model,
        *,
        location: str,
        basis: torch.Tensor,
        centre: torch.Tensor,
        mode: str,
        alpha: float = 1.0,
        all_positions: bool = False,
    ) -> None:
        if mode not in INTERVENTION_MODES:
            raise ValueError(f"unknown intervention mode {mode!r}")
        if basis.ndim != 2 or basis.shape[0] != GPT2_WIDTH:
            raise ValueError("basis must have shape [768, rank]")
        if centre.shape != (GPT2_WIDTH,):
            raise ValueError("centre must have shape [768]")
        if not torch.isfinite(basis).all() or not torch.isfinite(centre).all():
            raise ValueError("basis and centre must be finite")
        if not 0.0 < alpha <= 2.0:
            raise ValueError("alpha must lie in (0, 2]")
        identity = torch.eye(basis.shape[1], dtype=torch.float64)
        gram = basis.double().T @ basis.double()
        if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
            raise ValueError("basis columns must be orthonormal")

        transformer = _transformer(model)
        modules: dict[str, torch.nn.Module] = {
            "resid_embed": transformer.drop,
            "resid_ln_f": transformer.ln_f,
        }
        for layer, block in enumerate(transformer.h):
            modules[f"resid_block_{layer:02d}"] = block
            modules[f"attn_ln_{layer:02d}"] = block.ln_1
        if location not in modules:
            raise ValueError(f"location {location!r} is not intervenable")
        self.module = modules[location]
        self.location = location
        self.basis = basis.detach().cpu().float()
        self.centre = centre.detach().cpu().float()
        self.mode = mode
        self.alpha = float(alpha)
        self.applies_to_all_positions = bool(all_positions)
        self.active_mask: torch.Tensor | None = None
        self.calls = 0
        self.rows = 0
        self.projected_squared_norm = 0.0

    def _hook(self, _module, _inputs, output):
        if self.active_mask is None or not bool(self.active_mask.any()):
            return output
        hidden = _hidden_from_output(output)
        mask = self.active_mask.to(device=hidden.device)
        if mask.shape != (hidden.shape[0],):
            raise ValueError("layerwise intervention mask has the wrong batch shape")
        basis = self.basis.to(device=hidden.device)
        centre = self.centre.to(device=hidden.device)
        selected = hidden[:, -1, :].float()[mask]
        centered = selected - centre
        projected = (centered @ basis) @ basis.T
        if self.mode == "remove":
            replacement = selected - self.alpha * projected
        else:
            retained = centre + projected
            replacement = selected + self.alpha * (retained - selected)
        edited = hidden.clone()
        last = edited[:, -1, :].float()
        last[mask] = replacement
        edited[:, -1, :] = last.to(dtype=hidden.dtype)
        self.calls += 1
        self.rows += int(mask.sum())
        self.projected_squared_norm += float(projected.double().square().sum())
        return _restore_output(output, edited)

    @contextmanager
    def activate(self, mask: torch.Tensor):
        if self.active_mask is not None:
            raise RuntimeError("layerwise intervention cannot be nested")
        self.active_mask = mask.detach()
        handle = self.module.register_forward_hook(self._hook)
        try:
            yield self
        finally:
            handle.remove()
            self.active_mask = None

    def diagnostics(self) -> LayerwiseInterventionDiagnostics:
        rms = (
            (self.projected_squared_norm / self.rows) ** 0.5 if self.rows else 0.0
        )
        return LayerwiseInterventionDiagnostics(
            location=self.location,
            mode=self.mode,
            calls=self.calls,
            rows=self.rows,
            projected_rms=float(rms),
        )


def gpt2_qkv_response_bases(
    model,
    attention_bases: Mapping[int, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    """Map attention-input directions through GPT-2's fused QKV projection.

    Returned tensors have shape ``[768, rank]`` in concatenated-head Q/K/V feature
    space. The module is evaluated on ``basis`` and zero and differenced, rather
    than reading ``.weight`` directly. This includes the loaded LoRA update while
    cancelling the affine bias exactly.
    """
    transformer = _transformer(model)
    result: dict[int, dict[str, torch.Tensor]] = {}
    for layer, basis in attention_bases.items():
        layer = int(layer)
        if not 0 <= layer < GPT2_BLOCKS:
            raise ValueError("attention basis layer is out of range")
        if basis.ndim != 2 or basis.shape[0] != GPT2_WIDTH:
            raise ValueError("attention basis must have shape [768, rank]")
        module = transformer.h[layer].attn.c_attn
        reference = next(module.parameters())
        probe = basis.T.unsqueeze(0).to(device=reference.device, dtype=reference.dtype)
        with torch.inference_mode():
            response = module(probe) - module(torch.zeros_like(probe))
        if response.shape != (1, basis.shape[1], 3 * GPT2_WIDTH):
            raise RuntimeError(f"unexpected GPT-2 c_attn response shape {response.shape}")
        query, key, value = response[0].float().cpu().chunk(3, dim=-1)
        responses = {"query": query.T.contiguous(), "key": key.T.contiguous(),
                     "value": value.T.contiguous()}
        result[layer] = responses
    return result
