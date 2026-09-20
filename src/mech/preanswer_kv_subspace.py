"""Direct pre-answer KV gradients and variable-rank layer-local subspaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from src.mech.direct_layerwise_kv import LayerwiseEigensystem
from src.mech.official_codi_layerwise import GPT2_BLOCKS, GPT2_WIDTH
from src.mech.official_codi_target_utility import build_official_student_answer_io


@dataclass(frozen=True)
class PreAnswerKVResult:
    per_example_loss: torch.Tensor
    mean_loss: torch.Tensor
    first_logits: torch.Tensor
    first_targets: torch.Tensor
    key_states: torch.Tensor
    value_states: torch.Tensor
    key_gradients: torch.Tensor | None
    value_gradients: torch.Tensor | None
    gradient_connected: torch.Tensor | None
    full_key_gradients: torch.Tensor | None = None
    full_value_gradients: torch.Tensor | None = None


def cache_as_legacy_tuple(cache) -> tuple:
    """Expose exact cache tensors without stacking or detaching them."""
    if isinstance(cache, tuple):
        return cache
    if isinstance(cache, list):
        return tuple(cache)
    layers = getattr(cache, "layers", None)
    if layers is not None:
        result = tuple((layer.keys, layer.values) for layer in layers)
        if result and all(item[0] is not None and item[1] is not None for item in result):
            return result
    converter = getattr(cache, "to_legacy_cache", None)
    if converter is None:
        raise TypeError("cache does not expose a legacy key/value representation")
    return tuple(converter())


def latent_cache_tensor(cache, *, latent_positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,L,P,768] views of the final latent cache entries."""
    legacy = cache_as_legacy_tuple(cache)
    keys, values = [], []
    for entry in legacy:
        key, value = entry[:2]
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("cache layers must contain matching [B,H,T,Dh] K/V")
        if key.shape[2] < latent_positions:
            raise ValueError("cache is shorter than the requested latent window")
        batch, heads, _, head_width = key.shape
        keys.append(
            key[:, :, -latent_positions:, :].permute(0, 2, 1, 3).reshape(
                batch, latent_positions, heads * head_width
            )
        )
        values.append(
            value[:, :, -latent_positions:, :].permute(0, 2, 1, 3).reshape(
                batch, latent_positions, heads * head_width
            )
        )
    return torch.stack(keys, dim=1), torch.stack(values, dim=1)


def full_cache_tensor(cache) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matching ``[B,L,T,D]`` views of every cache row."""
    legacy = cache_as_legacy_tuple(cache)
    keys, values = [], []
    for entry in legacy:
        key, value = entry[:2]
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("cache layers must contain matching [B,H,T,Dh] K/V")
        batch, heads, tokens, head_width = key.shape
        keys.append(key.permute(0, 2, 1, 3).reshape(batch, tokens, heads * head_width))
        values.append(value.permute(0, 2, 1, 3).reshape(batch, tokens, heads * head_width))
    return torch.stack(keys, dim=1), torch.stack(values, dim=1)


def _cache_gradients(
    loss: torch.Tensor,
    legacy_cache: tuple,
    *,
    latent_positions: int,
    batch_scale: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = tuple(tensor for entry in legacy_cache for tensor in entry[:2])
    raw = torch.autograd.grad(loss, targets, allow_unused=True)
    key_rows, value_rows, connected = [], [], []
    for layer in range(len(legacy_cache)):
        key_target, value_target = legacy_cache[layer][:2]
        key_gradient, value_gradient = raw[2 * layer : 2 * layer + 2]
        layer_connected = (key_gradient is not None, value_gradient is not None)
        connected.append(layer_connected)
        if key_gradient is None:
            key_gradient = torch.zeros_like(key_target)
        if value_gradient is None:
            value_gradient = torch.zeros_like(value_target)
        key_rows.append(key_gradient * int(batch_scale))
        value_rows.append(value_gradient * int(batch_scale))
    gradient_cache = tuple(zip(key_rows, value_rows))
    full_key, full_value = full_cache_tensor(gradient_cache)
    key = full_key[:, :, -latent_positions:, :]
    value = full_value[:, :, -latent_positions:, :]
    return key, value, torch.tensor(connected, dtype=torch.bool), full_key, full_value


def official_codi_preanswer_kv_forward(
    model,
    batch,
    *,
    latent_positions: int,
    return_gradients: bool = False,
    gradient_objective: str = "answer_nll",
    kv_intervention=None,
    return_full_cache_gradients: bool = False,
) -> PreAnswerKVResult:
    """Score the answer using the exact pre-answer cache and optionally differentiate it.

    The gradient targets are the same tensor objects supplied as ``past_key_values``
    to the answer pass. This is the connectivity property missing from the earlier
    hooked-hidden-state experiment.
    """
    encoded = model.codi(
        input_ids=batch.student_question_ids,
        attention_mask=batch.student_question_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    cache = encoded.past_key_values
    latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
    for position in range(latent_positions):
        latent_output = model.codi(
            inputs_embeds=latent,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = latent_output.past_key_values
        if kv_intervention is not None:
            cache = kv_intervention(cache, position)
        latent = model.prj(latent_output.hidden_states[-1][:, -1, :].unsqueeze(1))

    # Normalize once and then use these exact tensors as both the decoder inputs and
    # the autograd targets. For gradient runs, leaf copies preserve the exact cache
    # values while making the answer loss differentiable even if all model parameters
    # are frozen. They also avoid retaining the prompt/latent construction graph.
    legacy_cache = cache_as_legacy_tuple(cache)
    if return_gradients:
        legacy_cache = tuple(
            (
                entry[0].detach().requires_grad_(True),
                entry[1].detach().requires_grad_(True),
                *entry[2:],
            )
            for entry in legacy_cache
        )
    key_states, value_states = latent_cache_tensor(
        legacy_cache, latent_positions=latent_positions
    )
    answer_inputs, answer_targets, answer_mask = build_official_student_answer_io(
        batch, eot_token_id=model.eot_id, pad_token_id=model.pad_token_id
    )
    decoded = model.codi(
        inputs_embeds=model.input_embeddings()(answer_inputs),
        past_key_values=legacy_cache,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    token_loss = F.cross_entropy(
        decoded.logits.transpose(1, 2), answer_targets, reduction="none"
    )
    weights = answer_mask.to(token_loss.dtype)
    per_example = (token_loss * weights).sum(-1) / weights.sum(-1).clamp_min(1)
    first_positions = answer_mask.to(torch.int64).argmax(-1)
    rows = torch.arange(answer_inputs.shape[0], device=answer_inputs.device)
    first_logits = decoded.logits[rows, first_positions]
    first_targets = answer_targets[rows, first_positions]

    if gradient_objective not in {"answer_nll", "first_token_margin"}:
        raise ValueError(f"unknown cache-gradient objective {gradient_objective!r}")
    key_gradients = value_gradients = connected = None
    full_key_gradients = full_value_gradients = None
    if return_gradients:
        gradient_loss = per_example.mean()
        if gradient_objective == "first_token_margin":
            # Preserve the dense model's own top-1 decision without using a gold
            # answer label.  The top-two identities are treated as fixed while
            # autograd measures cache directions that support their logit margin.
            top_two = first_logits.detach().topk(2, dim=-1).indices
            top = first_logits.gather(1, top_two[:, :1]).squeeze(1)
            runner_up = first_logits.gather(1, top_two[:, 1:2]).squeeze(1)
            gradient_loss = -(top - runner_up).mean()
        (
            key_gradients,
            value_gradients,
            connected,
            full_key_gradients,
            full_value_gradients,
        ) = _cache_gradients(
            gradient_loss, legacy_cache,
            latent_positions=latent_positions,
            batch_scale=per_example.shape[0],
        )
        if not return_full_cache_gradients:
            full_key_gradients = full_value_gradients = None
    return PreAnswerKVResult(
        per_example_loss=per_example,
        mean_loss=per_example.mean(),
        first_logits=first_logits,
        first_targets=first_targets,
        key_states=key_states,
        value_states=value_states,
        key_gradients=key_gradients,
        value_gradients=value_gradients,
        gradient_connected=connected,
        full_key_gradients=full_key_gradients,
        full_value_gradients=full_value_gradients,
    )


def _bh_qvalues(p_values: torch.Tensor) -> torch.Tensor:
    flat = p_values.flatten().double()
    order = torch.argsort(flat)
    ordered = flat[order]
    scale = flat.numel() / torch.arange(
        1, flat.numel() + 1, dtype=torch.double, device=flat.device
    )
    adjusted = ordered * scale
    adjusted = torch.flip(torch.cummin(torch.flip(adjusted, dims=(0,)), dim=0).values, dims=(0,))
    q_values = torch.empty_like(adjusted)
    q_values[order] = adjusted.clamp(max=1)
    return q_values.reshape(p_values.shape).float()


def score_preanswer_kv_directions(
    states: torch.Tensor,
    key_gradients: torch.Tensor,
    value_gradients: torch.Tensor,
    eigensystem: LayerwiseEigensystem,
    key_responses: torch.Tensor,
    value_responses: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Rank hidden-state eigenvectors by their direct first-order KV effect."""
    if states.shape != key_gradients.shape or states.shape != value_gradients.shape:
        raise ValueError("states and K/V gradients must share [N,L,P,D]")
    if states.ndim != 4 or states.shape[1:] != (
        GPT2_BLOCKS, states.shape[2], GPT2_WIDTH
    ):
        raise ValueError("expected states with shape [N,12,P,768]")
    expected = (GPT2_BLOCKS, GPT2_WIDTH, GPT2_WIDTH)
    if key_responses.shape != expected or value_responses.shape != expected:
        raise ValueError("full K/V response bases must have shape [12,768,768]")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(states.shape[0], generator=generator)
    split_z, effects, positive, split_means = [], [], [], []
    for layer in range(GPT2_BLOCKS):
        centered = states[:, layer].double() - eigensystem.means[layer].double()
        coefficients = torch.einsum(
            "npd,dk->npk", centered, eigensystem.eigenvectors[layer].double()
        )
        key_sensitivity = torch.einsum(
            "npd,dk->npk", key_gradients[:, layer].double(),
            key_responses[layer].double()
        )
        value_sensitivity = torch.einsum(
            "npd,dk->npk", value_gradients[:, layer].double(),
            value_responses[layer].double()
        )
        predicted_removal_damage = -coefficients * (
            key_sensitivity + value_sensitivity
        )
        shuffled_damage = -coefficients * (
            key_sensitivity.index_select(0, permutation)
            + value_sensitivity.index_select(0, permutation)
        )
        per_example = predicted_removal_damage.mean(1) - shuffled_damage.mean(1)
        layer_z, layer_positive, layer_means = [], [], []
        for parity in (0, 1):
            values = per_example[parity::2]
            mean = values.mean(0)
            standard_error = values.std(0, unbiased=True) / values.shape[0] ** 0.5
            layer_means.append(mean)
            layer_z.append(mean / standard_error.clamp_min(1e-12))
            layer_positive.append(mean > 0)
        split_z.append(torch.minimum(layer_z[0], layer_z[1]))
        positive.append(layer_positive[0] & layer_positive[1])
        split_means.append(torch.stack(layer_means))
        effects.append(per_example.mean(0))
    z_scores = torch.stack(split_z)
    # Conservative conjunction p-value: a direction must be strong in its weaker half.
    p_values = 0.5 * torch.erfc(z_scores / 2 ** 0.5)
    q_values = _bh_qvalues(p_values)
    return {
        "split_stable_z": z_scores.float(),
        "excess_predicted_removal_damage": torch.stack(effects).float(),
        "positive_both_splits": torch.stack(positive),
        "split_means": torch.stack(split_means).float(),
        "p_values": p_values.float(),
        "q_values": q_values,
    }


def select_variable_layerwise_bases(
    eigensystem: LayerwiseEigensystem,
    scores: Mapping[str, torch.Tensor],
    *,
    maximum_rank: int,
    minimum_split_z: float,
    fdr_q: float,
    retained_effect_fraction: float = 0.95,
    validation_scores: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], list[int]]:
    """Return statistically validated PCs and a parsimonious rank per layer."""
    bases, indices, ranks = {}, {}, []
    for layer in range(GPT2_BLOCKS):
        eligible = (
            scores["positive_both_splits"][layer]
            & (scores["split_stable_z"][layer] >= minimum_split_z)
            & (scores["q_values"][layer] <= fdr_q)
        )
        candidates = torch.where(eligible)[0]
        if candidates.numel():
            order = torch.argsort(
                scores["excess_predicted_removal_damage"][layer, candidates],
                descending=True,
            )
            candidates = candidates[order][:maximum_rank]
            rank_scores = validation_scores or scores
            if validation_scores is not None:
                rank_stable = (
                    rank_scores["positive_both_splits"][layer, candidates]
                    & (rank_scores["excess_predicted_removal_damage"][layer, candidates] > 0)
                )
                candidates = candidates[rank_stable]
            weights = rank_scores[
                "excess_predicted_removal_damage"
            ][layer, candidates].clamp_min(0)
            if float(weights.sum()) > 0:
                cumulative = weights.cumsum(0) / weights.sum()
                operational_rank = int(torch.where(
                    cumulative >= retained_effect_fraction
                )[0][0]) + 1
                candidates = candidates[:operational_rank]
        indices[layer] = candidates
        bases[layer] = eigensystem.eigenvectors[layer].index_select(1, candidates)
        ranks.append(int(candidates.numel()))
    return bases, indices, ranks
