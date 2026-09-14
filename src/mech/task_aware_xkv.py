"""Task-aware residual xKV primitives for the CODI confirmation experiment.

The implementation is deliberately a quality/storage reference path.  It
reconstructs dense caches so that an unmodified Transformers model can consume
them.  A production speed claim requires a kernel that reads the factors and
the sparse protected residual directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from src.mech.causal_xkv import (
    _legacy_cache,
    concatenate_group_cache,
    relative_reconstruction_error,
    split_group_cache,
)


@dataclass(frozen=True)
class WeightedFactorization:
    """A low-token-rank matrix factorization with explicit storage accounting."""

    code: torch.Tensor
    decoder: torch.Tensor
    storage_bits: int
    quantization_bits: int | None = None

    def reconstruct(self) -> torch.Tensor:
        return self.code @ self.decoder


def _validate_bits(bits: int | None) -> None:
    if bits is not None and not 2 <= int(bits) <= 16:
        raise ValueError("quantization bits must be in [2,16]")


def symmetric_fake_quantize(
    tensor: torch.Tensor,
    *,
    bits: int,
    reduce_dim: int,
    scale_bits: int = 16,
) -> tuple[torch.Tensor, int]:
    """Fake-quantize with one symmetric scale along ``reduce_dim``.

    The returned tensor stays floating point for the dense quality proxy.  The
    second return value is the modeled packed storage, including FP16 scales.
    """
    _validate_bits(bits)
    if tensor.ndim == 0:
        raise ValueError("cannot quantize a scalar")
    reduce_dim %= tensor.ndim
    bound = 2 ** (int(bits) - 1) - 1
    maximum = tensor.float().abs().amax(dim=reduce_dim, keepdim=True)
    scale = (maximum / max(bound, 1)).clamp_min(torch.finfo(torch.float32).eps)
    quantized = torch.clamp(torch.round(tensor.float() / scale), -bound, bound)
    reconstructed = quantized * scale
    modeled_bits = tensor.numel() * int(bits) + scale.numel() * int(scale_bits)
    return reconstructed.to(tensor.dtype), int(modeled_bits)


def weighted_svd_factorization(
    matrix: torch.Tensor,
    rank: int,
    *,
    feature_weights: torch.Tensor | None = None,
    quantization_bits: int | None = None,
    storage_element_bits: int = 16,
) -> WeightedFactorization:
    """Fit SVD under a diagonal feature-error metric.

    With positive weights ``s``, this minimizes ``||(X-X_hat) diag(s)||_F``.
    Gradient-RMS weights therefore provide a deployable, calibration-fitted
    answer-Fisher proxy without using any test answer or future test query.
    """
    if matrix.ndim != 2 or not 0 < int(rank) <= min(matrix.shape):
        raise ValueError("rank must fit the matrix")
    _validate_bits(quantization_bits)
    if feature_weights is None:
        weights = torch.ones(matrix.shape[1], device=matrix.device, dtype=torch.float32)
    else:
        if feature_weights.shape != (matrix.shape[1],):
            raise ValueError("feature weights must have one value per matrix column")
        weights = feature_weights.to(matrix.device, torch.float32)
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("feature weights must be finite and positive")
    transformed = matrix.float() * weights
    left, singular, right = torch.linalg.svd(transformed, full_matrices=False)
    code = left[:, :rank] * singular[:rank]
    decoder = right[:rank] / weights
    if quantization_bits is None:
        storage_bits = (code.numel() + decoder.numel()) * int(storage_element_bits)
    else:
        # Per-component scales: one scale per code column and decoder row.
        code, code_bits = symmetric_fake_quantize(
            code, bits=quantization_bits, reduce_dim=0
        )
        decoder, decoder_bits = symmetric_fake_quantize(
            decoder, bits=quantization_bits, reduce_dim=1
        )
        storage_bits = code_bits + decoder_bits
    return WeightedFactorization(
        code=code,
        decoder=decoder,
        storage_bits=int(storage_bits),
        quantization_bits=quantization_bits,
    )


def allocate_rank_budget(
    utilities: Sequence[float],
    *,
    total_rank: int,
    caps: Sequence[int],
    minimum_rank: int = 1,
) -> list[int]:
    """Allocate an integer rank budget to groups using diminishing returns."""
    if len(utilities) != len(caps) or not utilities:
        raise ValueError("utilities and caps must have the same non-zero length")
    if any(cap < 0 for cap in caps) or total_rank < 0:
        raise ValueError("rank budgets and caps must be non-negative")
    target = min(int(total_rank), sum(int(cap) for cap in caps))
    ranks = [min(int(minimum_rank), int(cap)) for cap in caps]
    while sum(ranks) > target:
        removable = [index for index, value in enumerate(ranks) if value]
        index = min(removable, key=lambda item: float(utilities[item]))
        ranks[index] -= 1
    safe_utilities = [max(float(value), 1e-12) for value in utilities]
    while sum(ranks) < target:
        eligible = [index for index, cap in enumerate(caps) if ranks[index] < int(cap)]
        if not eligible:
            break
        # Concave utility prevents all rank from collapsing into one group.
        index = max(
            eligible,
            key=lambda item: safe_utilities[item] / (ranks[item] + 1) ** 0.5,
        )
        ranks[index] += 1
    return ranks


def gradient_rms_group_metrics(
    key_gradients: torch.Tensor,
    value_gradients: torch.Tensor,
    groups: Sequence[Sequence[int]],
    *,
    minimum_weight: float = 0.25,
    maximum_weight: float = 4.0,
) -> tuple[dict[tuple[int, ...], torch.Tensor], dict[tuple[int, ...], float]]:
    """Fit fixed group feature weights from exact answer-loss K/V gradients."""
    if key_gradients.shape != value_gradients.shape or key_gradients.ndim != 4:
        raise ValueError("K/V gradients must share [N,L,P,D]")
    if not 0 < minimum_weight <= maximum_weight:
        raise ValueError("invalid weight clipping interval")
    feature_weights: dict[tuple[int, ...], torch.Tensor] = {}
    utilities: dict[tuple[int, ...], float] = {}
    for group_value in groups:
        group = tuple(int(layer) for layer in group_value)
        blocks = []
        for layer in group:
            for gradients in (key_gradients, value_gradients):
                rms = gradients[:, layer].double().square().mean((0, 1)).sqrt().float()
                blocks.append(rms)
        raw = torch.cat(blocks).clamp_min(torch.finfo(torch.float32).eps)
        normalized = raw / raw.mean().clamp_min(torch.finfo(torch.float32).eps)
        feature_weights[group] = normalized.clamp(minimum_weight, maximum_weight)
        utilities[group] = float(raw.double().square().mean())
    return feature_weights, utilities


def _basis_for(
    protected_bases: Mapping[tuple[int, str], torch.Tensor] | None,
    layer: int,
    kind: str,
) -> torch.Tensor | None:
    if protected_bases is None:
        return None
    basis = protected_bases.get((int(layer), kind))
    if basis is None or basis.shape[1] == 0:
        return None
    if basis.ndim != 2:
        raise ValueError("protected bases must be matrices")
    orthogonal, _ = torch.linalg.qr(basis.float(), mode="reduced")
    return orthogonal


def _apply_sparse_protected_residual(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    group: tuple[int, ...],
    width: int,
    latent_positions: int,
    protected_bases: Mapping[tuple[int, str], torch.Tensor] | None,
    residual_bits: int = 16,
) -> tuple[torch.Tensor, int, int]:
    """Correct protected feature coordinates only on CODI's latent cache rows."""
    if not 0 < latent_positions <= original.shape[0]:
        raise ValueError("latent positions must fit the cache token dimension")
    result = reconstructed.clone()
    stored_bits = protected_coordinates = 0
    rows = slice(original.shape[0] - latent_positions, original.shape[0])
    for local, layer in enumerate(group):
        for kind_offset, kind in enumerate(("key", "value")):
            basis = _basis_for(protected_bases, layer, kind)
            if basis is None:
                continue
            if basis.shape[0] != width:
                raise ValueError("protected basis width does not match cache width")
            start = (2 * local + kind_offset) * width
            stop = start + width
            error = original[rows, start:stop].float() - result[rows, start:stop].float()
            coordinates = error @ basis.to(error)
            result[rows, start:stop] += coordinates @ basis.to(error).T
            protected_coordinates += coordinates.numel()
            stored_bits += coordinates.numel() * int(residual_bits)
    return result, int(stored_bits), int(protected_coordinates)


def compress_reconstruct_task_aware_cache(
    cache,
    *,
    groups: Sequence[Sequence[int]],
    rank: int,
    feature_weights: Mapping[tuple[int, ...], torch.Tensor] | None = None,
    group_utilities: Mapping[tuple[int, ...], float] | None = None,
    adaptive_ranks: bool = False,
    protected_bases: Mapping[tuple[int, str], torch.Tensor] | None = None,
    latent_positions: int = 6,
    factor_quantization_bits: int | None = None,
    storage_element_bits: int = 16,
) -> tuple[object, dict]:
    """Compress grouped cache matrices and reconstruct them for quality testing.

    The ordinary and protected arms use the same xKV rank.  Protected arms add
    sparse per-request coordinates only for the last ``latent_positions`` rows;
    the fixed bases are model metadata, not duplicated in each request's cache.
    """
    legacy, restore = _legacy_cache(cache)
    layer_count = len(legacy)
    normalized_groups = tuple(tuple(int(layer) for layer in group) for group in groups)
    flattened = [layer for group in normalized_groups for layer in group]
    if sorted(flattened) != list(range(layer_count)) or len(set(flattened)) != layer_count:
        raise ValueError("groups must partition every cache layer exactly once")
    _validate_bits(factor_quantization_bits)
    updated = [[entry[0].clone(), entry[1].clone(), *entry[2:]] for entry in legacy]
    records = []
    dense_bits_total = factor_bits_total = protected_bits_total = 0
    batch = legacy[0][0].shape[0]
    for row in range(batch):
        matrices = []
        metadata = []
        for group in normalized_groups:
            first = legacy[group[0]][0]
            if first.ndim != 4:
                raise ValueError("cache tensors must have [B,H,T,Dh]")
            _, heads, tokens, head_width = first.shape
            width = heads * head_width
            keys = [
                legacy[layer][0][row].transpose(0, 1).reshape(tokens, width)
                for layer in group
            ]
            values = [
                legacy[layer][1][row].transpose(0, 1).reshape(tokens, width)
                for layer in group
            ]
            matrices.append(concatenate_group_cache(keys, values))
            metadata.append((tokens, heads, head_width, width))
        caps = [min(matrix.shape) for matrix in matrices]
        uniform = [min(int(rank), cap) for cap in caps]
        if adaptive_ranks:
            if group_utilities is None:
                raise ValueError("adaptive ranks require group utilities")
            utilities = [float(group_utilities[group]) for group in normalized_groups]
            ranks = allocate_rank_budget(
                utilities, total_rank=sum(uniform), caps=caps, minimum_rank=1
            )
        else:
            ranks = uniform
        for group, matrix, effective_rank, meta in zip(
            normalized_groups, matrices, ranks, metadata
        ):
            tokens, heads, head_width, width = meta
            weights = None if feature_weights is None else feature_weights.get(group)
            factor = weighted_svd_factorization(
                matrix,
                effective_rank,
                feature_weights=None if weights is None else weights.to(matrix),
                quantization_bits=factor_quantization_bits,
                storage_element_bits=storage_element_bits,
            )
            reconstructed = factor.reconstruct()
            reconstructed, protected_bits, protected_coordinates = (
                _apply_sparse_protected_residual(
                    matrix,
                    reconstructed,
                    group=group,
                    width=width,
                    latent_positions=latent_positions,
                    protected_bases=protected_bases,
                    residual_bits=storage_element_bits,
                )
                if protected_bases
                else (reconstructed, 0, 0)
            )
            new_keys, new_values = split_group_cache(
                reconstructed.to(legacy[group[0]][0].dtype),
                layers=len(group),
                width=width,
            )
            for local, layer in enumerate(group):
                updated[layer][0][row] = new_keys[local].reshape(
                    tokens, heads, head_width
                ).transpose(0, 1)
                updated[layer][1][row] = new_values[local].reshape(
                    tokens, heads, head_width
                ).transpose(0, 1)
            dense_bits = matrix.numel() * int(storage_element_bits)
            dense_bits_total += dense_bits
            factor_bits_total += factor.storage_bits
            protected_bits_total += protected_bits
            records.append({
                "group": list(group),
                "row": row,
                "tokens": tokens,
                "effective_rank": effective_rank,
                "weighted": weights is not None,
                "factor_quantization_bits": factor_quantization_bits,
                "protected_coordinates": protected_coordinates,
                "relative_reconstruction_error": relative_reconstruction_error(
                    matrix, reconstructed
                ),
                "dense_bits": dense_bits,
                "factor_bits": factor.storage_bits,
                "protected_bits": protected_bits,
            })
    restored_entries = []
    for original, values in zip(legacy, updated):
        restored_entries.append(tuple(values) if isinstance(original, tuple) else values)
    cache_bits = factor_bits_total + protected_bits_total
    return restore(tuple(restored_entries)), {
        "dense_bits": int(dense_bits_total),
        "factor_bits": int(factor_bits_total),
        "protected_residual_bits": int(protected_bits_total),
        "cache_bits": int(cache_bits),
        "modelled_compression_ratio": dense_bits_total / max(1, cache_bits),
        "records": records,
        "runtime_representation": "dense_reconstruction_quality_proxy",
    }


def quantize_reconstruct_cache(
    cache,
    *,
    bits: int,
    dense_storage_bits: int = 16,
    scale_bits: int = 16,
) -> tuple[object, dict]:
    """Per-token, per-head symmetric dense-cache quantization quality proxy."""
    _validate_bits(bits)
    legacy, restore = _legacy_cache(cache)
    updated = []
    dense_bits_total = quantized_bits_total = 0
    for entry in legacy:
        values = []
        for tensor in entry[:2]:
            reconstructed, modeled_bits = symmetric_fake_quantize(
                tensor, bits=bits, reduce_dim=-1, scale_bits=scale_bits
            )
            values.append(reconstructed)
            dense_bits_total += tensor.numel() * int(dense_storage_bits)
            quantized_bits_total += modeled_bits
        values.extend(entry[2:])
        updated.append(tuple(values) if isinstance(entry, tuple) else values)
    return restore(tuple(updated)), {
        "dense_bits": int(dense_bits_total),
        "cache_bits": int(quantized_bits_total),
        "modelled_compression_ratio": dense_bits_total / max(1, quantized_bits_total),
        "quantization_bits": int(bits),
        "runtime_representation": "fake_quantized_dense_quality_proxy",
        "quantization_scheme": "symmetric_per_token_per_head_with_fp16_scale",
    }
