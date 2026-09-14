"""Causal-subspace-protected cross-layer KV factorization primitives.

This is deliberately a kernel-independent reference implementation.  It can
reconstruct a cache for quality experiments and evaluate attention directly in
the reduced coordinates.  Production speed requires fusing the latter operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class MatrixFactorization:
    code: torch.Tensor       # [tokens, rank]
    decoder: torch.Tensor    # [rank, features]
    protected_rank: int = 0

    def reconstruct(self) -> torch.Tensor:
        return self.code @ self.decoder

    @property
    def storage_elements(self) -> int:
        return self.code.numel() + self.decoder.numel()

    @property
    def dense_elements(self) -> int:
        return self.code.shape[0] * self.decoder.shape[1]

    @property
    def compression_ratio(self) -> float:
        return self.dense_elements / self.storage_elements


def truncated_svd_factorization(matrix: torch.Tensor, rank: int) -> MatrixFactorization:
    if matrix.ndim != 2 or not 0 < rank <= min(matrix.shape):
        raise ValueError("rank must not exceed either matrix dimension")
    u, singular, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return MatrixFactorization(u[:, :rank] * singular[:rank], vh[:rank])


def causal_protected_factorization(
    matrix: torch.Tensor,
    protected_basis: torch.Tensor,
    rank: int,
) -> MatrixFactorization:
    """Preserve the feature-space protected projection, SVD-compress the rest."""
    if matrix.ndim != 2 or protected_basis.ndim != 2:
        raise ValueError("matrix and protected basis must be two dimensional")
    if protected_basis.shape[0] != matrix.shape[1]:
        raise ValueError("protected basis has the wrong feature dimension")
    core_rank = protected_basis.shape[1]
    if not core_rank <= rank <= min(matrix.shape):
        raise ValueError("total rank must cover the protected core and fit the matrix")
    q, _ = torch.linalg.qr(protected_basis.float(), mode="reduced")
    core_code = matrix.float() @ q
    residual = matrix.float() - core_code @ q.T
    residual_rank = rank - core_rank
    if residual_rank:
        residual_factor = truncated_svd_factorization(residual, residual_rank)
        code = torch.cat((core_code, residual_factor.code), dim=1)
        decoder = torch.cat((q.T, residual_factor.decoder), dim=0)
    else:
        code, decoder = core_code, q.T
    return MatrixFactorization(code, decoder, protected_rank=core_rank)


def concatenate_group_cache(
    keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Concatenate per-layer ``[tokens, width]`` K and V into one xKV matrix."""
    if not keys or len(keys) != len(values):
        raise ValueError("keys and values need the same non-zero layer count")
    reference = keys[0].shape
    if len(reference) != 2 or any(x.shape != reference for x in (*keys, *values)):
        raise ValueError("all cache matrices must share [tokens, width]")
    columns = []
    for key, value in zip(keys, values):
        columns.extend((key, value))
    return torch.cat(columns, dim=1)


def split_group_cache(
    matrix: torch.Tensor, *, layers: int, width: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if matrix.ndim != 2 or matrix.shape[1] != 2 * layers * width:
        raise ValueError("group cache has the wrong feature width")
    chunks = list(matrix.split(width, dim=1))
    return chunks[0::2], chunks[1::2]


def mapped_group_causal_basis(
    key_bases: Sequence[torch.Tensor], value_bases: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Stack matched layer/direction K,V responses into a group feature basis."""
    if not key_bases or len(key_bases) != len(value_bases):
        raise ValueError("key and value basis lists must match")
    width, rank = key_bases[0].shape
    if any(x.shape != (width, rank) for x in (*key_bases, *value_bases)):
        raise ValueError("all mapped bases must share [width, rank]")
    columns = []
    for direction in range(rank):
        pieces = []
        for key, value in zip(key_bases, value_bases):
            pieces.extend((key[:, direction], value[:, direction]))
        columns.append(torch.cat(pieces))
    stacked = torch.stack(columns, dim=1).float()
    basis, _ = torch.linalg.qr(stacked, mode="reduced")
    return basis


def mapped_group_variable_causal_basis(
    key_bases: Sequence[torch.Tensor], value_bases: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Embed every layer-local K/V direction independently in group feature space.

    Unlike ``mapped_group_causal_basis``, directions do not need to be aligned or
    share a rank across layers. The protected rank is the sum of layer-local ranks.
    """
    if not key_bases or len(key_bases) != len(value_bases):
        raise ValueError("key and value basis lists must match")
    width = key_bases[0].shape[0]
    if any(
        key.ndim != 2 or value.shape != key.shape or key.shape[0] != width
        for key, value in zip(key_bases, value_bases)
    ):
        raise ValueError("each layer must provide matching [width, rank] K/V bases")
    feature_width = 2 * len(key_bases) * width
    columns = []
    for layer, (key, value) in enumerate(zip(key_bases, value_bases)):
        for direction in range(key.shape[1]):
            column = key.new_zeros(feature_width)
            start = 2 * layer * width
            column[start : start + width] = key[:, direction]
            column[start + width : start + 2 * width] = value[:, direction]
            columns.append(column)
    if not columns:
        return key_bases[0].new_zeros((feature_width, 0))
    stacked = torch.stack(columns, dim=1).float()
    basis, _ = torch.linalg.qr(stacked, mode="reduced")
    return basis


def reduced_attention(
    query: torch.Tensor,
    code: torch.Tensor,
    key_decoder: torch.Tensor,
    value_decoder: torch.Tensor,
) -> torch.Tensor:
    """Attention from factors without materializing dense K or V.

    ``K = code @ key_decoder`` and ``V = code @ value_decoder``.  The algebra
    evaluates ``q K^T`` and ``softmax(.) V`` entirely through the shared code.
    """
    if query.ndim != 2 or code.ndim != 2:
        raise ValueError("query and code must be matrices")
    if key_decoder.shape != value_decoder.shape:
        raise ValueError("key and value decoder shapes must match")
    if code.shape[1] != key_decoder.shape[0] or query.shape[1] != key_decoder.shape[1]:
        raise ValueError("incompatible attention factor shapes")
    scale = query.shape[1] ** -0.5
    logits = (query @ key_decoder.T) @ code.T * scale
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities @ code) @ value_decoder


def relative_reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    if original.shape != reconstructed.shape:
        raise ValueError("matrices must share shape")
    denominator = original.double().square().sum().sqrt()
    if float(denominator) == 0:
        return 0.0
    return float((original.double() - reconstructed.double()).square().sum().sqrt() / denominator)


def _legacy_cache(cache):
    if isinstance(cache, tuple):
        return cache, lambda value: tuple(value)
    if isinstance(cache, list):
        return tuple(cache), lambda value: list(value)
    converter = getattr(cache, "to_legacy_cache", None)
    factory = getattr(type(cache), "from_legacy_cache", None)
    if converter is None or factory is None:
        raise TypeError("cache must be legacy or support legacy conversion")
    return converter(), factory


def compress_reconstruct_cache(
    cache,
    *,
    groups: Sequence[Sequence[int]],
    rank: int,
    protected_bases: dict[tuple[int, ...], torch.Tensor] | None = None,
) -> tuple[object, dict]:
    """Factorize cross-layer K/V groups and return a reconstructed cache.

    This path is for causal quality validation with an unmodified Transformers
    model.  Its memory remains dense after reconstruction.  The report separately
    records the factor storage that a custom reduced-attention kernel would use.
    """
    legacy, restore = _legacy_cache(cache)
    layer_count = len(legacy)
    flattened = [int(layer) for group in groups for layer in group]
    if sorted(flattened) != list(range(layer_count)) or len(set(flattened)) != layer_count:
        raise ValueError("groups must partition every cache layer exactly once")
    updated = [[entry[0].clone(), entry[1].clone(), *entry[2:]] for entry in legacy]
    records = []
    dense_total = factor_total = 0
    for group_value in groups:
        group = tuple(int(layer) for layer in group_value)
        first = legacy[group[0]][0]
        if first.ndim != 4:
            raise ValueError("cache tensors must have shape [B,H,T,D]")
        batch, heads, tokens, head_width = first.shape
        width = heads * head_width
        basis = None if protected_bases is None else protected_bases.get(group)
        for row in range(batch):
            keys = [legacy[layer][0][row].transpose(0, 1).reshape(tokens, width) for layer in group]
            values = [legacy[layer][1][row].transpose(0, 1).reshape(tokens, width) for layer in group]
            matrix = concatenate_group_cache(keys, values)
            requested_core = 0 if basis is None else basis.shape[1]
            effective_rank = min(int(rank), min(matrix.shape))
            if effective_rank < requested_core:
                # A short context cannot carry a rank-28 token factor. Leaving it
                # dense is honest and avoids silently changing the protected rank.
                records.append({"group": list(group), "row": row, "tokens": tokens,
                                "status": "dense_short_context", "requested_rank": rank,
                                "protected_rank": requested_core})
                dense_total += matrix.numel()
                factor_total += matrix.numel()
                continue
            factor = (truncated_svd_factorization(matrix, effective_rank)
                      if basis is None else causal_protected_factorization(matrix, basis.to(matrix), effective_rank))
            reconstructed = factor.reconstruct().to(dtype=first.dtype)
            new_keys, new_values = split_group_cache(reconstructed, layers=len(group), width=width)
            for local, layer in enumerate(group):
                updated[layer][0][row] = new_keys[local].reshape(tokens, heads, head_width).transpose(0, 1)
                updated[layer][1][row] = new_values[local].reshape(tokens, heads, head_width).transpose(0, 1)
            dense_total += factor.dense_elements
            factor_total += factor.storage_elements
            records.append({
                "group": list(group), "row": row, "tokens": tokens, "status": "compressed",
                "effective_rank": effective_rank, "protected_rank": factor.protected_rank,
                "relative_error": relative_reconstruction_error(matrix, reconstructed),
                "dense_elements": factor.dense_elements, "factor_elements": factor.storage_elements,
            })
    restored_entries = []
    for original, values in zip(legacy, updated):
        restored_entries.append(tuple(values) if isinstance(original, tuple) else values)
    return restore(tuple(restored_entries)), {
        "dense_elements": dense_total,
        "factor_elements": factor_total,
        "modelled_compression_ratio": dense_total / max(1, factor_total),
        "records": records,
        "runtime_representation": "dense_reconstruction_quality_proxy",
    }
