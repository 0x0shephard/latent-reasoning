"""Storage-matched, component-wise adaptive rank allocation for CODI xKV.

This module is a quality/storage reference implementation.  It reconstructs
dense caches for an unmodified Transformers model; a production speed claim
still requires a reduced-attention kernel that consumes the factors directly.
"""
from __future__ import annotations

from collections.abc import Mapping
import heapq

import torch

from src.mech.causal_xkv import _legacy_cache, relative_reconstruction_error
from src.mech.preanswer_kv_subspace import cache_as_legacy_tuple
from src.mech.task_aware_xkv import weighted_svd_factorization


Component = tuple[int, str]


def _component_order(layer_count: int) -> tuple[Component, ...]:
    return tuple(
        (layer, kind)
        for layer in range(int(layer_count))
        for kind in ("key", "value")
    )


def _flatten_component(tensor: torch.Tensor, row: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("cache tensors must have shape [B,H,T,Dh]")
    return tensor[row].transpose(0, 1).reshape(tensor.shape[2], -1)


class CalibrationSpectrumCollector:
    """Collect per-SVD-mode reconstruction and answer-Fisher utilities.

    The callback observes the exact dense pre-answer cache.  After the answer
    backward pass, :meth:`consume` pairs each retained SVD mode with the exact
    answer-NLL gradient on every pre-answer cache row.  No cache or example-level tensor
    is retained after a batch has been consumed.
    """

    def __init__(self, *, latent_positions: int, maximum_rank: int):
        if latent_positions <= 0 or maximum_rank <= 0:
            raise ValueError("latent_positions and maximum_rank must be positive")
        self.latent_positions = int(latent_positions)
        self.maximum_rank = int(maximum_rank)
        self.pending: dict[tuple[int, int, str], tuple[torch.Tensor, ...]] = {}
        self.reconstruction_sums: dict[Component, torch.Tensor] = {}
        self.fisher_sums: dict[Component, torch.Tensor] = {}
        self.counts: dict[Component, int] = {}
        self.token_counts: list[int] = []

    def __call__(self, cache, latent_position):
        if int(latent_position) != self.latent_positions - 1:
            return cache
        if self.pending:
            raise RuntimeError("consume the previous calibration batch first")
        legacy = cache_as_legacy_tuple(cache)
        batch = legacy[0][0].shape[0]
        self.token_counts.extend([int(legacy[0][0].shape[2])] * batch)
        matrices = []
        for layer, entry in enumerate(legacy):
            for kind_index, kind in enumerate(("key", "value")):
                tensor = entry[kind_index]
                matrices.append(tensor.permute(0, 2, 1, 3).reshape(
                    tensor.shape[0], tensor.shape[2], -1
                ))
        matrix = torch.stack(matrices, dim=1).detach().float()
        component_count, tokens, width = matrix.shape[1:]
        flat = matrix.reshape(batch * component_count, tokens, width)
        q = min(self.maximum_rank, tokens, width)
        left, singular, right = torch.linalg.svd(flat, full_matrices=False)
        components = _component_order(len(legacy))
        for row in range(batch):
            for component_index, (layer, kind) in enumerate(components):
                index = row * component_count + component_index
                self.pending[(row, layer, kind)] = (
                    left[index, :, :q], singular[index, :q], right[index, :q]
                )
        return cache

    def consume(
        self,
        key_gradients: torch.Tensor,
        value_gradients: torch.Tensor,
    ) -> None:
        if key_gradients.shape != value_gradients.shape or key_gradients.ndim != 4:
            raise ValueError("K/V gradients must share [B,L,P,D]")
        expected = key_gradients.shape[0] * key_gradients.shape[1] * 2
        if len(self.pending) != expected:
            raise RuntimeError("calibration spectra do not match gradient batch")
        for row in range(key_gradients.shape[0]):
            for layer in range(key_gradients.shape[1]):
                for kind, gradients in (
                    ("key", key_gradients), ("value", value_gradients)
                ):
                    left, singular, right = self.pending[(row, layer, kind)]
                    gradient = gradients[row, layer].float().to(right.device)
                    if gradient.shape[0] != left.shape[0]:
                        raise ValueError("gradient token rows do not match calibration spectra")
                    projected = gradient @ right.T
                    first_order = singular * (left * projected).sum(0)
                    reconstruction = singular.double().square().cpu()
                    fisher = first_order.double().square().cpu()
                    component = (layer, kind)
                    if component not in self.reconstruction_sums:
                        self.reconstruction_sums[component] = torch.zeros(
                            self.maximum_rank, dtype=torch.float64
                        )
                        self.fisher_sums[component] = torch.zeros(
                            self.maximum_rank, dtype=torch.float64
                        )
                        self.counts[component] = 0
                    self.reconstruction_sums[component][: len(reconstruction)] += reconstruction
                    self.fisher_sums[component][: len(fisher)] += fisher
                    self.counts[component] += 1
        self.pending.clear()

    def curves(self) -> tuple[dict[Component, torch.Tensor], dict[Component, torch.Tensor]]:
        if self.pending:
            raise RuntimeError("the final calibration batch has not been consumed")
        if not self.counts:
            raise RuntimeError("no calibration batches were collected")
        reconstruction = {
            component: self.reconstruction_sums[component] / self.counts[component]
            for component in self.counts
        }
        fisher = {
            component: self.fisher_sums[component] / self.counts[component]
            for component in self.counts
        }
        return reconstruction, fisher


def blend_component_utilities(
    reconstruction: Mapping[Component, torch.Tensor],
    answer_fisher: Mapping[Component, torch.Tensor],
    *,
    answer_weight: float,
) -> dict[Component, torch.Tensor]:
    """Blend globally normalized marginal utility curves and enforce concavity."""
    weight = float(answer_weight)
    if not 0 <= weight <= 1:
        raise ValueError("answer_weight must lie in [0,1]")
    if set(reconstruction) != set(answer_fisher) or not reconstruction:
        raise ValueError("utility families must use identical non-empty components")
    recon_total = sum(float(value.double().clamp_min(0).sum()) for value in reconstruction.values())
    fisher_total = sum(float(value.double().clamp_min(0).sum()) for value in answer_fisher.values())
    if recon_total <= 0 or fisher_total <= 0:
        raise ValueError("utility families must contain positive mass")
    result = {}
    for component in sorted(reconstruction):
        left = reconstruction[component].double().clamp_min(0) / recon_total
        right = answer_fisher[component].double().clamp_min(0) / fisher_total
        if left.shape != right.shape or left.ndim != 1:
            raise ValueError("component utilities must be matching vectors")
        marginal = (1 - weight) * left + weight * right
        # Prefix ranks require diminishing marginal utility for greedy allocation.
        marginal = torch.cummin(marginal, dim=0).values
        result[component] = marginal.clamp_min(0).cpu()
    return result


def allocate_component_ranks(
    utilities: Mapping[Component, torch.Tensor],
    *,
    total_rank: int,
    caps: Mapping[Component, int],
    minimum_rank: int = 1,
) -> dict[Component, int]:
    """Allocate a fixed number of rank units across K/V components.

    Curves must contain non-increasing marginal utilities.  Under this
    diminishing-returns contract, the heap allocation is the exact optimum.
    """
    components = tuple(sorted(utilities))
    if not components or set(components) != set(caps):
        raise ValueError("utilities and caps must use identical non-empty components")
    ranks = {}
    for component in components:
        curve = utilities[component]
        cap = int(caps[component])
        if curve.ndim != 1 or cap < 0 or cap > len(curve):
            raise ValueError("each cap must fit its one-dimensional utility curve")
        if bool((curve[1:] > curve[:-1] + 1e-15).any()):
            raise ValueError("marginal utilities must be non-increasing")
        ranks[component] = min(int(minimum_rank), cap)
    target = min(max(int(total_rank), 0), sum(int(caps[key]) for key in components))
    if sum(ranks.values()) > target:
        allocated = sum(ranks.values())
        for component in reversed(components):
            while ranks[component] and allocated > target:
                ranks[component] -= 1
                allocated -= 1
    else:
        allocated = sum(ranks.values())
    heap = []
    for index, component in enumerate(components):
        rank = ranks[component]
        if rank < int(caps[component]):
            heapq.heappush(heap, (-float(utilities[component][rank]), index, component))
    while allocated < target and heap:
        _, index, component = heapq.heappop(heap)
        ranks[component] += 1
        allocated += 1
        rank = ranks[component]
        if rank < int(caps[component]):
            heapq.heappush(heap, (-float(utilities[component][rank]), index, component))
    return ranks


def ordinary_per_layer_budget_bits(
    *, layer_count: int, tokens: int, width: int, rank: int,
    storage_element_bits: int = 16,
) -> int:
    effective = min(int(rank), int(tokens), 2 * int(width))
    return int(layer_count) * effective * (int(tokens) + 2 * int(width)) * int(storage_element_bits)


def compress_reconstruct_adaptive_component_cache(
    cache,
    *,
    baseline_rank: int,
    utilities: Mapping[Component, torch.Tensor],
    maximum_component_rank: int,
    storage_element_bits: int = 16,
) -> tuple[object, dict]:
    """Independently factorize layer K/V with the ordinary-xKV bit budget."""
    legacy, restore = _legacy_cache(cache)
    layer_count = len(legacy)
    expected = set(_component_order(layer_count))
    if set(utilities) != expected:
        raise ValueError("utilities must cover every layer key and value component")
    updated = [[entry[0].clone(), entry[1].clone(), *entry[2:]] for entry in legacy]
    records = []
    dense_bits_total = factor_bits_total = padding_bits_total = budget_bits_total = 0
    batch = legacy[0][0].shape[0]
    for row in range(batch):
        reference = legacy[0][0]
        _, heads, tokens, head_width = reference.shape
        width = heads * head_width
        caps = {
            component: min(int(maximum_component_rank), tokens, width, len(utilities[component]))
            for component in expected
        }
        budget_bits = ordinary_per_layer_budget_bits(
            layer_count=layer_count, tokens=tokens, width=width,
            rank=baseline_rank, storage_element_bits=storage_element_bits,
        )
        unit_bits = (tokens + width) * int(storage_element_bits)
        ranks = allocate_component_ranks(
            utilities, total_rank=budget_bits // unit_bits, caps=caps, minimum_rank=1
        )
        row_factor_bits = 0
        for layer in range(layer_count):
            for kind_index, kind in enumerate(("key", "value")):
                matrix = _flatten_component(legacy[layer][kind_index], row)
                rank = ranks[(layer, kind)]
                factor = weighted_svd_factorization(
                    matrix, rank, storage_element_bits=storage_element_bits
                )
                reconstructed = factor.reconstruct().to(matrix.dtype)
                updated[layer][kind_index][row] = reconstructed.reshape(
                    tokens, heads, head_width
                ).transpose(0, 1)
                dense_bits = matrix.numel() * int(storage_element_bits)
                dense_bits_total += dense_bits
                row_factor_bits += factor.storage_bits
                records.append({
                    "layer": layer, "cache_tensor": kind, "row": row,
                    "tokens": tokens, "effective_rank": rank,
                    "relative_reconstruction_error": relative_reconstruction_error(
                        matrix, reconstructed
                    ),
                    "dense_bits": dense_bits, "factor_bits": factor.storage_bits,
                })
        if row_factor_bits > budget_bits:
            raise RuntimeError("adaptive factors exceeded the ordinary-xKV bit budget")
        factor_bits_total += row_factor_bits
        padding_bits_total += budget_bits - row_factor_bits
        budget_bits_total += budget_bits
    restored_entries = [
        tuple(values) if isinstance(original, tuple) else values
        for original, values in zip(legacy, updated)
    ]
    return restore(tuple(restored_entries)), {
        "dense_bits": int(dense_bits_total),
        "factor_bits": int(factor_bits_total),
        "budget_padding_bits": int(padding_bits_total),
        "cache_bits": int(budget_bits_total),
        "modelled_compression_ratio": dense_bits_total / max(1, budget_bits_total),
        "baseline_rank": int(baseline_rank),
        "records": records,
        "runtime_representation": "dense_reconstruction_quality_proxy",
    }


class AdaptiveComponentFactorizer:
    """Apply the frozen component allocator once after CODI latent step six."""

    def __init__(
        self, *, latent_positions: int, baseline_rank: int,
        utilities: Mapping[Component, torch.Tensor], maximum_component_rank: int,
    ):
        self.latent_positions = int(latent_positions)
        self.baseline_rank = int(baseline_rank)
        self.utilities = dict(utilities)
        self.maximum_component_rank = int(maximum_component_rank)
        self.reports: list[dict] = []

    def __call__(self, cache, latent_position):
        if int(latent_position) != self.latent_positions - 1:
            return cache
        transformed, report = compress_reconstruct_adaptive_component_cache(
            cache,
            baseline_rank=self.baseline_rank,
            utilities=self.utilities,
            maximum_component_rank=self.maximum_component_rank,
        )
        self.reports.append(report)
        return transformed

    def summary(self) -> dict:
        records = [record for report in self.reports for record in report["records"]]
        ranks: dict[str, list[int]] = {}
        for record in records:
            key = f'{record["layer"]}:{record["cache_tensor"]}'
            ranks.setdefault(key, []).append(int(record["effective_rank"]))
        dense_bits = sum(report["dense_bits"] for report in self.reports)
        cache_bits = sum(report["cache_bits"] for report in self.reports)
        return {
            "batches": len(self.reports),
            "dense_bits": int(dense_bits),
            "factor_bits": int(sum(report["factor_bits"] for report in self.reports)),
            "budget_padding_bits": int(sum(report["budget_padding_bits"] for report in self.reports)),
            "cache_bits": int(cache_bits),
            "modelled_compression_ratio": dense_bits / max(1, cache_bits),
            "effective_rank_by_component": {
                key: {"minimum": min(values), "mean": sum(values) / len(values), "maximum": max(values)}
                for key, values in ranks.items()
            },
            "mean_relative_reconstruction_error": (
                sum(record["relative_reconstruction_error"] for record in records) / len(records)
                if records else None
            ),
            "runtime_representation": (
                self.reports[0]["runtime_representation"] if self.reports else None
            ),
        }


def select_adaptive_candidate(records: list[dict]) -> dict | None:
    eligible = [record for record in records if bool(record.get("screen_passed"))]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            float(record["modelled_compression_ratio"]),
            float(record["nll_bootstrap_95ci"][0]),
            float(record["first_token_fidelity"]),
            -abs(float(record["answer_weight"]) - 0.5),
        ),
    )
