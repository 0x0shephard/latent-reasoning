"""Fixed per-head K/V bases for CODI: fit once, project every cached vector.

xKV factorizes each request's cache matrix and must store a per-request decoder,
so at CODI's ~100-token cache length rank 64 barely compresses.  This module
instead fits one orthonormal basis per (layer, kind, head) on calibration
questions and projects every key and value written by that head onto its leading
``r`` directions.  The basis is model metadata; each cached vector costs ``r``
coordinates instead of ``head_dim``.

Because GPT-2 uses learned absolute positions (no rotary embedding on keys), the
projection folds algebraically into attention: ``q . (U U^T k) = (U^T q) . (U^T k)``
and ``sum_i p_i U U^T v_i = U (sum_i p_i U^T v_i)``.  The hook below implements
the projected model exactly; it is the unfused reference for an r-dimensional
per-head attention.

Everything here operates on hidden states and module outputs only.  Dataset and
protocol code lives in the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Sequence

import torch
import torch.nn as nn

from src.models.official_codi import (
    _normalized_official_questions,
    official_codi_base_model,
)


ROW_TYPES = ("question", "latent", "cue", "answer")
KINDS = ("key", "value")


def attention_qkv_modules(model) -> list[nn.Module]:
    """Return each block's fused query/key/value projection in layer order."""
    base = official_codi_base_model(model)
    transformer = getattr(base, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        raise TypeError("expected a GPT-2 style causal LM with transformer.h blocks")
    modules = []
    for block in blocks:
        module = getattr(getattr(block, "attn", None), "c_attn", None)
        if module is None:
            raise TypeError("expected every block to expose attn.c_attn")
        modules.append(module)
    return modules


def head_geometry(model) -> tuple[int, int, int]:
    """(layers, heads, head_dim) from the model configuration."""
    config = model.config
    layers = int(config.n_layer)
    heads = int(config.n_head)
    width = int(config.n_embd)
    if width % heads:
        raise ValueError("hidden width must divide evenly across heads")
    return layers, heads, width // heads


def _split_qkv(output: torch.Tensor, heads: int, head_dim: int):
    width = heads * head_dim
    if output.shape[-1] != 3 * width:
        raise ValueError("fused QKV output width does not match the head geometry")
    query = output[..., :width]
    key = output[..., width : 2 * width]
    value = output[..., 2 * width :]
    return query, key, value


class KVSecondMomentCollector:
    """Accumulate per-head, per-row-type second moments of cached keys and values.

    Attach it to a model, call :meth:`set_context` before every forward pass with
    the row type and a ``[batch, tokens]`` validity mask, and read the pooled and
    per-type moments afterwards.  Nothing about generation is changed.
    """

    def __init__(self, *, layers: int, heads: int, head_dim: int):
        if min(layers, heads, head_dim) <= 0:
            raise ValueError("layers, heads and head_dim must be positive")
        self.layers, self.heads, self.head_dim = int(layers), int(heads), int(head_dim)
        # Accumulators follow the observed tensors onto the accelerator.  Copying a
        # moment to the host on every forward pass would force one synchronization
        # per layer per step, which dominates calibration on a real model.
        self._moments = torch.zeros(
            len(ROW_TYPES), self.layers, len(KINDS), self.heads, self.head_dim, self.head_dim,
            dtype=torch.float64,
        )
        self._counts = torch.zeros(len(ROW_TYPES), dtype=torch.long)
        self._row_type: int | None = None
        self._mask: torch.Tensor | None = None
        self._handles: list = []

    @property
    def moments(self) -> torch.Tensor:
        """Second moments per row type, always on the host."""
        return self._moments.cpu()

    @property
    def counts(self) -> torch.Tensor:
        return self._counts.cpu()

    def _align_to(self, tensor: torch.Tensor) -> None:
        if self._moments.device != tensor.device:
            self._moments = self._moments.to(tensor.device)
            self._counts = self._counts.to(tensor.device)

    def set_context(self, row_type: str, mask: torch.Tensor | None) -> None:
        if row_type not in ROW_TYPES:
            raise ValueError(f"unknown row type {row_type!r}")
        self._row_type = ROW_TYPES.index(row_type)
        self._mask = None if mask is None else mask.detach().bool()

    def clear_context(self) -> None:
        self._row_type, self._mask = None, None

    def attach(self, model) -> "KVSecondMomentCollector":
        modules = attention_qkv_modules(model)
        if len(modules) != self.layers:
            raise ValueError("collector layer count does not match the model")
        for layer, module in enumerate(modules):
            self._handles.append(module.register_forward_hook(self._make_hook(layer)))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            if self._row_type is None:
                raise RuntimeError("set_context must be called before every forward pass")
            _, key, value = _split_qkv(output.detach(), self.heads, self.head_dim)
            batch, tokens = key.shape[:2]
            if self._mask is None:
                mask = torch.ones(batch, tokens, dtype=torch.bool, device=key.device)
            else:
                mask = self._mask.to(key.device)
                if mask.shape != (batch, tokens):
                    raise ValueError("validity mask must match [batch, tokens]")
            self._align_to(key)
            for kind_index, tensor in enumerate((key, value)):
                selected = tensor[mask].reshape(-1, self.heads, self.head_dim)
                if selected.shape[0] == 0:
                    continue
                # Accumulate in float64 but contract in the tensor's own dtype: the
                # contraction is a matmul (slow in float64 on consumer GPUs) while the
                # accumulation is an elementwise add (memory bound either way).
                moment = torch.einsum("nhd,nhe->hde", selected, selected).double()
                self._moments[self._row_type, layer, kind_index] += moment
            if layer == 0:
                self._counts[self._row_type] += mask.sum()
            return None
        return hook

    def pooled_moments(self) -> torch.Tensor:
        return self._moments.sum(0).cpu()

    def row_type_moments(self, row_type: str) -> torch.Tensor:
        return self._moments[ROW_TYPES.index(row_type)].cpu()

    def count_summary(self) -> dict[str, int]:
        counts = self._counts.cpu()
        return {name: int(counts[index]) for index, name in enumerate(ROW_TYPES)}


@dataclass
class HeadBases:
    """Orthonormal per-head bases with descending eigenvalues.

    ``vectors[layer, kind, head]`` is ``[head_dim, head_dim]`` with basis vectors as
    columns ordered by descending eigenvalue; ``eigenvalues`` is ``[..., head_dim]``.
    """

    vectors: torch.Tensor
    eigenvalues: torch.Tensor

    @property
    def head_dim(self) -> int:
        return int(self.vectors.shape[-1])


def fit_head_bases(moments: torch.Tensor) -> HeadBases:
    """Eigendecompose per-head second moments into descending orthonormal bases."""
    if moments.ndim != 5 or moments.shape[-1] != moments.shape[-2]:
        raise ValueError("moments must have shape [layers, kinds, heads, D, D]")
    symmetric = 0.5 * (moments.double() + moments.double().transpose(-1, -2))
    if not bool(torch.isfinite(symmetric).all()):
        raise ValueError("second moments contain non-finite values")
    eigenvalues, vectors = torch.linalg.eigh(symmetric)
    order = torch.arange(eigenvalues.shape[-1] - 1, -1, -1)
    eigenvalues = eigenvalues[..., order].clamp_min(0)
    vectors = vectors[..., :, order]
    return HeadBases(vectors=vectors.contiguous(), eigenvalues=eigenvalues.contiguous())


def random_head_bases(
    *, layers: int, kinds: int, heads: int, head_dim: int, seed: int
) -> HeadBases:
    """Seeded random orthonormal bases with the same shape as fitted bases."""
    generator = torch.Generator().manual_seed(int(seed))
    gaussian = torch.randn(
        layers, kinds, heads, head_dim, head_dim, generator=generator, dtype=torch.float64
    )
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(-2)
    eigenvalues = torch.full(q.shape[:-1], float("nan"), dtype=torch.float64)
    return HeadBases(vectors=q.contiguous(), eigenvalues=eigenvalues)


def energy_fraction_curves(eigenvalues: torch.Tensor) -> torch.Tensor:
    """Cumulative retained-energy fraction along the descending eigenvalue axis."""
    total = eigenvalues.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    return (eigenvalues.cumsum(-1) / total).clamp(0, 1)


def rank_for_energy(eigenvalues: torch.Tensor, fraction: float) -> torch.Tensor:
    """Smallest rank whose retained energy reaches ``fraction``."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    curves = energy_fraction_curves(eigenvalues)
    reached = curves >= float(fraction) - 1e-12
    return reached.long().argmax(-1) + 1


def uniform_ranks(
    *, layers: int, heads: int, rank: int, key_rank: int | None = None,
    value_rank: int | None = None,
) -> torch.Tensor:
    """Per-component rank tensor ``[layers, 2, heads]``; kinds may differ."""
    key_rank = int(rank if key_rank is None else key_rank)
    value_rank = int(rank if value_rank is None else value_rank)
    if min(key_rank, value_rank) <= 0:
        raise ValueError("ranks must be positive")
    ranks = torch.empty(layers, len(KINDS), heads, dtype=torch.long)
    ranks[:, 0] = key_rank
    ranks[:, 1] = value_rank
    return ranks


def allocate_energy_ranks(
    eigenvalues: torch.Tensor, *, total_rank: int, minimum_rank: int = 1
) -> torch.Tensor:
    """Spend a total rank budget across components by descending eigenvalue.

    Eigenvalues are non-increasing within each component, so marginal energy gains
    are concave and the greedy heap allocation maximizes total retained energy.
    """
    if eigenvalues.ndim != 4:
        raise ValueError("eigenvalues must have shape [layers, kinds, heads, D]")
    layers, kinds, heads, head_dim = eigenvalues.shape
    components = layers * kinds * heads
    minimum_rank = int(minimum_rank)
    if not 0 < minimum_rank <= head_dim:
        raise ValueError("minimum_rank must lie in [1, head_dim]")
    if not components * minimum_rank <= int(total_rank) <= components * head_dim:
        raise ValueError("total_rank must be feasible for the component count")
    flat = eigenvalues.reshape(components, head_dim).double()
    if bool((flat[:, 1:] > flat[:, :-1] + 1e-9 * flat.abs().max()).any()):
        raise ValueError("eigenvalues must be non-increasing within each component")
    ranks = [minimum_rank] * components
    allocated = components * minimum_rank
    heap = [
        (-float(flat[index, ranks[index]]), index)
        for index in range(components) if ranks[index] < head_dim
    ]
    heapq.heapify(heap)
    while allocated < int(total_rank) and heap:
        _, index = heapq.heappop(heap)
        ranks[index] += 1
        allocated += 1
        if ranks[index] < head_dim:
            heapq.heappush(heap, (-float(flat[index, ranks[index]]), index))
    return torch.tensor(ranks, dtype=torch.long).reshape(layers, kinds, heads)


def projectors_from_ranks(bases: HeadBases, ranks: torch.Tensor) -> torch.Tensor:
    """Orthogonal projectors ``U_r U_r^T`` per component; full rank is exact identity."""
    vectors = bases.vectors
    if ranks.shape != vectors.shape[:3]:
        raise ValueError("ranks must have shape [layers, kinds, heads]")
    head_dim = bases.head_dim
    if bool((ranks <= 0).any()) or bool((ranks > head_dim).any()):
        raise ValueError("ranks must lie in [1, head_dim]")
    positions = torch.arange(head_dim)
    keep = (positions[None, None, None, :] < ranks[..., None]).to(vectors.dtype)
    kept = vectors * keep.unsqueeze(-2)
    projectors = kept @ kept.transpose(-1, -2)
    full = ranks == head_dim
    if bool(full.any()):
        projectors[full] = torch.eye(head_dim, dtype=vectors.dtype)
    return projectors


class HeadProjectionHook:
    """Project every key and value a block writes onto its fixed per-head basis."""

    def __init__(self, projectors: torch.Tensor, *, ranks: torch.Tensor):
        if projectors.ndim != 5 or projectors.shape[-1] != projectors.shape[-2]:
            raise ValueError("projectors must have shape [layers, kinds, heads, D, D]")
        self.projectors = projectors
        self.ranks = ranks.clone()
        self.layers, self.kinds, self.heads, self.head_dim = (
            int(projectors.shape[0]), int(projectors.shape[1]),
            int(projectors.shape[2]), int(projectors.shape[3]),
        )
        self._handles: list = []
        self._device_projectors: torch.Tensor | None = None

    def attach(self, model) -> "HeadProjectionHook":
        modules = attention_qkv_modules(model)
        if len(modules) != self.layers:
            raise ValueError("projector layer count does not match the model")
        for layer, module in enumerate(modules):
            self._handles.append(module.register_forward_hook(self._make_hook(layer)))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()

    def _projectors_for(self, tensor: torch.Tensor) -> torch.Tensor:
        cached = self._device_projectors
        if cached is None or cached.device != tensor.device or cached.dtype != tensor.dtype:
            cached = self.projectors.to(device=tensor.device, dtype=tensor.dtype)
            self._device_projectors = cached
        return cached

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            projectors = self._projectors_for(output)
            query, key, value = _split_qkv(output, self.heads, self.head_dim)
            batch, tokens = key.shape[:2]
            pieces = [query]
            for kind_index, tensor in enumerate((key, value)):
                shaped = tensor.reshape(batch, tokens, self.heads, self.head_dim)
                projected = torch.einsum(
                    "bthd,hde->bthe", shaped, projectors[layer, kind_index]
                )
                pieces.append(projected.reshape(batch, tokens, self.heads * self.head_dim))
            return torch.cat(pieces, dim=-1)
        return hook

    def summary(self) -> dict:
        return storage_report(self.ranks, head_dim=self.head_dim)


def storage_report(ranks: torch.Tensor, *, head_dim: int) -> dict:
    """Modeled per-token cache storage for fixed-basis coordinates."""
    if ranks.ndim != 3:
        raise ValueError("ranks must have shape [layers, kinds, heads]")
    dense_units = int(ranks.numel()) * int(head_dim)
    stored_units = int(ranks.sum())
    compressed = ranks < int(head_dim)
    basis_parameters = int((ranks[compressed] * int(head_dim)).sum())
    return {
        "dense_units_per_token": dense_units,
        "stored_units_per_token": stored_units,
        "compression_ratio": dense_units / max(1, stored_units),
        "mean_rank": float(ranks.double().mean()),
        "key_mean_rank": float(ranks[:, 0].double().mean()),
        "value_mean_rank": float(ranks[:, 1].double().mean()),
        "basis_parameters": basis_parameters,
        "runtime_representation": "exact_projected_attention_reference",
    }


def subspace_capture(
    reference: torch.Tensor, candidate: torch.Tensor, rank: int
) -> torch.Tensor:
    """Fraction of the rank-``r`` reference subspace captured by the candidate's.

    ``||U_ref^T U_cand||_F^2 / r`` per component; one for identical subspaces and
    ``r / D`` in expectation for an isotropic random candidate.
    """
    if reference.shape != candidate.shape or reference.ndim < 2:
        raise ValueError("reference and candidate bases must share a shape")
    rank = int(rank)
    if not 0 < rank <= reference.shape[-1]:
        raise ValueError("rank must fit the basis")
    left = reference[..., :, :rank].double()
    right = candidate[..., :, :rank].double()
    overlap = left.transpose(-1, -2) @ right
    return overlap.square().sum((-1, -2)) / rank


@torch.inference_mode()
def collect_kv_second_moments(
    model,
    tokenizer,
    questions: Sequence[str],
    *,
    collector: KVSecondMomentCollector,
    latent_iterations: int,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    answer_cue: str = "The answer is:",
) -> dict:
    """Run the released forced-cue generation path while the collector observes.

    Mirrors ``generate_official_codi`` step for step so the second moments come
    from the exact positions CODI writes into its cache: question tokens plus the
    ``<bot>`` marker, six latent rows, the forced answer cue, and greedily decoded
    answer tokens up to EOS.
    """
    if latent_iterations <= 0 or max_new_tokens <= 0 or batch_size <= 0:
        raise ValueError("latent_iterations, max_new_tokens and batch_size must be positive")
    model.eval()
    embedding = model.input_embeddings()
    normalized = _normalized_official_questions(questions)
    cue_ids = list(tokenizer(f" {answer_cue}", add_special_tokens=False)["input_ids"])
    if not cue_ids:
        raise ValueError("answer cue must tokenize to at least one token")
    outputs: list[str] = []
    for start in range(0, len(normalized), batch_size):
        chunk = normalized[start : start + batch_size]
        batch = tokenizer(
            chunk, return_tensors="pt", padding="longest", add_special_tokens=False
        ).to(device)
        bot = torch.full((len(chunk), 1), model.bot_id, dtype=torch.long, device=device)
        input_ids = torch.cat((batch["input_ids"], bot), dim=1)
        attention_mask = torch.cat((batch["attention_mask"], torch.ones_like(bot)), dim=1)

        collector.set_context("question", attention_mask.bool())
        encoded = model.codi(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
        cache = encoded.past_key_values
        latent = model.prj(encoded.hidden_states[-1][:, -1, :].unsqueeze(1))
        for _ in range(latent_iterations):
            collector.set_context("latent", None)
            latent_output = model.codi(
                inputs_embeds=latent, past_key_values=cache, use_cache=True,
                output_hidden_states=True, return_dict=True,
            )
            cache = latent_output.past_key_values
            latent = model.prj(latent_output.hidden_states[-1][:, -1, :].unsqueeze(1))

        forced = torch.tensor(
            [model.eot_id, *cue_ids], dtype=torch.long, device=device
        ).unsqueeze(0).expand(len(chunk), -1)
        collector.set_context("cue", None)
        decoded = model.codi(
            inputs_embeds=embedding(forced), past_key_values=cache, use_cache=True,
            return_dict=True,
        )
        cache = decoded.past_key_values
        finished = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in chunk]
        next_token = decoded.logits[:, -1, : model.eot_id].argmax(dim=-1)
        for row, token_id in enumerate(next_token.tolist()):
            generated[row].append(int(token_id))
            if token_id == tokenizer.eos_token_id:
                finished[row] = True
        token_embedding = embedding(next_token).unsqueeze(1)
        for _ in range(1, max_new_tokens):
            if bool(finished.all()):
                break
            collector.set_context("answer", (~finished).unsqueeze(1))
            decoded = model.codi(
                inputs_embeds=token_embedding, past_key_values=cache, use_cache=True,
                return_dict=True,
            )
            cache = decoded.past_key_values
            next_token = decoded.logits[:, -1, : model.eot_id].argmax(dim=-1)
            for row, token_id in enumerate(next_token.tolist()):
                if not finished[row]:
                    generated[row].append(int(token_id))
                    if token_id == tokenizer.eos_token_id:
                        finished[row] = True
            token_embedding = embedding(next_token).unsqueeze(1)
        collector.clear_context()
        outputs.extend(
            tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in generated
        )
    return {"outputs": outputs, "row_counts": collector.count_summary()}
