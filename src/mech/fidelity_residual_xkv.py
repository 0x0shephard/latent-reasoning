"""Fidelity-sensitive residual bases and frozen selection rules for xKV."""
from __future__ import annotations

from collections.abc import Mapping

import torch


def fit_margin_gradient_bases(
    key_gradients: torch.Tensor,
    value_gradients: torch.Tensor,
    *,
    maximum_rank: int,
    seed: int,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, dict]]:
    """Fit low-rank feature bases to dense top-1 margin cache gradients.

    Gradients have shape ``[examples, layers, latent_positions, width]``.  The
    right singular vectors identify feature directions to which the dense
    model's top-1-versus-runner-up margin is most sensitive.  The basis is
    static model metadata; only per-request residual coordinates count as KV
    storage in the downstream reference implementation.
    """
    if key_gradients.shape != value_gradients.shape or key_gradients.ndim != 4:
        raise ValueError("K/V gradients must share [N,L,P,D]")
    if not 0 < int(maximum_rank) <= key_gradients.shape[-1]:
        raise ValueError("maximum_rank must fit the cache width")
    result: dict[tuple[int, str], torch.Tensor] = {}
    audit: dict[str, dict] = {}
    for layer in range(key_gradients.shape[1]):
        for kind_index, (kind, gradients) in enumerate(
            (("key", key_gradients), ("value", value_gradients))
        ):
            matrix = gradients[:, layer].reshape(-1, gradients.shape[-1]).float()
            if not bool(torch.isfinite(matrix).all()) or not float(matrix.norm()):
                raise ValueError(f"invalid {kind} gradients at layer {layer}")
            q = min(matrix.shape, int(maximum_rank) + 4)
            with torch.random.fork_rng():
                torch.manual_seed(int(seed) + 100 * layer + kind_index)
                _, singular, vectors = torch.pca_lowrank(
                    matrix, q=q, center=False, niter=4
                )
            basis = vectors[:, : int(maximum_rank)].contiguous().cpu()
            result[(layer, kind)] = basis
            total_energy = float(matrix.double().square().sum())
            captured = singular[: int(maximum_rank)].double().square().cumsum(0)
            audit[f"layer_{layer:02d}_{kind}"] = {
                "gradient_frobenius_norm": float(matrix.double().norm()),
                "maximum_rank": int(maximum_rank),
                "cumulative_gradient_energy_fraction": [
                    float(value / total_energy) for value in captured
                ],
            }
    return result, audit


def truncate_residual_bases(
    bases: Mapping[tuple[int, str], torch.Tensor], rank: int
) -> dict[tuple[int, str], torch.Tensor]:
    if rank <= 0:
        raise ValueError("residual rank must be positive")
    result = {}
    for component, basis in bases.items():
        if basis.ndim != 2 or basis.shape[1] < int(rank):
            raise ValueError("basis does not contain the requested residual rank")
        result[(int(component[0]), str(component[1]))] = basis[:, : int(rank)]
    return result


def select_residual_candidate(records: list[dict]) -> dict | None:
    """Choose compression first, then fidelity/KL, then NLL evidence."""
    eligible = [record for record in records if bool(record.get("screen_passed"))]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            float(record["modelled_compression_ratio"]),
            float(record["first_token_fidelity"]),
            -float(record["mean_first_token_kl_from_dense"]),
            float(record["nll_bootstrap_95ci"][0]),
            -int(record["residual_rank"]),
        ),
    )
