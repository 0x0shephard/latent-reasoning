"""Bridge a discovered activation subspace to a learned low-rank vocabulary head.

The endpoint experiment describes a column subspace ``U`` in hidden-state space,
whereas a factorized vocabulary head learns a down projection ``D``.  The rows of
``D`` span the hidden-state subspace visible to that head.  This module provides the
small, model-independent pieces needed to compare those spaces and to intervene on
the input of an arbitrary vocabulary head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SubspaceOverlap:
    """Principal-angle and asymmetric capture statistics for two subspaces."""

    ambient_dimension: int
    reference_rank: int
    candidate_rank: int
    shared_energy: float
    reference_capture_fraction: float
    candidate_occupancy_fraction: float
    mean_squared_cosine: float
    minimum_cosine: float
    mean_cosine: float
    maximum_cosine: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _orthonormal_columns(basis: torch.Tensor, *, name: str) -> torch.Tensor:
    if basis.ndim != 2 or min(basis.shape) <= 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if basis.shape[1] > basis.shape[0]:
        raise ValueError(f"{name} cannot have more columns than ambient dimensions")
    if not torch.isfinite(basis).all():
        raise ValueError(f"{name} contains non-finite values")
    result = torch.linalg.qr(basis.double(), mode="reduced").Q
    return result.to(device=basis.device, dtype=basis.dtype)


def orthonormal_row_space(matrix: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal column basis for the row span of ``matrix``.

    A learned down projection has shape ``[rank, hidden]``.  Its factorization is
    not unique, but its row span is invariant to any invertible change of bottleneck
    coordinates, making this the correct object to compare with activation PCs.
    """
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("matrix must be a non-empty rank-by-hidden matrix")
    if matrix.shape[0] > matrix.shape[1]:
        raise ValueError("row-space extraction expects rank <= hidden dimension")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    # SVD handles a numerically rank-deficient learned factor more honestly than QR.
    _, singular_values, right_t = torch.linalg.svd(matrix.double(), full_matrices=False)
    tolerance = (
        max(matrix.shape)
        * torch.finfo(torch.float64).eps
        * float(singular_values.max())
    )
    numerical_rank = int((singular_values > tolerance).sum())
    if numerical_rank == 0:
        raise ValueError("matrix has zero numerical rank")
    return right_t[:numerical_rank].T.to(device=matrix.device, dtype=matrix.dtype)


def subspace_overlap(reference: torch.Tensor, candidate: torch.Tensor) -> SubspaceOverlap:
    """Measure how much of ``reference`` lies in ``candidate``.

    ``reference_capture_fraction`` is ``||Q_candidate^T Q_reference||_F^2 / r_ref``.
    It equals one when every reference direction is available to the candidate.
    ``candidate_occupancy_fraction`` divides the same shared energy by candidate
    rank and answers the distinct question: how much of the candidate is occupied
    by the reference space?
    """
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("subspaces must be matrices")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("subspaces must have the same ambient dimension")
    left = _orthonormal_columns(reference, name="reference")
    right = _orthonormal_columns(candidate, name="candidate")
    cosines = torch.linalg.svdvals(left.double().T @ right.double())
    shared = float(cosines.square().sum())
    return SubspaceOverlap(
        ambient_dimension=int(reference.shape[0]),
        reference_rank=int(left.shape[1]),
        candidate_rank=int(right.shape[1]),
        shared_energy=shared,
        reference_capture_fraction=shared / left.shape[1],
        candidate_occupancy_fraction=shared / right.shape[1],
        mean_squared_cosine=float(cosines.square().mean()),
        minimum_cosine=float(cosines.min()),
        mean_cosine=float(cosines.mean()),
        maximum_cosine=float(cosines.max()),
    )


def edit_hidden_subspace(
    hidden: torch.Tensor,
    basis: torch.Tensor,
    centre: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Retain or remove a centred subspace from hidden states of any leading shape."""
    if mode not in {"retain", "remove"}:
        raise ValueError("mode must be 'retain' or 'remove'")
    if hidden.shape[-1] != basis.shape[0] or centre.shape != (basis.shape[0],):
        raise ValueError("hidden, basis, and centre dimensions do not match")
    columns = _orthonormal_columns(basis.to(hidden), name="basis")
    origin = centre.to(hidden)
    centered = hidden - origin
    projected = (centered @ columns) @ columns.T
    return origin + projected if mode == "retain" else hidden - projected


class SubspaceInterventionVocabularyHead(nn.Module):
    """Apply a hidden-space edit immediately before a wrapped vocabulary head.

    ``active_positions=None`` edits every visible answer position.  Otherwise the
    decoder must call :meth:`set_answer_position`; only listed zero-based answer
    positions are edited.  The position is propagated to the wrapped head so this
    composes with nested or position-conditioned heads.
    """

    def __init__(
        self,
        head: nn.Module,
        basis: torch.Tensor,
        centre: torch.Tensor,
        *,
        mode: str,
        vocabulary_size: int,
        active_positions: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"retain", "remove"}:
            raise ValueError("mode must be 'retain' or 'remove'")
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        columns = _orthonormal_columns(basis.detach(), name="basis")
        if centre.shape != (columns.shape[0],):
            raise ValueError("centre width must match the basis ambient dimension")
        positions = (
            None
            if active_positions is None
            else frozenset(int(x) for x in active_positions)
        )
        if positions is not None and any(value < 0 for value in positions):
            raise ValueError("active positions must be non-negative")
        self.head = head
        self.register_buffer("basis", columns.clone())
        self.register_buffer("centre", centre.detach().to(columns).clone())
        self.mode = mode
        self.vocabulary_size = int(vocabulary_size)
        self.active_positions = positions
        self._answer_position: int | None = None

    def set_answer_position(self, position: int | None) -> None:
        if position is not None and int(position) < 0:
            raise ValueError("answer position must be non-negative")
        self._answer_position = None if position is None else int(position)
        setter = getattr(self.head, "set_answer_position", None)
        if callable(setter):
            setter(position)

    def is_active(self) -> bool:
        if self._answer_position is None:
            return False
        return self.active_positions is None or self._answer_position in self.active_positions

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = hidden_states
        if self.is_active():
            values = edit_hidden_subspace(
                values, self.basis, self.centre, mode=self.mode
            )
        logits = self.head(values)
        if logits.shape[-1] != self.vocabulary_size:
            raise ValueError(
                f"wrapped head returned vocabulary {logits.shape[-1]}, "
                f"expected {self.vocabulary_size}"
            )
        return logits
