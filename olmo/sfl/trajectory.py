"""Minimal empirical SFL meaning-trajectory utilities.

The current port uses nine-dimensional projections of clause-level meaning
states. A transition is formed only between adjacent clauses within the
same source document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

STATE_DIM = 9


@dataclass(frozen=True)
class TrajectoryArrays:
    """Validated arrays loaded from a UAM meaning-trajectory NPZ artifact."""

    matrices: np.ndarray
    deltas: np.ndarray
    doc_ids: np.ndarray
    clause_ids: np.ndarray


def load_trajectory_arrays(path: str | Path) -> TrajectoryArrays:
    """Load the established UAM trajectory artifact and validate its contract."""
    with np.load(path, allow_pickle=False) as data:
        required = {"matrices", "deltas", "doc_ids", "clause_ids"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"trajectory artifact is missing arrays: {sorted(missing)}")
        matrices = np.asarray(data["matrices"], dtype=np.float32)
        deltas = np.asarray(data["deltas"], dtype=np.float32)
        doc_ids = np.asarray(data["doc_ids"])
        clause_ids = np.asarray(data["clause_ids"])

    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"matrices must have shape (N, 3, 3), got {matrices.shape}")
    if deltas.shape != matrices.shape:
        raise ValueError("deltas must have the same shape as matrices")
    n = matrices.shape[0]
    if doc_ids.shape != (n,) or clause_ids.shape != (n,):
        raise ValueError("doc_ids and clause_ids must have shape (N,)")
    return TrajectoryArrays(matrices, deltas, doc_ids, clause_ids)


def meaning_projection(matrices: np.ndarray) -> np.ndarray:
    """Return the row-major nine-dimensional current projection interface."""
    matrices = np.asarray(matrices, dtype=np.float32)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError("meaning matrices must end with shape (3, 3)")
    return matrices.reshape(*matrices.shape[:-2], STATE_DIM)


def successive_clause_indices(doc_ids: np.ndarray, clause_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return index pairs for adjacent ordered clauses inside each document."""
    doc_ids = np.asarray(doc_ids)
    clause_ids = np.asarray(clause_ids)
    if doc_ids.ndim != 1 or clause_ids.ndim != 1 or len(doc_ids) != len(clause_ids):
        raise ValueError("doc_ids and clause_ids must be equally sized one-dimensional arrays")

    current: list[int] = []
    following: list[int] = []
    for document in np.unique(doc_ids):
        members = np.flatnonzero(doc_ids == document)
        ordered = members[np.argsort(clause_ids[members], kind="stable")]
        current.extend(ordered[:-1])
        following.extend(ordered[1:])
    return np.asarray(current, dtype=np.int64), np.asarray(following, dtype=np.int64)


class SuccessiveClauseDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Observed current-state, state-change, and next-state triples."""

    def __init__(self, arrays: TrajectoryArrays):
        states = meaning_projection(arrays.matrices)
        current, following = successive_clause_indices(arrays.doc_ids, arrays.clause_ids)
        self.current = torch.from_numpy(states[current])
        self.change = torch.from_numpy(states[following] - states[current])
        self.next = torch.from_numpy(states[following])

    def __len__(self) -> int:
        return self.current.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.current[index], self.change[index], self.next[index]


class DeltaTransitionModel(nn.Module):
    """A compact random-initialised model of observed 9D state change."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, STATE_DIM),
        )

    def forward(self, current: Tensor) -> Tensor:
        return self.network(current)


def trajectory_loss(current: Tensor, observed_change: Tensor, next_state: Tensor, predicted_change: Tensor) -> Tensor:
    """Combine observed-change MSE with succeeding-state directional agreement."""
    predicted_next = current + predicted_change
    delta_error = F.mse_loss(predicted_change, observed_change)
    directional_error = 1.0 - F.cosine_similarity(predicted_next, next_state, dim=-1).mean()
    return delta_error + 0.5 * directional_error
