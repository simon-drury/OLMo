"""The initial computational contract for explicit SFL meaning states.

This module intentionally defines the theory-to-computation boundary before a
UAM adapter or a training loop is attached. It does not claim that a nine-value
vector exhausts Systemic Functional Linguistic description.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence, Tuple


# =============================================================================
# SFL MEANING-STATE ARCHITECTURE: THEORY, MATHEMATICS, AND IMPLEMENTATION NOTE
# =============================================================================
#
# SYMBOL LEGEND
# -------------
# t          : discourse / trajectory step
# M[t]       : rich SFL meaning state at step t
# z[t]       : current 9D numerical representation of M[t]
# project_9D : named projection map, MeaningState -> R^9
# R^9        : nine-dimensional space of real-number vectors
#
# PURPOSE
# -------
# The native semantic state is an explicit Systemic Functional Linguistic (SFL)
# meaning configuration. The 3 x 3 grid below is its stable top-level spine:
#
#                    field              tenor              mode
#                +------------------+------------------+------------------+
# ideational     | M[I,F,t]         | M[I,Tn,t]        | M[I,M,t]         |
#                +------------------+------------------+------------------+
# interpersonal  | M[P,F,t]         | M[P,Tn,t]        | M[P,M,t]         |
#                +------------------+------------------+------------------+
# textual        | M[X,F,t]         | M[X,Tn,t]        | M[X,M,t]         |
#                +------------------+------------------+------------------+
#
# Formally:
#
#     M[t] =
#       [ M[I,F,t]  M[I,Tn,t]  M[I,M,t]  ]
#       [ M[P,F,t]  M[P,Tn,t]  M[P,M,t]  ]
#       [ M[X,F,t]  M[X,Tn,t]  M[X,M,t]  ]
#
# Rows are metafunctional dimensions:
#     I = ideational, P = interpersonal, X = textual.
#
# Columns are register variables:
#     F = field, Tn = tenor, M = mode.
#
# A CELL IS A CONTAINER, NOT A SCALAR
# -----------------------------------
# The grid does not claim that SFL has only nine atomic variables. Each cell is
# an expandable meaning-state container. A future cell may contain system-
# network selections, continuous features, relations, uncertainty, confidence,
# and provenance at increasing systemic delicacy.
#
# The textual metafunction is a metafunctional dimension, not a flat variable.
# Its internal systems may include thematic structure (Theme/Rheme), information
# structure (Given/New), cohesive relations, periodicity, and discourse phase.
# Theme/Rheme and Given/New are related but distinct systems.
#
# COMPUTATIONAL PROJECTION FOR THE INITIAL EMPIRICAL RUN
# ------------------------------------------------------
# The initial trainable interface is a compact projection of the richer state:
#
#     z[t] = project_9D(M[t]) in R^9
#
# Read aloud: “z at time t is the 9D projection of the meaning state M at
# time t.”
#
#     project_9D : MeaningState -> R^9
#
# In this first contract, project_9D preserves the row-major top-level order:
#
#     z[t] = [ z[I,F], z[I,Tn], z[I,M],
#              z[P,F], z[P,Tn], z[P,M],
#              z[X,F], z[X,Tn], z[X,M] ]^T
#
# Read aloud: “z is the nine-dimensional vector obtained by reading the
# ideational row, then the interpersonal row, then the textual row.”
#
# The nine values are a computational interface, not the ontology of the model.
# A trajectory model can later predict successive projected meaning states:
#
#     z_hat[t + 1] = f_theta(z[0:t], C[t])
#
# Read aloud: “the predicted meaning representation at the next step equals a
# trainable function of the preceding representations and current context.”
#
# A later adapter must construct MeaningState3x3 objects only from versioned,
# empirically specified UAM trajectory records. This module deliberately has no
# tokeniser, vocabulary projection, text corpus fallback, or token loss.
# =============================================================================


@dataclass(frozen=True)
class MeaningState3x3:
    """A compact 3 x 3 numerical projection of a richer SFL meaning state.

    `values` follows the documented row-major order. The immutable container
    preserves the semantic ordering at the boundary between empirical data and
    a future SFL trajectory model.
    """

    values: Tuple[float, float, float, float, float, float, float, float, float]

    SHAPE: ClassVar[Tuple[int, int]] = (3, 3)
    DIMENSION: ClassVar[int] = 9
    ROWS: ClassVar[Tuple[str, str, str]] = (
        "ideational",
        "interpersonal",
        "textual",
    )
    COLUMNS: ClassVar[Tuple[str, str, str]] = ("field", "tenor", "mode")

    def __post_init__(self) -> None:
        if len(self.values) != self.DIMENSION:
            raise ValueError(
                f"MeaningState3x3 requires {self.DIMENSION} values; "
                f"received {len(self.values)}."
            )

    @classmethod
    def from_flat_9d(cls, values: Sequence[float]) -> "MeaningState3x3":
        """Construct the current 3 x 3 projection from nine ordered values."""
        if len(values) != cls.DIMENSION:
            raise ValueError(
                f"MeaningState3x3 requires {cls.DIMENSION} values; "
                f"received {len(values)}."
            )
        return cls(tuple(float(value) for value in values))  # type: ignore[arg-type]

    def at(self, metafunction: str, register_variable: str) -> float:
        """Return one top-level cell using explicit SFL axis names."""
        try:
            row = self.ROWS.index(metafunction)
            column = self.COLUMNS.index(register_variable)
        except ValueError as exc:
            raise KeyError(
                "Unknown SFL coordinate. Expected metafunction in "
                f"{self.ROWS} and register variable in {self.COLUMNS}."
            ) from exc
        return self.values[row * self.SHAPE[1] + column]

    def project_9d(self) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """Return z[t] = project_9D(M[t]) in documented row-major order."""
        return self.values


def project_9d(
    meaning_state: MeaningState3x3,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Map a compact meaning state to its current nine-dimensional interface."""
    return meaning_state.project_9d()
