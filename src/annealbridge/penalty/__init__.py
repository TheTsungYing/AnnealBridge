"""Penalty strategies (spec §18)."""

from annealbridge.penalty.strategy import (
    PenaltyStrategy,
    ScaledPenaltyStrategy,
    compute_objective_scale,
)

__all__ = ["PenaltyStrategy", "ScaledPenaltyStrategy", "compute_objective_scale"]
