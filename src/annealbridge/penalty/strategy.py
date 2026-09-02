"""Penalty strategies for hard-constraint weighting (spec §18).

``compute_objective_scale`` is the pure formula shared by any strategy — it
lives in ``annealbridge.validation.estimates`` (Phase 2 spec §20, so the
problem validator reports the same scale) and is re-exported here.
``ScaledPenaltyStrategy`` is the Phase 1 implementation. Both are fully
deterministic: the same problem always yields the same penalties.
"""

from typing import Protocol

from annealbridge.models import OptimizationProblem
from annealbridge.validation.estimates import compute_objective_scale

__all__ = ["PenaltyStrategy", "ScaledPenaltyStrategy", "compute_objective_scale"]


class PenaltyStrategy(Protocol):
    """Chooses the hard-constraint penalty for each solve attempt."""

    def initial_penalty(self, problem: OptimizationProblem) -> float:
        """Penalty for the first attempt."""
        ...

    def next_penalty(self, previous: float, attempt: int) -> float:
        """Penalty for the attempt following ``attempt`` (1-based)."""
        ...

    def objective_scale(self, problem: OptimizationProblem) -> float:
        """Estimated range of the objective, for trace output."""
        ...


class ScaledPenaltyStrategy:
    """Scales the hard penalty from the objective's coefficient range.

    ``initial_penalty = objective_scale * solver.penalty_multiplier`` and
    every retry doubles the previous penalty. Not claimed to be
    mathematically optimal; the multiplier is overridable for debugging.
    """

    def objective_scale(self, problem: OptimizationProblem) -> float:
        """Return the §18 objective scale for ``problem``."""
        return compute_objective_scale(problem.objective)

    def initial_penalty(self, problem: OptimizationProblem) -> float:
        """Return ``objective_scale * penalty_multiplier``."""
        return self.objective_scale(problem) * problem.solver.penalty_multiplier

    def next_penalty(self, previous: float, attempt: int) -> float:
        """Double the previous penalty for the next retry."""
        return previous * 2.0
