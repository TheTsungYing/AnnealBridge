"""Penalty strategies for hard-constraint weighting (spec §18).

``compute_objective_scale`` / ``compute_penalty_scale`` are the pure formulas
shared by any strategy — they live in ``annealbridge.validation.estimates``
(Phase 2 spec §20, so the problem validator reports the same scale) and are
re-exported here.
``ScaledPenaltyStrategy`` is the Phase 1 implementation. Both are fully
deterministic: the same problem always yields the same penalties.
"""

from typing import Protocol

from annealbridge.models import OptimizationProblem
from annealbridge.validation.estimates import (
    compute_objective_scale,
    compute_penalty_scale,
    variable_bounds,
)

__all__ = [
    "PenaltyStrategy",
    "ScaledPenaltyStrategy",
    "compute_objective_scale",
    "compute_penalty_scale",
]


class PenaltyStrategy(Protocol):
    """Chooses the hard-constraint penalty for each solve attempt."""

    def initial_penalty(self, problem: OptimizationProblem) -> float:
        """Penalty for the first attempt."""
        ...

    def next_penalty(self, previous: float, attempt: int) -> float:
        """Penalty for the attempt following ``attempt`` (1-based)."""
        ...

    def objective_scale(self, problem: OptimizationProblem) -> float:
        """Estimated range of the objective alone, for trace output."""
        ...

    def penalty_scale(self, problem: OptimizationProblem) -> float:
        """Energy range the hard penalty must dominate (objective + soft terms)."""
        ...


class ScaledPenaltyStrategy:
    """Scales the hard penalty from the non-penalty energy range.

    ``initial_penalty = penalty_scale * solver.penalty_multiplier`` where
    ``penalty_scale = objective_scale + soft energy bound`` (spec §18), and
    every retry doubles the previous penalty. Without soft constraints
    ``penalty_scale == objective_scale``. Not claimed to be mathematically
    optimal; the multiplier is overridable for debugging.

    With integer variables (3b §8, corrected by the 2026-09-09 review,
    F-06) ``objective_scale`` is an upper bound on the objective's *range*
    ``objective_max - objective_min`` over the declared bounds, which is
    the quantity the §18 derivation needs; the soft bound is taken over the
    same ranges. With ``penalty_multiplier > 1`` the compiled model's global
    minimum is therefore feasible whenever a feasible assignment exists
    (given integer coefficients, so a violation costs at least one unit).
    For an all-binary problem every number is identical to 3a.
    """

    def objective_scale(self, problem: OptimizationProblem) -> float:
        """Return the §18 objective-only scale for ``problem``."""
        return compute_objective_scale(problem.objective, variable_bounds(problem))

    def penalty_scale(self, problem: OptimizationProblem) -> float:
        """Return the §18 penalty scale (objective scale + soft bound)."""
        return compute_penalty_scale(problem)

    def initial_penalty(self, problem: OptimizationProblem) -> float:
        """Return ``penalty_scale * penalty_multiplier``."""
        return self.penalty_scale(problem) * problem.solver.penalty_multiplier

    def next_penalty(self, previous: float, attempt: int) -> float:
        """Double the previous penalty for the next retry."""
        return previous * 2.0
