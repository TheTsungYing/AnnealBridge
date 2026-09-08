"""Binary slack encoding for inequality constraints (spec §14).

Implemented from scratch on purpose (no ``dimod.add_linear_inequality_constraint``)
so that internal variable naming, constraint traces, and future scaling stay
fully under our control.

The pure arithmetic (term accumulation, ``<=`` normalization, slack range and
binary-expansion coefficients) lives in ``annealbridge.validation.estimates``
so the problem validator's ``estimated_compiled_variables`` can never drift
from what this module actually encodes (Phase 2 spec §20). This module keeps
the compiler-only concerns: raising for infeasible hard constraints, clamping
soft ones with a warning, and naming the generated slack variables.
"""

import logging
from dataclasses import dataclass

from annealbridge.exceptions import CompilationError
from annealbridge.models import Constraint
from annealbridge.validation.estimates import (
    Bounds,
    accumulate_terms,
    analyze_inequality,
    compute_slack_coefficients,
)

__all__ = [
    "InequalityEncoding",
    "accumulate_terms",
    "compute_slack_coefficients",
    "encode_slack",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InequalityEncoding:
    """Result of slack-encoding one inequality constraint.

    ``coefficients`` and ``constant`` are already normalized to the ``<=``
    form (a ``>=`` constraint has both sides multiplied by -1), so the
    penalty is ``lambda * (sum(coefficients) + slack - (-constant))^2``,
    i.e. ``constant`` can be fed directly into the squared-penalty expansion.
    ``slack_range`` is ``None`` for a redundant constraint, ``0`` when the
    encoding was clamped (soft, trivially infeasible) or the range is exactly
    zero, and positive otherwise.
    """

    coefficients: dict[str, float]
    slack_coefficients: dict[str, int]
    constant: float
    slack_range: int | None
    redundant: bool


def encode_slack(
    constraint: Constraint, bounds: Bounds | None = None
) -> InequalityEncoding:
    """Encode a ``<=`` or ``>=`` constraint with binary slack variables.

    ``bounds`` (``variable_bounds(problem)``) sizes the slack range from the
    variables' real ranges (3b §8); ``None`` treats every variable as binary,
    which is the Phase 1 behaviour. Production callers always pass it.

    Raises :class:`CompilationError` for a hard constraint that can never be
    satisfied. The validator judges trivial infeasibility on the same
    accumulated coefficients, so a validated problem never reaches this
    branch; it stays as a defence for callers that skip validation. A soft
    constraint in the same situation is clamped to zero slack bits with a
    warning: its penalty then degrades to the squared minimal violation,
    which is exactly the pressure a violated soft constraint should exert.
    """
    if constraint.operator not in ("<=", ">="):
        raise CompilationError(
            f"encode_slack only handles inequalities, got {constraint.operator!r} "
            f"for constraint {constraint.id}"
        )

    analysis = analyze_inequality(constraint, bounds)
    if analysis.redundant:
        return InequalityEncoding(
            coefficients=analysis.coefficients,
            slack_coefficients={},
            constant=-analysis.rhs,
            slack_range=None,
            redundant=True,
        )

    slack_range = analysis.slack_range
    assert slack_range is not None  # non-redundant analysis always sets it
    if slack_range < 0:
        if constraint.type == "hard":
            raise CompilationError(
                f"Hard constraint {constraint.id} can never be satisfied after "
                f"accumulating terms: lhs minimum {analysis.lhs_min} exceeds "
                f"rhs {analysis.rhs}"
            )
        logger.warning(
            "Soft constraint %s can never be satisfied (lhs range starts at %s, "
            "rhs %s); clamping slack to 0 bits so its penalty tracks the "
            "minimal violation",
            constraint.id,
            analysis.lhs_min,
            analysis.rhs,
        )
        slack_range = 0

    slack_coefficients = {
        f"__slack_{constraint.id}_{k}": value
        for k, value in enumerate(compute_slack_coefficients(slack_range))
    }
    return InequalityEncoding(
        coefficients=analysis.coefficients,
        slack_coefficients=slack_coefficients,
        constant=-analysis.rhs,
        slack_range=slack_range,
        redundant=False,
    )
