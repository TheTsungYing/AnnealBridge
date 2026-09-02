"""Shared pure arithmetic for validation, penalty and compilation.

Single source of truth (Phase 2 spec §20) for the formulas that both the
problem validator (estimates, warnings) and the compiler / penalty strategy
rely on. Everything here is deterministic arithmetic over the raw problem
models: no BQM is ever built, nothing is logged, and no solver is known.

``compiler/slack.py`` and ``penalty/strategy.py`` import these functions so
the validator's estimates can never drift from what compilation actually
produces.
"""

from dataclasses import dataclass
from typing import Iterable

from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
)


def accumulate_terms(terms: Iterable[LinearTerm]) -> dict[str, float]:
    """Sum coefficients per variable; duplicate variables accumulate (spec §15)."""
    coefficients: dict[str, float] = {}
    for term in terms:
        coefficients[term.variable] = coefficients.get(term.variable, 0.0) + term.coefficient
    return coefficients


def lhs_bounds(coefficients: dict[str, float]) -> tuple[float, float]:
    """Return ``(lhs_min, lhs_max)`` of ``sum(a_i * x_i)`` over binary ``x_i``.

    Must be called on *accumulated* coefficients (one entry per variable):
    per-term bounds over-estimate the range when a variable repeats with
    mixed signs, which is exactly how validator and compiler used to
    disagree on trivial infeasibility.
    """
    lhs_min = sum(min(value, 0.0) for value in coefficients.values())
    lhs_max = sum(max(value, 0.0) for value in coefficients.values())
    return lhs_min, lhs_max


def compute_slack_coefficients(slack_range: int) -> list[int]:
    """Return binary-expansion coefficients for a slack in ``0..slack_range``.

    Uses ``m = slack_range.bit_length()`` bits (== ceil(log2(S+1)) without
    floating point): the first ``m - 1`` coefficients are ``1, 2, ..., 2^(m-2)``
    and the last is the remainder ``S - (2^(m-1) - 1)``, so every integer in
    ``0..S`` is representable and no bit combination exceeds ``S``.
    """
    if slack_range < 0:
        raise ValueError(f"slack range must be >= 0, got {slack_range}")
    if slack_range == 0:
        return []
    num_bits = slack_range.bit_length()
    coefficients = [1 << k for k in range(num_bits - 1)]
    coefficients.append(slack_range - ((1 << (num_bits - 1)) - 1))
    return coefficients


def compute_objective_scale(objective: Objective) -> float:
    """Return ``max(1.0, sum(|linear coeffs|) + sum(|quadratic coeffs|))``.

    For binary variables this is an upper bound on the objective's range,
    which penalty strategies use to scale hard penalties (spec §18). The
    term lists are used verbatim (duplicates are not merged first): the
    result is an upper bound either way and the spec formula reads over
    the raw coefficient lists.
    """
    total = sum(abs(term.coefficient) for term in objective.linear_terms)
    total += sum(abs(term.coefficient) for term in objective.quadratic_terms)
    return max(1.0, total)


@dataclass(frozen=True)
class InequalityAnalysis:
    """Pure-arithmetic view of one inequality constraint.

    ``coefficients`` and ``rhs`` are normalized to the ``<=`` form (a ``>=``
    constraint has both sides multiplied by -1) with zero coefficients
    dropped. ``slack_range`` keeps the raw ``int(round(rhs - lhs_min))``
    value — it is negative for a trivially infeasible constraint (a hard one
    is rejected by the validator before compilation; the compiler clamps a
    soft one) and is ``None`` when ``redundant``.
    """

    coefficients: dict[str, float]
    rhs: float
    lhs_min: float
    lhs_max: float
    redundant: bool
    slack_range: int | None


def analyze_inequality(constraint: Constraint) -> InequalityAnalysis:
    """Analyze a ``<=`` or ``>=`` constraint without building anything."""
    if constraint.operator not in ("<=", ">="):
        raise ValueError(
            f"analyze_inequality only handles inequalities, got "
            f"{constraint.operator!r} for constraint {constraint.id}"
        )

    coefficients = {
        variable: value
        for variable, value in accumulate_terms(constraint.terms).items()
        if value != 0.0
    }
    rhs = constraint.rhs
    if constraint.operator == ">=":
        coefficients = {variable: -value for variable, value in coefficients.items()}
        rhs = -rhs

    lhs_min, lhs_max = lhs_bounds(coefficients)
    redundant = lhs_max <= rhs
    slack_range = None if redundant else int(round(rhs - lhs_min))
    return InequalityAnalysis(
        coefficients=coefficients,
        rhs=rhs,
        lhs_min=lhs_min,
        lhs_max=lhs_max,
        redundant=redundant,
        slack_range=slack_range,
    )


def count_slack_bits(constraint: Constraint) -> int:
    """Number of slack variables compilation will generate for ``constraint``.

    Equality constraints and redundant inequalities need none. A trivially
    infeasible *soft* inequality counts as zero bits, matching the
    compiler's explicit clamp in ``encode_slack``. A trivially infeasible
    *hard* inequality is never silently clamped: ``validate_problem`` rejects
    it with TRIVIALLY_INFEASIBLE before any estimate is made, so reaching it
    here is a programming error and raises. The count goes through
    :func:`compute_slack_coefficients` — the same function ``encode_slack``
    uses — so the two can never drift.
    """
    if constraint.operator not in ("<=", ">="):
        return 0
    analysis = analyze_inequality(constraint)
    if analysis.redundant:
        return 0
    slack_range = analysis.slack_range
    assert slack_range is not None  # non-redundant analysis always sets it
    if slack_range < 0:
        if constraint.type == "hard":
            raise ValueError(
                f"Hard constraint {constraint.id} is trivially infeasible; "
                "validate_problem must reject it before estimating"
            )
        return 0
    return len(compute_slack_coefficients(slack_range))


def estimate_compiled_variables(problem: OptimizationProblem) -> int:
    """Estimate the compiled variable count: declared variables + slack bits.

    Pure arithmetic (spec §20) — no BQM is built. For a problem that passes
    ``validate_problem`` this equals ``BQMCompiler.compile(...).num_variables``
    exactly: the compiler adds every declared variable plus one variable per
    slack bit.
    """
    declared = len({variable.name for variable in problem.variables})
    slack_bits = sum(count_slack_bits(constraint) for constraint in problem.constraints)
    return declared + slack_bits
