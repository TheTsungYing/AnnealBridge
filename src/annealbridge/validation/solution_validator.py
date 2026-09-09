"""Constraint validation of candidate solutions (spec §23).

Feasibility is judged against the *original* OptimizationProblem, never
against BQM energy. The sample must already have internal (slack)
variables removed by the caller.

Two entry points share one arithmetic:

- :func:`validate` evaluates a single sample and builds the full
  ``ValidationResult`` with one ``ConstraintEvaluation`` per constraint.
- :func:`validate_batch` evaluates a whole matrix of candidates at once and
  returns only the per-candidate verdicts (``feasible`` and
  ``soft_violation_score``) as arrays. Every candidate is still re-checked
  against the original problem (overview principle 2); only the report
  objects are skipped, because building thousands of Pydantic models per
  solve dominated the run time while the ranking only ever reads the
  evaluations of the top-k.

Both paths accumulate the constraint's left-hand side term by term in
term order, compare with the same hybrid tolerance rule of
:mod:`annealbridge.validation.tolerance` (spec §23.1, review F-05), and compute
``weight * violation * violation`` in the same association, so their
results are bit-identical — not merely close. ``tests/unit`` asserts that
equality on random problems; keep the two kernels in lock-step when
touching either.
"""

import logging
from dataclasses import dataclass

import numpy as np

from annealbridge.models import (
    Constraint,
    ConstraintEvaluation,
    OptimizationProblem,
    ValidationResult,
)
from annealbridge.validation.tolerance import EPSILON, tolerance, tolerance_array

logger = logging.getLogger(__name__)

__all__ = ["EPSILON", "BatchValidation", "validate", "validate_batch"]


def validate(problem: OptimizationProblem, sample: dict[str, int]) -> ValidationResult:
    """Evaluate every constraint of ``problem`` against a business sample.

    ``sample`` maps each business variable name to its value (0/1 for a
    binary variable, an integer within its bounds for an integer one), with
    internal variables already stripped.
    """
    evaluations = [_evaluate(constraint, sample) for constraint in problem.constraints]
    hard_violations = [
        evaluation
        for evaluation in evaluations
        if evaluation.constraint_type == "hard" and not evaluation.satisfied
    ]
    soft_violations = [
        evaluation
        for evaluation in evaluations
        if evaluation.constraint_type == "soft" and not evaluation.satisfied
    ]
    soft_violation_score = 0.0
    for evaluation in soft_violations:
        soft_violation_score += evaluation.weighted_penalty
    return ValidationResult(
        feasible=not hard_violations,
        evaluations=evaluations,
        hard_violations=hard_violations,
        soft_violations=soft_violations,
        soft_violation_score=soft_violation_score,
    )


def _evaluate(constraint: Constraint, sample: dict[str, int]) -> ConstraintEvaluation:
    actual = 0.0
    for term in constraint.terms:
        actual += term.coefficient * sample[term.variable]
    rhs = constraint.rhs
    tol = tolerance(actual, rhs)

    if constraint.operator == "==":
        satisfied = abs(actual - rhs) <= tol
        violation = abs(actual - rhs)
    elif constraint.operator == "<=":
        satisfied = actual <= rhs + tol
        violation = max(0.0, actual - rhs)
    else:
        satisfied = actual >= rhs - tol
        violation = max(0.0, rhs - actual)

    violation_amount = 0.0 if satisfied else violation

    weighted_penalty: float | None = None
    if constraint.type == "soft":
        weighted_penalty = constraint.weight * violation_amount * violation_amount

    return ConstraintEvaluation(
        constraint_id=constraint.id,
        constraint_type=constraint.type,
        satisfied=satisfied,
        actual_value=actual,
        operator=constraint.operator,
        expected_value=rhs,
        violation_amount=violation_amount,
        weighted_penalty=weighted_penalty,
    )


@dataclass(frozen=True)
class BatchValidation:
    """Per-candidate verdicts of :func:`validate_batch`, aligned by row."""

    feasible: np.ndarray  # bool, shape (candidates,)
    soft_violation_score: np.ndarray  # float64, shape (candidates,)


def validate_batch(
    problem: OptimizationProblem,
    variables: list[str],
    samples: np.ndarray,
) -> BatchValidation:
    """Evaluate every constraint against every row of ``samples`` at once.

    ``samples`` is an integer matrix of shape ``(candidates, len(variables))``
    whose column ``j`` holds the value of business variable
    ``variables[j]`` (0/1 for a binary variable, an integer within its
    bounds for an integer one); internal variables must already be
    stripped. Returns
    the same ``feasible`` verdict and ``soft_violation_score`` that
    :func:`validate` would produce for each row, computed with identical
    arithmetic, without building per-constraint report objects.
    """
    if samples.ndim != 2 or samples.shape[1] != len(variables):
        raise ValueError(
            f"samples must have shape (candidates, {len(variables)}), "
            f"got {samples.shape}"
        )
    column = {name: index for index, name in enumerate(variables)}
    count = samples.shape[0]
    feasible = np.ones(count, dtype=bool)
    soft_violation_score = np.zeros(count, dtype=np.float64)

    for constraint in problem.constraints:
        actual = np.zeros(count, dtype=np.float64)
        for term in constraint.terms:
            try:
                actual += term.coefficient * samples[:, column[term.variable]]
            except KeyError:
                raise KeyError(term.variable) from None
        rhs = constraint.rhs
        tol = tolerance_array(actual, rhs)

        if constraint.operator == "==":
            satisfied = np.abs(actual - rhs) <= tol
            violation = np.abs(actual - rhs)
        elif constraint.operator == "<=":
            satisfied = actual <= rhs + tol
            violation = np.maximum(0.0, actual - rhs)
        else:
            satisfied = actual >= rhs - tol
            violation = np.maximum(0.0, rhs - actual)

        if constraint.type == "hard":
            feasible &= satisfied
        else:
            violation_amount = np.where(satisfied, 0.0, violation)
            soft_violation_score += (
                constraint.weight * violation_amount * violation_amount
            )

    return BatchValidation(feasible=feasible, soft_violation_score=soft_violation_score)
