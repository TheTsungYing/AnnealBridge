"""Constraint validation of candidate solutions (spec §23).

Feasibility is judged against the *original* OptimizationProblem, never
against BQM energy. The sample must already have internal (slack)
variables removed by the caller.

Two entry points share one arithmetic:

- :func:`validate` evaluates a single sample and builds the full
  ``ValidationResult`` with one ``ConstraintEvaluation`` per constraint.
- :func:`validate_batch` evaluates a whole matrix of candidates at once and
  returns only the per-candidate verdicts (``feasible``,
  ``soft_violation_score`` and ``hard_violation_total``) plus a per-hard-
  constraint violation tally as arrays. Every candidate is still re-checked
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

Soft scores do not use the feasibility tolerance
------------------------------------------------
The tolerance decides ``satisfied`` for every constraint, and it zeroes a
*hard* constraint's ``violation_amount`` once it holds. A **soft**
constraint's ``violation_amount`` is instead the *exact* residual
(``abs(actual - rhs)``, ``max(0, actual - rhs)`` or ``max(0, rhs - actual)``),
whatever ``satisfied`` says, and ``weighted_penalty`` squares it. That is
what the compilers write into the model: the BQM squared penalty and the
CQM's native weighted constraint both cost ``weight * residual**2`` with no
tolerance band, so rounding a residual inside the band down to zero would
make the ranking prefer assignments the solver is paying to avoid — visibly
so at a huge weight (review F-05, 2026-09-11). ``soft_violation_score`` is
therefore the sum of ``weighted_penalty`` over **every** soft constraint,
which is exactly the solver's soft energy. ``soft_violations`` still lists
only the soft constraints whose ``satisfied`` is false, so a residual inside
the tolerance scores a tiny penalty without being reported as a violation.
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
    # Every soft constraint contributes, not just the violated ones: the
    # compiled model pays ``weight * residual**2`` for a residual inside the
    # tolerance too (review F-05). Accumulated in constraint order so
    # :func:`validate_batch` stays bit-identical.
    soft_violation_score = 0.0
    for evaluation in evaluations:
        if evaluation.constraint_type == "soft":
            assert evaluation.weighted_penalty is not None  # set for every soft
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

    # Hard: the tolerance decides, and a satisfied constraint reports no
    # violation at all. Soft: the exact residual, because that is what the
    # compiled model charges for (review F-05); ``satisfied`` above still
    # uses the tolerance and still drives ``soft_violations``.
    weighted_penalty: float | None = None
    if constraint.type == "soft":
        violation_amount = violation
        weighted_penalty = constraint.weight * violation_amount * violation_amount
    else:
        violation_amount = 0.0 if satisfied else violation

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
    """Per-candidate verdicts of :func:`validate_batch`, aligned by row.

    The hard tallies exist so an attempt that found nothing feasible can
    still be explained without a second pass over the candidates:
    ``hard_violation_total`` is Σ ``violation_amount`` over the hard
    constraints per candidate (zero for a satisfied constraint, exactly as
    :func:`validate` reports it), and ``hard_violated_counts[i]`` is how
    many candidates violated ``hard_constraint_ids[i]``. A per-candidate,
    per-constraint matrix is deliberately not kept: on an exhaustive
    backend it would cost candidates × hard constraints in memory.
    """

    feasible: np.ndarray  # bool, shape (candidates,)
    soft_violation_score: np.ndarray  # float64, shape (candidates,)
    hard_violation_total: np.ndarray  # float64, shape (candidates,)
    hard_constraint_ids: list[str]  # hard constraints, in problem order
    hard_violated_counts: np.ndarray  # int64, shape (len(hard_constraint_ids),)


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
    hard_violation_total = np.zeros(count, dtype=np.float64)
    hard_constraint_ids: list[str] = []
    hard_violated_counts: list[int] = []

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
            # The scalar kernel's ``0.0 if satisfied else violation``: the
            # tolerance decides, so a within-band residual counts as zero
            # here too. Accumulated in constraint order like the soft score.
            hard_violation_total += np.where(satisfied, 0.0, violation)
            hard_constraint_ids.append(constraint.id)
            hard_violated_counts.append(int(np.count_nonzero(~satisfied)))
        else:
            # Exact residual, no tolerance band (review F-05) — the same
            # value, in the same association and the same constraint order,
            # as the scalar kernel above.
            violation_amount = violation
            soft_violation_score += (
                constraint.weight * violation_amount * violation_amount
            )

    return BatchValidation(
        feasible=feasible,
        soft_violation_score=soft_violation_score,
        hard_violation_total=hard_violation_total,
        hard_constraint_ids=hard_constraint_ids,
        hard_violated_counts=np.asarray(hard_violated_counts, dtype=np.int64),
    )
