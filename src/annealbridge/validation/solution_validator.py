"""Constraint validation of candidate solutions (spec §23).

Feasibility is judged against the *original* OptimizationProblem, never
against BQM energy. The sample must already have internal (slack)
variables removed by the caller.
"""

import logging

from annealbridge.models import (
    Constraint,
    ConstraintEvaluation,
    OptimizationProblem,
    ValidationResult,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-8


def validate(problem: OptimizationProblem, sample: dict[str, int]) -> ValidationResult:
    """Evaluate every constraint of ``problem`` against a business sample.

    ``sample`` maps each business variable name to its 0/1 assignment,
    with internal variables already stripped.
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
    return ValidationResult(
        feasible=not hard_violations,
        evaluations=evaluations,
        hard_violations=hard_violations,
        soft_violations=soft_violations,
        soft_violation_score=sum(
            evaluation.weighted_penalty for evaluation in soft_violations
        ),
    )


def _evaluate(constraint: Constraint, sample: dict[str, int]) -> ConstraintEvaluation:
    actual = sum(term.coefficient * sample[term.variable] for term in constraint.terms)
    rhs = constraint.rhs

    if constraint.operator == "==":
        satisfied = abs(actual - rhs) <= EPSILON
        violation = abs(actual - rhs)
    elif constraint.operator == "<=":
        satisfied = actual <= rhs + EPSILON
        violation = max(0.0, actual - rhs)
    else:
        satisfied = actual >= rhs - EPSILON
        violation = max(0.0, rhs - actual)

    violation_amount = 0.0 if satisfied else violation

    weighted_penalty: float | None = None
    if constraint.type == "soft":
        weighted_penalty = constraint.weight * violation_amount**2

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
