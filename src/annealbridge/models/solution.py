"""Solution, validation and solve-result models."""

from typing import Literal

from pydantic import BaseModel

from annealbridge.models.metadata import SolverExecutionMetadata


class ConstraintEvaluation(BaseModel):
    """Result of evaluating one constraint against a candidate solution."""

    constraint_id: str
    constraint_type: Literal["hard", "soft"]
    satisfied: bool
    actual_value: float
    operator: str
    expected_value: float
    violation_amount: float
    weighted_penalty: float | None


class ValidationResult(BaseModel):
    """Feasibility verdict for one candidate solution."""

    feasible: bool
    evaluations: list[ConstraintEvaluation]
    hard_violations: list[ConstraintEvaluation]
    soft_violations: list[ConstraintEvaluation]
    soft_violation_score: float


class Solution(BaseModel):
    """A ranked feasible business solution."""

    rank: int
    variables: dict[str, int]
    objective_value: float
    soft_violation_score: float
    ranking_score: float
    energy: float | None
    hard_constraints_satisfied: bool
    constraint_evaluations: list[ConstraintEvaluation]


class SolveAttempt(BaseModel):
    """Statistics for one compile/solve/validate attempt."""

    attempt: int
    # None on a path whose compiler uses no hard penalty (3a spec §16.4).
    penalty: float | None
    samples_received: int
    unique_samples: int
    feasible_samples: int


class SolveError(BaseModel):
    """A structured validation or solver error (Phase 2 spec §13)."""

    code: str
    path: str | None = None
    message: str
    retryable: bool = False
    recommended_action: str | None = None


# Phase 1 name kept as an alias; SolveError is a field superset.
ProblemError = SolveError


class SolveResult(BaseModel):
    """The final outcome of solving an optimization problem."""

    status: Literal[
        "success",
        "infeasible",
        "invalid_problem",
        "solver_error",
        "backend_unavailable",
        "resource_limit_exceeded",
        "configuration_error",
    ]
    backend: str | None
    objective_direction: Literal["minimize", "maximize"] | None
    solutions: list[Solution]
    attempts: list[SolveAttempt]
    infeasibility_proven: bool = False
    errors: list[SolveError] = []
    warnings: list[SolveError] = []
    metadata: SolverExecutionMetadata | None = None
    message: str | None = None
