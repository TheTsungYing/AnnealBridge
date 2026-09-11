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
    # How many rows of this attempt's raw solver output carried this business
    # assignment, before deduplication. Not a confidence measure: on an
    # exhaustive backend every business assignment is enumerated once per
    # combination of the slack and integer-encoding bits, so the count only
    # reflects how many internal variables the compiled model happened to
    # have.
    sample_count: int
    hard_constraints_satisfied: bool
    constraint_evaluations: list[ConstraintEvaluation]


class ClosestCandidate(BaseModel):
    """The deduplicated candidate that came nearest to feasibility.

    The one infeasible assignment a result ever exposes: the candidate
    with the smallest total hard-constraint violation, re-evaluated by the
    validator so its evaluations use the original problem's arithmetic.
    """

    variables: dict[str, int]
    # Σ violation_amount over the hard constraints, in the constraints'
    # own units; strictly positive, or the candidate would be feasible.
    hard_violation_total: float
    constraint_evaluations: list[ConstraintEvaluation]


class HardViolationRate(BaseModel):
    """How often one hard constraint failed among an attempt's candidates."""

    constraint_id: str
    violated_candidates: int
    # Deduplicated candidates of the attempt; the same for every entry.
    candidates: int
    violated_fraction: float


class InfeasibilityDiagnostics(BaseModel):
    """Why the last attempt found nothing feasible (hard constraints only).

    Built from the batch re-validation of the last attempt's deduplicated
    candidates, so it is the validator's view, never the solver's.
    """

    closest_candidate: ClosestCandidate
    # One entry per hard constraint, in problem order.
    hard_violation_rates: list[HardViolationRate]


class SolveAttempt(BaseModel):
    """Statistics for one compile/solve/validate attempt."""

    attempt: int
    # None on a path whose compiler uses no hard penalty (3a spec §16.4).
    penalty: float | None
    samples_received: int
    unique_samples: int
    feasible_samples: int
    # Actual size of the compiled model this attempt ran (not the
    # pre-compile estimate ``validate`` reports).
    compiled_variables: int | None = None
    compiled_interactions: int | None = None
    # Wall-clock milliseconds measured by the service around each stage:
    # compile, the backend call, and decode + dedup + validate + rank.
    # Independent of the vendor-reported ``metadata.timing_us``.
    compile_ms: float | None = None
    solve_ms: float | None = None
    validate_ms: float | None = None


class SolveError(BaseModel):
    """A structured validation or solver error (Phase 2 spec §13)."""

    code: str
    path: str | None = None
    message: str
    retryable: bool = False
    recommended_action: str | None = None


# Phase 1 name kept as an alias; SolveError is a field superset. Kept for
# Phase 1 compatibility only — no production code path uses this name, only
# tests reference it.
ProblemError = SolveError


# The result vocabulary (spec §27). Named so the service's failure helpers
# and the availability map can be typed against it instead of ``str``
# (2026-09-09 review F-03).
SolveStatus = Literal[
    "success",
    "infeasible",
    "invalid_problem",
    "solver_error",
    "backend_unavailable",
    "resource_limit_exceeded",
    "configuration_error",
]


class SolveResult(BaseModel):
    """The final outcome of solving an optimization problem."""

    status: SolveStatus
    backend: str | None
    objective_direction: Literal["minimize", "maximize"] | None
    solutions: list[Solution]
    attempts: list[SolveAttempt]
    infeasibility_proven: bool = False
    # Only on ``infeasible``, and only when the last attempt had candidates
    # to diagnose; None on every other status.
    infeasibility: InfeasibilityDiagnostics | None = None
    # True only when an exhaustive backend enumerated every assignment, so
    # rank 1 is the global optimum of the ranking score, not merely the
    # best candidate seen.
    optimality_proven: bool = False
    errors: list[SolveError] = []
    warnings: list[SolveError] = []
    metadata: SolverExecutionMetadata | None = None
    message: str | None = None
    # Wall-clock milliseconds from entering ``solve`` to returning, problem
    # validation and any wait for a concurrency slot included.
    elapsed_ms: float | None = None
    annealbridge_version: str | None = None
