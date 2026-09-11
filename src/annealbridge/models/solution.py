"""Solution, validation and solve-result models.

Every field carries a ``description`` so the MCP output schema derived from
these models tells an agent what a field *means*, not just its type. The
wording follows ``docs/output-format.md``; keep the two in step.
"""

from typing import Literal

from pydantic import BaseModel, Field

from annealbridge.models.metadata import SolverExecutionMetadata


class ConstraintEvaluation(BaseModel):
    """Result of evaluating one constraint against a candidate solution."""

    constraint_id: str = Field(
        description=(
            "The constraint's id from the submitted problem, so an evaluation "
            "traces back to it."
        )
    )
    constraint_type: Literal["hard", "soft"] = Field(
        description=(
            "Echoed from the problem: a hard constraint must hold for a "
            "candidate to be feasible, a soft one is only penalized."
        )
    )
    satisfied: bool = Field(
        description=(
            "Whether this constraint holds for this assignment, judged by the "
            "validator against the original problem with the feasibility "
            "tolerance applied."
        )
    )
    actual_value: float = Field(
        description="The constraint's left-hand side evaluated at this assignment."
    )
    operator: str = Field(
        description='The constraint\'s operator: "==", "<=" or ">=".'
    )
    expected_value: float = Field(
        description="The constraint's right-hand side (the problem's rhs)."
    )
    violation_amount: float = Field(
        description=(
            "How far the constraint is from being satisfied. For a hard "
            "constraint it is 0 whenever the constraint holds. For a soft "
            "constraint it is always the exact residual, so it can be a tiny "
            "non-zero value while satisfied is still true — that is what the "
            "solver was charged for."
        )
    )
    weighted_penalty: float | None = Field(
        description=(
            "weight × violation_amount² for a soft constraint; null for a "
            "hard one, which is not weighted."
        )
    )


class ValidationResult(BaseModel):
    """Feasibility verdict for one candidate solution."""

    feasible: bool = Field(
        description=(
            "Whether every hard constraint holds, recomputed from the "
            "original problem. Never taken from solver energy or from a "
            "sampler's own feasibility flag."
        )
    )
    evaluations: list[ConstraintEvaluation] = Field(
        description="One entry per constraint, hard and soft, in problem order."
    )
    hard_violations: list[ConstraintEvaluation] = Field(
        description=(
            "The unsatisfied hard constraints among the evaluations. Empty "
            "exactly when feasible is true."
        )
    )
    soft_violations: list[ConstraintEvaluation] = Field(
        description=(
            "The unsatisfied soft constraints among the evaluations. They "
            "never affect feasibility, only the score."
        )
    )
    soft_violation_score: float = Field(
        description=(
            "Σ weight × violation² over all soft constraints, computed from "
            "the exact residual."
        )
    )


class Solution(BaseModel):
    """A ranked feasible business solution."""

    rank: int = Field(
        description=(
            "1-based rank among the returned solutions; 1 is the best "
            "ranking_score."
        )
    )
    variables: dict[str, int] = Field(
        description=(
            "The business variables only. Slack and integer-encoding bits are "
            "stripped, and integer variables are decoded to plain int values "
            "inside their declared bounds."
        )
    )
    objective_value: float = Field(
        description=(
            "The objective recomputed from the original problem, including "
            "its constant. Never derived from the compiled model."
        )
    )
    soft_violation_score: float = Field(
        description=(
            "Σ weight × violation² over all soft constraints, recomputed by "
            "the validator from the exact residual. The feasibility tolerance "
            "is deliberately not applied here, so this equals the soft energy "
            "the solver minimized."
        )
    )
    ranking_score: float = Field(
        description=(
            "objective_value + soft_violation_score when minimizing, "
            "objective_value − soft_violation_score when maximizing. The sort "
            "key behind rank."
        )
    )
    energy: float | None = Field(
        description=(
            "The compiled model's energy, which includes hard penalties, "
            "slack terms and encoding bits. For debugging only: it never "
            "feeds feasibility, the objective or the ranking. Null when the "
            "backend reported none."
        )
    )
    sample_count: int = Field(
        description=(
            "How many rows of this attempt's raw solver output carried this "
            "business assignment, before deduplication. Not a confidence "
            "measure: on an exhaustive backend every business assignment is "
            "enumerated once per combination of the slack and "
            "integer-encoding bits, so the count only reflects how many "
            "internal variables the compiled model happened to have."
        )
    )
    hard_constraints_satisfied: bool = Field(
        description=(
            "Always true for a returned solution — only feasible candidates "
            "are ranked."
        )
    )
    constraint_evaluations: list[ConstraintEvaluation] = Field(
        description="One entry per constraint, hard and soft, in problem order."
    )


class ClosestCandidate(BaseModel):
    """The deduplicated candidate that came nearest to feasibility.

    The one infeasible assignment a result ever exposes: the candidate
    with the smallest total hard-constraint violation, re-evaluated by the
    validator so its evaluations use the original problem's arithmetic.
    """

    variables: dict[str, int] = Field(
        description=(
            "The business variables only, like a ranked solution's: slack and "
            "integer-encoding bits are stripped and integers are decoded. A "
            "diagnostic aid, not an answer — it breaks at least one hard "
            "constraint."
        )
    )
    hard_violation_total: float = Field(
        description=(
            "Σ violation_amount over the hard constraints, in the "
            "constraints' own units. Strictly positive, or the candidate "
            "would be feasible; a hard constraint that holds within the "
            "feasibility tolerance contributes 0. It ranks candidates rather "
            "than measuring them, since different units are simply added."
        )
    )
    constraint_evaluations: list[ConstraintEvaluation] = Field(
        description=(
            "One entry per constraint, hard and soft, exactly as for a ranked "
            "solution. The hard entries with satisfied false are the binding "
            "requirements."
        )
    )


class HardViolationRate(BaseModel):
    """How often one hard constraint failed among an attempt's candidates."""

    constraint_id: str = Field(
        description="The constraint's id from the submitted problem."
    )
    violated_candidates: int = Field(
        description=(
            "How many of the attempt's candidates this constraint rejected, "
            "judged with the same feasibility tolerance as everywhere else."
        )
    )
    candidates: int = Field(
        description=(
            "The attempt's deduplicated candidate count — the same value for "
            "every entry, and equal to attempts[-1].unique_samples."
        )
    )
    violated_fraction: float = Field(
        description=(
            "violated_candidates / candidates, between 0 and 1. The rates are "
            "independent per-constraint counts, so they do not sum to 1 and a "
            "rate below 1 does not imply a feasible candidate exists."
        )
    )


class InfeasibilityDiagnostics(BaseModel):
    """Why the last attempt found nothing feasible (hard constraints only).

    Built from the batch re-validation of the last attempt's deduplicated
    candidates, so it is the validator's view, never the solver's.
    """

    closest_candidate: ClosestCandidate = Field(
        description=(
            "The candidate that came nearest to feasibility, chosen by the "
            "smallest total hard violation and, on a tie, by the one the "
            "solver listed first."
        )
    )
    hard_violation_rates: list[HardViolationRate] = Field(
        description=(
            "One entry per hard constraint, in the problem's constraint "
            "order. Soft constraints never appear: they cannot make a "
            "candidate infeasible."
        )
    )


class SolveAttempt(BaseModel):
    """Statistics for one compile/solve/validate attempt."""

    attempt: int = Field(
        description=(
            "1-based attempt number. An attempt is recorded even when it "
            "produced nothing feasible, so the retry ladder is visible: "
            "attempt 2 carries double attempt 1's penalty."
        )
    )
    # None on a path whose compiler uses no hard penalty (3a spec §16.4).
    penalty: float | None = Field(
        description=(
            "The hard-constraint penalty λ used for this attempt. Null on a "
            "path whose compiler uses no hard penalty (the CQM path)."
        )
    )
    samples_received: int = Field(
        description="Rows the backend returned, before decoding and deduplication."
    )
    unique_samples: int = Field(
        description=(
            "Distinct business assignments among the returned rows, after "
            "decoding and deduplication — the candidate count everything "
            "downstream works on."
        )
    )
    feasible_samples: int = Field(
        description=(
            "How many of those deduplicated candidates satisfied every hard "
            "constraint under independent re-validation. Fewer entries in "
            "solutions than this means the list was truncated to "
            "solver.top_k."
        )
    )
    compiled_variables: int | None = Field(
        default=None,
        description=(
            "The compiled model's actual variable count — business variables "
            "plus slack and integer-encoding bits — as opposed to the "
            "pre-compile estimate validate reports."
        ),
    )
    compiled_interactions: int | None = Field(
        default=None,
        description=(
            "The compiled model's quadratic terms: the BQM's interactions, "
            "or on the CQM path the objective's plus every constraint's."
        ),
    )
    compile_ms: float | None = Field(
        default=None,
        description=(
            "Wall-clock milliseconds the compile stage took, measured by the "
            "service. Unrelated to the vendor-reported metadata.timing_us, "
            "and different on every run."
        ),
    )
    solve_ms: float | None = Field(
        default=None,
        description=(
            "Wall-clock milliseconds the backend call took, network "
            "round-trips included on a remote backend. Measured by the "
            "service, unrelated to metadata.timing_us, and different on every "
            "run."
        ),
    )
    validate_ms: float | None = Field(
        default=None,
        description=(
            "Wall-clock milliseconds for decoding, deduplication, "
            "re-validation and ranking of the returned samples. Measured by "
            "the service, unrelated to metadata.timing_us, and different on "
            "every run."
        ),
    )


class SolveError(BaseModel):
    """A structured validation or solver error (Phase 2 spec §13)."""

    code: str = Field(
        description=(
            "A stable code from the error catalog, the same vocabulary for "
            "errors and warnings."
        )
    )
    path: str | None = Field(
        default=None,
        description=(
            "JSON path into the submitted problem, e.g. "
            "constraints[0].weight or solver.num_reads; null when the entry "
            "is not about one input field."
        ),
    )
    message: str = Field(
        description="What went wrong, with the concrete values involved."
    )
    retryable: bool = Field(
        default=False,
        description=(
            "Whether the same request may succeed later without any change. "
            "Always false for a warning."
        ),
    )
    recommended_action: str | None = Field(
        default=None,
        description=(
            "Fixed categorical guidance for this code — stable wording, never "
            "containing configuration values. Null when the catalog has none."
        ),
    )


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

    status: SolveStatus = Field(
        description=(
            "The single verdict for the request. success: at least one "
            "feasible solution was found and ranked. infeasible: the pipeline "
            "ran but no candidate satisfied every hard constraint under "
            "re-validation — check infeasibility_proven before concluding "
            "none exists. invalid_problem: the problem failed validation and "
            "no backend was invoked. solver_error: the backend failed while "
            "executing, or a remote vendor reported an error. "
            "backend_unavailable: the requested backend is not registered, "
            "not installed, disabled by policy or missing credentials — there "
            "is never a silent fallback. resource_limit_exceeded: a request "
            "parameter or the compiled size exceeded a server-side ceiling; "
            "values are refused, never clamped. configuration_error: the "
            "server's own configuration is wrong, not the problem."
        )
    )
    backend: str | None = Field(
        description=(
            "Registry name of the backend that ran, or null when the request "
            "failed before a backend was chosen."
        )
    )
    objective_direction: Literal["minimize", "maximize"] | None = Field(
        description=(
            "Echoed from the problem, so a consumer can interpret "
            "objective_value without re-reading the input. Null when the "
            "request failed before the problem was read."
        )
    )
    solutions: list[Solution] = Field(
        description=(
            "Ranked feasible solutions, best first, at most solver.top_k. "
            "Empty unless status is success."
        )
    )
    attempts: list[SolveAttempt] = Field(
        description="One entry per compile/solve/validate attempt, in order."
    )
    infeasibility_proven: bool = Field(
        default=False,
        description=(
            "True only when an exhaustive backend actually enumerated every "
            "assignment and found none feasible. On a heuristic or remote "
            "backend an infeasible result only means 'not found under this "
            "configuration'."
        ),
    )
    infeasibility: InfeasibilityDiagnostics | None = Field(
        default=None,
        description=(
            "Why the last attempt found nothing feasible. Present only when "
            "status is infeasible and that attempt had candidates to "
            "diagnose; null on every other status, and on an attempt that "
            "received no samples at all."
        ),
    )
    optimality_proven: bool = Field(
        default=False,
        description=(
            "True only on success when an exhaustive backend enumerated every "
            "assignment: rank 1 is then the global optimum of ranking_score, "
            "not merely the best candidate seen. Always false on a heuristic "
            "or remote backend."
        ),
    )
    errors: list[SolveError] = Field(
        default=[], description="Structured failures. Empty on success."
    )
    warnings: list[SolveError] = Field(
        default=[],
        description=(
            "Non-blocking advice, same structure as an error: the warnings "
            "validate gives for this backend, then any raised during the run. "
            "Present whatever the status, except invalid_problem."
        ),
    )
    metadata: SolverExecutionMetadata | None = Field(
        default=None,
        description=(
            "Sanitized execution facts about the last completed attempt, "
            "local backends included. Null when the request failed before any "
            "solve finished; the service never invents it."
        ),
    )
    message: str | None = Field(
        default=None,
        description=(
            "Human-readable summary, mainly used to explain an infeasible "
            "result. Null when there is nothing to add."
        ),
    )
    elapsed_ms: float | None = Field(
        default=None,
        description=(
            "Wall-clock milliseconds measured by the service from entering "
            "solve to returning, problem validation and any wait for a "
            "concurrency slot included. Unrelated to metadata.timing_us, "
            "which is what a vendor reports about its own side, and different "
            "on every run."
        ),
    )
    annealbridge_version: str | None = Field(
        default=None,
        description=(
            'The installed package version that produced this result; '
            '"unknown" outside an installed distribution.'
        ),
    )
