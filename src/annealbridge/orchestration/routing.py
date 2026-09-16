"""Solver routing: rank the registry's backends for a problem (3a spec §23).

**Advisory only, never a substitute.** ``solve`` uses the backend the user
named and nothing here writes to ``problem.solver.backend`` (overview
principle 5). The ranking is deterministic and free: it reads each
backend's *declaration* and the policy, runs the validator's advisory
layer, and never solves, compiles, builds a sampler, touches the network,
calls ``resolve_time_limit()`` or takes a concurrency slot.

This module knows no concrete backend or compiler (spec §4, §13.3): it
dispatches on ``SolverCapabilities`` flags and on the ``ModelCompiler`` /
``SolverBackend`` protocols only.
"""

from annealbridge.compiler.base import ModelCompiler
from annealbridge.models import (
    ModelType,
    OptimizationProblem,
    SolveError,
    SolverCapabilities,
)
from annealbridge.orchestration.limits import (
    exact_variable_limit_error,
    gate_errors,
    no_compiler_error,
    preference_limit_errors,
    select_model_type,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers.registry import SolverRegistry
from annealbridge.validation import (
    BackendRecommendation,
    BackendRecommendationResult,
    validate_problem,
    validate_problem_full,
)

# Reason codes are a fixed vocabulary (spec §23.4); they are not error codes
# and live outside the error catalog. A test checks every code emitted by
# ``recommend`` has an entry here.
REASON_DESCRIPTIONS: dict[str, str] = {
    "R_UNUSABLE": "A solve on this backend would fail right now; see `blocking`.",
    "R_EXACT_FITS": (
        "The estimated compiled variables fit the exhaustive backend limit, so "
        "it can prove optimality and infeasibility."
    ),
    "R_EXACT_NEAR_LIMIT": (
        "Within the exhaustive backend limit but close to it; expect noticeable "
        "run time and memory."
    ),
    "R_EXACT_OVER_LIMIT": (
        "The estimated compiled variables exceed the exhaustive backend limit; "
        "a solve would be refused."
    ),
    "R_LOCAL_HEURISTIC": (
        "Local heuristic: free and retryable, but optimality is not guaranteed."
    ),
    "R_NATIVE_CONSTRAINTS": (
        "Hard constraints are expressed natively in the model, with no penalty "
        "or slack variables."
    ),
    "R_REMOTE": "Remote backend; consumes quota.",
    "R_SINGLE_SAMPLE": (
        "Usually returns a single sample, so the effective top_k is at most 1."
    ),
    "R_DENSE_FOR_QPU": (
        "The problem is likely too dense or too large for minor-embedding."
    ),
    # 3b spec §17: integer variables.
    "R_INTEGER_NATIVE": (
        "Integer variables are passed to the model natively, with no binary "
        "encoding."
    ),
    "R_INTEGER_ENCODED": (
        "Integer variables are binary-encoded; the compiled size grows with "
        "the range."
    ),
    "R_INTEGER_BLOWUP": (
        "Binary-encoding the integer variables yields many quadratic "
        "interactions on this backend (INTEGER_QUADRATIC_BLOWUP); a backend "
        "that takes integers natively ranks ahead of it."
    ),
}


def _has_hard_constraint(problem: OptimizationProblem) -> bool:
    return any(constraint.type == "hard" for constraint in problem.constraints)


def _has_integer_variable(problem: OptimizationProblem) -> bool:
    return any(variable.type == "integer" for variable in problem.variables)


def _tier(
    caps: SolverCapabilities,
    model_type: ModelType | None,
    warning_codes: set[str],
    problem: OptimizationProblem,
    reasons: list[str],
) -> int:
    """Spec §23.3 step 3.3: the capability tier, appending its reason code(s)."""
    if caps.exhaustive:
        if "EXACT_OVER_LIMIT" in warning_codes:
            reasons.append("R_EXACT_OVER_LIMIT")
            return 1
        if "EXACT_NEAR_LIMIT" in warning_codes:
            reasons.append("R_EXACT_NEAR_LIMIT")
            return 1
        reasons.append("R_EXACT_FITS")
        return 0
    if not caps.remote:
        reasons.append("R_LOCAL_HEURISTIC")
        return 2
    if model_type == "cqm" and _has_hard_constraint(problem):
        reasons.append("R_NATIVE_CONSTRAINTS")
        return 3
    reasons.append("R_REMOTE")
    if not caps.returns_multiple_samples and problem.solver.top_k > 1:
        reasons.append("R_SINGLE_SAMPLE")
    return 4


def _assess(
    problem: OptimizationProblem,
    backend_name: str,
    registry: SolverRegistry,
    policy: ExecutionPolicy,
    compilers: dict[ModelType, ModelCompiler],
) -> tuple[tuple[int, int, int], BackendRecommendation]:
    """Spec §23.3 step 2 for one backend.

    Returns its sort key (registry order excluded) and its still-unranked
    recommendation entry.
    """
    backend = registry.get(backend_name)
    caps = backend.capabilities

    # Same gates and the same limit check as solve (§16.2 steps 3–5, 8),
    # applied to *this* candidate: the user's preferences may exceed one
    # backend's declared limit and not another's. gate_errors calls
    # is_available() lazily and that check reads local config only.
    blocking: list[SolveError] = []
    gate = gate_errors(backend_name, backend, policy)
    if gate is not None:
        blocking.extend(gate[2])
    blocking.extend(preference_limit_errors(caps, problem.solver, policy))

    # Same rule as solve/validate (§16.1), so the estimate describes the
    # path the problem would actually take on this backend.
    model_type = select_model_type(caps, compilers)
    if model_type is None:
        blocking.append(no_compiler_error(backend_name, caps, path="solver.backend"))

    variable_limit = int(policy.required_limit("variables"))
    validation = validate_problem_full(
        problem,
        capabilities=caps,
        max_compiled_variables=variable_limit,
        model_type=model_type,
    )
    # recommend() has already refused every backend-independent error, so
    # an error left here is this backend's own (a seed outside the range it
    # declares). solve would refuse the problem with it, so it blocks.
    blocking.extend(validation.errors)
    warning_codes = {warning.code for warning in validation.warnings}
    if caps.exhaustive and "EXACT_OVER_LIMIT" in warning_codes:
        # What is only advice for validate() is a refusal for solve
        # (§16.2 step 9), so it blocks here.
        blocking.append(
            exact_variable_limit_error(
                validation.estimated_compiled_variables,
                variable_limit,
                path="solver.backend",
            )
        )
    usable = not blocking

    reasons: list[str] = []
    if not usable:
        reasons.append("R_UNUSABLE")
    dense = "DENSE_FOR_QPU" in warning_codes
    if dense:
        reasons.append("R_DENSE_FOR_QPU")
    # 3b spec §17: a BQM path whose integer encoding blows up the quadratic
    # interactions sorts with DENSE_FOR_QPU, behind the backends that do not.
    blowup = "INTEGER_QUADRATIC_BLOWUP" in warning_codes
    if blowup:
        reasons.append("R_INTEGER_BLOWUP")
    tier = _tier(caps, model_type, warning_codes, problem, reasons)
    # Informational only (no tier change, §17): how this path takes integers.
    if _has_integer_variable(problem):
        if model_type == "cqm":
            reasons.append("R_INTEGER_NATIVE")
        elif model_type == "bqm":
            reasons.append("R_INTEGER_ENCODED")

    key = (0 if usable else 1, 1 if dense or blowup else 0, tier)
    entry = BackendRecommendation(
        rank=0,  # assigned after sorting
        backend=backend_name,
        usable=usable,
        model_type=model_type,
        reasons=reasons,
        blocking=blocking,
        warnings=validation.warnings,
        estimated_compiled_variables=validation.estimated_compiled_variables,
    )
    return key, entry


def recommend(
    problem: OptimizationProblem,
    registry: SolverRegistry,
    policy: ExecutionPolicy,
    compilers: dict[ModelType, ModelCompiler],
) -> BackendRecommendationResult:
    """Rank every registered backend for ``problem`` (spec §23.3).

    Deterministic: the same problem, registry, policy and compilers give
    the same result. Sort key, ascending: usable first, then neither
    DENSE_FOR_QPU nor INTEGER_QUADRATIC_BLOWUP, then capability tier
    (exhaustive that fits → local heuristic → remote with native constraints
    → other remote), then registry order. ``rank`` counts from 1. Integer
    variables add a reason (native or binary-encoded) but no tier (3b §17).
    """
    errors = validate_problem(problem)
    if errors:
        return BackendRecommendationResult(valid=False, errors=errors)

    assessed = [
        (_assess(problem, name, registry, policy, compilers), index)
        for index, name in enumerate(registry.names())
    ]
    assessed.sort(key=lambda item: (*item[0][0], item[1]))
    recommendations = [
        entry.model_copy(update={"rank": rank})
        for rank, ((_key, entry), _index) in enumerate(assessed, start=1)
    ]
    return BackendRecommendationResult(valid=True, recommendations=recommendations)
