"""Pre-compilation validation of an OptimizationProblem (spec §12, Phase 2 §20).

``validate_problem`` collects *all* errors in a single pass and returns them
as a list of ``SolveError`` (each carrying the catalog's fixed
``recommended_action``); it never raises and never stops at the first
problem. Duplicate objective terms are legal (the compiler sums coefficients)
and only produce a DUPLICATE_TERM_MERGED warning plus one log line (spec §8).

``validate_problem_full`` wraps the same error pass and adds the Phase 2 §20
advisory layer: non-blocking warnings, ``estimated_compiled_variables`` and
``objective_scale``, so an agent can fix a problem or switch backend before
spending quota. Warnings never affect ``valid``.
"""

import logging
import math
from collections import Counter

from pydantic import BaseModel

from annealbridge.models import (
    Constraint,
    Objective,
    OptimizationProblem,
    SolveError,
    SolverPreferences,
    catalog_error,
)
from annealbridge.validation.estimates import (
    accumulate_terms,
    analyze_inequality,
    compute_objective_scale,
    count_slack_bits,
    estimate_compiled_variables,
    lhs_bounds,
)

logger = logging.getLogger(__name__)

# Phase 2 §20 warning thresholds (heuristics, tunable as module constants).
SOFT_WEIGHT_SMALL_RATIO = 0.01
LARGE_SLACK_BITS_THRESHOLD = 10
EXACT_NEAR_LIMIT_RATIO = 0.8
QPU_DENSE_VARIABLE_THRESHOLD = 150
QPU_DENSE_CONSTRAINT_VARIABLE_THRESHOLD = 30

# Backends whose parameters the validator warns about (§15.2, §16). The spec's
# §20 table conditions on backend *names*; SolverCapabilities lives in the
# solvers layer, which validation must not import (§4).
SEED_IGNORING_BACKENDS = frozenset({"dwave_qpu", "leap_hybrid_bqm"})
HYBRID_IGNORED_PARAMETER_FIELDS = ("num_reads", "num_sweeps")

# Defaults are read off the model so they can never drift from the schema.
_DEFAULT_SOLVER_PREFERENCES = SolverPreferences()

# Every error code this validator can emit (Phase 1 spec §12 plus
# NO_VARIABLES). Each must have a fixed recommended_action in the error
# catalog; tests/unit/test_error_catalog.py enforces the coverage.
VALIDATOR_ERROR_CODES: frozenset[str] = frozenset(
    {
        "DUPLICATE_CONSTRAINT_ID",
        "DUPLICATE_VARIABLE",
        "EMPTY_CONSTRAINT",
        "HARD_CONSTRAINT_HAS_WEIGHT",
        "INVALID_SOLVER_PREFERENCE",
        "NON_FINITE_COEFFICIENT",
        "NON_INTEGER_INEQUALITY",
        "NO_VARIABLES",
        "RESERVED_VARIABLE_NAME",
        "SELF_QUADRATIC_TERM",
        "SOFT_CONSTRAINT_MISSING_WEIGHT",
        "TRIVIALLY_INFEASIBLE",
        "UNKNOWN_VARIABLE",
    }
)

# Backend-specific option fields that must be finite and strictly positive
# when given: (SolverPreferences attribute, option field).
_POSITIVE_BACKEND_OPTION_FIELDS = (
    ("dwave_qpu", "annealing_time_us"),
    ("dwave_qpu", "chain_strength"),
    ("leap_hybrid_bqm", "time_limit_seconds"),
)

# Fixed categorical guidance per warning code (mirrors the error catalog's
# style; §13.2's RECOMMENDED_ACTIONS stays an error-code vocabulary).
_WARNING_RECOMMENDED_ACTIONS: dict[str, str] = {
    "SOFT_WEIGHT_SMALL": (
        "The soft constraint's weight is tiny compared to the objective's "
        "scale, so it will barely influence solutions; raise the weight if "
        "the preference matters."
    ),
    "LARGE_SLACK_RANGE": (
        "This inequality needs many slack bits, which enlarges the compiled "
        "model; tighten the bound or split the constraint if possible."
    ),
    "EXACT_NEAR_LIMIT": (
        "The estimated compiled size is close to the exact solver's variable "
        "limit; consider simulated_annealing before the problem grows."
    ),
    "EXACT_OVER_LIMIT": (
        "The estimated compiled size exceeds the exact solver's variable "
        "limit; solving will fail with resource_limit_exceeded — reduce the "
        "problem or use simulated_annealing."
    ),
    "DENSE_FOR_QPU": (
        "The problem is likely too dense or too large to embed well on the "
        "QPU; consider leap_hybrid_bqm or simulated_annealing."
    ),
    "SEED_IGNORED": (
        "This backend does not support seeding; remove solver.seed or use a "
        "local backend if reproducibility is required."
    ),
    "PARAMETER_IGNORED": (
        "This parameter has no effect on the selected backend and will be "
        "ignored; remove it to avoid confusion."
    ),
    "DUPLICATE_TERM_MERGED": (
        "Duplicate objective terms are summed by the compiler; merge them in "
        "the input if the duplication is unintentional."
    ),
    "REDUNDANT_CONSTRAINT": (
        "This inequality is always satisfied and adds nothing to the model; "
        "remove it, or fix its bound if it was meant to restrict solutions."
    ),
}


class ProblemValidationResult(BaseModel):
    """Full validation outcome (Phase 2 spec §20).

    ``valid`` is decided by ``errors`` alone; warnings are advisory and never
    block. Estimates are only computed for a problem with no errors — that
    gate is what guarantees the slack arithmetic is safe (finite, integral
    inequalities, unique names) and the estimate matches compilation exactly.
    """

    valid: bool
    errors: list[SolveError] = []
    warnings: list[SolveError] = []
    estimated_compiled_variables: int | None = None  # 原始變數 + slack bits，純算術
    objective_scale: float | None = None


def validate_problem(problem: OptimizationProblem) -> list[SolveError]:
    """Validate a problem before compilation, returning all errors found.

    An empty list means the problem is safe to hand to the compiler.
    """
    errors, _ = _collect(problem)
    return errors


def _collect(
    problem: OptimizationProblem,
) -> tuple[list[SolveError], list[SolveError]]:
    """Single error pass plus the spec §8 duplicate-term warnings.

    Both public entry points go through here so a duplicate term is logged
    exactly once, whichever of them is called.
    """
    errors: list[SolveError] = []
    known_variables = {variable.name for variable in problem.variables}

    _check_variables(problem, errors)
    _check_constraint_ids(problem, errors)
    _check_objective(problem, known_variables, errors)
    for index, constraint in enumerate(problem.constraints):
        _check_constraint(constraint, index, known_variables, errors)
    _check_solver_preferences(problem, errors)

    duplicate_warnings: list[SolveError] = []
    _warn_duplicate_terms(problem.objective, duplicate_warnings)
    return errors, duplicate_warnings


def validate_problem_full(
    problem: OptimizationProblem,
    *,
    exact_max_variables: int | None = None,
) -> ProblemValidationResult:
    """Validate a problem and add the §20 advisory layer.

    Reuses :func:`validate_problem` unchanged for errors. Warnings and
    estimates are only produced when there are no errors: an erroneous
    problem must be fixed first anyway, and the no-error gate is exactly
    what makes the pure slack arithmetic well-defined.

    ``exact_max_variables`` is the caller-supplied policy limit for the
    exact backend (validation must not read policy itself, §4); when it is
    ``None`` the EXACT_NEAR_LIMIT / EXACT_OVER_LIMIT checks are skipped.
    """
    errors, duplicate_warnings = _collect(problem)
    if errors:
        return ProblemValidationResult(valid=False, errors=errors)

    objective_scale = compute_objective_scale(problem.objective)
    estimated = estimate_compiled_variables(problem)

    warnings: list[SolveError] = []
    _warn_soft_weights(problem, objective_scale, warnings)
    _warn_inequalities(problem, warnings)
    _warn_backend_fit(problem, estimated, exact_max_variables, warnings)
    _warn_ignored_parameters(problem, warnings)
    warnings.extend(duplicate_warnings)

    return ProblemValidationResult(
        valid=True,
        errors=[],
        warnings=warnings,
        estimated_compiled_variables=estimated,
        objective_scale=objective_scale,
    )


def _warning(code: str, path: str | None, message: str) -> SolveError:
    return SolveError(
        code=code,
        path=path,
        message=message,
        retryable=False,
        recommended_action=_WARNING_RECOMMENDED_ACTIONS[code],
    )


def _error(*, code: str, path: str | None, message: str) -> SolveError:
    """Build a validation error with the catalog's fixed recommended_action."""
    return catalog_error(code, message, path=path)


def _warn_soft_weights(
    problem: OptimizationProblem, objective_scale: float, warnings: list[SolveError]
) -> None:
    threshold = objective_scale * SOFT_WEIGHT_SMALL_RATIO
    for index, constraint in enumerate(problem.constraints):
        if constraint.type != "soft" or constraint.weight is None:
            continue
        if constraint.weight < threshold:
            warnings.append(
                _warning(
                    "SOFT_WEIGHT_SMALL",
                    f"constraints[{index}].weight",
                    (
                        f"Soft constraint {constraint.id} has weight "
                        f"{constraint.weight}, below {SOFT_WEIGHT_SMALL_RATIO} "
                        f"of the objective scale {objective_scale}; it will "
                        "barely influence solutions"
                    ),
                )
            )


def _warn_inequalities(
    problem: OptimizationProblem, warnings: list[SolveError]
) -> None:
    for index, constraint in enumerate(problem.constraints):
        if constraint.operator not in ("<=", ">="):
            continue
        analysis = analyze_inequality(constraint)
        if analysis.redundant:
            warnings.append(
                _warning(
                    "REDUNDANT_CONSTRAINT",
                    f"constraints[{index}]",
                    (
                        f"Inequality constraint {constraint.id} is always "
                        f"satisfied: lhs maximum {analysis.lhs_max} never "
                        f"exceeds the bound"
                    ),
                )
            )
            continue
        slack_bits = count_slack_bits(constraint)
        if slack_bits > LARGE_SLACK_BITS_THRESHOLD:
            warnings.append(
                _warning(
                    "LARGE_SLACK_RANGE",
                    f"constraints[{index}]",
                    (
                        f"Inequality constraint {constraint.id} needs "
                        f"{slack_bits} slack bits (more than "
                        f"{LARGE_SLACK_BITS_THRESHOLD}); the compiled model "
                        "grows accordingly"
                    ),
                )
            )


def _warn_backend_fit(
    problem: OptimizationProblem,
    estimated: int,
    exact_max_variables: int | None,
    warnings: list[SolveError],
) -> None:
    backend = problem.solver.backend

    if backend == "exact" and exact_max_variables is not None:
        if estimated > exact_max_variables:
            warnings.append(
                _warning(
                    "EXACT_OVER_LIMIT",
                    "solver.backend",
                    (
                        f"Estimated compiled variables ({estimated}) exceed "
                        f"the exact solver limit ({exact_max_variables}); "
                        "solving will fail with resource_limit_exceeded"
                    ),
                )
            )
        elif estimated > exact_max_variables * EXACT_NEAR_LIMIT_RATIO:
            warnings.append(
                _warning(
                    "EXACT_NEAR_LIMIT",
                    "solver.backend",
                    (
                        f"Estimated compiled variables ({estimated}) are above "
                        f"{EXACT_NEAR_LIMIT_RATIO:.0%} of the exact solver "
                        f"limit ({exact_max_variables})"
                    ),
                )
            )

    if backend == "dwave_qpu":
        max_constraint_variables = max(
            (
                len({term.variable for term in constraint.terms})
                for constraint in problem.constraints
            ),
            default=0,
        )
        reasons: list[str] = []
        if estimated > QPU_DENSE_VARIABLE_THRESHOLD:
            reasons.append(
                f"estimated compiled variables ({estimated}) exceed "
                f"{QPU_DENSE_VARIABLE_THRESHOLD}"
            )
        if max_constraint_variables > QPU_DENSE_CONSTRAINT_VARIABLE_THRESHOLD:
            reasons.append(
                f"largest constraint involves {max_constraint_variables} "
                f"variables (more than "
                f"{QPU_DENSE_CONSTRAINT_VARIABLE_THRESHOLD})"
            )
        if reasons:
            warnings.append(
                _warning(
                    "DENSE_FOR_QPU",
                    "solver.backend",
                    (
                        "Problem is likely too dense for QPU embedding: "
                        + "; ".join(reasons)
                    ),
                )
            )


def _warn_ignored_parameters(
    problem: OptimizationProblem, warnings: list[SolveError]
) -> None:
    solver = problem.solver

    # seed gets its dedicated code on every backend that ignores it (§15);
    # PARAMETER_IGNORED below deliberately skips it to avoid double-warning.
    if solver.seed is not None and solver.backend in SEED_IGNORING_BACKENDS:
        warnings.append(
            _warning(
                "SEED_IGNORED",
                "solver.seed",
                (
                    f"Backend {solver.backend} does not support seeding; "
                    "solver.seed will be ignored"
                ),
            )
        )

    if solver.backend == "leap_hybrid_bqm":
        for field in HYBRID_IGNORED_PARAMETER_FIELDS:
            value = getattr(solver, field)
            if value != getattr(_DEFAULT_SOLVER_PREFERENCES, field):
                warnings.append(
                    _warning(
                        "PARAMETER_IGNORED",
                        f"solver.{field}",
                        (
                            f"solver.{field}={value} has no effect on the "
                            "leap_hybrid_bqm backend and will be ignored"
                        ),
                    )
                )


def _warn_duplicate_terms(objective: Objective, warnings: list[SolveError]) -> None:
    """Spec §8: duplicate objective terms are merged, not rejected.

    The structured DUPLICATE_TERM_MERGED warning is the primary signal; each
    one is also logged once so the merge stays visible on the errors-only
    path the service uses.
    """
    first_new = len(warnings)
    for variable, count in _duplicate_linear_variables(objective):
        warnings.append(
            _warning(
                "DUPLICATE_TERM_MERGED",
                "objective.linear_terms",
                (
                    f"Variable {variable} appears {count} times in "
                    "objective.linear_terms; coefficients will be summed by "
                    "the compiler"
                ),
            )
        )
    for pair, count in _duplicate_quadratic_pairs(objective):
        warnings.append(
            _warning(
                "DUPLICATE_TERM_MERGED",
                "objective.quadratic_terms",
                (
                    f"Variable pair {pair} appears {count} times in "
                    "objective.quadratic_terms; coefficients will be summed "
                    "by the compiler"
                ),
            )
        )
    for warning in warnings[first_new:]:
        logger.warning("%s", warning.message)


def _check_variables(problem: OptimizationProblem, errors: list[SolveError]) -> None:
    if not problem.variables:
        errors.append(
            _error(
                code="NO_VARIABLES",
                path="variables",
                message="Problem declares no variables; there is nothing to optimize",
            )
        )
    seen: set[str] = set()
    for index, variable in enumerate(problem.variables):
        if variable.name in seen:
            errors.append(
                _error(
                    code="DUPLICATE_VARIABLE",
                    path=f"variables[{index}]",
                    message=f"Variable {variable.name} is declared more than once",
                )
            )
        seen.add(variable.name)
        if variable.name.startswith("__"):
            errors.append(
                _error(
                    code="RESERVED_VARIABLE_NAME",
                    path=f"variables[{index}]",
                    message=(
                        f"Variable name {variable.name} is reserved: names starting "
                        "with '__' are for internal variables"
                    ),
                )
            )


def _check_constraint_ids(
    problem: OptimizationProblem, errors: list[SolveError]
) -> None:
    seen: set[str] = set()
    for index, constraint in enumerate(problem.constraints):
        if constraint.id in seen:
            errors.append(
                _error(
                    code="DUPLICATE_CONSTRAINT_ID",
                    path=f"constraints[{index}]",
                    message=f"Constraint id {constraint.id} is used more than once",
                )
            )
        seen.add(constraint.id)


def _check_objective(
    problem: OptimizationProblem,
    known_variables: set[str],
    errors: list[SolveError],
) -> None:
    objective = problem.objective

    for index, term in enumerate(objective.linear_terms):
        path = f"objective.linear_terms[{index}]"
        if term.variable not in known_variables:
            errors.append(_unknown_variable(term.variable, path))
        if not math.isfinite(term.coefficient):
            errors.append(_non_finite(f"coefficient {term.coefficient}", path))

    for index, term in enumerate(objective.quadratic_terms):
        path = f"objective.quadratic_terms[{index}]"
        for variable in (term.variable1, term.variable2):
            if variable not in known_variables:
                errors.append(_unknown_variable(variable, path))
        if term.variable1 == term.variable2:
            errors.append(
                _error(
                    code="SELF_QUADRATIC_TERM",
                    path=path,
                    message=(
                        f"Quadratic term references {term.variable1} twice; for "
                        "binary variables x*x = x, use a linear term instead"
                    ),
                )
            )
        if not math.isfinite(term.coefficient):
            errors.append(_non_finite(f"coefficient {term.coefficient}", path))

    if not math.isfinite(objective.constant):
        errors.append(
            _non_finite(f"constant {objective.constant}", "objective.constant")
        )


def _duplicate_linear_variables(objective: Objective) -> list[tuple[str, int]]:
    counts = Counter(term.variable for term in objective.linear_terms)
    return [(variable, count) for variable, count in counts.items() if count > 1]


def _duplicate_quadratic_pairs(
    objective: Objective,
) -> list[tuple[tuple[str, ...], int]]:
    counts = Counter(
        frozenset((term.variable1, term.variable2))
        for term in objective.quadratic_terms
    )
    return [
        (tuple(sorted(pair)), count) for pair, count in counts.items() if count > 1
    ]


def _check_constraint(
    constraint: Constraint,
    index: int,
    known_variables: set[str],
    errors: list[SolveError],
) -> None:
    base = f"constraints[{index}]"

    if not constraint.terms:
        errors.append(
            _error(
                code="EMPTY_CONSTRAINT",
                path=base,
                message=f"Constraint {constraint.id} has no terms",
            )
        )

    is_inequality = constraint.operator in ("<=", ">=")
    for term_index, term in enumerate(constraint.terms):
        path = f"{base}.terms[{term_index}]"
        if term.variable not in known_variables:
            errors.append(_unknown_variable(term.variable, path))
        if not math.isfinite(term.coefficient):
            errors.append(_non_finite(f"coefficient {term.coefficient}", path))
        elif is_inequality and not term.coefficient.is_integer():
            errors.append(_non_integer_inequality(constraint, term.coefficient, path))

    if not math.isfinite(constraint.rhs):
        errors.append(_non_finite(f"rhs {constraint.rhs}", f"{base}.rhs"))
    elif is_inequality and not constraint.rhs.is_integer():
        errors.append(_non_integer_inequality(constraint, constraint.rhs, f"{base}.rhs"))

    if constraint.type == "hard":
        if constraint.weight is not None:
            errors.append(
                _error(
                    code="HARD_CONSTRAINT_HAS_WEIGHT",
                    path=f"{base}.weight",
                    message=(
                        f"Hard constraint {constraint.id} must not carry a weight; "
                        "hard penalty strength is chosen by the penalty strategy"
                    ),
                )
            )
    else:
        if constraint.weight is None or constraint.weight <= 0:
            errors.append(
                _error(
                    code="SOFT_CONSTRAINT_MISSING_WEIGHT",
                    path=f"{base}.weight",
                    message=(
                        f"Soft constraint {constraint.id} requires a weight > 0, "
                        f"got {constraint.weight}"
                    ),
                )
            )
        elif not math.isfinite(constraint.weight):
            errors.append(
                _non_finite(f"weight {constraint.weight}", f"{base}.weight")
            )

    _check_trivially_infeasible(constraint, base, errors)


def _check_trivially_infeasible(
    constraint: Constraint, base: str, errors: list[SolveError]
) -> None:
    """Reject a hard constraint no binary assignment can satisfy.

    The lhs range is taken over the *accumulated* coefficients (one per
    variable), exactly as the compiler sums them, so a constraint accepted
    here can never fail slack encoding later. The message reports the
    user's own operator and rhs, never the compiler's normalized form.
    """
    if constraint.type != "hard" or not constraint.terms:
        return
    if not all(math.isfinite(term.coefficient) for term in constraint.terms):
        return
    if not math.isfinite(constraint.rhs):
        return

    lhs_min, lhs_max = lhs_bounds(accumulate_terms(constraint.terms))
    rhs = constraint.rhs

    if constraint.operator == "<=":
        infeasible = lhs_min > rhs
    elif constraint.operator == ">=":
        infeasible = lhs_max < rhs
    else:
        infeasible = rhs < lhs_min or rhs > lhs_max

    if infeasible:
        errors.append(
            _error(
                code="TRIVIALLY_INFEASIBLE",
                path=base,
                message=(
                    f"Hard constraint {constraint.id} can never be satisfied: "
                    f"lhs range (after summing repeated variables) is "
                    f"[{lhs_min}, {lhs_max}] but requires "
                    f"{constraint.operator} {rhs}"
                ),
            )
        )


def _check_solver_preferences(
    problem: OptimizationProblem, errors: list[SolveError]
) -> None:
    solver = problem.solver
    checks = [
        ("top_k", solver.top_k, solver.top_k <= 0, "must be > 0"),
        ("num_reads", solver.num_reads, solver.num_reads <= 0, "must be > 0"),
        ("num_sweeps", solver.num_sweeps, solver.num_sweeps <= 0, "must be > 0"),
        ("max_retries", solver.max_retries, solver.max_retries < 0, "must be >= 0"),
        (
            "penalty_multiplier",
            solver.penalty_multiplier,
            solver.penalty_multiplier <= 0,
            "must be > 0",
        ),
    ]
    for field, value, is_bad, rule in checks:
        if is_bad:
            errors.append(
                _error(
                    code="INVALID_SOLVER_PREFERENCE",
                    path=f"solver.{field}",
                    message=f"solver.{field} {rule}, got {value}",
                )
            )

    # Backend-specific options: the schema already rejects NaN / ±inf
    # (allow_inf_nan=False); the semantic "> 0" rule lives here. Finiteness
    # is re-checked so a model built without validation cannot slip through
    # (nan > limit is False, so a policy comparison would silently pass).
    for options_field, field in _POSITIVE_BACKEND_OPTION_FIELDS:
        options = getattr(solver, options_field)
        if options is None:
            continue
        value = getattr(options, field)
        if value is None:
            continue
        if not math.isfinite(value) or value <= 0:
            errors.append(
                _error(
                    code="INVALID_SOLVER_PREFERENCE",
                    path=f"solver.{options_field}.{field}",
                    message=(
                        f"solver.{options_field}.{field} must be a finite "
                        f"number > 0, got {value}"
                    ),
                )
            )


def _unknown_variable(variable: str, path: str) -> SolveError:
    return _error(
        code="UNKNOWN_VARIABLE",
        path=path,
        message=f"Variable {variable} does not exist",
    )


def _non_finite(what: str, path: str) -> SolveError:
    return _error(
        code="NON_FINITE_COEFFICIENT",
        path=path,
        message=f"Value is not finite: {what}",
    )


def _non_integer_inequality(
    constraint: Constraint, value: float, path: str
) -> SolveError:
    return _error(
        code="NON_INTEGER_INEQUALITY",
        path=path,
        message=(
            f"Inequality constraint {constraint.id} requires integer coefficients "
            f"and rhs for slack encoding, got {value}"
        ),
    )
