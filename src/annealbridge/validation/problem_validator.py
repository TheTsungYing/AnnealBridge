"""Pre-compilation validation of an OptimizationProblem (spec §12, Phase 2 §20, 3a §9).

``validate_problem`` collects *all* errors in a single pass and returns them
as a list of ``SolveError`` (each carrying the catalog's fixed
``recommended_action``); it never raises and never stops at the first
problem. Duplicate objective terms are legal (the compiler sums coefficients)
and only produce a DUPLICATE_TERM_MERGED warning plus one log line (spec §8).

``validate_problem_full`` wraps the same error pass and adds the Phase 2 §20
advisory layer: non-blocking warnings, ``estimated_compiled_variables`` and
``objective_scale``, so an agent can fix a problem or switch backend before
spending quota. Warnings never affect ``valid``.

Backend-dependent advice is driven purely by the injected
``SolverCapabilities`` declaration (3a §9): this module never names a
backend, never reads policy and never imports ``solvers`` (spec §4).
"""

import logging
import math
import types
import typing
from collections import Counter

from pydantic import BaseModel

from annealbridge.models import (
    Constraint,
    ModelType,
    Objective,
    OptimizationProblem,
    SolveError,
    SolverCapabilities,
    SolverPreferences,
    Variable,
    catalog_error,
)
from annealbridge.validation.estimates import (
    Bounds,
    accumulate_terms,
    analyze_inequality,
    compute_objective_scale,
    constraint_bit_count,
    count_slack_bits,
    estimate_compiled_variables,
    estimate_cqm_variables,
    estimate_encoded_interactions,
    integer_encoding_bits,
    lhs_bounds,
    variable_bounds,
)

logger = logging.getLogger(__name__)

# Phase 2 §20 warning thresholds (heuristics, tunable as module constants).
SOFT_WEIGHT_SMALL_RATIO = 0.01
LARGE_SLACK_BITS_THRESHOLD = 10
EXACT_NEAR_LIMIT_RATIO = 0.8
QPU_DENSE_VARIABLE_THRESHOLD = 150
QPU_DENSE_CONSTRAINT_VARIABLE_THRESHOLD = 30
# 3b §9.3 integer-encoding thresholds (BQM path only).
LARGE_INTEGER_BITS_THRESHOLD = 10
INTEGER_QUADRATIC_BLOWUP_THRESHOLD = 2000
# 3b §7: integer bounds must lie within ±(2^31-1) so every encoded value,
# every product of two values and every float64 evaluation stays exact.
INTEGER_BOUND_LIMIT = 2**31 - 1

# Defaults are read off the model so they can never drift from the schema.
_DEFAULT_SOLVER_PREFERENCES = SolverPreferences()

# Every error code this validator can emit (Phase 1 spec §12 plus
# NO_VARIABLES, plus the 3b §9.1 integer rules). Each must have a fixed
# recommended_action in the error catalog; tests/unit/test_error_catalog.py
# enforces the coverage.
VALIDATOR_ERROR_CODES: frozenset[str] = frozenset(
    {
        "BOUNDS_ON_BINARY",
        "DUPLICATE_CONSTRAINT_ID",
        "DUPLICATE_VARIABLE",
        "EMPTY_CONSTRAINT",
        "HARD_CONSTRAINT_HAS_WEIGHT",
        "INTEGER_BOUNDS_INVALID",
        "INTEGER_BOUNDS_MISSING",
        "INTEGER_RANGE_TOO_LARGE",
        "INTEGER_REQUIRES_VERSION_1_1",
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


def _optional_members(annotation: object) -> list[object] | None:
    """The non-None members of an optional union annotation, else None.

    Accepts both spellings — ``types.UnionType`` (``X | None``) and
    ``typing.Union`` / ``Optional[X]``; anything that is not a union
    (a bare type, a Literal, ...) yields None.
    """
    origin = typing.get_origin(annotation)
    if origin is not types.UnionType and origin is not typing.Union:
        return None
    return [arg for arg in typing.get_args(annotation) if arg is not type(None)]


def _optional_model(annotation: object) -> type[BaseModel] | None:
    """The ``Model`` in a ``Model | None`` annotation, else None (3a §9.4).

    Only a union of exactly one BaseModel subclass with None counts as an
    option block; ``int | None`` (``seed``) and bare types do not.
    """
    members = _optional_members(annotation)
    if members is None or len(members) != 1:
        return None
    (member,) = members
    if isinstance(member, type) and issubclass(member, BaseModel):
        return member
    return None


def _optional_number(annotation: object) -> bool:
    """True for ``int | None`` / ``float | None`` (either union spelling)."""
    members = _optional_members(annotation)
    return members is not None and len(members) == 1 and members[0] in (int, float)


def _option_blocks() -> dict[str, type[BaseModel]]:
    """``SolverPreferences`` fields that are backend option blocks (3a §9.4).

    Reflected from the model so a new backend's block (with its own
    numeric options) is covered without touching this module.
    """
    blocks: dict[str, type[BaseModel]] = {}
    for name, field in SolverPreferences.model_fields.items():
        model = _optional_model(field.annotation)
        if model is not None:
            blocks[name] = model
    return blocks


# (option block name, option field) pairs whose value, when given, must be
# finite and strictly positive (3a §9.4). bool fields are deliberately
# excluded.
_POSITIVE_OPTION_FIELDS: tuple[tuple[str, str], ...] = tuple(
    (block, field)
    for block, model in _option_blocks().items()
    for field, info in model.model_fields.items()
    if _optional_number(info.annotation)
)

# Fixed categorical guidance per warning code (mirrors the error catalog's
# style; §13.2's RECOMMENDED_ACTIONS stays an error-code vocabulary). The
# text describes backends by capability, never by name (3a §13.2).
_WARNING_RECOMMENDED_ACTIONS: dict[str, str] = {
    "SOFT_WEIGHT_SMALL": (
        "The soft constraint's weight is tiny compared to the objective's "
        "scale, so it will barely influence solutions; raise the weight if "
        "the preference matters."
    ),
    "LARGE_SLACK_RANGE": (
        "This inequality needs many slack bits on a BQM backend, which "
        "enlarges the compiled model; tighten the bound or split the "
        "constraint if possible."
    ),
    "EXACT_NEAR_LIMIT": (
        "The estimated compiled size is close to the exhaustive backend's "
        "variable limit; consider a local heuristic backend before the "
        "problem grows."
    ),
    "EXACT_OVER_LIMIT": (
        "The estimated compiled size exceeds the exhaustive backend's "
        "variable limit; solving will fail with resource_limit_exceeded — "
        "reduce the problem or use a local heuristic backend."
    ),
    "DENSE_FOR_QPU": (
        "The problem is likely too dense or too large to minor-embed on a "
        "backend that requires embedding; consider a hybrid backend or a "
        "local heuristic backend."
    ),
    "SEED_IGNORED": (
        "This backend does not support seeding; remove solver.seed or use a "
        "backend that supports seeding if reproducibility is required."
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
    # 3b §9.3 / §19 (the bit threshold is LARGE_INTEGER_BITS_THRESHOLD and
    # is reported in the message, never in this fixed guidance).
    "LARGE_INTEGER_RANGE": (
        "An integer variable needs more encoding bits on a BQM backend than "
        "is comfortable; tighten its bounds, rescale its unit, or use a "
        "backend that accepts integer variables natively."
    ),
    "INTEGER_QUADRATIC_BLOWUP": (
        "Binary-encoding the integer variables produces many quadratic "
        "interactions on a BQM backend; prefer a backend that accepts "
        "integer variables natively (model type cqm), or reduce the ranges."
    ),
}


class ProblemValidationResult(BaseModel):
    """Full validation outcome (Phase 2 spec §20, 3a §9).

    ``valid`` is decided by ``errors`` alone; warnings are advisory and never
    block. Estimates are only computed for a problem with no errors — that
    gate is what guarantees the slack arithmetic is safe (finite, integral
    inequalities, unique names) and the estimate matches compilation exactly.
    ``model_type`` records which compiler path the estimate assumed.
    """

    valid: bool
    errors: list[SolveError] = []
    warnings: list[SolveError] = []
    estimated_compiled_variables: int | None = None  # 原始變數 + slack bits，純算術
    objective_scale: float | None = None
    model_type: ModelType | None = None


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
    # Error-pass views of the variables: a type per name and a bounds tuple
    # per name that is None when the declared bounds are illegal (3b §9.2).
    # ``Variable.bounds()`` is never called here — it raises on exactly the
    # variables this pass is about to report.
    variable_types = {variable.name: variable.type for variable in problem.variables}
    safe_bounds = {
        variable.name: _declared_bounds(variable) for variable in problem.variables
    }

    _check_variables(problem, errors)
    _check_version(problem, errors)
    _check_constraint_ids(problem, errors)
    _check_objective(problem, known_variables, variable_types, errors)
    for index, constraint in enumerate(problem.constraints):
        _check_constraint(constraint, index, known_variables, safe_bounds, errors)
    _check_solver_preferences(problem, errors)

    duplicate_warnings: list[SolveError] = []
    _warn_duplicate_terms(problem.objective, duplicate_warnings)
    return errors, duplicate_warnings


def validate_problem_full(
    problem: OptimizationProblem,
    *,
    capabilities: SolverCapabilities | None = None,
    max_compiled_variables: int | None = None,
    model_type: ModelType | None = None,
) -> ProblemValidationResult:
    """Validate a problem and add the §20 advisory layer (3a §9).

    Reuses :func:`validate_problem` unchanged for errors. Warnings and
    estimates are only produced when there are no errors: an erroneous
    problem must be fixed first anyway, and the no-error gate is exactly
    what makes the pure slack arithmetic well-defined.

    ``capabilities`` is the declaration of the backend the problem names,
    supplied by the caller (the validator knows no registry). When it is
    None only backend-independent checks run: no backend-fit and no
    ignored-parameter warnings.

    ``max_compiled_variables`` is the caller-supplied policy ceiling for an
    exhaustive backend (validation must not read policy itself, §4); it is
    only consulted when ``capabilities.exhaustive`` and skipped when None.

    ``model_type`` is the compiler path the caller will actually take; it
    falls back to ``capabilities.preferred_model_type`` (``"bqm"`` without
    capabilities). The BQM estimate counts encoding and slack bits; the CQM
    estimate is the variable count plus the integer slacks of 3b §15.3
    (§9.4).
    """
    errors, duplicate_warnings = _collect(problem)
    if errors:
        return ProblemValidationResult(valid=False, errors=errors)

    if model_type is None:
        model_type = capabilities.preferred_model_type if capabilities else "bqm"

    # Safe from here on: the error pass guarantees every bound is legal.
    bounds = variable_bounds(problem)
    objective_scale = compute_objective_scale(problem.objective, bounds)
    estimated = (
        estimate_compiled_variables(problem)
        if model_type == "bqm"
        else estimate_cqm_variables(problem)
    )

    warnings: list[SolveError] = []
    _warn_soft_weights(problem, objective_scale, warnings)
    _warn_inequalities(problem, bounds, warnings)
    _warn_integer_encoding(problem, bounds, model_type, warnings)
    if capabilities is not None:
        _warn_backend_fit(
            problem, estimated, bounds, capabilities, max_compiled_variables, warnings
        )
        _warn_ignored_parameters(problem, capabilities, model_type, warnings)
    warnings.extend(duplicate_warnings)

    return ProblemValidationResult(
        valid=True,
        errors=[],
        warnings=warnings,
        estimated_compiled_variables=estimated,
        objective_scale=objective_scale,
        model_type=model_type,
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
    problem: OptimizationProblem, bounds: Bounds, warnings: list[SolveError]
) -> None:
    for index, constraint in enumerate(problem.constraints):
        if constraint.operator not in ("<=", ">="):
            continue
        analysis = analyze_inequality(constraint, bounds)
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
        slack_bits = count_slack_bits(constraint, bounds)
        if slack_bits > LARGE_SLACK_BITS_THRESHOLD:
            warnings.append(
                _warning(
                    "LARGE_SLACK_RANGE",
                    f"constraints[{index}]",
                    (
                        f"Inequality constraint {constraint.id} would need "
                        f"{slack_bits} slack bits on a BQM backend (more than "
                        f"{LARGE_SLACK_BITS_THRESHOLD}); the compiled model "
                        "grows accordingly"
                    ),
                )
            )


def _warn_integer_encoding(
    problem: OptimizationProblem,
    bounds: Bounds,
    model_type: ModelType,
    warnings: list[SolveError],
) -> None:
    """3b §9.3: cost of binary-encoding integer variables on the BQM path.

    A CQM path takes integer variables natively, so neither warning applies
    there.
    """
    if model_type != "bqm":
        return
    has_integer = False
    for index, variable in enumerate(problem.variables):
        if variable.type != "integer":
            continue
        has_integer = True
        bits = integer_encoding_bits(*bounds[variable.name])
        if bits > LARGE_INTEGER_BITS_THRESHOLD:
            warnings.append(
                _warning(
                    "LARGE_INTEGER_RANGE",
                    f"variables[{index}]",
                    (
                        f"Integer variable {variable.name} with bounds "
                        f"{list(bounds[variable.name])} needs {bits} encoding bits "
                        f"on a BQM backend (more than "
                        f"{LARGE_INTEGER_BITS_THRESHOLD}); the compiled model "
                        "grows accordingly"
                    ),
                )
            )
    if not has_integer:
        return
    interactions = estimate_encoded_interactions(problem)
    if interactions > INTEGER_QUADRATIC_BLOWUP_THRESHOLD:
        warnings.append(
            _warning(
                "INTEGER_QUADRATIC_BLOWUP",
                "variables",
                (
                    f"Binary-encoding the integer variables yields up to "
                    f"{interactions} quadratic interactions on a BQM backend "
                    f"(more than {INTEGER_QUADRATIC_BLOWUP_THRESHOLD})"
                ),
            )
        )


def _warn_backend_fit(
    problem: OptimizationProblem,
    estimated: int,
    bounds: Bounds,
    caps: SolverCapabilities,
    max_compiled_variables: int | None,
    warnings: list[SolveError],
) -> None:
    """3a §9.3: size advice from the backend's declared capabilities."""
    if caps.exhaustive and max_compiled_variables is not None:
        if estimated > max_compiled_variables:
            warnings.append(
                _warning(
                    "EXACT_OVER_LIMIT",
                    "solver.backend",
                    (
                        f"Estimated compiled variables ({estimated}) exceed "
                        f"the exhaustive backend limit "
                        f"({max_compiled_variables}); solving will fail with "
                        "resource_limit_exceeded"
                    ),
                )
            )
        elif estimated > max_compiled_variables * EXACT_NEAR_LIMIT_RATIO:
            warnings.append(
                _warning(
                    "EXACT_NEAR_LIMIT",
                    "solver.backend",
                    (
                        f"Estimated compiled variables ({estimated}) are above "
                        f"{EXACT_NEAR_LIMIT_RATIO:.0%} of the exhaustive "
                        f"backend limit ({max_compiled_variables})"
                    ),
                )
            )

    if caps.requires_embedding:
        # A squared penalty forms its clique over compiled *bits* (3b §9.3):
        # one per binary variable, the encoding bits of an integer one, plus
        # the constraint's slack bits.
        max_constraint_variables = max(
            (
                constraint_bit_count(constraint, bounds)
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
                f"largest constraint spans {max_constraint_variables} "
                f"compiled bits (more than "
                f"{QPU_DENSE_CONSTRAINT_VARIABLE_THRESHOLD})"
            )
        if reasons:
            warnings.append(
                _warning(
                    "DENSE_FOR_QPU",
                    "solver.backend",
                    (
                        "Problem is likely too dense for a backend that "
                        "requires minor-embedding: " + "; ".join(reasons)
                    ),
                )
            )


def _is_default(solver: SolverPreferences, field: str) -> bool:
    return getattr(solver, field) == getattr(_DEFAULT_SOLVER_PREFERENCES, field)


def _warn_ignored_parameters(
    problem: OptimizationProblem,
    caps: SolverCapabilities,
    model_type: ModelType,
    warnings: list[SolveError],
) -> None:
    """3a §9.3: preferences the declared backend / model path will ignore."""
    solver = problem.solver

    # seed gets its dedicated code on every backend that ignores it (§15);
    # PARAMETER_IGNORED below deliberately skips it to avoid double-warning.
    if solver.seed is not None and not caps.supports_seed:
        warnings.append(
            _warning(
                "SEED_IGNORED",
                "solver.seed",
                (
                    f"Backend {caps.name} does not support seeding; "
                    "solver.seed will be ignored"
                ),
            )
        )

    ignored: list[tuple[str, str]] = []  # (field, reason)
    if not _is_default(solver, "num_reads") and not caps.supports_num_reads:
        ignored.append(("num_reads", "does not take a number of reads"))
    if not _is_default(solver, "num_sweeps") and not caps.supports_num_sweeps:
        ignored.append(("num_sweeps", "does not take a number of sweeps"))
    if not _is_default(solver, "penalty_multiplier") and model_type == "cqm":
        ignored.append(
            ("penalty_multiplier", "applies no hard penalty on the CQM path")
        )
    if not _is_default(solver, "max_retries") and (
        model_type == "cqm" or caps.exhaustive
    ):
        reason = (
            "applies no hard penalty on the CQM path, so there is nothing to retry"
            if model_type == "cqm"
            else "is exhaustive, so a retry can never surface new samples"
        )
        ignored.append(("max_retries", reason))
    for field, reason in ignored:
        warnings.append(
            _warning(
                "PARAMETER_IGNORED",
                f"solver.{field}",
                (
                    f"solver.{field}={getattr(solver, field)} has no effect: "
                    f"backend {caps.name} {reason}; it will be ignored"
                ),
            )
        )

    # An option block filled in for a backend other than the selected one
    # (block field name != capabilities.name, 3a §9.3 naming contract).
    for block in _option_blocks():
        if getattr(solver, block) is not None and block != caps.name:
            warnings.append(
                _warning(
                    "PARAMETER_IGNORED",
                    f"solver.{block}",
                    (
                        f"solver.{block} holds options for a different backend "
                        f"than the selected {caps.name}; they will be ignored"
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
        _check_variable_bounds(variable, f"variables[{index}]", errors)


def _check_variable_bounds(variable: Variable, path: str, errors: list[SolveError]) -> None:
    """3b §9.1: integer variables need legal bounds, binary ones take none."""
    lower, upper = variable.lower_bound, variable.upper_bound
    if variable.type == "binary":
        if lower is not None or upper is not None:
            errors.append(
                _error(
                    code="BOUNDS_ON_BINARY",
                    path=path,
                    message=(
                        f"Binary variable {variable.name} declares bounds "
                        f"[{lower}, {upper}]; binary variables are always 0/1"
                    ),
                )
            )
        return

    if lower is None or upper is None:
        errors.append(
            _error(
                code="INTEGER_BOUNDS_MISSING",
                path=path,
                message=(
                    f"Integer variable {variable.name} needs both lower_bound and "
                    f"upper_bound, got lower_bound={lower}, upper_bound={upper}"
                ),
            )
        )
        return
    if upper <= lower:
        errors.append(
            _error(
                code="INTEGER_BOUNDS_INVALID",
                path=path,
                message=(
                    f"Integer variable {variable.name} has upper_bound {upper} "
                    f"<= lower_bound {lower}; a variable needs at least two values"
                ),
            )
        )
    if abs(lower) > INTEGER_BOUND_LIMIT or abs(upper) > INTEGER_BOUND_LIMIT:
        errors.append(
            _error(
                code="INTEGER_RANGE_TOO_LARGE",
                path=path,
                message=(
                    f"Integer variable {variable.name} has bounds [{lower}, {upper}] "
                    f"outside the supported range of ±{INTEGER_BOUND_LIMIT}"
                ),
            )
        )


def _declared_bounds(variable: Variable) -> tuple[int, int] | None:
    """The variable's range if its declaration is legal, else ``None``.

    Mirrors :func:`_check_variable_bounds` without raising: the error pass
    uses it to skip range-based checks on variables it has already reported.
    """
    if variable.type == "binary":
        if variable.lower_bound is None and variable.upper_bound is None:
            return (0, 1)
        return None
    lower, upper = variable.lower_bound, variable.upper_bound
    if lower is None or upper is None or upper <= lower:
        return None
    if abs(lower) > INTEGER_BOUND_LIMIT or abs(upper) > INTEGER_BOUND_LIMIT:
        return None
    return (lower, upper)


def _check_version(problem: OptimizationProblem, errors: list[SolveError]) -> None:
    """3b §7: integer variables exist only from schema version 1.1 on."""
    if problem.version != "1.0":
        return
    integer_names = [v.name for v in problem.variables if v.type == "integer"]
    if integer_names:
        errors.append(
            _error(
                code="INTEGER_REQUIRES_VERSION_1_1",
                path="version",
                message=(
                    f"Problem declares integer variables ({', '.join(integer_names)}) "
                    f"but version is {problem.version!r}; integer variables require "
                    'version "1.1"'
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
    variable_types: dict[str, str],
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
        # x*x collapses to x only for a binary variable; an integer's square
        # is a legitimate quadratic term (3b §9.2). Unknown names keep the
        # Phase 1 verdict (they are reported as UNKNOWN_VARIABLE as well).
        if (
            term.variable1 == term.variable2
            and variable_types.get(term.variable1, "binary") == "binary"
        ):
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
    safe_bounds: dict[str, tuple[int, int] | None],
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

    _check_trivially_infeasible(constraint, base, safe_bounds, errors)


def _check_trivially_infeasible(
    constraint: Constraint,
    base: str,
    safe_bounds: dict[str, tuple[int, int] | None],
    errors: list[SolveError],
) -> None:
    """Reject a hard constraint no assignment within the bounds can satisfy.

    The lhs range is taken over the *accumulated* coefficients (one per
    variable) and the variables' declared ranges, exactly as the compiler
    sums them, so a constraint accepted here can never fail slack encoding
    later. A constraint mentioning a variable whose bounds are illegal is
    skipped: that variable already carries its own error (3b §9.2). The
    message reports the user's own operator and rhs, never the compiler's
    normalized form.
    """
    if constraint.type != "hard" or not constraint.terms:
        return
    if not all(math.isfinite(term.coefficient) for term in constraint.terms):
        return
    if not math.isfinite(constraint.rhs):
        return
    if any(safe_bounds.get(term.variable, (0, 1)) is None for term in constraint.terms):
        return

    bounds: Bounds = {
        name: declared for name, declared in safe_bounds.items() if declared is not None
    }
    lhs_min, lhs_max = lhs_bounds(accumulate_terms(constraint.terms), bounds)
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
    for options_field, field in _POSITIVE_OPTION_FIELDS:
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
