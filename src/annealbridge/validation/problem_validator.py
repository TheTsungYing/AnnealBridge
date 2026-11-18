"""Pre-compilation validation of an OptimizationProblem (spec §12, Phase 2 §20, 3a §9).

``validate_problem`` collects *all* errors in a single pass and returns them
as a list of ``SolveError`` (each carrying the catalog's fixed
``recommended_action``); it never raises and never stops at the first
problem. Duplicate objective terms are legal (the compiler sums coefficients)
and only produce a DUPLICATE_TERM_MERGED warning plus one log line (spec §8).

``validate_problem_full`` wraps the same error pass and adds the Phase 2 §20
advisory layer: non-blocking warnings, ``estimated_compiled_variables`` and
``objective_scale``, so an agent can fix a problem or switch backend before
spending quota. Warnings never affect ``valid``. A soft constraint that no
assignment can satisfy is legal but warned about (``SOFT_ALWAYS_VIOLATED``,
review F-04 / F-12) with the very judgement ``TRIVIALLY_INFEASIBLE`` uses
for hard ones, so the compilers never disagree with this module.

Backend-dependent advice is driven purely by the injected
``SolverCapabilities`` declaration (3a §9): this module never names a
backend, never reads policy and never imports ``solvers`` (spec §4).

Within one call everything that does not depend on the backend is computed
once and held in a :class:`ValidationCache` (2026-11-18 review R-06); a
caller validating one problem against several backends passes its own to
:func:`validate_expanded` so they share it. The cache changes how often
something is computed, never what is reported.
"""

import logging
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from annealbridge.models import (
    CardinalityConstraint,
    Constraint,
    LoweredCardinalityConstraint,
    ModelType,
    Objective,
    OptimizationProblem,
    SolveError,
    SolverCapabilities,
    SolverPreferences,
    Variable,
)
from annealbridge.models.reflection import is_number_type, model_class, union_members
from annealbridge.validation.estimates import (
    Bounds,
    ProblemAnalysis,
    accumulate_terms,
    analyze_problem,
    compute_objective_scale,
    compute_soft_energy_total,
    constraint_bit_count,
    count_slack_bits,
    equality_violation_step_below_one,
    estimate_encoded_interactions,
    estimate_model_variables,
    integer_encoding_bits,
    lhs_bounds,
    variable_bounds,
)
from annealbridge.validation.expansion import (
    DEFAULT_MAX_TEMPLATE_BINDINGS,
    ERRORS_PER_SOURCE,
    ExpandedProblem,
    expand_problem,
)

# Shared with the template expander (schema 1.3 spec §14.8): moved to
# ``validation/issues`` unchanged and re-exported here under the same names.
from annealbridge.validation.issues import (  # noqa: F401
    _WARNING_RECOMMENDED_ACTIONS,
    INTEGER_BOUND_LIMIT,
    _check_variable_bounds,
    _duplicate_linear_variables,
    _duplicate_quadratic_pairs,
    _error,
    _warning,
)
from annealbridge.validation.tolerance import satisfies, tolerance

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
# 2026-09-09 review F-24: the bound alone does not make the *slack range*
# exact. ``analyze_inequality`` computes ``rhs - lhs_min`` in float64 from
# ``coefficient * bound`` products, and float64 represents every integer
# up to 2^53 exactly but not beyond. An inequality whose per-term
# ``sum(|c| * max|bound|) + |rhs|`` stays within that limit has every
# product and every partial sum exact, so the slack encoding is exact;
# above it the range can be off by hundreds of units and a feasible
# assignment may become unreachable (an exhaustive backend would then
# "prove" infeasibility wrongly). Binary variables count with bound 1.
INEQUALITY_MAGNITUDE_LIMIT = 2**53

# Defaults are read off the model so they can never drift from the schema.
_DEFAULT_SOLVER_PREFERENCES = SolverPreferences()

# Every error code this validator can emit (Phase 1 spec §12 plus
# NO_VARIABLES, plus the 3b §9.1 integer rules). Each must have a fixed
# recommended_action in the error catalog; tests/unit/test_error_catalog.py
# enforces the coverage.
VALIDATOR_ERROR_CODES: frozenset[str] = frozenset(
    {
        "BOUNDS_ON_BINARY",
        "CARDINALITY_VARIABLE_NOT_BINARY",
        "COEFFICIENT_MAGNITUDE_TOO_LARGE",
        "DUPLICATE_CARDINALITY_VARIABLE",
        "DUPLICATE_CONSTRAINT_ID",
        "DUPLICATE_TEMPLATE_NAME",
        "DUPLICATE_VARIABLE",
        "EMPTY_CONSTRAINT",
        "FEATURE_REQUIRES_NEWER_VERSION",
        "HARD_CONSTRAINT_HAS_WEIGHT",
        "INDEX_SET_INVALID",
        "INEQUALITY_MAGNITUDE_TOO_LARGE",
        "INTEGER_BOUNDS_INVALID",
        "INTEGER_BOUNDS_MISSING",
        "INTEGER_RANGE_TOO_LARGE",
        "INTEGER_REQUIRES_VERSION_1_1",
        "INVALID_SOLVER_PREFERENCE",
        "NON_FINITE_COEFFICIENT",
        "NON_INTEGER_INEQUALITY",
        "NO_VARIABLES",
        "PARAMETER_TABLE_INVALID",
        "PARAMETER_VALUE_MISSING",
        "RESERVED_VARIABLE_NAME",
        "SELF_QUADRATIC_TERM",
        "SOFT_CONSTRAINT_MISSING_WEIGHT",
        "TEMPLATE_EXPANSION_LIMIT",
        "TEMPLATE_REFERENCE_INVALID",
        "TRIVIALLY_INFEASIBLE",
        "UNKNOWN_VARIABLE",
        "WALL_CLOCK_LIMIT_UNSUPPORTED",
    }
)


def _optional_model(annotation: object) -> type[BaseModel] | None:
    """The ``Model`` in a ``Model | None`` annotation, else None (3a §9.4).

    Only a union of exactly one BaseModel subclass with None counts as an
    option block; ``int | None`` (``seed``), bare types and a model wrapped
    in ``Annotated`` or ``list`` do not. Both union spellings are accepted
    (``X | None`` and ``Optional[X]``).
    """
    members = union_members(annotation)
    if members is None or len(members) != 1:
        return None
    return model_class(members[0])


def _optional_number(annotation: object) -> bool:
    """True for ``int | None`` / ``float | None`` (either union spelling).

    ``Annotated`` wrappers are transparent, so ``Quantity | None`` counts;
    ``bool | None`` and a bare ``int`` do not, nor does a union of several
    numeric types.
    """
    members = union_members(annotation)
    return members is not None and len(members) == 1 and is_number_type(members[0])


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

# Schema versions that predate cardinality_constraints (schema 1.2 spec
# §6.4). Gated on the list being non-empty, never on the key being present:
# a dump / validate round trip always carries the empty default.
_VERSIONS_WITHOUT_CARDINALITY: frozenset[str] = frozenset({"1.0", "1.1"})


class ProblemValidationResult(BaseModel):
    """Full validation outcome (Phase 2 spec §20, 3a §9).

    ``valid`` is decided by ``errors`` alone; warnings are advisory and never
    block. Estimates are only computed for a problem with no errors — that
    gate is what guarantees the slack arithmetic is safe (finite, integral
    inequalities, unique names) and the estimate matches compilation exactly.
    ``model_type`` records which compiler path the estimate assumed.
    """

    valid: bool = Field(
        description=(
            "Decided by errors alone. Warnings are advisory and never make a "
            "problem invalid."
        )
    )
    errors: list[SolveError] = Field(
        default=[],
        description=(
            "Every error found in one pass — validation does not stop at the "
            "first. Empty means the problem is safe to compile."
        ),
    )
    warnings: list[SolveError] = Field(
        default=[],
        description=(
            "Advisory findings that do not block a solve. Only produced when "
            "there are no errors."
        ),
    )
    # 3b §9.4：依 model_type 決定用哪條路徑；兩者都是純算術，不建模型。
    estimated_compiled_variables: int | None = Field(
        default=None,
        description=(
            "Compiled size without building a model: on the BQM path, binary "
            "variables + integer-encoding bits + slack bits; on the CQM path, "
            "variables + integer slacks. Null when the problem is invalid."
        ),
    )
    objective_scale: float | None = Field(
        default=None,
        description=(
            "The upper bound on the objective's range, used to size hard "
            "penalties and to judge whether a soft weight is meaningful. Null "
            "when the problem is invalid."
        ),
    )
    model_type: ModelType | None = Field(
        default=None,
        description=(
            "Which compiler path the estimate assumed. Null when the problem "
            "is invalid."
        ),
    )


def validate_problem(
    problem: OptimizationProblem,
    *,
    max_template_bindings: int = DEFAULT_MAX_TEMPLATE_BINDINGS,
) -> list[SolveError]:
    """Validate a problem before compilation, returning all errors found.

    An empty list means the problem is safe to hand to the compiler. A
    problem with schema 1.3 templates is expanded first (schema 1.3 spec
    §14.8) and the errors point into the templates; ``max_template_bindings``
    is the expansion ceiling, the policy default unless given. For such a
    problem only the errors come back, never the expanded problem: to
    compile or solve it, expand it with :func:`expand_problem` and hand on
    its ``problem``.
    """
    if not problem.has_templates():
        errors, _ = _collect(problem)
        return errors
    expansion = expand_problem(problem, max_template_bindings=max_template_bindings)
    return validate_expanded(expansion, errors_only=True).errors


def _collect(
    problem: OptimizationProblem,
    errors: list[SolveError] | None = None,
    *,
    duplicate_terms: bool = True,
    cache: "ValidationCache | None" = None,
) -> tuple[list[SolveError], list[SolveError]]:
    """Single error pass plus the spec §8 duplicate-term warnings.

    Both public entry points go through here so a duplicate term is logged
    exactly once, whichever of them is called. ``errors`` is the list to
    append to (a fresh one by default; :func:`validate_expanded` passes its
    bounded collector), and ``duplicate_terms=False`` leaves the duplicate
    terms to :func:`validate_expanded`, which attributes them to templates.
    ``cache``, when given, receives the variable bounds and the objective
    scale a clean pass computes, for the advisory layer to reuse.
    """
    errors = [] if errors is None else errors
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
    _check_feature_versions(problem, errors)
    _check_constraint_ids(problem, errors)
    _check_objective(problem, known_variables, variable_types, errors)
    for index, constraint in enumerate(problem.constraints):
        _check_constraint(constraint, index, known_variables, safe_bounds, errors)
    for index, cardinality in enumerate(problem.cardinality_constraints):
        _check_cardinality_constraint(cardinality, index, variable_types, errors)
    _check_solver_preferences(problem, errors)
    if not errors:
        # Needs legal bounds, so only once everything above is clean.
        bounds = variable_bounds(problem)
        objective_scale = compute_objective_scale(problem.objective, bounds)
        _check_objective_range(objective_scale, errors)
        if cache is not None:
            cache._bounds = bounds
            cache._objective_scale = objective_scale

    duplicate_warnings: list[SolveError] = []
    if duplicate_terms:
        _warn_duplicate_terms(problem.objective, duplicate_warnings)
    return errors, duplicate_warnings


def validate_problem_full(
    problem: OptimizationProblem,
    *,
    capabilities: SolverCapabilities | None = None,
    max_compiled_variables: int | None = None,
    model_type: ModelType | None = None,
    max_template_bindings: int = DEFAULT_MAX_TEMPLATE_BINDINGS,
) -> ProblemValidationResult:
    """Validate a problem and add the §20 advisory layer (3a §9).

    Reuses :func:`validate_problem` unchanged for errors, then adds the
    backend-dependent errors: a ``solver.seed`` outside the range
    ``capabilities`` declares (see :func:`_check_seed_range`), and a
    ``solver.wall_clock_limit_seconds`` on a backend that cannot stop
    part-way (see :func:`_check_wall_clock_limit`). Warnings and
    estimates are only produced when there are no errors: an erroneous
    problem must be fixed first anyway, and the no-error gate is exactly
    what makes the pure slack arithmetic well-defined.

    ``capabilities`` is the declaration of the backend the problem names,
    supplied by the caller (the validator knows no registry). When it is
    None only backend-independent checks run: no seed-range error, no
    backend-fit and no ignored-parameter warnings.

    ``max_compiled_variables`` is the caller-supplied policy ceiling for an
    exhaustive backend (validation must not read policy itself, §4); it is
    only consulted when ``capabilities.exhaustive`` and skipped when None.

    ``model_type`` is the compiler path the caller will actually take; it
    falls back to ``capabilities.preferred_model_type`` (``"bqm"`` without
    capabilities). The BQM estimate counts encoding and slack bits; the CQM
    estimate is the variable count plus the integer slacks of 3b §15.3
    (§9.4).

    A problem with schema 1.3 templates is expanded first, under the
    ceiling ``max_template_bindings``, and validated through
    :func:`validate_expanded` (schema 1.3 spec §14.8).
    """
    if problem.has_templates():
        return validate_expanded(
            expand_problem(problem, max_template_bindings=max_template_bindings),
            capabilities=capabilities,
            max_compiled_variables=max_compiled_variables,
            model_type=model_type,
        )
    return _validate_full(
        ValidationCache()._bound_to(problem),
        capabilities=capabilities,
        max_compiled_variables=max_compiled_variables,
        model_type=model_type,
    )


def _validate_full(
    cache: "ValidationCache",
    *,
    capabilities: SolverCapabilities | None,
    max_compiled_variables: int | None,
    model_type: ModelType | None,
) -> ProblemValidationResult:
    """:func:`validate_problem_full` of the problem ``cache`` is bound to.

    The one sequence of checks for every problem, with or without
    templates. What does not depend on the backend comes from ``cache``,
    computed on first use; each call then continues on its own copy of the
    collectors, so the issues of one backend are appended in the order they
    always were (the error pass, the backend's own errors, the compiler
    path's error; then the shared advisories, the compiler path's, the
    backend's, the duplicate terms) and a second backend sees none of them.
    For a templated problem the expansion's own warnings lead the list
    (:meth:`ValidationCache._finished_warnings`).
    """
    problem = cache._problem
    assert problem is not None  # bound
    errors = _fork(cache._error_pass())
    if capabilities is not None:
        _check_seed_range(problem, capabilities, errors)
        _check_wall_clock_limit(problem, capabilities, errors)
    if errors:
        return ProblemValidationResult(valid=False, errors=cache._finished_errors(errors))

    if model_type is None:
        model_type = capabilities.preferred_model_type if capabilities else "bqm"

    # Safe from here on: the error pass guarantees every bound is legal.
    path = cache._model_path(model_type, capabilities)
    if path.error is not None:
        errors.append(path.error)
        return ProblemValidationResult(valid=False, errors=cache._finished_errors(errors))
    assert path.estimated is not None and path.warnings is not None

    warnings = _fork(path.warnings)
    if capabilities is not None:
        _warn_exact_limits(path.estimated, capabilities, max_compiled_variables, warnings)
        _warn_embedding_density(
            path.estimated, cache._largest_constraint_bits, capabilities, warnings
        )
        _warn_ignored_parameters(problem, capabilities, model_type, warnings)

    return ProblemValidationResult(
        valid=True,
        errors=[],
        warnings=cache._finished_warnings(warnings),
        estimated_compiled_variables=path.estimated,
        objective_scale=cache._objective_scale,
        model_type=model_type,
    )


def validate_expanded(
    expansion: ExpandedProblem,
    *,
    capabilities: SolverCapabilities | None = None,
    max_compiled_variables: int | None = None,
    model_type: ModelType | None = None,
    errors_only: bool = False,
    cache: "ValidationCache | None" = None,
) -> ProblemValidationResult:
    """Validate the outcome of :func:`expand_problem` (schema 1.3 spec §14.8).

    The one path from an expansion to a validation result, shared by the
    service, routing and the two public validators so none of them expands
    twice or reports differently:

    * expansion errors are the result, ``valid: false``, nothing else;
    * a problem that had no templates is validated exactly as always;
    * an expanded one is validated with a bounded collector
      (:class:`_TemplateIssues`) that maps every path on a generated entry
      back to its template, keeps at most twenty errors per template and one
      warning per (code, template path), and counts the rest (spec §14.11).

    Warnings -- the expansion's own ones first -- are only reported when
    there is no error at all, as for every problem. ``errors_only`` runs the
    error pass alone, as :func:`validate_problem` does.

    ``cache`` is for a caller that validates the same ``expansion`` more
    than once within one call of its own (routing: the error pass, then
    every backend): pass the same fresh :class:`ValidationCache` each time
    and the work that does not depend on the backend is done once. The
    results are the ones separate calls give, log lines included.
    """
    if expansion.errors:
        return ProblemValidationResult(valid=False, errors=list(expansion.errors))
    problem = expansion.problem
    assert problem is not None  # no errors, so expanded
    cache = (ValidationCache() if cache is None else cache)._bound_to(problem, expansion)
    if errors_only:
        final = cache._finished_errors(_fork(cache._error_pass()))
        return ProblemValidationResult(valid=not final, errors=final)
    return _validate_full(
        cache,
        capabilities=capabilities,
        max_compiled_variables=max_compiled_variables,
        model_type=model_type,
    )


@dataclass
class _ModelPath:
    """What :class:`ValidationCache` holds for one compiler path.

    Either the path's own ``error``, or its compiled-size estimate and the
    advisories up to the last one that depends on nothing but the path.
    """

    error: SolveError | None = None
    estimated: int | None = None
    warnings: list[SolveError] | None = None


class ValidationCache:
    """What one validation call computes once (2026-11-18 review R-06).

    Validating a problem against a backend is mostly work that does not
    depend on the backend: the error pass, the variable bounds, the
    objective scale, lowering and analysing every constraint, and the
    advisories about the constraints themselves. A cache holds that work so
    it is done once per call rather than once per check, and once per
    ``recommend`` rather than once per backend. What depends on the compiler
    path (the size estimate, the BQM-only checks) is held per path, keyed
    by the model type and by whether the backend is exhaustive, so a BQM
    result is never handed to a CQM backend. What depends on the backend
    itself is never cached.

    Create one, pass it to every :func:`validate_expanded` call that
    validates the same expansion within one call of your own, and let it
    go: it describes the problem as it was when first used, so it must not
    be kept across calls, stored on the problem or shared between problems.
    Using it for a second problem raises ``ValueError``. Without one,
    :func:`validate_expanded` makes its own, so there is one validation
    path either way.

    Order is part of the output. The cache never merges finished lists: it
    keeps the *collector* as it stood after each shared stage, and every
    backend continues on a copy (:func:`_fork`), so each backend's
    collector receives exactly the appends it always received. That is
    what keeps the per-template caps and counts of :class:`_TemplateIssues`
    unchanged. A duplicate objective term is still logged every time a
    validation reports it, although it is only looked for once.

    The lists of two results are never shared, but the issues in them may
    be the same ``SolveError`` objects: an issue found once is reported by
    every validation it applies to. Treat them as read-only.
    """

    def __init__(self) -> None:
        self._subject: object = None
        self._problem: OptimizationProblem | None = None
        # Set for an expanded (templated) problem only.
        self._expansion: ExpandedProblem | None = None
        # The error pass.
        self._errors: list[SolveError] | None = None
        self._duplicate_warnings: list[SolveError] = []
        self._bounds: Bounds | None = None
        self._objective_scale: float | None = None
        # The advisory layer, each part on first use.
        self._paths: list[tuple[str, Constraint]] | None = None
        self._analysis: ProblemAnalysis | None = None
        self._shared_warnings: list[SolveError] | None = None
        self._soft_energy_total: float | None = None
        self._estimates: dict[ModelType, int] = {}
        self._interactions: int | None = None
        self._constraint_bits: int | None = None
        self._model_paths: dict[tuple[ModelType, bool], _ModelPath] = {}
        self._template_duplicates: list[SolveError] | None = None

    def _bound_to(
        self, problem: OptimizationProblem, expansion: ExpandedProblem | None = None
    ) -> "ValidationCache":
        """Bind to the problem on first use; refuse another one afterwards."""
        subject: object = problem if expansion is None else expansion
        if self._subject is None:
            self._subject = subject
            self._problem = problem
            if expansion is not None and expansion.templated:
                self._expansion = expansion
        elif self._subject is not subject:
            raise ValueError(
                "a ValidationCache holds the validation of one problem; "
                "create a new one for each problem and each call"
            )
        return self

    def _collector(self, *, errors: bool) -> list[SolveError]:
        """An empty issue list: bounded and path-mapping for a templated problem."""
        if self._expansion is None:
            return []
        return _TemplateIssues(self._expansion, errors=errors)

    def _error_pass(self) -> list[SolveError]:
        """The collector as the error pass left it. Shared: fork it to append.

        The pass runs once. A later call logs the duplicate terms again, as
        the pass itself would have.
        """
        if self._errors is None:
            problem = self._problem
            assert problem is not None  # bound
            errors = self._collector(errors=True)
            # A templated problem's duplicate terms are attributed to their
            # templates at the end (:meth:`_template_duplicate_warnings`).
            _, self._duplicate_warnings = _collect(
                problem, errors, duplicate_terms=self._expansion is None, cache=self
            )
            self._errors = errors
        else:
            _log_warnings(self._duplicate_warnings)
        return self._errors

    def _finished_errors(self, errors: list[SolveError]) -> list[SolveError]:
        """A call's own error collector as the list to report."""
        if isinstance(errors, _TemplateIssues):
            return errors.finish()
        return errors

    def _constraints(self) -> tuple[list[tuple[str, Constraint]], ProblemAnalysis]:
        """Every constraint with its path, and its analysis: built once."""
        if self._paths is None or self._analysis is None:
            problem = self._problem
            assert problem is not None and self._bounds is not None  # clean error pass
            self._paths = _constraint_paths(problem)
            self._analysis = analyze_problem(
                problem,
                bounds=self._bounds,
                constraints=[constraint for _, constraint in self._paths],
            )
        return self._paths, self._analysis

    def _shared_advisories(self) -> list[SolveError]:
        """The advisories about the constraints alone. Shared: fork to append."""
        if self._shared_warnings is None:
            assert self._objective_scale is not None  # clean error pass
            paths, analysis = self._constraints()
            warnings = self._collector(errors=False)
            _warn_soft_weights(paths, self._objective_scale, warnings)
            _warn_inequalities(paths, analysis, warnings)
            _warn_constraint_ranges(paths, analysis.bounds, warnings)
            self._shared_warnings = warnings
        return self._shared_warnings

    def _soft_energy(self) -> float:
        if self._soft_energy_total is None:
            problem = self._problem
            assert problem is not None  # bound
            self._soft_energy_total = compute_soft_energy_total(
                problem, problem_analysis=self._constraints()[1]
            )
        return self._soft_energy_total

    def _estimated(self, model_type: ModelType) -> int:
        if model_type not in self._estimates:
            problem = self._problem
            assert problem is not None  # bound
            self._estimates[model_type] = estimate_model_variables(
                problem, model_type, problem_analysis=self._constraints()[1]
            )
        return self._estimates[model_type]

    def _encoded_interactions(self) -> int:
        if self._interactions is None:
            problem = self._problem
            assert problem is not None  # bound
            self._interactions = estimate_encoded_interactions(
                problem, problem_analysis=self._constraints()[1]
            )
        return self._interactions

    def _largest_constraint_bits(self) -> int:
        """The most compiled bits one constraint's penalty couples (3b §9.3)."""
        if self._constraint_bits is None:
            analysis = self._constraints()[1]
            self._constraint_bits = max(
                (
                    constraint_bit_count(constraint, analysis.bounds, analysis=inequality)
                    for constraint, inequality in zip(
                        analysis.all_constraints, analysis.inequalities, strict=True
                    )
                ),
                default=0,
            )
        return self._constraint_bits

    def _model_path(
        self, model_type: ModelType, capabilities: SolverCapabilities | None
    ) -> _ModelPath:
        """The compiler path's error, or its estimate and advisories so far.

        Keyed by everything those depend on: the model type and, for
        :func:`_warn_fractional_equalities`, whether the backend is
        exhaustive. ``warnings`` is shared: fork it to append.
        """
        key = (model_type, capabilities is not None and capabilities.exhaustive)
        path = self._model_paths.get(key)
        if path is None:
            path = self._model_paths[key] = self._build_model_path(model_type, capabilities)
        return path

    def _build_model_path(
        self, model_type: ModelType, capabilities: SolverCapabilities | None
    ) -> _ModelPath:
        problem = self._problem
        assert problem is not None and self._objective_scale is not None  # clean error pass
        errors: list[SolveError] = []
        _check_soft_energy_range(self._objective_scale, self._soft_energy, model_type, errors)
        if errors:
            return _ModelPath(error=errors[0])
        estimated = self._estimated(model_type)

        paths, analysis = self._constraints()
        warnings = _fork(self._shared_advisories())
        _warn_cardinality_form(problem, model_type, warnings)
        _warn_fractional_equalities(paths, model_type, capabilities, warnings)
        _warn_integer_encoding(
            problem, analysis.bounds, model_type, self._encoded_interactions, warnings
        )
        return _ModelPath(estimated=estimated, warnings=warnings)

    def _template_duplicate_warnings(self) -> list[SolveError]:
        """:func:`_template_duplicate_terms`, computed once and logged each time."""
        assert self._expansion is not None  # templated
        if self._template_duplicates is None:
            self._template_duplicates = _template_duplicate_terms(self._expansion)
        else:
            _log_warnings(self._template_duplicates)
        return self._template_duplicates

    def _finished_warnings(self, warnings: list[SolveError]) -> list[SolveError]:
        """A call's own warning collector as the list to report.

        The duplicate objective terms close the list. For a templated
        problem the expansion's own warnings open it and the collector adds
        its counts (schema 1.3 spec §14.11).
        """
        if self._expansion is None:
            warnings.extend(self._duplicate_warnings)
            return warnings
        assert isinstance(warnings, _TemplateIssues)
        return [
            *self._expansion.warnings,
            *warnings.finish(),
            *self._template_duplicate_warnings(),
        ]


def _fork(issues: list[SolveError]) -> list[SolveError]:
    """An independent copy of a collector, in the state it is in now.

    Appending to the copy is appending to the original at this point of its
    history, without the original seeing it: how several validations
    continue from one shared stage (:class:`ValidationCache`).
    """
    if isinstance(issues, _TemplateIssues):
        return issues.fork()
    return list(issues)


def _log_warnings(warnings: list[SolveError]) -> None:
    for warning in warnings:
        logger.warning("%s", warning.message)


class _TemplateIssues(list):
    """The bounded, path-mapping collector of :func:`validate_expanded`.

    Stands in for the plain ``errors`` / ``warnings`` list the validator
    appends to (schema 1.3 spec §14.11). An issue on an explicit entry is
    kept as it is. One on a generated entry is rewritten on arrival: its
    path points into the template and its message says so; then an error is
    kept while its template has fewer than twenty kept errors, and a warning
    only as the first of its (code, path). The rest are counted, never
    stored, so the objects the validator builds for them are freed at once
    and memory does not grow with the number of generated entries.

    ``len()`` is the number of issues *appended*, kept or counted, so every
    check the validator makes on the length of its list (``if errors:``,
    ``len(errors) != errors_before``) means what it always meant.
    """

    def __init__(self, expansion: ExpandedProblem, *, errors: bool) -> None:
        super().__init__()
        self._expansion = expansion
        self._errors = errors
        self._counted = 0
        self._kept_per_source: Counter[str] = Counter()
        self._last_of_source: dict[str, int] = {}
        self._omitted: dict[str, Counter[str]] = {}
        self._warning_at: dict[tuple[str, str], int] = {}
        self._warning_count: Counter[tuple[str, str]] = Counter()

    def append(self, issue: SolveError) -> None:
        location = self._expansion.locate(issue.path)
        if location is None:
            super().append(issue)
            return
        if self._errors:
            source = location.source
            if self._kept_per_source[source] >= ERRORS_PER_SOURCE:
                self._omitted.setdefault(source, Counter())[issue.code] += 1
                self._counted += 1
                return
            self._kept_per_source[source] += 1
            self._last_of_source[source] = list.__len__(self)
        else:
            key = (issue.code, location.path)
            self._warning_count[key] += 1
            if key in self._warning_at:
                self._counted += 1
                return
            self._warning_at[key] = list.__len__(self)
        super().append(
            issue.model_copy(
                update={"path": location.path, "message": issue.message + location.suffix}
            )
        )

    def extend(self, issues) -> None:  # type: ignore[override]
        for issue in issues:
            self.append(issue)

    def __len__(self) -> int:
        return list.__len__(self) + self._counted

    def fork(self) -> "_TemplateIssues":
        """An independent collector in exactly this one's state.

        Every kept issue and every count is copied, in order, so what is
        appended to the copy is capped, aggregated and counted as if it had
        been appended here. The kept issues are bounded (twenty errors per
        template, one warning per code and template path), and so is the
        copy.
        """
        twin = _TemplateIssues(self._expansion, errors=self._errors)
        list.extend(twin, list.__iter__(self))
        twin._counted = self._counted
        twin._kept_per_source = Counter(self._kept_per_source)
        twin._last_of_source = dict(self._last_of_source)
        twin._omitted = {source: Counter(counts) for source, counts in self._omitted.items()}
        twin._warning_at = dict(self._warning_at)
        twin._warning_count = Counter(self._warning_count)
        return twin

    def finish(self) -> list[SolveError]:
        """The kept issues, with counts and cross-reference notes appended."""
        items = [item for item in list.__iter__(self)]
        if self._errors:
            for source, counts in self._omitted.items():
                at = self._last_of_source[source]
                total = sum(counts.values())
                detail = ", ".join(f"{code} ×{count}" for code, count in counts.items())
                items[at] = _with_note(
                    items[at],
                    f"; {total} more error{'s' if total != 1 else ''} from {source} "
                    f"{'are' if total != 1 else 'is'} not listed ({detail})",
                )
            context = _NoteContext(self._expansion)
            return [context.note(item) for item in items]
        for key, at in self._warning_at.items():
            count = self._warning_count[key]
            if count > 1:
                items[at] = _with_note(
                    items[at], f"; {count} generated entries from this template get this warning"
                )
        return items

class _NoteContext:
    """Cross-reference notes for errors on explicit entries (spec §14.11).

    The lookups they need -- the first position of each linear constraint
    id, the declared names -- are built once, on first use, so the notes
    cost one pass over the problem, not one per error.
    """

    def __init__(self, expansion: ExpandedProblem) -> None:
        self._expansion = expansion
        self._problem = expansion.problem
        assert self._problem is not None
        self._first_linear_id: dict[str, int] | None = None
        self._declared: set[str] | None = None

    def note(self, error: SolveError) -> SolveError:
        """Point an error on an explicit entry at the template involved."""
        expansion = self._expansion
        if error.code == "NO_VARIABLES":
            left_out = expansion.all_generated_variables_left_out
            if left_out:
                return _with_note(
                    error,
                    f" (variable_families generated {left_out} variables, but no "
                    "objective term or constraint uses any of them, so all were "
                    "left out)",
                )
        elif error.code == "DUPLICATE_CONSTRAINT_ID" and error.path is not None:
            if expansion.locate(error.path) is None:
                note = self._generated_id_note(error.path)
                if note:
                    return _with_note(error, note)
        elif error.code == "UNKNOWN_VARIABLE" and error.path is not None:
            if expansion.locate(error.path) is None:
                if self._declared is None:
                    self._declared = {variable.name for variable in self._problem.variables}
                families = expansion.family_names
                for name in _names_at(self._problem, error.path):
                    if name not in self._declared and any(
                        name.startswith(family + "[") for family in families
                    ):
                        return _with_note(
                            error,
                            " (a generated variable is named exactly family[e1,e2], "
                            "without spaces, for elements of the family's index sets)",
                        )
        return error

    def _generated_id_note(self, path: str) -> str:
        """For an explicit cardinality id that a linear template also generates.

        ``_check_constraint_ids`` reports the second occurrence in
        ``constraints`` then ``cardinality_constraints`` order, so a generated
        linear constraint sharing an explicit cardinality constraint's id puts
        the error on the explicit entry; the note names the template.
        """
        if not path.startswith("cardinality_constraints["):
            return ""
        problem = self._problem
        if self._first_linear_id is None:
            self._first_linear_id = {}
            for position, constraint in enumerate(problem.constraints):
                self._first_linear_id.setdefault(constraint.id, position)
        index = int(path[len("cardinality_constraints[") : path.index("]")])
        position = self._first_linear_id.get(problem.cardinality_constraints[index].id)
        if position is None:
            return ""
        location = self._expansion.locate(f"constraints[{position}]")
        if location is None:
            return ""
        return location.suffix.replace(" (generated by", " (the same id is generated by", 1)


def _with_note(issue: SolveError, note: str) -> SolveError:
    return issue.model_copy(update={"message": issue.message + note})


def _names_at(problem: OptimizationProblem, path: str) -> list[str]:
    """The variable names an explicit entry at ``path`` references."""
    head, _, rest = path.partition("[")
    try:
        index = int(rest[: rest.index("]")])
        tail = rest[rest.index("]") + 1 :]
        if head == "objective.linear_terms":
            return [problem.objective.linear_terms[index].variable]
        if head == "objective.quadratic_terms":
            term = problem.objective.quadratic_terms[index]
            return [term.variable1, term.variable2]
        if head == "constraints" and tail.startswith(".terms["):
            j = int(tail[len(".terms[") : tail.index("]")])
            return [problem.constraints[index].terms[j].variable]
        if head == "cardinality_constraints" and tail.startswith(".variables["):
            j = int(tail[len(".variables[") : tail.index("]")])
            return [problem.cardinality_constraints[index].variables[j]]
    except (ValueError, IndexError):
        return []
    return []


def _template_duplicate_terms(expansion: ExpandedProblem) -> list[SolveError]:
    """DUPLICATE_TERM_MERGED for an expanded objective, per template (§14.11).

    Its path is the whole term list, which no generated position can map.
    The duplicates are recomputed with the very functions the validator
    uses; each is attributed to the template that generated its first
    repeat after the first occurrence, and aggregated per template. One
    made of explicit terms only keeps its path and message. One log line
    per warning, not per duplicate.
    """
    problem = expansion.problem
    assert problem is not None
    objective = problem.objective
    result: list[SolveError] = []
    groups = (
        (
            "linear",
            _duplicate_linear_variables(objective),
            objective.linear_terms,
            lambda term: term.variable,
            lambda key: key,
        ),
        (
            "quadratic",
            _duplicate_quadratic_pairs(objective),
            objective.quadratic_terms,
            lambda term: frozenset((term.variable1, term.variable2)),
            frozenset,
        ),
    )
    for kind, duplicates, terms, key_of, lookup in groups:
        if not duplicates:
            continue
        wanted = {lookup(key) for key, _ in duplicates}
        positions: dict[object, list[int]] = {}
        for position, term in enumerate(terms):
            key = key_of(term)
            if key in wanted:
                positions.setdefault(key, []).append(position)
        explicit = expansion.objective_explicit(kind)
        aggregated: dict[str, list] = {}
        for key, count in duplicates:
            warning = _duplicate_term_warning(kind, key, count)
            repeats = [p for p in positions[lookup(key)][1:] if p >= explicit]
            source = expansion.objective_source(kind, repeats[0]) if repeats else None
            if source is None:
                result.append(warning)
                continue
            if source in aggregated:
                aggregated[source][1] += 1
                continue
            aggregated[source] = [len(result), 1]
            result.append(
                warning.model_copy(
                    update={
                        "path": source,
                        "message": f"{warning.message} (generated by {source})",
                    }
                )
            )
        for source, (at, count) in aggregated.items():
            if count > 1:
                result[at] = _with_note(
                    result[at],
                    f"; {count} generated entries from this template get this warning",
                )
    _log_warnings(result)
    return result


def _constraint_paths(problem: OptimizationProblem) -> list[tuple[str, Constraint]]:
    """``all_constraints()`` paired with the path each one was written at.

    ``constraints[i]`` for a linear constraint, ``cardinality_constraints[i]``
    for a lowered cardinality one (schema 1.2 spec §6.1), in
    ``all_constraints()`` order. For a problem without cardinality
    constraints this is exactly the old ``enumerate(problem.constraints)``
    paths, so every advisory keeps its path.
    """
    return [
        (f"constraints[{index}]", constraint)
        for index, constraint in enumerate(problem.constraints)
    ] + [
        (f"cardinality_constraints[{index}]", cardinality.lowered())
        for index, cardinality in enumerate(problem.cardinality_constraints)
    ]


def _count_phrase(count: int) -> str:
    """``"counts n variables, so between 0 and n are chosen"`` (spec §6.5)."""
    noun = "variable" if count == 1 else "variables"
    return f"counts {count} {noun}, so between 0 and {count} are chosen"


def _warn_soft_weights(
    paths: list[tuple[str, Constraint]], objective_scale: float, warnings: list[SolveError]
) -> None:
    threshold = objective_scale * SOFT_WEIGHT_SMALL_RATIO
    for base, constraint in paths:
        if constraint.type != "soft" or constraint.weight is None:
            continue
        if constraint.weight < threshold:
            warnings.append(
                _warning(
                    "SOFT_WEIGHT_SMALL",
                    f"{base}.weight",
                    (
                        f"Soft constraint {constraint.id} has weight "
                        f"{constraint.weight}, below {SOFT_WEIGHT_SMALL_RATIO} "
                        f"of the objective scale {objective_scale}; it will "
                        "barely influence solutions"
                    ),
                )
            )


def _warn_inequalities(
    paths: list[tuple[str, Constraint]],
    problem_analysis: ProblemAnalysis,
    warnings: list[SolveError],
) -> None:
    """``paths`` and ``problem_analysis`` list the same constraints, in order."""
    bounds = problem_analysis.bounds
    for (base, constraint), analysis in zip(paths, problem_analysis.inequalities, strict=True):
        if analysis is None:  # an equality
            continue
        is_cardinality = isinstance(constraint, LoweredCardinalityConstraint)
        if analysis.redundant:
            if is_cardinality:
                message = (
                    f"Cardinality constraint {constraint.id} "
                    f"{_count_phrase(len(constraint.terms))}, and requires "
                    f"{constraint.operator} {int(constraint.rhs)}, which always holds"
                )
            else:
                message = (
                    f"Inequality constraint {constraint.id} is always "
                    f"satisfied: lhs maximum {analysis.lhs_max} never "
                    f"exceeds the bound"
                )
            warnings.append(_warning("REDUNDANT_CONSTRAINT", base, message))
            continue
        slack_bits = count_slack_bits(constraint, bounds, analysis=analysis)
        if slack_bits > LARGE_SLACK_BITS_THRESHOLD:
            noun = "Cardinality" if is_cardinality else "Inequality"
            warnings.append(
                _warning(
                    "LARGE_SLACK_RANGE",
                    base,
                    (
                        f"{noun} constraint {constraint.id} would need "
                        f"{slack_bits} slack bits on a BQM backend (more than "
                        f"{LARGE_SLACK_BITS_THRESHOLD}); the compiled model "
                        "grows accordingly"
                    ),
                )
            )


def _warn_constraint_ranges(
    paths: list[tuple[str, Constraint]], bounds: Bounds, warnings: list[SolveError]
) -> None:
    """Range verdicts the error pass leaves to advice (review F-04 / F-12).

    A *soft* constraint no assignment within the bounds can satisfy is a
    legal problem: its weight is simply always paid and both compilers write
    the squared minimal violation (clamping any slack to zero). The agent
    must still hear about it, so it gets ``SOFT_ALWAYS_VIOLATED``, decided by
    the same :func:`_never_satisfiable` as ``TRIVIALLY_INFEASIBLE`` -- the
    zero-coefficient case (``x:+1, x:-1 == 5``) included, which is what the
    CQM compiler's constant branch judges with the same tolerance.

    A *soft* equality whose coefficients cancel to zero and whose rhs is
    zero within tolerance is always satisfied; it gets
    ``REDUNDANT_CONSTRAINT`` like the always-true inequalities of
    :func:`_warn_inequalities` (which already covers the zero-coefficient
    inequality with ``0 <= rhs``). The hard equality of the same shape stays
    silent as before: the Phase 3a golden recording
    (``tests/golden/phase3a_compile.json``) pins the full validator output of
    a 1.0 problem containing one, and that recording is a contract.
    """
    for path, constraint in paths:
        if constraint.type != "soft":
            continue  # both verdicts below are about soft constraints only
        coefficients = accumulate_terms(constraint.terms)
        lhs_min, lhs_max = lhs_bounds(coefficients, bounds)
        constant = not any(value != 0.0 for value in coefficients.values())
        rhs = constraint.rhs

        if constraint.type == "soft" and _never_satisfiable(
            constraint.operator, lhs_min, lhs_max, rhs
        ):
            if isinstance(constraint, LoweredCardinalityConstraint):
                # Schema 1.2 spec §6.5: the count, not the lhs range.
                warnings.append(
                    _warning(
                        "SOFT_ALWAYS_VIOLATED",
                        path,
                        (
                            f"Soft cardinality constraint {constraint.id} can "
                            f"never be satisfied: it {_count_phrase(len(constraint.terms))}, "
                            f"but requires {constraint.operator} {int(rhs)}; "
                            f"every solution pays weight {constraint.weight}"
                        ),
                    )
                )
                continue
            if constant:
                detail = (
                    "its coefficients sum to zero for every variable, so the "
                    "lhs is always 0"
                )
            else:
                detail = (
                    "lhs range (after summing repeated variables) is "
                    f"[{lhs_min}, {lhs_max}]"
                )
            warnings.append(
                _warning(
                    "SOFT_ALWAYS_VIOLATED",
                    path,
                    (
                        f"Soft constraint {constraint.id} can never be satisfied: "
                        f"{detail} but requires {constraint.operator} {rhs}; "
                        f"every solution pays weight {constraint.weight}"
                    ),
                )
            )
        elif (
            constraint.type == "soft"
            and constant
            and constraint.operator == "=="
            and satisfies("==", 0.0, rhs)
        ):
            warnings.append(
                _warning(
                    "REDUNDANT_CONSTRAINT",
                    path,
                    (
                        f"Equality constraint {constraint.id} is always satisfied: "
                        f"its coefficients sum to zero for every variable and "
                        f"0 == {rhs} holds"
                    ),
                )
            )


def _warn_cardinality_form(
    problem: OptimizationProblem, model_type: ModelType, warnings: list[SolveError]
) -> None:
    """Schema 1.2 spec §6.3: a linear at-most-one that could be declared.

    Only a declared cardinality constraint gets the slack-free pairwise
    encoding (never a linear one: that would change how a ``1.0`` document
    compiles), so a hard ``"<=" 1`` over two or more distinct binary
    variables, every coefficient exactly 1, is pointed at the declaration.
    From version 1.2 on only (older documents cannot declare it, and their
    advisories must not change) and on the BQM path only (the CQM path
    compiles both forms alike).
    """
    if problem.version in _VERSIONS_WITHOUT_CARDINALITY or model_type != "bqm":
        return
    binary = {variable.name for variable in problem.variables if variable.type == "binary"}
    for index, constraint in enumerate(problem.constraints):
        if (
            constraint.type != "hard"
            or constraint.operator != "<="
            or constraint.rhs != 1
        ):
            continue
        names = [term.variable for term in constraint.terms]
        if (
            len(names) < 2
            or len(set(names)) != len(names)
            or any(term.coefficient != 1 for term in constraint.terms)
            or any(name not in binary for name in names)
        ):
            continue
        warnings.append(
            _warning(
                "CARDINALITY_FORM_AVAILABLE",
                f"constraints[{index}]",
                (
                    f"Hard constraint {constraint.id} is an at-most-one over "
                    f"{len(names)} binary variables; declared in "
                    'cardinality_constraints (operator "<=", rhs 1) it compiles '
                    "on a BQM backend without a slack variable"
                ),
            )
        )


def _warn_fractional_equalities(
    paths: list[tuple[str, Constraint]],
    model_type: ModelType,
    capabilities: SolverCapabilities | None,
    warnings: list[SolveError],
) -> None:
    """2026-11-18 review R-01: a hard equality a BQM penalty may not enforce.

    The hard penalty is derived for violations of at least one unit
    (``compute_penalty_scale``). A hard ``==`` whose coefficients and rhs
    are not all integers can be violated by less
    (:func:`equality_violation_step_below_one`), and then the compiled
    model's minimum may sit on an infeasible assignment: a heuristic backend
    returns no feasible sample and the solve ends ``infeasible``.

    Advisory only; nothing about compilation changes. On the BQM path only
    (the CQM path keeps hard constraints native), and not for an exhaustive
    backend, which enumerates every assignment and is re-validated against
    the original problem whatever the penalty is. A cardinality constraint
    has unit coefficients and an integer rhs, so it never qualifies.
    """
    if model_type != "bqm":
        return
    if capabilities is not None and capabilities.exhaustive:
        return
    for path, constraint in paths:
        if constraint.type != "hard" or constraint.operator != "==":
            continue
        if not equality_violation_step_below_one(constraint):
            continue
        warnings.append(
            _warning(
                "FRACTIONAL_HARD_EQUALITY",
                path,
                (
                    f"Hard equality constraint {constraint.id} has a "
                    "non-integer coefficient or rhs, so it can be violated by "
                    "less than 1; on a BQM backend the hard penalty is derived "
                    "for violations of at least one unit and may be too weak "
                    "to rule such a violation out, in which case no feasible "
                    "solution is found"
                ),
            )
        )


def _warn_integer_encoding(
    problem: OptimizationProblem,
    bounds: Bounds,
    model_type: ModelType,
    encoded_interactions: Callable[[], int],
    warnings: list[SolveError],
) -> None:
    """3b §9.3: cost of binary-encoding integer variables on the BQM path.

    A CQM path takes integer variables natively, so neither warning applies
    there. ``encoded_interactions`` gives ``estimate_encoded_interactions``
    of the problem; it is only called when there is an integer variable.
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
    interactions = encoded_interactions()
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


def _warn_exact_limits(
    estimated: int,
    caps: SolverCapabilities,
    max_compiled_variables: int | None,
    warnings: list[SolveError],
) -> None:
    """3a §9.3: size advice for an exhaustive backend."""
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


def _warn_embedding_density(
    estimated: int,
    largest_constraint_bits: Callable[[], int],
    caps: SolverCapabilities,
    warnings: list[SolveError],
) -> None:
    """3a §9.3 / 3b §9.3: density advice for a backend that needs minor-embedding.

    ``largest_constraint_bits`` gives the largest ``constraint_bit_count``
    over the problem's constraints; it is only called for such a backend.
    """
    if caps.requires_embedding:
        # A squared penalty forms its clique over compiled *bits* (3b §9.3):
        # one per binary variable, the encoding bits of an integer one, plus
        # the constraint's slack bits.
        max_constraint_variables = largest_constraint_bits()
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


def _ignored_parameters(
    solver: SolverPreferences, caps: SolverCapabilities, model_type: ModelType
) -> list[tuple[str, str]]:
    """Non-default preferences the backend or the model path will ignore.

    ``(field, reason)`` pairs, in the order the warnings are reported.
    """
    ignored: list[tuple[str, str]] = []
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
    # Batch 4 (G), postprocess spec §2-§3: post-processing is skipped on an
    # exhaustive backend (it already has every assignment), and its sample
    # count means nothing while it is off. One warning per cause: on an
    # exhaustive backend the count is not warned about separately.
    if solver.postprocess != "none" and caps.exhaustive:
        ignored.append(
            (
                "postprocess",
                "is exhaustive and already returns every assignment, so there "
                "is nothing to repair or improve",
            )
        )
    elif not _is_default(solver, "postprocess_candidates") and (
        solver.postprocess == "none"
    ):
        ignored.append(
            (
                "postprocess_candidates",
                "runs no post-processing while solver.postprocess is none",
            )
        )
    return ignored


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

    for field, reason in _ignored_parameters(solver, caps, model_type):
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
        warnings.append(_duplicate_term_warning("linear", variable, count))
    for pair, count in _duplicate_quadratic_pairs(objective):
        warnings.append(_duplicate_term_warning("quadratic", pair, count))
    _log_warnings(warnings[first_new:])


def _duplicate_term_warning(kind: str, key, count: int) -> SolveError:
    """One DUPLICATE_TERM_MERGED warning (a variable, or a sorted pair)."""
    if kind == "linear":
        return _warning(
            "DUPLICATE_TERM_MERGED",
            "objective.linear_terms",
            (
                f"Variable {key} appears {count} times in "
                "objective.linear_terms; coefficients will be summed by "
                "the compiler"
            ),
        )
    return _warning(
        "DUPLICATE_TERM_MERGED",
        "objective.quadratic_terms",
        (
            f"Variable pair {key} appears {count} times in "
            "objective.quadratic_terms; coefficients will be summed "
            "by the compiler"
        ),
    )


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


def _check_feature_versions(
    problem: OptimizationProblem, errors: list[SolveError]
) -> None:
    """Schema 1.2 spec §6.4: a feature is only legal from its own version on.

    ``cardinality_constraints`` needs ``"1.2"`` or later. The integer rule
    of ``"1.1"`` keeps its own code (:func:`_check_version`) so a ``1.0``
    document's output is unchanged; when both apply, the message says that
    the one version fixes both, so an agent can correct it in one step.
    """
    if not problem.cardinality_constraints:
        return
    if problem.version not in _VERSIONS_WITHOUT_CARDINALITY:
        return
    notes: list[str] = []
    if "version" not in problem.model_fields_set:
        notes.append('version defaults to "1.0" when omitted')
    if problem.version == "1.0" and any(
        variable.type == "integer" for variable in problem.variables
    ):
        notes.append('"1.2" also allows the integer variables')
    message = (
        'cardinality_constraints requires version "1.2" or later, but version '
        f"is {problem.version!r}"
    )
    if notes:
        message += "; " + "; ".join(notes)
    errors.append(
        _error(code="FEATURE_REQUIRES_NEWER_VERSION", path="version", message=message)
    )


def _check_constraint_ids(
    problem: OptimizationProblem, errors: list[SolveError]
) -> None:
    """Every constraint id is unique across both constraint lists.

    Reads the ids only (never ``lowered()``), so it holds for any document
    the schema accepted (schema 1.2 spec §6.2).
    """
    entries = [
        (f"constraints[{index}]", constraint.id)
        for index, constraint in enumerate(problem.constraints)
    ] + [
        (f"cardinality_constraints[{index}]", cardinality.id)
        for index, cardinality in enumerate(problem.cardinality_constraints)
    ]
    seen: set[str] = set()
    for path, constraint_id in entries:
        if constraint_id in seen:
            errors.append(
                _error(
                    code="DUPLICATE_CONSTRAINT_ID",
                    path=path,
                    message=f"Constraint id {constraint_id} is used more than once",
                )
            )
        seen.add(constraint_id)


def _check_cardinality_constraint(
    cardinality: CardinalityConstraint,
    index: int,
    variable_types: dict[str, str],
    errors: list[SolveError],
) -> None:
    """Schema 1.2 spec §6.2: a cardinality constraint in its own terms.

    The paths point into ``cardinality_constraints`` and the messages talk
    about counting chosen variables; the recommended actions are the
    catalog's, shared with linear constraints. The range verdict runs only
    on a declaration without structural errors, where exactly
    ``len(variables)`` distinct binary variables are counted and the count
    ranges over ``[0, n]``; it is :func:`_never_satisfiable`, the judgement
    ``TRIVIALLY_INFEASIBLE`` applies to linear constraints, over the same
    range the lowered linear constraint has, so it agrees with
    ``validate_solution``.
    """
    base = f"cardinality_constraints[{index}]"
    errors_before = len(errors)

    if not cardinality.variables:
        errors.append(
            _error(
                code="EMPTY_CONSTRAINT",
                path=base,
                message=f"Cardinality constraint {cardinality.id} lists no variables",
            )
        )

    seen: set[str] = set()
    for position, name in enumerate(cardinality.variables):
        path = f"{base}.variables[{position}]"
        variable_type = variable_types.get(name)
        if variable_type is None:
            errors.append(
                _error(
                    code="UNKNOWN_VARIABLE",
                    path=path,
                    message=(
                        f"Cardinality constraint {cardinality.id} counts variable "
                        f"{name!r}, which is not declared in variables"
                    ),
                )
            )
        elif variable_type != "binary":
            errors.append(
                _error(
                    code="CARDINALITY_VARIABLE_NOT_BINARY",
                    path=path,
                    message=(
                        f"Cardinality constraint {cardinality.id} counts {name!r}, "
                        f"which is an {variable_type} variable; only binary "
                        "variables can be counted"
                    ),
                )
            )
        if name in seen:
            errors.append(
                _error(
                    code="DUPLICATE_CARDINALITY_VARIABLE",
                    path=path,
                    message=(
                        f"Cardinality constraint {cardinality.id} lists variable "
                        f"{name!r} more than once; each variable is counted once"
                    ),
                )
            )
        seen.add(name)

    if cardinality.type == "hard":
        if cardinality.weight is not None:
            errors.append(
                _error(
                    code="HARD_CONSTRAINT_HAS_WEIGHT",
                    path=f"{base}.weight",
                    message=(
                        f"Hard constraint {cardinality.id} must not carry a weight; "
                        "hard penalty strength is chosen by the penalty strategy"
                    ),
                )
            )
    else:
        if cardinality.weight is None or cardinality.weight <= 0:
            errors.append(
                _error(
                    code="SOFT_CONSTRAINT_MISSING_WEIGHT",
                    path=f"{base}.weight",
                    message=(
                        f"Soft constraint {cardinality.id} requires a weight > 0, "
                        f"got {cardinality.weight}"
                    ),
                )
            )
        elif not math.isfinite(cardinality.weight):
            errors.append(
                _non_finite(f"weight {cardinality.weight}", f"{base}.weight")
            )

    if len(errors) != errors_before or cardinality.type != "hard":
        return
    count = len(cardinality.variables)
    if _never_satisfiable(cardinality.operator, 0.0, float(count), float(cardinality.rhs)):
        errors.append(
            _error(
                code="TRIVIALLY_INFEASIBLE",
                path=base,
                message=(
                    f"Hard cardinality constraint {cardinality.id} "
                    f"{_count_phrase(count)}, but requires "
                    f"{cardinality.operator} {cardinality.rhs}"
                ),
            )
        )


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

    # The range judgement below trusts float64 arithmetic on the lhs; it is
    # skipped for a constraint whose magnitude puts that arithmetic outside
    # the exact range, the same way it is skipped for illegal bounds.
    if _check_inequality_magnitude(constraint, base, safe_bounds, errors):
        _check_trivially_infeasible(constraint, base, safe_bounds, errors)


def _check_objective_range(objective_scale: float, errors: list[SolveError]) -> None:
    """2026-11-18 review R-11: finite coefficients whose range is not finite.

    Every coefficient passed the checks above, yet their magnitudes can
    still add up beyond float64 (two coefficients of ``1e308``). Then
    ``objective_scale`` is infinite and objective values of different
    assignments can no longer be compared, on any compiler path and
    whatever the backend, so this belongs to the error pass. Before this
    check the service reported such a problem as PENALTY_OVERFLOW after
    zero attempts, even without a single hard constraint.

    ``objective_scale`` is ``compute_objective_scale`` over the variables'
    declared ranges, which must be legal: the caller computes it only when
    the pass is otherwise clean, and keeps it for the advisory layer.
    """
    if math.isfinite(objective_scale):
        return
    errors.append(
        _error(
            code="COEFFICIENT_MAGNITUDE_TOO_LARGE",
            path="objective",
            message=(
                "The objective's coefficients are too large: the sum of "
                "their magnitudes over the variables' ranges exceeds the "
                "floating-point range, so objective values cannot be "
                "compared"
            ),
        )
    )


def _check_soft_energy_range(
    objective_scale: float,
    soft_energy_total: Callable[[], float],
    model_type: ModelType,
    errors: list[SolveError],
) -> None:
    """2026-11-18 review R-11: the BQM path's penalty scale must be finite.

    ``objective_scale`` (finite: :func:`_check_objective_range` passed) plus
    the soft energy bound is the penalty scale of ``compute_penalty_scale``;
    when it overflows no hard penalty can be derived from it, and the
    service used to report PENALTY_OVERFLOW after zero attempts although no
    penalty ladder was involved. Depends on the compiler path, like the
    advisories beside it: the CQM path derives no hard penalty, and its
    compiler reports an overflowing soft term itself. ``soft_energy_total``
    gives ``compute_soft_energy_total`` of the problem; it is only called on
    the BQM path.
    """
    if model_type != "bqm":
        return
    if math.isfinite(objective_scale + soft_energy_total()):
        return
    errors.append(
        _error(
            code="COEFFICIENT_MAGNITUDE_TOO_LARGE",
            path="constraints",
            message=(
                "The soft constraints' weights and coefficients are too "
                "large: the bound on their total penalty exceeds the "
                "floating-point range, so the model cannot be scaled on a "
                "BQM backend"
            ),
        )
    )


def _check_inequality_magnitude(
    constraint: Constraint,
    base: str,
    safe_bounds: dict[str, tuple[int, int] | None],
    errors: list[SolveError],
) -> bool:
    """2026-09-09 review F-24: keep the slack arithmetic inside float64 exactness.

    For a ``<=`` / ``>=`` constraint the compiler sizes its slack from
    ``rhs - lhs_min`` computed in float64 (``analyze_inequality``); that
    value is exact only while every ``coefficient * bound`` product and
    every partial sum stays within ±2^53. The per-term sum
    ``sum(|c| * max(|lower|, |upper|)) + |rhs|`` bounds all of them at
    once (it is at least the accumulated-coefficient sum the estimates
    use), so a constraint within it is encoded exactly and one above it is
    rejected with INEQUALITY_MAGNITUDE_TOO_LARGE. Non-finite values and
    variables with illegal bounds are skipped: each already carries its
    own error. Returns False when the constraint was rejected, so the
    caller can skip the range-based judgement that would rely on the same
    inexact arithmetic.
    """
    if constraint.operator not in ("<=", ">=") or not constraint.terms:
        return True
    # Only integer-valued, finite numbers are summed (anything else already
    # carries NON_FINITE_COEFFICIENT / NON_INTEGER_INEQUALITY), and they are
    # summed as Python integers so the comparison with the limit is itself
    # exact right at the boundary.
    values = [term.coefficient for term in constraint.terms] + [constraint.rhs]
    if not all(math.isfinite(value) and value.is_integer() for value in values):
        return True
    if any(safe_bounds.get(term.variable, (0, 1)) is None for term in constraint.terms):
        return True

    magnitude = abs(int(constraint.rhs))
    for term in constraint.terms:
        lower, upper = safe_bounds.get(term.variable) or (0, 1)
        magnitude += abs(int(term.coefficient)) * max(abs(lower), abs(upper))
    if magnitude <= INEQUALITY_MAGNITUDE_LIMIT:
        return True
    errors.append(
        _error(
            code="INEQUALITY_MAGNITUDE_TOO_LARGE",
            path=base,
            message=(
                f"Inequality constraint {constraint.id} is too large to encode "
                f"exactly: sum(|coefficient| * max|bound|) + |rhs| is "
                f"{magnitude}, above the 2^53 limit "
                f"({INEQUALITY_MAGNITUDE_LIMIT}) within which the slack range "
                f"is computed exactly"
            ),
        )
    )
    return False


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

    The comparison is :func:`_never_satisfiable`, shared with the soft
    ``SOFT_ALWAYS_VIOLATED`` warning: what is rejected here is exactly what
    ``validate_solution`` would reject for every assignment, and what passes
    here has at least one lhs value it would accept.
    """
    if constraint.type != "hard" or not constraint.terms:
        return
    if not all(math.isfinite(term.coefficient) for term in constraint.terms):
        return
    if not math.isfinite(constraint.rhs):
        return
    if any(safe_bounds.get(term.variable, (0, 1)) is None for term in constraint.terms):
        return

    # Only this constraint's variables: ``lhs_bounds`` reads no other, and a
    # dict over every variable would make the error pass O(variables x
    # constraints), which a schema 1.3 template reaches from a few lines
    # (2026-09-28 batch 8 diff review). Same entries, so the same result.
    bounds: Bounds = {}
    for term in constraint.terms:
        declared = safe_bounds.get(term.variable)
        if declared is not None:
            bounds[term.variable] = declared
    lhs_min, lhs_max = lhs_bounds(accumulate_terms(constraint.terms), bounds)
    rhs = constraint.rhs

    if _never_satisfiable(constraint.operator, lhs_min, lhs_max, rhs):
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


def _never_satisfiable(operator: str, lhs_min: float, lhs_max: float, rhs: float) -> bool:
    """Whether no lhs value in ``[lhs_min, lhs_max]`` passes the §23.1 test.

    The one judgement behind ``TRIVIALLY_INFEASIBLE`` (hard, an error) and
    ``SOFT_ALWAYS_VIOLATED`` (soft, a warning), so the two can never drift.
    It uses the solution validator's hybrid tolerance (review F-25):
    ``actual - tol(actual)`` and ``actual + tol(actual)`` are monotone in
    ``actual`` (the relative part is 1e-12), so the most favourable end of
    the range decides -- ``lhs_min`` for ``<=``, ``lhs_max`` for ``>=`` --
    and for ``==`` the rhs must fall outside the range widened by the
    tolerance at each end. With ``lhs_min == lhs_max == 0`` (no non-zero
    coefficient) this is exactly ``not satisfies(operator, 0.0, rhs)``, the
    CQM compiler's constant-constraint test.
    """
    if operator == "<=":
        return not satisfies("<=", lhs_min, rhs)
    if operator == ">=":
        return not satisfies(">=", lhs_max, rhs)
    return rhs < lhs_min - tolerance(lhs_min, rhs) or rhs > lhs_max + tolerance(lhs_max, rhs)


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
            "postprocess_candidates",
            solver.postprocess_candidates,
            solver.postprocess_candidates <= 0,
            "must be > 0",
        ),
        # Finiteness re-checked here (F-08): the schema already rejects
        # NaN / inf, but a model built without validation must not slip a
        # ``nan`` penalty through (``nan <= 0`` is False).
        (
            "penalty_multiplier",
            solver.penalty_multiplier,
            not math.isfinite(solver.penalty_multiplier)
            or solver.penalty_multiplier <= 0,
            "must be a finite number > 0",
        ),
    ]
    # Batch 6 (J): the wall-clock limit, when set. Finiteness re-checked
    # for the same reason as penalty_multiplier.
    limit = solver.wall_clock_limit_seconds
    if limit is not None:
        checks.append(
            (
                "wall_clock_limit_seconds",
                limit,
                not math.isfinite(limit) or limit <= 0,
                "must be a finite number > 0",
            )
        )
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


def _check_seed_range(
    problem: OptimizationProblem, caps: SolverCapabilities, errors: list[SolveError]
) -> None:
    """A ``solver.seed`` outside the range the backend declares.

    The range is the backend's own limit, read from ``seed_min`` /
    ``seed_max`` on its declaration, never from its name; refused rather
    than left for the backend to reject mid-solve. A backend that declares
    no range is not checked, and one that ignores seeds is not checked
    either: it gets the SEED_IGNORED warning for any value instead.
    """
    seed = problem.solver.seed
    if seed is None or not caps.supports_seed:
        return
    if caps.seed_min is None or caps.seed_max is None:
        return
    if caps.seed_min <= seed <= caps.seed_max:
        return
    errors.append(
        _error(
            code="INVALID_SOLVER_PREFERENCE",
            path="solver.seed",
            message=(
                f"solver.seed must be an integer between {caps.seed_min} and "
                f"{caps.seed_max} on backend {caps.name}, got {seed}"
            ),
        )
    )


def _check_wall_clock_limit(
    problem: OptimizationProblem, caps: SolverCapabilities, errors: list[SolveError]
) -> None:
    """A ``solver.wall_clock_limit_seconds`` the backend cannot honour.

    Read from ``supports_interrupt`` on the declaration, never from the
    backend's name. Refused rather than warned about: ignoring the limit
    would let the solve run past a ceiling the caller set, silently.
    """
    if problem.solver.wall_clock_limit_seconds is None or caps.supports_interrupt:
        return
    errors.append(
        _error(
            code="WALL_CLOCK_LIMIT_UNSUPPORTED",
            path="solver.wall_clock_limit_seconds",
            message=(
                f"Backend {caps.name} cannot stop part-way through a solve, so "
                f"it cannot honour solver.wall_clock_limit_seconds"
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
