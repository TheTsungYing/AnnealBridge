"""CQM compiler: OptimizationProblem -> dimod.ConstrainedQuadraticModel (3a spec §15; 3b §15).

Every declared variable is a native CQM variable: binary ones ``BINARY``,
integer ones ``INTEGER`` with their own bounds (3b §15.1), so there is no
binary expansion and no ``integer_encodings`` on this path. Hard
constraints become native CQM constraints (``weight=None``, i.e. must be
satisfied) whatever their variables, so no penalty lambda, no slack, no
generated variables and no penalty strategy are involved: the model
expresses feasibility itself.

Soft constraints come in two writings (3b §15.3), decided *before* the
constraint is added because dimod records a constraint before rejecting
its penalty type:

* **all-binary**: the 3a native form ``weight=w, penalty="quadratic"``,
  whose energy contribution is ``w * violation**2`` — the same formula
  ``solution_validator`` uses for ``weighted_penalty``, so the preference
  strength the solver sees and the score the ranking uses are one thing
  (overview principle 3; proven by the §21.2 test);
* **at least one integer variable**: dimod only allows ``penalty="linear"``
  (``w * |violation|``) there, which is *not* the validator's formula, so
  the penalty is written into the objective instead: ``w * (lhs - rhs)**2``
  for an equality, and for an inequality (normalised to ``<=`` by
  ``analyze_inequality``) ``w * (lhs + s - rhs)**2`` with one internal
  ``INTEGER`` slack ``__slack_<id>`` in ``[0, S]`` when ``S > 0``, no slack
  when the constraint is always violated (``S <= 0``, clamped with a
  warning like ``encode_slack``) and nothing at all when it is redundant.
  Minimising over the slack gives back ``w * max(0, violation)**2``, so
  the identity ``min_s energy == sign * objective + sum(w * violation**2)``
  holds for every business assignment (proven by
  ``test_cqm_compiler_integer.py``).

A constraint whose accumulated coefficients are all zero is a constant
``0 <op> rhs``, judged with the validator's tolerance (:func:`satisfies`).
When it holds it is redundant and adds nothing; a *hard* one that fails is
a ``CompilationError`` (the validator's ``TRIVIALLY_INFEASIBLE`` rejects it
first, this is the last defence); a *soft* one that fails is a legal
problem (review F-04) and becomes the constant penalty ``w * rhs**2`` in
the objective, exactly what the BQM path puts in its offset, so the same
problem never compiles on one path and fails on the other.

The objective is built by :func:`build_objective_qm` (maximize negates
everything, constant included) and the squared penalties are expanded by
:func:`expand_square_qm`, a thin adapter over the same ``expand_square``
the BQM penalty terms use (review F-13a). Both expansions are plain float
arithmetic, so the finished model is checked for non-finite biases before
it leaves ``compile`` (:func:`_check_finite`, review F-06), exactly as the
BQM path does. Nothing here degrades the model to a BQM (that conversion is a
test-only cross-check, never production code) and the compiler takes no
options (§15.2).
"""

import logging
import math
from typing import TYPE_CHECKING

import dimod

from annealbridge.compiler.base import select_business_columns
from annealbridge.compiler.integer_encoding import expand_square_qm
from annealbridge.compiler.objective import (
    add_model_variable,
    build_objective_qm,
    has_finite_biases,
)
from annealbridge.exceptions import CompilationError, NonFiniteModelError
from annealbridge.models import (
    CompiledProblem,
    Constraint,
    ConstraintTrace,
    ModelType,
    OptimizationProblem,
    Variable,
)
from annealbridge.validation.estimates import (
    Bounds,
    analyze_inequality,
    compute_objective_scale,
    nonzero_coefficients,
    variable_bounds,
)
from annealbridge.validation.tolerance import satisfies

if TYPE_CHECKING:
    from annealbridge.solvers.base import RawSolverResult

logger = logging.getLogger(__name__)

_COMPILER_NAME = "CQMCompiler"


def _constant_constraint_holds(operator: str, rhs: float) -> bool:
    """Whether ``0 <operator> rhs`` holds for a constraint with no variables.

    Judged with the §23.1 tolerance, the same test the problem validator
    applies to the constant lhs range ``[0, 0]`` (review F-04): an exact
    ``rhs == 0.0`` used to refuse ``rhs = 1e-9`` that the validator accepts.
    """
    return satisfies(operator, 0.0, rhs)


def _check_finite(
    cqm: dimod.ConstrainedQuadraticModel, problem: OptimizationProblem
) -> None:
    """Raise :class:`NonFiniteModelError` unless every bias of ``cqm`` is finite.

    The CQM twin of the BQM path's guard (2026-09-11 review F06), sharing
    its :func:`has_finite_biases` test: a finite problem can still compile
    to a model holding ``inf``, either because an objective-form soft
    penalty expanded ``weight * coefficient**2`` past the float range
    (``expand_square_qm``) or because a constraint's accumulated lhs
    coefficients overflowed. Both the objective and every constraint (lhs
    biases and rhs) are checked, so the promise that no backend is ever
    called with a non-finite model holds on this path too.

    There is no hard penalty here (§10.4), and the soft weights themselves
    come from a validated problem, so nothing re-checks them: only what the
    compiler's own arithmetic produced.
    """
    finite = has_finite_biases(cqm.objective) and all(
        math.isfinite(comparison.rhs) and has_finite_biases(comparison.lhs)
        for comparison in cqm.constraints.values()
    )
    if not finite:
        raise NonFiniteModelError(
            f"Compiled model of problem {problem.name} has a non-finite bias: "
            f"the soft weight and coefficient arithmetic overflowed the "
            f"floating-point range while expanding the model"
        )


class CQMCompiler:
    """Compiles an :class:`OptimizationProblem` into a constrained quadratic model.

    Variables and constraints are added in problem order (an integer
    slack right after the soft constraint that needs it), so the output is
    deterministic (§15.3). The input problem is never mutated.
    """

    @property
    def model_type(self) -> ModelType:
        return "cqm"

    @property
    def uses_hard_penalty(self) -> bool:
        return False

    def compile(
        self,
        problem: OptimizationProblem,
        hard_penalty: float | None,
    ) -> CompiledProblem:
        """Compile ``problem``; ``hard_penalty`` must be ``None``.

        The CQM expresses hard constraints natively, so a penalty value is a
        caller (service) error, not a problem error.
        """
        if hard_penalty is not None:
            raise CompilationError(
                f"CQMCompiler does not use a hard_penalty; got {hard_penalty!r}"
            )
        bounds = variable_bounds(problem)
        declared = {variable.name: variable for variable in problem.variables}
        cqm = dimod.ConstrainedQuadraticModel()
        for variable in problem.variables:
            add_model_variable(cqm, variable)

        # Built first, set last: the objective-form soft constraints add
        # their squared penalties (and slack variables) to it on the way.
        objective = build_objective_qm(problem.objective, problem.variables)
        internal_variables: set[str] = set()
        constraint_trace = [
            self._compile_constraint(
                cqm, objective, constraint, declared, bounds, internal_variables
            )
            for constraint in problem.constraints
        ]
        cqm.set_objective(objective)
        # 2026-09-11 review (F06): the soft-penalty expansion and the
        # accumulated constraint coefficients are plain float arithmetic,
        # which overflows to ``inf`` silently; a model with a non-finite
        # bias must never reach a backend.
        _check_finite(cqm, problem)

        compiled = CompiledProblem(
            model_type=self.model_type,
            model=cqm,
            original_problem=problem,
            internal_variables=internal_variables,
            constraint_trace=constraint_trace,
            hard_penalty=None,
            objective_scale=compute_objective_scale(problem.objective, bounds),
            # dimod 0.12.22: ``ConstrainedQuadraticModel.num_variables`` is a
            # method, not a property.
            num_variables=cqm.num_variables(),
            num_interactions=cqm.objective.num_interactions
            + sum(view.lhs.num_interactions for view in cqm.constraints.values()),
        )
        logger.info(
            "Compiled problem %s as CQM: %d variables (%d internal), "
            "%d constraints (%d soft)",
            problem.name,
            compiled.num_variables,
            len(internal_variables),
            len(cqm.constraints),
            sum(1 for constraint in problem.constraints if constraint.type == "soft"),
        )
        return compiled

    def decode(
        self, compiled: CompiledProblem, raw: "RawSolverResult"
    ) -> "RawSolverResult":
        """Business-variable view of a CQM backend's result (3b §13).

        CQM backends already return integer values (int64); the decode drops
        the internal columns (the ``__slack_*`` of §15.3) and restores the
        problem's variable order, which ``ExactCQMSolver`` does not preserve.
        """
        return select_business_columns(compiled, raw)

    def _compile_constraint(
        self,
        cqm: dimod.ConstrainedQuadraticModel,
        objective: dimod.QuadraticModel,
        constraint: Constraint,
        declared: dict[str, Variable],
        bounds: Bounds,
        internal_variables: set[str],
    ) -> ConstraintTrace:
        # §10.4: the soft weight is the only lambda-like value here, and it
        # is the constraint's own; hard constraints carry no penalty at all.
        if constraint.type == "soft" and constraint.weight is None:
            raise CompilationError(f"Soft constraint {constraint.id} has no weight")

        coefficients = nonzero_coefficients(constraint.terms)

        if not coefficients:
            # A constant constraint: ``0 <op> rhs``, judged with the
            # validator's tolerance.
            if _constant_constraint_holds(constraint.operator, constraint.rhs):
                return self._trace(constraint, redundant=True, native=True)
            if constraint.type == "hard":
                # The validator's TRIVIALLY_INFEASIBLE rejects this with the
                # same test; this is the consistency defence (COMPILATION_FAILED).
                raise CompilationError(
                    f"Constraint {constraint.id} has no non-zero coefficients and "
                    f"0 {constraint.operator} {constraint.rhs} does not hold"
                )
            # Soft (review F-04): a legal problem whose weight is always paid.
            # The validator warns SOFT_ALWAYS_VIOLATED; the penalty is the
            # constant ``w * rhs**2`` in the objective, the BQM path's offset
            # term, and the trace mirrors its clamped inequality (slack 0).
            weight = constraint.weight
            assert weight is not None  # checked above
            logger.warning(
                "Soft constraint %s has no non-zero coefficients and 0 %s %s never "
                "holds; writing its constant penalty into the objective",
                constraint.id,
                constraint.operator,
                constraint.rhs,
            )
            self._add_squared(objective, {}, -constraint.rhs, weight, declared)
            return self._trace(
                constraint,
                native=False,
                slack_range=None if constraint.operator == "==" else 0,
            )

        # The writing is decided here, before anything touches the CQM:
        # dimod records a soft constraint and only then rejects
        # ``penalty="quadratic"`` over integer variables (3b §15.3).
        involves_integer = any(declared[name].type == "integer" for name in coefficients)
        if constraint.type == "hard" or not involves_integer:
            self._add_native(cqm, constraint, coefficients, declared)
            return self._trace(constraint, native=True)

        weight = constraint.weight
        assert weight is not None  # checked above
        if constraint.operator == "==":
            self._add_squared(objective, coefficients, -constraint.rhs, weight, declared)
            return self._trace(constraint, native=False)

        analysis = analyze_inequality(constraint, bounds)
        if analysis.redundant:
            # ``lhs_max <= rhs``: never violated, so no penalty at all
            # (same rule as ``encode_slack``).
            return self._trace(constraint, redundant=True, native=False)

        slack_range = analysis.slack_range
        assert slack_range is not None  # non-redundant analysis always sets it
        generated_variables: list[str] = []
        penalty_coefficients = dict(analysis.coefficients)
        if slack_range > 0:
            slack = f"__slack_{constraint.id}"
            cqm.add_variable("INTEGER", slack, lower_bound=0, upper_bound=slack_range)
            objective.add_variable("INTEGER", slack, lower_bound=0, upper_bound=slack_range)
            internal_variables.add(slack)
            generated_variables.append(slack)
            penalty_coefficients[slack] = 1.0
        elif slack_range < 0:
            logger.warning(
                "Soft constraint %s can never be satisfied (lhs range starts at %s, "
                "rhs %s); writing its penalty without a slack so it tracks the "
                "minimal violation",
                constraint.id,
                analysis.lhs_min,
                analysis.rhs,
            )
            slack_range = 0
        # ``slack_range == 0``: the lhs minimum already meets the rhs, so a
        # slack could only be 0 anyway; ``w * (lhs - rhs)**2`` is exact.
        self._add_squared(objective, penalty_coefficients, -analysis.rhs, weight, declared)
        return self._trace(
            constraint,
            native=False,
            generated_variables=generated_variables,
            slack_range=slack_range,
        )

    @staticmethod
    def _add_native(
        cqm: dimod.ConstrainedQuadraticModel,
        constraint: Constraint,
        coefficients: dict[str, float],
        declared: dict[str, Variable],
    ) -> None:
        """Add ``constraint`` as a CQM constraint with a QM lhs (3b §15.2)."""
        lhs = dimod.QuadraticModel()
        for name, value in coefficients.items():
            add_model_variable(lhs, declared[name])
            lhs.add_linear(name, value)
        if constraint.type == "hard":
            cqm.add_constraint_from_model(
                lhs,
                sense=constraint.operator,
                rhs=constraint.rhs,
                label=constraint.id,
                weight=None,
            )
        else:
            cqm.add_constraint_from_model(
                lhs,
                sense=constraint.operator,
                rhs=constraint.rhs,
                label=constraint.id,
                weight=constraint.weight,
                penalty="quadratic",
            )

    @staticmethod
    def _add_squared(
        objective: dimod.QuadraticModel,
        coefficients: dict[str, float],
        constant: float,
        weight: float,
        declared: dict[str, Variable],
    ) -> None:
        """Add ``weight * (sum(coefficients) + constant)**2`` to the objective QM."""
        for name in coefficients:
            # Business variables the objective did not mention yet; the
            # slack (not in ``declared``) is declared by the caller.
            if name not in objective.variables:
                add_model_variable(objective, declared[name])
        expand_square_qm(objective, coefficients, constant, weight)

    @staticmethod
    def _trace(
        constraint: Constraint,
        *,
        native: bool,
        redundant: bool = False,
        generated_variables: list[str] | None = None,
        slack_range: int | None = None,
    ) -> ConstraintTrace:
        return ConstraintTrace(
            constraint_id=constraint.id,
            constraint_type=constraint.type,
            operator=constraint.operator,
            source_description=constraint.description,
            generated_variables=generated_variables or [],
            penalty=None if constraint.type == "hard" else constraint.weight,
            slack_range=slack_range,
            redundant=redundant,
            native=native,
            compiler=_COMPILER_NAME,
        )
