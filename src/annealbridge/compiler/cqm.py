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

The objective is built by :func:`build_objective_qm` (maximize negates
everything, constant included) and the squared penalties are expanded by
:func:`expand_square_qm`, which shares its expansion core with the BQM
path. Nothing here degrades the model to a BQM (that conversion is a
test-only cross-check, never production code) and the compiler takes no
options (§15.2).
"""

import logging
from typing import TYPE_CHECKING

import dimod

from annealbridge.compiler.base import select_business_columns
from annealbridge.compiler.integer_encoding import expand_square_qm
from annealbridge.compiler.objective import add_model_variable, build_objective_qm
from annealbridge.exceptions import CompilationError
from annealbridge.models import (
    CompiledProblem,
    Constraint,
    ConstraintTrace,
    ModelType,
    OptimizationProblem,
    Variable,
)
from annealbridge.penalty.strategy import compute_objective_scale
from annealbridge.validation.estimates import (
    Bounds,
    accumulate_terms,
    analyze_inequality,
    variable_bounds,
)

if TYPE_CHECKING:
    from annealbridge.solvers.base import RawSolverResult

logger = logging.getLogger(__name__)

_COMPILER_NAME = "CQMCompiler"


def _constant_constraint_holds(operator: str, rhs: float) -> bool:
    """Whether ``0 <operator> rhs`` holds for a constraint with no variables."""
    if operator == "==":
        return rhs == 0.0
    if operator == "<=":
        return 0.0 <= rhs
    return 0.0 >= rhs


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

        coefficients = {
            variable: value
            for variable, value in accumulate_terms(constraint.terms).items()
            if value != 0.0
        }

        if not coefficients:
            # A constant constraint: ``0 <op> rhs``. The validator's
            # TRIVIALLY_INFEASIBLE check should already have rejected the
            # failing case; this is the consistency defence (COMPILATION_FAILED).
            if not _constant_constraint_holds(constraint.operator, constraint.rhs):
                raise CompilationError(
                    f"Constraint {constraint.id} has no non-zero coefficients and "
                    f"0 {constraint.operator} {constraint.rhs} does not hold"
                )
            return self._trace(constraint, redundant=True, native=True)

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
