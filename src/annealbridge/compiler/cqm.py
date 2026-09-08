"""CQM compiler: OptimizationProblem -> dimod.ConstrainedQuadraticModel (3a spec §15).

Hard constraints become native CQM constraints (``weight=None``, i.e. must
be satisfied), so no penalty lambda, no generated variables and no penalty
strategy are involved: the model expresses feasibility itself. Soft
constraints become dimod *soft* constraints with
``weight=constraint.weight, penalty="quadratic"``, whose energy contribution
is ``weight * violation**2`` — the same formula ``solution_validator`` uses
for ``weighted_penalty``, so the preference strength the solver sees and the
score the ranking uses are one thing (overview principle 3; proven by the
§21.2 test). The objective is built by the shared
:func:`build_objective_bqm` (maximize negates everything, constant included),
so ``energy == sign * objective + sum(weight * violation**2)`` holds for
every sample.

Nothing here degrades the model to a BQM (that conversion is a test-only
cross-check, never production code) and the compiler takes no options
(§15.2).
"""

import logging

import dimod

from annealbridge.compiler.objective import build_objective_bqm
from annealbridge.exceptions import CompilationError
from annealbridge.models import (
    CompiledProblem,
    Constraint,
    ConstraintTrace,
    ModelType,
    OptimizationProblem,
)
from annealbridge.penalty.strategy import compute_objective_scale
from annealbridge.validation.estimates import accumulate_terms, variable_bounds

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

    Variables and constraints are added in problem order, so the output is
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
        cqm = dimod.ConstrainedQuadraticModel()
        for variable in problem.variables:
            cqm.add_variable("BINARY", variable.name)

        cqm.set_objective(build_objective_bqm(problem.objective))

        constraint_trace = [
            self._compile_constraint(cqm, constraint) for constraint in problem.constraints
        ]

        compiled = CompiledProblem(
            model_type=self.model_type,
            model=cqm,
            original_problem=problem,
            internal_variables=set(),
            constraint_trace=constraint_trace,
            hard_penalty=None,
            objective_scale=compute_objective_scale(problem.objective, bounds),
            # dimod 0.12.22: ``ConstrainedQuadraticModel.num_variables`` is a
            # method, not a property.
            num_variables=len(cqm.variables),
        )
        logger.info(
            "Compiled problem %s as CQM: %d variables, %d constraints (%d soft)",
            problem.name,
            compiled.num_variables,
            len(cqm.constraints),
            sum(1 for constraint in problem.constraints if constraint.type == "soft"),
        )
        return compiled

    def _compile_constraint(
        self, cqm: dimod.ConstrainedQuadraticModel, constraint: Constraint
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

        redundant = False
        if not coefficients:
            # A constant constraint: ``0 <op> rhs``. The validator's
            # TRIVIALLY_INFEASIBLE check should already have rejected the
            # failing case; this is the consistency defence (COMPILATION_FAILED).
            if not _constant_constraint_holds(constraint.operator, constraint.rhs):
                raise CompilationError(
                    f"Constraint {constraint.id} has no non-zero coefficients and "
                    f"0 {constraint.operator} {constraint.rhs} does not hold"
                )
            redundant = True
        else:
            lhs = dimod.BinaryQuadraticModel(coefficients, {}, 0.0, "BINARY")
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

        return ConstraintTrace(
            constraint_id=constraint.id,
            constraint_type=constraint.type,
            operator=constraint.operator,
            source_description=constraint.description,
            generated_variables=[],
            penalty=None if constraint.type == "hard" else constraint.weight,
            slack_range=None,
            redundant=redundant,
            native=True,
            compiler=_COMPILER_NAME,
        )
