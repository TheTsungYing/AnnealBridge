"""BQM compiler: OptimizationProblem -> dimod.BinaryQuadraticModel (spec §15)."""

import logging

import dimod

from annealbridge.compiler.objective import build_objective_bqm
from annealbridge.compiler.slack import accumulate_terms, encode_slack
from annealbridge.exceptions import CompilationError
from annealbridge.models import (
    CompiledProblem,
    Constraint,
    ConstraintTrace,
    ModelType,
    Objective,
    OptimizationProblem,
)
from annealbridge.penalty.strategy import compute_objective_scale

logger = logging.getLogger(__name__)

_COMPILER_NAME = "BQMCompiler"


def _add_squared_penalty(
    bqm: dimod.BinaryQuadraticModel,
    coefficients: dict[str, float],
    constant: float,
    lam: float,
) -> None:
    """Add ``lam * (sum(c_i * y_i) + constant)^2`` to ``bqm``.

    For binary variables ``y^2 == y``, so the diagonal of the expansion folds
    into the linear bias. All contributions accumulate (dimod ``add_*``
    semantics), never overwrite.
    """
    items = list(coefficients.items())
    for variable, value in items:
        bqm.add_linear(variable, lam * (value * value + 2.0 * constant * value))
    for i, (var_i, value_i) in enumerate(items):
        for var_j, value_j in items[i + 1 :]:
            bqm.add_quadratic(var_i, var_j, 2.0 * lam * value_i * value_j)
    bqm.offset += lam * constant * constant


class BQMCompiler:
    """Compiles an :class:`OptimizationProblem` into a binary quadratic model.

    Maximization objectives are converted to minimization energy by negating
    every objective coefficient including the constant, so that
    ``energy == sign * objective + sum(penalties)`` holds exactly.
    The input problem is never mutated.
    """

    @property
    def model_type(self) -> ModelType:
        return "bqm"

    @property
    def uses_hard_penalty(self) -> bool:
        return True

    def compile(
        self,
        problem: OptimizationProblem,
        hard_penalty: float | None,
    ) -> CompiledProblem:
        """Compile ``problem``; hard constraints use ``hard_penalty`` as lambda.

        ``hard_penalty`` must be a float: the BQM compiler expresses hard
        constraints as penalty terms, so ``None`` is a caller (service)
        error, not a problem error.
        """
        if hard_penalty is None:
            raise CompilationError("BQMCompiler requires a hard_penalty; got None")
        bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
        for variable in problem.variables:
            bqm.add_variable(variable.name)

        self._compile_objective(bqm, problem.objective)

        internal_variables: set[str] = set()
        constraint_trace = [
            self._compile_constraint(bqm, constraint, hard_penalty, internal_variables)
            for constraint in problem.constraints
        ]

        compiled = CompiledProblem(
            model_type=self.model_type,
            model=bqm,
            original_problem=problem,
            internal_variables=internal_variables,
            constraint_trace=constraint_trace,
            hard_penalty=hard_penalty,
            objective_scale=compute_objective_scale(problem.objective),
            num_variables=bqm.num_variables,
        )
        logger.info(
            "Compiled problem %s: %d variables (%d internal), hard_penalty=%s",
            problem.name,
            compiled.num_variables,
            len(internal_variables),
            hard_penalty,
        )
        return compiled

    def _compile_objective(
        self, bqm: dimod.BinaryQuadraticModel, objective: Objective
    ) -> None:
        # Shared with the CQM compiler (3a §14). ``update`` adds the other
        # model's biases into the pre-registered variables exactly like the
        # former inline ``add_*`` loop did, so the output is unchanged.
        bqm.update(build_objective_bqm(objective))

    def _compile_constraint(
        self,
        bqm: dimod.BinaryQuadraticModel,
        constraint: Constraint,
        hard_penalty: float,
        internal_variables: set[str],
    ) -> ConstraintTrace:
        # §10.4: hard penalty and soft weight come from different sources and
        # must never substitute for each other.
        if constraint.type == "hard":
            lam = hard_penalty
        else:
            if constraint.weight is None:
                raise CompilationError(
                    f"Soft constraint {constraint.id} has no weight"
                )
            lam = constraint.weight

        generated_variables: list[str] = []
        slack_range: int | None = None
        redundant = False

        if constraint.operator == "==":
            coefficients = {
                variable: value
                for variable, value in accumulate_terms(constraint.terms).items()
                if value != 0.0
            }
            _add_squared_penalty(bqm, coefficients, -constraint.rhs, lam)
        else:
            encoding = encode_slack(constraint)
            redundant = encoding.redundant
            slack_range = encoding.slack_range
            if not redundant:
                generated_variables = list(encoding.slack_coefficients)
                for name in generated_variables:
                    bqm.add_variable(name)
                internal_variables.update(generated_variables)
                coefficients = dict(encoding.coefficients)
                for name, value in encoding.slack_coefficients.items():
                    coefficients[name] = coefficients.get(name, 0.0) + float(value)
                _add_squared_penalty(bqm, coefficients, encoding.constant, lam)

        return ConstraintTrace(
            constraint_id=constraint.id,
            constraint_type=constraint.type,
            operator=constraint.operator,
            source_description=constraint.description,
            generated_variables=generated_variables,
            penalty=lam,
            slack_range=slack_range,
            redundant=redundant,
            compiler=_COMPILER_NAME,
        )
