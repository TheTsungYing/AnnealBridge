"""BQM compiler: OptimizationProblem -> dimod.BinaryQuadraticModel (spec §15)."""

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import dimod
import numpy as np

from annealbridge.compiler.base import select_business_columns
from annealbridge.compiler.integer_encoding import (
    AffineForm,
    encode_integer_variables,
    substitute_linear,
)
from annealbridge.compiler.objective import build_objective_bqm
from annealbridge.compiler.slack import accumulate_terms, encode_slack
from annealbridge.exceptions import CompilationError, NonFiniteModelError
from annealbridge.models import (
    CompiledProblem,
    Constraint,
    ConstraintTrace,
    ModelType,
    Objective,
    OptimizationProblem,
)
from annealbridge.penalty.strategy import compute_objective_scale
from annealbridge.validation.estimates import Bounds, variable_bounds

if TYPE_CHECKING:
    from annealbridge.solvers.base import RawSolverResult

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


def _check_finite(
    bqm: dimod.BinaryQuadraticModel, problem: OptimizationProblem, hard_penalty: float
) -> None:
    """Raise :class:`NonFiniteModelError` if any bias of ``bqm`` is not finite."""
    finite = (
        math.isfinite(bqm.offset)
        and all(math.isfinite(bias) for bias in bqm.linear.values())
        and all(math.isfinite(bias) for bias in bqm.quadratic.values())
    )
    if not finite:
        raise NonFiniteModelError(
            f"Compiled model of problem {problem.name} has a non-finite bias "
            f"at hard_penalty={hard_penalty!r}: the penalty or coefficient "
            f"arithmetic overflowed the floating-point range"
        )


class BQMCompiler:
    """Compiles an :class:`OptimizationProblem` into a binary quadratic model.

    Maximization objectives are converted to minimization energy by negating
    every objective coefficient including the constant, so that
    ``energy == sign * objective + sum(penalties)`` holds exactly.
    The input problem is never mutated.

    Integer variables (IR 1.1) are binary-expanded into ``__int_<name>_<k>``
    bits (3b §14): every objective and constraint expression is rewritten
    through the variables' affine forms before it enters the model, and
    :meth:`decode` folds the bits back into integer values. A binary-only
    problem has identity forms and compiles exactly as before.
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
        bounds = variable_bounds(problem)
        forms, encodings = encode_integer_variables(problem)
        bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
        internal_variables: set[str] = set()
        for variable in problem.variables:
            # A binary variable is registered under its own name (the 3a
            # order and naming, which the golden test pins); an integer
            # variable contributes its encoding bits in ``k`` order instead.
            encoding = encodings.get(variable.name)
            if encoding is None:
                bqm.add_variable(variable.name)
                continue
            for bit in encoding.bits:
                bqm.add_variable(bit)
            internal_variables.update(encoding.bits)

        self._compile_objective(bqm, problem.objective, forms)

        constraint_trace = [
            self._compile_constraint(
                bqm, constraint, hard_penalty, internal_variables, bounds, forms
            )
            for constraint in problem.constraints
        ]
        # 2026-09-09 review (F-07): the penalty arithmetic is plain float
        # multiplication, which overflows to ``inf`` silently; a model with
        # a non-finite bias must never reach a backend.
        _check_finite(bqm, problem, hard_penalty)

        compiled = CompiledProblem(
            model_type=self.model_type,
            model=bqm,
            original_problem=problem,
            internal_variables=internal_variables,
            constraint_trace=constraint_trace,
            hard_penalty=hard_penalty,
            objective_scale=compute_objective_scale(problem.objective, bounds),
            num_variables=bqm.num_variables,
            integer_encodings=encodings,
        )
        logger.info(
            "Compiled problem %s: %d variables (%d internal, %d integer encoded), "
            "hard_penalty=%s",
            problem.name,
            compiled.num_variables,
            len(internal_variables),
            len(encodings),
            hard_penalty,
        )
        return compiled

    def decode(
        self, compiled: CompiledProblem, raw: "RawSolverResult"
    ) -> "RawSolverResult":
        """Business-variable view of a BQM backend's bit matrix (3b §13).

        Without integer variables this is pure column selection and keeps
        the backend's ``int8`` bit matrix. With integer variables the
        result is ``int64``: each integer column is
        ``lower + bits @ coefficients`` over its encoding bits, binary
        columns are copied, and the bit and slack columns are dropped. The
        encoding guarantees every value lies within the variable's bounds,
        so nothing is clamped. ``energies``, the row order, ``backend`` and
        ``metadata`` pass through untouched.
        """
        encodings = compiled.integer_encodings
        if not encodings:
            return select_business_columns(compiled, raw)

        column = {name: index for index, name in enumerate(raw.variables)}

        def column_of(name: str, kind: str) -> int:
            try:
                return column[name]
            except KeyError:
                raise ValueError(
                    f"solver result from backend '{raw.backend}' lacks {kind} {name!r}"
                ) from None

        names = [variable.name for variable in compiled.original_problem.variables]
        samples = np.empty((raw.num_samples, len(names)), dtype=np.int64)
        for position, name in enumerate(names):
            encoding = encodings.get(name)
            if encoding is None:
                samples[:, position] = raw.samples[:, column_of(name, "business variable")]
                continue
            bit_columns = [column_of(bit, "encoding bit") for bit in encoding.bits]
            bits = raw.samples[:, bit_columns].astype(np.int64, copy=False)
            coefficients = np.asarray(encoding.coefficients, dtype=np.int64)
            samples[:, position] = encoding.lower + bits @ coefficients
        return raw.model_copy(update={"variables": names, "samples": samples})

    def _compile_objective(
        self,
        bqm: dimod.BinaryQuadraticModel,
        objective: Objective,
        forms: Mapping[str, AffineForm],
    ) -> None:
        # Shared with the CQM compiler (3a §14). ``update`` adds the other
        # model's biases into the pre-registered variables exactly like the
        # former inline ``add_*`` loop did, so the output is unchanged.
        bqm.update(build_objective_bqm(objective, forms))

    def _compile_constraint(
        self,
        bqm: dimod.BinaryQuadraticModel,
        constraint: Constraint,
        hard_penalty: float,
        internal_variables: set[str],
        bounds: Bounds,
        forms: Mapping[str, AffineForm],
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

        # 3b §14.1 / §14.3: the business coefficients are rewritten through
        # the affine forms, which turns an integer variable into its bits
        # and moves ``sum(c * lower)`` into the penalty's constant. Identity
        # forms (binary variables) leave both exactly as they were.
        if constraint.operator == "==":
            coefficients = {
                variable: value
                for variable, value in accumulate_terms(constraint.terms).items()
                if value != 0.0
            }
            bit_coefficients, shift = substitute_linear(coefficients, forms)
            _add_squared_penalty(bqm, bit_coefficients, shift - constraint.rhs, lam)
        else:
            # ``encode_slack`` sizes the slack from the variables' bounds via
            # the same ``analyze_inequality`` the estimates use, so the
            # range ``rhs - lhs_min`` already covers the integer lhs.
            encoding = encode_slack(constraint, bounds)
            redundant = encoding.redundant
            slack_range = encoding.slack_range
            if not redundant:
                generated_variables = list(encoding.slack_coefficients)
                for name in generated_variables:
                    bqm.add_variable(name)
                internal_variables.update(generated_variables)
                coefficients, shift = substitute_linear(encoding.coefficients, forms)
                for name, value in encoding.slack_coefficients.items():
                    coefficients[name] = coefficients.get(name, 0.0) + float(value)
                _add_squared_penalty(bqm, coefficients, encoding.constant + shift, lam)

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
