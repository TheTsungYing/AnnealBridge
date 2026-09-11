"""BQM compiler: OptimizationProblem -> dimod.BinaryQuadraticModel (spec §15)."""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import dimod
import numpy as np

from annealbridge.compiler.base import select_business_columns
from annealbridge.compiler.integer_encoding import (
    AffineForm,
    encode_integer_variables,
    expand_square,
    substitute_linear,
)
from annealbridge.compiler.objective import bqm_has_finite_biases, build_objective_bqm
from annealbridge.compiler.slack import encode_slack, nonzero_coefficients
from annealbridge.exceptions import CompilationError, NonFiniteModelError
from annealbridge.models import (
    CompiledProblem,
    Constraint,
    ConstraintTrace,
    ModelType,
    Objective,
    OptimizationProblem,
)
from annealbridge.validation.estimates import (
    Bounds,
    compute_objective_scale,
    variable_bounds,
)

if TYPE_CHECKING:
    from annealbridge.solvers.base import RawSolverResult

logger = logging.getLogger(__name__)

_COMPILER_NAME = "BQMCompiler"


class _BiasAccumulator:
    """Linear, quadratic and offset contributions gathered before dimod builds the model.

    The penalty terms used to go into a live ``dimod.BinaryQuadraticModel``
    one ``add_linear`` / ``add_quadratic`` call at a time, about a
    microsecond each through the cython layer, which was most of a compile
    on a model with tens of thousands of interactions (2026-09-11 batch 4
    performance fix). This accumulator keeps the very same arithmetic:
    every bias is the same chain of float additions in the very order the
    ``add_*`` calls ran (the objective first, then each constraint's
    penalty in constraint order), and :meth:`to_bqm` hands the finished
    vectors to ``from_numpy_vectors``, which only copies them. A pair is
    keyed by variable index without orientation, exactly as dimod keys an
    interaction, so a contribution to ``(v, u)`` lands on the ``(u, v)``
    entry; the first contribution sets the entry (dimod inserts a new
    interaction's bias as given) and every later one adds. A variable is
    registered with linear bias ``0.0``, as ``add_variable`` does, and the
    registration order is the model's variable order, which the 3a golden
    test pins.
    """

    def __init__(self) -> None:
        self._order: list[str] = []
        self._index: dict[str, int] = {}
        self._linear: list[float] = []
        self._quadratic: dict[tuple[int, int], float] = {}
        self.offset: float = 0.0

    def add_variable(self, name: str) -> None:
        if name in self._index:
            return
        self._index[name] = len(self._order)
        self._order.append(name)
        self._linear.append(0.0)

    def add_linear(self, name: str, bias: float) -> None:
        self._linear[self._index[name]] += bias

    def add_quadratic(self, u: str, v: str, bias: float) -> None:
        i = self._index[u]
        j = self._index[v]
        if i == j:
            raise ValueError(f"{u!r} cannot have an interaction with itself")
        key = (i, j) if i < j else (j, i)
        if key in self._quadratic:
            self._quadratic[key] += bias
        else:
            self._quadratic[key] = bias

    def to_bqm(self) -> dimod.BinaryQuadraticModel:
        count = len(self._quadratic)
        pairs = np.fromiter(
            self._quadratic.keys(), dtype=np.dtype((np.int64, 2)), count=count
        ).reshape(count, 2)
        biases = np.fromiter(self._quadratic.values(), dtype=np.float64, count=count)
        return dimod.BinaryQuadraticModel.from_numpy_vectors(
            np.array(self._linear, dtype=np.float64),
            (
                np.ascontiguousarray(pairs[:, 0]),
                np.ascontiguousarray(pairs[:, 1]),
                biases,
            ),
            self.offset,
            "BINARY",
            variable_order=self._order,
        )


def _add_squared_penalty(
    model: _BiasAccumulator,
    coefficients: dict[str, float],
    constant: float,
    lam: float,
) -> None:
    """Add ``lam * (sum(c_i * y_i) + constant)^2`` to ``model``.

    A thin adapter over :func:`expand_square` (2026-09-09 review F-13a: the
    same expansion the CQM path's ``expand_square_qm`` uses). Every
    compiled variable is a bit here, so ``y^2 == y`` and the whole diagonal
    folds into the linear bias. All contributions accumulate (dimod
    ``add_*`` semantics), never overwrite.
    """
    linear, quadratic, offset = expand_square(coefficients, constant, lam)
    for variable, value in linear.items():
        model.add_linear(variable, value)
    for (var_i, var_j), value in quadratic.items():
        model.add_quadratic(var_i, var_j, value)
    model.offset += offset


def _check_finite(
    bqm: dimod.BinaryQuadraticModel, problem: OptimizationProblem, hard_penalty: float
) -> None:
    """Raise :class:`NonFiniteModelError` if any bias of ``bqm`` is not finite.

    The test itself is :func:`bqm_has_finite_biases`, the vectorised form of
    the very predicate the CQM path runs as :func:`has_finite_biases`
    (review F-06); the wording stays here because the hard penalty is this
    path's own.
    """
    if not bqm_has_finite_biases(bqm):
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
        # The biases accumulate in Python and dimod builds the model once at
        # the end (batch 4); the registration order below is the model's
        # variable order.
        model = _BiasAccumulator()
        internal_variables: set[str] = set()
        for variable in problem.variables:
            # A binary variable is registered under its own name (the 3a
            # order and naming, which the golden test pins); an integer
            # variable contributes its encoding bits in ``k`` order instead.
            encoding = encodings.get(variable.name)
            if encoding is None:
                model.add_variable(variable.name)
                continue
            for bit in encoding.bits:
                model.add_variable(bit)
            internal_variables.update(encoding.bits)

        self._compile_objective(model, problem.objective, forms)

        constraint_trace = [
            self._compile_constraint(
                model, constraint, hard_penalty, internal_variables, bounds, forms
            )
            for constraint in problem.constraints
        ]
        bqm = model.to_bqm()
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
            num_interactions=bqm.num_interactions,
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
        model: _BiasAccumulator,
        objective: Objective,
        forms: Mapping[str, AffineForm],
    ) -> None:
        # Shared with the CQM compiler (3a §14). The objective model's biases
        # are added into the pre-registered variables exactly like ``update``
        # did (``0.0 + bias`` per linear entry, one entry per pair, the
        # offset added once), so the output is unchanged.
        objective_bqm = build_objective_bqm(objective, forms)
        for variable, bias in objective_bqm.linear.items():
            model.add_linear(variable, float(bias))
        for u, v, bias in objective_bqm.iter_quadratic():
            model.add_quadratic(u, v, float(bias))
        model.offset += float(objective_bqm.offset)

    def _compile_constraint(
        self,
        model: _BiasAccumulator,
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
            coefficients = nonzero_coefficients(constraint.terms)
            bit_coefficients, shift = substitute_linear(coefficients, forms)
            _add_squared_penalty(model, bit_coefficients, shift - constraint.rhs, lam)
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
                    model.add_variable(name)
                internal_variables.update(generated_variables)
                coefficients, shift = substitute_linear(encoding.coefficients, forms)
                for name, value in encoding.slack_coefficients.items():
                    coefficients[name] = coefficients.get(name, 0.0) + float(value)
                _add_squared_penalty(model, coefficients, encoding.constant + shift, lam)

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
