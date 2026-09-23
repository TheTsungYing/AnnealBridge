"""BQM compiler: OptimizationProblem -> dimod.BinaryQuadraticModel (spec §15)."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dimod
import numpy as np

from annealbridge.compiler.base import select_business_columns
from annealbridge.compiler.integer_encoding import (
    AffineForm,
    Linear,
    Quadratic,
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
    IntegerEncoding,
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

    def copy(self) -> "_BiasAccumulator":
        """An independent accumulator in the same state.

        The containers are new and the floats immutable, so adding to the
        copy never touches this one (the prepared snapshot of
        :class:`PreparedBQM`).
        """
        other = _BiasAccumulator()
        other._order = list(self._order)
        other._index = dict(self._index)
        other._linear = list(self._linear)
        other._quadratic = dict(self._quadratic)
        other.offset = self.offset
        return other

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
    _add_expansion(model, *expand_square(coefficients, constant, lam))


def _add_expansion(
    model: _BiasAccumulator, linear: Linear, quadratic: Quadratic, offset: float
) -> None:
    """Add one :func:`expand_square` result to ``model``.

    Linear entries first, then the pairs, then the offset: the order the
    penalty term has always been added in. A soft constraint's cached
    expansion (:class:`_ConstraintStep`) is replayed through here, so it
    lands exactly as a fresh one would.
    """
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


@dataclass(frozen=True)
class _ConstraintStep:
    """One constraint's penalty term with everything but the hard penalty settled.

    ``coefficients`` and ``constant`` are the squared term over compiled
    variables, slack bits already merged in; ``slack_variables`` are
    registered right before the term is added, as a fresh compile does. A
    soft constraint's term does not depend on the hard penalty, so it is
    expanded once with the constraint's own weight (``soft_weight``) into
    ``soft_expansion``; a hard one is expanded per compile. A redundant
    inequality adds nothing. Read only: every :meth:`PreparedBQM.compile`
    replays the same steps.
    """

    constraint: Constraint
    slack_variables: tuple[str, ...]
    coefficients: Mapping[str, float]
    constant: float
    slack_range: int | None
    redundant: bool
    soft_weight: float | None
    soft_expansion: tuple[Linear, Quadratic, float] | None


def _replay_step(
    model: _BiasAccumulator,
    step: _ConstraintStep,
    hard_penalty: float,
    internal_variables: set[str],
) -> ConstraintTrace:
    """Add ``step``'s penalty term to ``model`` and return its trace."""
    constraint = step.constraint
    # §10.4: hard penalty and soft weight come from different sources and
    # must never substitute for each other.
    if constraint.type == "hard":
        lam = hard_penalty
    else:
        assert step.soft_weight is not None  # set by _prepare_constraint
        lam = step.soft_weight
    if not step.redundant:
        for name in step.slack_variables:
            model.add_variable(name)
        internal_variables.update(step.slack_variables)
        if step.soft_expansion is None:
            _add_squared_penalty(model, step.coefficients, step.constant, lam)
        else:
            _add_expansion(model, *step.soft_expansion)
    return ConstraintTrace(
        constraint_id=constraint.id,
        constraint_type=constraint.type,
        operator=constraint.operator,
        source_description=constraint.description,
        generated_variables=list(step.slack_variables),
        penalty=lam,
        slack_range=step.slack_range,
        redundant=step.redundant,
        compiler=_COMPILER_NAME,
    )


@dataclass(frozen=True)
class PreparedBQM:
    """:meth:`BQMCompiler.prepare`'s result: a compile short of the hard penalty.

    ``base`` holds every business variable (or encoding bit) registered and
    the objective accumulated; ``steps`` holds one :class:`_ConstraintStep`
    per constraint. :meth:`compile` copies ``base`` and replays the steps in
    constraint order, so every bias is the same chain of float additions a
    from-scratch compile makes, and nothing held here ever changes: any
    number of compiles, at any penalties, in any order, each equal bit for
    bit to ``BQMCompiler().compile(problem, hard_penalty)`` (variable order,
    biases and offset). Each compiled problem gets its own copy of the
    integer encodings, as a fresh compile would.
    """

    problem: OptimizationProblem
    base: _BiasAccumulator
    encoding_bits: frozenset[str]
    integer_encodings: dict[str, IntegerEncoding]
    objective_scale: float
    steps: tuple[_ConstraintStep, ...]

    def compile(self, hard_penalty: float | None) -> CompiledProblem:
        """Finish the compile with ``hard_penalty`` as the hard-constraint lambda."""
        if hard_penalty is None:
            raise CompilationError("BQMCompiler requires a hard_penalty; got None")
        model = self.base.copy()
        internal_variables = set(self.encoding_bits)
        constraint_trace = [
            _replay_step(model, step, hard_penalty, internal_variables)
            for step in self.steps
        ]
        bqm = model.to_bqm()
        # 2026-09-09 review (F-07): the penalty arithmetic is plain float
        # multiplication, which overflows to ``inf`` silently; a model with
        # a non-finite bias must never reach a backend.
        _check_finite(bqm, self.problem, hard_penalty)

        compiled = CompiledProblem(
            model_type="bqm",
            model=bqm,
            original_problem=self.problem,
            internal_variables=internal_variables,
            constraint_trace=constraint_trace,
            hard_penalty=hard_penalty,
            objective_scale=self.objective_scale,
            num_variables=bqm.num_variables,
            num_interactions=bqm.num_interactions,
            integer_encodings={
                name: encoding.model_copy(deep=True)
                for name, encoding in self.integer_encodings.items()
            },
        )
        logger.info(
            "Compiled problem %s: %d variables (%d internal, %d integer encoded), "
            "hard_penalty=%s",
            self.problem.name,
            compiled.num_variables,
            len(internal_variables),
            len(self.integer_encodings),
            hard_penalty,
        )
        return compiled


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
        return self.prepare(problem).compile(hard_penalty)

    def prepare(self, problem: OptimizationProblem) -> PreparedBQM:
        """Every step of :meth:`compile` that does not depend on the hard penalty.

        Integer encodings, variable registration, the objective, each
        constraint's substitution and slack encoding and every soft
        penalty run here, once and in :meth:`compile`'s order (so a problem
        error surfaces exactly as it would there); the returned
        :class:`PreparedBQM` finishes the compile for any hard penalty.
        :meth:`compile` itself is ``prepare(problem).compile(hard_penalty)``,
        so there is one code path.
        """
        bounds = variable_bounds(problem)
        forms, encodings = encode_integer_variables(problem)
        # The biases accumulate in Python and dimod builds the model once at
        # the end (batch 4); the registration order below is the model's
        # variable order.
        model = _BiasAccumulator()
        encoding_bits: set[str] = set()
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
            encoding_bits.update(encoding.bits)

        self._compile_objective(model, problem.objective, forms)

        steps = tuple(
            self._prepare_constraint(constraint, bounds, forms)
            for constraint in problem.constraints
        )
        return PreparedBQM(
            problem=problem,
            base=model,
            encoding_bits=frozenset(encoding_bits),
            integer_encodings=encodings,
            objective_scale=compute_objective_scale(problem.objective, bounds),
            steps=steps,
        )

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

    def _prepare_constraint(
        self,
        constraint: Constraint,
        bounds: Bounds,
        forms: Mapping[str, AffineForm],
    ) -> _ConstraintStep:
        # §10.4: a soft constraint's lambda is its own weight, never the
        # hard penalty; a hard constraint's is supplied per compile.
        soft_weight: float | None = None
        if constraint.type != "hard":
            if constraint.weight is None:
                raise CompilationError(
                    f"Soft constraint {constraint.id} has no weight"
                )
            soft_weight = constraint.weight

        slack_variables: tuple[str, ...] = ()
        slack_range: int | None = None
        redundant = False
        coefficients: dict[str, float] = {}
        constant = 0.0

        # 3b §14.1 / §14.3: the business coefficients are rewritten through
        # the affine forms, which turns an integer variable into its bits
        # and moves ``sum(c * lower)`` into the penalty's constant. Identity
        # forms (binary variables) leave both exactly as they were.
        if constraint.operator == "==":
            coefficients, shift = substitute_linear(
                nonzero_coefficients(constraint.terms), forms
            )
            constant = shift - constraint.rhs
        else:
            # ``encode_slack`` sizes the slack from the variables' bounds via
            # the same ``analyze_inequality`` the estimates use, so the
            # range ``rhs - lhs_min`` already covers the integer lhs.
            encoding = encode_slack(constraint, bounds)
            redundant = encoding.redundant
            slack_range = encoding.slack_range
            if not redundant:
                slack_variables = tuple(encoding.slack_coefficients)
                coefficients, shift = substitute_linear(encoding.coefficients, forms)
                for name, value in encoding.slack_coefficients.items():
                    coefficients[name] = coefficients.get(name, 0.0) + float(value)
                constant = encoding.constant + shift

        # The soft term is fixed by the constraint's own weight, so its
        # expansion is done once here; a hard term waits for the penalty.
        soft_expansion = None
        if soft_weight is not None and not redundant:
            soft_expansion = expand_square(coefficients, constant, soft_weight)
        return _ConstraintStep(
            constraint=constraint,
            slack_variables=slack_variables,
            coefficients=coefficients,
            constant=constant,
            slack_range=slack_range,
            redundant=redundant,
            soft_weight=soft_weight,
            soft_expansion=soft_expansion,
        )
