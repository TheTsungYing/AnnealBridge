"""``ModelCompiler.decode`` and the service step that calls it (3b spec §13, §16).

Before 3b the orchestration layer stripped internal columns itself, using
``compiled.internal_variables`` and whatever column order the backend
happened to return. 3b moves that job into the compiler: step 12a of §16
inserts ``decoded = compiler.decode(compiled, raw)`` between the solve and
``process_candidates``, and §13 fixes what ``decode`` guarantees:

* the returned ``variables`` are exactly
  ``[v.name for v in compiled.original_problem.variables]`` -- the
  *problem's* order on both paths, never the backend's;
* internal columns (``__slack...``, encoding bits) are gone, so no key of a
  reported ``Solution`` starts with ``__``;
* ``energies``, the row order, ``backend`` and ``metadata`` pass through
  untouched, and the sample dtype is preserved for a binary-only problem;
* a result missing a business variable is a backend-contract violation
  (``ValueError``), not a silently shortened matrix.

The service-level half also pins the two counters §13 keeps distinct:
``samples_received`` still counts the backend's *raw* rows, while
``unique_samples`` counts distinct *business* assignments after decode --
so duplicate rows that differ only in their slack bits collapse.

§13 also records one deliberate observable change from 3a: on the CQM path
``Solution.variables`` used to follow ``ExactCQMSolver``'s alphabetical
order and now follows the problem's declaration order.
"""

import random

import numpy as np
import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    SolverExecutionMetadata,
    SolverPreferences,
    Variable,
)
from annealbridge.orchestration import OptimizationService, evaluate_objective
from annealbridge.solvers import RawSolverResult, SolverRegistry
from annealbridge.solvers.base import AvailabilityStatus, SolverCapabilities
from tests.fakes.local_cqm_backend import FAKE_LOCAL_CQM_NAME, FakeLocalCQMBackend

SHUFFLING_BQM_NAME = "fake_shuffling_bqm"


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------


def bqm_problem() -> OptimizationProblem:
    """Variables declared ``c, a, b`` -- deliberately not alphabetical.

    The ``<=`` hard constraint makes the BQM compiler emit slack bits, so
    the compiled model has internal columns for ``decode`` to drop.
    """
    return OptimizationProblem(
        name="decode-bqm",
        variables=[Variable(name="c"), Variable(name="a"), Variable(name="b")],
        objective=Objective(
            direction="maximize",
            linear_terms=[
                LinearTerm(variable="c", coefficient=3.0),
                LinearTerm(variable="a", coefficient=2.0),
                LinearTerm(variable="b", coefficient=1.0),
            ],
            constant=0.5,
        ),
        constraints=[
            Constraint(
                id="cap",
                type="hard",
                terms=[
                    LinearTerm(variable="c", coefficient=1.0),
                    LinearTerm(variable="a", coefficient=1.0),
                    LinearTerm(variable="b", coefficient=1.0),
                ],
                operator="<=",
                rhs=2,
            )
        ],
    )


def cqm_problem() -> OptimizationProblem:
    """Variables declared ``b, a, c`` -- ``ExactCQMSolver`` will re-sort them."""
    return OptimizationProblem(
        name="decode-cqm",
        variables=[Variable(name="b"), Variable(name="a"), Variable(name="c")],
        objective=Objective(
            direction="maximize",
            linear_terms=[
                LinearTerm(variable="b", coefficient=1.0),
                LinearTerm(variable="a", coefficient=3.0),
                LinearTerm(variable="c", coefficient=2.0),
            ],
        ),
        constraints=[
            Constraint(
                id="cap",
                type="hard",
                terms=[
                    LinearTerm(variable="b", coefficient=1.0),
                    LinearTerm(variable="a", coefficient=1.0),
                    LinearTerm(variable="c", coefficient=1.0),
                ],
                operator="<=",
                rhs=2,
            )
        ],
    )


def route_to(problem: OptimizationProblem, backend: str) -> OptimizationProblem:
    """Point ``problem`` at a test-only backend.

    ``SolverPreferences.backend`` is a Literal of the shipped names, so a
    fake is set through ``model_construct`` (defaults are still filled in),
    the same way ``test_service_cqm_flow.py`` does it.
    """
    return problem.model_copy(
        update={"solver": SolverPreferences.model_construct(backend=backend)}
    )


# --------------------------------------------------------------------------
# A BQM backend that returns its columns in an arbitrary order
# --------------------------------------------------------------------------

_SHUFFLING_CAPABILITIES = SolverCapabilities(
    name=SHUFFLING_BQM_NAME,
    remote=False,
    heuristic=True,
    exhaustive=False,
    supports_seed=False,
    supports_num_reads=True,
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    description=(
        "Test-only BQM backend returning its sample columns in a shuffled "
        "order (3b spec §13); never production."
    ),
)


class ShufflingBQMBackend:
    """Returns fixed business rows with the model columns shuffled.

    Nothing in the ``SolverBackend`` contract promises a column order, and
    ``decode`` must not depend on one. Each business row given to the
    constructor becomes one returned read; every compiled variable that is
    not a business variable (the ``__slack`` bits) is filled with a value
    that *varies per row*, so two reads that agree on the business
    assignment but differ in slack still count as one unique candidate
    after decode.
    """

    def __init__(self, rows: list[dict[str, int]], *, shuffle_seed: int = 1) -> None:
        self._rows = [dict(row) for row in rows]
        self._shuffle_seed = shuffle_seed
        self.solve_calls = 0
        self.last_raw: RawSolverResult | None = None
        self.last_compiled = None

    @property
    def capabilities(self) -> SolverCapabilities:
        return _SHUFFLING_CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        return AvailabilityStatus(category="available")

    @property
    def name(self) -> str:
        return _SHUFFLING_CAPABILITIES.name

    @property
    def is_exhaustive(self) -> bool:
        return False

    def resolve_time_limit(self, compiled_problem, preferences) -> float | None:
        return None

    def solve(self, compiled_problem, preferences) -> RawSolverResult:
        self.solve_calls += 1
        self.last_compiled = compiled_problem
        bqm = compiled_problem.model
        names = [str(variable) for variable in bqm.variables]
        order = list(names)
        random.Random(self._shuffle_seed).shuffle(order)
        business = {v.name for v in compiled_problem.original_problem.variables}
        samples = []
        for index, row in enumerate(self._rows):
            samples.append(
                {
                    name: int(row[name]) if name in business else index % 2
                    for name in order
                }
            )
        raw = RawSolverResult.from_dicts(
            samples,
            [float(bqm.energy(sample)) for sample in samples],
            backend=self.name,
            variables=order,
        )
        self.last_raw = raw
        return raw


def make_service(backend) -> OptimizationService:
    return OptimizationService(registry=SolverRegistry({backend.name: backend}))


# --------------------------------------------------------------------------
# 1. The service's BQM path
# --------------------------------------------------------------------------

# Six reads over four distinct business assignments, all feasible for
# "c + a + b <= 2"; rows 0/4 and 1/5 repeat, so unique < received.
BQM_ROWS = [
    {"c": 1, "a": 1, "b": 0},
    {"c": 1, "a": 0, "b": 0},
    {"c": 0, "a": 1, "b": 1},
    {"c": 0, "a": 0, "b": 0},
    {"c": 1, "a": 1, "b": 0},
    {"c": 1, "a": 0, "b": 0},
]


class TestServiceBQMPath:
    @pytest.fixture
    def backend(self) -> ShufflingBQMBackend:
        return ShufflingBQMBackend(BQM_ROWS)

    @pytest.fixture
    def result(self, backend):
        result = make_service(backend).solve(route_to(bqm_problem(), backend.name))
        assert result.status == "success", result.errors
        return result

    def test_the_backend_really_did_shuffle_and_include_internal_columns(
        self, backend, result
    ):
        # Guard on the fixture: without a shuffled, slack-carrying raw
        # result the assertions below would prove nothing.
        raw = backend.last_raw
        assert raw is not None
        assert raw.variables != ["c", "a", "b", "__slack_cap_0", "__slack_cap_1"]
        assert sorted(raw.variables) == sorted(
            ["c", "a", "b", "__slack_cap_0", "__slack_cap_1"]
        )
        internal = [
            index for index, name in enumerate(raw.variables) if name.startswith("__")
        ]
        business = [
            index
            for index, name in enumerate(raw.variables)
            if not name.startswith("__")
        ]
        # At least one internal column is sandwiched between business ones,
        # so decode cannot get away with trimming a prefix or a suffix.
        assert any(
            min(business) < index < max(business) for index in internal
        ), raw.variables
        # ... and the business columns themselves are not in problem order.
        assert [raw.variables[index] for index in business] != ["c", "a", "b"]

    def test_solutions_carry_business_variables_in_problem_order(self, result):
        problem = bqm_problem()
        expected = [variable.name for variable in problem.variables]
        assert expected == ["c", "a", "b"]
        for solution in result.solutions:
            assert list(solution.variables) == expected, solution.variables

    def test_no_internal_variable_reaches_a_solution(self, result):
        for solution in result.solutions:
            assert not [key for key in solution.variables if key.startswith("__")]
            assert set(solution.variables) == {"c", "a", "b"}

    def test_samples_received_counts_raw_rows_and_unique_counts_assignments(
        self, result
    ):
        attempt = result.attempts[0]
        assert attempt.samples_received == len(BQM_ROWS) == 6
        distinct = {tuple(sorted(row.items())) for row in BQM_ROWS}
        assert attempt.unique_samples == len(distinct) == 4
        assert attempt.feasible_samples == 4

    def test_values_and_objective_are_recomputed_from_the_original_problem(
        self, result
    ):
        problem = bqm_problem()
        for solution in result.solutions:
            assert solution.variables in BQM_ROWS
            assert solution.objective_value == evaluate_objective(
                problem.objective, solution.variables
            )
        best = result.solutions[0]
        # max 3c + 2a + b + 0.5 subject to c + a + b <= 2 over the rows given.
        assert best.variables == {"c": 1, "a": 1, "b": 0}
        assert best.objective_value == 5.5

    def test_one_solve_per_attempt_and_only_one_attempt(self, backend, result):
        assert backend.solve_calls == 1
        assert len(result.attempts) == 1


# --------------------------------------------------------------------------
# 2. The service's CQM path -- the 3a -> 3b observable change of §13
# --------------------------------------------------------------------------


class TestServiceCQMPath:
    @pytest.fixture
    def backend(self) -> FakeLocalCQMBackend:
        return FakeLocalCQMBackend()

    @pytest.fixture
    def result(self, backend):
        result = make_service(backend).solve(route_to(cqm_problem(), backend.name))
        assert result.status == "success", result.errors
        return result

    def test_exact_cqm_solver_really_re_sorts_the_variables(self):
        # Guard on the premise of the next test: the backend's own order is
        # alphabetical, which is *not* the problem's declaration order.
        compiled = CQMCompiler().compile(cqm_problem(), None)
        backend = FakeLocalCQMBackend()
        raw = backend.solve(compiled, SolverPreferences())
        assert raw.variables == ["a", "b", "c"]
        assert [v.name for v in cqm_problem().variables] == ["b", "a", "c"]

    def test_solution_keys_follow_the_problem_not_the_sampler(self, result):
        # 3b §13: in 3a this was the sampler's alphabetical order.
        for solution in result.solutions:
            assert list(solution.variables) == ["b", "a", "c"], solution.variables

    def test_no_internal_variable_and_correct_optimum(self, result):
        problem = cqm_problem()
        best = result.solutions[0]
        assert not [key for key in best.variables if key.startswith("__")]
        # max b + 3a + 2c with a + b + c <= 2 -> a = c = 1, b = 0, value 5.
        assert best.variables == {"b": 0, "a": 1, "c": 1}
        assert best.objective_value == 5.0
        assert best.objective_value == evaluate_objective(
            problem.objective, best.variables
        )

    def test_every_assignment_is_enumerated_and_deduplicated(self, result):
        attempt = result.attempts[0]
        assert attempt.samples_received == 8  # 2^3, ExactCQMSolver
        assert attempt.unique_samples == 8


# --------------------------------------------------------------------------
# 3. The compilers' ``decode`` on hand-built results
# --------------------------------------------------------------------------


def shuffled_raw(
    names: list[str],
    rows: list[list[int]],
    energies: list[float],
    *,
    dtype,
    metadata: SolverExecutionMetadata | None = None,
    backend: str = "hand-built",
) -> RawSolverResult:
    return RawSolverResult(
        variables=list(names),
        samples=np.array(rows, dtype=dtype).reshape(len(rows), len(names)),
        energies=np.array(energies, dtype=np.float64),
        backend=backend,
        metadata=metadata,
    )


METADATA = SolverExecutionMetadata(
    backend="hand-built", remote=False, num_reads_requested=4
)


class TestBQMCompilerDecode:
    """``BQMCompiler.decode`` on a hand-built, shuffled, slack-carrying result."""

    @staticmethod
    def compiled():
        return BQMCompiler().compile(bqm_problem(), 10.0)

    # Columns: b, __slack_cap_1, c, __slack_cap_0, a -- business names out of
    # order with internal ones interleaved.
    NAMES = ["b", "__slack_cap_1", "c", "__slack_cap_0", "a"]
    ROWS = [
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 1, 1, 0, 1],
    ]
    ENERGIES = [-3.5, 2.0, -7.25]

    def test_reorders_drops_internals_and_keeps_everything_else(self):
        compiled = self.compiled()
        raw = shuffled_raw(
            self.NAMES, self.ROWS, self.ENERGIES, dtype=np.int8, metadata=METADATA
        )

        decoded = BQMCompiler().decode(compiled, raw)

        assert decoded.variables == ["c", "a", "b"]
        assert decoded.samples.tolist() == [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
        assert np.array_equal(decoded.energies, raw.energies)
        assert decoded.backend == raw.backend
        assert decoded.metadata is raw.metadata
        # The input is untouched: decode returns a new result.
        assert raw.variables == self.NAMES
        assert raw.samples.shape == (3, 5)

    def test_int8_stays_int8(self):
        raw = shuffled_raw(self.NAMES, self.ROWS, self.ENERGIES, dtype=np.int8)
        decoded = BQMCompiler().decode(self.compiled(), raw)
        assert raw.samples.dtype == np.int8
        assert decoded.samples.dtype == np.int8

    def test_int64_stays_int64(self):
        raw = shuffled_raw(self.NAMES, self.ROWS, self.ENERGIES, dtype=np.int64)
        decoded = BQMCompiler().decode(self.compiled(), raw)
        assert raw.samples.dtype == np.int64
        assert decoded.samples.dtype == np.int64

    def test_a_missing_business_variable_is_a_contract_violation(self):
        raw = shuffled_raw(
            ["b", "__slack_cap_1", "__slack_cap_0", "a"],
            [[1, 0, 1, 0], [0, 1, 0, 1]],
            [0.0, 1.0],
            dtype=np.int8,
            backend="incomplete",
        )
        with pytest.raises(ValueError, match="lacks business variable 'c'"):
            BQMCompiler().decode(self.compiled(), raw)

    def test_an_empty_result_decodes_to_zero_rows(self):
        raw = RawSolverResult(
            variables=self.NAMES, samples=[], energies=[], backend="empty"
        )
        decoded = BQMCompiler().decode(self.compiled(), raw)

        assert decoded.variables == ["c", "a", "b"]
        assert decoded.samples.shape == (0, 3)
        assert decoded.energies.shape == (0,)
        assert decoded.num_samples == 0


class TestCQMCompilerDecode:
    """``CQMCompiler.decode`` on a hand-built result in the sampler's order."""

    @staticmethod
    def compiled():
        return CQMCompiler().compile(cqm_problem(), None)

    NAMES = ["a", "b", "c"]  # ExactCQMSolver's alphabetical order
    ROWS = [[1, 0, 1], [0, 1, 1], [0, 0, 0]]
    ENERGIES = [-5.0, -3.0, 0.0]

    def test_restores_the_problem_order(self):
        raw = shuffled_raw(
            self.NAMES, self.ROWS, self.ENERGIES, dtype=np.int64, metadata=METADATA
        )

        decoded = CQMCompiler().decode(self.compiled(), raw)

        assert decoded.variables == ["b", "a", "c"]
        assert decoded.samples.tolist() == [[0, 1, 1], [1, 0, 1], [0, 0, 0]]
        assert np.array_equal(decoded.energies, raw.energies)
        assert decoded.backend == raw.backend
        assert decoded.metadata is raw.metadata

    def test_int64_stays_int64_and_int8_stays_int8(self):
        wide = shuffled_raw(self.NAMES, self.ROWS, self.ENERGIES, dtype=np.int64)
        narrow = shuffled_raw(self.NAMES, self.ROWS, self.ENERGIES, dtype=np.int8)

        assert CQMCompiler().decode(self.compiled(), wide).samples.dtype == np.int64
        assert CQMCompiler().decode(self.compiled(), narrow).samples.dtype == np.int8

    def test_a_missing_business_variable_is_a_contract_violation(self):
        raw = shuffled_raw(
            ["a", "c"], [[1, 1], [0, 0]], [0.0, 1.0], dtype=np.int64, backend="partial"
        )
        with pytest.raises(ValueError, match="lacks business variable 'b'"):
            CQMCompiler().decode(self.compiled(), raw)

    def test_an_empty_result_decodes_to_zero_rows(self):
        raw = RawSolverResult(
            variables=self.NAMES, samples=[], energies=[], backend="empty"
        )
        decoded = CQMCompiler().decode(self.compiled(), raw)

        assert decoded.variables == ["b", "a", "c"]
        assert decoded.samples.shape == (0, 3)
        assert decoded.num_samples == 0
