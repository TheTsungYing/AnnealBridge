"""Bounded-integer knapsack across all three paths (3b spec §26.3).

``examples/integer_knapsack.json`` is the first shipped ``version 1.1``
problem: four ``[0, 3]`` integer variables, a hard capacity constraint and
one soft preference. It goes JSON -> ``OptimizationService`` -> result on
three independent paths — the exhaustive BQM backend (``exact``), the
heuristic BQM backend (``simulated_annealing``) and a constraint-model
backend (``FakeLocalCQMBackend``, 3a §26.1) — and all three must return the
same business optimum, with the integer values decoded back to plain
``int`` inside their declared bounds and no binary-expansion variable left
in sight.

Nothing here calls a compiler, a decoder or a validator by hand: the whole
point is that the *service* pipeline gets the integers right end to end.
The only exception is :func:`brute_force_optimum`, which is the independent
oracle the expected answer is checked against.
"""

import itertools

import pytest

from annealbridge.models import OptimizationProblem, SolverPreferences
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.optimizer import evaluate_objective
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import validate_solution
from tests.fakes.local_cqm_backend import FAKE_LOCAL_CQM_NAME, FakeLocalCQMBackend

# Capacity 18, each item taken 0..3 times: A(w6,v10) B(w5,v9) C(w4,v7)
# D(w3,v6), plus a soft preference for b + c <= 2 (weight 3).
# ``brute_force_optimum`` enumerates all 4^4 = 256 assignments and proves
# this one is the *unique* feasible maximum (see
# ``test_brute_force_oracle_confirms_a_unique_optimum``), so every path may
# assert the exact variable assignment and not merely the objective value.
INTEGER_KNAPSACK_OPTIMUM_VALUE = 34.0
INTEGER_KNAPSACK_OPTIMUM_SELECTION = {"item_a": 0, "item_b": 1, "item_c": 1, "item_d": 3}

# The three paths under test: (backend name, solver overrides).
# SA takes 500 reads: the landscape has a strong attractor at value 32
# ({B, C, D×3} minus one D), and at 100 reads roughly half of all seeds
# settle there instead of the optimum — as ``test_knapsack.py`` notes for
# the binary knapsack. 500 reads reach 34 for every seed tried but one.
SA_OVERRIDES = {"seed": 1234, "num_reads": 500}
PATHS = [
    ("exact", {}),
    ("simulated_annealing", SA_OVERRIDES),
    (FAKE_LOCAL_CQM_NAME, {}),
]


def brute_force_optimum(
    problem: OptimizationProblem,
) -> tuple[float, list[dict[str, int]]]:
    """Enumerate every assignment; return ``(best value, best samples)``.

    Independent of the compilers and of the solvers: the ranges come from
    each variable's own ``bounds()`` (here ``range(0, 4)`` four times),
    feasibility from the solution validator and the value from
    ``evaluate_objective``, all against the *original* problem. Key order
    follows ``problem.variables``.
    """
    names = [variable.name for variable in problem.variables]
    ranges = [
        range(lower, upper + 1)
        for lower, upper in (variable.bounds() for variable in problem.variables)
    ]

    best_value: float | None = None
    best_samples: list[dict[str, int]] = []
    for values in itertools.product(*ranges):
        sample = dict(zip(names, values, strict=True))
        if not validate_solution(problem, sample).feasible:
            continue
        value = evaluate_objective(problem.objective, sample)
        if best_value is None or value > best_value:
            best_value, best_samples = value, [sample]
        elif value == best_value:
            best_samples.append(sample)
    assert best_value is not None, "the problem has no feasible assignment"
    return best_value, best_samples


@pytest.fixture
def load_problem(load_example):
    """Load ``examples/integer_knapsack.json`` with the given solver block.

    ``backend`` is a Literal of shipped names, so the test-only CQM fake is
    set via ``model_construct`` (see ``test_knapsack_cqm.py``).
    """

    def _load(backend: str, **solver_overrides) -> OptimizationProblem:
        problem = OptimizationProblem.model_validate(
            load_example("integer_knapsack.json")
        )
        if backend == FAKE_LOCAL_CQM_NAME:
            solver = SolverPreferences.model_construct(backend=backend, **solver_overrides)
        else:
            solver = SolverPreferences(backend=backend, **solver_overrides)
        return problem.model_copy(update={"solver": solver})

    return _load


def registry_with_fake(fake: FakeLocalCQMBackend) -> SolverRegistry:
    defaults = SolverRegistry.default()
    backends = {name: defaults.get(name) for name in defaults.names()}
    backends[FAKE_LOCAL_CQM_NAME] = fake
    return SolverRegistry(backends)


@pytest.fixture
def solve_on(load_problem):
    """``solve_on(backend, **overrides)`` -> the service result for one path."""

    def _solve(backend: str, **solver_overrides):
        if backend == FAKE_LOCAL_CQM_NAME:
            registry = registry_with_fake(FakeLocalCQMBackend())
            service = OptimizationService(registry=registry)
        else:
            service = OptimizationService()
        return service.solve(load_problem(backend, **solver_overrides))

    return _solve


class TestIntegerKnapsackOracle:
    def test_brute_force_oracle_confirms_a_unique_optimum(self, load_problem):
        problem = load_problem("exact")

        best_value, best_samples = brute_force_optimum(problem)

        assert best_value == pytest.approx(INTEGER_KNAPSACK_OPTIMUM_VALUE)
        assert len(best_samples) == 1
        assert best_samples[0] == INTEGER_KNAPSACK_OPTIMUM_SELECTION


class TestIntegerKnapsackExact:
    def test_exact_backend_finds_the_optimum(self, solve_on):
        result = solve_on("exact")

        assert result.status == "success"
        assert result.backend == "exact"
        assert result.objective_direction == "maximize"
        assert result.infeasibility_proven is False

        best = result.solutions[0]
        assert best.rank == 1
        assert best.objective_value == pytest.approx(INTEGER_KNAPSACK_OPTIMUM_VALUE)
        assert best.variables == INTEGER_KNAPSACK_OPTIMUM_SELECTION
        assert best.hard_constraints_satisfied is True


class TestIntegerKnapsackSimulatedAnnealing:
    def test_sa_with_fixed_seed_finds_the_optimum(self, solve_on):
        result = solve_on("simulated_annealing", **SA_OVERRIDES)

        assert result.status == "success"
        assert result.backend == "simulated_annealing"

        best = result.solutions[0]
        assert best.objective_value == pytest.approx(INTEGER_KNAPSACK_OPTIMUM_VALUE)
        assert best.variables == INTEGER_KNAPSACK_OPTIMUM_SELECTION
        assert best.hard_constraints_satisfied is True


class TestIntegerKnapsackCQM:
    def test_cqm_backend_finds_the_optimum(self, solve_on):
        result = solve_on(FAKE_LOCAL_CQM_NAME)

        assert result.status == "success"
        assert result.backend == FAKE_LOCAL_CQM_NAME
        assert result.metadata is not None
        assert result.metadata.model_type == "cqm"

        best = result.solutions[0]
        assert best.objective_value == pytest.approx(INTEGER_KNAPSACK_OPTIMUM_VALUE)
        assert best.variables == INTEGER_KNAPSACK_OPTIMUM_SELECTION
        assert best.hard_constraints_satisfied is True


class TestIntegerKnapsackAcrossPaths:
    def test_three_paths_agree_on_the_first_solution(self, load_problem, solve_on):
        results = [solve_on(backend, **overrides) for backend, overrides in PATHS]

        firsts = [result.solutions[0].variables for result in results]
        assert firsts[0] == firsts[1] == firsts[2]

        # Key order is the problem's declared variable order on every path,
        # not the sampler's or the binary expansion's.
        expected_order = [
            variable.name for variable in load_problem("exact").variables
        ]
        for first in firsts:
            assert list(first) == expected_order

    @pytest.mark.parametrize(
        ("backend", "overrides"), PATHS, ids=[path[0] for path in PATHS]
    )
    def test_every_solution_is_a_clean_integer_assignment(
        self, load_problem, solve_on, backend, overrides
    ):
        bounds = {
            variable.name: variable.bounds()
            for variable in load_problem("exact").variables
        }

        result = solve_on(backend, **overrides)

        assert result.solutions
        for solution in result.solutions:
            assert set(solution.variables) == set(bounds)
            for name, value in solution.variables.items():
                assert not name.startswith("__")
                # Plain Python ``int``: not numpy.int64, and not bool
                # (``type(True) is int`` is False, so this covers both).
                assert type(value) is int
                lower, upper = bounds[name]
                assert lower <= value <= upper
