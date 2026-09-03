"""Knapsack scenario on the CQM path (3a spec §26.3).

The shipped ``examples/knapsack.json`` goes JSON -> ``OptimizationService``
-> result, exactly like ``test_knapsack.py``, but through a backend that
declares constraint models only (``FakeLocalCQMBackend``). The answer must
be the same optimum the exact BQM path finds, with no slack variable in
sight and no hard penalty on the single attempt.
"""

import pytest

from annealbridge.models import OptimizationProblem, SolverPreferences
from annealbridge.orchestration import OptimizationService
from annealbridge.solvers import SolverRegistry
from tests.fakes.local_cqm_backend import FAKE_LOCAL_CQM_NAME, FakeLocalCQMBackend

KNAPSACK_OPTIMUM_VALUE = 17.0
KNAPSACK_BUSINESS_NAMES = {"item_a", "item_b", "item_c", "item_d"}


def registry_with_fake(fake: FakeLocalCQMBackend) -> SolverRegistry:
    defaults = SolverRegistry.default()
    backends = {name: defaults.get(name) for name in defaults.names()}
    backends[FAKE_LOCAL_CQM_NAME] = fake
    return SolverRegistry(backends)


@pytest.fixture
def load_problem(load_example):
    """Load the example JSON; ``backend=fake`` routes it to the CQM fake.

    The backend name is a Literal of shipped names, so the fake is set via
    ``model_construct`` (see ``test_fifth_backend.py``).
    """

    def _load(backend: str, **solver_overrides) -> OptimizationProblem:
        problem = OptimizationProblem.model_validate(load_example("knapsack.json"))
        if backend == FAKE_LOCAL_CQM_NAME:
            solver = SolverPreferences.model_construct(backend=backend, **solver_overrides)
        else:
            solver = SolverPreferences(backend=backend, **solver_overrides)
        return problem.model_copy(update={"solver": solver})

    return _load


class TestKnapsackCQM:
    def test_cqm_backend_finds_the_same_optimum_as_exact(self, load_problem):
        fake = FakeLocalCQMBackend()
        service = OptimizationService(registry=registry_with_fake(fake))

        result = service.solve(load_problem(FAKE_LOCAL_CQM_NAME))
        exact = service.solve(load_problem("exact"))

        assert result.status == "success"
        assert result.backend == FAKE_LOCAL_CQM_NAME
        assert result.objective_direction == "maximize"
        assert result.metadata is not None
        assert result.metadata.model_type == "cqm"

        best = result.solutions[0]
        assert best.rank == 1
        assert best.objective_value == pytest.approx(KNAPSACK_OPTIMUM_VALUE)
        assert best.variables == exact.solutions[0].variables
        assert best.hard_constraints_satisfied is True

    def test_single_attempt_without_hard_penalty(self, load_problem):
        service = OptimizationService(registry=registry_with_fake(FakeLocalCQMBackend()))

        result = service.solve(load_problem(FAKE_LOCAL_CQM_NAME))

        assert len(result.attempts) == 1
        assert result.attempts[0].penalty is None
        assert result.infeasibility_proven is False
        assert result.warnings == []

    def test_solutions_contain_business_variables_only(self, load_problem):
        service = OptimizationService(registry=registry_with_fake(FakeLocalCQMBackend()))

        result = service.solve(load_problem(FAKE_LOCAL_CQM_NAME, top_k=5))

        assert len(result.solutions) == 5
        for solution in result.solutions:
            assert set(solution.variables) == KNAPSACK_BUSINESS_NAMES
            assert not any(name.startswith("__") for name in solution.variables)
