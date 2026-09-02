"""Retry scenario: tiny penalty forces SA retries; exact never retries (spec §32)."""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

# objective_scale = 10 + 8 + 7 + 6 = 31, so multiplier 0.01 gives an
# initial hard penalty of 0.31. At that strength every over-weight subset
# is cheaper in energy than the best feasible one (e.g. {A,C,D}: value 23,
# excess 3 -> energy -23 + 0.31*9 = -20.21 vs. {A,C} at -17), so SA's
# first attempt lands on infeasible attractors. The feasible optimum only
# becomes the ground state once the doubled penalty exceeds 1.0 (binding
# case {B,C,D}: -21 + 4*lambda > -17 requires lambda > 1), which the retry
# doubling reaches by attempt 3 (0.31 -> 0.62 -> 1.24).
TINY_MULTIPLIER = 0.01
SEED = 0
KNAPSACK_OPTIMUM_VALUE = 17.0


@pytest.fixture
def load_problem(load_example):
    def _load(**solver_overrides) -> OptimizationProblem:
        return OptimizationProblem.model_validate(
            load_example("knapsack.json", **solver_overrides)
        )

    return _load


class TestSimulatedAnnealingRetry:
    def solve(self, load_problem):
        problem = load_problem(
            backend="simulated_annealing",
            seed=SEED,
            num_reads=100,
            penalty_multiplier=TINY_MULTIPLIER,
        )
        return OptimizationService().solve(problem)

    def test_retries_then_succeeds(self, load_problem):
        result = self.solve(load_problem)

        assert result.status == "success"
        assert len(result.attempts) > 1
        assert result.attempts[0].feasible_samples == 0
        assert result.attempts[-1].feasible_samples > 0
        assert result.solutions[0].objective_value == pytest.approx(
            KNAPSACK_OPTIMUM_VALUE
        )

    def test_penalty_increases_each_attempt(self, load_problem):
        result = self.solve(load_problem)

        penalties = [attempt.penalty for attempt in result.attempts]
        assert penalties[0] == pytest.approx(31.0 * TINY_MULTIPLIER)
        for previous, current in zip(penalties, penalties[1:]):
            assert current == pytest.approx(previous * 2.0)
            assert current > previous

    def test_attempt_numbers_are_sequential(self, load_problem):
        result = self.solve(load_problem)
        assert [attempt.attempt for attempt in result.attempts] == list(
            range(1, len(result.attempts) + 1)
        )

    def test_retries_are_bounded_by_max_retries(self, load_problem):
        result = self.solve(load_problem)
        assert len(result.attempts) <= 1 + 3  # default max_retries = 3


class TestExactNeverRetries:
    def test_exact_backend_solves_in_one_attempt(self, load_problem):
        # Same tiny penalty multiplier: the exhaustive backend still sees
        # every assignment, so feasible solutions are found regardless of
        # lambda and no retry is ever attempted (spec §19, §45).
        problem = load_problem(backend="exact", penalty_multiplier=TINY_MULTIPLIER)
        result = OptimizationService().solve(problem)

        assert result.status == "success"
        assert len(result.attempts) == 1
        assert result.solutions[0].objective_value == pytest.approx(
            KNAPSACK_OPTIMUM_VALUE
        )
