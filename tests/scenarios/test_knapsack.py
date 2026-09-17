"""Knapsack scenario through the full service pipeline (spec §30, §34)."""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

# Spec §30: capacity 10, items A(w6,v10) B(w5,v8) C(w4,v7) D(w3,v6).
# {A, C} weighs 6 + 4 = 10 (feasible) and is worth 10 + 7 = 17; the next
# best feasible subsets are {B, C} = 15, {B, D} = 14 and {C, D} = 13, and
# any subset containing more weight is infeasible, so 17 is the optimum.
KNAPSACK_OPTIMUM_VALUE = 17.0
KNAPSACK_OPTIMUM_SELECTION = {"item_a": 1, "item_b": 0, "item_c": 1, "item_d": 0}


@pytest.fixture
def load_problem(load_example):
    """Load the example JSON, optionally overriding solver preferences."""

    def _load(**solver_overrides) -> OptimizationProblem:
        return OptimizationProblem.model_validate(
            load_example("knapsack.json", **solver_overrides)
        )

    return _load


class TestKnapsackExact:
    def test_exact_backend_finds_optimum(self, load_problem):
        result = OptimizationService().solve(load_problem(backend="exact"))

        assert result.status == "success"
        assert result.backend == "exact"
        assert result.objective_direction == "maximize"
        assert len(result.attempts) == 1
        assert result.infeasibility_proven is False

        best = result.solutions[0]
        assert best.rank == 1
        assert best.variables == KNAPSACK_OPTIMUM_SELECTION
        assert best.objective_value == pytest.approx(KNAPSACK_OPTIMUM_VALUE)
        assert best.hard_constraints_satisfied is True
        assert all(
            evaluation.satisfied
            for evaluation in best.constraint_evaluations
            if evaluation.constraint_type == "hard"
        )

    def test_solutions_contain_business_variables_only(self, load_problem):
        result = OptimizationService().solve(load_problem(backend="exact"))
        business_names = {"item_a", "item_b", "item_c", "item_d"}
        for solution in result.solutions:
            assert set(solution.variables) == business_names

    def test_sample_count_covers_every_slack_combination(self, load_problem):
        # The compiled model has 8 variables: 4 items plus the 4 slack bits
        # of the capacity constraint, and ``exact`` enumerates all 256 rows.
        # Each business assignment therefore appears once per slack pattern,
        # 2**4 = 16 times -- which is why this number says nothing about how
        # good a solution is.
        result = OptimizationService().solve(load_problem(backend="exact", top_k=5))

        assert result.solutions
        for solution in result.solutions:
            assert solution.sample_count == 16
        assert sum(s.sample_count for s in result.solutions) <= (
            result.attempts[0].samples_received
        )

    def test_ranking_is_descending_for_maximize(self, load_problem):
        result = OptimizationService().solve(load_problem(backend="exact", top_k=5))
        scores = [solution.ranking_score for solution in result.solutions]
        assert scores == sorted(scores, reverse=True)
        assert [solution.rank for solution in result.solutions] == list(
            range(1, len(result.solutions) + 1)
        )


class TestKnapsackSimulatedAnnealing:
    def test_sa_with_fixed_seed_finds_feasible(self, load_problem):
        result = OptimizationService().solve(
            load_problem(backend="simulated_annealing", seed=1234, num_reads=100)
        )

        assert result.status == "success"
        assert result.backend == "simulated_annealing"
        assert result.solutions
        for solution in result.solutions:
            assert solution.hard_constraints_satisfied is True
            # A sampler reaches an assignment as often as it happens to; all
            # that holds is that it was reached, and that the ranked
            # solutions cannot claim more rows than the backend returned.
            assert solution.sample_count >= 1
        assert sum(s.sample_count for s in result.solutions) <= (
            result.attempts[0].samples_received
        )

    def test_sa_with_more_reads_finds_optimum(self, load_problem):
        # Asserting the optimum with only 100 reads would be brittle for a
        # stochastic sampler; 500 reads on this 8-variable model is ample.
        result = OptimizationService().solve(
            load_problem(backend="simulated_annealing", seed=1234, num_reads=500)
        )

        assert result.status == "success"
        best = result.solutions[0]
        assert best.objective_value == pytest.approx(KNAPSACK_OPTIMUM_VALUE)
        assert best.variables == KNAPSACK_OPTIMUM_SELECTION


class TestKnapsackTabu:
    def test_tabu_with_fixed_seed_finds_the_optimum(self, load_problem):
        # The other local heuristic on the same problem: a tabu search from
        # 100 random starts reaches the optimum of this 8-variable model.
        result = OptimizationService().solve(
            load_problem(backend="tabu", seed=1234, num_reads=100)
        )

        assert result.status == "success"
        assert result.backend == "tabu"
        assert result.objective_direction == "maximize"

        best = result.solutions[0]
        assert best.objective_value == pytest.approx(KNAPSACK_OPTIMUM_VALUE)
        assert best.variables == KNAPSACK_OPTIMUM_SELECTION
        for solution in result.solutions:
            assert solution.hard_constraints_satisfied is True
            assert solution.sample_count >= 1
        assert sum(s.sample_count for s in result.solutions) <= (
            result.attempts[0].samples_received
        )
