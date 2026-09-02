"""Assignment scenario through the full service pipeline (spec §31, §34)."""

from pathlib import Path

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

ASSIGNMENT_PATH = Path(__file__).resolve().parents[2] / "examples" / "assignment.json"

# Cost matrix (workers x tasks):
#           clean  cook  drive
#   alice     4      2     8
#   bob       3      7     5
#   carol     6      4     3
# All 6 permutations by hand:
#   alice=clean, bob=cook,  carol=drive: 4 + 7 + 3 = 14
#   alice=clean, bob=drive, carol=cook : 4 + 5 + 4 = 13
#   alice=cook,  bob=clean, carol=drive: 2 + 3 + 3 = 8   <-- optimum
#   alice=cook,  bob=drive, carol=clean: 2 + 5 + 6 = 13
#   alice=drive, bob=clean, carol=cook : 8 + 3 + 4 = 15
#   alice=drive, bob=cook,  carol=clean: 8 + 7 + 6 = 21
ASSIGNMENT_OPTIMUM_COST = 8.0
ASSIGNMENT_OPTIMUM = {
    "alice_clean": 0,
    "alice_cook": 1,
    "alice_drive": 0,
    "bob_clean": 1,
    "bob_cook": 0,
    "bob_drive": 0,
    "carol_clean": 0,
    "carol_cook": 0,
    "carol_drive": 1,
}


def load_problem() -> OptimizationProblem:
    return OptimizationProblem.model_validate_json(ASSIGNMENT_PATH.read_text())


class TestAssignmentExact:
    def test_exact_backend_finds_minimum_cost(self):
        problem = load_problem()
        assert problem.solver.backend == "exact"
        # 9 variables, equality constraints only (no slack), within the
        # exact backend's 24-variable limit.
        assert len(problem.variables) == 9

        result = OptimizationService().solve(problem)

        assert result.status == "success"
        assert result.backend == "exact"
        assert result.objective_direction == "minimize"
        assert len(result.attempts) == 1

        best = result.solutions[0]
        assert best.rank == 1
        assert best.objective_value == pytest.approx(ASSIGNMENT_OPTIMUM_COST)
        assert best.variables == ASSIGNMENT_OPTIMUM

    def test_all_one_hot_constraints_satisfied(self):
        result = OptimizationService().solve(load_problem())

        best = result.solutions[0]
        assert best.hard_constraints_satisfied is True
        assert len(best.constraint_evaluations) == 6
        for evaluation in best.constraint_evaluations:
            assert evaluation.constraint_type == "hard"
            assert evaluation.satisfied is True
            assert evaluation.actual_value == pytest.approx(1.0)

    def test_every_returned_solution_is_a_valid_assignment(self):
        result = OptimizationService().solve(load_problem())

        assert result.solutions
        for solution in result.solutions:
            assert solution.hard_constraints_satisfied is True
            # A valid assignment selects exactly 3 of the 9 variables.
            assert sum(solution.variables.values()) == 3

    def test_ranking_is_ascending_for_minimize(self):
        result = OptimizationService().solve(load_problem())
        scores = [solution.ranking_score for solution in result.solutions]
        assert scores == sorted(scores)
