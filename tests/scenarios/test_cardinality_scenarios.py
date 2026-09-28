"""Scenario: a schema 1.2 problem written with cardinality constraints.

JSON text -> ``OptimizationProblem`` -> ``OptimizationService.solve`` ->
result, the route an MCP or CLI caller takes (schema 1.2 spec 2026-09-25
§13 item 7). Three jobs go to three machines:

* each job runs on exactly one machine (hard one-hot, ``"==", 1``);
* machines ``m1`` and ``m2`` take at most one job each (hard at-most-one,
  ``"<=", 1``: the pairwise encoding on the BQM path);
* ``m3`` should take at most one job too, but may take more at a cost
  (soft ``"<=", 1``, weight 2).

Costs per machine (m1, m2, m3): a 1 4 3, b 2 3 2, c 1 5 2. Without the soft
rule the cheapest schedule is a=m1, b=m3, c=m3 (cost 5, soft penalty 2,
ranking 7); with it the unique optimum is a=m1, b=m2, c=m3: cost 6, soft
score 0, ranking 6. ``test_the_optimum_is_proven_by_enumeration`` checks
this over all 512 assignments with the solution validator.
"""

import itertools
import json

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.candidates import evaluate_objective
from annealbridge.validation import validate_problem, validate_solution

PROBLEM_JSON = """
{
  "version": "1.2",
  "name": "machine_assignment",
  "variables": [
    {"name": "a_m1"}, {"name": "a_m2"}, {"name": "a_m3"},
    {"name": "b_m1"}, {"name": "b_m2"}, {"name": "b_m3"},
    {"name": "c_m1"}, {"name": "c_m2"}, {"name": "c_m3"}
  ],
  "objective": {
    "direction": "minimize",
    "linear_terms": [
      {"variable": "a_m1", "coefficient": 1},
      {"variable": "a_m2", "coefficient": 4},
      {"variable": "a_m3", "coefficient": 3},
      {"variable": "b_m1", "coefficient": 2},
      {"variable": "b_m2", "coefficient": 3},
      {"variable": "b_m3", "coefficient": 2},
      {"variable": "c_m1", "coefficient": 1},
      {"variable": "c_m2", "coefficient": 5},
      {"variable": "c_m3", "coefficient": 2}
    ]
  },
  "constraints": [],
  "cardinality_constraints": [
    {"id": "a_once", "type": "hard", "variables": ["a_m1", "a_m2", "a_m3"],
     "operator": "==", "rhs": 1},
    {"id": "b_once", "type": "hard", "variables": ["b_m1", "b_m2", "b_m3"],
     "operator": "==", "rhs": 1},
    {"id": "c_once", "type": "hard", "variables": ["c_m1", "c_m2", "c_m3"],
     "operator": "==", "rhs": 1},
    {"id": "m1_single", "type": "hard", "variables": ["a_m1", "b_m1", "c_m1"],
     "operator": "<=", "rhs": 1},
    {"id": "m2_single", "type": "hard", "variables": ["a_m2", "b_m2", "c_m2"],
     "operator": "<=", "rhs": 1},
    {"id": "m3_prefer_single", "type": "soft", "weight": 2,
     "variables": ["a_m3", "b_m3", "c_m3"], "operator": "<=", "rhs": 1,
     "description": "m3 is slow when shared"}
  ]
}
"""

CONSTRAINT_IDS = ["a_once", "b_once", "c_once", "m1_single", "m2_single", "m3_prefer_single"]
OPTIMUM = {
    "a_m1": 1, "a_m2": 0, "a_m3": 0,
    "b_m1": 0, "b_m2": 1, "b_m3": 0,
    "c_m1": 0, "c_m2": 0, "c_m3": 1,
}  # fmt: skip
OPTIMAL_OBJECTIVE = 6.0
OPTIMAL_RANKING = 6.0


def load(**solver) -> OptimizationProblem:
    data = json.loads(PROBLEM_JSON)
    data["solver"] = solver
    return OptimizationProblem.model_validate_json(json.dumps(data))


def test_the_optimum_is_proven_by_enumeration():
    problem = load(backend="exact")
    assert validate_problem(problem) == []
    names = [variable.name for variable in problem.variables]
    rankings: dict[tuple[int, ...], float] = {}
    for values in itertools.product((0, 1), repeat=len(names)):
        sample = dict(zip(names, values))
        check = validate_solution(problem, sample)
        if check.feasible:
            objective = evaluate_objective(problem.objective, sample)
            rankings[values] = objective + check.soft_violation_score
    best = min(rankings.values())
    winners = [values for values, score in rankings.items() if score == best]
    assert best == OPTIMAL_RANKING
    assert winners == [tuple(OPTIMUM[name] for name in names)]
    # The soft rule binds: without it a cheaper schedule exists.
    cheaper = {"a_m1": 1, "b_m3": 1, "c_m3": 1}
    assert evaluate_objective(problem.objective, {n: cheaper.get(n, 0) for n in names}) == 5.0


@pytest.fixture(scope="module")
def exact_result():
    return OptimizationService().solve(load(backend="exact"))


@pytest.fixture(scope="module")
def annealing_result():
    return OptimizationService().solve(load(backend="simulated_annealing", seed=11, num_reads=200))


class TestExact:
    @pytest.fixture
    def result(self, exact_result):
        return exact_result

    def test_the_proven_optimum_is_returned(self, result):
        assert result.status == "success", result.errors
        assert result.optimality_proven is True
        best = result.solutions[0]
        assert best.variables == OPTIMUM
        assert best.objective_value == OPTIMAL_OBJECTIVE
        assert best.soft_violation_score == 0.0
        assert best.ranking_score == OPTIMAL_RANKING
        assert best.hard_constraints_satisfied is True

    def test_every_constraint_is_reported_in_declaration_order(self, result):
        for solution in result.solutions:
            assert [e.constraint_id for e in solution.constraint_evaluations] == CONSTRAINT_IDS
        m3 = result.solutions[0].constraint_evaluations[-1]
        assert (m3.constraint_type, m3.operator, m3.actual_value, m3.expected_value) == (
            "soft",
            "<=",
            1.0,
            1.0,
        )

    def test_the_compiled_model_has_one_slack_bit_only(self, result):
        """One-hots and pairwise at-most-ones need none; the soft <= 1 needs one."""
        (attempt,) = result.attempts
        assert attempt.compiled_variables == 9 + 1


class TestSimulatedAnnealing:
    @pytest.fixture
    def result(self, annealing_result):
        return annealing_result

    def test_a_feasible_optimum_is_found(self, result):
        assert result.status == "success", result.errors
        best = result.solutions[0]
        assert best.hard_constraints_satisfied is True
        assert validate_solution(load(backend="exact"), best.variables).feasible is True
        assert best.objective_value == OPTIMAL_OBJECTIVE
        assert best.ranking_score == OPTIMAL_RANKING
