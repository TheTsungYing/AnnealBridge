"""Unit tests for candidate processing and ranking (spec §25, §25.1, §33)."""

import pytest

from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    Variable,
)
from annealbridge.orchestration import process_candidates
from annealbridge.solvers import RawSolverResult


def make_problem(
    direction: str = "minimize",
    coefficients: dict[str, float] | None = None,
    constraints: list[Constraint] | None = None,
) -> OptimizationProblem:
    coefficients = coefficients if coefficients is not None else {"x": 1.0, "y": 2.0}
    return OptimizationProblem(
        name="ranking-test",
        variables=[Variable(name=name) for name in coefficients],
        objective=Objective(
            direction=direction,
            linear_terms=[
                LinearTerm(variable=name, coefficient=value)
                for name, value in coefficients.items()
            ],
        ),
        constraints=constraints or [],
    )


def raw(samples: list[dict[str, int]], energies: list[float]) -> RawSolverResult:
    return RawSolverResult.from_dicts(samples, energies, "exact")


class TestDeduplication:
    def test_same_business_solution_kept_once_with_min_energy(self):
        problem = make_problem()
        # Two samples differ only in the internal slack bit -> one business
        # solution; the kept energy must be the minimum of the duplicates.
        result = raw(
            [
                {"x": 1, "y": 0, "__slack_c_0": 0},
                {"x": 1, "y": 0, "__slack_c_0": 1},
                {"x": 0, "y": 1, "__slack_c_0": 0},
            ],
            [5.0, 3.0, 4.0],
        )
        solutions, unique, feasible = process_candidates(
            problem, result, {"__slack_c_0"}, top_k=10
        )

        assert unique == 2
        assert feasible == 2
        assert len(solutions) == 2
        by_vars = {tuple(sorted(s.variables.items())): s for s in solutions}
        merged = by_vars[(("x", 1), ("y", 0))]
        assert merged.energy == 3.0

    def test_internal_variables_never_appear_in_solutions(self):
        problem = make_problem()
        result = raw([{"x": 1, "y": 1, "__slack_c_0": 1}], [0.0])
        solutions, _, _ = process_candidates(problem, result, {"__slack_c_0"}, top_k=5)
        assert solutions[0].variables == {"x": 1, "y": 1}


class TestSoftViolationRanking:
    def test_soft_violation_can_outweigh_better_objective(self):
        # maximize 5a + 4b with soft "a <= 0" (weight 2): a=1 scores
        # 5 - 2 = 3, b=1 scores 4 - 0 = 4, so b=1 must rank first even
        # though its raw objective is lower.
        soft = Constraint(
            id="avoid_a",
            type="soft",
            terms=[LinearTerm(variable="a", coefficient=1)],
            operator="<=",
            rhs=0,
            weight=2.0,
        )
        problem = make_problem(
            direction="maximize",
            coefficients={"a": 5.0, "b": 4.0},
            constraints=[soft],
        )
        result = raw([{"a": 1, "b": 0}, {"a": 0, "b": 1}], [-5.0, -4.0])

        solutions, _, feasible = process_candidates(problem, result, set(), top_k=5)

        assert feasible == 2
        assert solutions[0].variables == {"a": 0, "b": 1}
        assert solutions[0].ranking_score == pytest.approx(4.0)
        assert solutions[0].soft_violation_score == pytest.approx(0.0)
        assert solutions[1].variables == {"a": 1, "b": 0}
        assert solutions[1].ranking_score == pytest.approx(3.0)
        assert solutions[1].soft_violation_score == pytest.approx(2.0)
        assert solutions[1].objective_value == pytest.approx(5.0)
        assert [s.rank for s in solutions] == [1, 2]


class TestTieBreak:
    def test_equal_scores_break_by_assignment_tuple(self):
        # minimize a + b: both single-selection solutions score 1.0; the
        # name-sorted assignment tuple (0, 1) < (1, 0) puts a=0,b=1 first.
        problem = make_problem(coefficients={"a": 1.0, "b": 1.0})
        samples = [{"a": 1, "b": 0}, {"a": 0, "b": 1}]
        energies = [1.0, 1.0]

        forward, _, _ = process_candidates(
            problem, raw(samples, energies), set(), top_k=5
        )
        reverse, _, _ = process_candidates(
            problem, raw(samples[::-1], energies[::-1]), set(), top_k=5
        )

        expected_order = [{"a": 0, "b": 1}, {"a": 1, "b": 0}]
        assert [s.variables for s in forward] == expected_order
        assert [s.variables for s in reverse] == expected_order

    def test_tie_break_prefers_objective_before_assignment(self):
        soft = Constraint(
            id="want_x",
            type="soft",
            terms=[LinearTerm(variable="x", coefficient=1)],
            operator=">=",
            rhs=1,
            weight=1.0,
        )
        problem = make_problem(
            direction="minimize",
            coefficients={"x": 2.0, "y": 1.0},
            constraints=[soft],
        )
        # x=1,y=0: obj 2, soft 0 -> score 2; x=0,y=1: obj 1, soft 1 -> score 2.
        # Equal ranking_score, so the lower objective (minimize) ranks first.
        result = raw([{"x": 1, "y": 0}, {"x": 0, "y": 1}], [0.0, 0.0])
        solutions, _, _ = process_candidates(problem, result, set(), top_k=5)

        assert solutions[0].ranking_score == pytest.approx(2.0)
        assert solutions[1].ranking_score == pytest.approx(2.0)
        assert solutions[0].variables == {"x": 0, "y": 1}
        assert solutions[0].objective_value == pytest.approx(1.0)


class TestFeasibilityFilterAndTopK:
    def test_infeasible_candidates_are_dropped(self):
        hard = Constraint(
            id="pick_one",
            type="hard",
            terms=[
                LinearTerm(variable="x", coefficient=1),
                LinearTerm(variable="y", coefficient=1),
            ],
            operator="==",
            rhs=1,
        )
        problem = make_problem(constraints=[hard])
        result = raw(
            [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
            [0.0, 1.0, 3.0],
        )
        solutions, unique, feasible = process_candidates(problem, result, set(), top_k=5)

        assert unique == 3
        assert feasible == 1
        assert len(solutions) == 1
        assert solutions[0].variables == {"x": 1, "y": 0}
        assert solutions[0].hard_constraints_satisfied is True

    def test_top_k_limits_and_ranks_from_one(self):
        problem = make_problem(coefficients={"x": 1.0, "y": 2.0})
        result = raw(
            [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}, {"x": 1, "y": 1}],
            [0.0, 1.0, 2.0, 3.0],
        )
        solutions, unique, feasible = process_candidates(problem, result, set(), top_k=2)

        assert unique == 4
        assert feasible == 4
        assert [s.rank for s in solutions] == [1, 2]
        assert solutions[0].variables == {"x": 0, "y": 0}
        assert solutions[1].variables == {"x": 1, "y": 0}
