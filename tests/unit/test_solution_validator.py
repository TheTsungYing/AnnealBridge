"""Tests for the Solution Validator (spec §23 / §23.1)."""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.validation import EPSILON, validate_solution


def make_problem(constraints: list[dict]) -> OptimizationProblem:
    """A knapsack-shaped problem with caller-supplied constraints."""
    return OptimizationProblem.model_validate(
        {
            "name": "solution validator test",
            "variables": [{"name": "x1"}, {"name": "x2"}, {"name": "x3"}],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "x1", "coefficient": 10},
                    {"variable": "x2", "coefficient": 8},
                    {"variable": "x3", "coefficient": 7},
                ],
            },
            "constraints": constraints,
        }
    )


def hard(operator: str, rhs: float, coefficients: dict[str, float]) -> dict:
    return {
        "id": f"hard_{operator}_{rhs}",
        "type": "hard",
        "terms": [
            {"variable": name, "coefficient": value}
            for name, value in coefficients.items()
        ],
        "operator": operator,
        "rhs": rhs,
    }


def soft(operator: str, rhs: float, coefficients: dict[str, float], weight: float) -> dict:
    return {
        "id": f"soft_{operator}_{rhs}",
        "type": "soft",
        "terms": [
            {"variable": name, "coefficient": value}
            for name, value in coefficients.items()
        ],
        "operator": operator,
        "rhs": rhs,
        "weight": weight,
    }


class TestValidSolution:
    def test_all_constraints_satisfied(self):
        problem = make_problem(
            [
                hard("<=", 10, {"x1": 6, "x2": 5}),
                soft("==", 1, {"x3": 1}, weight=4.0),
            ]
        )
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 1})
        assert result.feasible is True
        assert result.hard_violations == []
        assert result.soft_violations == []
        assert result.soft_violation_score == 0.0
        assert len(result.evaluations) == 2
        assert all(evaluation.satisfied for evaluation in result.evaluations)
        assert all(
            evaluation.violation_amount == 0.0 for evaluation in result.evaluations
        )

    def test_evaluation_fields(self):
        problem = make_problem([hard("<=", 10, {"x1": 6, "x2": 5})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        evaluation = result.evaluations[0]
        assert evaluation.constraint_id == "hard_<=_10"
        assert evaluation.constraint_type == "hard"
        assert evaluation.actual_value == 6
        assert evaluation.operator == "<="
        assert evaluation.expected_value == 10
        assert evaluation.weighted_penalty is None


class TestHardViolation:
    def test_violated_hard_constraint(self):
        problem = make_problem([hard("<=", 10, {"x1": 6, "x2": 5})])
        result = validate_solution(problem, {"x1": 1, "x2": 1, "x3": 0})
        assert result.feasible is False
        assert len(result.hard_violations) == 1
        evaluation = result.hard_violations[0]
        assert evaluation.satisfied is False
        assert evaluation.actual_value == 11
        assert evaluation.violation_amount == 1
        assert evaluation.weighted_penalty is None
        assert result.soft_violation_score == 0.0

    def test_one_violated_among_many(self):
        problem = make_problem(
            [
                hard("<=", 10, {"x1": 6, "x2": 5}),
                hard(">=", 1, {"x3": 1}),
            ]
        )
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.feasible is False
        assert [e.constraint_id for e in result.hard_violations] == ["hard_>=_1"]


class TestSoftViolation:
    def test_weighted_penalty_is_weight_times_violation_squared(self):
        problem = make_problem([soft("==", 3, {"x1": 1}, weight=4.0)])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.feasible is True  # only soft constraints violated
        assert len(result.soft_violations) == 1
        evaluation = result.soft_violations[0]
        assert evaluation.violation_amount == 2  # |1 - 3|
        assert evaluation.weighted_penalty == 4.0 * 2**2
        assert result.soft_violation_score == 16.0

    def test_score_sums_over_soft_violations(self):
        problem = make_problem(
            [
                soft("==", 2, {"x1": 1}, weight=3.0),  # violation 1 -> penalty 3
                soft(">=", 2, {"x2": 1}, weight=5.0),  # violation 2 -> penalty 20
            ]
        )
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.soft_violation_score == 3.0 + 20.0

    def test_satisfied_soft_constraint_contributes_zero(self):
        problem = make_problem([soft("<=", 1, {"x1": 1}, weight=9.0)])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.soft_violations == []
        assert result.evaluations[0].weighted_penalty == 0.0
        assert result.soft_violation_score == 0.0


class TestTolerance:
    """§23.1: EPSILON = 1e-8 on each operator's boundary."""

    def test_eq_within_epsilon_satisfied(self):
        problem = make_problem([hard("==", 1 + 0.5 * EPSILON, {"x1": 1})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.feasible is True
        assert result.evaluations[0].violation_amount == 0.0

    def test_eq_beyond_epsilon_violated(self):
        problem = make_problem([hard("==", 1 + 3 * EPSILON, {"x1": 1})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.feasible is False
        assert result.evaluations[0].violation_amount == pytest.approx(3 * EPSILON)

    def test_le_within_epsilon_satisfied(self):
        problem = make_problem([hard("<=", 1 - 0.5 * EPSILON, {"x1": 1})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.feasible is True
        assert result.evaluations[0].violation_amount == 0.0

    def test_le_beyond_epsilon_violated(self):
        problem = make_problem([hard("<=", 1 - 3 * EPSILON, {"x1": 1})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.feasible is False

    def test_ge_within_epsilon_satisfied(self):
        problem = make_problem([hard(">=", 1 + 0.5 * EPSILON, {"x1": 1})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.feasible is True
        assert result.evaluations[0].violation_amount == 0.0

    def test_ge_beyond_epsilon_violated(self):
        problem = make_problem([hard(">=", 1 + 3 * EPSILON, {"x1": 1})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.feasible is False


class TestViolationAmount:
    def test_eq_uses_absolute_difference(self):
        problem = make_problem([hard("==", 2, {"x1": 5})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.evaluations[0].violation_amount == 3  # |5 - 2|
        below = validate_solution(problem, {"x1": 0, "x2": 0, "x3": 0})
        assert below.evaluations[0].violation_amount == 2  # |0 - 2|

    def test_le_uses_positive_excess(self):
        problem = make_problem([hard("<=", 3, {"x1": 5})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.evaluations[0].violation_amount == 2  # max(0, 5 - 3)

    def test_ge_uses_positive_shortfall(self):
        problem = make_problem([hard(">=", 4, {"x1": 1})])
        result = validate_solution(problem, {"x1": 1, "x2": 0, "x3": 0})
        assert result.evaluations[0].violation_amount == 3  # max(0, 4 - 1)


class TestPurity:
    def test_inputs_not_mutated(self):
        problem = make_problem([hard("<=", 10, {"x1": 6, "x2": 5})])
        snapshot = problem.model_copy(deep=True)
        sample = {"x1": 1, "x2": 1, "x3": 0}
        validate_solution(problem, sample)
        assert problem == snapshot
        assert sample == {"x1": 1, "x2": 1, "x3": 0}
