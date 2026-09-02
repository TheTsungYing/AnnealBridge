"""Unit tests for the penalty strategy (spec §18, §33)."""

import pytest

from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    QuadraticTerm,
    SolverPreferences,
    Variable,
)
from annealbridge.penalty import ScaledPenaltyStrategy


def make_problem(penalty_multiplier: float = 2.0) -> OptimizationProblem:
    """A small problem with |linear| = 10 + 8 and |quadratic| = 3 -> scale 21."""
    return OptimizationProblem(
        name="penalty-test",
        variables=[Variable(name="x"), Variable(name="y")],
        objective=Objective(
            direction="maximize",
            linear_terms=[
                LinearTerm(variable="x", coefficient=10),
                LinearTerm(variable="y", coefficient=-8),
            ],
            quadratic_terms=[
                QuadraticTerm(variable1="x", variable2="y", coefficient=-3)
            ],
        ),
        constraints=[
            Constraint(
                id="cap",
                type="hard",
                terms=[
                    LinearTerm(variable="x", coefficient=1),
                    LinearTerm(variable="y", coefficient=1),
                ],
                operator="<=",
                rhs=1,
            )
        ],
        solver=SolverPreferences(penalty_multiplier=penalty_multiplier),
    )


EXPECTED_SCALE = 21.0  # |10| + |-8| + |-3|


class TestScaledPenaltyStrategy:
    def test_objective_scale(self):
        strategy = ScaledPenaltyStrategy()
        assert strategy.objective_scale(make_problem()) == EXPECTED_SCALE

    def test_initial_penalty_uses_scale_and_multiplier(self):
        strategy = ScaledPenaltyStrategy()
        assert strategy.initial_penalty(make_problem()) == pytest.approx(
            EXPECTED_SCALE * 2.0
        )

    def test_deterministic_for_same_problem(self):
        strategy = ScaledPenaltyStrategy()
        first = strategy.initial_penalty(make_problem())
        second = strategy.initial_penalty(make_problem())
        assert first == second
        assert strategy.objective_scale(make_problem()) == strategy.objective_scale(
            make_problem()
        )

    def test_next_penalty_doubles_each_retry(self):
        strategy = ScaledPenaltyStrategy()
        penalty = strategy.initial_penalty(make_problem())
        for attempt in range(1, 4):
            following = strategy.next_penalty(penalty, attempt)
            assert following == pytest.approx(penalty * 2.0)
            assert following > penalty
            penalty = following

    def test_penalty_multiplier_override(self):
        strategy = ScaledPenaltyStrategy()
        assert strategy.initial_penalty(
            make_problem(penalty_multiplier=0.5)
        ) == pytest.approx(EXPECTED_SCALE * 0.5)
        assert strategy.initial_penalty(
            make_problem(penalty_multiplier=10.0)
        ) == pytest.approx(EXPECTED_SCALE * 10.0)
