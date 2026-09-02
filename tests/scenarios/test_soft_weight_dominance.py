"""Regression: a soft weight far above the objective must not hide hard
constraints from simulated annealing (spec §18 ``penalty_scale``).

Before the penalty scale accounted for soft terms, ``initial_penalty`` was
``objective_scale * multiplier`` (here 2 * 2 = 4) and doubling it four times
never caught up with a soft term worth 1000 * 2^2 = 4000 energy, so every
attempt reported zero feasible samples although ``x = 1, y = 0`` is feasible.
"""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

KNAPSACK_OPTIMUM_VALUE = 17.0
SEED = 1


def soft_dominated_problem(backend: str) -> OptimizationProblem:
    """minimize x + y; hard x + y >= 1; soft x + y == 0 with weight 1000."""
    return OptimizationProblem.model_validate(
        {
            "name": "soft-dominated",
            "variables": [{"name": "x"}, {"name": "y"}],
            "objective": {
                "direction": "minimize",
                "linear_terms": [
                    {"variable": "x", "coefficient": 1},
                    {"variable": "y", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "at_least_one",
                    "type": "hard",
                    "terms": [
                        {"variable": "x", "coefficient": 1},
                        {"variable": "y", "coefficient": 1},
                    ],
                    "operator": ">=",
                    "rhs": 1,
                },
                {
                    "id": "prefer_none",
                    "type": "soft",
                    "weight": 1000,
                    "terms": [
                        {"variable": "x", "coefficient": 1},
                        {"variable": "y", "coefficient": 1},
                    ],
                    "operator": "==",
                    "rhs": 0,
                },
            ],
            "solver": {"backend": backend, "seed": SEED},
        }
    )


class TestSoftWeightDominatesObjective:
    def test_simulated_annealing_finds_the_feasible_optimum(self):
        result = OptimizationService().solve(soft_dominated_problem("simulated_annealing"))

        assert result.status == "success"
        # penalty_scale = 2 + 1000 * 2^2 = 4002 -> initial penalty 8004, so
        # the very first attempt is already feasible: no retry needed.
        assert len(result.attempts) == 1
        assert result.attempts[0].penalty == pytest.approx(8004.0)
        assert result.attempts[0].feasible_samples > 0
        assert result.solutions[0].objective_value == pytest.approx(1.0)
        assert result.solutions[0].hard_constraints_satisfied is True

    def test_matches_exact_backend(self):
        sa = OptimizationService().solve(soft_dominated_problem("simulated_annealing"))
        exact = OptimizationService().solve(soft_dominated_problem("exact"))

        assert exact.status == "success"
        assert sa.solutions[0].objective_value == pytest.approx(
            exact.solutions[0].objective_value
        )
        assert sa.attempts[0].penalty == pytest.approx(exact.attempts[0].penalty)


@pytest.fixture
def knapsack_with_soft_take_everything(load_example):
    """The Phase 1 knapsack plus a soft 'take every item' preference.

    Taking all four items weighs 18 > capacity 10, so the soft preference
    pulls straight into the infeasible region; with ``weight`` far above the
    objective scale (31) the old objective-only penalty could not compete.
    """

    def _build(weight: float) -> OptimizationProblem:
        data = load_example(
            "knapsack.json", backend="simulated_annealing", seed=SEED, num_reads=100
        )
        data["constraints"].append(
            {
                "id": "take_everything",
                "type": "soft",
                "weight": weight,
                "terms": [
                    {"variable": name, "coefficient": 1}
                    for name in ("item_a", "item_b", "item_c", "item_d")
                ],
                "operator": "==",
                "rhs": 4,
            }
        )
        return OptimizationProblem.model_validate(data)

    return _build


class TestKnapsackWithDominantSoftWeight:
    WEIGHT = 10_000.0

    def test_simulated_annealing_stays_feasible(self, knapsack_with_soft_take_everything):
        result = OptimizationService().solve(knapsack_with_soft_take_everything(self.WEIGHT))

        assert result.status == "success"
        assert result.attempts[0].feasible_samples > 0
        for solution in result.solutions:
            assert solution.hard_constraints_satisfied is True
            total_weight = sum(
                w for name, w in (("item_a", 6), ("item_b", 5), ("item_c", 4), ("item_d", 3))
                if solution.variables[name] == 1
            )
            assert total_weight <= 10

    def test_penalty_scale_includes_soft_bound(self, knapsack_with_soft_take_everything):
        # objective_scale 31; soft max |sum - 4| over binaries is 4 -> 10000 * 16.
        result = OptimizationService().solve(knapsack_with_soft_take_everything(self.WEIGHT))
        assert result.attempts[0].penalty == pytest.approx((31.0 + 160_000.0) * 2.0)

    def test_best_solution_is_the_feasible_ranking_optimum(self, knapsack_with_soft_take_everything):
        # No feasible subset holds more than two items (the lightest three,
        # b + c + d, weigh 12 > 10), so every two-item subset carries the same
        # soft violation (4 - 2)^2 and ranking falls back to the objective:
        # {a, c} = 17, the Phase 1 optimum.
        result = OptimizationService().solve(knapsack_with_soft_take_everything(self.WEIGHT))
        assert result.solutions[0].objective_value == pytest.approx(KNAPSACK_OPTIMUM_VALUE)
