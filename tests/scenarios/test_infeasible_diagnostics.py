"""Scenario: an infeasible problem explains itself.

JSON -> ``OptimizationProblem`` -> ``OptimizationService`` -> result, the
same route an MCP or CLI caller takes. Two hard constraints that cannot
hold together: ``all_three`` wants every item, ``budget`` allows at most
the cheapest one. Neither is unsatisfiable on its own, so the problem
validator passes it and only the candidate re-validation can say why
nothing worked.

The diagnosis is the validator's view of the *last* attempt, never the
solver's: the closest candidate is chosen by the smallest total hard
violation recomputed from the original problem, and each hard constraint's
rate counts that attempt's deduplicated candidates.
"""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

SEED = 7


def jointly_infeasible_problem(backend: str, **solver_overrides) -> OptimizationProblem:
    """maximize x1 + x2 + x3, hard x1+x2+x3 >= 3 and 3x1+2x2+x3 <= 1."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.0",
            "name": "jointly-infeasible",
            "variables": [{"name": "x1"}, {"name": "x2"}, {"name": "x3"}],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "x1", "coefficient": 1},
                    {"variable": "x2", "coefficient": 1},
                    {"variable": "x3", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "all_three",
                    "type": "hard",
                    "terms": [
                        {"variable": "x1", "coefficient": 1},
                        {"variable": "x2", "coefficient": 1},
                        {"variable": "x3", "coefficient": 1},
                    ],
                    "operator": ">=",
                    "rhs": 3,
                },
                {
                    "id": "budget",
                    "type": "hard",
                    "terms": [
                        {"variable": "x1", "coefficient": 3},
                        {"variable": "x2", "coefficient": 2},
                        {"variable": "x3", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 1,
                },
            ],
            "solver": {"backend": backend, **solver_overrides},
        }
    )


class TestExhaustiveDiagnosis:
    """The exhaustive backend enumerates all eight assignments, so every
    number below is a complete census, not a sample."""

    @pytest.fixture
    def result(self):
        return OptimizationService().solve(jointly_infeasible_problem("exact"))

    def test_status_is_a_proven_infeasibility(self, result):
        assert result.status == "infeasible"
        assert result.infeasibility_proven is True
        assert result.solutions == []
        assert len(result.attempts) == 1
        assert result.attempts[0].unique_samples == 8
        assert result.attempts[0].feasible_samples == 0

    def test_closest_candidate_is_the_cheapest_single_item(self, result):
        # x3 alone: budget holds exactly (1 <= 1) and all_three misses by 2.
        # Every other assignment is further away in total.
        closest = result.infeasibility.closest_candidate
        assert closest.variables == {"x1": 0, "x2": 0, "x3": 1}
        assert closest.hard_violation_total == 2.0

    def test_closest_candidate_carries_the_validator_s_evaluations(self, result):
        evaluations = result.infeasibility.closest_candidate.constraint_evaluations
        assert [
            (e.constraint_id, e.constraint_type, e.satisfied, e.violation_amount)
            for e in evaluations
        ] == [
            ("all_three", "hard", False, 2.0),
            ("budget", "hard", True, 0.0),
        ]
        # The total is exactly the sum of those hard violation amounts.
        assert result.infeasibility.closest_candidate.hard_violation_total == sum(
            e.violation_amount for e in evaluations if e.constraint_type == "hard"
        )

    def test_rates_are_a_census_over_all_eight_assignments(self, result):
        rates = result.infeasibility.hard_violation_rates
        assert [
            (r.constraint_id, r.violated_candidates, r.candidates, r.violated_fraction)
            for r in rates
        ] == [
            ("all_three", 7, 8, 0.875),  # only x1=x2=x3=1 satisfies it
            ("budget", 6, 8, 0.75),  # only {} and {x3} stay within 1
        ]
        assert all(
            rate.candidates == result.attempts[-1].unique_samples for rate in rates
        )


class TestHeuristicDiagnosis:
    """The same problem on a heuristic backend: nothing is proven, but the
    attempt that ran is still explained."""

    @pytest.fixture
    def result(self):
        return OptimizationService().solve(
            jointly_infeasible_problem("simulated_annealing", seed=SEED, num_reads=50)
        )

    def test_infeasible_but_not_proven(self, result):
        assert result.status == "infeasible"
        assert result.infeasibility_proven is False
        assert result.solutions == []

    def test_diagnosis_describes_the_last_attempt(self, result):
        diagnostics = result.infeasibility
        assert diagnostics is not None
        assert diagnostics.closest_candidate.hard_violation_total > 0
        assert [r.constraint_id for r in diagnostics.hard_violation_rates] == [
            "all_three",
            "budget",
        ]
        # Retries mean several attempts; the rates count the last one's
        # deduplicated candidates, not the whole run's.
        last = result.attempts[-1]
        assert last.feasible_samples == 0
        for rate in diagnostics.hard_violation_rates:
            assert rate.candidates == last.unique_samples
            assert 0 <= rate.violated_candidates <= rate.candidates
            assert rate.violated_fraction == pytest.approx(
                rate.violated_candidates / rate.candidates
            )

    def test_every_hard_constraint_fails_somewhere_in_the_closest_candidate(
        self, result
    ):
        closest = result.infeasibility.closest_candidate
        assert set(closest.variables) == {"x1", "x2", "x3"}
        assert [e.constraint_id for e in closest.constraint_evaluations] == [
            "all_three",
            "budget",
        ]
        assert not all(e.satisfied for e in closest.constraint_evaluations)
