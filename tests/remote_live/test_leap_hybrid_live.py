"""Live Leap Hybrid test (spec §28): smallest possible knapsack.

Opt-in only — consumes Leap quota. See conftest.py for the gating fixtures.
No time_limit is set, so the backend uses the sampler's min_time_limit.
"""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

pytestmark = pytest.mark.remote


def _minimal_knapsack() -> OptimizationProblem:
    """3-variable knapsack; optimum is {a, c} with value 5 and weight 3."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.0",
            "name": "live-minimal-knapsack",
            "variables": [
                {"name": "a", "type": "binary"},
                {"name": "b", "type": "binary"},
                {"name": "c", "type": "binary"},
            ],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "a", "coefficient": 3},
                    {"variable": "b", "coefficient": 2},
                    {"variable": "c", "coefficient": 2},
                ],
            },
            "constraints": [
                {
                    "id": "capacity",
                    "type": "hard",
                    "terms": [
                        {"variable": "a", "coefficient": 2},
                        {"variable": "b", "coefficient": 2},
                        {"variable": "c", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 3,
                }
            ],
            "solver": {"backend": "leap_hybrid_bqm"},
        }
    )


def test_leap_hybrid_solves_minimal_knapsack(live_policy):
    result = OptimizationService(policy=live_policy).solve(_minimal_knapsack())

    assert result.status == "success", (result.status, result.errors, result.message)
    assert result.backend == "leap_hybrid_bqm"
    # Leap Hybrid returns a single sample (documented limitation).
    assert len(result.solutions) == 1

    best = result.solutions[0]
    # Trust only the re-validation against the original JSON, never energy.
    hard = [e for e in best.constraint_evaluations if e.constraint_type == "hard"]
    assert hard and all(e.satisfied for e in hard)
    recomputed = 3 * best.variables["a"] + 2 * best.variables["b"] + 2 * best.variables["c"]
    assert best.objective_value == recomputed
    weight = 2 * best.variables["a"] + 2 * best.variables["b"] + 1 * best.variables["c"]
    assert weight <= 3
