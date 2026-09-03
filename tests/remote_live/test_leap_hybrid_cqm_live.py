"""Live Leap hybrid CQM test (3a spec §26.6): smallest possible knapsack.

Opt-in only — consumes Leap quota. See conftest.py for the gating fixtures.
``time_limit_seconds=5`` is the Leap CQM solver's documented minimum
(§17.6), so this is the cheapest possible submission.
"""

import logging

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
            "solver": {
                "backend": "leap_hybrid_cqm",
                "leap_hybrid_cqm": {"time_limit_seconds": 5},
            },
        }
    )


def test_leap_hybrid_cqm_solves_minimal_knapsack(live_policy, caplog, monkeypatch):
    import os

    token = os.environ["DWAVE_API_TOKEN"]
    caplog.set_level(logging.DEBUG, logger="annealbridge")

    result = OptimizationService(policy=live_policy).solve(_minimal_knapsack())

    assert result.status == "success", (result.status, result.errors, result.message)
    assert result.backend == "leap_hybrid_cqm"
    assert result.solutions
    assert result.metadata is not None
    assert result.metadata.model_type == "cqm"
    assert result.metadata.effective_time_limit_seconds >= 5
    # One attempt, no penalty: hard constraints travel natively (§16.3).
    assert len(result.attempts) == 1
    assert result.attempts[0].penalty is None

    best = result.solutions[0]
    # Trust only the re-validation against the original JSON, never energy.
    hard = [e for e in best.constraint_evaluations if e.constraint_type == "hard"]
    assert hard and all(e.satisfied for e in hard)
    recomputed = 3 * best.variables["a"] + 2 * best.variables["b"] + 2 * best.variables["c"]
    assert best.objective_value == recomputed
    weight = 2 * best.variables["a"] + 2 * best.variables["b"] + 1 * best.variables["c"]
    assert weight <= 3
    # No business variables were invented (no slack on the CQM path).
    assert set(best.variables) == {"a", "b", "c"}

    # Phase 2 §19 / 3a §28: the token never leaves through any channel.
    assert token not in result.model_dump_json()
    for record in caplog.records:
        assert token not in record.getMessage()
