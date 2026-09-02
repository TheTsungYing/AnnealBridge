"""Live QPU test (spec §28): smallest possible knapsack, tiny num_reads.

Opt-in only — consumes Leap quota. See conftest.py for the gating fixtures.
"""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

pytestmark = pytest.mark.remote


def _minimal_knapsack(backend: str, **solver_overrides) -> OptimizationProblem:
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
            "solver": {"backend": backend, **solver_overrides},
        }
    )


def test_qpu_solves_minimal_knapsack(live_policy):
    # num_reads kept tiny on purpose: this is a wiring test, not a benchmark.
    problem = _minimal_knapsack("dwave_qpu", num_reads=10)
    result = OptimizationService(policy=live_policy).solve(problem)

    assert result.status == "success", (result.status, result.errors, result.message)
    assert result.backend == "dwave_qpu"
    assert result.solutions

    best = result.solutions[0]
    # Trust only the re-validation against the original JSON, never energy:
    # every hard constraint of the ranked best solution must hold, and the
    # objective value must equal a recomputation from the raw assignment.
    hard = [e for e in best.constraint_evaluations if e.constraint_type == "hard"]
    assert hard and all(e.satisfied for e in hard)
    recomputed = 3 * best.variables["a"] + 2 * best.variables["b"] + 2 * best.variables["c"]
    assert best.objective_value == recomputed
    weight = 2 * best.variables["a"] + 2 * best.variables["b"] + 1 * best.variables["c"]
    assert weight <= 3
