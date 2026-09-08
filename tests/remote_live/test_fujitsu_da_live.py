"""Live Fujitsu Digital Annealer test (3b spec §26.6): smallest knapsack.

Opt-in only — consumes the account's paid DA quota and one of its job
slots. Two gates have to open before this runs (see ``conftest.py``):
``pytest -m remote`` and a ``FUJITSU_DA_API_KEY`` in the environment.

``REQUIRED_ENV`` below is what the shared ``_require_live_opt_in`` fixture
reads, so this module skips on the *Fujitsu* key rather than the D-Wave
one. That skip is the only kind of skip the suite allows (spec §28, §36),
and it is deliberate here for the same reason as for the D-Wave live
tests: without a Fujitsu account this file simply never runs (decision 3
of the 3b plan) — it hides no bug, it only prevents accidental spend.

``time_limit_seconds`` is 1, the smallest value the schema accepts, so a
run costs the minimum possible solver time.
"""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

pytestmark = pytest.mark.remote

# Read by tests/remote_live/conftest.py::_require_live_opt_in.
REQUIRED_ENV = "FUJITSU_DA_API_KEY"


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
                "backend": "fujitsu_da",
                "fujitsu_da": {"time_limit_seconds": 1},
            },
        }
    )


def test_fujitsu_da_solves_minimal_knapsack(live_policy):
    result = OptimizationService(policy=live_policy).solve(_minimal_knapsack())

    assert result.status == "success", (result.status, result.errors, result.message)
    assert result.backend == "fujitsu_da"
    assert result.solutions

    best = result.solutions[0]
    # Trust only the re-validation against the original JSON, never energy.
    hard = [e for e in best.constraint_evaluations if e.constraint_type == "hard"]
    assert hard and all(e.satisfied for e in hard)
    recomputed = 3 * best.variables["a"] + 2 * best.variables["b"] + 2 * best.variables["c"]
    assert best.objective_value == recomputed
    weight = 2 * best.variables["a"] + 2 * best.variables["b"] + 1 * best.variables["c"]
    assert weight <= 3

    assert result.metadata is not None
    assert result.metadata.solver_id == "fujitsuDA3/v4"
    assert result.metadata.backend == "fujitsu_da"
    assert result.metadata.remote is True
