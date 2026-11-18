"""An interrupted enumeration proves neither optimality nor infeasibility.

2026-11-18 review R-18. ``optimality_proven`` and ``infeasibility_proven``
rest on an exhaustive backend having enumerated every assignment; a result
the wall-clock limit cut short (``RawSolverResult.interrupted``) is partial
and must not set either flag. No shipped backend is exhaustive *and*
interruptible, so a test-only one (tests/fakes/exhaustive_interruptible_backend.py)
stands in. The service only hands a backend ``interrupt=`` when
``solver.wall_clock_limit_seconds`` is set, as in tests/unit/test_wall_clock_limit.py.
"""

import json

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.solvers import SolverRegistry
from tests.conftest import EXAMPLES_DIR
from tests.fakes.exhaustive_interruptible_backend import (
    SCRIPTED_EXHAUSTIVE_NAME,
    ScriptedExhaustiveBackend,
)
from tests.fakes.interruptible_backend import FakeClock, route

LIMIT_REACHED = "WALL_CLOCK_LIMIT_REACHED"
INFEASIBLE_CUT_PREFIX = (
    "No feasible solution found before solver.wall_clock_limit_seconds ran out"
)


def knapsack_problem() -> OptimizationProblem:
    """``examples/knapsack.json``: optimum {item_a, item_c}, value 17."""
    payload = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
    return OptimizationProblem.model_validate(payload)


def knapsack(*items: str) -> dict[str, int]:
    row = dict.fromkeys(["item_a", "item_b", "item_c", "item_d"], 0)
    for item in items:
        row[f"item_{item}"] = 1
    return row


FEASIBLE = [knapsack("a", "c"), knapsack("b", "d")]
INFEASIBLE = [knapsack("a", "b", "c", "d")]  # weight 18 > 10


def exhaustive_service(rows, action: str):
    backend = ScriptedExhaustiveBackend(rows, script=lambda call, interrupt: action)
    service = OptimizationService(
        registry=SolverRegistry({backend.name: backend}),
        clock=FakeClock(),
    )
    return backend, service


def on_exhaustive(**preferences) -> OptimizationProblem:
    return route(knapsack_problem(), SCRIPTED_EXHAUSTIVE_NAME, **preferences)


def codes(errors) -> list[str]:
    return [error.code for error in errors]


class TestTheFakeIsAccepted:
    def test_validate_accepts_a_limit_on_the_exhaustive_interruptible_backend(self):
        _, service = exhaustive_service(FEASIBLE, "full")

        result = service.validate(on_exhaustive(wall_clock_limit_seconds=10))

        assert result.valid
        assert result.errors == []


class TestInterruptedSuccess:
    def test_feasible_but_interrupted_proves_no_optimality(self):
        backend, service = exhaustive_service(FEASIBLE, "partial")

        result = service.solve(on_exhaustive(wall_clock_limit_seconds=10))

        assert backend.received_interrupt == [True]
        assert result.status == "success"
        assert result.optimality_proven is False
        assert "optimality is not proven" in result.message
        assert "proved optimality" not in result.message
        assert result.wall_clock_limit_reached is True
        assert LIMIT_REACHED in codes(result.warnings)
        assert result.solutions[0].variables == knapsack("a", "c")


class TestInterruptedInfeasible:
    def test_infeasible_and_interrupted_proves_no_infeasibility(self):
        backend, service = exhaustive_service(INFEASIBLE, "partial")

        result = service.solve(on_exhaustive(wall_clock_limit_seconds=10))

        assert backend.received_interrupt == [True]
        assert result.status == "infeasible"
        assert result.infeasibility_proven is False
        assert result.message.startswith(INFEASIBLE_CUT_PREFIX)
        assert "wall_clock_limit_seconds" in result.message
        assert "enumerated every assignment" not in result.message
        assert result.wall_clock_limit_reached is True
        assert LIMIT_REACHED in codes(result.warnings)


class TestNotInterruptedStillProves:
    """Regression: a complete enumeration keeps both proofs."""

    def test_feasible_and_complete_proves_optimality(self):
        backend, service = exhaustive_service(FEASIBLE, "full")

        result = service.solve(on_exhaustive(wall_clock_limit_seconds=10))

        assert backend.received_interrupt == [True]
        assert result.status == "success"
        assert result.optimality_proven is True
        assert "proved optimality" in result.message
        assert result.wall_clock_limit_reached is False
        assert LIMIT_REACHED not in codes(result.warnings)

    def test_infeasible_and_complete_proves_infeasibility(self):
        backend, service = exhaustive_service(INFEASIBLE, "full")

        result = service.solve(on_exhaustive(wall_clock_limit_seconds=10))

        assert backend.received_interrupt == [True]
        assert result.status == "infeasible"
        assert result.infeasibility_proven is True
        assert "enumerated every assignment" in result.message
        assert result.wall_clock_limit_reached is False

    def test_without_a_limit_the_backend_gets_no_interrupt_and_still_proves(self):
        backend, service = exhaustive_service(FEASIBLE, "full")

        result = service.solve(on_exhaustive())

        assert backend.received_interrupt == [False]
        assert result.status == "success"
        assert result.optimality_proven is True
