"""``SolverExecutionError`` and the status it names (3b spec §20.8).

A remote backend can tell the service *how* a failure should be reported:
``SolverExecutionError.status`` names the ``SolveResult.status`` to use.
``None`` keeps the historical default, ``"solver_error"``; a backend sets
``"configuration_error"`` when the vendor rejected the shape of *our*
request (a malformed header, say), because telling the agent to change its
problem would be misleading.

The service half is proved through ``FakeDeclaredBackend``: a backend the
core has never heard of, whose ``solve()`` raises. Nothing in
``orchestration/`` names it, so this pins the generic path.
"""

import json
from pathlib import Path

import pytest

from annealbridge.exceptions import OptimizerError, SolverExecutionError
from annealbridge.models import OptimizationProblem, SolverPreferences
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import SolverRegistry
from tests.fakes.declared_backend import (
    FAKE_DECLARED_NAME,
    FAKE_LIMIT_KEY,
    FakeDeclaredBackend,
)

KNAPSACK = Path(__file__).resolve().parents[2] / "examples" / "knapsack.json"


class TestSolverExecutionError:
    def test_defaults(self):
        error = SolverExecutionError("m")

        assert isinstance(error, OptimizerError)
        assert str(error) == "m"
        assert error.code is None
        assert error.status is None

    def test_code_and_status_are_kept(self):
        error = SolverExecutionError("m", code="X", status="configuration_error")

        assert error.code == "X"
        assert error.status == "configuration_error"


class RaisingBackend(FakeDeclaredBackend):
    """A declared backend whose ``solve()`` always fails."""

    def __init__(self, error: SolverExecutionError) -> None:
        super().__init__()
        self._error = error

    def solve(self, compiled_problem, preferences):
        self.solve_calls += 1
        self.last_preferences = preferences
        raise self._error


class ConfigurationErrorBackend(RaisingBackend):
    """Names ``configuration_error``: the vendor rejected our request shape."""

    def __init__(self) -> None:
        super().__init__(
            SolverExecutionError(
                "bad header",
                code="BACKEND_CONFIG_INVALID",
                status="configuration_error",
            )
        )


class StatuslessBackend(RaisingBackend):
    """Names no status, so the service keeps the ``solver_error`` default."""

    def __init__(self) -> None:
        super().__init__(SolverExecutionError("gateway timed out", code="REMOTE_TIMEOUT"))


def make_service(backend) -> OptimizationService:
    return OptimizationService(
        registry=SolverRegistry({FAKE_DECLARED_NAME: backend}),
        policy=ExecutionPolicy(allow_remote=True, limits={FAKE_LIMIT_KEY: 1000}),
    )


def make_knapsack() -> OptimizationProblem:
    """``examples/knapsack.json`` routed to the declared test backend.

    ``SolverPreferences.backend`` is a Literal of the shipped backend
    names, so a test-only backend cannot be named through
    ``model_validate``; ``model_construct`` takes the same path every real
    backend takes once its name is in the Literal.
    """
    problem = OptimizationProblem.model_validate(
        json.loads(KNAPSACK.read_text(encoding="utf-8"))
    )
    solver = SolverPreferences.model_construct(backend=FAKE_DECLARED_NAME)
    return problem.model_copy(update={"solver": solver})


class TestServiceHonoursTheNamedStatus:
    @pytest.fixture
    def configuration_error_backend(self) -> ConfigurationErrorBackend:
        return ConfigurationErrorBackend()

    @pytest.fixture
    def statusless_backend(self) -> StatuslessBackend:
        return StatuslessBackend()

    def test_named_status_reaches_the_result(self, configuration_error_backend):
        result = make_service(configuration_error_backend).solve(make_knapsack())

        assert configuration_error_backend.solve_calls == 1
        assert result.status == "configuration_error"
        assert result.backend == FAKE_DECLARED_NAME
        assert result.solutions == []
        (error,) = result.errors
        assert error.code == "BACKEND_CONFIG_INVALID"
        assert "bad header" in error.message

    def test_without_a_status_the_default_is_solver_error(self, statusless_backend):
        result = make_service(statusless_backend).solve(make_knapsack())

        assert statusless_backend.solve_calls == 1
        assert result.status == "solver_error"
        (error,) = result.errors
        assert error.code == "REMOTE_TIMEOUT"
