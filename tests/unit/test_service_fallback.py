"""Service-layer guarantees of ``solve()`` (2026-09-09 review F-03).

``OptimizationService.solve`` promises never to raise for domain errors
(Phase 2 spec §14 step 10: any backend exception → ``solver_error``;
Phase 1 §36). Before this review the promise was outsourced to each
backend's own wrapping, so four paths let raw exceptions escape to the
MCP / CLI caller. Two are pinned here through ``FakeDeclaredBackend``, a
backend the core has never heard of:

* a result that lacks a business variable (the compiler's decode raises a
  bare ``ValueError``); and
* a backend whose ``solve()`` raises something that is not an
  ``OptimizerError`` — a third-party plugin, or code outside
  ``guarded_call`` — whose text may embed a credential.

The third path, a construction-time check of every declared preference
path (3a spec §11.3: fail when the service is built, not on every solve),
is pinned at the bottom. The fourth (``SolverExecutionError.status``) lives
in ``test_exceptions.py``.
"""

import json
import logging
from pathlib import Path

import pytest

from annealbridge.interfaces.composition import SettingsError, build_state_from_policy
from annealbridge.models import (
    OptimizationProblem,
    ParameterLimit,
    SolverCapabilities,
    SolverExecutionMetadata,
    SolverPreferences,
)
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import RawSolverResult, SolverRegistry
from tests.fakes.declared_backend import (
    FAKE_DECLARED_NAME,
    FAKE_LIMIT_ERROR_CODE,
    FAKE_LIMIT_KEY,
    FakeDeclaredBackend,
)

KNAPSACK = Path(__file__).resolve().parents[2] / "examples" / "knapsack.json"
FAKE_KEY = "fake-key-ABC123"


def make_service(backend, **policy_overrides) -> OptimizationService:
    policy = ExecutionPolicy(
        allow_remote=True, limits={FAKE_LIMIT_KEY: 1000}, **policy_overrides
    )
    return OptimizationService(
        registry=SolverRegistry({FAKE_DECLARED_NAME: backend}), policy=policy
    )


def make_knapsack(**solver_overrides) -> OptimizationProblem:
    problem = OptimizationProblem.model_validate(
        json.loads(KNAPSACK.read_text(encoding="utf-8"))
    )
    solver = SolverPreferences.model_construct(
        backend=FAKE_DECLARED_NAME, **solver_overrides
    )
    return problem.model_copy(update={"solver": solver})


class ScriptedBackend(FakeDeclaredBackend):
    """A declared backend whose ``solve()`` follows a per-call script.

    Each step is either an exception instance (raised as-is, so the test
    controls the class, ``BaseException`` included), a ``RawSolverResult``
    returned verbatim, or ``None`` for the fake's normal behaviour.
    """

    def __init__(self, steps: list[object], **kwargs) -> None:
        super().__init__(**kwargs)
        self._steps = list(steps)

    def solve(self, compiled_problem, preferences):
        step = self._steps.pop(0) if self._steps else None
        if isinstance(step, BaseException):
            self.solve_calls += 1
            self.last_preferences = preferences
            raise step
        if isinstance(step, RawSolverResult):
            self.solve_calls += 1
            self.last_preferences = preferences
            return step
        return super().solve(compiled_problem, preferences)


def missing_column_result() -> RawSolverResult:
    """A result carrying only ``item_a``; the knapsack has four items."""
    return RawSolverResult.from_dicts(
        [{"item_a": 1}], [0.0], backend=FAKE_DECLARED_NAME, variables=["item_a"]
    )


class TestUnexpectedFailuresBecomeSolverError:
    def test_result_lacking_a_business_variable_is_a_solver_error(self):
        backend = ScriptedBackend([missing_column_result()])

        result = make_service(backend).solve(make_knapsack())

        assert backend.solve_calls == 1
        assert result.status == "solver_error"
        assert result.backend == FAKE_DECLARED_NAME
        assert result.solutions == []
        (error,) = result.errors
        assert error.code == "SOLVER_ERROR"
        assert "ValueError" in error.message
        assert "item_b" in error.message
        assert result.message == error.message

    def test_non_optimizer_error_is_a_redacted_solver_error(self, monkeypatch, caplog):
        monkeypatch.setenv("FUJITSU_DA_API_KEY", FAKE_KEY)
        backend = ScriptedBackend([RuntimeError(f"vendor sdk blew up with {FAKE_KEY}")])
        caplog.set_level(logging.DEBUG, logger="annealbridge")

        result = make_service(backend).solve(make_knapsack())

        assert backend.solve_calls == 1
        assert result.status == "solver_error"
        (error,) = result.errors
        assert error.code == "SOLVER_ERROR"
        # The class name is categorical and stays; the text is redacted.
        assert "RuntimeError" in error.message
        assert "***" in error.message
        assert FAKE_KEY not in result.model_dump_json()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "the fallback must leave a WARNING trace"
        assert any("RuntimeError" in r.getMessage() for r in warnings)
        for record in caplog.records:
            assert FAKE_KEY not in record.getMessage()

    def test_completed_attempts_and_metadata_are_kept(self):
        first = SolverExecutionMetadata(
            backend=FAKE_DECLARED_NAME, remote=True, solver_id="first-attempt"
        )
        # All four items weigh 18 > capacity 10: attempt 1 is infeasible, so
        # the hard-penalty ladder retries and attempt 2 blows up.
        infeasible = RawSolverResult.from_dicts(
            [{"item_a": 1, "item_b": 1, "item_c": 1, "item_d": 1}],
            [0.0],
            backend=FAKE_DECLARED_NAME,
            metadata=first,
        )
        backend = ScriptedBackend([infeasible, RuntimeError("second attempt died")])

        result = make_service(backend, allow_remote_retries=True).solve(
            make_knapsack(max_retries=2)
        )

        assert backend.solve_calls == 2
        assert result.status == "solver_error"
        assert len(result.attempts) == 1
        assert result.attempts[0].attempt == 1
        assert result.attempts[0].feasible_samples == 0
        assert result.metadata is not None
        assert result.metadata.solver_id == "first-attempt"
        assert result.metadata.model_type == "bqm"

    def test_base_exceptions_are_not_swallowed(self):
        backend = ScriptedBackend([KeyboardInterrupt(), None])
        service = make_service(backend, max_concurrent_solves=1)

        with pytest.raises(KeyboardInterrupt):
            service.solve(make_knapsack())

        # The concurrency slot was released on the way out: a second solve
        # runs instead of reporting CONCURRENCY_LIMIT.
        result = service.solve(make_knapsack())
        assert result.status == "success"
        assert backend.solve_calls == 2


def capabilities_with(preference: str) -> SolverCapabilities:
    base = FakeDeclaredBackend().capabilities
    return base.model_copy(
        update={
            "parameter_limits": [
                ParameterLimit(
                    preference=preference,
                    limit=FAKE_LIMIT_KEY,
                    error_code=FAKE_LIMIT_ERROR_CODE,
                )
            ]
        }
    )


class MisdeclaredBackend(FakeDeclaredBackend):
    """Declares a limit on a preference path that does not exist."""

    def __init__(self, preference: str) -> None:
        super().__init__()
        self._caps = capabilities_with(preference)

    @property
    def capabilities(self) -> SolverCapabilities:
        return self._caps


class TestDeclaredPreferencePathsAreCheckedAtConstruction:
    @pytest.mark.parametrize("path", ["num_readz", "fujitsu_da.nope", "num_reads.x"])
    def test_service_refuses_an_unknown_preference_path(self, path):
        with pytest.raises(ValueError, match=path.replace(".", r"\.")) as exc_info:
            make_service(MisdeclaredBackend(path))

        message = str(exc_info.value)
        assert FAKE_DECLARED_NAME in message
        assert FAKE_LIMIT_KEY in message

    def test_a_valid_declaration_still_builds(self):
        service = make_service(FakeDeclaredBackend())

        assert service.solve(make_knapsack()).status == "success"

    def test_composition_reports_it_as_a_settings_error(self):
        registry = SolverRegistry({FAKE_DECLARED_NAME: MisdeclaredBackend("num_readz")})
        policy = ExecutionPolicy(allow_remote=True, limits={FAKE_LIMIT_KEY: 1000})

        with pytest.raises(SettingsError, match="num_readz"):
            build_state_from_policy(policy, registry)
