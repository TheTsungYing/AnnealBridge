"""Credential-leak tests for the remote solve path (Phase 2 spec §19, §27).

A fake D-Wave token is placed in ``DWAVE_API_TOKEN`` and a fake sampler
raises an exception whose text embeds it. Nothing that leaves the service —
the ``SolveResult`` JSON, its errors/warnings, or any ``annealbridge`` log
record — may contain the token.

The token deliberately contains hyphens, so it does *not* match the
``DEV-[A-Za-z0-9]{20,}`` redaction pattern: masking has to come from the
live environment-variable candidate that :func:`redact` resolves. With the
env var set, ``ocean_config_status()`` already reports ``"ok"`` (the
``dwave`` extra is not installed, so the env var is the only config
source), which is exactly the path under test — only
``dwave_system_installed`` is patched.
"""

import logging
import traceback

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers import (
    DWaveQPUBackend,
    LeapHybridBQMBackend,
    SolverRegistry,
)
import annealbridge.solvers.metadata as metadata_module
from tests.remote_mock.conftest import (
    FAKE_UNPATTERNED_TOKEN,
    FakeLeapHybridSampler,
    FakeQPUSampler,
    make_problem,
)

# Hyphenated on purpose: not matched by the DEV- pattern, so only the live
# environment-variable candidate can mask it. Never a real token.
FAKE_TOKEN = FAKE_UNPATTERNED_TOKEN


def solve_with_leaky_sampler(monkeypatch, caplog, message: str, *, lazy: bool = False):
    """Run a full remote solve whose sampler raises ``message``."""
    monkeypatch.setenv("DWAVE_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(metadata_module, "dwave_system_installed", lambda: True)
    # The env token alone makes ocean_config_status() report "ok"; that path
    # is under test, so it is deliberately not patched.
    assert metadata_module.ocean_config_status() == "ok"

    fake = FakeQPUSampler(raise_on_sample=RuntimeError(message), lazy=lazy)
    service = OptimizationService(
        registry=SolverRegistry({"dwave_qpu": DWaveQPUBackend(lambda: fake)}),
        policy=ExecutionPolicy(allow_remote=True),
    )
    caplog.set_level(logging.DEBUG, logger="annealbridge")
    result = service.solve(make_problem())
    assert fake.sample_calls == 1
    return result


class TestTokenNeverLeaves:
    # Each test drives its own solve so the log records land in the calling
    # test's ``caplog`` phase rather than in a fixture's setup phase.
    def solve(self, monkeypatch, caplog):
        return solve_with_leaky_sampler(
            monkeypatch,
            caplog,
            f"connection rejected for token={FAKE_TOKEN} "
            f"at https://cloud.dwavesys.com",
        )

    def test_solve_fails_as_a_structured_solver_error(self, monkeypatch, caplog):
        result = self.solve(monkeypatch, caplog)

        assert result.status == "solver_error"
        assert result.backend == "dwave_qpu"
        assert result.solutions == []
        assert result.errors

    def test_token_absent_from_the_serialized_result(self, monkeypatch, caplog):
        result = self.solve(monkeypatch, caplog)

        assert FAKE_TOKEN not in result.model_dump_json()

    def test_token_absent_from_every_log_record(self, monkeypatch, caplog):
        self.solve(monkeypatch, caplog)

        assert caplog.records  # the solve really did log something
        for record in caplog.records:
            assert FAKE_TOKEN not in record.getMessage()

    def test_token_absent_from_errors_and_warnings(self, monkeypatch, caplog):
        result = self.solve(monkeypatch, caplog)

        for entry in [*result.errors, *result.warnings]:
            assert FAKE_TOKEN not in entry.message

    def test_error_message_is_visibly_redacted(self, monkeypatch, caplog):
        result = self.solve(monkeypatch, caplog)

        assert "***" in result.errors[0].message

    def test_error_carries_catalog_guidance(self, monkeypatch, caplog):
        error = self.solve(monkeypatch, caplog).errors[0]

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert error.recommended_action is not None


class TestBareTokenIsMasked:
    """A token with no ``token=`` prefix: only the env candidate can mask it."""

    def test_bare_token_is_still_redacted(self, monkeypatch, caplog):
        result = solve_with_leaky_sampler(
            monkeypatch,
            caplog,
            f"connection rejected (credential {FAKE_TOKEN}) "
            f"at https://cloud.dwavesys.com",
        )

        assert result.status == "solver_error"
        assert FAKE_TOKEN not in result.model_dump_json()
        assert "***" in result.errors[0].message
        assert caplog.records
        for record in caplog.records:
            assert FAKE_TOKEN not in record.getMessage()


class TestLazyResolveFailureIsRedacted:
    """The token must not leak when the failure surfaces at resolve time."""

    def solve(self, monkeypatch, caplog):
        return solve_with_leaky_sampler(
            monkeypatch,
            caplog,
            f"request failed for token={FAKE_TOKEN} (credential {FAKE_TOKEN})",
            lazy=True,
        )

    def test_structured_error_instead_of_an_escaping_exception(self, monkeypatch, caplog):
        result = self.solve(monkeypatch, caplog)

        assert result.status == "solver_error"
        assert result.errors[0].code == "REMOTE_SOLVER_ERROR"

    def test_token_absent_from_result_and_logs(self, monkeypatch, caplog):
        result = self.solve(monkeypatch, caplog)

        assert FAKE_TOKEN not in result.model_dump_json()
        assert "***" in result.errors[0].message
        assert caplog.records
        for record in caplog.records:
            assert FAKE_TOKEN not in record.getMessage()


def compile_problem():
    return BQMCompiler().compile(make_problem(), hard_penalty=100.0)


def hybrid_preferences():
    return make_problem().solver.model_copy(update={"backend": "leap_hybrid_bqm"})


class TestExceptionChainCarriesNoToken:
    """The raw Ocean exception must not hang off the wrapped error: anything
    formatting the chain (``logger.exception``, traceback tooling) would
    otherwise print the unredacted text."""

    @pytest.fixture(autouse=True)
    def _token_in_env(self, monkeypatch):
        monkeypatch.setenv("DWAVE_API_TOKEN", FAKE_TOKEN)

    def assert_chain_is_clean(self, exc: BaseException) -> None:
        assert exc.__cause__ is None
        assert exc.__context__ is None
        formatted = "".join(traceback.format_exception(exc))
        assert FAKE_TOKEN not in formatted
        assert "***" in formatted

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def test_qpu_sample_failure(self, lazy):
        fake = FakeQPUSampler(
            raise_on_sample=RuntimeError(f"rejected token={FAKE_TOKEN}"), lazy=lazy
        )
        backend = DWaveQPUBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compile_problem(), make_problem().solver)

        self.assert_chain_is_clean(exc_info.value)

    def test_qpu_sampler_construction_failure(self):
        def failing_factory():
            raise ValueError(f"invalid endpoint for token={FAKE_TOKEN}")

        backend = DWaveQPUBackend(sampler_factory=failing_factory)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compile_problem(), make_problem().solver)

        assert exc_info.value.code == "DWAVE_CONFIG_INVALID"
        self.assert_chain_is_clean(exc_info.value)

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def test_hybrid_sample_failure(self, lazy):
        fake = FakeLeapHybridSampler(
            raise_on_sample=RuntimeError(f"rejected token={FAKE_TOKEN}"), lazy=lazy
        )
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compile_problem(), hybrid_preferences())

        assert fake.sample_calls == 1
        self.assert_chain_is_clean(exc_info.value)

    def test_hybrid_time_limit_resolution_failure(self):
        def failing_factory():
            raise RuntimeError(f"cannot reach solver, token={FAKE_TOKEN}")

        backend = LeapHybridBQMBackend(sampler_factory=failing_factory)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.resolve_time_limit(compile_problem(), hybrid_preferences())

        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"
        self.assert_chain_is_clean(exc_info.value)
