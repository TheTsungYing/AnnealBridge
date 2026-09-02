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
``_dwave_system_installed`` is patched.
"""

import logging

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers import DWaveQPUBackend, SolverRegistry
import annealbridge.solvers.dwave_qpu as qpu_module

# Hyphenated on purpose: not matched by the DEV- pattern. Never a real token.
FAKE_TOKEN = "DEV-FAKE-TOKEN-1234567890abcdefghij"


class LeakyQPUSampler:
    """Fake sampler whose failure text embeds the configured token."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.sample_calls = 0

    def sample(self, bqm, **kwargs):
        self.sample_calls += 1
        raise RuntimeError(self.message)


def make_problem() -> OptimizationProblem:
    """maximize 2a + b s.t. a + b <= 1 on the remote QPU backend."""
    return OptimizationProblem.model_validate(
        {
            "name": "credential leak problem",
            "variables": [{"name": "a"}, {"name": "b"}],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "a", "coefficient": 2},
                    {"variable": "b", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "at_most_one",
                    "type": "hard",
                    "terms": [
                        {"variable": "a", "coefficient": 1},
                        {"variable": "b", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
            "solver": {"backend": "dwave_qpu"},
        }
    )


def solve_with_leaky_sampler(monkeypatch, caplog, message: str):
    """Run a full remote solve whose sampler raises ``message``."""
    monkeypatch.setenv("DWAVE_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(qpu_module, "_dwave_system_installed", lambda: True)
    # The env token alone makes ocean_config_status() report "ok"; that path
    # is under test, so it is deliberately not patched.
    assert qpu_module.ocean_config_status() == "ok"

    fake = LeakyQPUSampler(message)
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
