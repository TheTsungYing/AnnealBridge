"""Mock tests for FujitsuDABackend (Phase 3b spec §20, §26.5).

Never talks to the real Digital Annealer: a scripted
:class:`~tests.fakes.FakeDATransport` is injected through the backend's
only network seam, and ``clock`` / ``sleep`` are injected so the polling
loop is exercised without waiting. Every request the backend makes is
recorded, so the tests assert on the exact wire shape (URL, headers,
``binary_polynomial`` terms) as well as on the classification of every
row of the §20.8 failure table.

``FUJITSU_DA_API_KEY`` is deliberately a value that no redaction *pattern*
matches (see :data:`FAKE_KEY`), so "the key never leaks" can only pass
because ``redact`` masks the live environment value.
"""

import http.client
import json
import logging
import sys
import traceback
import urllib.error

import numpy as np
import pydantic
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    FujitsuDAOptions,
    OptimizationProblem,
    SolverPreferences,
)
from annealbridge.models.error_catalog import RETRYABLE_CODES
from annealbridge.orchestration import OptimizationService
from annealbridge.solvers import (
    AvailabilityStatus,
    ExactSolverBackend,
    FujitsuDABackend,
    SolverCapabilities,
)
from annealbridge.solvers.fujitsu_da import (
    DEFAULT_BASE_URL,
    REASON_API_KEY_MISSING,
    REASON_URL_NOT_HTTPS,
    SOLVER_ID,
)
from tests.fakes import (
    FAKE_JOB_ID,
    FakeDATransport,
    FakeSolution,
    bits_solution,
    json_response,
)
from tests.remote_mock.conftest import make_problem
from tests.remote_mock.test_service_remote_flow import (
    assert_actions_present,
    make_service,
    make_zero_infeasible_problem,
)

# Not matched by any pattern in ``_REDACTION_PATTERNS``: masking it can only
# come from the live environment-variable candidate. Never a real key.
FAKE_KEY = "fujitsu-fake-api-key-Z9Z9-0001"

SUBMIT_PATH = "/v4/async/qubo/solve"
RESULT_PATH = f"/v4/async/jobs/result/{FAKE_JOB_ID}"
CANCEL_PATH = "/v4/async/jobs/cancel"

PROBLEM_NAME = "remote mock problem"


def set_key(monkeypatch, key: str = FAKE_KEY) -> str:
    monkeypatch.setenv("FUJITSU_DA_API_KEY", key)
    return key


def make_compiled(**solver_overrides) -> CompiledProblem:
    """The shared 0/1 problem compiled for the DA backend (adds slack)."""
    return BQMCompiler().compile(
        make_problem(backend="fujitsu_da", **solver_overrides), hard_penalty=100.0
    )


def make_preferences(**options) -> SolverPreferences:
    """Preferences for the DA backend; no kwargs means no option block at all."""
    return SolverPreferences(
        backend="fujitsu_da",
        fujitsu_da=FujitsuDAOptions(**options) if options else None,
    )


def zero_rows(compiled: CompiledProblem, count: int = 1) -> list[FakeSolution]:
    """``count`` well-formed all-zero solutions with energies 0, 1, 2 …"""
    width = len(compiled.model.variables)
    return [bits_solution([0] * width, float(index)) for index in range(count)]


def make_backend(fake: FakeDATransport, **kwargs) -> tuple[FujitsuDABackend, list[float]]:
    """A backend wired to ``fake`` plus the list its ``sleep`` records into.

    ``clock`` is frozen at 0 so the polling deadline is never reached
    unless a test injects its own clock.
    """
    sleeps: list[float] = []
    kwargs.setdefault("clock", lambda: 0.0)
    backend = FujitsuDABackend(transport=fake, sleep=sleeps.append, **kwargs)
    return backend, sleeps


def solve(
    monkeypatch,
    fake: FakeDATransport,
    preferences: SolverPreferences | None = None,
    compiled: CompiledProblem | None = None,
    **kwargs,
):
    """Run one direct ``backend.solve()`` against the scripted transport."""
    set_key(monkeypatch)
    compiled = compiled if compiled is not None else make_compiled()
    backend, sleeps = make_backend(fake, **kwargs)
    return backend.solve(compiled, preferences or make_preferences()), sleeps


def expect_failure(monkeypatch, fake: FakeDATransport, **kwargs) -> SolverExecutionError:
    """Run a solve that must fail and return the raised error."""
    with pytest.raises(SolverExecutionError) as exc_info:
        solve(monkeypatch, fake, **kwargs)
    return exc_info.value


def expected_terms(bqm) -> list[dict]:
    """The ``binary_polynomial.terms`` the spec (§20.7 step 2) prescribes."""
    index_of = {variable: index for index, variable in enumerate(bqm.variables)}
    terms = [
        {"coefficient": float(bqm.linear[variable]), "polynomials": [index_of[variable]]}
        for variable in bqm.variables
    ]
    terms += [
        {"coefficient": float(bias), "polynomials": [index_of[u], index_of[v]]}
        for (u, v), bias in bqm.quadratic.items()
    ]
    terms.append({"coefficient": float(bqm.offset), "polynomials": []})
    return terms


def make_da_service(fake: FakeDATransport, **policy_kwargs) -> OptimizationService:
    """A service holding exactly one DA backend wired to ``fake``."""
    backend, _ = make_backend(fake)
    policy_kwargs.setdefault("allow_remote", True)
    return make_service(backend, "fujitsu_da", **policy_kwargs)


class TestCapabilities:
    """§20.1: what the backend declares about itself."""

    def test_capability_fields(self):
        capabilities = FujitsuDABackend().capabilities

        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "fujitsu_da"
        assert capabilities.remote is True
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is False
        assert capabilities.supports_num_reads is False
        assert capabilities.supports_time_limit is True
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is True
        assert capabilities.requires_embedding is False
        assert capabilities.supports_num_sweeps is False

    def test_declared_parameter_limits(self):
        capabilities = FujitsuDABackend().capabilities

        assert [
            (limit.preference, limit.limit, limit.error_code)
            for limit in capabilities.parameter_limits
        ] == [("fujitsu_da.time_limit_seconds", "time_seconds", "REMOTE_TIME_LIMIT")]

    def test_description_names_the_credential(self):
        assert "API key" in FujitsuDABackend().capabilities.description

    def test_properties_alias_capabilities(self):
        backend = FujitsuDABackend()

        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive

    def test_no_third_party_http_package_is_imported(self):
        # §20.4: HTTP goes through the standard library only. The backend
        # module is imported at the top of this file, so by the time this
        # runs the import has happened.
        assert "annealbridge.solvers.fujitsu_da" in sys.modules
        assert "requests" not in sys.modules
        assert "httpx" not in sys.modules


class TestRequestShape:
    """§20.7 step 2 / §26.5: the exact bytes that go on the wire."""

    def test_default_base_url_and_submit_path(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        assert fake.submit_request.url == f"{DEFAULT_BASE_URL}{SUBMIT_PATH}"

    def test_url_env_overrides_the_base_and_loses_its_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("FUJITSU_DA_URL", "https://example.test/da/")
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        assert fake.submit_request.url == f"https://example.test/da{SUBMIT_PATH}"

    def test_headers_are_exactly_the_three_declared_ones(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        headers = fake.submit_request.headers
        assert set(headers) == {"X-Api-Key", "Content-Type", "Accept"}
        assert "X-Access-Token" not in headers
        assert headers["X-Api-Key"] == FAKE_KEY
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    def test_every_request_carries_the_same_headers(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        assert fake.requests
        for recorded in fake.requests:
            assert set(recorded.headers) == {"X-Api-Key", "Content-Type", "Accept"}

    def test_binary_polynomial_matches_the_bqm_term_by_term(self, monkeypatch):
        compiled = make_compiled()
        fake = FakeDATransport(solutions=zero_rows(compiled))

        solve(monkeypatch, fake, compiled=compiled)

        terms = fake.submit_request.json["binary_polynomial"]["terms"]
        assert terms == expected_terms(compiled.model)

    def test_zero_bias_variables_and_the_offset_are_still_sent(self, monkeypatch):
        compiled = make_compiled()
        fake = FakeDATransport(solutions=zero_rows(compiled))

        solve(monkeypatch, fake, compiled=compiled)

        terms = fake.submit_request.json["binary_polynomial"]["terms"]
        linear = [term for term in terms if len(term["polynomials"]) == 1]
        constant = [term for term in terms if term["polynomials"] == []]
        # One linear term per variable, zero biases included, plus exactly
        # one constant term even when the offset is 0.
        assert len(linear) == len(compiled.model.variables)
        assert [term["polynomials"][0] for term in linear] == list(
            range(len(compiled.model.variables))
        )
        assert len(constant) == 1
        assert constant[0]["coefficient"] == float(compiled.model.offset)

    def test_time_limit_sec_is_the_resolved_value(self, monkeypatch):
        compiled = make_compiled()
        preferences = make_preferences(time_limit_seconds=42)
        fake = FakeDATransport(solutions=zero_rows(compiled))
        set_key(monkeypatch)
        backend, _ = make_backend(fake)

        backend.solve(compiled, preferences)

        expected = backend.resolve_time_limit(compiled, preferences)
        assert fake.submit_request.json["fujitsuDA3"]["time_limit_sec"] == int(expected)
        assert expected == 42.0

    def test_unset_options_are_omitted_and_only_the_default_time_limit_is_sent(
        self, monkeypatch
    ):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        assert fake.submit_request.json["fujitsuDA3"] == {"time_limit_sec": 10}

    def test_given_options_are_all_forwarded(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(
            monkeypatch,
            fake,
            make_preferences(num_run=8, num_group=2, num_output_solution=3),
        )

        assert fake.submit_request.json["fujitsuDA3"] == {
            "time_limit_sec": 10,
            "num_run": 8,
            "num_group": 2,
            "num_output_solution": 3,
        }

    def test_body_is_exactly_the_solver_block_and_the_polynomial(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        assert set(fake.submit_request.json) == {"fujitsuDA3", "binary_polynomial"}

    def test_no_business_strings_are_sent(self, monkeypatch):
        # §24: the vendor sees numbers, never the problem's own text.
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        body = fake.submit_request.body.decode("utf-8")
        assert PROBLEM_NAME not in body
        assert "description" not in body
        assert "at_most_one" not in body

    def test_request_timeout_is_forwarded_to_the_transport(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake, request_timeout_seconds=7.5)

        assert [recorded.timeout for recorded in fake.requests] == [7.5] * len(
            fake.requests
        )


class TestResolveTimeLimit:
    """§20.6: the service must learn the submitted value before submission."""

    @pytest.mark.parametrize(
        ("options", "expected"),
        [({}, 10.0), ({"time_limit_seconds": 30}, 30.0)],
        ids=["default", "user-value"],
    )
    def test_resolved_value(self, options, expected):
        fake = FakeDATransport()
        backend, _ = make_backend(fake)

        resolved = backend.resolve_time_limit(make_compiled(), make_preferences(**options))

        assert resolved == expected
        assert isinstance(resolved, float)
        assert fake.requests == []


class TestPolling:
    """§20.7 steps 3–5: poll until Done, then release the job slot."""

    def test_default_script_polls_three_times_and_sleeps_twice(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        _, sleeps = solve(monkeypatch, fake, poll_interval_seconds=1.5)

        assert fake.poll_calls == 3
        assert len(fake.requests_for("GET", RESULT_PATH)) == 3
        assert sleeps == [1.5, 1.5]

    def test_immediate_done_never_sleeps(self, monkeypatch):
        fake = FakeDATransport(
            solutions=zero_rows(make_compiled()), poll_statuses=("Done",)
        )

        _, sleeps = solve(monkeypatch, fake)

        assert fake.poll_calls == 1
        assert sleeps == []

    def test_result_is_deleted_exactly_once_after_done(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        assert fake.delete_calls == 1
        deletes = fake.requests_for("DELETE")
        assert len(deletes) == 1
        assert deletes[0].url == f"{DEFAULT_BASE_URL}{RESULT_PATH}"
        assert deletes[0].body is None

    def test_delete_http_failure_is_only_a_warning(self, monkeypatch, caplog):
        fake = FakeDATransport(
            solutions=zero_rows(make_compiled()),
            delete_response=(500, b"internal error"),
        )

        with caplog.at_level(logging.WARNING, logger="annealbridge.solvers.fujitsu_da"):
            result, _ = solve(monkeypatch, fake)

        assert result.num_samples == 1
        assert fake.delete_calls == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        assert "could not be deleted" in warnings[0].getMessage()

    def test_delete_exception_is_only_a_warning(self, monkeypatch, caplog):
        fake = FakeDATransport(
            solutions=zero_rows(make_compiled()), delete_response=OSError("boom")
        )

        with caplog.at_level(logging.WARNING, logger="annealbridge.solvers.fujitsu_da"):
            result, _ = solve(monkeypatch, fake)

        assert result.num_samples == 1
        assert fake.delete_calls == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        assert "could not be deleted" in warnings[0].getMessage()
        assert "OSError" in warnings[0].getMessage()

    def test_result_url_is_polled(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        solve(monkeypatch, fake)

        for recorded in fake.requests_for("GET"):
            assert recorded.url == f"{DEFAULT_BASE_URL}{RESULT_PATH}"
            assert recorded.body is None


class TestResultConversion:
    """§20.7 step 6: configuration → int8 rows, energies, metadata."""

    def test_samples_are_int8_rows_in_solution_order(self, monkeypatch):
        compiled = make_compiled()
        width = len(compiled.model.variables)
        first = [0] * width
        second = [1] + [0] * (width - 1)
        fake = FakeDATransport(
            solutions=[bits_solution(first, -1.0), bits_solution(second, -2.0)]
        )

        result, _ = solve(monkeypatch, fake, compiled=compiled)

        assert result.samples.dtype == np.int8
        assert result.samples.tolist() == [first, second]
        assert result.variables == [str(v) for v in compiled.model.variables]
        assert result.backend == "fujitsu_da"

    def test_energies_come_from_the_solutions(self, monkeypatch):
        compiled = make_compiled()
        width = len(compiled.model.variables)
        fake = FakeDATransport(
            solutions=[
                bits_solution([0] * width, -1.5),
                bits_solution([0] * width, 2.0),
                bits_solution([0] * width, 0.0),
            ]
        )

        result, _ = solve(monkeypatch, fake, compiled=compiled)

        assert result.energies.tolist() == [-1.5, 2.0, 0.0]

    def test_frequency_is_not_expanded_into_repeated_rows(self, monkeypatch):
        compiled = make_compiled()
        width = len(compiled.model.variables)
        fake = FakeDATransport(
            solutions=[bits_solution([0] * width, 0.0, frequency=4.0)]
        )

        result, _ = solve(monkeypatch, fake, compiled=compiled)

        assert result.num_samples == 1

    def test_empty_solution_list_is_zero_samples_not_an_error(self, monkeypatch):
        compiled = make_compiled()
        fake = FakeDATransport(solutions=[])

        result, _ = solve(monkeypatch, fake, compiled=compiled)

        assert result.num_samples == 0
        assert result.samples.shape == (0, len(compiled.model.variables))
        assert result.energies.tolist() == []

    def test_missing_variable_index_is_a_solver_error(self, monkeypatch):
        compiled = make_compiled()
        width = len(compiled.model.variables)
        solution = bits_solution([0] * width, 0.0)
        del solution.configuration[str(width - 1)]
        fake = FakeDATransport(solutions=[solution])

        error = expect_failure(monkeypatch, fake, compiled=compiled)

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert "variable index" in str(error)

    @pytest.mark.parametrize(
        "value", [1, "true", 1.0, None], ids=["int", "string", "float", "null"]
    )
    def test_non_boolean_configuration_value_is_a_solver_error(
        self, monkeypatch, value
    ):
        compiled = make_compiled()
        width = len(compiled.model.variables)
        solution = bits_solution([0] * width, 0.0)
        solution.configuration["0"] = value
        fake = FakeDATransport(solutions=[solution])

        error = expect_failure(monkeypatch, fake, compiled=compiled)

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert "variable index 0" in str(error)

    def test_metadata_timing_solver_id_and_time_limit(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        result, _ = solve(monkeypatch, fake, make_preferences(time_limit_seconds=12))

        metadata = result.metadata
        assert metadata is not None
        # The vendor reports milliseconds as strings; metadata is µs floats.
        assert metadata.timing_us == {
            "solve_time": 5041 * 1000.0,
            "total_elapsed_time": 5050 * 1000.0,
        }
        assert metadata.solver_id == SOLVER_ID == "fujitsuDA3/v4"
        assert metadata.effective_time_limit_seconds == 12.0
        assert metadata.backend == "fujitsu_da"
        assert metadata.remote is True

    def test_missing_timing_block_yields_no_timing(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()), timing=None)

        result, _ = solve(monkeypatch, fake)

        assert result.metadata.timing_us == {}

    def test_num_reads_requested_defaults_to_the_vendor_defaults(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        result, _ = solve(monkeypatch, fake)

        assert result.metadata.num_reads_requested == 5

    def test_num_reads_requested_multiplies_outputs_by_groups(self, monkeypatch):
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        result, _ = solve(
            monkeypatch, fake, make_preferences(num_output_solution=3, num_group=2)
        )

        assert result.metadata.num_reads_requested == 6


class TestPolicyLimits:
    """§26.5: preferences over a policy ceiling are refused, never clamped."""

    def test_time_limit_over_policy_is_refused_before_submission(self, monkeypatch):
        set_key(monkeypatch)
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))
        service = make_da_service(fake)

        result = service.solve(
            make_problem(backend="fujitsu_da", fujitsu_da={"time_limit_seconds": 400})
        )

        assert result.status == "resource_limit_exceeded"
        assert result.backend == "fujitsu_da"
        assert result.solutions == []
        assert [error.code for error in result.errors] == ["REMOTE_TIME_LIMIT"]
        assert "400" in result.errors[0].message
        assert fake.submit_calls == 0
        assert fake.requests == []
        assert_actions_present(result)

    def test_time_limit_at_the_ceiling_still_submits(self, monkeypatch):
        set_key(monkeypatch)
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))
        service = make_da_service(fake)

        result = service.solve(
            make_problem(backend="fujitsu_da", fujitsu_da={"time_limit_seconds": 300})
        )

        assert result.status == "success"
        assert fake.submit_calls == 1
        assert fake.submit_request.json["fujitsuDA3"]["time_limit_sec"] == 300

    def test_num_run_over_the_schema_range_is_rejected_at_model_build(self):
        # Schema-level range (§20.2): the request never reaches the service.
        with pytest.raises(pydantic.ValidationError):
            make_problem(backend="fujitsu_da", fujitsu_da={"num_run": 2000})


class TestTransportExceptionClassification:
    """§20.8: exceptions the transport *raises*."""

    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (TimeoutError("read timed out"), "REMOTE_TIMEOUT"),
            (urllib.error.URLError(TimeoutError("connect timed out")), "REMOTE_TIMEOUT"),
            (
                urllib.error.URLError(ConnectionRefusedError("refused")),
                "REMOTE_SOLVER_ERROR",
            ),
            (
                http.client.RemoteDisconnected("gone"),
                "REMOTE_SOLVER_ERROR",
            ),
            (OSError("network down"), "REMOTE_SOLVER_ERROR"),
        ],
        ids=["timeout", "url-timeout", "refused", "disconnected", "oserror"],
    )
    def test_submit_exception_maps_to_a_code(
        self, monkeypatch, exception, expected_code
    ):
        fake = FakeDATransport(submit_response=exception)

        error = expect_failure(monkeypatch, fake)

        assert error.code == expected_code
        assert error.status is None
        assert error.__cause__ is None
        assert error.__context__ is None
        assert FAKE_KEY not in str(error)
        assert FAKE_KEY not in "".join(traceback.format_exception(error))

    def test_poll_exception_is_classified_too(self, monkeypatch):
        fake = FakeDATransport(poll_responses={0: TimeoutError("gone quiet")})

        error = expect_failure(monkeypatch, fake)

        assert error.code == "REMOTE_TIMEOUT"
        assert fake.submit_calls == 1
        assert FAKE_KEY not in str(error)


class TestHttpStatusClassification:
    """§20.8: statuses the transport *returns*."""

    @pytest.mark.parametrize(
        ("status", "body", "expected_code", "expected_status"),
        [
            (401, b'{"message": "Unauthorized"}', "REMOTE_AUTH_FAILED", None),
            (403, b'{"message": "Forbidden"}', "REMOTE_AUTH_FAILED", None),
            (
                400,
                b'{"message": "Monthly usage exceeds specified metering limit."}',
                "REMOTE_QUOTA_EXCEEDED",
                None,
            ),
            (
                400,
                b'{"message": "Invalid request header: X-Access-Token"}',
                "BACKEND_CONFIG_INVALID",
                "configuration_error",
            ),
            (
                400,
                b'{"message": "X-Api-Key is required"}',
                "BACKEND_CONFIG_INVALID",
                "configuration_error",
            ),
            (
                400,
                b'{"message": "the number of variables exceeds the limit (100000)"}',
                "REMOTE_SOLVER_ERROR",
                None,
            ),
            (413, b"Request Entity Too Large", "REMOTE_SOLVER_ERROR", None),
            (429, b'{"message": "Too Many Requests"}', "REMOTE_BUSY", None),
            (500, b'{"message": "Internal Server Error"}', "REMOTE_SOLVER_ERROR", None),
            (503, b'{"message": "Service Unavailable"}', "REMOTE_SOLVER_ERROR", None),
        ],
        ids=[
            "401",
            "403",
            "400-quota",
            "400-access-token-header",
            "400-api-key-header",
            "400-problem-level",
            "413",
            "429",
            "500",
            "503",
        ],
    )
    def test_status_maps_to_a_code(
        self, monkeypatch, status, body, expected_code, expected_status
    ):
        fake = FakeDATransport(submit_response=(status, body))

        error = expect_failure(monkeypatch, fake)

        assert error.code == expected_code
        assert error.status == expected_status
        assert error.__cause__ is None
        assert error.__context__ is None
        assert FAKE_KEY not in str(error)
        assert FAKE_KEY not in "".join(traceback.format_exception(error))

    def test_413_message_names_the_payload_size_not_the_body(self, monkeypatch):
        fake = FakeDATransport(submit_response=(413, b"Request Entity Too Large"))

        error = expect_failure(monkeypatch, fake)

        assert "payload too large" in str(error)

    def test_429_is_retryable_in_the_catalog(self):
        assert "REMOTE_BUSY" in RETRYABLE_CODES

    def test_status_is_also_classified_while_polling(self, monkeypatch):
        fake = FakeDATransport(poll_responses={1: (401, b'{"message": "Unauthorized"}')})

        error = expect_failure(monkeypatch, fake)

        assert error.code == "REMOTE_AUTH_FAILED"
        assert fake.poll_calls == 2


class TestMalformedResponses:
    """§20.8 bottom rows: a 2xx whose body is not what the API promises."""

    def test_non_json_2xx_is_a_solver_error(self, monkeypatch):
        fake = FakeDATransport(submit_response=(200, b"<html>not json</html>"))

        error = expect_failure(monkeypatch, fake)

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert "not valid JSON" in str(error)

    def test_missing_job_id_is_a_solver_error(self, monkeypatch):
        fake = FakeDATransport(submit_response=json_response(200, {"status": "ok"}))

        error = expect_failure(monkeypatch, fake)

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert "no job_id" in str(error)

    def test_done_without_qubo_solution_is_a_solver_error(self, monkeypatch):
        fake = FakeDATransport(result_response=json_response(200, {"status": "Done"}))

        error = expect_failure(monkeypatch, fake)

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert "no qubo_solution" in str(error)

    def test_job_status_error_reports_the_message_and_frees_the_slot(self, monkeypatch):
        fake = FakeDATransport(
            poll_statuses=("Error",), error_message="the number of variables exceeds"
        )

        error = expect_failure(monkeypatch, fake)

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert "the number of variables exceeds" in str(error)
        # A failed job still occupies a slot, so it is deleted too.
        assert fake.delete_calls == 1
        assert error.__cause__ is None

    def test_unexpected_job_status_is_a_solver_error(self, monkeypatch):
        fake = FakeDATransport(
            poll_responses={0: json_response(200, {"status": "Canceled"})}
        )

        error = expect_failure(monkeypatch, fake)

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert "Canceled" in str(error)
        assert fake.delete_calls == 0


class TestPollTimeout:
    """§20.7: a job that never finishes is cancelled and reported."""

    def make_fake(self, **kwargs) -> FakeDATransport:
        return FakeDATransport(poll_statuses=("Running",), **kwargs)

    def stepping_clock(self):
        """A clock that jumps 100 s per call, so the deadline is hit at once."""
        state = {"now": 0.0}

        def clock() -> float:
            state["now"] += 100.0
            return state["now"]

        return clock

    def test_timeout_cancels_the_job(self, monkeypatch):
        fake = self.make_fake()

        error = expect_failure(monkeypatch, fake, clock=self.stepping_clock())

        assert error.code == "REMOTE_TIMEOUT"
        assert fake.cancel_calls == 1
        cancels = fake.requests_for("POST", CANCEL_PATH)
        assert len(cancels) == 1
        assert cancels[0].url == f"{DEFAULT_BASE_URL}{CANCEL_PATH}"
        assert cancels[0].json == {"job_id": FAKE_JOB_ID}

    def test_timeout_message_names_the_budget(self, monkeypatch):
        fake = self.make_fake()

        error = expect_failure(monkeypatch, fake, clock=self.stepping_clock())

        assert "70" in str(error)
        assert FAKE_KEY not in str(error)

    def test_cancel_failure_does_not_hide_the_timeout(self, monkeypatch):
        fake = self.make_fake(cancel_response=OSError("cancel refused"))

        error = expect_failure(monkeypatch, fake, clock=self.stepping_clock())

        assert error.code == "REMOTE_TIMEOUT"
        assert fake.cancel_calls == 1

    def test_no_result_is_deleted_on_timeout(self, monkeypatch):
        fake = self.make_fake()

        expect_failure(monkeypatch, fake, clock=self.stepping_clock())

        assert fake.delete_calls == 0


class TestServiceStatusMapping:
    """§20.8 / §26.5: the status a failure is reported under by the service."""

    def solve_with(self, monkeypatch, fake: FakeDATransport):
        set_key(monkeypatch)
        service = make_da_service(fake)
        return service.solve(make_problem(backend="fujitsu_da"))

    def test_header_rejection_is_a_configuration_error(self, monkeypatch):
        fake = FakeDATransport(
            submit_response=(400, b'{"message": "Invalid request header: X-Api-Key"}')
        )

        result = self.solve_with(monkeypatch, fake)

        assert result.status == "configuration_error"
        assert [error.code for error in result.errors] == ["BACKEND_CONFIG_INVALID"]
        assert result.solutions == []
        assert_actions_present(result)

    def test_problem_level_rejection_is_a_solver_error(self, monkeypatch):
        fake = FakeDATransport(
            submit_response=(
                400,
                b'{"message": "the number of variables exceeds the limit (100000)"}',
            )
        )

        result = self.solve_with(monkeypatch, fake)

        assert result.status == "solver_error"
        assert [error.code for error in result.errors] == ["REMOTE_SOLVER_ERROR"]
        assert_actions_present(result)

    def test_busy_is_reported_as_retryable(self, monkeypatch):
        fake = FakeDATransport(submit_response=(429, b'{"message": "Too Many Requests"}'))

        result = self.solve_with(monkeypatch, fake)

        assert result.status == "solver_error"
        assert [error.code for error in result.errors] == ["REMOTE_BUSY"]
        assert result.errors[0].retryable is True
        assert_actions_present(result)

    def test_auth_failure_is_a_solver_error(self, monkeypatch):
        fake = FakeDATransport(submit_response=(401, b'{"message": "Unauthorized"}'))

        result = self.solve_with(monkeypatch, fake)

        assert result.status == "solver_error"
        assert [error.code for error in result.errors] == ["REMOTE_AUTH_FAILED"]
        assert_actions_present(result)

    def test_successful_solve_reaches_success(self, monkeypatch):
        # The all-zero sample satisfies "a + b <= 1", so it is a feasible
        # candidate and the whole flow reaches success.
        fake = FakeDATransport(solutions=zero_rows(make_compiled()))

        result = self.solve_with(monkeypatch, fake)

        assert result.status == "success", (result.status, result.errors)
        assert result.solutions[0].variables == {"a": 0, "b": 0}
        assert result.metadata.solver_id == SOLVER_ID
        assert result.metadata.model_type == "bqm"
        assert result.metadata.backend == "fujitsu_da"


class TestIsAvailable:
    """§20.5: credentials and endpoint sanity, with no network I/O at all."""

    def check(self, fake: FakeDATransport) -> AvailabilityStatus:
        backend, _ = make_backend(fake)
        status = backend.is_available()
        # The answer is local: no request is ever made to decide it.
        assert fake.requests == []
        return status

    def test_missing_key(self):
        fake = FakeDATransport()

        assert self.check(fake) == AvailabilityStatus(
            category="credentials_missing", detail=REASON_API_KEY_MISSING
        )

    def test_empty_key_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("FUJITSU_DA_API_KEY", "")
        fake = FakeDATransport()

        assert self.check(fake) == AvailabilityStatus(
            category="credentials_missing", detail=REASON_API_KEY_MISSING
        )

    def test_non_https_url_is_a_configuration_error(self, monkeypatch):
        set_key(monkeypatch)
        monkeypatch.setenv("FUJITSU_DA_URL", "http://insecure.test/da")
        fake = FakeDATransport()

        assert self.check(fake) == AvailabilityStatus(
            category="config_invalid",
            detail=REASON_URL_NOT_HTTPS,
            error_code="BACKEND_CONFIG_INVALID",
        )

    def test_available_with_key_and_default_url(self, monkeypatch):
        set_key(monkeypatch)
        fake = FakeDATransport()

        assert self.check(fake) == AvailabilityStatus(category="available")

    def test_detail_never_contains_the_url_or_the_key(self, monkeypatch):
        set_key(monkeypatch)
        monkeypatch.setenv("FUJITSU_DA_URL", "http://insecure.test/da")
        fake = FakeDATransport()

        detail = self.check(fake).detail
        assert "insecure.test" not in detail
        assert FAKE_KEY not in detail

    def test_solve_without_a_key_never_touches_the_transport(self):
        fake = FakeDATransport()
        backend, _ = make_backend(fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled(), make_preferences())

        assert exc_info.value.code == "REMOTE_CREDENTIALS_MISSING"
        assert fake.requests == []

    def test_solve_with_a_non_https_url_never_touches_the_transport(self, monkeypatch):
        set_key(monkeypatch)
        monkeypatch.setenv("FUJITSU_DA_URL", "http://insecure.test/da")
        fake = FakeDATransport()
        backend, _ = make_backend(fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled(), make_preferences())

        assert exc_info.value.code == "BACKEND_CONFIG_INVALID"
        assert exc_info.value.status == "configuration_error"
        assert fake.requests == []


class TestIntegerProblemThroughTheService:
    """§26.5: bits the DA returns decode back to integer business values."""

    def load(self, examples_dir) -> OptimizationProblem:
        payload = json.loads((examples_dir / "integer_knapsack.json").read_text())
        payload["solver"] = {**payload["solver"], "backend": "fujitsu_da"}
        return OptimizationProblem.model_validate(payload)

    def best_bits(self, problem: OptimizationProblem) -> tuple[list[int], float]:
        """The optimum bit row, found locally by the exhaustive backend."""
        compiled = BQMCompiler().compile(problem, hard_penalty=100.0)
        raw = ExactSolverBackend().solve(compiled, problem.solver)
        index = int(np.argmin(raw.energies))
        return raw.samples[index].tolist(), float(raw.energies[index])

    def solve(self, monkeypatch, examples_dir):
        set_key(monkeypatch)
        problem = self.load(examples_dir)
        bits, energy = self.best_bits(problem)
        fake = FakeDATransport(solutions=[bits_solution(bits, energy)])
        result = make_da_service(fake).solve(problem)
        return fake, bits, result

    def test_status_and_decoded_assignment(self, monkeypatch, examples_dir):
        _, _, result = self.solve(monkeypatch, examples_dir)

        assert result.status == "success", (result.status, result.errors, result.message)
        assert result.backend == "fujitsu_da"
        assert result.solutions
        assert result.solutions[0].variables == {
            "item_a": 0,
            "item_b": 1,
            "item_c": 1,
            "item_d": 3,
        }
        assert result.solutions[0].objective_value == 34

    def test_values_stay_inside_the_declared_bounds(self, monkeypatch, examples_dir):
        _, _, result = self.solve(monkeypatch, examples_dir)

        for solution in result.solutions:
            assert set(solution.variables) == {"item_a", "item_b", "item_c", "item_d"}
            for name, value in solution.variables.items():
                assert isinstance(value, int)
                assert 0 <= value <= 3, name

    def test_no_encoding_variable_leaks_into_the_solution(self, monkeypatch, examples_dir):
        _, _, result = self.solve(monkeypatch, examples_dir)

        for solution in result.solutions:
            assert not any("__" in name for name in solution.variables)

    def test_every_bit_of_the_compiled_model_was_submitted(
        self, monkeypatch, examples_dir
    ):
        fake, bits, _ = self.solve(monkeypatch, examples_dir)

        terms = fake.submit_request.json["binary_polynomial"]["terms"]
        linear = [term for term in terms if len(term["polynomials"]) == 1]
        assert len(linear) == len(bits)


class TestRemoteRetries:
    """§14 step 14: remote retries burn quota, so policy must opt in."""

    def solve(self, monkeypatch, *, allow_remote_retries: bool, max_retries: int = 2):
        set_key(monkeypatch)
        problem = make_zero_infeasible_problem(
            backend="fujitsu_da", max_retries=max_retries
        )
        width = len(BQMCompiler().compile(problem, hard_penalty=100.0).model.variables)
        # The all-zero sample violates "a + b >= 1", so no attempt can ever
        # produce a feasible candidate and every retry is actually taken.
        fake = FakeDATransport(solutions=[bits_solution([0] * width, 0.0)])
        service = make_da_service(fake, allow_remote_retries=allow_remote_retries)
        return fake, service.solve(problem)

    def test_retries_disabled_makes_exactly_one_attempt(self, monkeypatch):
        fake, result = self.solve(monkeypatch, allow_remote_retries=False)

        assert result.status == "infeasible"
        assert len(result.attempts) == 1
        assert fake.submit_calls == 1
        assert fake.delete_calls == 1
        assert [warning.code for warning in result.warnings] == [
            "REMOTE_RETRIES_DISABLED"
        ]
        assert_actions_present(result)

    def test_retries_enabled_uses_one_plus_max_retries(self, monkeypatch):
        fake, result = self.solve(monkeypatch, allow_remote_retries=True)

        assert result.status == "infeasible"
        assert len(result.attempts) == 3
        assert [attempt.attempt for attempt in result.attempts] == [1, 2, 3]
        assert fake.submit_calls == 3
        assert fake.delete_calls == 3
        assert result.warnings == []

    def test_over_the_remote_retry_ceiling_submits_nothing(self, monkeypatch):
        # 2026-09-09 review (F-07): opting in to remote retries does not lift
        # the ceiling. max_retries=4 is over the default 3, so the request is
        # refused before any vendor job exists — zero quota burnt.
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=True, max_retries=4
        )

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["RETRY_LIMIT"]
        assert "4" in result.errors[0].message
        assert result.solutions == []
        assert result.attempts == []
        assert fake.submit_calls == 0
        assert fake.delete_calls == 0

    def test_exactly_the_remote_retry_ceiling_is_allowed(self, monkeypatch):
        # The ceiling is inclusive: max_retries == the ceiling still solves.
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=True, max_retries=3
        )

        assert result.status == "infeasible"
        assert len(result.attempts) == 4
        assert fake.submit_calls == 4


class TestRedaction:
    """§20.5: the key never survives into anything that leaves the backend."""

    def test_key_in_a_transport_exception_is_masked(self, monkeypatch):
        fake = FakeDATransport(
            submit_response=OSError(f"TLS handshake failed for X-Api-Key {FAKE_KEY}")
        )

        error = expect_failure(monkeypatch, fake)

        message = str(error)
        assert FAKE_KEY not in message
        assert "***" in message
        assert FAKE_KEY not in "".join(traceback.format_exception(error))

    @pytest.mark.parametrize(
        "body",
        [
            "y" * 190 + " " + FAKE_KEY + " rejected",  # key straddles the 200 cut
            "x" * 180 + "\n" * 60 + FAKE_KEY + "\n   tail",  # collapse moves it onto the cut
        ],
        ids=["straddles_cut", "moved_by_collapse"],
    )
    def test_body_summary_masks_the_key_before_truncating(self, monkeypatch, body):
        # F-01: the summary helper must redact first; truncating or collapsing
        # first leaves a key fragment that no longer equals the env value.
        from annealbridge.solvers.fujitsu_da import _redacted_summary

        set_key(monkeypatch)
        summary = _redacted_summary(body)

        assert "***" in summary
        assert len(summary) <= 201  # the 200-character cap plus the ellipsis
        for start in range(len(FAKE_KEY) - 7):
            assert FAKE_KEY[start : start + 8] not in summary
