"""Credential-leak tests for the remote solve path (Phase 2 spec §19, §27;
3a spec §26.5, §28).

A fake credential is placed in the backend's environment variable
(``DWAVE_API_TOKEN`` for the D-Wave kinds, ``FUJITSU_DA_API_KEY`` for the
Fujitsu kind) and a fake sampler / transport raises an exception whose text
embeds it. Nothing that leaves the service — the ``SolveResult`` JSON, its
errors/warnings, or any ``annealbridge`` log record — may contain the
credential. The service-level tests run once per **every remote backend
in the default registry** (3b spec §22; 2026-09-09 review F-10): the kinds
are generated from ``SolverRegistry.default()``, the env var comes from
each backend's own ``capabilities.credentials`` declaration, and a remote
backend without a fake builder here — or without a declared credential
env var — fails the suite instead of silently going untested.

The token deliberately contains hyphens, so it does *not* match the
``DEV-[A-Za-z0-9]{20,}`` value pattern the D-Wave backends declare:
masking has to come from the live environment-variable candidate that
:func:`redact` resolves. With the env var set, ``ocean_config_status()``
already reports ``"ok"`` (the ``dwave`` extra is not installed, so the env
var is the only config source), which is exactly the path under test —
only ``dwave_system_installed`` is patched.
"""

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers import (
    DWaveQPUBackend,
    FujitsuDABackend,
    LeapHybridBQMBackend,
    LeapHybridCQMBackend,
    SolverRegistry,
)
import annealbridge.solvers.ocean as ocean_module
from tests.fakes import FakeDATransport, bits_solution, json_response
from tests.remote_mock.conftest import (
    FAKE_UNPATTERNED_TOKEN,
    FakeCQMSampler,
    FakeLeapHybridSampler,
    FakeQPUSampler,
    make_problem,
)

# Hyphenated on purpose: not matched by the DEV- pattern, so only the live
# environment-variable candidate can mask it. Never a real token.
FAKE_TOKEN = FAKE_UNPATTERNED_TOKEN


@dataclass(frozen=True)
class RemoteKind:
    """How to drive one remote backend kind into a credential-bearing failure.

    ``env_var`` is where the backend reads its credential; ``build`` returns
    ``(fake, backend)`` whose first remote call raises an exception with the
    given message (``lazy`` only matters for the Ocean fakes); ``calls``
    counts how many times the fake was asked to solve.
    """

    env_var: str
    build: Callable[[str, bool], tuple[Any, Any]]
    calls: Callable[[Any], int]


def _build_qpu(message: str, lazy: bool):
    fake = FakeQPUSampler(raise_on_sample=RuntimeError(message), lazy=lazy)
    return fake, DWaveQPUBackend(sampler_factory=lambda: fake)


def _build_cqm(message: str, lazy: bool):
    fake = FakeCQMSampler(raise_on_sample=RuntimeError(message), lazy=lazy)
    return fake, LeapHybridCQMBackend(sampler_factory=lambda: fake)


def _build_hybrid(message: str, lazy: bool):
    fake = FakeLeapHybridSampler(raise_on_sample=RuntimeError(message), lazy=lazy)
    return fake, LeapHybridBQMBackend(sampler_factory=lambda: fake)


def _build_da(message: str, lazy: bool):
    # The transport raises on the very first request (job submission); the
    # DA flow has no lazy resolve step, so ``lazy`` is irrelevant here.
    fake = FakeDATransport(submit_response=RuntimeError(message))
    return fake, FujitsuDABackend(transport=fake, sleep=lambda _seconds: None)


# How to drive each shipped remote backend *class* into a failure with a
# fake. Keyed by class, not by name: the names and env vars come from the
# default registry below, so adding a remote backend without adding a fake
# here fails ``_remote_kinds`` loudly.
_FAKE_BUILDERS: dict[type, tuple[Callable[[str, bool], tuple[Any, Any]], Callable[[Any], int]]] = {
    DWaveQPUBackend: (_build_qpu, lambda fake: fake.sample_calls),
    LeapHybridBQMBackend: (_build_hybrid, lambda fake: fake.sample_calls),
    LeapHybridCQMBackend: (_build_cqm, lambda fake: fake.sample_calls),
    FujitsuDABackend: (_build_da, lambda fake: fake.submit_calls),
}


def _remote_kinds() -> dict[str, RemoteKind]:
    """Every remote backend of the default registry, driven by its own declaration."""
    kinds: dict[str, RemoteKind] = {}
    defaults = SolverRegistry.default()
    for name in defaults.names():
        backend = defaults.get(name)
        if not backend.capabilities.remote:
            continue
        assert type(backend) in _FAKE_BUILDERS, (
            f"remote backend {name!r} has no fake builder in the credential-leak suite"
        )
        env_vars = backend.capabilities.credentials.env_vars
        assert env_vars, f"remote backend {name!r} declares no credential env var"
        build, calls = _FAKE_BUILDERS[type(backend)]
        kinds[name] = RemoteKind(env_vars[0], build, calls)
    return kinds


REMOTE_KINDS: dict[str, RemoteKind] = _remote_kinds()


@pytest.fixture(params=sorted(REMOTE_KINDS))
def remote_kind(request) -> str:
    return request.param


def solve_with_leaky_sampler(
    monkeypatch, caplog, message: str, *, kind: str, lazy: bool = False
):
    """Run a full remote solve on ``kind`` whose sampler raises ``message``."""
    remote = REMOTE_KINDS[kind]
    monkeypatch.setenv(remote.env_var, FAKE_TOKEN)
    # Only installability is patched (harmless for non-Ocean kinds): an env
    # token alone makes ``ocean_config_status()`` report "ok", and that path
    # is under test, so it is deliberately left real.
    monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
    if remote.env_var == ocean_module.TOKEN_ENV:
        assert ocean_module.ocean_config_status() == "ok"

    fake, backend = remote.build(message, lazy)
    service = OptimizationService(
        registry=SolverRegistry({kind: backend}),
        policy=ExecutionPolicy(allow_remote=True),
    )
    caplog.set_level(logging.DEBUG, logger="annealbridge")
    result = service.solve(make_problem(backend=kind))
    assert remote.calls(fake) == 1
    return result


class TestTokenNeverLeaves:
    # Each test drives its own solve so the log records land in the calling
    # test's ``caplog`` phase rather than in a fixture's setup phase.
    def solve(self, monkeypatch, caplog, kind: str):
        return solve_with_leaky_sampler(
            monkeypatch,
            caplog,
            f"connection rejected for token={FAKE_TOKEN} "
            f"at https://cloud.dwavesys.com",
            kind=kind,
        )

    def test_solve_fails_as_a_structured_solver_error(
        self, monkeypatch, caplog, remote_kind
    ):
        result = self.solve(monkeypatch, caplog, remote_kind)

        assert result.status == "solver_error"
        assert result.backend == remote_kind
        assert result.solutions == []
        assert result.errors

    def test_token_absent_from_the_serialized_result(
        self, monkeypatch, caplog, remote_kind
    ):
        result = self.solve(monkeypatch, caplog, remote_kind)

        assert FAKE_TOKEN not in result.model_dump_json()

    def test_token_absent_from_every_log_record(self, monkeypatch, caplog, remote_kind):
        self.solve(monkeypatch, caplog, remote_kind)

        assert caplog.records  # the solve really did log something
        for record in caplog.records:
            assert FAKE_TOKEN not in record.getMessage()

    def test_token_absent_from_errors_and_warnings(
        self, monkeypatch, caplog, remote_kind
    ):
        result = self.solve(monkeypatch, caplog, remote_kind)

        for entry in [*result.errors, *result.warnings]:
            assert FAKE_TOKEN not in entry.message

    def test_error_message_is_visibly_redacted(self, monkeypatch, caplog, remote_kind):
        result = self.solve(monkeypatch, caplog, remote_kind)

        assert "***" in result.errors[0].message

    def test_error_carries_catalog_guidance(self, monkeypatch, caplog, remote_kind):
        error = self.solve(monkeypatch, caplog, remote_kind).errors[0]

        assert error.code == "REMOTE_SOLVER_ERROR"
        assert error.recommended_action is not None


class TestBareTokenIsMasked:
    """A token with no ``token=`` prefix: only the env candidate can mask it."""

    def test_bare_token_is_still_redacted(self, monkeypatch, caplog, remote_kind):
        result = solve_with_leaky_sampler(
            monkeypatch,
            caplog,
            f"connection rejected (credential {FAKE_TOKEN}) "
            f"at https://cloud.dwavesys.com",
            kind=remote_kind,
        )

        assert result.status == "solver_error"
        assert FAKE_TOKEN not in result.model_dump_json()
        assert "***" in result.errors[0].message
        assert caplog.records
        for record in caplog.records:
            assert FAKE_TOKEN not in record.getMessage()


class TestLazyResolveFailureIsRedacted:
    """The token must not leak when the failure surfaces at resolve time."""

    def solve(self, monkeypatch, caplog, kind: str):
        return solve_with_leaky_sampler(
            monkeypatch,
            caplog,
            f"request failed for token={FAKE_TOKEN} (credential {FAKE_TOKEN})",
            kind=kind,
            lazy=True,
        )

    def test_structured_error_instead_of_an_escaping_exception(
        self, monkeypatch, caplog, remote_kind
    ):
        result = self.solve(monkeypatch, caplog, remote_kind)

        assert result.status == "solver_error"
        assert result.errors[0].code == "REMOTE_SOLVER_ERROR"

    def test_token_absent_from_result_and_logs(self, monkeypatch, caplog, remote_kind):
        result = self.solve(monkeypatch, caplog, remote_kind)

        assert FAKE_TOKEN not in result.model_dump_json()
        assert "***" in result.errors[0].message
        assert caplog.records
        for record in caplog.records:
            assert FAKE_TOKEN not in record.getMessage()


def compile_problem():
    return BQMCompiler().compile(make_problem(), hard_penalty=100.0)


def compile_cqm_problem():
    return CQMCompiler().compile(make_problem(backend="leap_hybrid_cqm"), hard_penalty=None)


def hybrid_preferences():
    return make_problem().solver.model_copy(update={"backend": "leap_hybrid_bqm"})


def cqm_preferences():
    return make_problem(backend="leap_hybrid_cqm").solver


def da_preferences():
    return make_problem(backend="fujitsu_da").solver


def da_backend(fake: FakeDATransport) -> FujitsuDABackend:
    return FujitsuDABackend(transport=fake, sleep=lambda _seconds: None)


class TestFujitsuResponseBodyEchoingTheKeyIsMasked:
    """3b §22: the key may come back in an HTTP error *body* (not only in an
    exception); the 400 classification runs on the raw body while the
    message that leaves the backend is redacted."""

    @pytest.fixture(autouse=True)
    def _key_in_env(self, monkeypatch):
        monkeypatch.setenv("FUJITSU_DA_API_KEY", FAKE_TOKEN)

    def solve(self, caplog, body: bytes):
        fake = FakeDATransport(submit_response=(400, body))
        service = OptimizationService(
            registry=SolverRegistry({"fujitsu_da": da_backend(fake)}),
            policy=ExecutionPolicy(allow_remote=True),
        )
        caplog.set_level(logging.DEBUG, logger="annealbridge")
        return service.solve(make_problem(backend="fujitsu_da"))

    def test_problem_level_rejection_echoing_the_key(self, caplog):
        body = ('{"message": "rejected request from ' + FAKE_TOKEN + '"}').encode()
        result = self.solve(caplog, body)

        assert result.status == "solver_error"
        assert result.errors[0].code == "REMOTE_SOLVER_ERROR"
        assert FAKE_TOKEN not in result.model_dump_json()
        assert "***" in result.errors[0].message
        assert caplog.records
        for record in caplog.records:
            assert FAKE_TOKEN not in record.getMessage()

    def test_header_rejection_echoing_the_key(self, caplog):
        body = (
            '{"message": "Invalid request header:X-Api-Key ' + FAKE_TOKEN + '"}'
        ).encode()
        result = self.solve(caplog, body)

        assert result.status == "configuration_error"
        assert result.errors[0].code == "BACKEND_CONFIG_INVALID"
        assert FAKE_TOKEN not in result.model_dump_json()
        assert "***" in result.errors[0].message
        for record in caplog.records:
            assert FAKE_TOKEN not in record.getMessage()

    def test_header_line_pattern_masks_even_without_the_env_value(
        self, caplog, monkeypatch
    ):
        # A key that is *not* the configured one (so the env candidate cannot
        # mask it) is still hidden by the ``X-Api-Key: …`` line pattern.
        monkeypatch.setenv("FUJITSU_DA_API_KEY", "some-other-configured-key")
        body = b'{"message": "Invalid request header:\\nX-Api-Key: leaked-value-123\\n"}'
        result = self.solve(caplog, body)

        assert result.status == "configuration_error"
        assert "leaked-value-123" not in result.model_dump_json()
        assert "X-Api-Key: ***" in result.errors[0].message



def assert_no_key_fragment(text: str, key: str, width: int = 8) -> None:
    """No ``width``-character window of ``key`` survives in ``text``.

    A truncated or whitespace-split key is still a leak: F-01 showed that a
    partial key defeats the literal env-value replacement in ``redact()``.
    """
    for start in range(len(key) - width + 1):
        fragment = key[start : start + width]
        assert fragment not in text, f"key fragment {fragment!r} leaked in {text!r}"


class TestFujitsuTruncatedBodyStillMasksTheKey:
    """F-01: the body summary is length-capped at 200 characters and
    whitespace-collapsed. Redaction must run *before* that, or a key that
    straddles the cut (or is moved onto it by the collapse) leaves an
    unmasked prefix in the error message and the logs."""

    @pytest.fixture(autouse=True)
    def _key_in_env(self, monkeypatch):
        monkeypatch.setenv("FUJITSU_DA_API_KEY", FAKE_TOKEN)

    def solve_via_service(self, caplog, fake: FakeDATransport):
        service = OptimizationService(
            registry=SolverRegistry({"fujitsu_da": da_backend(fake)}),
            policy=ExecutionPolicy(allow_remote=True),
        )
        caplog.set_level(logging.DEBUG, logger="annealbridge")
        return service.solve(make_problem(backend="fujitsu_da"))

    def assert_clean(self, caplog, result) -> None:
        assert_no_key_fragment(result.model_dump_json(), FAKE_TOKEN)
        assert "***" in result.errors[0].message
        assert caplog.records
        for record in caplog.records:
            assert_no_key_fragment(record.getMessage(), FAKE_TOKEN)

    def test_key_straddling_the_200_character_cut(self, caplog):
        # The key starts at character ~190 of the collapsed body, so the old
        # order truncated it mid-way and only a prefix reached redact().
        body = (
            "Invalid request header X-Api-Key value " + "y" * 150 + " " + FAKE_TOKEN
            + " rejected"
        ).encode()
        result = self.solve_via_service(
            caplog, FakeDATransport(submit_response=(400, body))
        )

        # "Invalid request header" classifies as a configuration error (§20.8);
        # the classification reads the raw body, the message must not.
        assert result.status == "configuration_error"
        assert result.errors[0].code == "BACKEND_CONFIG_INVALID"
        self.assert_clean(caplog, result)

    def test_key_moved_onto_the_cut_by_whitespace_collapsing(self, caplog):
        # In the raw body the key begins after 240 characters (past the cap);
        # collapsing the 60 newlines to one space pulls it back to 181, so
        # the cut lands inside it.
        body = ("x" * 180 + "\n" * 60 + FAKE_TOKEN + "\n\n   tail").encode()
        result = self.solve_via_service(
            caplog, FakeDATransport(submit_response=(400, body))
        )

        assert result.status == "solver_error"
        self.assert_clean(caplog, result)

    def test_delete_failure_warning_with_the_key_on_the_cut(self, caplog):
        compiled = BQMCompiler().compile(make_problem(backend="fujitsu_da"), hard_penalty=100.0)
        width = len(compiled.model.variables)
        fake = FakeDATransport(
            solutions=[bits_solution([0] * width, 0.0)],
            delete_response=(500, ("internal error " + "z" * 175 + " " + FAKE_TOKEN).encode()),
        )
        backend = da_backend(fake)

        with caplog.at_level(logging.WARNING, logger="annealbridge.solvers.fujitsu_da"):
            raw = backend.solve(compiled, da_preferences())

        assert raw.num_samples == 1
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "could not be deleted" in warnings[0]
        assert "***" in warnings[0]
        assert_no_key_fragment(warnings[0], FAKE_TOKEN)

    def test_non_json_2xx_body_with_the_key_on_the_cut(self):
        body = ("<html>" + "w" * 185 + " " + FAKE_TOKEN + "</html>").encode()
        fake = FakeDATransport(submit_response=(200, body))
        backend = da_backend(fake)
        compiled = BQMCompiler().compile(make_problem(backend="fujitsu_da"), hard_penalty=100.0)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compiled, da_preferences())

        message = str(exc_info.value)
        assert "not valid JSON" in message
        assert "***" in message
        assert_no_key_fragment(message, FAKE_TOKEN)
        assert_no_key_fragment("".join(traceback.format_exception(exc_info.value)), FAKE_TOKEN)


class TestExceptionChainCarriesNoToken:
    """The raw Ocean exception must not hang off the wrapped error: anything
    formatting the chain (``logger.exception``, traceback tooling) would
    otherwise print the unredacted text."""

    @pytest.fixture(autouse=True)
    def _token_in_env(self, monkeypatch):
        monkeypatch.setenv("DWAVE_API_TOKEN", FAKE_TOKEN)
        monkeypatch.setenv("FUJITSU_DA_API_KEY", FAKE_TOKEN)

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

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def test_cqm_sample_failure(self, lazy):
        fake = FakeCQMSampler(
            raise_on_sample=RuntimeError(f"rejected token={FAKE_TOKEN}"), lazy=lazy
        )
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compile_cqm_problem(), cqm_preferences())

        assert fake.sample_calls == 1
        self.assert_chain_is_clean(exc_info.value)

    def test_cqm_sampler_construction_failure(self):
        def failing_factory():
            raise ValueError(f"invalid endpoint for token={FAKE_TOKEN}")

        backend = LeapHybridCQMBackend(sampler_factory=failing_factory)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compile_cqm_problem(), cqm_preferences())

        assert exc_info.value.code == "DWAVE_CONFIG_INVALID"
        self.assert_chain_is_clean(exc_info.value)

    def test_cqm_time_limit_resolution_failure(self):
        fake = FakeCQMSampler(
            raise_on_min_time_limit=RuntimeError(f"cannot reach solver, token={FAKE_TOKEN}")
        )
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.resolve_time_limit(compile_cqm_problem(), cqm_preferences())

        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"
        self.assert_chain_is_clean(exc_info.value)

    def test_da_transport_failure(self):
        fake = FakeDATransport(
            submit_response=OSError(f"connection refused for X-Api-Key {FAKE_TOKEN}")
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            da_backend(fake).solve(compile_problem(), da_preferences())

        assert fake.submit_calls == 1
        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"
        self.assert_chain_is_clean(exc_info.value)

    def test_da_poll_failure_after_submission(self):
        fake = FakeDATransport(
            poll_responses={0: RuntimeError(f"lost connection, key={FAKE_TOKEN}")}
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            da_backend(fake).solve(compile_problem(), da_preferences())

        assert fake.poll_calls == 1
        self.assert_chain_is_clean(exc_info.value)

    def test_da_error_body_echoing_the_key(self):
        fake = FakeDATransport(
            submit_response=json_response(
                500, {"title": "Internal Server Error", "message": f"key {FAKE_TOKEN}"}
            )
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            da_backend(fake).solve(compile_problem(), da_preferences())

        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"
        self.assert_chain_is_clean(exc_info.value)

    def test_da_invalid_json_body_echoing_the_key(self):
        fake = FakeDATransport(
            submit_response=(200, f"<html>{FAKE_TOKEN}</html>".encode())
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            da_backend(fake).solve(compile_problem(), da_preferences())

        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"
        self.assert_chain_is_clean(exc_info.value)
