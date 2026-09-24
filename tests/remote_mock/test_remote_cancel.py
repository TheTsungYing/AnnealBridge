"""Cancellation and the wall-clock limit on the remote backends (mocked).

Batch 6 (J), time-limit spec 2026-09-24 §5, §6.4 and §9 item 6. No remote
backend declares ``supports_interrupt``, so:

* a cancel token acts only at the service's own checkpoints. A vendor call
  already running is left to finish, and cancelling must never cost an
  extra remote call: no vendor cancel, no DELETE beyond the one the normal
  flow makes, no retry. The next attempt of the retry ladder is not
  started, and the solve raises ``SolveCancelled``;
* ``solver.wall_clock_limit_seconds`` is refused as
  WALL_CLOCK_LIMIT_UNSUPPORTED by validate, solve and recommend, before any
  remote call at all.

Nothing here talks to a vendor: the Fujitsu backend runs against the
scripted :class:`~tests.fakes.FakeDATransport` (every HTTP request
recorded) and the Ocean backends against the counting fake samplers of
``tests/remote_mock/conftest.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

import annealbridge.solvers.dwave_qpu as qpu_module
import annealbridge.solvers.leap_hybrid_bqm as leap_module
import annealbridge.solvers.leap_hybrid_cqm as cqm_module
from annealbridge.compiler import BQMCompiler
from annealbridge.orchestration import (
    CancelToken,
    ExecutionPolicy,
    OptimizationService,
    SolveCancelled,
)
from annealbridge.solvers import (
    DWaveQPUBackend,
    FujitsuDABackend,
    LeapHybridBQMBackend,
    LeapHybridCQMBackend,
    SolverRegistry,
)
from tests.fakes import FakeDATransport, bits_solution
from tests.remote_mock.conftest import (
    FAKE_MINIMAL_SAMPLESET_INFO,
    CountingFactory,
    FakeCQMSampler,
    FakeLeapHybridSampler,
    FakeQPUSampler,
    make_problem,
    make_remote_available,
)
from tests.remote_mock.test_fujitsu_da_mock import set_key
from tests.remote_mock.test_service_remote_flow import make_zero_infeasible_problem

# One attempt, then two retries -- all of them taken when nothing cancels,
# because the all-zero sample the fakes return is infeasible.
MAX_RETRIES = 2
ATTEMPTS_WITHOUT_CANCEL = 1 + MAX_RETRIES


def retrying_service(backend, name: str, **policy) -> OptimizationService:
    return OptimizationService(
        registry=SolverRegistry({name: backend}),
        policy=ExecutionPolicy(
            allow_remote=True, allow_remote_retries=True, **policy
        ),
    )


def wire_shape(fake: FakeDATransport) -> list[tuple[str, str]]:
    """``(method, path)`` of every recorded request, in order."""
    return [
        (request.method, request.url.split("://", 1)[-1].split("/", 1)[-1])
        for request in fake.requests
    ]


# ---------------------------------------------------------------------------
# Fujitsu DA: HTTP, every request recorded.
# ---------------------------------------------------------------------------


class TestFujitsuCancel:
    """A cancel during the vendor job's polling costs nothing extra."""

    def build(self, monkeypatch, on_sleep: Callable[[float], None] = lambda _: None):
        set_key(monkeypatch)
        problem = make_zero_infeasible_problem(
            backend="fujitsu_da", max_retries=MAX_RETRIES
        )
        width = len(BQMCompiler().compile(problem, hard_penalty=100.0).model.variables)
        fake = FakeDATransport(solutions=[bits_solution([0] * width, 0.0)])
        # The backend sleeps between polls: the one moment a test can act
        # while a vendor job is in flight.
        backend = FujitsuDABackend(transport=fake, clock=lambda: 0.0, sleep=on_sleep)
        return problem, fake, retrying_service(backend, "fujitsu_da")

    def test_without_cancel_every_retry_is_submitted(self, monkeypatch):
        problem, fake, service = self.build(monkeypatch)

        result = service.solve(problem, cancel=CancelToken())

        assert result.status == "infeasible"
        assert len(result.attempts) == ATTEMPTS_WITHOUT_CANCEL
        assert fake.submit_calls == ATTEMPTS_WITHOUT_CANCEL
        assert fake.delete_calls == ATTEMPTS_WITHOUT_CANCEL
        assert fake.cancel_calls == 0

    def test_cancel_mid_job_adds_no_request_and_starts_no_retry(self, monkeypatch):
        token = CancelToken()
        problem, fake, service = self.build(monkeypatch, lambda _: token.cancel())
        # What one uncancelled attempt sends: submit, polls, cleanup DELETE.
        _, reference, reference_service = self.build(monkeypatch)
        single = problem.solver.model_copy(update={"max_retries": 0})
        reference_service.solve(problem.model_copy(update={"solver": single}))
        assert ("POST", "v4/async/jobs/cancel") not in wire_shape(reference)

        with pytest.raises(SolveCancelled):
            service.solve(problem, cancel=token)

        # The first poll-sleep cancelled the token; the job already
        # submitted ran to the end exactly as an uncancelled single
        # attempt does (submit, polls, the normal cleanup DELETE) -- no
        # vendor cancel, no extra DELETE, and no second submit.
        assert token.cancelled
        assert wire_shape(fake) == wire_shape(reference)
        assert fake.submit_calls == 1
        assert fake.delete_calls == 1
        assert fake.cancel_calls == 0

    def test_a_token_cancelled_up_front_makes_no_request(self, monkeypatch):
        problem, fake, service = self.build(monkeypatch)
        token = CancelToken()
        token.cancel()

        with pytest.raises(SolveCancelled):
            service.solve(problem, cancel=token)

        assert fake.requests == []

    def test_the_concurrency_slot_is_released_after_a_cancel(self, monkeypatch):
        token = CancelToken()
        problem, fake, _ = self.build(monkeypatch)
        backend = FujitsuDABackend(
            transport=fake, clock=lambda: 0.0, sleep=lambda _: token.cancel()
        )
        service = retrying_service(backend, "fujitsu_da", max_concurrent_solves=1)

        with pytest.raises(SolveCancelled):
            service.solve(problem, cancel=token)
        # One slot only: a leaked slot would make this CONCURRENCY_LIMIT.
        result = service.solve(problem)

        assert result.status == "infeasible"
        assert len(result.attempts) == ATTEMPTS_WITHOUT_CANCEL


# ---------------------------------------------------------------------------
# D-Wave QPU: the Ocean SDK seam, sampler calls counted.
# ---------------------------------------------------------------------------


class CancellingQPUSampler(FakeQPUSampler):
    """QPU fake whose ``sample()`` cancels a token while the call runs."""

    def __init__(self, token: CancelToken | None) -> None:
        super().__init__(
            assignments=[{"a": 0, "b": 0}], info=FAKE_MINIMAL_SAMPLESET_INFO
        )
        self.token = token

    def sample(self, bqm, **kwargs):
        if self.token is not None:
            self.token.cancel()
        return super().sample(bqm, **kwargs)


class TestQPUCancel:
    def solve(self, monkeypatch, token: CancelToken | None):
        make_remote_available(monkeypatch, qpu_module)
        sampler = CancellingQPUSampler(token)
        factory = CountingFactory(sampler)
        service = retrying_service(
            DWaveQPUBackend(sampler_factory=factory), "dwave_qpu"
        )
        problem = make_zero_infeasible_problem(max_retries=MAX_RETRIES)
        return sampler, factory, service, problem

    def test_without_cancel_every_retry_samples(self, monkeypatch):
        sampler, factory, service, problem = self.solve(monkeypatch, None)

        result = service.solve(problem, cancel=CancelToken())

        assert result.status == "infeasible"
        assert sampler.sample_calls == ATTEMPTS_WITHOUT_CANCEL

    def test_cancel_during_sampling_starts_no_further_attempt(self, monkeypatch):
        token = CancelToken()
        sampler, factory, service, problem = self.solve(monkeypatch, token)

        with pytest.raises(SolveCancelled):
            service.solve(problem, cancel=token)

        assert sampler.sample_calls == 1
        assert factory.calls == 1


# ---------------------------------------------------------------------------
# §9 item 6: every remote backend refuses a wall-clock limit, remotely silent.
# ---------------------------------------------------------------------------


@dataclass
class WiredRemote:
    """A remote backend on its fake, plus the count of remote calls made."""

    backend: object
    remote_calls: Callable[[], int]


def _wire_qpu(monkeypatch) -> WiredRemote:
    make_remote_available(monkeypatch, qpu_module)
    sampler = FakeQPUSampler()
    factory = CountingFactory(sampler)
    return WiredRemote(
        DWaveQPUBackend(sampler_factory=factory),
        lambda: factory.calls + sampler.sample_calls,
    )


def _wire_hybrid_bqm(monkeypatch) -> WiredRemote:
    make_remote_available(monkeypatch, leap_module)
    sampler = FakeLeapHybridSampler()
    factory = CountingFactory(sampler)
    return WiredRemote(
        LeapHybridBQMBackend(sampler_factory=factory),
        lambda: factory.calls + sampler.sample_calls + sampler.min_time_limit_calls,
    )


def _wire_hybrid_cqm(monkeypatch) -> WiredRemote:
    make_remote_available(monkeypatch, cqm_module)
    sampler = FakeCQMSampler()
    factory = CountingFactory(sampler)
    return WiredRemote(
        LeapHybridCQMBackend(sampler_factory=factory),
        lambda: factory.calls + sampler.sample_calls + sampler.min_time_limit_calls,
    )


def _wire_fujitsu(monkeypatch) -> WiredRemote:
    set_key(monkeypatch)
    fake = FakeDATransport()
    return WiredRemote(
        FujitsuDABackend(transport=fake, clock=lambda: 0.0, sleep=lambda _: None),
        lambda: len(fake.requests),
    )


WIRING: dict[str, Callable[..., WiredRemote]] = {
    "dwave_qpu": _wire_qpu,
    "leap_hybrid_bqm": _wire_hybrid_bqm,
    "leap_hybrid_cqm": _wire_hybrid_cqm,
    "fujitsu_da": _wire_fujitsu,
}


def _remote_backend_names() -> list[str]:
    defaults = SolverRegistry.default()
    return [
        name for name in defaults.names() if defaults.get(name).capabilities.remote
    ]


def test_every_remote_backend_is_wired_here():
    # A new remote backend must join the refusal test below.
    assert sorted(_remote_backend_names()) == sorted(WIRING)


@pytest.mark.parametrize("name", _remote_backend_names())
def test_remote_backend_refuses_a_wall_clock_limit_without_calling_out(
    monkeypatch, name
):
    wired = WIRING[name](monkeypatch)
    assert wired.backend.capabilities.supports_interrupt is False
    service = OptimizationService(
        registry=SolverRegistry({name: wired.backend}),
        policy=ExecutionPolicy(allow_remote=True),
    )
    problem = make_problem(backend=name, wall_clock_limit_seconds=5)

    validation = service.validate(problem)
    result = service.solve(problem, cancel=CancelToken())
    (entry,) = service.recommend(problem).recommendations

    assert validation.valid is False
    assert [e.code for e in validation.errors] == ["WALL_CLOCK_LIMIT_UNSUPPORTED"]
    assert result.status == "invalid_problem"
    assert [e.code for e in result.errors] == ["WALL_CLOCK_LIMIT_UNSUPPORTED"]
    assert result.errors[0].path == "solver.wall_clock_limit_seconds"
    assert result.attempts == []
    assert entry.backend == name
    assert entry.usable is False
    assert "WALL_CLOCK_LIMIT_UNSUPPORTED" in [b.code for b in entry.blocking]
    assert wired.remote_calls() == 0
