"""The stop condition types and their declaration rules (batch 6 J).

Time-limit spec 2026-09-24 §5, §6.1 and §9 item 7:

* ``CancelToken`` is a one-way, idempotent flag; ``Interrupt`` answers
  ``should_stop`` from a deadline on an injected clock, a token, or both,
  with no side effect and from any number of threads;
* ``SolveCancelled`` is not an ``OptimizerError`` and is exported from
  ``annealbridge.orchestration`` together with ``CancelToken``;
* ``SolverCapabilities`` refuses ``exhaustive`` together with
  ``supports_interrupt``;
* ``OptimizationService`` refuses, when it is built, a backend that
  declares ``supports_interrupt`` but whose ``solve`` takes no
  ``interrupt`` keyword.
"""

import threading

import pytest
from pydantic import ValidationError

from annealbridge.exceptions import OptimizerError, SolveCancelled
from annealbridge.interrupt import CancelToken, Interrupt
from annealbridge.orchestration import OptimizationService
from annealbridge.solvers import SolverRegistry
from annealbridge.solvers.base import (
    AvailabilityStatus,
    RawSolverResult,
    SolverCapabilities,
)
from tests.fakes.interruptible_backend import FakeClock, ScriptedInterruptBackend


class TestCancelToken:
    def test_a_new_token_is_not_cancelled(self):
        assert CancelToken().cancelled is False

    def test_cancel_is_permanent_and_idempotent(self):
        token = CancelToken()
        token.cancel()
        token.cancel()
        assert token.cancelled is True

    def test_cancel_from_another_thread_is_seen(self):
        token = CancelToken()
        thread = threading.Thread(target=token.cancel)
        thread.start()
        thread.join()
        assert token.cancelled is True


class TestInterrupt:
    def test_neither_deadline_nor_token_never_stops(self):
        interrupt = Interrupt(deadline=None, token=None, clock=FakeClock(1e9))
        assert interrupt.should_stop() is False
        assert interrupt.cancelled is False
        assert interrupt.deadline_passed() is False

    def test_the_deadline_is_reached_at_exactly_the_deadline(self):
        clock = FakeClock(0.0)
        interrupt = Interrupt(deadline=10.0, token=None, clock=clock)
        clock.now = 9.999
        assert interrupt.deadline_passed() is False
        assert interrupt.should_stop() is False
        clock.now = 10.0
        assert interrupt.deadline_passed() is True
        assert interrupt.should_stop() is True
        assert interrupt.cancelled is False

    def test_a_cancelled_token_stops_without_a_deadline(self):
        token = CancelToken()
        interrupt = Interrupt(deadline=None, token=token, clock=FakeClock())
        assert interrupt.should_stop() is False
        token.cancel()
        assert interrupt.cancelled is True
        assert interrupt.deadline_passed() is False
        assert interrupt.should_stop() is True

    def test_both_reasons_are_reported_separately(self):
        token = CancelToken()
        clock = FakeClock(0.0)
        interrupt = Interrupt(deadline=5.0, token=token, clock=clock)
        clock.now = 6.0
        token.cancel()
        assert interrupt.cancelled is True
        assert interrupt.deadline_passed() is True

    def test_should_stop_has_no_side_effect(self):
        clock = FakeClock(0.0)
        interrupt = Interrupt(deadline=5.0, token=CancelToken(), clock=clock)
        answers = {interrupt.should_stop() for _ in range(100)}
        assert answers == {False}
        # Only the clock is read; asking never moves anything forward.
        assert clock.now == 0.0

    def test_the_default_clock_is_a_real_monotonic_clock(self):
        # No clock injected: a deadline already in the past has passed, one
        # far in the future has not.
        assert Interrupt(deadline=float("-inf"), token=None).deadline_passed()
        assert not Interrupt(deadline=float("inf"), token=None).deadline_passed()

    def test_should_stop_from_many_threads_at_once(self):
        token = CancelToken()
        interrupt = Interrupt(deadline=None, token=token, clock=FakeClock())
        start = threading.Barrier(8)
        seen: list[bool] = []
        lock = threading.Lock()

        def poll() -> None:
            start.wait()
            answers = [interrupt.should_stop() for _ in range(1000)]
            with lock:
                seen.extend(answers)

        threads = [threading.Thread(target=poll) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert seen == [False] * 8000
        token.cancel()
        assert interrupt.should_stop() is True


class TestExports:
    def test_solve_cancelled_is_not_a_domain_failure(self):
        # The service's handlers turn every OptimizerError into a structured
        # solver_error; a cancellation must get through them.
        assert issubclass(SolveCancelled, Exception)
        assert not issubclass(SolveCancelled, OptimizerError)

    def test_orchestration_exports_the_caller_facing_types(self):
        import annealbridge.orchestration as orchestration

        assert orchestration.CancelToken is CancelToken
        assert orchestration.SolveCancelled is SolveCancelled
        assert {"CancelToken", "SolveCancelled"} <= set(orchestration.__all__)


def _capabilities(**overrides) -> SolverCapabilities:
    values = {
        "name": "declared",
        "remote": False,
        "heuristic": True,
        "exhaustive": False,
        "supports_seed": False,
        "supports_num_reads": False,
        "supports_time_limit": False,
        "supported_model_types": ["bqm"],
        "returns_multiple_samples": True,
        "description": "test-only",
    }
    values.update(overrides)
    return SolverCapabilities(**values)


class TestCapabilityDeclaration:
    def test_supports_interrupt_defaults_to_false(self):
        assert _capabilities().supports_interrupt is False

    def test_an_exhaustive_backend_cannot_declare_supports_interrupt(self):
        with pytest.raises(ValidationError, match="supports_interrupt"):
            _capabilities(exhaustive=True, heuristic=False, supports_interrupt=True)

    def test_each_flag_alone_is_accepted(self):
        assert _capabilities(exhaustive=True, heuristic=False).exhaustive is True
        assert _capabilities(supports_interrupt=True).supports_interrupt is True

    def test_the_shipped_declarations(self):
        # Read from the declarations, which is what the service and the
        # validator read too; exact enumerates (a cut enumeration proves
        # nothing) and the remote backends are out of scope for batch 6.
        registry = SolverRegistry.default()
        declared = {
            name
            for name in registry.names()
            if registry.get(name).capabilities.supports_interrupt
        }
        assert declared == {"simulated_annealing", "tabu", "simulated_bifurcation"}
        for name in registry.names():
            caps = registry.get(name).capabilities
            assert not (caps.exhaustive and caps.supports_interrupt), name


class _BackendBase:
    """The protocol members every backend below shares; only solve differs."""

    def __init__(self, supports_interrupt: bool = True) -> None:
        self.capabilities = _capabilities(supports_interrupt=supports_interrupt)

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return False

    def is_available(self) -> AvailabilityStatus:
        return AvailabilityStatus(category="available")

    def resolve_time_limit(self, compiled, preferences):
        return None


class _NoInterruptParameter(_BackendBase):
    def solve(self, compiled, preferences) -> RawSolverResult:  # pragma: no cover
        raise AssertionError("never called")


class _PositionalVarargsOnly(_BackendBase):
    def solve(self, compiled, preferences, *args) -> RawSolverResult:  # pragma: no cover
        raise AssertionError("never called")


class _PositionalOnlyInterrupt(_BackendBase):
    def solve(self, compiled, preferences, interrupt=None, /) -> RawSolverResult:  # pragma: no cover
        raise AssertionError("never called")


class _KeywordOnlyInterrupt(_BackendBase):
    def solve(self, compiled, preferences, *, interrupt=None) -> RawSolverResult:  # pragma: no cover
        raise AssertionError("never called")


class _PositionalOrKeywordInterrupt(_BackendBase):
    def solve(self, compiled, preferences, interrupt=None) -> RawSolverResult:  # pragma: no cover
        raise AssertionError("never called")


def _service(backend) -> OptimizationService:
    return OptimizationService(registry=SolverRegistry({backend.name: backend}))


class TestConstructionCheck:
    @pytest.mark.parametrize(
        "backend_class",
        [_NoInterruptParameter, _PositionalVarargsOnly, _PositionalOnlyInterrupt],
    )
    def test_a_declared_backend_without_the_keyword_is_refused(self, backend_class):
        with pytest.raises(ValueError, match="declares supports_interrupt") as exc_info:
            _service(backend_class())
        assert "'declared'" in str(exc_info.value)
        assert "'interrupt'" in str(exc_info.value)

    @pytest.mark.parametrize(
        "backend_class", [_KeywordOnlyInterrupt, _PositionalOrKeywordInterrupt]
    )
    def test_a_declared_backend_taking_the_keyword_is_accepted(self, backend_class):
        _service(backend_class())

    def test_var_keyword_is_accepted(self):
        _service(ScriptedInterruptBackend([], supports_interrupt=True))

    def test_an_undeclared_backend_is_never_inspected(self):
        # No declaration, no requirement: the two-argument solve stays valid.
        _service(_NoInterruptParameter(supports_interrupt=False))

    def test_the_default_registry_passes_the_check(self):
        OptimizationService()
