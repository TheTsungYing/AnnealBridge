"""The shared D-Wave backend steps in ``solvers.ocean`` (2026-09-09 review F-13c).

``resolve_hybrid_time_limit`` used to exist twice, once per Leap hybrid
backend, word for word apart from the message label; ``create_sampler``
three times. The rule is pinned here once, with a fake sampler; the
backend-level tests in ``tests/remote_mock`` still exercise each backend
through the shared code. Nothing here imports ``dwave.system``.
"""

import threading
import time

import pytest

from annealbridge.exceptions import SolverExecutionError
import annealbridge.solvers.ocean as ocean_module
from annealbridge.solvers.ocean import (
    HYBRID_SAMPLE_EXCEPTION_CODES,
    SAMPLER_INIT_EXCEPTION_CODES,
    LazySampler,
    call_ocean,
    create_sampler,
    resolve_hybrid_time_limit,
)


class FakeSampler:
    def __init__(self, minimum: float = 5.0) -> None:
        self.minimum = minimum
        self.seen_models: list[object] = []

    def min_time_limit(self, model: object) -> float:
        self.seen_models.append(model)
        return self.minimum


class RequestTimeout(Exception):
    """Same class name as Ocean's, which is all the classifier looks at."""


class SolverAuthenticationError(Exception):
    """Same class name as Ocean's → REMOTE_AUTH_FAILED."""


class CountingFactory:
    """Returns a brand-new object per call and counts the calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return object()


def holder_of(sampler: object) -> LazySampler:
    return LazySampler(lambda: sampler, lambda: sampler)


MODEL = object()


class TestResolveHybridTimeLimit:
    @pytest.mark.parametrize(
        "user_value,expected",
        [(None, 5.0), (10, 10.0), (10.5, 10.5), (3, 5.0), (5.0, 5.0)],
    )
    def test_user_value_is_floored_at_the_sampler_minimum(self, user_value, expected):
        sampler = FakeSampler(minimum=5.0)

        result = resolve_hybrid_time_limit(
            holder_of(sampler), MODEL, user_value, label="Leap hybrid"
        )

        assert result == expected
        assert isinstance(result, float)
        # The minimum is asked for exactly this model, once.
        assert sampler.seen_models == [MODEL]

    def test_sampler_construction_failure_is_a_config_error_with_the_label(self):
        def factory():
            raise ValueError("bad region")

        holder = LazySampler(factory, factory)

        with pytest.raises(SolverExecutionError) as info:
            resolve_hybrid_time_limit(holder, MODEL, None, label="Leap hybrid CQM")

        assert info.value.code == "DWAVE_CONFIG_INVALID"
        assert str(info.value).startswith("Leap hybrid CQM sampler could not be created")
        assert info.value.__cause__ is None and info.value.__context__ is None

    def test_min_time_limit_failure_is_classified_with_the_label(self):
        class Sampler:
            def min_time_limit(self, model):
                raise RequestTimeout("slow")

        with pytest.raises(SolverExecutionError) as info:
            resolve_hybrid_time_limit(holder_of(Sampler()), MODEL, 7.0, label="Leap hybrid")

        assert info.value.code == "REMOTE_TIMEOUT"
        assert str(info.value).startswith(
            "Leap hybrid minimum time limit could not be determined"
        )


class TestCreateSampler:
    def test_returns_the_cached_sampler(self):
        sampler = FakeSampler()
        holder = holder_of(sampler)

        assert create_sampler(holder, "D-Wave QPU") is sampler
        assert create_sampler(holder, "D-Wave QPU") is sampler

    def test_failure_message_carries_the_label(self):
        def factory():
            raise ValueError("no such solver")

        with pytest.raises(SolverExecutionError) as info:
            create_sampler(LazySampler(factory, factory), "D-Wave QPU")

        assert info.value.code == "DWAVE_CONFIG_INVALID"
        assert str(info.value).startswith("D-Wave QPU sampler could not be created")


class TestLazySamplerCache:
    """2026-09-09 review F-21: the cache is bound to the credential state.

    A sampler used to be cached for the life of the backend instance, so a
    long-running server that had its Leap token rotated kept talking to
    the cloud with a client built from the old one. The cache key is now a
    *fingerprint* of the credential sources; the default one reads the env
    token and the Ocean config files, and tests inject their own.
    """

    def test_a_stable_fingerprint_keeps_one_sampler(self):
        state = {"fp": "a"}
        factory = CountingFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: state["fp"])

        first = holder.get()

        assert holder.get() is first
        assert factory.calls == 1

    def test_a_changed_fingerprint_rebuilds_the_sampler(self):
        state = {"fp": "a"}
        factory = CountingFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: state["fp"])
        first = holder.get()

        state["fp"] = "b"
        second = holder.get()

        assert second is not first
        assert factory.calls == 2

    def test_returning_to_an_earlier_fingerprint_still_rebuilds(self):
        """One entry, not a growing map of every credential ever seen:
        going back to a previous fingerprint rebuilds rather than serving
        a stale client the intervening rotation may have invalidated."""
        state = {"fp": "a"}
        factory = CountingFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: state["fp"])
        first = holder.get()
        state["fp"] = "b"
        second = holder.get()

        state["fp"] = "a"
        third = holder.get()

        assert third is not first and third is not second
        assert factory.calls == 3

    def test_invalidate_forces_the_next_get_to_rebuild(self):
        factory = CountingFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: "fixed")
        first = holder.get()

        holder.invalidate()
        second = holder.get()

        assert second is not first
        assert factory.calls == 2

    def test_concurrent_first_use_builds_exactly_one_sampler(self):
        """Thundering herd: a real sampler construction takes seconds, so
        four threads racing on first use must not build four clients."""
        factory_calls: list[int] = []

        def slow_factory() -> object:
            factory_calls.append(1)
            time.sleep(0.05)
            return object()

        holder = LazySampler(slow_factory, slow_factory, fingerprint=lambda: "fixed")
        results: list[object] = []
        results_lock = threading.Lock()

        def worker() -> None:
            sampler = holder.get()
            with results_lock:
                results.append(sampler)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(factory_calls) == 1
        assert len(results) == 4
        assert all(sampler is results[0] for sampler in results)

    def test_the_default_fingerprint_follows_the_env_token(self, monkeypatch):
        """No explicit fingerprint: rotating ``DWAVE_API_TOKEN`` is enough."""
        monkeypatch.setenv(ocean_module.TOKEN_ENV, "DEV-FAKE-TOKEN-1234567890abcdefghij")
        factory = CountingFactory()
        holder = LazySampler(factory, factory)
        first = holder.get()

        assert factory.calls == 1

        monkeypatch.setenv(
            ocean_module.TOKEN_ENV, "DEV-FAKE-TOKEN-ROTATED-0987654321zyxwv"
        )
        second = holder.get()

        assert second is not first
        assert factory.calls == 2
        assert holder.get() is second
        assert factory.calls == 2


class TestCallOceanInvalidation:
    """F-21, second half: an auth failure drops the cached sampler.

    A fingerprint only notices credentials this process can *see*. When
    the cloud itself rejects the client (revoked token, expired session),
    the cached sampler is known-bad, so the guard that classifies the
    failure also throws it away — the next solve builds a fresh one.
    """

    def _holder(self) -> tuple[LazySampler, CountingFactory]:
        factory = CountingFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: "fixed")
        holder.get()
        return holder, factory

    def test_an_auth_failure_invalidates_the_holder(self):
        holder, factory = self._holder()

        def boom():
            raise SolverAuthenticationError("denied")

        with pytest.raises(SolverExecutionError) as info:
            call_ocean("solve", HYBRID_SAMPLE_EXCEPTION_CODES, boom, holder=holder)

        assert info.value.code == "REMOTE_AUTH_FAILED"
        assert info.value.__cause__ is None and info.value.__context__ is None

        holder.get()

        assert factory.calls == 2

    def test_another_failure_keeps_the_cached_sampler(self):
        """A timeout says nothing about the credentials; rebuilding the
        sampler on every slow request would be pure waste."""
        holder, factory = self._holder()

        def boom():
            raise RequestTimeout("slow")

        with pytest.raises(SolverExecutionError) as info:
            call_ocean("solve", HYBRID_SAMPLE_EXCEPTION_CODES, boom, holder=holder)

        assert info.value.code == "REMOTE_TIMEOUT"

        holder.get()

        assert factory.calls == 1

    def test_an_auth_failure_without_a_holder_still_raises(self):
        """``create_sampler`` passes no holder: there is nothing to drop."""

        def boom():
            raise SolverAuthenticationError("denied")

        with pytest.raises(SolverExecutionError) as info:
            call_ocean("creating sampler", SAMPLER_INIT_EXCEPTION_CODES, boom)

        assert info.value.code == "REMOTE_AUTH_FAILED"
