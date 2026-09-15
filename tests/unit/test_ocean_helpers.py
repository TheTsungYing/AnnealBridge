"""The shared D-Wave backend steps in ``solvers.ocean`` (2026-09-09 review F-13c).

``resolve_hybrid_time_limit`` used to exist twice, once per Leap hybrid
backend, word for word apart from the message label; ``create_sampler``
three times. The rule is pinned here once, with a fake sampler; the
backend-level tests in ``tests/remote_mock`` still exercise each backend
through the shared code. Nothing here imports ``dwave.system``.
"""

import gc
import logging
import threading
import time

import pytest

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CredentialDeclaration, SolverCapabilities
import annealbridge.solvers.metadata as metadata_module
import annealbridge.solvers.ocean as ocean_module
from annealbridge.solvers.ocean import (
    HYBRID_SAMPLE_EXCEPTION_CODES,
    SAMPLER_INIT_EXCEPTION_CODES,
    HybridTimeLimitMemo,
    LazySampler,
    call_ocean,
    create_sampler,
    resolve_hybrid_sampler,
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
    """Returns a brand-new object per call and counts the calls.

    A bare ``object()`` has no ``close`` attribute, which is also what the
    ``tests/remote_mock`` fake samplers look like: dropping one must stay a
    no-op (review F-07).
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return object()


class ClosableSampler:
    """A sampler that records how often it was closed (review F-07)."""

    def __init__(self, name: str = "s", fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        if self.fail:
            raise RuntimeError("client shutdown failed")


class ClosableFactory:
    """Hands out a fresh :class:`ClosableSampler` per call, keeping them all."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.built: list[ClosableSampler] = []

    def __call__(self) -> ClosableSampler:
        sampler = ClosableSampler(name=f"s{len(self.built)}", fail=self.fail)
        self.built.append(sampler)
        return sampler


def holder_of(sampler: object) -> LazySampler:
    return LazySampler(lambda: sampler, lambda: sampler)


MODEL = object()


class TestOceanSamplerHolder:
    """2026-09-15 consolidation: the holder a D-Wave constructor gets comes
    with its redaction.

    ``ocean_sampler_holder`` is the only way the Ocean backends build their
    :class:`LazySampler`, so the credential declaration and the config-file
    token source cannot be forgotten by one of them. Pinned with a fake
    vendor, starting from empty process-level tables.
    """

    FAKE_ENV = "FAKE_OCEAN_LIKE_API_KEY"
    FAKE_VALUE = "fake-ocean-like-secret-0123456789"

    @pytest.fixture(autouse=True)
    def _clean_tables(self, monkeypatch):
        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})
        # The registered source must not read a real Ocean config file.
        monkeypatch.setattr(ocean_module, "_ocean_config_secrets", lambda: ())

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            name="fake_ocean_backend",
            remote=True,
            heuristic=True,
            exhaustive=False,
            supports_seed=False,
            supports_num_reads=False,
            supports_time_limit=True,
            supported_model_types=["bqm"],
            returns_multiple_samples=False,
            description="Test double.",
            credentials=CredentialDeclaration(env_vars=[self.FAKE_ENV]),
        )

    def test_declares_the_credentials_and_registers_the_config_source(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv(self.FAKE_ENV, self.FAKE_VALUE)
        assert metadata_module.redact(self.FAKE_VALUE) == self.FAKE_VALUE

        ocean_module.ocean_sampler_holder(None, CountingFactory(), self.capabilities())

        assert metadata_module.credential_env_vars() == [self.FAKE_ENV]
        assert "ocean_config" in metadata_module._SECRET_SOURCES
        assert metadata_module.redact(f"key {self.FAKE_VALUE}") == "key ***"

    def test_returns_a_lazy_holder_that_builds_nothing_yet(self) -> None:
        seam = CountingFactory()
        default = CountingFactory()

        holder = ocean_module.ocean_sampler_holder(seam, default, self.capabilities())

        assert isinstance(holder, LazySampler)
        assert (seam.calls, default.calls) == (0, 0)
        holder.get()
        assert (seam.calls, default.calls) == (1, 0)

    def test_without_a_seam_the_default_factory_is_used(self) -> None:
        default = CountingFactory()

        holder = ocean_module.ocean_sampler_holder(None, default, self.capabilities())
        holder.get()

        assert default.calls == 1


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


class Model:
    """A weakly referenceable stand-in for a compiled model (``object()`` is not)."""


class TestResolveHybridSampler:
    """The pair form: the same rule as ``resolve_hybrid_time_limit`` plus the sampler."""

    @pytest.mark.parametrize("user_value,expected", [(None, 5.0), (10, 10.0), (3, 5.0)])
    def test_returns_the_holder_sampler_and_the_same_limit(self, user_value, expected):
        sampler = FakeSampler(minimum=5.0)
        holder = holder_of(sampler)

        got_sampler, limit = resolve_hybrid_sampler(
            holder, MODEL, user_value, label="Leap hybrid"
        )

        assert got_sampler is sampler
        assert limit == expected
        assert limit == resolve_hybrid_time_limit(holder, MODEL, user_value, label="x")

    def test_without_a_memo_every_call_asks_the_sampler(self):
        sampler = FakeSampler(minimum=5.0)
        holder = holder_of(sampler)

        resolve_hybrid_sampler(holder, MODEL, None, label="Leap hybrid")
        resolve_hybrid_sampler(holder, MODEL, None, label="Leap hybrid")

        assert sampler.seen_models == [MODEL, MODEL]

    def test_construction_failure_is_classified_like_the_limit_form(self):
        def factory():
            raise ValueError("bad region")

        with pytest.raises(SolverExecutionError) as info:
            resolve_hybrid_sampler(LazySampler(factory, factory), MODEL, None, label="L")

        assert info.value.code == "DWAVE_CONFIG_INVALID"


class TestHybridTimeLimitMemo:
    """One attempt asks ``min_time_limit`` once; anything else is a miss.

    The memo is keyed by the identity of the sampler *and* the model, both
    held weakly: the service's pre-submission check and the solve of the
    same attempt share one model object and one sampler, while a rebuilt
    sampler, a recompiled model or a collected one must recompute. The
    value is never changed by a hit.
    """

    def test_the_same_sampler_and_model_ask_once(self):
        sampler = FakeSampler(minimum=5.0)
        holder = holder_of(sampler)
        model = Model()
        memo = HybridTimeLimitMemo()

        first = resolve_hybrid_sampler(holder, model, 2.0, label="L", memo=memo)
        second = resolve_hybrid_sampler(holder, model, 2.0, label="L", memo=memo)
        third = resolve_hybrid_time_limit(holder, model, 9.0, label="L", memo=memo)

        assert sampler.seen_models == [model]
        assert first == (sampler, 5.0)
        assert second == (sampler, 5.0)
        # The user's value still floors on top of the memoised minimum.
        assert third == 9.0

    def test_another_model_is_a_miss(self):
        sampler = FakeSampler(minimum=5.0)
        holder = holder_of(sampler)
        memo = HybridTimeLimitMemo()
        first, second = Model(), Model()

        resolve_hybrid_sampler(holder, first, None, label="L", memo=memo)
        resolve_hybrid_sampler(holder, second, None, label="L", memo=memo)
        resolve_hybrid_sampler(holder, first, None, label="L", memo=memo)

        # Single slot: going back to the first model recomputes too.
        assert sampler.seen_models == [first, second, first]

    def test_a_rebuilt_sampler_is_a_miss(self):
        samplers = [FakeSampler(minimum=5.0), FakeSampler(minimum=7.0)]
        holder = LazySampler(lambda: samplers.pop(0), lambda: None)
        model = Model()
        memo = HybridTimeLimitMemo()

        _, before = resolve_hybrid_sampler(holder, model, None, label="L", memo=memo)
        holder.invalidate()
        rebuilt, after = resolve_hybrid_sampler(holder, model, None, label="L", memo=memo)

        assert before == 5.0
        assert after == 7.0
        assert rebuilt.seen_models == [model]

    def test_a_collected_model_is_a_miss(self):
        sampler = FakeSampler(minimum=5.0)
        holder = holder_of(sampler)
        memo = HybridTimeLimitMemo()

        model = Model()
        resolve_hybrid_sampler(holder, model, None, label="L", memo=memo)
        del model
        gc.collect()
        replacement = Model()
        resolve_hybrid_sampler(holder, replacement, None, label="L", memo=memo)

        assert len(sampler.seen_models) == 2
        assert sampler.seen_models[-1] is replacement

    def test_a_model_that_cannot_be_weakly_referenced_is_not_memoised(self):
        sampler = FakeSampler(minimum=5.0)
        holder = holder_of(sampler)
        memo = HybridTimeLimitMemo()

        resolve_hybrid_sampler(holder, MODEL, None, label="L", memo=memo)
        resolve_hybrid_sampler(holder, MODEL, None, label="L", memo=memo)

        assert sampler.seen_models == [MODEL, MODEL]
        assert memo.lookup(sampler, MODEL) is None

    def test_a_failed_min_time_limit_stores_nothing(self):
        class Sampler:
            def __init__(self):
                self.calls = 0

            def min_time_limit(self, model):
                self.calls += 1
                if self.calls == 1:
                    raise RequestTimeout("slow")
                return 4.0

        sampler = Sampler()
        holder = holder_of(sampler)
        model = Model()
        memo = HybridTimeLimitMemo()

        with pytest.raises(SolverExecutionError):
            resolve_hybrid_sampler(holder, model, None, label="L", memo=memo)
        _, limit = resolve_hybrid_sampler(holder, model, None, label="L", memo=memo)

        assert limit == 4.0
        assert sampler.calls == 2


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


class TestLazySamplerClosesWhatItDrops:
    """2026-09-11 review F-07: a dropped sampler is released, not leaked.

    A real Ocean sampler owns a ``dwave.cloud.Client`` — worker threads and
    an HTTP session — and the holder used to drop its reference and nothing
    more, so every credential rotation and every ``REMOTE_AUTH_FAILED`` left
    one behind in a long-running MCP server. ``close()`` is best effort: a
    sampler that has none is left alone and a failing one is logged, never
    raised.
    """

    def test_a_superseded_sampler_is_closed_once_and_the_new_one_is_not(self):
        state = {"fp": "a"}
        factory = ClosableFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: state["fp"])
        first = holder.get()

        state["fp"] = "b"
        second = holder.get()

        assert second is not first
        assert first.closed == 1
        assert second.closed == 0
        # Still cached, so nothing more is closed.
        assert holder.get() is second
        assert first.closed == 1 and second.closed == 0

    def test_invalidate_closes_the_dropped_sampler(self):
        factory = ClosableFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: "fixed")
        first = holder.get()

        holder.invalidate()

        assert first.closed == 1

        second = holder.get()

        assert second is not first
        assert first.closed == 1
        assert second.closed == 0

    def test_repeated_invalidate_does_not_close_twice(self):
        """The slot is already empty: there is nothing left to close."""
        factory = ClosableFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: "fixed")
        first = holder.get()

        holder.invalidate()
        holder.invalidate()
        holder.invalidate()

        assert first.closed == 1

    def test_invalidate_before_any_get_closes_nothing(self):
        factory = ClosableFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: "fixed")

        holder.invalidate()

        assert factory.built == []

    def test_a_sampler_without_close_is_dropped_silently(self):
        """What every test fake (and ``tests/remote_mock``) looks like."""
        state = {"fp": "a"}
        factory = CountingFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: state["fp"])
        holder.get()

        state["fp"] = "b"
        holder.get()
        holder.invalidate()

        assert factory.calls == 2

    def test_a_failing_close_is_logged_and_never_propagates(self, caplog):
        state = {"fp": "a"}
        factory = ClosableFactory(fail=True)
        holder = LazySampler(factory, factory, fingerprint=lambda: state["fp"])
        first = holder.get()
        caplog.set_level(logging.WARNING, logger="annealbridge.solvers.ocean")

        state["fp"] = "b"
        second = holder.get()  # must not raise
        holder.invalidate()  # must not raise either

        assert second is not first
        assert first.closed == 1 and second.closed == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        for record in warnings:
            message = record.getMessage()
            assert "ClosableSampler" in message
            assert "RuntimeError" in message
            # The exception text itself is not logged: it could quote a
            # credential the vendor echoed back.
            assert "client shutdown failed" not in message

    def test_a_failed_rebuild_still_closes_the_sampler_it_evicted(self):
        """The old sampler has already left the cache, so it must be closed
        even though the replacement never arrived."""
        first = ClosableSampler()
        state = {"fp": "a", "boom": False}

        def factory():
            if state["boom"]:
                raise ValueError("bad region")
            return first

        holder = LazySampler(factory, factory, fingerprint=lambda: state["fp"])

        assert holder.get() is first

        state["fp"] = "b"
        state["boom"] = True
        with pytest.raises(ValueError):
            holder.get()

        assert first.closed == 1


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

    def test_an_auth_failure_closes_the_sampler_it_drops(self):
        """F-07 on this path too: the known-bad client is released, not just
        forgotten."""
        factory = ClosableFactory()
        holder = LazySampler(factory, factory, fingerprint=lambda: "fixed")
        sampler = holder.get()

        def boom():
            raise SolverAuthenticationError("denied")

        with pytest.raises(SolverExecutionError):
            call_ocean("solve", HYBRID_SAMPLE_EXCEPTION_CODES, boom, holder=holder)

        assert sampler.closed == 1

    def test_an_auth_failure_without_a_holder_still_raises(self):
        """``create_sampler`` passes no holder: there is nothing to drop."""

        def boom():
            raise SolverAuthenticationError("denied")

        with pytest.raises(SolverExecutionError) as info:
            call_ocean("creating sampler", SAMPLER_INIT_EXCEPTION_CODES, boom)

        assert info.value.code == "REMOTE_AUTH_FAILED"
