"""The shared D-Wave backend steps in ``solvers.ocean`` (2026-09-09 review F-13c).

``resolve_hybrid_time_limit`` used to exist twice, once per Leap hybrid
backend, word for word apart from the message label; ``create_sampler``
three times. The rule is pinned here once, with a fake sampler; the
backend-level tests in ``tests/remote_mock`` still exercise each backend
through the shared code. Nothing here imports ``dwave.system``.
"""

import pytest

from annealbridge.exceptions import SolverExecutionError
from annealbridge.solvers.ocean import (
    LazySampler,
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
