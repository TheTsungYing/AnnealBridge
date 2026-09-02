"""Mock tests for LeapHybridBQMBackend (Phase 2 spec §16, §19, §27).

Never talks to real D-Wave: a fake sampler is injected through the
backend's ``sampler_factory`` seam and returns a real ``dimod.SampleSet``
whose ``info`` carries fake timing plus dirty keys that must not survive
sanitization.

The fake also reproduces Ocean's lazy sampleset: with ``lazy=True``,
``sample()`` returns immediately and the cloud failure only surfaces when
the sampleset is resolved.
"""

import sys
import traceback

import dimod
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    LeapHybridBQMOptions,
    OptimizationProblem,
    SolverPreferences,
)
from annealbridge.solvers import LeapHybridBQMBackend, SolverCapabilities
import annealbridge.solvers.leap_hybrid_bqm as leap_module

FAKE_MIN_TIME_LIMIT = 3.0

# Matches the DEV-[A-Za-z0-9]{20,} redaction pattern; never a real token.
FAKE_TOKEN = "DEV-FAKETOKEN1234567890abcdefghij"

# Whitelist timing keys plus dirty keys that sanitization must drop.
FAKE_SAMPLESET_INFO = {
    "run_time": 2900000,
    "charge_time": 2871000,
    "qpu_access_time": 12345,
    "problem_id": "fake-problem-id-123",
    "messages": [{"nested": "structure"}],
    "raw_blob": b"\x00\x01\x02",
    "unexpected": {"deep": ("tuple", object())},
}


# The backend classifies Ocean exceptions by class name (dwave-cloud-client
# is not installed in mock CI), so the fakes carry the real names.
class SolverAuthenticationError(Exception):
    """Fake of dwave.cloud's SolverAuthenticationError."""


class RequestTimeout(Exception):
    """Fake of dwave.cloud's RequestTimeout."""


class SolverFailureError(Exception):
    """Fake of dwave.cloud's SolverFailureError (raised while resolving)."""


class SolverNotFoundError(Exception):
    """Fake of dwave.cloud's SolverNotFoundError."""


class ConfigFileError(Exception):
    """Fake of dwave.cloud's ConfigFileError."""


class ValidationError(ValueError):
    """Fake of pydantic's ValidationError, which subclasses ValueError."""


class FakeLeapHybridSampler:
    """Fake with the LeapHybridSampler surface the backend touches.

    ``lazy=True`` reproduces Ocean's real shape: ``sample()`` returns
    immediately with a ``SampleSet.from_future`` whose hook only runs (and
    only fails) when the sampleset is resolved.
    """

    def __init__(
        self,
        raise_on_sample: Exception | None = None,
        raise_on_min_time_limit: Exception | None = None,
        lazy: bool = False,
    ) -> None:
        self.raise_on_sample = raise_on_sample
        self.raise_on_min_time_limit = raise_on_min_time_limit
        self.lazy = lazy
        self.min_time_limit_bqm = None
        self.min_time_limit_calls = 0
        self.sample_bqm = None
        self.sample_kwargs = None
        self.sample_calls = 0

    def min_time_limit(self, bqm) -> float:
        self.min_time_limit_bqm = bqm
        self.min_time_limit_calls += 1
        if self.raise_on_min_time_limit is not None:
            raise self.raise_on_min_time_limit
        return FAKE_MIN_TIME_LIMIT

    def _build_sampleset(self, bqm) -> dimod.SampleSet:
        if self.raise_on_sample is not None:
            raise self.raise_on_sample
        # Hybrid solvers typically return exactly one sample.
        assignment = {variable: 0 for variable in bqm.variables}
        return dimod.SampleSet.from_samples(
            assignment,
            vartype=dimod.BINARY,
            energy=bqm.energy(assignment),
            info=dict(FAKE_SAMPLESET_INFO),
        )

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.sample_bqm = bqm
        self.sample_kwargs = kwargs
        self.sample_calls += 1
        if self.lazy:
            return dimod.SampleSet.from_future(
                object(), lambda _future: self._build_sampleset(bqm)
            )
        return self._build_sampleset(bqm)


class CountingFactory:
    """``sampler_factory`` seam that counts calls and can fail the first one."""

    def __init__(self, sampler=None, fail_first: Exception | None = None) -> None:
        self.sampler = sampler if sampler is not None else FakeLeapHybridSampler()
        self.fail_first = fail_first
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.fail_first is not None and self.calls == 1:
            raise self.fail_first
        return self.sampler


def make_compiled_problem() -> CompiledProblem:
    """Minimal 0/1 problem: maximize 2a + b s.t. a + b <= 1 (adds slack)."""
    problem = OptimizationProblem.model_validate(
        {
            "name": "leap hybrid mock problem",
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
            "solver": {"backend": "leap_hybrid_bqm"},
        }
    )
    return BQMCompiler().compile(problem, hard_penalty=100.0)


def make_preferences(time_limit_seconds: float | None = None) -> SolverPreferences:
    return SolverPreferences(
        backend="leap_hybrid_bqm",
        leap_hybrid_bqm=LeapHybridBQMOptions(time_limit_seconds=time_limit_seconds),
    )


def solve_with_fake(
    preferences: SolverPreferences,
    fake: FakeLeapHybridSampler | None = None,
) -> tuple[FakeLeapHybridSampler, CompiledProblem, object]:
    fake = fake if fake is not None else FakeLeapHybridSampler()
    backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)
    compiled = make_compiled_problem()
    result = backend.solve(compiled, preferences)
    return fake, compiled, result


class TestCapabilities:
    def test_capability_fields(self):
        capabilities = LeapHybridBQMBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "leap_hybrid_bqm"
        assert capabilities.remote is True
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is False
        assert capabilities.supports_num_reads is False
        assert capabilities.supports_time_limit is True
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is False

    def test_description_explains_single_sample(self):
        description = LeapHybridBQMBackend().capabilities.description
        assert "single sample" in description

    def test_properties_alias_capabilities(self):
        backend = LeapHybridBQMBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive


class TestLazyImport:
    def test_construction_does_not_import_dwave(self):
        LeapHybridBQMBackend()
        assert "dwave.system" not in sys.modules


class TestTimeLimitForwarding:
    def test_no_user_time_limit_uses_sampler_minimum(self):
        fake, compiled, result = solve_with_fake(make_preferences(None))

        assert fake.sample_kwargs == {"time_limit": FAKE_MIN_TIME_LIMIT}
        assert fake.min_time_limit_bqm is compiled.model
        assert result.metadata.effective_time_limit_seconds == FAKE_MIN_TIME_LIMIT

    def test_options_object_missing_entirely_uses_sampler_minimum(self):
        preferences = SolverPreferences(backend="leap_hybrid_bqm")
        assert preferences.leap_hybrid_bqm is None

        fake, _, result = solve_with_fake(preferences)

        assert fake.sample_kwargs == {"time_limit": FAKE_MIN_TIME_LIMIT}
        assert result.metadata.effective_time_limit_seconds == FAKE_MIN_TIME_LIMIT

    def test_user_time_limit_above_minimum_is_forwarded_verbatim(self):
        fake, _, result = solve_with_fake(make_preferences(10.0))

        assert fake.sample_kwargs == {"time_limit": 10.0}
        assert result.metadata.effective_time_limit_seconds == 10.0

    def test_user_time_limit_below_minimum_is_raised_to_minimum(self):
        fake, _, result = solve_with_fake(make_preferences(1.0))

        assert fake.sample_kwargs == {"time_limit": FAKE_MIN_TIME_LIMIT}
        assert result.metadata.effective_time_limit_seconds == FAKE_MIN_TIME_LIMIT

    def test_bqm_is_forwarded_unchanged(self):
        fake, compiled, _ = solve_with_fake(make_preferences(None))

        assert fake.sample_bqm is compiled.model

    def test_hybrid_meaningless_parameters_are_not_forwarded(self):
        preferences = SolverPreferences(
            backend="leap_hybrid_bqm",
            num_reads=7,
            num_sweeps=50,
            seed=42,
            leap_hybrid_bqm=LeapHybridBQMOptions(time_limit_seconds=5.0),
        )

        fake, _, _ = solve_with_fake(preferences)

        assert set(fake.sample_kwargs) == {"time_limit"}


class TestSampleSetConversion:
    def test_samples_and_energies_match_the_sampleset(self):
        fake, compiled, result = solve_with_fake(make_preferences(None))

        expected_sample = {str(v): 0 for v in compiled.model.variables}
        assert result.samples == [expected_sample]
        assert result.energies == [
            pytest.approx(compiled.model.energy({v: 0 for v in compiled.model.variables}))
        ]
        assert result.backend == "leap_hybrid_bqm"

    def test_samples_include_internal_slack_variables(self):
        _, compiled, result = solve_with_fake(make_preferences(None))

        assert compiled.internal_variables
        assert compiled.internal_variables <= set(result.samples[0])


class TestMetadataSanitization:
    def test_only_whitelisted_timing_keys_survive(self):
        _, _, result = solve_with_fake(make_preferences(None))

        assert result.metadata.timing_us == {
            "run_time": 2900000.0,
            "charge_time": 2871000.0,
            "qpu_access_time": 12345.0,
        }

    def test_dirty_info_keys_do_not_leak_anywhere(self):
        _, _, result = solve_with_fake(make_preferences(None))

        dumped = result.metadata.model_dump_json()
        assert "problem_id" not in dumped
        assert "fake-problem-id-123" not in dumped
        assert "nested" not in dumped
        assert "unexpected" not in dumped

    def test_backend_and_remote_flags(self):
        _, _, result = solve_with_fake(make_preferences(None))

        assert result.metadata.backend == "leap_hybrid_bqm"
        assert result.metadata.remote is True


class TestExceptionClassification:
    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (SolverAuthenticationError(f"invalid token={FAKE_TOKEN}"), "REMOTE_AUTH_FAILED"),
            (RequestTimeout(f"timed out, token={FAKE_TOKEN}"), "REMOTE_TIMEOUT"),
            (RuntimeError(f"solver exploded, token={FAKE_TOKEN}"), "REMOTE_SOLVER_ERROR"),
        ],
        ids=["auth", "timeout", "other"],
    )
    def test_sample_exceptions_map_to_codes_and_are_redacted(
        self, exception, expected_code
    ):
        fake = FakeLeapHybridSampler(raise_on_sample=exception)
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message

    def test_subclass_of_auth_error_is_classified_via_mro(self):
        class DerivedAuthError(SolverAuthenticationError):
            pass

        fake = FakeLeapHybridSampler(raise_on_sample=DerivedAuthError("denied"))
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "REMOTE_AUTH_FAILED"

    def test_factory_failure_is_classified_and_redacted(self):
        def failing_factory():
            raise SolverAuthenticationError(f"Authorization: Bearer {FAKE_TOKEN}")

        backend = LeapHybridBQMBackend(sampler_factory=failing_factory)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "REMOTE_AUTH_FAILED"
        assert FAKE_TOKEN not in str(exc_info.value)


class TestSamplerInitExceptionClassification:
    """Spec §16: sampler construction has its own classification table.

    ``LeapHybridSampler()`` runs Client.from_config() / get_solver(), so a
    ValueError there is a *configuration* problem, not a solve failure.
    """

    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (ValueError(f"invalid region, token={FAKE_TOKEN}"), "DWAVE_CONFIG_INVALID"),
            (
                ValidationError(f"1 validation error, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
            ),
            (
                SolverNotFoundError(f"no solver matches, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
            ),
            (
                ConfigFileError(f"bad dwave.conf, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
            ),
            (
                SolverAuthenticationError(f"invalid token={FAKE_TOKEN}"),
                "REMOTE_AUTH_FAILED",
            ),
            (RequestTimeout(f"timed out, token={FAKE_TOKEN}"), "REMOTE_TIMEOUT"),
            (
                RuntimeError(f"client exploded, token={FAKE_TOKEN}"),
                "REMOTE_SOLVER_ERROR",
            ),
        ],
        ids=[
            "value_error",
            "validation_error",
            "solver_not_found",
            "config_file_error",
            "auth",
            "timeout",
            "other",
        ],
    )
    def test_factory_exceptions_map_to_codes_and_are_redacted(
        self, exception, expected_code
    ):
        backend = LeapHybridBQMBackend(
            sampler_factory=CountingFactory(fail_first=exception)
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message

    @pytest.mark.parametrize(
        "exception",
        [
            ValueError(f"invalid endpoint, token={FAKE_TOKEN}"),
            ValidationError(f"1 validation error, token={FAKE_TOKEN}"),
        ],
        ids=["value_error", "validation_error"],
    )
    def test_factory_value_error_is_config_invalid_not_embedding_failed(
        self, exception
    ):
        backend = LeapHybridBQMBackend(
            sampler_factory=CountingFactory(fail_first=exception)
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "DWAVE_CONFIG_INVALID"
        assert exc_info.value.code != "EMBEDDING_FAILED"
        assert FAKE_TOKEN not in str(exc_info.value)
        assert "***" in str(exc_info.value)


class TestLazySampleSetResolution:
    """Spec §16/§19: the backend resolves the sampleset inside the guarded
    call, so a failure that only surfaces on resolve is still classified and
    redacted."""

    def test_lazy_success_matches_the_eager_result(self):
        _, _, expected = solve_with_fake(make_preferences(None))
        lazy, _, result = solve_with_fake(
            make_preferences(None), fake=FakeLeapHybridSampler(lazy=True)
        )

        assert lazy.sample_calls == 1
        assert result.samples == expected.samples
        assert result.energies == expected.energies
        assert result.backend == expected.backend
        assert result.metadata.model_dump() == expected.metadata.model_dump()

    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (RequestTimeout(f"timed out, token={FAKE_TOKEN}"), "REMOTE_TIMEOUT"),
            (
                SolverFailureError(f"solver failed, token={FAKE_TOKEN}"),
                "REMOTE_SOLVER_ERROR",
            ),
            (RuntimeError(f"boom, token={FAKE_TOKEN}"), "REMOTE_SOLVER_ERROR"),
        ],
        ids=["timeout", "solver_failure", "other"],
    )
    def test_failure_on_resolve_is_classified_and_redacted(
        self, exception, expected_code
    ):
        fake = FakeLeapHybridSampler(lazy=True, raise_on_sample=exception)
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message
        # sample() itself returned normally: the failure happened on resolve.
        assert fake.sample_calls == 1
        assert fake.sample_kwargs == {"time_limit": FAKE_MIN_TIME_LIMIT}


class TestOriginalExceptionIsNotReachable:
    """Spec §19: the wrapped error carries no chain back to the original,
    so a traceback dump cannot print credential-bearing text."""

    def test_sample_failure_has_no_cause_or_context(self):
        fake = FakeLeapHybridSampler(
            raise_on_sample=RuntimeError(f"solver exploded, token={FAKE_TOKEN}")
        )
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        formatted = "".join(traceback.format_exception(error))
        assert FAKE_TOKEN not in formatted

    def test_factory_failure_has_no_cause_or_context(self):
        backend = LeapHybridBQMBackend(
            sampler_factory=CountingFactory(
                fail_first=SolverNotFoundError(f"Authorization: Bearer {FAKE_TOKEN}")
            )
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        formatted = "".join(traceback.format_exception(error))
        assert FAKE_TOKEN not in formatted


class TestSamplerCaching:
    """Spec §16: LeapHybridSampler() fetches solver metadata, so it is built
    once per backend instance — and a failed construction is never cached."""

    def test_sampler_is_built_once_per_backend(self):
        factory = CountingFactory()
        backend = LeapHybridBQMBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        backend.solve(compiled, make_preferences(None))
        backend.solve(compiled, make_preferences(None))

        assert factory.calls == 1
        assert factory.sampler.sample_calls == 2

    def test_failed_construction_is_not_cached(self):
        factory = CountingFactory(
            fail_first=SolverAuthenticationError(f"denied, token={FAKE_TOKEN}")
        )
        backend = LeapHybridBQMBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compiled, make_preferences(None))
        assert exc_info.value.code == "REMOTE_AUTH_FAILED"

        result = backend.solve(compiled, make_preferences(None))

        assert factory.calls == 2
        assert result.backend == "leap_hybrid_bqm"
        assert factory.sampler.sample_calls == 1

    def test_each_backend_instance_builds_its_own_sampler(self):
        factory = CountingFactory()
        compiled = make_compiled_problem()

        LeapHybridBQMBackend(sampler_factory=factory).solve(
            compiled, make_preferences(None)
        )
        LeapHybridBQMBackend(sampler_factory=factory).solve(
            compiled, make_preferences(None)
        )

        assert factory.calls == 2


class TestResolveTimeLimit:
    """Spec §16/§20: the service must be able to learn the effective time
    limit *before* anything is submitted."""

    @pytest.mark.parametrize(
        ("user_time_limit", "expected"),
        [
            (None, FAKE_MIN_TIME_LIMIT),
            (10.0, 10.0),
            (1.0, FAKE_MIN_TIME_LIMIT),
        ],
        ids=["no-preference", "above-minimum", "below-minimum"],
    )
    def test_effective_time_limit_without_submitting(self, user_time_limit, expected):
        fake = FakeLeapHybridSampler()
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)
        compiled = make_compiled_problem()

        resolved = backend.resolve_time_limit(compiled, make_preferences(user_time_limit))

        assert resolved == expected
        assert fake.sample_calls == 0
        assert fake.min_time_limit_bqm is compiled.model

    def test_options_object_missing_entirely_uses_sampler_minimum(self):
        fake = FakeLeapHybridSampler()
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)
        preferences = SolverPreferences(backend="leap_hybrid_bqm")
        assert preferences.leap_hybrid_bqm is None

        assert (
            backend.resolve_time_limit(make_compiled_problem(), preferences)
            == FAKE_MIN_TIME_LIMIT
        )
        assert fake.sample_calls == 0

    @pytest.mark.parametrize(
        "user_time_limit", [None, 10.0, 1.0], ids=["none", "above", "below"]
    )
    def test_solve_submits_exactly_the_resolved_time_limit(self, user_time_limit):
        fake = FakeLeapHybridSampler()
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)
        compiled = make_compiled_problem()
        preferences = make_preferences(user_time_limit)

        result = backend.solve(compiled, preferences)

        expected = backend.resolve_time_limit(compiled, preferences)
        assert result.metadata.effective_time_limit_seconds == expected
        assert fake.sample_kwargs == {"time_limit": expected}

    def test_min_time_limit_failure_is_classified_and_redacted(self):
        fake = FakeLeapHybridSampler(
            raise_on_min_time_limit=RuntimeError(f"exploded, token={FAKE_TOKEN}")
        )
        backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.resolve_time_limit(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message
        assert fake.sample_calls == 0

    def test_factory_failure_during_resolve_is_config_invalid(self):
        backend = LeapHybridBQMBackend(
            sampler_factory=CountingFactory(
                fail_first=ValueError(f"invalid region, token={FAKE_TOKEN}")
            )
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.resolve_time_limit(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "DWAVE_CONFIG_INVALID"
        assert FAKE_TOKEN not in str(exc_info.value)
        assert "***" in str(exc_info.value)


class TestIsAvailable:
    def test_dwave_system_not_installed(self, monkeypatch):
        monkeypatch.setattr(leap_module, "_dwave_system_installed", lambda: False)

        assert LeapHybridBQMBackend().is_available() == (
            False,
            "dwave-system not installed",
        )

    def test_credentials_not_configured(self, monkeypatch):
        monkeypatch.setattr(leap_module, "_dwave_system_installed", lambda: True)
        monkeypatch.setattr(leap_module, "ocean_config_status", lambda: "missing")

        assert LeapHybridBQMBackend().is_available() == (
            False,
            "D-Wave credentials not configured",
        )

    def test_configuration_invalid(self, monkeypatch):
        monkeypatch.setattr(leap_module, "_dwave_system_installed", lambda: True)
        monkeypatch.setattr(leap_module, "ocean_config_status", lambda: "invalid")

        assert LeapHybridBQMBackend().is_available() == (
            False,
            "D-Wave configuration invalid",
        )

    def test_available_when_installed_and_configured(self, monkeypatch):
        monkeypatch.setattr(leap_module, "_dwave_system_installed", lambda: True)
        monkeypatch.setattr(leap_module, "ocean_config_status", lambda: "ok")

        assert LeapHybridBQMBackend().is_available() == (True, None)
