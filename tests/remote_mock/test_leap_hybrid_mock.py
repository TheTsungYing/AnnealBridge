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

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    LeapHybridBQMOptions,
    SolverPreferences,
)
from annealbridge.solvers import (
    AvailabilityStatus,
    LeapHybridBQMBackend,
    SolverCapabilities,
)
import annealbridge.solvers.ocean as ocean_module
from annealbridge.solvers.ocean import (
    REASON_CONFIG_INVALID,
    REASON_CREDENTIALS_MISSING,
    REASON_NOT_INSTALLED,
)
from tests.remote_mock.conftest import (
    FAKE_MIN_TIME_LIMIT,
    FAKE_TOKEN,
    ConfigFileError,
    CountingFactory,
    FakeLeapHybridSampler,
    RequestTimeout,
    SolverAuthenticationError,
    SolverFailureError,
    SolverNotFoundError,
    ValidationError,
    make_problem,
)


def make_compiled_problem() -> CompiledProblem:
    """Minimal 0/1 problem: maximize 2a + b s.t. a + b <= 1 (adds slack)."""
    return BQMCompiler().compile(
        make_problem(backend="leap_hybrid_bqm"), hard_penalty=100.0
    )


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
        assert capabilities.requires_embedding is False
        assert capabilities.supports_num_sweeps is False
        assert capabilities.remote is True
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is False
        assert capabilities.supports_num_reads is False
        assert capabilities.supports_time_limit is True
        # Spec 7.1: the hybrid time limit is a declaration, not a name check.
        assert [
            (limit.preference, limit.limit, limit.error_code)
            for limit in capabilities.parameter_limits
        ] == [("leap_hybrid_bqm.time_limit_seconds", "time_seconds", "REMOTE_TIME_LIMIT")]
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
        assert sorted(result.variables) == sorted(expected_sample)
        assert result.as_dicts() == [expected_sample]
        assert result.energies.tolist() == [
            pytest.approx(compiled.model.energy({v: 0 for v in compiled.model.variables}))
        ]
        assert result.backend == "leap_hybrid_bqm"

    def test_samples_include_internal_slack_variables(self):
        _, compiled, result = solve_with_fake(make_preferences(None))

        assert compiled.internal_variables
        assert compiled.internal_variables <= set(result.variables)


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
            sampler_factory=CountingFactory(
                FakeLeapHybridSampler(), failures=[exception]
            )
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
            sampler_factory=CountingFactory(
                FakeLeapHybridSampler(), failures=[exception]
            )
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
        assert result.variables == expected.variables
        assert result.samples.tolist() == expected.samples.tolist()
        assert result.energies.tolist() == expected.energies.tolist()
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
                FakeLeapHybridSampler(),
                failures=[SolverNotFoundError(f"Authorization: Bearer {FAKE_TOKEN}")],
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
        factory = CountingFactory(FakeLeapHybridSampler())
        backend = LeapHybridBQMBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        backend.solve(compiled, make_preferences(None))
        backend.solve(compiled, make_preferences(None))

        assert factory.calls == 1
        assert factory.sampler.sample_calls == 2

    def test_failed_construction_is_not_cached(self):
        factory = CountingFactory(
            FakeLeapHybridSampler(),
            failures=[SolverAuthenticationError(f"denied, token={FAKE_TOKEN}")],
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
        factory = CountingFactory(FakeLeapHybridSampler())
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
                FakeLeapHybridSampler(),
                failures=[ValueError(f"invalid region, token={FAKE_TOKEN}")],
            )
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.resolve_time_limit(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "DWAVE_CONFIG_INVALID"
        assert FAKE_TOKEN not in str(exc_info.value)
        assert "***" in str(exc_info.value)


class TestIsAvailable:
    """The backend answers through the shared check in solvers.ocean, so
    the installability / credential probes are patched there."""

    def test_dwave_system_not_installed(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: False)

        assert LeapHybridBQMBackend().is_available() == AvailabilityStatus(
            category="not_installed", detail=REASON_NOT_INSTALLED
        )

    def test_credentials_not_configured(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "missing")

        assert LeapHybridBQMBackend().is_available() == AvailabilityStatus(
            category="credentials_missing", detail=REASON_CREDENTIALS_MISSING
        )

    def test_configuration_invalid(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "invalid")

        assert LeapHybridBQMBackend().is_available() == AvailabilityStatus(
            category="config_invalid",
            detail=REASON_CONFIG_INVALID,
            error_code="DWAVE_CONFIG_INVALID",
        )

    def test_available_when_installed_and_configured(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "ok")

        assert LeapHybridBQMBackend().is_available() == AvailabilityStatus(
            category="available"
        )
