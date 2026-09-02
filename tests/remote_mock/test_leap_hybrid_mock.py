"""Mock tests for LeapHybridBQMBackend (Phase 2 spec §16, §27).

Never talks to real D-Wave: a fake sampler is injected through the
backend's ``sampler_factory`` seam and returns a real ``dimod.SampleSet``
whose ``info`` carries fake timing plus dirty keys that must not survive
sanitization.
"""

import sys

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


class FakeLeapHybridSampler:
    """Fake with the LeapHybridSampler surface the backend touches."""

    def __init__(self, raise_on_sample: Exception | None = None) -> None:
        self.raise_on_sample = raise_on_sample
        self.min_time_limit_bqm = None
        self.sample_bqm = None
        self.sample_kwargs = None

    def min_time_limit(self, bqm) -> float:
        self.min_time_limit_bqm = bqm
        return FAKE_MIN_TIME_LIMIT

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.sample_bqm = bqm
        self.sample_kwargs = kwargs
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
) -> tuple[FakeLeapHybridSampler, CompiledProblem, object]:
    fake = FakeLeapHybridSampler()
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
