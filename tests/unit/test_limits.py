"""Unit tests for the declaration-driven limit and gate checks (Phase 3a spec §12).

``orchestration/limits.py`` holds pure functions: they read a backend's
capabilities declaration and the policy, never a backend name. The fakes
here are minimal ``SolverBackend`` implementations so the gate order and
the reported names can be pinned without any shipped backend.
"""

import pytest

from annealbridge.models import (
    AvailabilityStatus,
    DWaveQPUOptions,
    LeapHybridBQMOptions,
    ParameterLimit,
    SolverCapabilities,
    SolverPreferences,
)
from annealbridge.orchestration.limits import (
    AVAILABILITY_MAP,
    gate_errors,
    preference_limit_errors,
    read_preference,
)
from annealbridge.orchestration.policy import ExecutionPolicy

QPU_LIMITS = [
    ParameterLimit(preference="num_reads", limit="reads", error_code="QPU_READS_LIMIT"),
    ParameterLimit(
        preference="dwave_qpu.annealing_time_us",
        limit="annealing_time_us",
        error_code="QPU_ANNEALING_TIME_LIMIT",
    ),
]
HYBRID_LIMITS = [
    ParameterLimit(
        preference="leap_hybrid_bqm.time_limit_seconds",
        limit="time_seconds",
        error_code="REMOTE_TIME_LIMIT",
    )
]


def make_capabilities(**overrides) -> SolverCapabilities:
    fields = dict(
        name="fake_remote",
        remote=True,
        heuristic=True,
        exhaustive=False,
        supports_seed=False,
        supports_num_reads=True,
        supports_time_limit=False,
        supported_model_types=["bqm"],
        returns_multiple_samples=True,
        description="Fake backend for the limits module tests.",
    )
    fields.update(overrides)
    return SolverCapabilities(**fields)


class SpyBackend:
    """A ``SolverBackend`` that counts ``is_available()`` calls."""

    def __init__(
        self,
        capabilities: SolverCapabilities,
        status: AvailabilityStatus = AvailabilityStatus(category="available"),
    ) -> None:
        self._capabilities = capabilities
        self._status = status
        self.availability_calls = 0

    @property
    def capabilities(self) -> SolverCapabilities:
        return self._capabilities

    def is_available(self) -> AvailabilityStatus:
        self.availability_calls += 1
        return self._status

    @property
    def name(self) -> str:
        return self._capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return self._capabilities.exhaustive

    def resolve_time_limit(self, compiled_problem, preferences):
        return None

    def solve(self, compiled_problem, preferences):
        raise AssertionError("solve() must not be reached by a gate test")


class TestReadPreference:
    def test_top_level_field(self):
        assert read_preference(SolverPreferences(num_reads=250), "num_reads") == 250

    def test_dotted_path_into_a_filled_option_block(self):
        preferences = SolverPreferences(
            dwave_qpu=DWaveQPUOptions(annealing_time_us=20.0)
        )

        assert read_preference(preferences, "dwave_qpu.annealing_time_us") == 20.0

    def test_unset_option_block_reads_as_none(self):
        assert read_preference(SolverPreferences(), "dwave_qpu.annealing_time_us") is None
        assert (
            read_preference(SolverPreferences(), "leap_hybrid_bqm.time_limit_seconds")
            is None
        )

    def test_unset_leaf_inside_a_filled_block_reads_as_none(self):
        preferences = SolverPreferences(leap_hybrid_bqm=LeapHybridBQMOptions())

        assert (
            read_preference(preferences, "leap_hybrid_bqm.time_limit_seconds") is None
        )

    def test_unknown_top_level_field_raises(self):
        with pytest.raises(ValueError, match="'no_such_field' does not exist"):
            read_preference(SolverPreferences(), "no_such_field")

    def test_unknown_nested_field_raises_even_when_the_block_is_unset(self):
        # A wrong declaration must fail loudly, not read as "no limit".
        with pytest.raises(ValueError, match="'dwave_qpu.no_such_field' does not exist"):
            read_preference(SolverPreferences(), "dwave_qpu.no_such_field")

    def test_unknown_nested_field_raises_when_the_block_is_filled(self):
        preferences = SolverPreferences(dwave_qpu=DWaveQPUOptions())

        with pytest.raises(ValueError, match="does not exist"):
            read_preference(preferences, "dwave_qpu.no_such_field")

    def test_path_through_a_scalar_field_raises(self):
        with pytest.raises(ValueError, match="not an option block"):
            read_preference(SolverPreferences(), "num_reads.deeper")


class TestPreferenceLimitErrors:
    def test_no_declarations_means_no_errors(self):
        errors = preference_limit_errors(
            make_capabilities(), SolverPreferences(num_reads=10**6), ExecutionPolicy()
        )

        assert errors == []

    def test_within_the_limits_is_clean(self):
        policy = ExecutionPolicy(max_qpu_reads=100, max_qpu_annealing_time_us=50.0)
        preferences = SolverPreferences(
            num_reads=100, dwave_qpu=DWaveQPUOptions(annealing_time_us=50.0)
        )

        # Exactly at the limit is allowed; only *over* is refused.
        assert (
            preference_limit_errors(
                make_capabilities(parameter_limits=QPU_LIMITS), preferences, policy
            )
            == []
        )

    def test_unset_option_block_is_not_a_violation(self):
        policy = ExecutionPolicy(max_qpu_annealing_time_us=1.0)

        errors = preference_limit_errors(
            make_capabilities(parameter_limits=QPU_LIMITS), SolverPreferences(), policy
        )

        assert errors == []

    def test_every_violation_is_collected(self):
        policy = ExecutionPolicy(max_qpu_reads=10, max_qpu_annealing_time_us=100.0)
        preferences = SolverPreferences(
            num_reads=100, dwave_qpu=DWaveQPUOptions(annealing_time_us=500.0)
        )

        errors = preference_limit_errors(
            make_capabilities(parameter_limits=QPU_LIMITS), preferences, policy
        )

        assert [error.code for error in errors] == [
            "QPU_READS_LIMIT",
            "QPU_ANNEALING_TIME_LIMIT",
        ]

    def test_message_uses_the_dotted_path_and_both_numbers(self):
        policy = ExecutionPolicy(max_qpu_annealing_time_us=100.0)
        preferences = SolverPreferences(dwave_qpu=DWaveQPUOptions(annealing_time_us=500.0))

        [error] = preference_limit_errors(
            make_capabilities(parameter_limits=QPU_LIMITS), preferences, policy
        )

        assert error.message == (
            "dwave_qpu.annealing_time_us 500.0 exceeds the server maximum of 100.0"
        )
        assert error.recommended_action

    def test_hybrid_time_limit_declaration(self):
        policy = ExecutionPolicy(max_remote_time_seconds=10)
        preferences = SolverPreferences(
            leap_hybrid_bqm=LeapHybridBQMOptions(time_limit_seconds=60.0)
        )

        [error] = preference_limit_errors(
            make_capabilities(parameter_limits=HYBRID_LIMITS), preferences, policy
        )

        assert error.code == "REMOTE_TIME_LIMIT"
        assert "60.0" in error.message
        assert "maximum of 10" in error.message

    def test_custom_limit_key_is_read_from_policy_limits(self):
        declaration = ParameterLimit(
            preference="num_sweeps", limit="iterations", error_code="QPU_READS_LIMIT"
        )
        policy = ExecutionPolicy(limits={"iterations": 500})

        errors = preference_limit_errors(
            make_capabilities(parameter_limits=[declaration]),
            SolverPreferences(num_sweeps=1000),
            policy,
        )

        assert [error.code for error in errors] == ["QPU_READS_LIMIT"]
        assert "num_sweeps 1000 exceeds the server maximum of 500.0" in errors[0].message

    def test_policy_without_the_declared_key_raises(self):
        declaration = ParameterLimit(
            preference="num_sweeps", limit="iterations", error_code="QPU_READS_LIMIT"
        )

        with pytest.raises(ValueError) as exc_info:
            preference_limit_errors(
                make_capabilities(name="custom", parameter_limits=[declaration]),
                SolverPreferences(),
                ExecutionPolicy(),
            )

        assert "custom" in str(exc_info.value)
        assert "iterations" in str(exc_info.value)

    def test_never_clamps(self):
        policy = ExecutionPolicy(max_qpu_reads=10)
        preferences = SolverPreferences(num_reads=100)

        preference_limit_errors(
            make_capabilities(parameter_limits=QPU_LIMITS), preferences, policy
        )

        assert preferences.num_reads == 100


class TestGateOrder:
    """§16.2 steps 3–5 short-circuit in order; the first two never touch
    ``is_available()`` (the D-Wave one reads a config file)."""

    def test_disabled_by_policy_fires_first_without_availability(self):
        backend = SpyBackend(make_capabilities())
        policy = ExecutionPolicy(allow_remote=False, enabled_backends={"exact"})

        result = gate_errors("fake_remote", backend, policy)

        assert result is not None
        status, name, errors = result
        assert status == "backend_unavailable"
        assert [error.code for error in errors] == ["BACKEND_DISABLED_BY_POLICY"]
        assert backend.availability_calls == 0

    def test_remote_disabled_fires_second_without_availability(self):
        backend = SpyBackend(
            make_capabilities(), AvailabilityStatus(category="not_installed")
        )
        policy = ExecutionPolicy(allow_remote=False)

        result = gate_errors("fake_remote", backend, policy)

        assert result is not None
        status, name, errors = result
        assert status == "backend_unavailable"
        assert [error.code for error in errors] == ["REMOTE_DISABLED"]
        assert backend.availability_calls == 0

    def test_availability_is_checked_last(self):
        backend = SpyBackend(
            make_capabilities(),
            AvailabilityStatus(category="not_installed", detail="dwave-system not installed"),
        )
        policy = ExecutionPolicy(allow_remote=True)

        result = gate_errors("fake_remote", backend, policy)

        assert result is not None
        status, name, errors = result
        assert status == "backend_unavailable"
        assert [error.code for error in errors] == ["BACKEND_NOT_INSTALLED"]
        assert "dwave-system not installed" in errors[0].message
        assert backend.availability_calls == 1

    def test_available_backend_passes(self):
        backend = SpyBackend(make_capabilities())

        assert gate_errors("fake_remote", backend, ExecutionPolicy(allow_remote=True)) is None
        assert backend.availability_calls == 1

    def test_local_backend_ignores_allow_remote(self):
        backend = SpyBackend(make_capabilities(name="fake_local", remote=False))

        assert gate_errors("fake_local", backend, ExecutionPolicy(allow_remote=False)) is None

    def test_registry_key_in_enabled_set_passes_the_policy_gate(self):
        backend = SpyBackend(make_capabilities())
        policy = ExecutionPolicy(allow_remote=True, enabled_backends={"custom_key"})

        assert gate_errors("custom_key", backend, policy) is None

    @pytest.mark.parametrize("category", sorted(AVAILABILITY_MAP))
    def test_category_maps_to_status_and_default_code(self, category):
        backend = SpyBackend(make_capabilities(), AvailabilityStatus(category=category))

        result = gate_errors("fake_remote", backend, ExecutionPolicy(allow_remote=True))

        assert result is not None
        status, _name, errors = result
        assert (status, errors[0].code) == AVAILABILITY_MAP[category]

    def test_backend_supplied_error_code_wins_over_the_default(self):
        backend = SpyBackend(
            make_capabilities(),
            AvailabilityStatus(category="config_invalid", error_code="DWAVE_CONFIG_INVALID"),
        )

        result = gate_errors("fake_remote", backend, ExecutionPolicy(allow_remote=True))

        assert result is not None
        status, _name, errors = result
        assert status == "configuration_error"
        assert errors[0].code == "DWAVE_CONFIG_INVALID"

    def test_unreported_detail_still_produces_a_message(self):
        backend = SpyBackend(make_capabilities(), AvailabilityStatus(category="unavailable"))

        result = gate_errors("fake_remote", backend, ExecutionPolicy(allow_remote=True))

        assert result is not None
        assert "no reason reported" in result[2][0].message


class TestReportedName:
    """Disabled-by-policy reports the registry key (what the user asked
    for); the other gates report ``capabilities.name`` (Phase 2 behaviour)."""

    def test_disabled_by_policy_reports_the_registry_key(self):
        backend = SpyBackend(make_capabilities(name="fake_remote"))
        policy = ExecutionPolicy(allow_remote=True, enabled_backends={"exact"})

        result = gate_errors("custom_key", backend, policy)

        assert result is not None
        assert result[1] == "custom_key"
        assert "'custom_key'" in result[2][0].message

    def test_remote_disabled_reports_the_capabilities_name(self):
        backend = SpyBackend(make_capabilities(name="fake_remote"))

        result = gate_errors("custom_key", backend, ExecutionPolicy(allow_remote=False))

        assert result is not None
        assert result[1] == "fake_remote"
        assert "'fake_remote'" in result[2][0].message

    def test_unavailable_reports_the_capabilities_name(self):
        backend = SpyBackend(
            make_capabilities(name="fake_remote"),
            AvailabilityStatus(category="credentials_missing"),
        )

        result = gate_errors("custom_key", backend, ExecutionPolicy(allow_remote=True))

        assert result is not None
        assert result[1] == "fake_remote"
        assert "'fake_remote'" in result[2][0].message
