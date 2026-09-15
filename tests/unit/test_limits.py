"""Unit tests for the declaration-driven limit and gate checks (Phase 3a spec §12).

``orchestration/limits.py`` holds pure functions: they read a backend's
capabilities declaration and the policy, never a backend name. The fakes
here are minimal ``SolverBackend`` implementations so the gate order and
the reported names can be pinned without any shipped backend.
"""

import pytest

from annealbridge.interfaces.capabilities import build_capabilities
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
    policy_gate_errors,
    preference_limit_errors,
    read_preference,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers import SolverRegistry
from tests.fakes.declared_backend import (
    FAKE_CREDENTIAL_ENV,
    FAKE_DECLARED_NAME,
    FakeDeclaredBackend,
)

FAKE_KEY = "fake-key-ABC123"

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
    """A ``SolverBackend`` that counts ``is_available()`` calls.

    ``raise_on_available`` makes the availability check raise instead of
    answering — a third-party backend whose credential lookup blows up
    (2026-09-09 review, addition 1).
    """

    def __init__(
        self,
        capabilities: SolverCapabilities,
        status: AvailabilityStatus = AvailabilityStatus(category="available"),
        raise_on_available: Exception | None = None,
    ) -> None:
        self._capabilities = capabilities
        self._status = status
        self.raise_on_available = raise_on_available
        self.availability_calls = 0

    @property
    def capabilities(self) -> SolverCapabilities:
        return self._capabilities

    def is_available(self) -> AvailabilityStatus:
        self.availability_calls += 1
        if self.raise_on_available is not None:
            raise self.raise_on_available
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


ALIAS_KEY = "alias_key"
OTHER_KEY = "other_key"
FAKE_CAPS_NAME = "fake_backend"

# (enabled_backends, whether the set lets ALIAS_KEY through). The registry
# key and ``capabilities.name`` differ on purpose: a gate that matched the
# set against the capabilities name would disagree with the listed answer.
ENABLED_SETS = [
    pytest.param(None, True, id="all-backends"),
    pytest.param({ALIAS_KEY}, True, id="set-with-key"),
    pytest.param({OTHER_KEY}, False, id="set-without-key"),
]
POLICY_GATE_CODES = {"BACKEND_DISABLED_BY_POLICY", "REMOTE_DISABLED"}
UNAVAILABLE = AvailabilityStatus(category="not_installed", detail="fake sdk not installed")


class TestCapabilitiesEnabledMatchesPolicyGates:
    """2026-09-15 consolidation: ``BackendCapability.enabled`` and
    ``gate_errors`` agree.

    The capabilities view reports ``enabled`` from the same two policy gates
    (``enabled_backends`` by registry key, then ``allow_remote``) that
    ``gate_errors`` checks before it ever calls ``is_available()``. Over the
    whole grid, "the view says enabled" must be exactly "``gate_errors`` did
    not refuse before availability", whatever the backend's availability.
    """

    @staticmethod
    def check(backend, enabled_backends, allow_remote):
        """Return the view's ``enabled``, the gate result and whether the gate
        consulted ``is_available()``."""
        registry = SolverRegistry({ALIAS_KEY: backend})
        policy = ExecutionPolicy(enabled_backends=enabled_backends, allow_remote=allow_remote)

        (entry,) = build_capabilities(registry, policy).backends
        assert entry.name == ALIAS_KEY

        calls_before = backend.availability_calls
        result = gate_errors(ALIAS_KEY, backend, policy)
        availability_consulted = backend.availability_calls - calls_before
        assert availability_consulted in (0, 1)

        blocked_before_availability = result is not None and availability_consulted == 0
        assert entry.enabled is (not blocked_before_availability)
        if blocked_before_availability:
            (error,) = result[2]
            assert error.code in POLICY_GATE_CODES
        return entry, result, availability_consulted

    @pytest.mark.parametrize("allow_remote", [True, False], ids=["remote-allowed", "remote-refused"])
    @pytest.mark.parametrize("remote", [True, False], ids=["remote", "local"])
    @pytest.mark.parametrize(("enabled_backends", "key_listed"), ENABLED_SETS)
    def test_available_backend(self, enabled_backends, key_listed, remote, allow_remote):
        backend = SpyBackend(make_capabilities(name=FAKE_CAPS_NAME, remote=remote))

        entry, result, consulted = self.check(backend, enabled_backends, allow_remote)

        assert entry.available is True
        assert entry.enabled is (key_listed and (not remote or allow_remote))
        if entry.enabled:
            assert result is None
            assert consulted == 1

    @pytest.mark.parametrize("allow_remote", [True, False], ids=["remote-allowed", "remote-refused"])
    @pytest.mark.parametrize("remote", [True, False], ids=["remote", "local"])
    @pytest.mark.parametrize(("enabled_backends", "key_listed"), ENABLED_SETS)
    def test_unavailable_backend(self, enabled_backends, key_listed, remote, allow_remote):
        backend = SpyBackend(make_capabilities(name=FAKE_CAPS_NAME, remote=remote), UNAVAILABLE)
        twin = SpyBackend(make_capabilities(name=FAKE_CAPS_NAME, remote=remote))

        entry, result, consulted = self.check(backend, enabled_backends, allow_remote)
        twin_entry, _twin_result, _ = self.check(twin, enabled_backends, allow_remote)

        # Availability never feeds ``enabled``.
        assert entry.available is False
        assert entry.enabled is twin_entry.enabled
        assert entry.enabled is (key_listed and (not remote or allow_remote))
        if entry.enabled:
            # Policy let it through, so the refusal is the availability one.
            assert result is not None
            status, name, errors = result
            assert consulted == 1
            assert (status, errors[0].code) == AVAILABILITY_MAP["not_installed"]
            assert name == FAKE_CAPS_NAME
            assert "fake sdk not installed" in errors[0].message


class TestPolicyGateErrors:
    """2026-09-15 consolidation: ``policy_gate_errors`` is exactly the policy
    half of ``gate_errors`` — the same refusal, message included, and never a
    call to ``is_available()``."""

    @pytest.mark.parametrize("availability", [None, UNAVAILABLE], ids=["available", "unavailable"])
    @pytest.mark.parametrize("allow_remote", [True, False], ids=["remote-allowed", "remote-refused"])
    @pytest.mark.parametrize("remote", [True, False], ids=["remote", "local"])
    @pytest.mark.parametrize(("enabled_backends", "key_listed"), ENABLED_SETS)
    def test_matches_gate_errors_before_availability(
        self, enabled_backends, key_listed, remote, allow_remote, availability
    ):
        capabilities = make_capabilities(name=FAKE_CAPS_NAME, remote=remote)
        backend = (
            SpyBackend(capabilities)
            if availability is None
            else SpyBackend(capabilities, availability)
        )
        policy = ExecutionPolicy(enabled_backends=enabled_backends, allow_remote=allow_remote)

        refused = policy_gate_errors(ALIAS_KEY, capabilities, policy)
        assert backend.availability_calls == 0

        gate = gate_errors(ALIAS_KEY, backend, policy)

        assert (refused is None) is (key_listed and (not remote or allow_remote))
        if refused is None:
            # Policy permits it; only then does gate_errors ask for availability.
            assert backend.availability_calls == 1
            if availability is None:
                assert gate is None
        else:
            assert backend.availability_calls == 0
            assert gate == refused

    def test_reports_the_registry_key_then_the_capabilities_name(self):
        capabilities = make_capabilities(name=FAKE_CAPS_NAME, remote=True)

        disabled = policy_gate_errors(
            ALIAS_KEY, capabilities, ExecutionPolicy(enabled_backends={OTHER_KEY})
        )
        remote = policy_gate_errors(ALIAS_KEY, capabilities, ExecutionPolicy())

        assert disabled is not None and remote is not None
        assert disabled[1] == ALIAS_KEY
        assert [error.code for error in disabled[2]] == ["BACKEND_DISABLED_BY_POLICY"]
        assert remote[1] == FAKE_CAPS_NAME
        assert [error.code for error in remote[2]] == ["REMOTE_DISABLED"]


class TestAvailabilityCheckFailures:
    """2026-09-09 review (additions 1 and 4): ``gate_errors`` is the one
    place ``is_available()`` is called, so it is where a backend that raises
    — or reports a category the map has never heard of — is turned into a
    structured refusal instead of an exception escaping to the caller."""

    def test_an_exception_becomes_a_backend_unavailable_error(self):
        backend = SpyBackend(
            make_capabilities(), raise_on_available=RuntimeError("boom")
        )

        result = gate_errors("fake_remote", backend, ExecutionPolicy(allow_remote=True))

        assert result is not None
        status, name, errors = result
        assert status == "backend_unavailable"
        assert name == backend.capabilities.name
        assert [error.code for error in errors] == ["BACKEND_UNAVAILABLE"]
        message = errors[0].message
        assert "RuntimeError" in message
        assert "availability check failed" in message
        assert "boom" in message
        assert backend.availability_calls == 1

    def test_the_message_is_redacted(self, monkeypatch):
        # Registering the fifth backend is what teaches the shared redaction
        # about its declared env var (review F-10), exactly as in
        # ``test_service_fallback.py::test_non_optimizer_error_is_a_redacted_solver_error``.
        monkeypatch.setenv(FAKE_CREDENTIAL_ENV, FAKE_KEY)
        SolverRegistry({FAKE_DECLARED_NAME: FakeDeclaredBackend()})
        backend = SpyBackend(
            make_capabilities(),
            raise_on_available=RuntimeError(f"vendor sdk blew up with {FAKE_KEY}"),
        )

        result = gate_errors("fake_remote", backend, ExecutionPolicy(allow_remote=True))

        assert result is not None
        message = result[2][0].message
        assert FAKE_KEY not in message
        assert "***" in message
        # The class name is categorical and survives redaction.
        assert "RuntimeError" in message

    def test_an_unknown_category_falls_back_to_the_same_pair(self):
        # ``AvailabilityCategory`` is a Literal, so an unknown value is only
        # reachable through ``model_construct``; it must still be a
        # structured refusal that names the offending category.
        backend = SpyBackend(
            make_capabilities(),
            AvailabilityStatus.model_construct(
                category="weird", detail=None, error_code=None
            ),
        )

        result = gate_errors("fake_remote", backend, ExecutionPolicy(allow_remote=True))

        assert result is not None
        status, _name, errors = result
        assert status == "backend_unavailable"
        assert [error.code for error in errors] == ["BACKEND_UNAVAILABLE"]
        assert "weird" in errors[0].message


class TestNonNumericPreferencePaths:
    """2026-09-09 review (addition 2 / F-26e): a declaration may only point
    at a numeric leaf. A ``Literal`` field, a whole option block or a bool
    would each read as "no limit" and silently disable the ceiling."""

    @pytest.mark.parametrize(
        "path", ["backend", "dwave_qpu", "dwave_qpu.auto_scale"]
    )
    def test_a_non_numeric_leaf_raises(self, path):
        with pytest.raises(ValueError, match="not a numeric preference"):
            read_preference(SolverPreferences(), path)

    def test_numeric_leaves_still_read_normally(self):
        preferences = SolverPreferences()

        assert read_preference(preferences, "seed") is None
        assert read_preference(preferences, "num_reads") == 100
        assert read_preference(preferences, "dwave_qpu.annealing_time_us") is None
