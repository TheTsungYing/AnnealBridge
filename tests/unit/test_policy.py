"""Unit tests for the execution policy defaults (Phase 2 spec §8)."""

import pytest
from pydantic import ValidationError

from annealbridge.orchestration import ExecutionPolicy as ExportedExecutionPolicy
from annealbridge.orchestration.policy import ExecutionPolicy


class TestExecutionPolicyDefaults:
    def test_allow_remote_defaults_to_false(self):
        assert ExecutionPolicy().allow_remote is False

    def test_allow_remote_retries_defaults_to_false(self):
        assert ExecutionPolicy().allow_remote_retries is False

    def test_exact_max_variables_defaults_to_24(self):
        assert ExecutionPolicy().exact_max_variables == 24

    def test_max_qpu_reads_defaults_to_1000(self):
        assert ExecutionPolicy().max_qpu_reads == 1000

    def test_max_qpu_annealing_time_us_defaults_to_2000(self):
        assert ExecutionPolicy().max_qpu_annealing_time_us == 2000.0

    def test_max_remote_time_seconds_defaults_to_300(self):
        assert ExecutionPolicy().max_remote_time_seconds == 300

    def test_max_concurrent_solves_defaults_to_4(self):
        assert ExecutionPolicy().max_concurrent_solves == 4

    def test_enabled_backends_defaults_to_none(self):
        # None means "every backend in the registry" (spec §8).
        assert ExecutionPolicy().enabled_backends is None


class TestExecutionPolicyOverrides:
    def test_fields_are_overridable(self):
        policy = ExecutionPolicy(
            allow_remote=True,
            allow_remote_retries=True,
            exact_max_variables=8,
            max_qpu_reads=10,
            max_qpu_annealing_time_us=123.5,
            max_remote_time_seconds=30,
            max_concurrent_solves=1,
            enabled_backends={"exact"},
        )

        assert policy.allow_remote is True
        assert policy.allow_remote_retries is True
        assert policy.exact_max_variables == 8
        assert policy.max_qpu_reads == 10
        assert policy.max_qpu_annealing_time_us == 123.5
        assert policy.max_remote_time_seconds == 30
        assert policy.max_concurrent_solves == 1
        assert policy.enabled_backends == {"exact"}


class TestExecutionPolicyExport:
    def test_package_export_is_the_same_class(self):
        assert ExportedExecutionPolicy is ExecutionPolicy


class TestExecutionPolicyBounds:
    """Every limit has a lower bound: a zero or negative ceiling is a config
    error, not a policy (max_concurrent_solves=-1 would even break the
    service's BoundedSemaphore at the first solve)."""

    LIMIT_FIELDS = [
        "exact_max_variables",
        "max_qpu_reads",
        "max_qpu_annealing_time_us",
        "max_remote_time_seconds",
        "max_concurrent_solves",
    ]

    @pytest.mark.parametrize("field", LIMIT_FIELDS)
    @pytest.mark.parametrize("value", [0, -1])
    def test_non_positive_limit_is_rejected(self, field, value):
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPolicy(**{field: value})

        assert [error["loc"] for error in exc_info.value.errors()] == [(field,)]

    @pytest.mark.parametrize("field", LIMIT_FIELDS)
    def test_smallest_positive_limit_is_accepted(self, field):
        policy = ExecutionPolicy(**{field: 1})

        assert getattr(policy, field) == 1

    def test_service_cannot_be_built_from_an_invalid_concurrency_limit(self):
        # Previously the bare ValueError surfaced from threading.BoundedSemaphore
        # on the first solve; now the policy itself refuses the value.
        with pytest.raises(ValidationError):
            ExecutionPolicy(max_concurrent_solves=-1)


# --- Phase 3a §11: generic limits ------------------------------------------

from annealbridge.config import SettingsError  # noqa: E402
from annealbridge.interfaces.composition import build_state_from_policy  # noqa: E402
from annealbridge.models import (  # noqa: E402
    AvailabilityStatus,
    ParameterLimit,
    SolverCapabilities,
)
from annealbridge.orchestration import OptimizationService  # noqa: E402
from annealbridge.solvers import SolverRegistry  # noqa: E402

COMPATIBILITY_KEYS = ["variables", "reads", "annealing_time_us", "time_seconds"]


def make_capabilities(**overrides) -> SolverCapabilities:
    fields = dict(
        name="fake",
        remote=False,
        heuristic=True,
        exhaustive=False,
        supports_seed=False,
        supports_num_reads=False,
        supports_time_limit=False,
        supported_model_types=["bqm"],
        returns_multiple_samples=True,
        description="Fake backend for policy tests.",
    )
    fields.update(overrides)
    return SolverCapabilities(**fields)


class TestLimitLookup:
    @pytest.mark.parametrize(
        "key, field",
        [
            ("variables", "exact_max_variables"),
            ("reads", "max_qpu_reads"),
            ("annealing_time_us", "max_qpu_annealing_time_us"),
            ("time_seconds", "max_remote_time_seconds"),
        ],
    )
    def test_compatibility_key_reads_the_phase2_field(self, key, field):
        policy = ExecutionPolicy(
            exact_max_variables=8,
            max_qpu_reads=10,
            max_qpu_annealing_time_us=123.5,
            max_remote_time_seconds=30,
        )

        assert policy.limit(key) == getattr(policy, field)

    def test_compatibility_key_keeps_the_field_type(self):
        policy = ExecutionPolicy()

        assert isinstance(policy.limit("variables"), int)
        assert isinstance(policy.limit("reads"), int)
        assert isinstance(policy.limit("time_seconds"), int)
        assert isinstance(policy.limit("annealing_time_us"), float)

    def test_custom_key_reads_limits_as_float(self):
        policy = ExecutionPolicy(limits={"iterations": 5})

        assert policy.limit("iterations") == 5.0
        assert isinstance(policy.limit("iterations"), float)

    def test_unknown_key_is_none(self):
        assert ExecutionPolicy().limit("iterations") is None

    def test_required_limit_raises_for_an_unknown_key(self):
        # The caller must have a ceiling: no value is an error, never a
        # comparison against None.
        assert ExecutionPolicy().limit("iterations") is None

        with pytest.raises(ValueError, match="iterations"):
            ExecutionPolicy().required_limit("iterations")

    def test_required_limit_equals_limit_when_present(self):
        policy = ExecutionPolicy(limits={"iterations": 5})

        assert policy.required_limit("variables") == policy.limit("variables")
        assert policy.required_limit("iterations") == policy.limit("iterations")

    def test_limits_default_to_empty(self):
        assert ExecutionPolicy().limits == {}


class TestLimitsValidation:
    @pytest.mark.parametrize("key", COMPATIBILITY_KEYS)
    def test_compatibility_key_in_limits_is_rejected(self, key):
        # One limit, one source: the Phase 2 field owns these keys.
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPolicy(limits={key: 5})

        assert [error["loc"] for error in exc_info.value.errors()] == [("limits",)]
        assert key in str(exc_info.value)

    @pytest.mark.parametrize(
        "value", [0, -1, float("inf"), float("-inf"), float("nan")]
    )
    def test_non_positive_or_non_finite_value_is_rejected(self, value):
        with pytest.raises(ValidationError) as exc_info:
            ExecutionPolicy(limits={"iterations": value})

        assert [error["loc"] for error in exc_info.value.errors()] == [("limits",)]

    def test_smallest_positive_value_is_accepted(self):
        assert ExecutionPolicy(limits={"iterations": 1e-9}).limit("iterations") == 1e-9

    def test_several_custom_keys(self):
        policy = ExecutionPolicy(limits={"iterations": 100, "bits": 2048})

        assert policy.limit("iterations") == 100.0
        assert policy.limit("bits") == 2048.0


class TestLimitsFor:
    """§12.3: the single source for the capabilities view and the service."""

    def test_shipped_backends_match_the_phase2_output_key_for_key(self):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy()

        limits = {
            name: policy.limits_for(registry.get(name).capabilities)
            for name in registry.names()
        }

        # The declared keys come first; the two service-level ceilings added
        # by the 2026-09-09 review (retries by the ``remote`` flag, then
        # top_k) close every backend's view.
        assert limits == {
            "exact": {
                "max_variables": 24,
                "max_local_retries": 10,
                "max_top_k": 1000,
            },
            "simulated_annealing": {
                "max_local_reads": 100000,
                "max_sweeps": 100000,
                "max_local_retries": 10,
                "max_top_k": 1000,
            },
            # ``tabu`` declares only a read ceiling: its sampler takes no
            # sweeps, so no sweep key is published for it.
            "tabu": {
                "max_local_reads": 100000,
                "max_local_retries": 10,
                "max_top_k": 1000,
            },
            # ``simulated_bifurcation`` reads sweeps as integration steps,
            # so it is capped under both local keys, like the annealer.
            "simulated_bifurcation": {
                "max_variables": 10000,
                "max_local_reads": 100000,
                "max_sweeps": 100000,
                "max_local_retries": 10,
                "max_top_k": 1000,
            },
            "dwave_qpu": {
                "max_reads": 1000,
                "max_annealing_time_us": 2000.0,
                "max_remote_retries": 3,
                "max_top_k": 1000,
            },
            "leap_hybrid_bqm": {
                "max_time_seconds": 300,
                "max_remote_retries": 3,
                "max_top_k": 1000,
            },
            "leap_hybrid_cqm": {
                "max_time_seconds": 300,
                "max_remote_retries": 3,
                "max_top_k": 1000,
            },
            "fujitsu_da": {
                "max_time_seconds": 300,
                "max_remote_retries": 3,
                "max_top_k": 1000,
            },
        }
        # Key order feeds the CLI table, so it is pinned too.
        assert list(limits["dwave_qpu"]) == [
            "max_reads",
            "max_annealing_time_us",
            "max_remote_retries",
            "max_top_k",
        ]
        assert list(limits["simulated_annealing"]) == [
            "max_local_reads",
            "max_sweeps",
            "max_local_retries",
            "max_top_k",
        ]
        assert list(limits["tabu"]) == [
            "max_local_reads",
            "max_local_retries",
            "max_top_k",
        ]
        assert list(limits["simulated_bifurcation"]) == [
            "max_variables",
            "max_local_reads",
            "max_sweeps",
            "max_local_retries",
            "max_top_k",
        ]

    def test_values_follow_the_policy(self):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy(
            exact_max_variables=8,
            max_qpu_reads=10,
            max_qpu_annealing_time_us=123.5,
            max_remote_time_seconds=30,
            max_local_reads=200,
            max_sweeps=300,
            max_local_retries=2,
            max_remote_retries=1,
            max_top_k=50,
        )

        assert policy.limits_for(registry.get("exact").capabilities) == {
            "max_variables": 8,
            "max_local_retries": 2,
            "max_top_k": 50,
        }
        assert policy.limits_for(
            registry.get("simulated_annealing").capabilities
        ) == {
            "max_local_reads": 200,
            "max_sweeps": 300,
            "max_local_retries": 2,
            "max_top_k": 50,
        }
        assert policy.limits_for(registry.get("dwave_qpu").capabilities) == {
            "max_reads": 10,
            "max_annealing_time_us": 123.5,
            "max_remote_retries": 1,
            "max_top_k": 50,
        }
        assert policy.limits_for(registry.get("leap_hybrid_bqm").capabilities) == {
            "max_time_seconds": 30,
            "max_remote_retries": 1,
            "max_top_k": 50,
        }

    def test_hybrid_cqm_style_declaration_yields_one_time_limit(self):
        # The leap_hybrid_cqm declaration (spec §7.1): the declared
        # time_seconds limit and the remote+time_limit flag name the same
        # key, so the view shows it once.
        caps = make_capabilities(
            name="leap_hybrid_cqm",
            remote=True,
            supports_time_limit=True,
            supported_model_types=["cqm"],
            parameter_limits=[
                ParameterLimit(
                    preference="leap_hybrid_bqm.time_limit_seconds",
                    limit="time_seconds",
                    error_code="REMOTE_TIME_LIMIT",
                )
            ],
        )

        assert ExecutionPolicy().limits_for(caps) == {
            "max_time_seconds": 300,
            "max_remote_retries": 3,
            "max_top_k": 1000,
        }

    def test_custom_declared_key_is_reported_from_limits(self):
        caps = make_capabilities(
            name="custom",
            remote=True,
            supports_num_reads=True,
            parameter_limits=[
                ParameterLimit(
                    preference="num_reads", limit="iterations", error_code="QPU_READS_LIMIT"
                )
            ],
        )
        policy = ExecutionPolicy(limits={"iterations": 100000})

        assert policy.limits_for(caps) == {
            "max_iterations": 100000.0,
            "max_remote_retries": 3,
            "max_top_k": 1000,
        }

    def test_remote_with_num_reads_but_no_declaration_has_only_service_limits(self):
        # The Phase 2 drift (spec §0 item 1) is gone: without a declaration
        # the view reports no read ceiling, and the service enforces none.
        # Only the two service-level ceilings (spec §11.4), which belong to
        # no backend, remain.
        caps = make_capabilities(remote=True, supports_num_reads=True)

        assert ExecutionPolicy().limits_for(caps) == {
            "max_remote_retries": 3,
            "max_top_k": 1000,
        }


class DeclaringBackend:
    """Minimal ``SolverBackend`` whose only purpose is its declaration."""

    def __init__(self, capabilities: SolverCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> SolverCapabilities:
        return self._capabilities

    def is_available(self) -> AvailabilityStatus:
        return AvailabilityStatus(category="available")

    @property
    def name(self) -> str:
        return self._capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return self._capabilities.exhaustive

    def resolve_time_limit(self, compiled_problem, preferences):
        return None

    def solve(self, compiled_problem, preferences):
        raise AssertionError("solve() is not exercised by construction tests")


def custom_limit_registry() -> SolverRegistry:
    backend = DeclaringBackend(
        make_capabilities(
            name="fake_custom",
            parameter_limits=[
                ParameterLimit(
                    preference="num_sweeps", limit="iterations", error_code="QPU_READS_LIMIT"
                )
            ],
        )
    )
    return SolverRegistry({"custom": backend})


class TestServiceDeclaredLimitConsistency:
    """§11.3: a declared limit the policy cannot value fails at construction."""

    def test_shipped_registry_builds_with_the_default_policy(self):
        OptimizationService(registry=SolverRegistry.default(), policy=ExecutionPolicy())

    def test_missing_policy_value_raises_naming_backend_and_key(self):
        with pytest.raises(ValueError) as exc_info:
            OptimizationService(registry=custom_limit_registry(), policy=ExecutionPolicy())

        message = str(exc_info.value)
        assert "'custom'" in message  # the registry key
        assert "'iterations'" in message

    def test_policy_with_the_value_builds(self):
        OptimizationService(
            registry=custom_limit_registry(),
            policy=ExecutionPolicy(limits={"iterations": 100}),
        )

    def test_composition_root_reports_it_as_a_settings_error(self):
        with pytest.raises(SettingsError) as exc_info:
            build_state_from_policy(ExecutionPolicy(), custom_limit_registry())

        message = str(exc_info.value)
        assert message.startswith("Invalid server settings")
        assert "'custom'" in message
        assert "'iterations'" in message
        assert exc_info.value.__cause__ is None

    def test_composition_root_builds_when_the_policy_has_the_value(self):
        state = build_state_from_policy(
            ExecutionPolicy(limits={"iterations": 100}), custom_limit_registry()
        )

        assert state.policy.limit("iterations") == 100.0
