"""Ceilings on the caller-controlled solve parameters (2026-09-09 review F-02 / F-07 / F-08).

``max_retries``, the simulated-annealing ``num_reads`` / ``num_sweeps`` and
``top_k`` are bounded by policy; a value over the ceiling is refused as a
structured ``resource_limit_exceeded`` result and is never clamped. The
hard penalty is stopped before it leaves the floating-point range, and
``penalty_multiplier`` must be a finite number.
"""

import math
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from annealbridge.config import ServerSettings, SettingsError, load_settings
from annealbridge.interfaces.capabilities import build_capabilities
from annealbridge.models import (
    CompiledProblem,
    OptimizationProblem,
    SolverExecutionMetadata,
    SolverPreferences,
)
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import (
    COMPATIBILITY_LIMIT_FIELDS,
    ExecutionPolicy,
)
from annealbridge.solvers import (
    AvailabilityStatus,
    RawSolverResult,
    SolverCapabilities,
    SolverRegistry,
)
from annealbridge.validation import validate_problem
from tests.unit.test_settings import (
    fixture_clean_env,  # noqa: F401  (registers the clean_env fixture)
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_problem(**solver_overrides) -> OptimizationProblem:
    """maximize 2a + b s.t. a + b >= 1 (feasible; the all-zero sample is not)."""
    return OptimizationProblem.model_validate(
        {
            "name": "parameter ceiling problem",
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
                    "id": "at_least_one",
                    "type": "hard",
                    "terms": [
                        {"variable": "a", "coefficient": 1},
                        {"variable": "b", "coefficient": 1},
                    ],
                    "operator": ">=",
                    "rhs": 1,
                }
            ],
            "solver": {"backend": "simulated_annealing", **solver_overrides},
        }
    )


class AlwaysZeroBackend:
    """A local BQM backend that only ever returns the all-zero sample.

    Every attempt is therefore infeasible for :func:`make_problem`, so the
    penalty ladder is climbed as far as the service lets it.
    """

    capabilities = SolverCapabilities(
        name="always_zero",
        remote=False,
        heuristic=True,
        exhaustive=False,
        supports_seed=False,
        supports_num_reads=False,
        supports_time_limit=False,
        supported_model_types=["bqm"],
        returns_multiple_samples=False,
        description="test-only",
    )

    def __init__(self, metadata: SolverExecutionMetadata | None = None) -> None:
        self.solve_calls = 0
        self.penalties: list[float] = []
        # A backend that reports execution facts (F-08): the ladder's last
        # completed attempt must keep them even when the ladder then stops.
        # ``None`` (the default) is the metadata-free backend the other
        # tests use.
        self._metadata = metadata

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

    def solve(self, compiled: CompiledProblem, preferences: SolverPreferences):
        self.solve_calls += 1
        self.penalties.append(compiled.hard_penalty)
        variables = [str(v) for v in compiled.model.variables]
        row = {v: 0 for v in variables}
        return RawSolverResult.from_dicts(
            [row], [float(compiled.model.energy(row))], backend=self.name,
            variables=variables, metadata=self._metadata,
        )


def make_zero_service(
    *, metadata: SolverExecutionMetadata | None = None, **policy_kwargs
) -> tuple[AlwaysZeroBackend, OptimizationService]:
    backend = AlwaysZeroBackend(metadata)
    registry = SolverRegistry({"simulated_annealing": backend})
    return backend, OptimizationService(
        registry=registry, policy=ExecutionPolicy(**policy_kwargs)
    )


def codes(result) -> list[str]:
    return [error.code for error in result.errors]


# --------------------------------------------------------------------------
# Policy / settings
# --------------------------------------------------------------------------

NEW_FIELDS = {
    "max_local_reads": ("local_reads", 100000),
    "max_sweeps": ("sweeps", 100000),
    "max_local_retries": ("local_retries", 10),
    "max_remote_retries": ("remote_retries", 3),
    "max_top_k": ("top_k", 1000),
}


class TestPolicyFields:
    def test_defaults(self):
        policy = ExecutionPolicy()
        for field, (_, default) in NEW_FIELDS.items():
            assert getattr(policy, field) == default, field

    def test_limit_keys_map_to_the_fields(self):
        policy = ExecutionPolicy(
            max_local_reads=7,
            max_sweeps=8,
            max_local_retries=0,
            max_remote_retries=1,
            max_top_k=2,
        )
        for field, (key, _) in NEW_FIELDS.items():
            assert COMPATIBILITY_LIMIT_FIELDS[key] == field
            assert policy.limit(key) == getattr(policy, field)

    @pytest.mark.parametrize("field", list(NEW_FIELDS))
    def test_generic_limits_refuse_the_built_in_keys(self, field):
        key = NEW_FIELDS[field][0]
        with pytest.raises(ValidationError, match=field):
            ExecutionPolicy(limits={key: 5})

    @pytest.mark.parametrize(
        "field, bad",
        [
            ("max_local_reads", 0),
            ("max_sweeps", 0),
            ("max_top_k", 0),
            ("max_local_retries", -1),
            ("max_remote_retries", -1),
        ],
    )
    def test_lower_bounds(self, field, bad):
        with pytest.raises(ValidationError):
            ExecutionPolicy(**{field: bad})

    def test_zero_retries_is_a_valid_ceiling(self):
        policy = ExecutionPolicy(max_local_retries=0, max_remote_retries=0)
        assert policy.limit("local_retries") == 0
        assert policy.limit("remote_retries") == 0


class TestSettings:
    def test_defaults_match_the_policy(self, clean_env):
        settings = ServerSettings()
        policy = settings.to_policy()
        for field, (_, default) in NEW_FIELDS.items():
            assert getattr(settings, field) == default
            assert getattr(policy, field) == default

    def test_env_overrides_reach_the_policy(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_MAX_LOCAL_READS", "500")
        clean_env.setenv("ANNEALBRIDGE_MAX_SWEEPS", "600")
        clean_env.setenv("ANNEALBRIDGE_MAX_LOCAL_RETRIES", "2")
        clean_env.setenv("ANNEALBRIDGE_MAX_REMOTE_RETRIES", "1")
        clean_env.setenv("ANNEALBRIDGE_MAX_TOP_K", "9")

        policy = load_settings().to_policy()

        assert policy.max_local_reads == 500
        assert policy.max_sweeps == 600
        assert policy.max_local_retries == 2
        assert policy.max_remote_retries == 1
        assert policy.max_top_k == 9

    @pytest.mark.parametrize(
        "variable, bad",
        [
            ("ANNEALBRIDGE_MAX_LOCAL_READS", "0"),
            ("ANNEALBRIDGE_MAX_SWEEPS", "-5"),
            ("ANNEALBRIDGE_MAX_LOCAL_RETRIES", "-1"),
            ("ANNEALBRIDGE_MAX_REMOTE_RETRIES", "abc"),
            ("ANNEALBRIDGE_MAX_TOP_K", "inf"),
        ],
    )
    def test_bad_values_are_settings_errors(self, clean_env, variable, bad):
        clean_env.setenv(variable, bad)
        with pytest.raises(SettingsError, match=variable):
            load_settings()

    @pytest.mark.parametrize("key", [key for key, _ in NEW_FIELDS.values()])
    def test_limits_json_refuses_the_built_in_keys(self, clean_env, key):
        clean_env.setenv("ANNEALBRIDGE_LIMITS", f'{{"{key}": 5}}')
        with pytest.raises(SettingsError, match=COMPATIBILITY_LIMIT_FIELDS[key]):
            load_settings()


# --------------------------------------------------------------------------
# Capabilities view
# --------------------------------------------------------------------------


class TestLimitsView:
    def test_shipped_backends(self):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy()
        limits = {
            name: policy.limits_for(registry.get(name).capabilities)
            for name in registry.names()
        }
        assert limits["simulated_annealing"] == {
            "max_local_reads": 100000,
            "max_sweeps": 100000,
            "max_local_retries": 10,
            "max_top_k": 1000,
        }
        assert limits["exact"] == {
            "max_variables": 24,
            "max_local_retries": 10,
            "max_top_k": 1000,
        }
        assert limits["dwave_qpu"] == {
            "max_reads": 1000,
            "max_annealing_time_us": 2000.0,
            "max_remote_retries": 3,
            "max_top_k": 1000,
        }
        for name in ("leap_hybrid_bqm", "leap_hybrid_cqm", "fujitsu_da"):
            assert limits[name] == {
                "max_time_seconds": 300,
                "max_remote_retries": 3,
                "max_top_k": 1000,
            }, name

    def test_build_capabilities_follows_the_policy(self):
        policy = ExecutionPolicy(
            max_local_reads=11, max_sweeps=12, max_local_retries=1, max_top_k=2
        )
        view = build_capabilities(SolverRegistry.default(), policy)
        by_name = {backend.name: backend for backend in view.backends}
        assert by_name["simulated_annealing"].limits == {
            "max_local_reads": 11,
            "max_sweeps": 12,
            "max_local_retries": 1,
            "max_top_k": 2,
        }
        # Key order feeds the CLI table: declared limits first, then the
        # service-level ceilings.
        assert list(by_name["simulated_annealing"].limits) == [
            "max_local_reads",
            "max_sweeps",
            "max_local_retries",
            "max_top_k",
        ]


# --------------------------------------------------------------------------
# Local ceilings through the service (never clamped)
# --------------------------------------------------------------------------


class TestLocalCeilings:
    @pytest.mark.parametrize(
        "preference, value, code",
        [
            ("num_reads", 100001, "LOCAL_READS_LIMIT"),
            ("num_sweeps", 100001, "SWEEPS_LIMIT"),
            ("max_retries", 11, "RETRY_LIMIT"),
            ("top_k", 1001, "TOP_K_LIMIT"),
        ],
    )
    def test_over_the_default_ceiling_is_refused(self, preference, value, code):
        service = OptimizationService()
        problem = make_problem(**{preference: value})

        result = service.solve(problem)

        assert result.status == "resource_limit_exceeded"
        assert codes(result) == [code]
        assert result.attempts == []
        assert result.solutions == []
        # The offending value is reported as given: nothing was clamped.
        assert str(value) in result.errors[0].message
        assert result.errors[0].recommended_action

    @pytest.mark.parametrize(
        "preference, value, policy_field",
        [
            ("num_reads", 10, "max_local_reads"),
            ("num_sweeps", 10, "max_sweeps"),
            ("max_retries", 1, "max_local_retries"),
            ("top_k", 1, "max_top_k"),
        ],
    )
    def test_equal_to_the_ceiling_is_allowed(self, preference, value, policy_field):
        service = OptimizationService(policy=ExecutionPolicy(**{policy_field: value}))
        problem = make_problem(**{preference: value, "seed": 1})

        result = service.solve(problem)

        assert result.status == "success", result.errors

    def test_refusal_never_reaches_the_backend(self):
        backend, service = make_zero_service(max_local_retries=2)

        result = service.solve(make_problem(max_retries=3))

        assert result.status == "resource_limit_exceeded"
        assert codes(result) == ["RETRY_LIMIT"]
        assert backend.solve_calls == 0

    def test_every_violation_is_reported_at_once(self):
        service = OptimizationService(
            policy=ExecutionPolicy(max_local_reads=1, max_sweeps=1, max_top_k=1)
        )
        result = service.solve(
            make_problem(num_reads=2, num_sweeps=2, top_k=2, max_retries=11)
        )
        assert result.status == "resource_limit_exceeded"
        assert codes(result) == [
            "LOCAL_READS_LIMIT",
            "SWEEPS_LIMIT",
            "RETRY_LIMIT",
            "TOP_K_LIMIT",
        ]

    def test_retry_ceiling_applies_to_the_exhaustive_backend_too(self):
        # The exact backend never retries, but the ceiling is one rule for
        # every backend: a value over it is refused rather than ignored.
        service = OptimizationService()
        result = service.solve(make_problem(backend="exact", max_retries=11))
        assert result.status == "resource_limit_exceeded"
        assert codes(result) == ["RETRY_LIMIT"]

    def test_recommend_reports_the_same_ceiling(self):
        service = OptimizationService()
        result = service.recommend(make_problem(num_sweeps=100001))
        entry = next(
            r for r in result.recommendations if r.backend == "simulated_annealing"
        )
        assert "SWEEPS_LIMIT" in [error.code for error in entry.blocking]


# --------------------------------------------------------------------------
# Penalty overflow (F-07)
# --------------------------------------------------------------------------


@pytest.fixture
def warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


class TestPenaltyOverflow:
    def test_initial_penalty_beyond_float_range_is_refused(self, warnings_are_errors):
        backend, service = make_zero_service()
        # penalty_scale (3) × 1e308 is already inf: nothing is compiled.
        result = service.solve(make_problem(penalty_multiplier=1e308))

        assert result.status == "resource_limit_exceeded"
        assert codes(result) == ["PENALTY_OVERFLOW"]
        assert result.attempts == []
        assert backend.solve_calls == 0
        assert result.errors[0].recommended_action

    def test_doubling_stops_before_the_penalty_leaves_float_range(
        self, warnings_are_errors
    ):
        backend, service = make_zero_service()
        # ~1e307 after scaling; doubling reaches inf within the 10 retries.
        result = service.solve(
            make_problem(penalty_multiplier=3.3e306, max_retries=10)
        )

        assert result.status == "resource_limit_exceeded"
        assert codes(result) == ["PENALTY_OVERFLOW"]
        assert 1 <= len(result.attempts) < 11
        assert len(result.attempts) == backend.solve_calls
        assert all(math.isfinite(attempt.penalty) for attempt in result.attempts)
        assert all(math.isfinite(p) for p in backend.penalties)
        message = result.errors[0].message
        assert str(len(result.attempts)) in message

    def test_finite_penalty_that_overflows_in_compile_is_refused(
        self, warnings_are_errors
    ):
        backend, service = make_zero_service()
        # A penalty near the float ceiling stays finite itself but the
        # squared-penalty expansion (lam * coefficient^2) overflows.
        result = service.solve(make_problem(penalty_multiplier=5e307))

        assert result.status == "resource_limit_exceeded"
        assert codes(result) == ["PENALTY_OVERFLOW"]
        assert backend.solve_calls == 0

    def test_last_completed_attempt_metadata_survives_the_ladder(
        self, warnings_are_errors
    ):
        # F-08: climbing the ladder may have spent real quota, so the
        # PENALTY_OVERFLOW result must carry the last completed attempt's
        # facts, exactly like the infeasible and solver_error paths do.
        backend, service = make_zero_service(
            metadata=SolverExecutionMetadata(
                backend="always_zero", remote=False, solver_id="zero-attempt"
            )
        )
        result = service.solve(
            make_problem(penalty_multiplier=3.3e306, max_retries=10)
        )

        assert result.status == "resource_limit_exceeded"
        assert codes(result) == ["PENALTY_OVERFLOW"]
        assert len(result.attempts) >= 1
        assert backend.solve_calls == len(result.attempts)
        assert result.metadata is not None
        assert result.metadata.solver_id == "zero-attempt"
        # The service, not the backend, stamps the model type (3a §22).
        assert result.metadata.model_type == "bqm"

    def test_no_metadata_is_invented_when_no_attempt_ran(self, warnings_are_errors):
        # The mirror image: the very first penalty is already inf, so no
        # attempt completed and there is nothing to report — the backend's
        # metadata must not be conjured out of a solve that never happened.
        backend, service = make_zero_service(
            metadata=SolverExecutionMetadata(
                backend="always_zero", remote=False, solver_id="zero-attempt"
            )
        )
        result = service.solve(make_problem(penalty_multiplier=1e308))

        assert result.status == "resource_limit_exceeded"
        assert codes(result) == ["PENALTY_OVERFLOW"]
        assert result.attempts == []
        assert backend.solve_calls == 0
        assert result.metadata is None

    def test_result_serialises_with_finite_penalties_only(self, warnings_are_errors):
        _, service = make_zero_service()
        result = service.solve(
            make_problem(penalty_multiplier=3.3e306, max_retries=10)
        )
        payload = result.model_dump(mode="json")
        assert all(attempt["penalty"] is not None for attempt in payload["attempts"])


# --------------------------------------------------------------------------
# penalty_multiplier finiteness (F-08)
# --------------------------------------------------------------------------


class TestPenaltyMultiplierFiniteness:
    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_schema_rejects_non_finite(self, value):
        with pytest.raises(ValidationError):
            SolverPreferences(penalty_multiplier=value)

    @pytest.mark.parametrize("value", [math.inf, math.nan])
    def test_validator_rejects_non_finite_when_the_schema_is_bypassed(self, value):
        problem = make_problem()
        problem.solver = SolverPreferences.model_construct(
            **{**problem.solver.model_dump(), "penalty_multiplier": value}
        )

        errors = validate_problem(problem)

        assert [error.code for error in errors] == ["INVALID_SOLVER_PREFERENCE"]
        assert errors[0].path == "solver.penalty_multiplier"

    def test_service_refuses_nan_instead_of_solving(self):
        backend, service = make_zero_service()
        problem = make_problem()
        problem.solver = SolverPreferences.model_construct(
            **{**problem.solver.model_dump(), "penalty_multiplier": math.nan}
        )

        result = service.solve(problem)

        assert result.status == "invalid_problem"
        assert "INVALID_SOLVER_PREFERENCE" in codes(result)
        assert backend.solve_calls == 0
        assert result.attempts == []

    def test_finite_multiplier_still_passes(self):
        assert validate_problem(make_problem(penalty_multiplier=0.5)) == []
        with pytest.raises(ValidationError):
            SolverPreferences(penalty_multiplier=np.inf)
