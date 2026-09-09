"""End-to-end service tests for the remote-backend flow (Phase 2 spec §14).

Everything goes through :meth:`OptimizationService.solve`; no internal
helper is called directly. Real D-Wave is never touched: fake samplers are
injected through each backend's ``sampler_factory`` seam, and availability
is forced by patching the ``dwave_availability`` name each backend module
imports from ``solvers.ocean`` (the real ``is_available()`` reports the
backend unusable whether or not the ``dwave`` extra is installed: no
package, or no credentials). Patching per module keeps the two backends
independently controllable.
"""

import threading

import dimod
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import (
    DWaveQPUOptions,
    LeapHybridBQMOptions,
    OptimizationProblem,
)
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.penalty import ScaledPenaltyStrategy
from annealbridge.solvers import (
    REASON_CONFIG_INVALID,
    REASON_CREDENTIALS_MISSING,
    REASON_NOT_INSTALLED,
    AvailabilityStatus,
    DWaveQPUBackend,
    ExactSolverBackend,
    LeapHybridBQMBackend,
    SimulatedAnnealingBackend,
    SolverCapabilities,
    SolverRegistry,
)
import annealbridge.solvers.dwave_qpu as qpu_module
import annealbridge.solvers.leap_hybrid_bqm as leap_module
from tests.remote_mock.conftest import (
    FAKE_MINIMAL_SAMPLESET_INFO,
    CountingFactory,
    FakeLeapHybridSampler,
    FakeQPUSampler,
    RequestTimeout,
    SolverFailureError,
    make_problem,
    make_remote_available,
)

# The business assignment the flow tests expect back from a fake sampler:
# the optimum of "maximize 2a + b s.t. a + b <= 1".
BEST = [{"a": 1, "b": 0}]


class BlockingQPUSampler(FakeQPUSampler):
    """QPU fake that parks inside ``sample()`` until it is released."""

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__(assignments=BEST)
        self.started = started
        self.release = release

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.started.set()
        assert self.release.wait(timeout=10)
        return super().sample(bqm, **kwargs)


class FakeUnavailableBackend:
    """A remote backend that reports the given availability status.

    Implements the ``SolverBackend`` protocol; ``solve()`` fails loudly so a
    test proves the service never reaches it.
    """

    def __init__(self, availability: AvailabilityStatus) -> None:
        self._availability = availability
        self.solve_calls = 0

    @property
    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(
            name="fake_remote",
            remote=True,
            heuristic=True,
            exhaustive=False,
            supports_seed=False,
            supports_num_reads=True,
            supports_time_limit=False,
            supported_model_types=["bqm"],
            returns_multiple_samples=True,
            description="Fake remote backend used to exercise availability gating.",
        )

    def is_available(self) -> AvailabilityStatus:
        return self._availability

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return self.capabilities.exhaustive

    def resolve_time_limit(self, compiled_problem, preferences):
        return None

    def solve(self, compiled_problem, preferences):
        self.solve_calls += 1
        raise AssertionError("solve() must not be reached for an unavailable backend")


def make_zero_infeasible_problem(
    backend: str = "dwave_qpu", **solver_overrides
) -> OptimizationProblem:
    """maximize 2a + b s.t. a + b >= 1, so the all-zero sample is infeasible."""
    return OptimizationProblem.model_validate(
        {
            "name": "service remote retry problem",
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
            "solver": {"backend": backend, **solver_overrides},
        }
    )


def internal_variables_of(problem: OptimizationProblem) -> set[str]:
    """Slack variables the compiler generates, using the service's penalty."""
    penalty = ScaledPenaltyStrategy().initial_penalty(problem)
    return BQMCompiler().compile(problem, penalty).internal_variables


def make_service(backend, key: str, **policy_kwargs) -> OptimizationService:
    """A service whose registry holds exactly ``backend`` under ``key``."""
    return OptimizationService(
        registry=SolverRegistry({key: backend}),
        policy=ExecutionPolicy(**policy_kwargs),
    )


def make_qpu_service(fake: FakeQPUSampler, **policy_kwargs) -> OptimizationService:
    backend = DWaveQPUBackend(sampler_factory=lambda: fake)
    return make_service(backend, "dwave_qpu", **policy_kwargs)


def make_hybrid_service(
    fake: FakeLeapHybridSampler, **policy_kwargs
) -> OptimizationService:
    backend = LeapHybridBQMBackend(sampler_factory=lambda: fake)
    return make_service(backend, "leap_hybrid_bqm", **policy_kwargs)


def assert_actions_present(result) -> None:
    """Every structured error/warning carries catalog guidance (spec §13.2)."""
    for entry in [*result.errors, *result.warnings]:
        assert entry.recommended_action is not None
        assert entry.recommended_action != ""


class TestRemoteDisabledByPolicy:
    """§14 step 4: the remote gate runs before availability is even checked."""

    def test_remote_backend_is_refused_without_touching_the_sampler(self):
        fake = FakeQPUSampler(assignments=BEST)
        service = make_qpu_service(fake, allow_remote=False)

        result = service.solve(make_problem())

        assert result.status == "backend_unavailable"
        assert result.backend == "dwave_qpu"
        assert result.solutions == []
        assert len(result.errors) == 1
        assert result.errors[0].code == "REMOTE_DISABLED"
        assert result.message
        assert_actions_present(result)

    def test_sampler_is_never_called(self):
        fake = FakeQPUSampler(assignments=BEST)
        service = make_qpu_service(fake, allow_remote=False)

        service.solve(make_problem())

        assert fake.sample_calls == 0
        assert fake.sample_bqm is None

    def test_no_availability_patching_is_required(self):
        # The gate fires even though the real is_available() reports the
        # backend unusable: "dwave-system not installed" in the mock venv,
        # or "credentials missing" once the dwave extra is installed (the
        # conftest clears DWAVE_API_TOKEN). REMOTE_DISABLED wins because
        # it is checked first.
        real = DWaveQPUBackend().is_available()
        assert real.category in {"not_installed", "credentials_missing"}, real

        result = make_qpu_service(FakeQPUSampler(assignments=BEST), allow_remote=False).solve(
            make_problem()
        )

        assert result.errors[0].code == "REMOTE_DISABLED"


class TestBackendDisabledByPolicy:
    """§14 step 3: enabled_backends is checked before anything else."""

    def make_service(self) -> OptimizationService:
        return OptimizationService(
            registry=SolverRegistry(
                {
                    "exact": ExactSolverBackend(),
                    "simulated_annealing": SimulatedAnnealingBackend(),
                }
            ),
            policy=ExecutionPolicy(enabled_backends={"exact"}),
        )

    def test_disabled_backend_is_refused(self):
        result = self.make_service().solve(
            make_problem(backend="simulated_annealing")
        )

        assert result.status == "backend_unavailable"
        assert result.backend == "simulated_annealing"
        assert result.solutions == []
        assert result.attempts == []
        assert len(result.errors) == 1
        assert result.errors[0].code == "BACKEND_DISABLED_BY_POLICY"
        assert_actions_present(result)

    def test_message_names_the_enabled_backends(self):
        result = self.make_service().solve(
            make_problem(backend="simulated_annealing")
        )

        assert "simulated_annealing" in result.errors[0].message
        assert "exact" in result.errors[0].message

    def test_enabled_backend_still_solves(self):
        result = self.make_service().solve(make_problem(backend="exact"))

        assert result.status == "success"
        assert result.backend == "exact"


class TestEnabledBackendsUseRegistryKeys:
    """enabled_backends is matched against the registry key the user requests
    (the same name the capabilities view reports), not capabilities.name —
    a custom registry may register a backend under a different key."""

    def make_service(self, enabled: set[str]) -> tuple[FakeUnavailableBackend, OptimizationService]:
        backend = FakeUnavailableBackend(
            AvailabilityStatus(
                category="credentials_missing", detail=REASON_CREDENTIALS_MISSING
            )
        )
        assert backend.name != "dwave_qpu"  # the key deliberately differs
        service = make_service(
            backend, "dwave_qpu", allow_remote=True, enabled_backends=enabled
        )
        return backend, service

    def test_registry_key_in_enabled_set_passes_the_gate(self):
        backend, service = self.make_service({"dwave_qpu"})

        result = service.solve(make_problem(backend="dwave_qpu"))

        # Past the policy gate: the next gate (availability) is the one that fires.
        assert result.errors[0].code == "REMOTE_CREDENTIALS_MISSING"
        assert backend.solve_calls == 0

    def test_capabilities_name_in_enabled_set_does_not_pass_the_gate(self):
        backend, service = self.make_service({"fake_remote"})

        result = service.solve(make_problem(backend="dwave_qpu"))

        assert result.status == "backend_unavailable"
        assert result.errors[0].code == "BACKEND_DISABLED_BY_POLICY"
        assert backend.solve_calls == 0

    def test_refusal_names_the_requested_backend_not_the_internal_one(self):
        _, service = self.make_service({"exact"})

        result = service.solve(make_problem(backend="dwave_qpu"))

        assert result.backend == "dwave_qpu"
        assert "'dwave_qpu'" in result.errors[0].message
        assert "fake_remote" not in result.errors[0].message


class TestAvailabilityClassification:
    """§14 step 5 / 3a §8.2: the availability *category* maps to a status
    and a default code; a backend-supplied ``error_code`` wins over the
    default. The service never compares reason strings."""

    @pytest.mark.parametrize(
        ("availability", "expected_status", "expected_code"),
        [
            (
                AvailabilityStatus(category="not_installed", detail=REASON_NOT_INSTALLED),
                "backend_unavailable",
                "BACKEND_NOT_INSTALLED",
            ),
            (
                AvailabilityStatus(
                    category="credentials_missing", detail=REASON_CREDENTIALS_MISSING
                ),
                "backend_unavailable",
                "REMOTE_CREDENTIALS_MISSING",
            ),
            (
                AvailabilityStatus(
                    category="config_invalid",
                    detail=REASON_CONFIG_INVALID,
                    error_code="DWAVE_CONFIG_INVALID",
                ),
                "configuration_error",
                "DWAVE_CONFIG_INVALID",
            ),
            (
                AvailabilityStatus(category="config_invalid", detail="config broken"),
                "configuration_error",
                "BACKEND_CONFIG_INVALID",
            ),
            (
                AvailabilityStatus(category="unavailable", detail="some new reason"),
                "backend_unavailable",
                "BACKEND_UNAVAILABLE",
            ),
            (
                AvailabilityStatus(category="unavailable"),
                "backend_unavailable",
                "BACKEND_UNAVAILABLE",
            ),
        ],
        ids=[
            "not-installed",
            "credentials",
            "invalid-config-dwave-code",
            "invalid-config-default-code",
            "unknown",
            "no-reason",
        ],
    )
    def test_reason_maps_to_status_and_code(
        self, availability, expected_status, expected_code
    ):
        backend = FakeUnavailableBackend(availability)
        service = make_service(backend, "dwave_qpu", allow_remote=True)

        result = service.solve(make_problem())

        assert result.status == expected_status
        assert result.backend == "fake_remote"
        assert result.solutions == []
        assert len(result.errors) == 1
        assert result.errors[0].code == expected_code
        assert result.errors[0].recommended_action
        expected_detail = availability.detail or "no reason reported"
        assert result.errors[0].message == (
            f"Backend 'fake_remote' is unavailable: {expected_detail}"
        )
        assert backend.solve_calls == 0
        assert_actions_present(result)

    def test_unreported_reason_still_produces_a_message(self):
        service = make_service(
            FakeUnavailableBackend(AvailabilityStatus(category="unavailable")),
            "dwave_qpu",
            allow_remote=True,
        )

        result = service.solve(make_problem())

        assert result.message
        assert "fake_remote" in result.errors[0].message


class TestPreferenceLimits:
    """§14 step 9: preferences are refused, never clamped, before solving."""

    def test_num_reads_over_the_limit_is_refused(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=BEST)
        service = make_qpu_service(fake, allow_remote=True, max_qpu_reads=10)

        result = service.solve(make_problem(num_reads=100))

        assert result.status == "resource_limit_exceeded"
        assert result.backend == "dwave_qpu"
        assert result.solutions == []
        assert result.attempts == []
        assert len(result.errors) == 1
        assert result.errors[0].code == "QPU_READS_LIMIT"
        assert_actions_present(result)

    def test_num_reads_message_reports_requested_and_maximum(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        service = make_qpu_service(
            FakeQPUSampler(assignments=BEST), allow_remote=True, max_qpu_reads=10
        )

        message = service.solve(make_problem(num_reads=100)).errors[0].message

        assert "100" in message
        assert "10" in message
        assert "num_reads 100" in message
        assert "maximum of 10" in message

    def test_num_reads_over_the_limit_never_reaches_the_sampler(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=BEST)
        service = make_qpu_service(fake, allow_remote=True, max_qpu_reads=10)

        service.solve(make_problem(num_reads=100))

        assert fake.sample_bqm is None
        assert fake.sample_calls == 0

    def test_annealing_time_over_the_limit_is_refused(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=BEST)
        service = make_qpu_service(
            fake, allow_remote=True, max_qpu_annealing_time_us=100.0
        )

        result = service.solve(
            make_problem(dwave_qpu=DWaveQPUOptions(annealing_time_us=500.0))
        )

        assert result.status == "resource_limit_exceeded"
        assert len(result.errors) == 1
        assert result.errors[0].code == "QPU_ANNEALING_TIME_LIMIT"
        assert "500.0" in result.errors[0].message
        assert "maximum of 100.0" in result.errors[0].message
        assert fake.sample_bqm is None
        assert fake.sample_calls == 0
        assert_actions_present(result)

    def test_hybrid_time_limit_over_the_limit_is_refused(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)
        fake = FakeLeapHybridSampler(assignments=BEST)
        service = make_hybrid_service(
            fake, allow_remote=True, max_remote_time_seconds=10
        )

        result = service.solve(
            make_problem(
                backend="leap_hybrid_bqm",
                leap_hybrid_bqm=LeapHybridBQMOptions(time_limit_seconds=60.0),
            )
        )

        assert result.status == "resource_limit_exceeded"
        assert result.backend == "leap_hybrid_bqm"
        assert len(result.errors) == 1
        assert result.errors[0].code == "REMOTE_TIME_LIMIT"
        assert "60.0" in result.errors[0].message
        assert "maximum of 10" in result.errors[0].message
        assert fake.sample_bqm is None
        assert fake.sample_calls == 0
        assert_actions_present(result)

    def test_all_violated_limits_are_reported_together(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=BEST)
        service = make_qpu_service(
            fake,
            allow_remote=True,
            max_qpu_reads=10,
            max_qpu_annealing_time_us=100.0,
        )

        result = service.solve(
            make_problem(
                num_reads=100,
                dwave_qpu=DWaveQPUOptions(annealing_time_us=500.0),
            )
        )

        assert result.status == "resource_limit_exceeded"
        assert len(result.errors) == 2
        assert {error.code for error in result.errors} == {
            "QPU_READS_LIMIT",
            "QPU_ANNEALING_TIME_LIMIT",
        }
        assert fake.sample_calls == 0
        assert_actions_present(result)

    def test_preferences_within_the_limits_still_solve(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=BEST)
        service = make_qpu_service(
            fake,
            allow_remote=True,
            max_qpu_reads=100,
            max_qpu_annealing_time_us=500.0,
        )

        result = service.solve(
            make_problem(
                num_reads=100,
                dwave_qpu=DWaveQPUOptions(annealing_time_us=500.0),
            )
        )

        # Exactly at the limit is allowed; only *over* the limit is refused.
        assert result.status == "success"
        assert fake.sample_calls == 1


class TestEffectiveHybridTimeLimit:
    """§14 step 9 / §16: the *submitted* time limit is checked, not just the
    user's. The sampler minimum can push it above policy even when the user
    gave nothing; that must be refused before any request is sent."""

    def make(self, *, minimum: float, maximum: int, **solver_overrides):
        fake = FakeLeapHybridSampler(assignments=BEST, min_time_limit=minimum)
        service = make_hybrid_service(
            fake, allow_remote=True, max_remote_time_seconds=maximum
        )
        problem = make_problem(backend="leap_hybrid_bqm", **solver_overrides)
        return fake, service.solve(problem)

    def test_omitted_time_limit_above_policy_is_refused(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)

        fake, result = self.make(minimum=5.0, maximum=3)

        assert result.status == "resource_limit_exceeded"
        assert result.backend == "leap_hybrid_bqm"
        assert result.solutions == []
        assert len(result.errors) == 1
        assert result.errors[0].code == "REMOTE_TIME_LIMIT"
        assert_actions_present(result)

    def test_refusal_happens_before_any_submission(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)

        fake, _ = self.make(minimum=5.0, maximum=3)

        assert fake.sample_calls == 0
        assert fake.sample_bqm is None

    def test_message_reports_effective_and_maximum(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)

        _, result = self.make(minimum=5.0, maximum=3)

        message = result.errors[0].message
        assert "5.0" in message
        assert "maximum of 3" in message

    def test_user_value_below_policy_but_minimum_above_is_refused(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)

        fake, result = self.make(
            minimum=5.0,
            maximum=3,
            leap_hybrid_bqm=LeapHybridBQMOptions(time_limit_seconds=2.0),
        )

        # The user's 2.0 passes the preference check; the effective 5.0
        # (raised to the minimum) does not, and nothing is submitted.
        assert result.status == "resource_limit_exceeded"
        assert result.errors[0].code == "REMOTE_TIME_LIMIT"
        assert "5.0" in result.errors[0].message
        assert fake.sample_calls == 0

    def test_effective_exactly_at_policy_still_solves(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)

        fake, result = self.make(minimum=3.0, maximum=3)

        assert result.status == "success"
        assert fake.sample_calls == 1
        assert fake.sample_kwargs == {"time_limit": 3.0}
        assert result.metadata.effective_time_limit_seconds == 3.0

    def test_effective_value_within_policy_is_the_one_submitted(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)

        fake, result = self.make(
            minimum=5.0,
            maximum=10,
            leap_hybrid_bqm=LeapHybridBQMOptions(time_limit_seconds=2.0),
        )

        assert result.status == "success"
        assert fake.sample_kwargs == {"time_limit": 5.0}
        assert result.metadata.effective_time_limit_seconds == 5.0

    def test_time_limit_resolution_failure_is_a_solver_error(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)
        fake = FakeLeapHybridSampler(assignments=BEST)
        factory = CountingFactory(fake, failures=[ValueError("bad region")])
        service = make_service(
            LeapHybridBQMBackend(sampler_factory=factory),
            "leap_hybrid_bqm",
            allow_remote=True,
        )

        result = service.solve(make_problem(backend="leap_hybrid_bqm"))

        assert result.status == "solver_error"
        assert result.errors[0].code == "DWAVE_CONFIG_INVALID"
        assert fake.sample_calls == 0
        assert_actions_present(result)


class TestLazySampleSetFailures:
    """Ocean samplesets resolve on first access; a cloud failure surfacing
    there must still come back as a structured solver_error, never as an
    exception escaping the service."""

    @pytest.mark.parametrize(
        ("failure", "expected_code"),
        [
            (RequestTimeout("remote solve timed out"), "REMOTE_TIMEOUT"),
            (SolverFailureError("solver rejected the problem"), "REMOTE_SOLVER_ERROR"),
        ],
        ids=["timeout", "solver-failure"],
    )
    def test_qpu_resolve_failure_is_structured(self, monkeypatch, failure, expected_code):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(raise_on_sample=failure, lazy=True)
        service = make_qpu_service(fake, allow_remote=True)

        result = service.solve(make_problem())

        assert result.status == "solver_error"
        assert result.backend == "dwave_qpu"
        assert result.solutions == []
        assert [error.code for error in result.errors] == [expected_code]
        assert fake.sample_calls == 1
        assert_actions_present(result)

    @pytest.mark.parametrize(
        ("failure", "expected_code"),
        [
            (RequestTimeout("remote solve timed out"), "REMOTE_TIMEOUT"),
            (SolverFailureError("solver rejected the problem"), "REMOTE_SOLVER_ERROR"),
        ],
        ids=["timeout", "solver-failure"],
    )
    def test_hybrid_resolve_failure_is_structured(
        self, monkeypatch, failure, expected_code
    ):
        make_remote_available(monkeypatch, leap_module)
        fake = FakeLeapHybridSampler(raise_on_sample=failure, lazy=True)
        service = make_hybrid_service(fake, allow_remote=True)

        result = service.solve(make_problem(backend="leap_hybrid_bqm"))

        assert result.status == "solver_error"
        assert result.backend == "leap_hybrid_bqm"
        assert [error.code for error in result.errors] == [expected_code]
        assert fake.sample_calls == 1
        assert_actions_present(result)

    def test_lazy_success_path_solves_normally(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(
            assignments=BEST, info=FAKE_MINIMAL_SAMPLESET_INFO, lazy=True
        )
        service = make_qpu_service(fake, allow_remote=True)

        result = service.solve(make_problem())

        assert result.status == "success"
        assert result.solutions[0].variables == {"a": 1, "b": 0}
        assert result.metadata.timing_us == {"qpu_access_time": 12345.0}


class TestSamplerConstructionFailures:
    """A sampler that cannot be built is a configuration problem, not an
    embedding problem: the agent must not be told to shrink the problem."""

    def test_qpu_factory_value_error_is_not_embedding_failed(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        factory = CountingFactory(FakeQPUSampler(assignments=BEST), failures=[ValueError("bad region")])
        service = make_service(
            DWaveQPUBackend(sampler_factory=factory), "dwave_qpu", allow_remote=True
        )

        result = service.solve(make_problem())

        assert result.status == "solver_error"
        assert result.errors[0].code == "DWAVE_CONFIG_INVALID"
        assert_actions_present(result)

    def test_qpu_sample_value_error_is_still_embedding_failed(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(raise_on_sample=ValueError("no embedding found"))
        service = make_qpu_service(fake, allow_remote=True)

        result = service.solve(make_problem())

        assert result.status == "solver_error"
        assert result.errors[0].code == "EMBEDDING_FAILED"


class TestSamplerReuse:
    """The sampler is built once per backend instance, not per solve/retry."""

    def test_qpu_sampler_is_built_once_across_retries(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=[{"a": 0, "b": 0}])
        factory = CountingFactory(fake)
        service = make_service(
            DWaveQPUBackend(sampler_factory=factory),
            "dwave_qpu",
            allow_remote=True,
            allow_remote_retries=True,
        )

        result = service.solve(make_zero_infeasible_problem(max_retries=2))

        assert result.status == "infeasible"
        assert fake.sample_calls == 3
        assert factory.calls == 1

    def test_qpu_sampler_is_reused_across_solves(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=BEST)
        factory = CountingFactory(fake)
        service = make_service(
            DWaveQPUBackend(sampler_factory=factory), "dwave_qpu", allow_remote=True
        )

        assert service.solve(make_problem()).status == "success"
        assert service.solve(make_problem()).status == "success"

        assert fake.sample_calls == 2
        assert factory.calls == 1

    def test_hybrid_sampler_is_built_once_for_check_and_solve(self, monkeypatch):
        make_remote_available(monkeypatch, leap_module)
        fake = FakeLeapHybridSampler(assignments=BEST)
        factory = CountingFactory(fake)
        service = make_service(
            LeapHybridBQMBackend(sampler_factory=factory),
            "leap_hybrid_bqm",
            allow_remote=True,
        )

        # The time-limit check and the solve share one sampler.
        assert service.solve(make_problem(backend="leap_hybrid_bqm")).status == "success"
        assert service.solve(make_problem(backend="leap_hybrid_bqm")).status == "success"

        assert fake.sample_calls == 2
        assert factory.calls == 1

    def test_failed_construction_is_not_cached(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=BEST)
        factory = CountingFactory(fake, failures=[RequestTimeout("connect timed out")])
        service = make_service(
            DWaveQPUBackend(sampler_factory=factory), "dwave_qpu", allow_remote=True
        )

        failed = service.solve(make_problem())
        recovered = service.solve(make_problem())

        assert failed.status == "solver_error"
        assert failed.errors[0].code == "REMOTE_TIMEOUT"
        assert recovered.status == "success"
        assert factory.calls == 2


class TestSuccessfulRemoteSolve:
    def solve(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(
            assignments=[{"a": 0, "b": 0}, {"a": 1, "b": 0}],
            info=FAKE_MINIMAL_SAMPLESET_INFO,
        )
        service = make_qpu_service(fake, allow_remote=True)
        return fake, service.solve(make_problem())

    def test_status_backend_and_attempts(self, monkeypatch):
        fake, result = self.solve(monkeypatch)

        assert result.status == "success"
        assert result.backend == "dwave_qpu"
        assert result.objective_direction == "maximize"
        assert result.errors == []
        assert len(result.attempts) == 1
        assert fake.sample_calls == 1

    def test_metadata_comes_from_the_backend(self, monkeypatch):
        _, result = self.solve(monkeypatch)

        assert result.metadata is not None
        assert result.metadata.backend == "dwave_qpu"
        assert result.metadata.remote is True
        assert result.metadata.num_reads_requested == 100
        assert result.metadata.timing_us == {"qpu_access_time": 12345.0}

    def test_solution_variables_exclude_internal_slack(self, monkeypatch):
        problem = make_problem()
        internal = internal_variables_of(problem)
        assert internal  # the <= constraint really does generate slack

        _, result = self.solve(monkeypatch)

        assert result.solutions
        for solution in result.solutions:
            assert set(solution.variables) == {"a", "b"}
            assert not internal & set(solution.variables)

    def test_ranking_uses_the_business_objective(self, monkeypatch):
        _, result = self.solve(monkeypatch)

        assert [solution.rank for solution in result.solutions] == [1, 2]
        assert result.solutions[0].variables == {"a": 1, "b": 0}
        assert result.solutions[0].objective_value == pytest.approx(2.0)
        assert result.solutions[0].hard_constraints_satisfied is True


class TestRemoteRetryPolicy:
    """§14 step 14: remote retries burn quota, so policy must opt in."""

    def solve(self, monkeypatch, *, allow_remote_retries: bool, max_retries: int):
        make_remote_available(monkeypatch, qpu_module)
        # Only the all-zero sample, which violates a + b >= 1, so no attempt
        # can ever produce a feasible candidate.
        fake = FakeQPUSampler(assignments=[{"a": 0, "b": 0}])
        service = make_qpu_service(
            fake,
            allow_remote=True,
            allow_remote_retries=allow_remote_retries,
        )
        problem = make_zero_infeasible_problem(max_retries=max_retries)
        return fake, service.solve(problem)

    def test_retries_disabled_makes_exactly_one_attempt(self, monkeypatch):
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=False, max_retries=3
        )

        assert result.status == "infeasible"
        assert result.solutions == []
        assert len(result.attempts) == 1
        assert fake.sample_calls == 1

    def test_retries_disabled_never_proves_infeasibility(self, monkeypatch):
        _, result = self.solve(monkeypatch, allow_remote_retries=False, max_retries=3)

        assert result.infeasibility_proven is False
        assert result.message

    def test_retries_disabled_emits_exactly_one_warning(self, monkeypatch):
        _, result = self.solve(monkeypatch, allow_remote_retries=False, max_retries=3)

        assert len(result.warnings) == 1
        warning = result.warnings[0]
        assert warning.code == "REMOTE_RETRIES_DISABLED"
        assert warning.recommended_action is not None
        assert result.errors == []
        assert_actions_present(result)

    def test_retries_enabled_uses_one_plus_max_retries(self, monkeypatch):
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=True, max_retries=2
        )

        assert result.status == "infeasible"
        assert len(result.attempts) == 3
        assert fake.sample_calls == 3
        assert [attempt.attempt for attempt in result.attempts] == [1, 2, 3]

    def test_retries_enabled_emits_no_retries_disabled_warning(self, monkeypatch):
        _, result = self.solve(monkeypatch, allow_remote_retries=True, max_retries=2)

        assert result.warnings == []
        assert result.infeasibility_proven is False

    def test_user_requested_single_attempt_emits_no_warning(self, monkeypatch):
        # max_retries=0 means the user asked for one attempt: policy blocked
        # nothing, so REMOTE_RETRIES_DISABLED would be a false report.
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=False, max_retries=0
        )

        assert result.status == "infeasible"
        assert len(result.attempts) == 1
        assert fake.sample_calls == 1
        assert result.warnings == []
        assert "server policy" not in (result.message or "")

    def test_warning_only_when_policy_actually_cut_retries(self, monkeypatch):
        _, cut = self.solve(monkeypatch, allow_remote_retries=False, max_retries=1)
        _, not_cut = self.solve(monkeypatch, allow_remote_retries=False, max_retries=0)

        assert [warning.code for warning in cut.warnings] == ["REMOTE_RETRIES_DISABLED"]
        assert not_cut.warnings == []

    def test_over_the_remote_retry_ceiling_never_reaches_the_sampler(self, monkeypatch):
        # 2026-09-09 review (F-07): the opt-in enables retries, it does not
        # raise their ceiling. 4 is over the default 3, so nothing is sampled.
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=True, max_retries=4
        )

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["RETRY_LIMIT"]
        assert "4" in result.errors[0].message
        assert result.solutions == []
        assert fake.sample_calls == 0

    def test_exactly_the_remote_retry_ceiling_is_allowed(self, monkeypatch):
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=True, max_retries=3
        )

        assert result.status == "infeasible"
        assert len(result.attempts) == 4
        assert fake.sample_calls == 4

    def test_ceiling_applies_even_when_retries_are_off(self, monkeypatch):
        # The ceiling is a bound on the requested value, not on what the
        # policy would actually run: the opt-in flag is irrelevant to it.
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=False, max_retries=4
        )

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["RETRY_LIMIT"]
        assert fake.sample_calls == 0


class FailAfterFirstQPUSampler(FakeQPUSampler):
    """Returns its assignments on the first call and raises on every later one."""

    def __init__(self, failure: Exception, **kwargs) -> None:
        super().__init__(**kwargs)
        self.failure = failure

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        if self.sample_calls >= 1:
            self.raise_on_sample = self.failure
        return super().sample(bqm, **kwargs)


class TestSolverErrorAfterACompletedAttempt:
    """A failure on a retry must not discard the completed attempt's metadata:
    quota was spent on it and the caller needs to see that."""

    def solve(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FailAfterFirstQPUSampler(
            RequestTimeout("remote solve timed out"),
            assignments=[{"a": 0, "b": 0}],
            info=FAKE_MINIMAL_SAMPLESET_INFO,
        )
        service = make_qpu_service(
            fake, allow_remote=True, allow_remote_retries=True
        )
        return fake, service.solve(make_zero_infeasible_problem(max_retries=1))

    def test_second_attempt_failure_is_a_solver_error(self, monkeypatch):
        fake, result = self.solve(monkeypatch)

        assert fake.sample_calls == 2
        assert result.status == "solver_error"
        assert result.errors[0].code == "REMOTE_TIMEOUT"
        assert [attempt.attempt for attempt in result.attempts] == [1]

    def test_first_attempt_metadata_is_kept(self, monkeypatch):
        _, result = self.solve(monkeypatch)

        assert result.metadata is not None
        assert result.metadata.backend == "dwave_qpu"
        assert result.metadata.timing_us == {"qpu_access_time": 12345.0}

    def test_failure_on_the_first_attempt_has_no_metadata(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(raise_on_sample=RequestTimeout("timed out"))
        service = make_qpu_service(fake, allow_remote=True)

        result = service.solve(make_problem())

        assert result.status == "solver_error"
        assert result.metadata is None


class TestSolverError:
    def solve(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(raise_on_sample=RequestTimeout("remote solve timed out"))
        service = make_qpu_service(fake, allow_remote=True)
        return service.solve(make_problem())

    def test_timeout_is_reported_as_a_retryable_solver_error(self, monkeypatch):
        result = self.solve(monkeypatch)

        assert result.status == "solver_error"
        assert result.backend == "dwave_qpu"
        assert result.solutions == []
        assert len(result.errors) == 1
        assert result.errors[0].code == "REMOTE_TIMEOUT"
        assert result.errors[0].retryable is True
        assert_actions_present(result)

    def test_message_is_non_empty(self, monkeypatch):
        result = self.solve(monkeypatch)

        assert result.message
        assert result.errors[0].message


class TestConcurrencyLimit:
    """§14 step 6: a full server rejects immediately instead of queueing."""

    def test_second_solve_is_rejected_while_a_slot_is_held(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        started = threading.Event()
        release = threading.Event()
        fake = BlockingQPUSampler(started=started, release=release)
        service = make_qpu_service(fake, allow_remote=True, max_concurrent_solves=1)
        first: dict[str, object] = {}

        def run_first() -> None:
            first["result"] = service.solve(make_problem())

        thread = threading.Thread(target=run_first, name="first-solve")
        thread.start()
        try:
            assert started.wait(timeout=10)
            rejected = service.solve(make_problem())
        finally:
            release.set()
            thread.join(timeout=10)

        assert rejected.status == "resource_limit_exceeded"
        assert rejected.solutions == []
        assert len(rejected.errors) == 1
        assert rejected.errors[0].code == "CONCURRENCY_LIMIT"
        assert rejected.errors[0].retryable is True
        assert_actions_present(rejected)

        assert not thread.is_alive()
        assert first["result"].status == "success"

        # The slot the first solve held is back: a later solve succeeds.
        assert service.solve(make_problem()).status == "success"


class TestSlotReleasedOnFailure:
    def test_slot_survives_a_solver_error(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(raise_on_sample=RequestTimeout("boom"))
        service = make_qpu_service(fake, allow_remote=True, max_concurrent_solves=1)

        failed = service.solve(make_problem())
        assert failed.status == "solver_error"

        fake.raise_on_sample = None
        recovered = service.solve(make_problem())

        assert [error.code for error in recovered.errors] == []
        assert recovered.status == "success"
        assert recovered.solutions
