"""End-to-end service tests for the remote-backend flow (Phase 2 spec §14).

Everything goes through :meth:`OptimizationService.solve`; no internal
helper is called directly. Real D-Wave is never touched: fake samplers are
injected through each backend's ``sampler_factory`` seam, and availability
is forced by patching the backend module's ``_dwave_system_installed`` /
``ocean_config_status`` (the ``dwave`` extra is not installed in mock CI,
so the real ``is_available()`` would report "dwave-system not installed").
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
    DWaveQPUBackend,
    ExactSolverBackend,
    LeapHybridBQMBackend,
    SimulatedAnnealingBackend,
    SolverCapabilities,
    SolverRegistry,
)
import annealbridge.solvers.dwave_qpu as qpu_module
import annealbridge.solvers.leap_hybrid_bqm as leap_module

FAKE_MIN_TIME_LIMIT = 3.0

# Whitelisted timing keys only; the point here is the service flow, the
# sanitization itself is covered by the per-backend mock tests.
FAKE_SAMPLESET_INFO = {"timing": {"qpu_access_time": 12345}}


class RequestTimeout(Exception):
    """Fake of dwave.cloud's RequestTimeout (classified as REMOTE_TIMEOUT)."""


def expand(bqm, assignment: dict[str, int]) -> dict:
    """Widen a business assignment over every compiled variable (slack = 0)."""
    return {
        variable: int(assignment.get(str(variable), 0)) for variable in bqm.variables
    }


class FakeQPUSampler:
    """Fake with the EmbeddingComposite surface the QPU backend touches.

    Returns exactly the caller-supplied business assignments, so a test can
    decide whether the service sees feasible samples or none at all.
    """

    def __init__(
        self,
        assignments: list[dict[str, int]] | None = None,
        raise_on_sample: Exception | None = None,
    ) -> None:
        self.assignments = (
            [{"a": 1, "b": 0}] if assignments is None else assignments
        )
        self.raise_on_sample = raise_on_sample
        self.sample_calls = 0
        self.sample_bqm = None
        self.sample_kwargs = None

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.sample_calls += 1
        self.sample_bqm = bqm
        self.sample_kwargs = kwargs
        if self.raise_on_sample is not None:
            raise self.raise_on_sample
        rows = [expand(bqm, assignment) for assignment in self.assignments]
        return dimod.SampleSet.from_samples(
            rows,
            vartype=dimod.BINARY,
            energy=[bqm.energy(row) for row in rows],
            info=dict(FAKE_SAMPLESET_INFO),
        )


class BlockingQPUSampler(FakeQPUSampler):
    """QPU fake that parks inside ``sample()`` until it is released."""

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.started = started
        self.release = release

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.started.set()
        assert self.release.wait(timeout=10)
        return super().sample(bqm, **kwargs)


class FakeLeapHybridSampler:
    """Fake with the LeapHybridSampler surface the hybrid backend touches."""

    def __init__(self, assignments: list[dict[str, int]] | None = None) -> None:
        self.assignments = (
            [{"a": 1, "b": 0}] if assignments is None else assignments
        )
        self.sample_calls = 0
        self.sample_bqm = None
        self.sample_kwargs = None

    def min_time_limit(self, bqm) -> float:
        return FAKE_MIN_TIME_LIMIT

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.sample_calls += 1
        self.sample_bqm = bqm
        self.sample_kwargs = kwargs
        rows = [expand(bqm, assignment) for assignment in self.assignments]
        return dimod.SampleSet.from_samples(
            rows,
            vartype=dimod.BINARY,
            energy=[bqm.energy(row) for row in rows],
            info=dict(FAKE_SAMPLESET_INFO),
        )


class FakeUnavailableBackend:
    """A remote backend that reports itself unavailable for a given reason.

    Implements the ``SolverBackend`` protocol; ``solve()`` fails loudly so a
    test proves the service never reaches it.
    """

    def __init__(self, availability: tuple[bool, str | None]) -> None:
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

    def is_available(self) -> tuple[bool, str | None]:
        return self._availability

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return self.capabilities.exhaustive

    def solve(self, compiled_problem, preferences):
        self.solve_calls += 1
        raise AssertionError("solve() must not be reached for an unavailable backend")


def make_remote_available(monkeypatch, module) -> None:
    """Force ``module``'s backend to report itself installed and configured."""
    monkeypatch.setattr(module, "_dwave_system_installed", lambda: True)
    monkeypatch.setattr(module, "ocean_config_status", lambda: "ok")


def make_problem(backend: str = "dwave_qpu", **solver_overrides) -> OptimizationProblem:
    """maximize 2a + b s.t. a + b <= 1; the ``<=`` adds an internal slack var."""
    return OptimizationProblem.model_validate(
        {
            "name": "service remote flow problem",
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
            "solver": {"backend": backend, **solver_overrides},
        }
    )


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
        fake = FakeQPUSampler()
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
        fake = FakeQPUSampler()
        service = make_qpu_service(fake, allow_remote=False)

        service.solve(make_problem())

        assert fake.sample_calls == 0
        assert fake.sample_bqm is None

    def test_no_availability_patching_is_required(self):
        # The gate fires even though the real is_available() would report
        # "dwave-system not installed": REMOTE_DISABLED wins because it is
        # checked first.
        assert DWaveQPUBackend().is_available() == (False, "dwave-system not installed")

        result = make_qpu_service(FakeQPUSampler(), allow_remote=False).solve(
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


class TestAvailabilityClassification:
    """§14 step 5: is_available() reasons map to categorical codes."""

    @pytest.mark.parametrize(
        ("availability", "expected_status", "expected_code"),
        [
            (
                (False, "dwave-system not installed"),
                "backend_unavailable",
                "BACKEND_NOT_INSTALLED",
            ),
            (
                (False, "D-Wave credentials not configured"),
                "backend_unavailable",
                "REMOTE_CREDENTIALS_MISSING",
            ),
            (
                (False, "D-Wave configuration invalid"),
                "configuration_error",
                "DWAVE_CONFIG_INVALID",
            ),
            (
                (False, "some new reason"),
                "backend_unavailable",
                "BACKEND_UNAVAILABLE",
            ),
            ((False, None), "backend_unavailable", "BACKEND_UNAVAILABLE"),
        ],
        ids=["not-installed", "credentials", "invalid-config", "unknown", "no-reason"],
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
        assert backend.solve_calls == 0
        assert_actions_present(result)

    def test_unreported_reason_still_produces_a_message(self):
        service = make_service(
            FakeUnavailableBackend((False, None)), "dwave_qpu", allow_remote=True
        )

        result = service.solve(make_problem())

        assert result.message
        assert "fake_remote" in result.errors[0].message


class TestPreferenceLimits:
    """§14 step 9: preferences are refused, never clamped, before solving."""

    def test_num_reads_over_the_limit_is_refused(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler()
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
            FakeQPUSampler(), allow_remote=True, max_qpu_reads=10
        )

        message = service.solve(make_problem(num_reads=100)).errors[0].message

        assert "100" in message
        assert "10" in message
        assert "num_reads 100" in message
        assert "maximum of 10" in message

    def test_num_reads_over_the_limit_never_reaches_the_sampler(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler()
        service = make_qpu_service(fake, allow_remote=True, max_qpu_reads=10)

        service.solve(make_problem(num_reads=100))

        assert fake.sample_bqm is None
        assert fake.sample_calls == 0

    def test_annealing_time_over_the_limit_is_refused(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler()
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
        fake = FakeLeapHybridSampler()
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
        fake = FakeQPUSampler()
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
        fake = FakeQPUSampler()
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


class TestSuccessfulRemoteSolve:
    def solve(self, monkeypatch):
        make_remote_available(monkeypatch, qpu_module)
        fake = FakeQPUSampler(assignments=[{"a": 0, "b": 0}, {"a": 1, "b": 0}])
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
