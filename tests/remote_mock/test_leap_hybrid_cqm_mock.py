"""Mock tests for LeapHybridCQMBackend (Phase 3a spec §17, §26.5).

Never talks to real D-Wave: a fake sampler is injected through the
backend's ``sampler_factory`` seam and returns a real ``dimod.SampleSet``
built with ``from_samples_cqm`` (so it carries ``is_feasible``) whose
``info`` is hybrid-shaped: whitelist timing at the top level plus dirty
keys that must not survive sanitization.

The first half drives the backend directly; the second half goes through
:meth:`OptimizationService.solve` so the CQM path is exercised end to end:
compiler choice from the declaration, the policy gates, the one-attempt
rule, and — above all — that the sampler's feasibility verdict is only
reported, never trusted (overview principle 2).
"""

import sys
import traceback
import types

import dimod
import numpy as np
import pytest

from annealbridge.compiler import CQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    LeapHybridCQMOptions,
    OptimizationProblem,
    SolverPreferences,
)
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.solvers import (
    AvailabilityStatus,
    LeapHybridCQMBackend,
    SolverCapabilities,
    SolverRegistry,
)
import annealbridge.solvers.leap_hybrid_cqm as cqm_module
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
    FakeCQMSampler,
    RequestTimeout,
    SolverAuthenticationError,
    SolverFailureError,
    SolverNotFoundError,
    ValidationError,
    make_problem,
    make_remote_available,
)

BACKEND = "leap_hybrid_cqm"

# The optimum of "maximize 2a + b s.t. a + b <= 1" (make_problem).
BEST = {"a": 1, "b": 0}
# Violates the hard constraint a + b <= 1.
VIOLATOR = {"a": 1, "b": 1}


def make_cqm_problem(**solver_overrides) -> OptimizationProblem:
    return make_problem(backend=BACKEND, **solver_overrides)


def make_compiled_problem() -> CompiledProblem:
    """The CQM the service would build: hard constraint native, no penalty."""
    return CQMCompiler().compile(make_cqm_problem(), hard_penalty=None)


def make_preferences(time_limit_seconds: float | None = None) -> SolverPreferences:
    return SolverPreferences(
        backend=BACKEND,
        leap_hybrid_cqm=LeapHybridCQMOptions(time_limit_seconds=time_limit_seconds),
    )


def solve_with_fake(
    preferences: SolverPreferences,
    fake: FakeCQMSampler | None = None,
) -> tuple[FakeCQMSampler, CompiledProblem, object]:
    fake = fake if fake is not None else FakeCQMSampler()
    backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)
    compiled = make_compiled_problem()
    result = backend.solve(compiled, preferences)
    return fake, compiled, result


def make_service(fake: FakeCQMSampler, **policy_kwargs) -> OptimizationService:
    """A service whose registry holds exactly the CQM backend over ``fake``."""
    backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)
    return OptimizationService(
        registry=SolverRegistry({BACKEND: backend}),
        policy=ExecutionPolicy(**policy_kwargs),
    )


def assert_actions_present(result) -> None:
    """Every structured error/warning carries catalog guidance (spec §13.2)."""
    for entry in [*result.errors, *result.warnings]:
        assert entry.recommended_action is not None
        assert entry.recommended_action != ""


# ---------------------------------------------------------------------------
# Backend level
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_capability_fields(self):
        capabilities = LeapHybridCQMBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == BACKEND
        assert capabilities.remote is True
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is False
        assert capabilities.supports_num_reads is False
        assert capabilities.supports_time_limit is True
        assert capabilities.supports_num_sweeps is False
        assert capabilities.requires_embedding is False
        assert capabilities.supported_model_types == ["cqm"]
        assert capabilities.preferred_model_type == "cqm"
        # §17.1: the CQM hybrid solver returns several samples, so top_k
        # is meaningful (unlike the BQM hybrid solver).
        assert capabilities.returns_multiple_samples is True

    def test_parameter_limit_declaration(self):
        # §7.1: the time limit is a declaration, and the preference path
        # names the option block (== backend name, §9.3 naming contract).
        capabilities = LeapHybridCQMBackend().capabilities
        assert [
            (limit.preference, limit.limit, limit.error_code)
            for limit in capabilities.parameter_limits
        ] == [("leap_hybrid_cqm.time_limit_seconds", "time_seconds", "REMOTE_TIME_LIMIT")]
        block, field = capabilities.parameter_limits[0].preference.split(".")
        assert block == capabilities.name
        assert field in LeapHybridCQMOptions.model_fields

    def test_description_states_leap_limits_and_ignored_flag(self):
        description = LeapHybridCQMBackend().capabilities.description
        assert "feasibility flag is ignored" in description
        assert "5,000,000 variables" in description
        assert "100,000 constraints" in description
        assert "minimum time_limit 5 s" in description
        assert "no penalty, no slack" in description

    def test_properties_alias_capabilities(self):
        backend = LeapHybridCQMBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive


class TestLazyImport:
    def test_construction_does_not_import_dwave(self):
        LeapHybridCQMBackend()
        assert "dwave.system" not in sys.modules

    def test_module_has_no_dwave_import_at_module_level(self):
        # §17.5 / §4: dwave.* only inside the default sampler factory.
        imported = [
            value.__name__
            for value in vars(cqm_module).values()
            if isinstance(value, types.ModuleType)
        ]
        assert not any(name.startswith("dwave") for name in imported)


class TestModelForwarding:
    def test_the_compiled_cqm_is_forwarded_unchanged(self):
        fake, compiled, _ = solve_with_fake(make_preferences(None))

        assert fake.sample_cqm_model is compiled.model
        assert isinstance(fake.sample_cqm_model, dimod.ConstrainedQuadraticModel)

    def test_hard_constraint_arrives_native_with_no_weight(self):
        fake, compiled, _ = solve_with_fake(make_preferences(None))

        cqm = fake.sample_cqm_model
        assert set(cqm.constraints) == {"at_most_one"}
        # dimod keeps ``weight=None`` (must be satisfied) as "not soft".
        assert "at_most_one" not in cqm._soft
        assert cqm.num_soft_constraints() == 0
        assert compiled.hard_penalty is None
        assert compiled.internal_variables == set()
        assert set(map(str, cqm.variables)) == {"a", "b"}

    def test_only_time_limit_is_forwarded(self):
        # §17.2: no label (problem name is business data), no num_reads,
        # no num_sweeps, no seed, no penalty_multiplier.
        preferences = SolverPreferences(
            backend=BACKEND,
            num_reads=7,
            num_sweeps=50,
            seed=42,
            penalty_multiplier=9.0,
            leap_hybrid_cqm=LeapHybridCQMOptions(time_limit_seconds=5.0),
        )

        fake, _, _ = solve_with_fake(preferences)

        assert fake.sample_kwargs == {"time_limit": 5.0}
        assert "label" not in fake.sample_kwargs
        assert "num_reads" not in fake.sample_kwargs


class TestTimeLimitForwarding:
    def test_no_user_time_limit_uses_sampler_minimum(self):
        fake, compiled, result = solve_with_fake(make_preferences(None))

        assert fake.sample_kwargs == {"time_limit": FAKE_MIN_TIME_LIMIT}
        assert fake.min_time_limit_cqm is compiled.model
        assert result.metadata.effective_time_limit_seconds == FAKE_MIN_TIME_LIMIT

    def test_options_object_missing_entirely_uses_sampler_minimum(self):
        preferences = SolverPreferences(backend=BACKEND)
        assert preferences.leap_hybrid_cqm is None

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


class TestResolveTimeLimit:
    """§17.2 / §20: the service learns the effective time limit *before*
    anything is submitted, and solve() submits exactly that number."""

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
        fake = FakeCQMSampler()
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)
        compiled = make_compiled_problem()

        resolved = backend.resolve_time_limit(compiled, make_preferences(user_time_limit))

        assert resolved == expected
        assert fake.sample_calls == 0
        assert fake.min_time_limit_cqm is compiled.model

    @pytest.mark.parametrize(
        "user_time_limit", [None, 10.0, 1.0], ids=["none", "above", "below"]
    )
    def test_solve_submits_exactly_the_resolved_time_limit(self, user_time_limit):
        fake = FakeCQMSampler()
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)
        compiled = make_compiled_problem()
        preferences = make_preferences(user_time_limit)

        result = backend.solve(compiled, preferences)

        expected = backend.resolve_time_limit(compiled, preferences)
        assert result.metadata.effective_time_limit_seconds == expected
        assert fake.sample_kwargs == {"time_limit": expected}

    def test_min_time_limit_failure_is_classified_and_redacted(self):
        fake = FakeCQMSampler(
            raise_on_min_time_limit=RuntimeError(f"exploded, token={FAKE_TOKEN}")
        )
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.resolve_time_limit(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message
        assert fake.sample_calls == 0

    def test_factory_failure_during_resolve_is_config_invalid(self):
        backend = LeapHybridCQMBackend(
            sampler_factory=CountingFactory(
                FakeCQMSampler(),
                failures=[ValueError(f"invalid region, token={FAKE_TOKEN}")],
            )
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.resolve_time_limit(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "DWAVE_CONFIG_INVALID"
        assert FAKE_TOKEN not in str(exc_info.value)
        assert "***" in str(exc_info.value)


class TestSampleSetConversion:
    def test_every_sample_is_returned_in_sampler_order(self):
        fake = FakeCQMSampler(assignments=[VIOLATOR, BEST, {"a": 0, "b": 0}])

        _, _, result = solve_with_fake(make_preferences(None), fake=fake)

        assert result.backend == BACKEND
        assert result.as_dicts() == [VIOLATOR, BEST, {"a": 0, "b": 0}]
        assert result.samples.dtype == np.int64
        assert result.num_samples == 3

    def test_energies_are_the_cqm_energies(self):
        fake = FakeCQMSampler(assignments=[BEST, {"a": 0, "b": 0}])

        _, compiled, result = solve_with_fake(make_preferences(None), fake=fake)

        # maximize 2a + b is compiled as minimize -(2a + b).
        assert result.energies.tolist() == [
            pytest.approx(compiled.model.objective.energy(BEST)),
            pytest.approx(compiled.model.objective.energy({"a": 0, "b": 0})),
        ]
        assert result.energies[0] == pytest.approx(-2.0)

    def test_infeasible_flagged_samples_are_not_filtered(self):
        # §17.3 / §29: never ``sampleset.filter(lambda d: d.is_feasible)``.
        fake = FakeCQMSampler(assignments=[VIOLATOR, BEST])

        _, _, result = solve_with_fake(make_preferences(None), fake=fake)

        assert result.as_dicts() == [VIOLATOR, BEST]
        assert result.metadata.sampler_reported_feasible == 1

    def test_samples_are_not_reordered_by_feasibility(self):
        fake = FakeCQMSampler(assignments=[VIOLATOR, {"a": 0, "b": 0}, BEST])

        _, _, result = solve_with_fake(make_preferences(None), fake=fake)

        assert result.as_dicts()[0] == VIOLATOR

    def test_sampler_reported_feasible_counts_the_flag_verbatim(self):
        # The fake lies: the violator is flagged feasible. The count reports
        # what the sampler said, nothing is re-judged here (§22).
        fake = FakeCQMSampler(
            assignments=[VIOLATOR, BEST], feasible_flags=[True, True]
        )

        _, _, result = solve_with_fake(make_preferences(None), fake=fake)

        assert result.metadata.sampler_reported_feasible == 2
        assert result.as_dicts() == [VIOLATOR, BEST]

    def test_missing_is_feasible_field_reports_none(self):
        class NoFlagSampler(FakeCQMSampler):
            def _build_sampleset(self, cqm):
                rows = self._rows(cqm)
                return dimod.SampleSet.from_samples(
                    rows,
                    vartype=dimod.BINARY,
                    energy=[cqm.objective.energy(row) for row in rows],
                    info=dict(self.info),
                )

        fake = NoFlagSampler(assignments=[BEST])

        _, _, result = solve_with_fake(make_preferences(None), fake=fake)

        assert result.metadata.sampler_reported_feasible is None
        assert result.as_dicts() == [BEST]

    def test_float_zero_one_samples_are_accepted(self):
        # Leap may return float-typed samples (§17.3); 0.0 / 1.0 are fine.
        fake = FakeCQMSampler(assignments=[{"a": 1.0, "b": 0.0}])

        _, _, result = solve_with_fake(make_preferences(None), fake=fake)

        assert result.as_dicts() == [BEST]
        assert result.samples.dtype == np.int64

    @pytest.mark.parametrize(
        "bad_row",
        [{"a": 2, "b": 1}, {"a": 0.5, "b": 0.0}, {"a": -1, "b": 0}],
        ids=["two", "half", "minus-one"],
    )
    def test_non_binary_value_is_a_remote_solver_error(self, bad_row):
        # §17.3 / 3b §11: the bounds assertion runs in the backend before
        # the shared conversion (whose int64 cast would otherwise truncate
        # silently). from_samples_cqm keeps the values as given (int8 for
        # 2 / -1, float64 for 0.5).
        fake = FakeCQMSampler(assignments=[BEST, bad_row])
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"
        assert "outside its bounds" in str(exc_info.value)
        assert fake.sample_calls == 1


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
        # from_samples_cqm adds ``constraint_labels`` to info; it is not
        # whitelisted and must not survive either.
        assert "constraint_labels" not in dumped
        assert "at_most_one" not in dumped

    def test_backend_and_remote_flags(self):
        _, _, result = solve_with_fake(make_preferences(None))

        assert result.metadata.backend == BACKEND
        assert result.metadata.remote is True

    def test_model_type_is_left_for_the_service(self):
        # §22: the backend never knows the model type; the service fills it.
        _, _, result = solve_with_fake(make_preferences(None))

        assert result.metadata.model_type is None


class TestExceptionClassification:
    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (SolverAuthenticationError(f"invalid token={FAKE_TOKEN}"), "REMOTE_AUTH_FAILED"),
            (RequestTimeout(f"timed out, token={FAKE_TOKEN}"), "REMOTE_TIMEOUT"),
            (SolverFailureError(f"rejected, token={FAKE_TOKEN}"), "REMOTE_SOLVER_ERROR"),
            (RuntimeError(f"solver exploded, token={FAKE_TOKEN}"), "REMOTE_SOLVER_ERROR"),
        ],
        ids=["auth", "timeout", "solver-failure", "other"],
    )
    def test_sample_exceptions_map_to_codes_and_are_redacted(
        self, exception, expected_code
    ):
        fake = FakeCQMSampler(raise_on_sample=exception)
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message

    def test_sample_stage_value_error_is_not_an_embedding_failure(self):
        # The hybrid table has no ValueError entry: a CQM never embeds.
        fake = FakeCQMSampler(raise_on_sample=ValueError("solver rejected the model"))
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "REMOTE_SOLVER_ERROR"

    def test_subclass_of_auth_error_is_classified_via_mro(self):
        class DerivedAuthError(SolverAuthenticationError):
            pass

        fake = FakeCQMSampler(raise_on_sample=DerivedAuthError("denied"))
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == "REMOTE_AUTH_FAILED"


class TestSamplerInitExceptionClassification:
    """§17.4: sampler construction has its own table — ``SolverNotFoundError``
    / ``ConfigFileError`` / ``ValueError`` there mean invalid configuration."""

    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (
                SolverNotFoundError(f"no solver matches, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
            ),
            (
                ConfigFileError(f"bad dwave.conf, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
            ),
            (ValueError(f"invalid region, token={FAKE_TOKEN}"), "DWAVE_CONFIG_INVALID"),
            (
                ValidationError(f"1 validation error, token={FAKE_TOKEN}"),
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
            "solver_not_found",
            "config_file_error",
            "value_error",
            "validation_error",
            "auth",
            "timeout",
            "other",
        ],
    )
    def test_factory_exceptions_map_to_codes_and_are_redacted(
        self, exception, expected_code
    ):
        backend = LeapHybridCQMBackend(
            sampler_factory=CountingFactory(FakeCQMSampler(), failures=[exception])
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message


class TestLazySampleSetResolution:
    """The backend resolves the sampleset inside the guarded call, so a
    failure that only surfaces on resolve is still classified and redacted."""

    def test_lazy_success_matches_the_eager_result(self):
        _, _, expected = solve_with_fake(make_preferences(None))
        lazy, _, result = solve_with_fake(
            make_preferences(None), fake=FakeCQMSampler(lazy=True)
        )

        assert lazy.sample_calls == 1
        assert result.variables == expected.variables
        assert result.samples.tolist() == expected.samples.tolist()
        assert result.energies.tolist() == expected.energies.tolist()
        assert result.metadata.model_dump() == expected.metadata.model_dump()

    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (RequestTimeout(f"timed out, token={FAKE_TOKEN}"), "REMOTE_TIMEOUT"),
            (
                SolverFailureError(f"solver failed, token={FAKE_TOKEN}"),
                "REMOTE_SOLVER_ERROR",
            ),
        ],
        ids=["timeout", "solver_failure"],
    )
    def test_failure_on_resolve_is_classified_and_redacted(
        self, exception, expected_code
    ):
        fake = FakeCQMSampler(lazy=True, raise_on_sample=exception)
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message
        assert fake.sample_calls == 1
        assert fake.sample_kwargs == {"time_limit": FAKE_MIN_TIME_LIMIT}


class TestOriginalExceptionIsNotReachable:
    """Phase 2 §19: no ``__cause__`` / ``__context__`` back to the raw error."""

    def test_sample_failure_has_no_cause_or_context(self):
        fake = FakeCQMSampler(
            raise_on_sample=RuntimeError(f"solver exploded, token={FAKE_TOKEN}")
        )
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        assert FAKE_TOKEN not in "".join(traceback.format_exception(error))

    def test_factory_failure_has_no_cause_or_context(self):
        backend = LeapHybridCQMBackend(
            sampler_factory=CountingFactory(
                FakeCQMSampler(),
                failures=[SolverNotFoundError(f"Authorization: Bearer {FAKE_TOKEN}")],
            )
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences(None))

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        assert FAKE_TOKEN not in "".join(traceback.format_exception(error))


class TestSamplerCaching:
    """§17.5 ``LazySampler``: built once per backend, failures never cached."""

    def test_sampler_is_built_once_per_backend(self):
        factory = CountingFactory(FakeCQMSampler())
        backend = LeapHybridCQMBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        backend.solve(compiled, make_preferences(None))
        backend.solve(compiled, make_preferences(None))

        assert factory.calls == 1
        assert factory.sampler.sample_calls == 2

    def test_failed_construction_is_not_cached(self):
        factory = CountingFactory(
            FakeCQMSampler(),
            failures=[SolverAuthenticationError(f"denied, token={FAKE_TOKEN}")],
        )
        backend = LeapHybridCQMBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compiled, make_preferences(None))
        assert exc_info.value.code == "REMOTE_AUTH_FAILED"

        result = backend.solve(compiled, make_preferences(None))

        assert factory.calls == 2
        assert result.backend == BACKEND
        assert factory.sampler.sample_calls == 1

    def test_each_backend_instance_builds_its_own_sampler(self):
        factory = CountingFactory(FakeCQMSampler())
        compiled = make_compiled_problem()

        LeapHybridCQMBackend(sampler_factory=factory).solve(
            compiled, make_preferences(None)
        )
        LeapHybridCQMBackend(sampler_factory=factory).solve(
            compiled, make_preferences(None)
        )

        assert factory.calls == 2

    def test_auth_failure_at_sampling_invalidates_the_cache(self):
        """2026-09-09 review F-21: a rejected client is known-bad, so the
        guard that classifies ``REMOTE_AUTH_FAILED`` also drops the cached
        sampler; the next solve builds a fresh one."""
        sampler = FakeCQMSampler(
            raise_on_sample=SolverAuthenticationError(f"denied, token={FAKE_TOKEN}")
        )
        factory = CountingFactory(sampler)
        backend = LeapHybridCQMBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compiled, make_preferences(None))
        assert exc_info.value.code == "REMOTE_AUTH_FAILED"
        assert factory.calls == 1

        sampler.raise_on_sample = None
        result = backend.solve(compiled, make_preferences(None))

        assert factory.calls == 2
        assert result.backend == BACKEND


class TestIsAvailable:
    """The backend answers through the shared check in solvers.ocean."""

    def test_dwave_system_not_installed(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: False)

        assert LeapHybridCQMBackend().is_available() == AvailabilityStatus(
            category="not_installed", detail=REASON_NOT_INSTALLED
        )

    def test_credentials_not_configured(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "missing")

        assert LeapHybridCQMBackend().is_available() == AvailabilityStatus(
            category="credentials_missing", detail=REASON_CREDENTIALS_MISSING
        )

    def test_configuration_invalid(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "invalid")

        assert LeapHybridCQMBackend().is_available() == AvailabilityStatus(
            category="config_invalid",
            detail=REASON_CONFIG_INVALID,
            error_code="DWAVE_CONFIG_INVALID",
        )

    def test_available_when_installed_and_configured(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "ok")

        assert LeapHybridCQMBackend().is_available() == AvailabilityStatus(
            category="available"
        )


# ---------------------------------------------------------------------------
# Service level: the CQM path end to end through OptimizationService.solve
# ---------------------------------------------------------------------------


class TestRemoteDisabledByPolicy:
    def test_refused_with_remote_disabled_before_touching_the_sampler(self):
        fake = FakeCQMSampler(assignments=[BEST])
        service = make_service(fake)  # allow_remote defaults to False

        result = service.solve(make_cqm_problem())

        assert result.status == "backend_unavailable"
        assert result.backend == BACKEND
        assert result.solutions == []
        assert [error.code for error in result.errors] == ["REMOTE_DISABLED"]
        assert fake.sample_calls == 0
        assert fake.min_time_limit_calls == 0
        assert_actions_present(result)


class TestSuccessfulSolve:
    def solve(self, monkeypatch, **solver_overrides):
        make_remote_available(monkeypatch, cqm_module)
        fake = FakeCQMSampler(assignments=[{"a": 0, "b": 0}, BEST, {"a": 0, "b": 1}])
        service = make_service(fake, allow_remote=True)
        return fake, service.solve(make_cqm_problem(**solver_overrides))

    def test_status_backend_and_ranking(self, monkeypatch):
        fake, result = self.solve(monkeypatch)

        assert result.status == "success"
        assert result.backend == BACKEND
        assert fake.sample_calls == 1
        # Ranking uses the business objective, not the sampler order.
        assert result.solutions[0].variables == BEST
        assert result.solutions[0].objective_value == 2.0
        assert [solution.variables for solution in result.solutions] == [
            BEST,
            {"a": 0, "b": 1},
            {"a": 0, "b": 0},
        ]
        assert result.warnings == []

    def test_metadata_model_type_is_cqm(self, monkeypatch):
        _, result = self.solve(monkeypatch)

        assert result.metadata is not None
        assert result.metadata.model_type == "cqm"
        assert result.metadata.backend == BACKEND
        assert result.metadata.remote is True
        assert result.metadata.sampler_reported_feasible == 3
        assert result.metadata.effective_time_limit_seconds == FAKE_MIN_TIME_LIMIT

    def test_exactly_one_attempt_with_no_penalty(self, monkeypatch):
        _, result = self.solve(monkeypatch, max_retries=3)

        assert len(result.attempts) == 1
        assert result.attempts[0].penalty is None
        assert result.attempts[0].feasible_samples == 3

    def test_hard_constraint_is_native_in_the_submitted_model(self, monkeypatch):
        fake, _ = self.solve(monkeypatch)

        cqm = fake.sample_cqm_model
        assert isinstance(cqm, dimod.ConstrainedQuadraticModel)
        assert "at_most_one" not in cqm._soft
        assert cqm.num_soft_constraints() == 0
        # No slack: exactly the business variables.
        assert set(map(str, cqm.variables)) == {"a", "b"}
        assert all("__" not in name for name in result_variables(fake))

    def test_no_label_and_no_num_reads_reach_the_sampler(self, monkeypatch):
        fake, _ = self.solve(monkeypatch, num_reads=42)

        assert set(fake.sample_kwargs) == {"time_limit"}


def result_variables(fake: FakeCQMSampler) -> list[str]:
    return [str(variable) for variable in fake.sample_cqm_model.variables]


class TestTimeLimitPolicy:
    """§17.2 / §26.5: the time limit is forwarded as resolved, floored at the
    sampler minimum, and refused — never clamped — above policy."""

    def make(self, monkeypatch, *, minimum: float, maximum: int, **solver_overrides):
        make_remote_available(monkeypatch, cqm_module)
        fake = FakeCQMSampler(assignments=[BEST], min_time_limit=minimum)
        service = make_service(
            fake, allow_remote=True, max_remote_time_seconds=maximum
        )
        return fake, service.solve(make_cqm_problem(**solver_overrides))

    def test_forwarded_time_limit_equals_resolve_time_limit(self, monkeypatch):
        make_remote_available(monkeypatch, cqm_module)
        fake = FakeCQMSampler(assignments=[BEST])
        backend = LeapHybridCQMBackend(sampler_factory=lambda: fake)
        service = OptimizationService(
            registry=SolverRegistry({BACKEND: backend}),
            policy=ExecutionPolicy(allow_remote=True),
        )
        problem = make_cqm_problem(
            leap_hybrid_cqm=LeapHybridCQMOptions(time_limit_seconds=7.5)
        )

        result = service.solve(problem)

        expected = backend.resolve_time_limit(
            CQMCompiler().compile(problem, hard_penalty=None), problem.solver
        )
        assert result.status == "success"
        assert fake.sample_kwargs == {"time_limit": expected}
        assert expected == 7.5

    def test_below_minimum_is_raised_and_recorded(self, monkeypatch):
        fake, result = self.make(
            monkeypatch,
            minimum=5.0,
            maximum=10,
            leap_hybrid_cqm=LeapHybridCQMOptions(time_limit_seconds=2.0),
        )

        assert result.status == "success"
        assert fake.sample_kwargs == {"time_limit": 5.0}
        assert result.metadata.effective_time_limit_seconds == 5.0

    def test_user_value_above_policy_is_refused_not_clamped(self, monkeypatch):
        fake, result = self.make(
            monkeypatch,
            minimum=3.0,
            maximum=300,
            leap_hybrid_cqm=LeapHybridCQMOptions(time_limit_seconds=500.0),
        )

        assert result.status == "resource_limit_exceeded"
        assert result.backend == BACKEND
        assert result.solutions == []
        assert [error.code for error in result.errors] == ["REMOTE_TIME_LIMIT"]
        assert "500" in result.errors[0].message
        assert "300" in result.errors[0].message
        assert fake.sample_calls == 0
        assert_actions_present(result)

    def test_effective_value_above_policy_is_refused_not_clamped(self, monkeypatch):
        # The user's 2.0 passes the preference check; the effective 5.0
        # (raised to the sampler minimum) does not, and nothing is sent.
        fake, result = self.make(
            monkeypatch,
            minimum=5.0,
            maximum=3,
            leap_hybrid_cqm=LeapHybridCQMOptions(time_limit_seconds=2.0),
        )

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["REMOTE_TIME_LIMIT"]
        assert "5.0" in result.errors[0].message
        assert "maximum of 3" in result.errors[0].message
        assert fake.sample_calls == 0
        assert fake.min_time_limit_calls >= 1

    def test_omitted_time_limit_with_minimum_above_policy_is_refused(self, monkeypatch):
        fake, result = self.make(monkeypatch, minimum=5.0, maximum=3)

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["REMOTE_TIME_LIMIT"]
        assert fake.sample_calls == 0

    def test_effective_exactly_at_policy_still_solves(self, monkeypatch):
        fake, result = self.make(monkeypatch, minimum=3.0, maximum=3)

        assert result.status == "success"
        assert fake.sample_kwargs == {"time_limit": 3.0}


class TestSamplerVerdictIsNotTrusted:
    """Overview principle 2 / §17.3: a sample flagged feasible by the sampler
    but violating a hard constraint never reaches ``solutions``; the flag
    is only counted into metadata."""

    def solve(self, monkeypatch):
        make_remote_available(monkeypatch, cqm_module)
        fake = FakeCQMSampler(
            assignments=[VIOLATOR, BEST, {"a": 0, "b": 1}],
            feasible_flags=[True, True, True],
        )
        service = make_service(fake, allow_remote=True)
        return fake, service.solve(make_cqm_problem())

    def test_violator_is_excluded_from_solutions(self, monkeypatch):
        _, result = self.solve(monkeypatch)

        assert result.status == "success"
        assert VIOLATOR not in [solution.variables for solution in result.solutions]
        assert [solution.variables for solution in result.solutions] == [
            BEST,
            {"a": 0, "b": 1},
        ]

    def test_sampler_count_is_reported_as_given(self, monkeypatch):
        _, result = self.solve(monkeypatch)

        assert result.metadata.sampler_reported_feasible == 3
        assert result.attempts[0].feasible_samples == 2

    def test_all_flagged_feasible_but_all_violating_is_infeasible(self, monkeypatch):
        make_remote_available(monkeypatch, cqm_module)
        fake = FakeCQMSampler(assignments=[VIOLATOR], feasible_flags=[True])
        service = make_service(fake, allow_remote=True)

        result = service.solve(make_cqm_problem())

        assert result.status == "infeasible"
        assert result.solutions == []
        assert result.infeasibility_proven is False
        assert result.metadata.sampler_reported_feasible == 1


class TestNonBinarySampleThroughService:
    def test_non_binary_value_is_a_structured_solver_error(self, monkeypatch):
        make_remote_available(monkeypatch, cqm_module)
        fake = FakeCQMSampler(assignments=[BEST, {"a": 2, "b": 0}])
        service = make_service(fake, allow_remote=True)

        result = service.solve(make_cqm_problem())

        assert result.status == "solver_error"
        assert result.backend == BACKEND
        assert result.solutions == []
        assert [error.code for error in result.errors] == ["REMOTE_SOLVER_ERROR"]
        assert fake.sample_calls == 1
        assert_actions_present(result)


class TestExceptionClassificationThroughService:
    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (SolverAuthenticationError(f"invalid token={FAKE_TOKEN}"), "REMOTE_AUTH_FAILED"),
            (RequestTimeout(f"timed out, token={FAKE_TOKEN}"), "REMOTE_TIMEOUT"),
            (RuntimeError(f"solver exploded, token={FAKE_TOKEN}"), "REMOTE_SOLVER_ERROR"),
        ],
        ids=["auth", "timeout", "other"],
    )
    def test_sample_failure_is_a_structured_solver_error(
        self, monkeypatch, exception, expected_code
    ):
        make_remote_available(monkeypatch, cqm_module)
        fake = FakeCQMSampler(raise_on_sample=exception)
        service = make_service(fake, allow_remote=True)

        result = service.solve(make_cqm_problem())

        assert result.status == "solver_error"
        assert result.backend == BACKEND
        assert [error.code for error in result.errors] == [expected_code]
        assert FAKE_TOKEN not in result.model_dump_json()
        assert fake.sample_calls == 1
        assert_actions_present(result)

    def test_solver_not_found_at_construction_is_config_invalid(self, monkeypatch):
        make_remote_available(monkeypatch, cqm_module)
        service = OptimizationService(
            registry=SolverRegistry(
                {
                    BACKEND: LeapHybridCQMBackend(
                        sampler_factory=CountingFactory(
                            FakeCQMSampler(),
                            failures=[
                                SolverNotFoundError(f"no CQM solver, token={FAKE_TOKEN}")
                            ],
                        )
                    )
                }
            ),
            policy=ExecutionPolicy(allow_remote=True),
        )

        result = service.solve(make_cqm_problem())

        assert result.status == "solver_error"
        assert [error.code for error in result.errors] == ["DWAVE_CONFIG_INVALID"]
        assert FAKE_TOKEN not in result.model_dump_json()
        assert_actions_present(result)


class TestRetryPolicyDoesNotApply:
    """§16.3 / §26.5: the CQM path has no penalty to retune, so it makes one
    attempt whatever ``allow_remote_retries`` and ``max_retries`` say — and
    never claims policy cut a retry (no ``REMOTE_RETRIES_DISABLED``)."""

    def solve(self, monkeypatch, *, allow_remote_retries: bool, assignments):
        make_remote_available(monkeypatch, cqm_module)
        fake = FakeCQMSampler(assignments=assignments)
        service = make_service(
            fake, allow_remote=True, allow_remote_retries=allow_remote_retries
        )
        return fake, service.solve(make_cqm_problem(max_retries=3))

    @pytest.mark.parametrize("allow_remote_retries", [False, True], ids=["off", "on"])
    def test_infeasible_run_makes_one_attempt_and_no_warning(
        self, monkeypatch, allow_remote_retries
    ):
        fake, result = self.solve(
            monkeypatch,
            allow_remote_retries=allow_remote_retries,
            assignments=[VIOLATOR],
        )

        assert result.status == "infeasible"
        assert result.infeasibility_proven is False
        assert fake.sample_calls == 1
        assert len(result.attempts) == 1
        assert result.attempts[0].penalty is None
        assert "REMOTE_RETRIES_DISABLED" not in [w.code for w in result.warnings]
        assert result.warnings == []

    @pytest.mark.parametrize("allow_remote_retries", [False, True], ids=["off", "on"])
    def test_successful_run_makes_one_attempt_and_no_warning(
        self, monkeypatch, allow_remote_retries
    ):
        fake, result = self.solve(
            monkeypatch, allow_remote_retries=allow_remote_retries, assignments=[BEST]
        )

        assert result.status == "success"
        assert fake.sample_calls == 1
        assert len(result.attempts) == 1
        assert result.attempts[0].penalty is None
        assert "REMOTE_RETRIES_DISABLED" not in [w.code for w in result.warnings]
