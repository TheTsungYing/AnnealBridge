"""Mock tests for DWaveQPUBackend (Phase 2 spec §15, §15.1, §19, §27).

Never talks to real D-Wave: a fake sampler is injected through the
backend's ``sampler_factory`` seam and returns a real ``dimod.SampleSet``
whose ``info`` carries fake QPU timing, an embedding context and dirty
keys that must not survive sanitization.

The fake also reproduces the two Ocean behaviours the backend defends
against: a sampleset that only fails when it is *resolved*
(``SampleSet.from_future``), and an ``embedding_context`` that appears in
``info`` only when ``return_embedding=True`` was actually requested.
"""

import sys
import traceback

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    DWaveQPUOptions,
    SolverPreferences,
)
from annealbridge.solvers import (
    AvailabilityStatus,
    DWaveQPUBackend,
    SolverCapabilities,
)
import annealbridge.solvers.ocean as ocean_module
from annealbridge.solvers.ocean import (
    REASON_CONFIG_INVALID,
    REASON_CREDENTIALS_MISSING,
    REASON_NOT_INSTALLED,
)
from tests.remote_mock.conftest import (
    FAKE_EMBEDDING_CONTEXT,
    FAKE_TOKEN,
    FAKE_UNPATTERNED_TOKEN,
    ConfigFileError,
    CountingFactory,
    EmbeddingError,
    FakeQPUSampler,
    RequestTimeout,
    SolverAuthenticationError,
    SolverFailureError,
    SolverNotFoundError,
    ValidationError,
    make_problem,
)


def make_compiled_problem(hard_penalty: float = 100.0) -> CompiledProblem:
    """Minimal 0/1 problem: maximize 2a + b s.t. a + b <= 1 (adds slack)."""
    return BQMCompiler().compile(
        make_problem(backend="dwave_qpu"), hard_penalty=hard_penalty
    )


def make_preferences(
    num_reads: int = 100,
    annealing_time_us: float | None = None,
    chain_strength: float | None = None,
    auto_scale: bool = True,
) -> SolverPreferences:
    return SolverPreferences(
        backend="dwave_qpu",
        num_reads=num_reads,
        dwave_qpu=DWaveQPUOptions(
            annealing_time_us=annealing_time_us,
            chain_strength=chain_strength,
            auto_scale=auto_scale,
        ),
    )


def solve_with_fake(
    preferences: SolverPreferences,
    fake: FakeQPUSampler | None = None,
    compiled: CompiledProblem | None = None,
) -> tuple[FakeQPUSampler, CompiledProblem, object]:
    fake = fake if fake is not None else FakeQPUSampler()
    backend = DWaveQPUBackend(sampler_factory=lambda: fake)
    compiled = compiled if compiled is not None else make_compiled_problem()
    result = backend.solve(compiled, preferences)
    return fake, compiled, result


class TestCapabilities:
    def test_capability_fields(self):
        capabilities = DWaveQPUBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "dwave_qpu"
        assert capabilities.requires_embedding is True
        # Spec 7.1: the two policy-limited QPU parameters, as declarations.
        assert [
            (limit.preference, limit.limit, limit.error_code)
            for limit in capabilities.parameter_limits
        ] == [
            ("num_reads", "reads", "QPU_READS_LIMIT"),
            ("dwave_qpu.annealing_time_us", "annealing_time_us", "QPU_ANNEALING_TIME_LIMIT"),
        ]
        assert capabilities.supports_num_sweeps is False
        assert capabilities.remote is True
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is False
        assert capabilities.supports_num_reads is True
        assert capabilities.supports_time_limit is False
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is True

    def test_properties_alias_capabilities(self):
        backend = DWaveQPUBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive


class TestLazyImport:
    def test_construction_does_not_import_dwave(self):
        DWaveQPUBackend()
        assert "dwave.system" not in sys.modules


class TestParameterForwarding:
    def test_num_reads_and_auto_scale_always_forwarded(self):
        fake, _, _ = solve_with_fake(SolverPreferences(backend="dwave_qpu"))

        assert fake.sample_kwargs == {
            "num_reads": 100,
            "auto_scale": True,
            "return_embedding": True,
        }

    def test_full_options_forwarded(self):
        fake, _, _ = solve_with_fake(
            make_preferences(
                num_reads=50,
                annealing_time_us=20.0,
                chain_strength=3.5,
                auto_scale=False,
            )
        )

        assert fake.sample_kwargs == {
            "num_reads": 50,
            "annealing_time": 20.0,
            "chain_strength": 3.5,
            "auto_scale": False,
            "return_embedding": True,
        }

    def test_chain_strength_none_is_omitted(self):
        fake, _, _ = solve_with_fake(make_preferences(chain_strength=None))

        assert "chain_strength" not in fake.sample_kwargs

    def test_no_options_object_omits_annealing_time(self):
        preferences = SolverPreferences(backend="dwave_qpu", num_reads=7)
        assert preferences.dwave_qpu is None

        fake, _, _ = solve_with_fake(preferences)

        assert fake.sample_kwargs == {
            "num_reads": 7,
            "auto_scale": True,
            "return_embedding": True,
        }

    def test_bqm_is_forwarded_unchanged(self):
        fake, compiled, _ = solve_with_fake(make_preferences())

        assert fake.sample_bqm is compiled.model

    def test_qpu_meaningless_parameters_are_not_forwarded(self):
        preferences = SolverPreferences(
            backend="dwave_qpu",
            num_reads=25,
            num_sweeps=50,
            seed=42,
            dwave_qpu=DWaveQPUOptions(annealing_time_us=10.0, chain_strength=1.0),
        )

        fake, _, _ = solve_with_fake(preferences)

        assert set(fake.sample_kwargs) == {
            "num_reads",
            "auto_scale",
            "annealing_time",
            "chain_strength",
            "return_embedding",
        }


class TestReturnEmbedding:
    """Spec §15: ``return_embedding=True`` is always requested, because
    EmbeddingComposite otherwise leaves ``info["embedding_context"]`` out."""

    def test_return_embedding_is_always_requested(self):
        fake, _, _ = solve_with_fake(make_preferences())

        assert fake.sample_kwargs["return_embedding"] is True

    def test_fake_omits_embedding_context_without_the_kwarg(self):
        # Guards the fake itself: the embedding context is only reported
        # when asked for, so the backend test above proves a real request.
        fake = FakeQPUSampler()

        sampleset = fake.sample(make_compiled_problem().model)

        assert "embedding_context" not in sampleset.info

    def test_fake_reports_embedding_context_with_the_kwarg(self):
        fake = FakeQPUSampler()

        sampleset = fake.sample(
            make_compiled_problem().model, return_embedding=True
        )

        assert sampleset.info["embedding_context"] == FAKE_EMBEDDING_CONTEXT


class TestChainStrengthIndependence:
    """Spec §15.1: chain strength is never derived from the hard penalty."""

    def test_chain_strength_unchanged_across_penalties(self):
        preferences = make_preferences(chain_strength=2.5)

        for hard_penalty in (100.0, 5000.0):
            fake, _, _ = solve_with_fake(
                preferences, compiled=make_compiled_problem(hard_penalty)
            )
            assert fake.sample_kwargs["chain_strength"] == 2.5

    def test_omitted_chain_strength_stays_omitted_across_penalties(self):
        preferences = make_preferences(chain_strength=None)

        for hard_penalty in (100.0, 5000.0):
            fake, _, _ = solve_with_fake(
                preferences, compiled=make_compiled_problem(hard_penalty)
            )
            assert "chain_strength" not in fake.sample_kwargs


class TestSampleSetConversion:
    def test_samples_and_energies_match_the_sampleset(self):
        fake, compiled, result = solve_with_fake(make_preferences())

        variables = list(compiled.model.variables)
        zeros = {variable: 0 for variable in variables}
        flipped = {**zeros, variables[0]: 1}
        assert sorted(result.variables) == sorted(str(variable) for variable in variables)
        assert result.as_dicts() == [
            {str(variable): int(value) for variable, value in row.items()}
            for row in (zeros, flipped)
        ]
        assert result.energies.tolist() == [
            pytest.approx(compiled.model.energy(zeros)),
            pytest.approx(compiled.model.energy(flipped)),
        ]
        assert result.backend == "dwave_qpu"

    def test_samples_include_internal_slack_variables(self):
        _, compiled, result = solve_with_fake(make_preferences())

        assert compiled.internal_variables
        assert compiled.internal_variables <= set(result.variables)


class TestMetadataSanitization:
    def test_only_whitelisted_timing_keys_survive(self):
        _, _, result = solve_with_fake(make_preferences())

        assert result.metadata.timing_us == {
            "qpu_access_time": 12345.0,
            "qpu_sampling_time": 6789.0,
            "qpu_anneal_time_per_sample": 20.0,
        }

    def test_dirty_info_keys_do_not_leak_anywhere(self):
        _, _, result = solve_with_fake(make_preferences())

        dumped = result.metadata.model_dump_json()
        assert "problem_id" not in dumped
        assert "fake-problem-id-456" not in dumped
        assert "nested" not in dumped
        assert "unexpected" not in dumped
        assert "embedding_context" not in dumped

    def test_backend_remote_and_num_reads_requested(self):
        _, _, result = solve_with_fake(make_preferences(num_reads=42))

        assert result.metadata.backend == "dwave_qpu"
        assert result.metadata.remote is True
        assert result.metadata.num_reads_requested == 42

    def test_average_chain_break_fraction_extracted(self):
        _, _, result = solve_with_fake(make_preferences())

        assert result.metadata.average_chain_break_fraction == pytest.approx(0.125)

    def test_missing_chain_break_fraction_is_none(self):
        fake = FakeQPUSampler(include_chain_break_fraction=False)

        _, _, result = solve_with_fake(make_preferences(), fake=fake)

        assert result.metadata.average_chain_break_fraction is None

    def test_embedding_max_chain_length_extracted(self):
        _, _, result = solve_with_fake(make_preferences())

        assert result.metadata.embedding_max_chain_length == 3

    @pytest.mark.parametrize(
        "embedding_context",
        [
            None,  # key absent entirely
            "not-a-dict",
            {"embedding": "not-a-dict"},
            {"embedding": {}},
            {"embedding": {"a": 3}},  # chain without len()
        ],
        ids=["absent", "context-not-dict", "embedding-not-dict", "empty", "no-len"],
    )
    def test_malformed_embedding_context_yields_none(self, embedding_context):
        # embedding_context=None means the sampler never injects the key.
        fake = FakeQPUSampler(
            info={"timing": {"qpu_access_time": 1}},
            embedding_context=embedding_context,
        )

        _, _, result = solve_with_fake(make_preferences(), fake=fake)

        assert result.metadata.embedding_max_chain_length is None


class TestExceptionClassification:
    @pytest.mark.parametrize(
        ("exception", "expected_code"),
        [
            (EmbeddingError(f"no embedding, token={FAKE_TOKEN}"), "EMBEDDING_FAILED"),
            (ValueError(f"no embedding found, token={FAKE_TOKEN}"), "EMBEDDING_FAILED"),
            (SolverAuthenticationError(f"invalid token={FAKE_TOKEN}"), "REMOTE_AUTH_FAILED"),
            (RequestTimeout(f"timed out, token={FAKE_TOKEN}"), "REMOTE_TIMEOUT"),
            (RuntimeError(f"solver exploded, token={FAKE_TOKEN}"), "REMOTE_SOLVER_ERROR"),
        ],
        ids=["embedding", "value_error", "auth", "timeout", "other"],
    )
    def test_sample_exceptions_map_to_codes_and_are_redacted(
        self, exception, expected_code
    ):
        fake = FakeQPUSampler(raise_on_sample=exception)
        backend = DWaveQPUBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message

    def test_subclass_of_embedding_error_is_classified_via_mro(self):
        class DisconnectedChainError(EmbeddingError):
            pass

        fake = FakeQPUSampler(raise_on_sample=DisconnectedChainError("chain broken"))
        backend = DWaveQPUBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        assert exc_info.value.code == "EMBEDDING_FAILED"

    def test_named_exception_subclassing_value_error_keeps_own_code(self):
        # The MRO walk starts at the most-derived class, so a named Ocean
        # exception wins over its ValueError base.
        class SolverAuthenticationError(ValueError):  # noqa: F811 — deliberate shadow
            pass

        fake = FakeQPUSampler(raise_on_sample=SolverAuthenticationError("denied"))
        backend = DWaveQPUBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        assert exc_info.value.code == "REMOTE_AUTH_FAILED"

    def test_factory_failure_is_classified_and_redacted(self):
        def failing_factory():
            raise SolverAuthenticationError(f"Authorization: Bearer {FAKE_TOKEN}")

        backend = DWaveQPUBackend(sampler_factory=failing_factory)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        assert exc_info.value.code == "REMOTE_AUTH_FAILED"
        assert FAKE_TOKEN not in str(exc_info.value)


class TestSamplerInitExceptionClassification:
    """Spec §15: sampler construction has its own classification table.

    ``DWaveSampler()`` runs Client.from_config() / get_solver(), so a
    ValueError there is a *configuration* problem — never an embedding one.
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
        factory = CountingFactory(FakeQPUSampler(), failures=[exception])
        backend = DWaveQPUBackend(sampler_factory=factory)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

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
        # The sampling-stage table maps ValueError to EMBEDDING_FAILED;
        # telling the agent to shrink the problem would be wrong here.
        backend = DWaveQPUBackend(
            sampler_factory=CountingFactory(FakeQPUSampler(), failures=[exception])
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        assert exc_info.value.code == "DWAVE_CONFIG_INVALID"
        assert exc_info.value.code != "EMBEDDING_FAILED"
        assert FAKE_TOKEN not in str(exc_info.value)
        assert "***" in str(exc_info.value)

    def test_sample_stage_value_error_still_means_embedding_failed(self):
        # The two tables really are independent: the same type, a different
        # stage, a different code.
        fake = FakeQPUSampler(raise_on_sample=ValueError("no embedding found"))
        backend = DWaveQPUBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        assert exc_info.value.code == "EMBEDDING_FAILED"


class TestLazySampleSetResolution:
    """Spec §15/§19: the backend resolves the sampleset inside the guarded
    call, so a failure that only surfaces on resolve is still classified and
    redacted."""

    def test_lazy_success_matches_the_eager_result(self):
        eager, _, expected = solve_with_fake(make_preferences())
        lazy, _, result = solve_with_fake(
            make_preferences(), fake=FakeQPUSampler(lazy=True)
        )

        assert lazy.sample_calls == 1
        assert result.variables == expected.variables
        assert result.samples.tolist() == expected.samples.tolist()
        assert result.energies.tolist() == expected.energies.tolist()
        assert result.backend == expected.backend
        assert result.metadata.model_dump() == expected.metadata.model_dump()
        assert result.metadata.embedding_max_chain_length == 3

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
        fake = FakeQPUSampler(lazy=True, raise_on_sample=exception)
        backend = DWaveQPUBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message
        # sample() itself returned normally: the failure happened on resolve.
        assert fake.sample_calls == 1
        assert fake.sample_kwargs == {
            "num_reads": 100,
            "auto_scale": True,
            "return_embedding": True,
        }


class TestOriginalExceptionIsNotReachable:
    """Spec §19: the wrapped error carries no chain back to the original,
    so a traceback dump cannot print credential-bearing text."""

    def test_sample_failure_has_no_cause_or_context(self):
        fake = FakeQPUSampler(
            raise_on_sample=RuntimeError(f"solver exploded, token={FAKE_TOKEN}")
        )
        backend = DWaveQPUBackend(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        formatted = "".join(traceback.format_exception(error))
        assert FAKE_TOKEN not in formatted

    def test_factory_failure_has_no_cause_or_context(self):
        factory = CountingFactory(
            FakeQPUSampler(),
            failures=[SolverNotFoundError(f"Authorization: Bearer {FAKE_TOKEN}")],
        )
        backend = DWaveQPUBackend(sampler_factory=factory)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(make_compiled_problem(), make_preferences())

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        formatted = "".join(traceback.format_exception(error))
        assert FAKE_TOKEN not in formatted


class TestSamplerCaching:
    """Spec §15: DWaveSampler() is expensive, so it is built once per
    backend instance — and a failed construction is never cached."""

    def test_sampler_is_built_once_per_backend(self):
        factory = CountingFactory(FakeQPUSampler())
        backend = DWaveQPUBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        backend.solve(compiled, make_preferences())
        backend.solve(compiled, make_preferences())

        assert factory.calls == 1
        assert factory.sampler.sample_calls == 2

    def test_failed_construction_is_not_cached(self):
        factory = CountingFactory(
            FakeQPUSampler(),
            failures=[SolverAuthenticationError(f"denied, token={FAKE_TOKEN}")],
        )
        backend = DWaveQPUBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compiled, make_preferences())
        assert exc_info.value.code == "REMOTE_AUTH_FAILED"

        result = backend.solve(compiled, make_preferences())

        assert factory.calls == 2
        assert result.backend == "dwave_qpu"
        assert factory.sampler.sample_calls == 1

    def test_each_backend_instance_builds_its_own_sampler(self):
        factory = CountingFactory(FakeQPUSampler())
        compiled = make_compiled_problem()

        DWaveQPUBackend(sampler_factory=factory).solve(compiled, make_preferences())
        DWaveQPUBackend(sampler_factory=factory).solve(compiled, make_preferences())

        assert factory.calls == 2

    def test_rotated_token_rebuilds_the_sampler(self, monkeypatch):
        """2026-09-09 review F-21: the cache is keyed on the credential
        fingerprint, so a rotated ``DWAVE_API_TOKEN`` is picked up on the
        next solve instead of living on in a client built from the old
        one. Both values are synthetic."""
        monkeypatch.setenv(ocean_module.TOKEN_ENV, FAKE_UNPATTERNED_TOKEN)
        factory = CountingFactory(FakeQPUSampler())
        backend = DWaveQPUBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        backend.solve(compiled, make_preferences())

        assert factory.calls == 1

        monkeypatch.setenv(
            ocean_module.TOKEN_ENV, "DEV-FAKE-TOKEN-ROTATED-0987654321zyxwv"
        )
        backend.solve(compiled, make_preferences())

        assert factory.calls == 2

    def test_auth_failure_at_sampling_invalidates_the_cache(self):
        """F-21: the cloud rejecting the client is the one signal a local
        fingerprint cannot see, so the cached sampler is dropped there."""
        sampler = FakeQPUSampler(
            raise_on_sample=SolverAuthenticationError(f"denied, token={FAKE_TOKEN}")
        )
        factory = CountingFactory(sampler)
        backend = DWaveQPUBackend(sampler_factory=factory)
        compiled = make_compiled_problem()

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compiled, make_preferences())
        assert exc_info.value.code == "REMOTE_AUTH_FAILED"
        assert factory.calls == 1

        sampler.raise_on_sample = None
        result = backend.solve(compiled, make_preferences())

        assert factory.calls == 2
        assert result.backend == "dwave_qpu"


class TestResolveTimeLimit:
    """Spec §16/§20: the QPU takes no time limit, and answering that
    question must not build a sampler (let alone submit anything)."""

    def test_resolve_time_limit_is_none_without_touching_the_factory(self):
        factory = CountingFactory(FakeQPUSampler())
        backend = DWaveQPUBackend(sampler_factory=factory)

        assert backend.resolve_time_limit(make_compiled_problem(), make_preferences()) is None
        assert factory.calls == 0
        assert factory.sampler.sample_calls == 0


class TestIsAvailable:
    """The backend answers through the shared check in solvers.ocean, so
    the installability / credential probes are patched there."""

    def test_dwave_system_not_installed(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: False)

        assert DWaveQPUBackend().is_available() == AvailabilityStatus(
            category="not_installed", detail=REASON_NOT_INSTALLED
        )

    def test_credentials_not_configured(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "missing")

        assert DWaveQPUBackend().is_available() == AvailabilityStatus(
            category="credentials_missing", detail=REASON_CREDENTIALS_MISSING
        )

    def test_configuration_invalid(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "invalid")

        assert DWaveQPUBackend().is_available() == AvailabilityStatus(
            category="config_invalid",
            detail=REASON_CONFIG_INVALID,
            error_code="DWAVE_CONFIG_INVALID",
        )

    def test_available_when_installed_and_configured(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "ok")

        assert DWaveQPUBackend().is_available() == AvailabilityStatus(category="available")
