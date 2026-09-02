"""Mock tests for DWaveQPUBackend (Phase 2 spec §15, §15.1, §27).

Never talks to real D-Wave: a fake sampler is injected through the
backend's ``sampler_factory`` seam and returns a real ``dimod.SampleSet``
whose ``info`` carries fake QPU timing, an embedding context and dirty
keys that must not survive sanitization.
"""

import sys

import dimod
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    DWaveQPUOptions,
    OptimizationProblem,
    SolverPreferences,
)
from annealbridge.solvers import DWaveQPUBackend, SolverCapabilities
import annealbridge.solvers.dwave_qpu as qpu_module

# Matches the DEV-[A-Za-z0-9]{20,} redaction pattern; never a real token.
FAKE_TOKEN = "DEV-FAKETOKEN1234567890abcdefghij"

# Nested QPU timing (whitelist keys), an embedding context, and dirty keys
# that sanitization must drop.
FAKE_SAMPLESET_INFO = {
    "timing": {
        "qpu_access_time": 12345,
        "qpu_sampling_time": 6789,
        "qpu_anneal_time_per_sample": 20,
    },
    "embedding_context": {
        "embedding": {"a": (0, 1, 2), "b": (3,), "slack": (4, 5)},
    },
    "problem_id": "fake-problem-id-456",
    "messages": [{"nested": "structure"}],
    "raw_blob": b"\x00\x01\x02",
    "unexpected": {"deep": ("tuple", object())},
}


# The backend classifies Ocean exceptions by class name (dwave-cloud-client
# and dwave-system are not installed in mock CI), so the fakes carry the
# real names.
class SolverAuthenticationError(Exception):
    """Fake of dwave.cloud's SolverAuthenticationError."""


class RequestTimeout(Exception):
    """Fake of dwave.cloud's RequestTimeout."""


class EmbeddingError(Exception):
    """Fake of dwave.embedding's EmbeddingError."""


class FakeQPUSampler:
    """Fake with the EmbeddingComposite surface the backend touches."""

    def __init__(
        self,
        raise_on_sample: Exception | None = None,
        include_chain_break_fraction: bool = True,
        info: dict | None = None,
    ) -> None:
        self.raise_on_sample = raise_on_sample
        self.include_chain_break_fraction = include_chain_break_fraction
        self.info = FAKE_SAMPLESET_INFO if info is None else info
        self.sample_bqm = None
        self.sample_kwargs = None

    def sample(self, bqm, **kwargs) -> dimod.SampleSet:
        self.sample_bqm = bqm
        self.sample_kwargs = kwargs
        if self.raise_on_sample is not None:
            raise self.raise_on_sample
        variables = list(bqm.variables)
        rows = [
            {variable: 0 for variable in variables},
            {**{variable: 0 for variable in variables}, variables[0]: 1},
        ]
        vectors = {}
        if self.include_chain_break_fraction:
            vectors["chain_break_fraction"] = [0.0, 0.25]
        return dimod.SampleSet.from_samples(
            rows,
            vartype=dimod.BINARY,
            energy=[bqm.energy(row) for row in rows],
            info=dict(self.info),
            **vectors,
        )


def make_compiled_problem(hard_penalty: float = 100.0) -> CompiledProblem:
    """Minimal 0/1 problem: maximize 2a + b s.t. a + b <= 1 (adds slack)."""
    problem = OptimizationProblem.model_validate(
        {
            "name": "dwave qpu mock problem",
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
            "solver": {"backend": "dwave_qpu"},
        }
    )
    return BQMCompiler().compile(problem, hard_penalty=hard_penalty)


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

        assert fake.sample_kwargs == {"num_reads": 100, "auto_scale": True}

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
        }

    def test_chain_strength_none_is_omitted(self):
        fake, _, _ = solve_with_fake(make_preferences(chain_strength=None))

        assert "chain_strength" not in fake.sample_kwargs

    def test_no_options_object_omits_annealing_time(self):
        preferences = SolverPreferences(backend="dwave_qpu", num_reads=7)
        assert preferences.dwave_qpu is None

        fake, _, _ = solve_with_fake(preferences)

        assert fake.sample_kwargs == {"num_reads": 7, "auto_scale": True}

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
        }


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
        assert result.samples == [
            {str(variable): int(value) for variable, value in row.items()}
            for row in (zeros, flipped)
        ]
        assert result.energies == [
            pytest.approx(compiled.model.energy(zeros)),
            pytest.approx(compiled.model.energy(flipped)),
        ]
        assert result.backend == "dwave_qpu"

    def test_samples_include_internal_slack_variables(self):
        _, compiled, result = solve_with_fake(make_preferences())

        assert compiled.internal_variables
        assert compiled.internal_variables <= set(result.samples[0])


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
        info = {"timing": {"qpu_access_time": 1}}
        if embedding_context is not None:
            info["embedding_context"] = embedding_context
        fake = FakeQPUSampler(info=info)

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


class TestIsAvailable:
    def test_dwave_system_not_installed(self, monkeypatch):
        monkeypatch.setattr(qpu_module, "_dwave_system_installed", lambda: False)

        assert DWaveQPUBackend().is_available() == (
            False,
            "dwave-system not installed",
        )

    def test_credentials_not_configured(self, monkeypatch):
        monkeypatch.setattr(qpu_module, "_dwave_system_installed", lambda: True)
        monkeypatch.setattr(qpu_module, "ocean_config_status", lambda: "missing")

        assert DWaveQPUBackend().is_available() == (
            False,
            "D-Wave credentials not configured",
        )

    def test_configuration_invalid(self, monkeypatch):
        monkeypatch.setattr(qpu_module, "_dwave_system_installed", lambda: True)
        monkeypatch.setattr(qpu_module, "ocean_config_status", lambda: "invalid")

        assert DWaveQPUBackend().is_available() == (
            False,
            "D-Wave configuration invalid",
        )

    def test_available_when_installed_and_configured(self, monkeypatch):
        monkeypatch.setattr(qpu_module, "_dwave_system_installed", lambda: True)
        monkeypatch.setattr(qpu_module, "ocean_config_status", lambda: "ok")

        assert DWaveQPUBackend().is_available() == (True, None)
