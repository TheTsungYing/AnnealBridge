"""Unit tests for the solver backends (spec §20–§22, §33)."""

import logging
import os

import dimod
import pytest
from dwave.samplers import SimulatedAnnealingSampler

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    OptimizationProblem,
    ParameterLimit,
    SolverPreferences,
)
from annealbridge.solvers import (
    AvailabilityStatus,
    ExactSolverBackend,
    SimulatedAnnealingBackend,
    SolverCapabilities,
    SolverRegistry,
)
from annealbridge.solvers.base import BackendAliases, log_solved, record_column
from annealbridge.solvers.simulated_annealing import (
    _SEED_LIMIT,
    READS_PER_SHARD,
    default_workers,
    shard_seeds,
    shard_sizes,
)

# Knapsack (spec §30): capacity 10, items A(w6,v10) B(w5,v8) C(w4,v7) D(w3,v6).
# {A, C} has weight 6 + 4 = 10 (feasible) and value 10 + 7 = 17; every other
# feasible subset is worth less ({B,C}=15, {B,D}=14, {C,D}=13, ...), so the
# global optimum objective value is 17.
KNAPSACK_OPTIMUM_VALUE = 17.0
KNAPSACK_OPTIMUM_SELECTION = {"item_a": 1, "item_b": 0, "item_c": 1, "item_d": 0}

# Well above the objective range (max 31) so violating the hard constraint
# always costs more than any objective gain.
HARD_PENALTY = 100.0


@pytest.fixture
def load_knapsack(load_example):
    def _load() -> OptimizationProblem:
        return OptimizationProblem.model_validate(load_example("knapsack.json"))

    return _load


@pytest.fixture
def compile_knapsack(load_knapsack):
    def _compile() -> CompiledProblem:
        return BQMCompiler().compile(load_knapsack(), hard_penalty=HARD_PENALTY)

    return _compile


class TestKnapsackExample:
    def test_example_file_parses(self, load_knapsack):
        problem = load_knapsack()
        assert problem.name == "knapsack"
        assert [v.name for v in problem.variables] == [
            "item_a",
            "item_b",
            "item_c",
            "item_d",
        ]
        assert problem.solver.backend == "exact"


class TestExactSolverBackend:
    def test_properties(self):
        backend = ExactSolverBackend()
        assert backend.name == "exact"
        assert backend.is_exhaustive is True

    def test_capabilities(self):
        capabilities = ExactSolverBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "exact"
        assert capabilities.remote is False
        assert capabilities.heuristic is False
        assert capabilities.exhaustive is True
        assert capabilities.supports_seed is False
        assert capabilities.supports_num_reads is False
        assert capabilities.supports_time_limit is False
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is True
        assert capabilities.description.strip()
        assert capabilities.supports_num_sweeps is False
        assert capabilities.requires_embedding is False
        assert capabilities.parameter_limits == []

    def test_is_available(self):
        status = ExactSolverBackend().is_available()
        assert isinstance(status, AvailabilityStatus)
        assert status.available is True
        assert status == AvailabilityStatus(category="available")

    def test_properties_alias_capabilities(self):
        backend = ExactSolverBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive

    def test_resolve_time_limit_is_none(self, compile_knapsack):
        # A local backend never submits a time limit (Phase 2 spec §10).
        compiled = compile_knapsack()
        assert ExactSolverBackend().resolve_time_limit(compiled, SolverPreferences()) is None

    def test_finds_known_optimum(self, compile_knapsack):
        compiled = compile_knapsack()
        result = ExactSolverBackend().solve(compiled, SolverPreferences())

        best_index = int(result.energies.argmin())
        best_sample = result.as_dicts()[best_index]
        business = {
            name: value
            for name, value in best_sample.items()
            if name not in compiled.internal_variables
        }
        assert business == KNAPSACK_OPTIMUM_SELECTION
        # Maximization compiles with sign -1 and the optimum is feasible
        # (zero penalty), so energy == -objective.
        assert result.energies[best_index] == pytest.approx(-KNAPSACK_OPTIMUM_VALUE)

    def test_returns_all_samples(self, compile_knapsack):
        compiled = compile_knapsack()
        result = ExactSolverBackend().solve(compiled, SolverPreferences())

        assert result.backend == "exact"
        assert result.num_samples == 2**compiled.num_variables
        assert result.samples.shape == (2**compiled.num_variables, compiled.num_variables)
        assert len(result.energies) == result.num_samples
        assert result.num_samples > 0

    def test_samples_include_internal_variables(self, compile_knapsack):
        compiled = compile_knapsack()
        result = ExactSolverBackend().solve(compiled, SolverPreferences())

        assert compiled.internal_variables
        assert compiled.internal_variables <= set(result.variables)
        assert compiled.internal_variables <= set(result.as_dicts()[0])

    def test_reports_local_execution_metadata(self, compile_knapsack):
        compiled = compile_knapsack()
        result = ExactSolverBackend().solve(compiled, SolverPreferences())

        assert result.metadata is not None
        assert result.metadata.backend == "exact"
        assert result.metadata.remote is False
        # A local run has no vendor facts: no timing, no solver id, no quota.
        assert result.metadata.timing_us == {}
        assert result.metadata.solver_id is None
        assert result.metadata.effective_time_limit_seconds is None
        assert result.metadata.average_chain_break_fraction is None
        assert result.metadata.embedding_max_chain_length is None
        assert result.metadata.sampler_reported_feasible is None
        # This backend takes no read count; the service stamps model_type.
        assert result.metadata.num_reads_requested is None
        assert result.metadata.model_type is None


class TestSimulatedAnnealingBackend:
    def test_properties(self):
        backend = SimulatedAnnealingBackend()
        assert backend.name == "simulated_annealing"
        assert backend.is_exhaustive is False

    def test_capabilities(self):
        capabilities = SimulatedAnnealingBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "simulated_annealing"
        assert capabilities.remote is False
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is True
        assert capabilities.supports_num_reads is True
        assert capabilities.supports_time_limit is False
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is True
        assert capabilities.description.strip()
        assert capabilities.supports_num_sweeps is True
        assert capabilities.requires_embedding is False
        # 2026-09-09 review (F-02): the two caller-controlled sampling
        # parameters are policy-limited under their own keys and codes.
        assert capabilities.parameter_limits == [
            ParameterLimit(
                preference="num_reads",
                limit="local_reads",
                error_code="LOCAL_READS_LIMIT",
            ),
            ParameterLimit(
                preference="num_sweeps",
                limit="sweeps",
                error_code="SWEEPS_LIMIT",
            ),
        ]

    def test_is_available(self):
        status = SimulatedAnnealingBackend().is_available()
        assert isinstance(status, AvailabilityStatus)
        assert status.available is True
        assert status == AvailabilityStatus(category="available")

    def test_properties_alias_capabilities(self):
        backend = SimulatedAnnealingBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive

    def test_resolve_time_limit_is_none(self, compile_knapsack):
        compiled = compile_knapsack()
        assert (
            SimulatedAnnealingBackend().resolve_time_limit(compiled, SolverPreferences())
            is None
        )

    def test_same_seed_is_reproducible(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=20, num_sweeps=100, seed=42)
        backend = SimulatedAnnealingBackend()

        first = backend.solve(compiled, preferences)
        second = backend.solve(compiled, preferences)

        assert first.variables == second.variables
        assert first.samples.tolist() == second.samples.tolist()
        assert first.energies.tolist() == second.energies.tolist()

    def test_returns_all_reads(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=20, num_sweeps=100, seed=7)
        result = SimulatedAnnealingBackend().solve(compiled, preferences)

        assert result.backend == "simulated_annealing"
        assert result.num_samples == preferences.num_reads
        assert result.samples.shape == (preferences.num_reads, compiled.num_variables)
        assert len(result.energies) == preferences.num_reads
        assert result.num_samples > 0

    def test_none_seed_still_solves(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=5, num_sweeps=50, seed=None)
        result = SimulatedAnnealingBackend().solve(compiled, preferences)

        assert result.num_samples == preferences.num_reads

    def test_reports_local_execution_metadata(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=20, num_sweeps=100, seed=7)
        result = SimulatedAnnealingBackend().solve(compiled, preferences)

        assert result.metadata is not None
        assert result.metadata.backend == "simulated_annealing"
        assert result.metadata.remote is False
        # A local run has no vendor facts: no timing, no solver id, no quota.
        assert result.metadata.timing_us == {}
        assert result.metadata.solver_id is None
        assert result.metadata.effective_time_limit_seconds is None
        assert result.metadata.average_chain_break_fraction is None
        assert result.metadata.embedding_max_chain_length is None
        assert result.metadata.sampler_reported_feasible is None
        # The reads asked of the sampler, whatever the shard layout was.
        assert result.metadata.num_reads_requested == preferences.num_reads
        # The service stamps the model type; the backend does not know it.
        assert result.metadata.model_type is None

    def test_metadata_read_count_is_independent_of_sharding(self, compile_knapsack):
        # More than one shard: the count reported is the request, not a shard.
        compiled = compile_knapsack()
        num_reads = READS_PER_SHARD * 2 + 3
        preferences = SolverPreferences(num_reads=num_reads, num_sweeps=10, seed=3)
        result = SimulatedAnnealingBackend(workers=2).solve(compiled, preferences)

        assert result.metadata is not None
        assert result.metadata.num_reads_requested == num_reads


class TestSimulatedAnnealingSharding:
    """Reads are sampled in READS_PER_SHARD-sized shards, ``workers`` at a
    time. The contract: the result depends on (problem, num_reads,
    num_sweeps, seed) only — never on the worker count or the machine."""

    def test_shard_sizes_fill_then_remainder(self):
        assert shard_sizes(0) == []
        assert shard_sizes(-3) == []
        assert shard_sizes(1) == [1]
        assert shard_sizes(READS_PER_SHARD) == [READS_PER_SHARD]
        assert shard_sizes(READS_PER_SHARD + 1) == [READS_PER_SHARD, 1]
        assert shard_sizes(60) == [25, 25, 10]
        assert sum(shard_sizes(12345)) == 12345

    def test_shard_seeds_are_pinned(self):
        # Derived through numpy's SeedSequence. The values are pinned so a
        # change in numpy's algorithm (or in our masking) is caught here
        # rather than silently changing every seeded result.
        assert shard_seeds(1234, 4) == [759113382, 1602421836, 216676975, 551138160]
        assert shard_seeds(None, 3) == [None, None, None]
        assert all(0 <= seed < 2**31 for seed in shard_seeds(2**31 - 1, 50))

    def test_workers_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="workers must be >= 1"):
            SimulatedAnnealingBackend(workers=0)
        with pytest.raises(ValueError, match="workers must be >= 1"):
            SimulatedAnnealingBackend(workers=-2)

    def test_workers_default_is_detected_and_explicit_is_kept(self):
        assert SimulatedAnnealingBackend().workers >= 1
        assert SimulatedAnnealingBackend(workers=3).workers == 3

    def test_default_workers_prefers_process_cpu_count(self, monkeypatch):
        # Python 3.13's os.process_cpu_count honours the affinity mask; the
        # attribute is absent on 3.11 / 3.12, so it is faked either way.
        monkeypatch.setattr(os, "process_cpu_count", lambda: 5, raising=False)
        assert default_workers() == 5

    def test_default_workers_falls_back_to_affinity_then_cpu_count(self, monkeypatch):
        monkeypatch.delattr(os, "process_cpu_count", raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
        assert default_workers() == 3

        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 7)
        assert default_workers() == 7

        monkeypatch.setattr(os, "cpu_count", lambda: None)
        assert default_workers() == 1

    def test_result_is_identical_for_any_worker_count(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=60, num_sweeps=100, seed=42)

        results = [
            SimulatedAnnealingBackend(workers=workers).solve(compiled, preferences)
            for workers in (1, 2, 3, 8)
        ]

        for other in results[1:]:
            assert other.variables == results[0].variables
            assert other.samples.tolist() == results[0].samples.tolist()
            assert other.energies.tolist() == results[0].energies.tolist()

    def test_sharded_run_returns_every_read_in_shard_order(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=60, num_sweeps=100, seed=42)

        result = SimulatedAnnealingBackend(workers=4).solve(compiled, preferences)

        assert result.num_samples == 60
        assert result.samples.shape == (60, compiled.num_variables)
        # Each shard is the plain sampler call under its derived seed, and
        # the rows are concatenated in shard order.
        sampler = SimulatedAnnealingSampler()
        expected = []
        for size, seed in zip([25, 25, 10], shard_seeds(42, 3)):
            sampleset = sampler.sample(compiled.model, num_reads=size, num_sweeps=100, seed=seed)
            expected.extend(sampleset.record.sample.tolist())
        assert result.samples.tolist() == expected

    def test_single_shard_passes_the_seed_straight_through(self, compile_knapsack):
        # At most READS_PER_SHARD reads is one shard: the user's seed goes to
        # the sampler unchanged, so these results are exactly what the
        # backend returned before sharding existed.
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=READS_PER_SHARD, num_sweeps=100, seed=42)

        result = SimulatedAnnealingBackend(workers=4).solve(compiled, preferences)
        direct = SimulatedAnnealingSampler().sample(
            compiled.model, num_reads=READS_PER_SHARD, num_sweeps=100, seed=42
        )

        assert result.samples.tolist() == direct.record.sample.tolist()
        assert result.energies.tolist() == direct.record.energy.tolist()

    def test_one_read_past_the_shard_size_takes_two_shards(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(
            num_reads=READS_PER_SHARD + 1, num_sweeps=100, seed=42
        )

        result = SimulatedAnnealingBackend(workers=2).solve(compiled, preferences)

        assert result.num_samples == READS_PER_SHARD + 1

    @pytest.mark.parametrize("seed", [-1, 2**31])
    @pytest.mark.parametrize("num_reads", [10, 60])
    def test_out_of_range_seed_is_a_solver_error_for_any_read_count(
        self, compile_knapsack, seed, num_reads
    ):
        # The sampler accepts 0 <= seed < 2**31. One shard lets the sampler
        # reject it; several shards check the same rule before deriving
        # shard seeds, so the status is the same either way.
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=num_reads, num_sweeps=10, seed=seed)

        with pytest.raises(SolverExecutionError, match="between 0 and"):
            SimulatedAnnealingBackend(workers=2).solve(compiled, preferences)

    @pytest.mark.parametrize("seed", [2**31, -1])
    def test_out_of_range_seed_fails_the_same_way_on_both_paths(
        self, compile_knapsack, seed
    ):
        # One shard (10 reads) lets the sampler reject the seed; several
        # shards (60 reads) reject it before deriving shard seeds. The caller
        # must not be able to tell the paths apart by exception type, cause
        # type or message frame. The upper bound quoted in between is not
        # pinned: the vendor's text says "2^32 - 1" although it checks
        # ``< 2**31``.
        compiled = compile_knapsack()
        errors = []
        for num_reads in (10, 60):
            preferences = SolverPreferences(num_reads=num_reads, num_sweeps=10, seed=seed)
            with pytest.raises(SolverExecutionError) as exc_info:
                SimulatedAnnealingBackend(workers=2).solve(compiled, preferences)
            errors.append(exc_info.value)

        single_shard, sharded = errors
        assert type(single_shard) is type(sharded) is SolverExecutionError
        assert type(single_shard.__cause__) is type(sharded.__cause__) is ValueError
        for error in errors:
            message = str(error)
            assert message.startswith(
                "Simulated annealing solver failed: "
                "'seed' should be an integer between 0 and "
            )
            assert message.endswith(f"value = {seed}")

    @pytest.mark.parametrize("num_reads", [10, 60])
    def test_largest_accepted_seed_solves_on_both_paths(self, compile_knapsack, num_reads):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=num_reads, num_sweeps=10, seed=2**31 - 1)

        result = SimulatedAnnealingBackend(workers=2).solve(compiled, preferences)

        assert result.num_samples == num_reads

    def test_vendor_sampler_accepts_the_largest_seed_the_shard_check_allows(self):
        """Pin dwave-samplers' own seed rule, which the sharded path copies.

        The single-shard path hands the seed straight to
        ``SimulatedAnnealingSampler``; the sharded path re-implements the
        sampler's range check through ``_SEED_LIMIT`` so both paths reject
        the same seeds. If a dwave-samplers upgrade changes the accepted
        range, this test (and its rejecting twin) turns red and the copied
        limit has to be revisited.
        """
        bqm = dimod.BinaryQuadraticModel({"a": 1.0}, {}, 0.0, dimod.BINARY)

        sampleset = SimulatedAnnealingSampler().sample(
            bqm, num_reads=1, num_sweeps=1, seed=_SEED_LIMIT - 1
        )

        assert len(sampleset) == 1

    @pytest.mark.parametrize("seed", [_SEED_LIMIT, -1], ids=["seed_limit", "negative"])
    def test_vendor_sampler_rejects_the_seeds_the_shard_check_rejects(self, seed):
        """The rejecting half of the pinned vendor rule (see the accepting twin)."""
        bqm = dimod.BinaryQuadraticModel({"a": 1.0}, {}, 0.0, dimod.BINARY)

        with pytest.raises(ValueError):
            SimulatedAnnealingSampler().sample(bqm, num_reads=1, num_sweeps=1, seed=seed)

    def test_failed_shard_fails_the_solve(self, compile_knapsack, monkeypatch):
        compiled = compile_knapsack()
        backend = SimulatedAnnealingBackend(workers=2)
        original = backend._sample

        def failing(model, num_reads, num_sweeps, seed):
            if num_reads == 10:  # the remainder shard
                raise RuntimeError("shard exploded")
            return original(model, num_reads, num_sweeps, seed)

        monkeypatch.setattr(backend, "_sample", failing)

        with pytest.raises(SolverExecutionError, match="shard exploded"):
            backend.solve(compiled, SolverPreferences(num_reads=60, num_sweeps=10, seed=1))


class TestBackendAliases:
    """2026-09-09 review F-26c: ``name``, ``is_exhaustive`` and the "no time
    limit" ``resolve_time_limit`` come from one mixin instead of twelve
    verbatim copies. The test fakes deliberately keep their own copies, so
    the ``SolverBackend`` protocol stays structural.
    """

    @staticmethod
    def built_in_backends() -> list:
        registry = SolverRegistry.default()
        return [registry.get(name) for name in registry.names()]

    def test_every_built_in_backend_derives_its_aliases_from_the_declaration(self):
        backends = self.built_in_backends()

        assert len(backends) == 6
        for backend in backends:
            assert isinstance(backend, BackendAliases), backend
            assert backend.name == backend.capabilities.name
            assert backend.is_exhaustive == backend.capabilities.exhaustive

    @pytest.mark.parametrize("name", ["exact", "simulated_annealing", "dwave_qpu"])
    def test_a_backend_without_a_time_limit_resolves_to_none(self, name):
        backend = SolverRegistry.default().get(name)

        assert backend.capabilities.supports_time_limit is False
        # The mixin's default never looks at the compiled problem.
        assert backend.resolve_time_limit(None, SolverPreferences()) is None

    def test_declaring_a_time_limit_without_implementing_it_raises(self):
        declared = SolverCapabilities(
            name="claims_a_time_limit",
            remote=True,
            heuristic=True,
            exhaustive=False,
            supports_seed=False,
            supports_num_reads=False,
            supports_time_limit=True,
            supported_model_types=["bqm"],
            returns_multiple_samples=True,
            description="Declares a time limit but never implements one.",
        )

        class Incomplete(BackendAliases):
            @property
            def capabilities(self) -> SolverCapabilities:
                return declared

        backend = Incomplete()
        assert backend.name == "claims_a_time_limit"
        assert backend.is_exhaustive is False
        with pytest.raises(NotImplementedError, match="resolve_time_limit"):
            backend.resolve_time_limit(None, SolverPreferences())


class TestLogSolved:
    """The shared solved-log line in ``solvers.base``."""

    LOGGER = "annealbridge.solvers.test_log_solved"

    def test_without_extra_there_are_no_parentheses(self, caplog):
        logger = logging.getLogger(self.LOGGER)

        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            log_solved(logger, "exact", "demo problem", 3, 8)

        (record,) = caplog.records
        assert record.levelno == logging.INFO
        assert record.getMessage() == (
            "Backend exact solved problem demo problem: 3 variables, 8 samples"
        )

    def test_extra_is_appended_in_the_given_order(self, caplog):
        logger = logging.getLogger(self.LOGGER)

        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            log_solved(logger, "b", "demo", 3, 2, zeta=1, alpha=None, mid=2.5)

        (record,) = caplog.records
        assert record.getMessage() == (
            "Backend b solved problem demo: 3 variables, 2 samples "
            "(zeta=1, alpha=None, mid=2.5)"
        )

    def test_values_are_lazy_logger_arguments(self, caplog):
        class Probe:
            calls = 0

            def __str__(self) -> str:
                Probe.calls += 1
                return "probe%"

        logger = logging.getLogger(self.LOGGER)
        probe = Probe()

        # INFO disabled: nothing may be formatted, so ``__str__`` never runs.
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            log_solved(logger, "b", "demo", 3, 2, job_id=probe)
        assert Probe.calls == 0
        assert caplog.records == []

        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            log_solved(logger, "b", "demo", 3, 2, job_id=probe)

        (record,) = caplog.records
        assert record.msg == (
            "Backend %s solved problem %s: %d variables, %d samples (job_id=%s)"
        )
        assert record.args == ("b", "demo", 3, 2, probe)
        assert record.getMessage().endswith("(job_id=probe%)")


class TestRecordColumn:
    """Optional ``sampleset.record`` fields, looked up through ``dtype.names``."""

    def test_present_field_is_returned(self):
        sampleset = dimod.SampleSet.from_samples(
            [{"a": 0}, {"a": 1}],
            vartype="BINARY",
            energy=[0.0, 1.0],
            is_feasible=[True, False],
        )

        values = record_column(sampleset, "is_feasible")

        assert values is not None
        assert values.tolist() == [True, False]

    def test_missing_field_is_none(self):
        sampleset = dimod.SampleSet.from_samples(
            [{"a": 0}], vartype="BINARY", energy=[0.0]
        )

        assert record_column(sampleset, "is_feasible") is None
        # Why presence goes through ``dtype.names``: plain attribute access raises.
        with pytest.raises(AttributeError):
            sampleset.record.is_feasible
