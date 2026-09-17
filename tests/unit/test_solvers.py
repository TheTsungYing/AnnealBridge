"""Unit tests for the solver backends (spec §20–§22, §33)."""

import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import dimod
import numpy as np
import pytest
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler

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
    SimulatedBifurcationBackend,
    SolverCapabilities,
    SolverRegistry,
    TabuBackend,
)
from annealbridge.solvers.base import BackendAliases, log_solved, record_column
from annealbridge.solvers.sharding import (
    READS_PER_SHARD,
    default_workers,
    shard_seeds,
    shard_sizes,
)
from annealbridge.solvers.simulated_annealing import _SEED_LIMIT
from annealbridge.solvers.simulated_bifurcation import (
    _CAPABILITIES,
    AGENTS_PER_BATCH,
    _batch_sizes,
    _dense_ising,
    _initial_states,
)
from annealbridge.solvers.tabu import _SEED_LIMIT as _TABU_SEED_LIMIT

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


class TestTabuBackend:
    def test_properties(self):
        backend = TabuBackend()
        assert backend.name == "tabu"
        assert backend.is_exhaustive is False

    def test_capabilities(self):
        capabilities = TabuBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "tabu"
        assert capabilities.remote is False
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is True
        assert capabilities.supports_num_reads is True
        assert capabilities.supports_time_limit is False
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is True
        assert capabilities.description.strip()
        # Tabu search takes no sweeps at all, unlike the annealer.
        assert capabilities.supports_num_sweeps is False
        assert capabilities.requires_embedding is False
        # The sampler's own seed range (0 .. 2**32 - 1), wider than the
        # simulated annealer's, declared so validation refuses an
        # out-of-range seed before anything runs.
        assert capabilities.seed_min == 0
        assert capabilities.seed_max == 2**32 - 1
        # Reads are the one caller-controlled parameter this backend takes,
        # limited under the local key.
        assert capabilities.parameter_limits == [
            ParameterLimit(
                preference="num_reads",
                limit="local_reads",
                error_code="LOCAL_READS_LIMIT",
            ),
        ]

    def test_is_available(self):
        status = TabuBackend().is_available()
        assert isinstance(status, AvailabilityStatus)
        assert status.available is True
        assert status == AvailabilityStatus(category="available")

    def test_properties_alias_capabilities(self):
        backend = TabuBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive

    def test_resolve_time_limit_is_none(self, compile_knapsack):
        compiled = compile_knapsack()
        assert TabuBackend().resolve_time_limit(compiled, SolverPreferences()) is None

    def test_same_seed_is_reproducible(self, compile_knapsack):
        # 60 reads is the sharded path (> READS_PER_SHARD): the promise holds
        # there too, because the search is bounded by a count and not by a
        # wall clock.
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=60, seed=42)
        backend = TabuBackend()

        first = backend.solve(compiled, preferences)
        second = backend.solve(compiled, preferences)

        assert first.variables == second.variables
        assert first.samples.tolist() == second.samples.tolist()
        assert first.energies.tolist() == second.energies.tolist()

    def test_returns_all_reads(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=20, seed=7)
        result = TabuBackend().solve(compiled, preferences)

        assert result.backend == "tabu"
        assert result.num_samples == preferences.num_reads
        assert result.samples.shape == (preferences.num_reads, compiled.num_variables)
        assert len(result.energies) == preferences.num_reads
        assert result.num_samples > 0

    def test_none_seed_still_solves(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=5, seed=None)
        result = TabuBackend().solve(compiled, preferences)

        assert result.num_samples == preferences.num_reads

    def test_reports_local_execution_metadata(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=20, seed=7)
        result = TabuBackend().solve(compiled, preferences)

        assert result.metadata is not None
        assert result.metadata.backend == "tabu"
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
        preferences = SolverPreferences(num_reads=num_reads, seed=3)
        result = TabuBackend(workers=2).solve(compiled, preferences)

        assert result.metadata is not None
        assert result.metadata.num_reads_requested == num_reads

    @pytest.mark.parametrize("seed", [11, None], ids=["seeded", "unseeded"])
    def test_exactly_the_declared_parameters_reach_the_sampler(
        self, compile_knapsack, monkeypatch, seed
    ):
        # ``TabuSampler.sample`` ends in ``**kwargs``, so an unknown name
        # would be swallowed in silence: the backend must submit the
        # caller's ``num_reads`` / ``seed`` plus its own fixed search bound,
        # and nothing else. ``num_sweeps`` is non-default here and must not
        # reach the sampler; ``timeout`` must be off and ``num_restarts``
        # zero, which is what makes a seeded run reproducible.
        compiled = compile_knapsack()
        backend = TabuBackend()
        original = backend._sampler.sample
        seen: list[dict] = []

        def recording(model, **kwargs):
            seen.append(kwargs)
            return original(model, **kwargs)

        monkeypatch.setattr(backend._sampler, "sample", recording)
        backend.solve(compiled, SolverPreferences(num_reads=10, num_sweeps=7, seed=seed))

        expected = {"num_reads", "timeout", "num_restarts"}
        if seed is not None:
            expected.add("seed")
        assert seen
        for kwargs in seen:
            assert set(kwargs) == expected
            assert "num_sweeps" not in kwargs
            assert kwargs["timeout"] is None
            assert kwargs["num_restarts"] == 0

    @pytest.mark.parametrize("seed", [-1, 2**32])
    @pytest.mark.parametrize("num_reads", [10, 60])
    def test_out_of_range_seed_is_a_solver_error_for_any_read_count(
        self, compile_knapsack, seed, num_reads
    ):
        # The sampler accepts 0 <= seed <= 2**32 - 1. One shard lets the
        # sampler reject it; several shards check the same rule before
        # deriving shard seeds, so the status is the same either way.
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=num_reads, seed=seed)

        with pytest.raises(SolverExecutionError, match="between 0 and"):
            TabuBackend(workers=2).solve(compiled, preferences)

    @pytest.mark.parametrize("seed", [-1, 2**32])
    def test_out_of_range_seed_fails_the_same_way_on_both_paths(
        self, compile_knapsack, seed
    ):
        # One shard (10 reads) lets the sampler reject the seed; several
        # shards (60 reads) reject it before deriving shard seeds. Here the
        # two messages must be *identical*, not merely alike: the sharded
        # path raises the vendor's own wording verbatim, so a caller cannot
        # tell from the error which path ran. (The annealer's twin can only
        # compare the frame, because that vendor message quotes the offending
        # value.)
        compiled = compile_knapsack()
        errors = []
        for num_reads in (10, 60):
            preferences = SolverPreferences(num_reads=num_reads, seed=seed)
            with pytest.raises(SolverExecutionError) as exc_info:
                TabuBackend(workers=2).solve(compiled, preferences)
            errors.append(exc_info.value)

        single_shard, sharded = errors
        assert type(single_shard) is type(sharded) is SolverExecutionError
        assert str(single_shard) == str(sharded)
        assert type(single_shard.__cause__) is type(sharded.__cause__) is ValueError
        assert str(single_shard.__cause__) == str(sharded.__cause__)
        assert str(single_shard) == (
            "Tabu search solver failed: Seed must be between 0 and 2**32 - 1"
        )

    @pytest.mark.parametrize("num_reads", [10, 60])
    def test_largest_accepted_seed_solves_on_both_paths(self, compile_knapsack, num_reads):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=num_reads, seed=2**32 - 1)

        result = TabuBackend(workers=2).solve(compiled, preferences)

        assert result.num_samples == num_reads

    def test_vendor_sampler_accepts_the_largest_seed_the_shard_check_allows(self):
        """Pin dwave-samplers' tabu seed rule, which the sharded path copies.

        It is not the simulated annealer's rule: ``TabuSampler`` accepts up
        to ``2**32 - 1`` where the annealer stops at ``2**31``. If an upgrade
        changes the range, this test (and its rejecting twin) turns red and
        the copied limit has to be revisited.
        """
        bqm = dimod.BinaryQuadraticModel({"a": 1.0}, {}, 0.0, dimod.BINARY)

        sampleset = TabuSampler().sample(bqm, num_reads=1, seed=_TABU_SEED_LIMIT - 1)

        assert len(sampleset) == 1

    @pytest.mark.parametrize(
        "seed", [_TABU_SEED_LIMIT, -1], ids=["seed_limit", "negative"]
    )
    def test_vendor_sampler_rejects_the_seeds_the_shard_check_rejects(self, seed):
        """The rejecting half of the pinned vendor rule (see the accepting twin)."""
        bqm = dimod.BinaryQuadraticModel({"a": 1.0}, {}, 0.0, dimod.BINARY)

        with pytest.raises(ValueError):
            TabuSampler().sample(bqm, num_reads=1, seed=seed)

    def test_workers_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="workers must be >= 1"):
            TabuBackend(workers=0)
        with pytest.raises(ValueError, match="workers must be >= 1"):
            TabuBackend(workers=-2)

    def test_workers_default_is_detected_and_explicit_is_kept(self):
        assert TabuBackend().workers >= 1
        assert TabuBackend(workers=3).workers == 3

    def test_result_is_identical_for_any_worker_count(self, compile_knapsack):
        # Same contract as the annealer: the shard layout and shard seeds are
        # a function of (num_reads, seed) only, and the search is bounded by
        # a count rather than a wall clock, so workers change wall time and
        # nothing else. 200 reads is 8 shards, so 4 and 8 workers both run
        # every shard concurrently.
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=200, seed=42)

        results = [
            TabuBackend(workers=workers).solve(compiled, preferences)
            for workers in (1, 4, 8)
        ]

        for other in results[1:]:
            assert other.variables == results[0].variables
            assert other.samples.tolist() == results[0].samples.tolist()
            assert other.energies.tolist() == results[0].energies.tolist()

    def test_sharded_run_returns_every_read_in_shard_order(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=60, seed=42)

        result = TabuBackend(workers=2).solve(compiled, preferences)

        assert result.num_samples == 60
        assert result.samples.shape == (60, compiled.num_variables)
        # Each shard is the plain sampler call under its derived seed, with
        # this backend's fixed search bound, and the rows are concatenated in
        # shard order.
        sampler = TabuSampler()
        expected = []
        sizes = shard_sizes(60)
        for size, seed in zip(sizes, shard_seeds(42, len(sizes))):
            sampleset = sampler.sample(
                compiled.model, num_reads=size, seed=seed, timeout=None, num_restarts=0
            )
            expected.extend(sampleset.record.sample.tolist())
        assert result.samples.tolist() == expected

    def test_single_shard_passes_the_seed_straight_through(
        self, compile_knapsack, monkeypatch
    ):
        # At most READS_PER_SHARD reads is one shard: no seed derivation at
        # all, the user's seed reaches the sampler unchanged, so the result is
        # exactly what a plain sampler call gives.
        compiled = compile_knapsack()
        backend = TabuBackend(workers=4)
        original = backend._sampler.sample
        seen: list[dict] = []

        def recording(model, **kwargs):
            seen.append(kwargs)
            return original(model, **kwargs)

        monkeypatch.setattr(backend._sampler, "sample", recording)
        result = backend.solve(
            compiled, SolverPreferences(num_reads=READS_PER_SHARD, seed=42)
        )

        assert [kwargs["num_reads"] for kwargs in seen] == [READS_PER_SHARD]
        assert seen[0]["seed"] == 42
        direct = TabuSampler().sample(
            compiled.model,
            num_reads=READS_PER_SHARD,
            seed=42,
            timeout=None,
            num_restarts=0,
        )
        assert result.samples.tolist() == direct.record.sample.tolist()
        assert result.energies.tolist() == direct.record.energy.tolist()

    def test_vendor_sampler_is_deterministic_under_concurrency_without_a_timeout(self):
        """Pin the vendor behaviour the worker-count promise rests on.

        With ``timeout=None`` and ``num_restarts=0`` a seeded read is a
        fixed amount of work, so the same call gives the same sample set
        whether it runs alone or alongside three others. The vendor's own
        default (``timeout=20``, a 20 ms wall clock per read) does *not*:
        concurrency changes how much search fits in the budget, which is
        why the backend submits both parameters explicitly. If an upgrade
        makes the untimed search non-deterministic, this turns red before
        the reproducibility tests above do.
        """
        bqm = dimod.BinaryQuadraticModel(
            {f"x{i}": float(i % 3) - 1.0 for i in range(12)},
            {(f"x{i}", f"x{j}"): float((i * j) % 5) - 2.0 for i in range(12) for j in range(i + 1, 12)},
            0.0,
            dimod.BINARY,
        )
        sampler = TabuSampler()

        def sample(_: int) -> dimod.SampleSet:
            return sampler.sample(bqm, num_reads=5, seed=99, timeout=None, num_restarts=0)

        serial = [sample(index) for index in range(4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            concurrent = list(executor.map(sample, range(4)))

        expected = serial[0].record.sample.tolist()
        expected_energies = serial[0].record.energy.tolist()
        for sampleset in serial + concurrent:
            assert sampleset.record.sample.tolist() == expected
            assert sampleset.record.energy.tolist() == expected_energies

    def test_failed_shard_fails_the_solve(self, compile_knapsack, monkeypatch):
        compiled = compile_knapsack()
        backend = TabuBackend(workers=2)
        original = backend._sample

        def failing(model, num_reads, seed):
            if num_reads == 10:  # the remainder shard
                raise RuntimeError("shard exploded")
            return original(model, num_reads, seed)

        monkeypatch.setattr(backend, "_sample", failing)

        with pytest.raises(SolverExecutionError, match="shard exploded"):
            backend.solve(compiled, SolverPreferences(num_reads=60, seed=1))


class TestSimulatedBifurcationBackend:
    """The eighth backend: Toshiba's simulated bifurcation, in numpy."""

    @staticmethod
    def compiled_from_bqm(
        bqm: dimod.BinaryQuadraticModel, problem: OptimizationProblem
    ) -> CompiledProblem:
        """A compiled problem around an arbitrary BQM (the backend only reads
        ``model`` and the counts; the provenance fields are not used)."""
        return CompiledProblem(
            model=bqm,
            original_problem=problem,
            internal_variables=set(),
            constraint_trace=[],
            hard_penalty=None,
            objective_scale=1.0,
            num_variables=bqm.num_variables,
            num_interactions=bqm.num_interactions,
        )

    @staticmethod
    def dense_bqm(n: int, seed: int) -> dimod.BinaryQuadraticModel:
        """A dense +-1 SK instance with a random field, as a binary model."""
        rng = np.random.default_rng(seed)
        upper = np.triu(rng.choice([-1.0, 1.0], size=(n, n)), 1)
        spin = dimod.BinaryQuadraticModel(
            {f"s{i}": float(rng.normal()) for i in range(n)},
            {(f"s{i}", f"s{j}"): upper[i, j] for i in range(n) for j in range(i + 1, n)},
            0.0,
            dimod.SPIN,
        )
        return spin.change_vartype(dimod.BINARY, inplace=False)

    def test_properties(self):
        backend = SimulatedBifurcationBackend()
        assert backend.name == "simulated_bifurcation"
        assert backend.is_exhaustive is False
        assert backend.device == "cpu"
        assert backend.max_variables == 10_000

    def test_capabilities(self):
        capabilities = SimulatedBifurcationBackend().capabilities
        assert isinstance(capabilities, SolverCapabilities)
        assert capabilities.name == "simulated_bifurcation"
        assert capabilities.remote is False
        assert capabilities.heuristic is True
        assert capabilities.exhaustive is False
        assert capabilities.supports_seed is True
        assert capabilities.supports_num_reads is True
        # Steps are sweeps: the one knob on the work per read.
        assert capabilities.supports_num_sweeps is True
        # No wall clock anywhere, so no time limit to declare.
        assert capabilities.supports_time_limit is False
        assert capabilities.supported_model_types == ["bqm"]
        assert capabilities.returns_multiple_samples is True
        assert capabilities.requires_embedding is False
        assert capabilities.description.strip()
        # The same range as tabu (the two backends added after the annealer
        # agree), declared so validation refuses it before anything runs.
        assert capabilities.seed_min == 0
        assert capabilities.seed_max == 2**32 - 1
        assert capabilities.parameter_limits == [
            ParameterLimit(
                preference="num_reads",
                limit="local_reads",
                error_code="LOCAL_READS_LIMIT",
            ),
            ParameterLimit(
                preference="num_sweeps", limit="sweeps", error_code="SWEEPS_LIMIT"
            ),
        ]
        assert capabilities.credentials.empty

    def test_is_available_on_cpu(self):
        status = SimulatedBifurcationBackend().is_available()
        assert isinstance(status, AvailabilityStatus)
        assert status == AvailabilityStatus(category="available")

    def test_properties_alias_capabilities(self):
        backend = SimulatedBifurcationBackend()
        assert backend.name == backend.capabilities.name
        assert backend.is_exhaustive == backend.capabilities.exhaustive

    def test_resolve_time_limit_is_none(self, compile_knapsack):
        compiled = compile_knapsack()
        backend = SimulatedBifurcationBackend()
        assert backend.resolve_time_limit(compiled, SolverPreferences()) is None

    @pytest.mark.parametrize(
        ("device", "max_variables"),
        [("gpu", 10), ("cpu", 0), ("cpu", -1)],
        ids=["unknown-device", "zero-cap", "negative-cap"],
    )
    def test_constructor_rejects_bad_arguments(self, device, max_variables):
        with pytest.raises(ValueError):
            SimulatedBifurcationBackend(device=device, max_variables=max_variables)

    def test_same_seed_is_reproducible_across_batches(self, compile_knapsack):
        # More reads than one batch holds: the initial state of every read
        # depends on the seed and its index only, never on its batch.
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=AGENTS_PER_BATCH + 7, seed=42)
        backend = SimulatedBifurcationBackend()

        first = backend.solve(compiled, preferences)
        second = backend.solve(compiled, preferences)

        assert first.variables == second.variables
        assert first.samples.tolist() == second.samples.tolist()
        assert first.energies.tolist() == second.energies.tolist()

    def test_returns_all_reads_as_binary_samples(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=20, seed=7)
        result = SimulatedBifurcationBackend().solve(compiled, preferences)

        assert result.backend == "simulated_bifurcation"
        assert result.num_samples == preferences.num_reads
        assert result.samples.shape == (preferences.num_reads, compiled.num_variables)
        assert result.samples.dtype == np.int8
        assert set(np.unique(result.samples).tolist()) <= {0, 1}
        assert len(result.energies) == preferences.num_reads
        assert result.variables == list(compiled.model.variables)

    def test_energies_are_the_models_own(self, compile_knapsack):
        # Every sample is re-scored by the binary BQM (offset included), not
        # by the dynamics.
        compiled = compile_knapsack()
        result = SimulatedBifurcationBackend().solve(
            compiled, SolverPreferences(num_reads=30, seed=3)
        )
        expected = compiled.model.energies((result.samples, result.variables))
        assert result.energies.dtype == np.float64
        assert result.energies.tolist() == list(expected)

    def test_none_seed_still_solves(self, compile_knapsack):
        compiled = compile_knapsack()
        result = SimulatedBifurcationBackend().solve(
            compiled, SolverPreferences(num_reads=5, seed=None)
        )
        assert result.num_samples == 5

    def test_reports_local_execution_metadata(self, compile_knapsack):
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=AGENTS_PER_BATCH + 1, seed=7)
        result = SimulatedBifurcationBackend().solve(compiled, preferences)

        assert result.metadata is not None
        assert result.metadata.backend == "simulated_bifurcation"
        assert result.metadata.remote is False
        assert result.metadata.timing_us == {}
        assert result.metadata.solver_id is None
        assert result.metadata.effective_time_limit_seconds is None
        assert result.metadata.average_chain_break_fraction is None
        assert result.metadata.embedding_max_chain_length is None
        assert result.metadata.sampler_reported_feasible is None
        # The whole request, whatever the batch layout was.
        assert result.metadata.num_reads_requested == preferences.num_reads
        assert result.metadata.model_type is None

    @pytest.mark.parametrize("seed", [-1, 2**32])
    def test_out_of_range_seed_is_a_solver_error(self, compile_knapsack, seed):
        compiled = compile_knapsack()
        preferences = SolverPreferences.model_construct(num_reads=10, seed=seed)
        with pytest.raises(SolverExecutionError) as excinfo:
            SimulatedBifurcationBackend().solve(compiled, preferences)
        assert str(excinfo.value) == (
            "Simulated bifurcation solver failed: seed must be between 0 and "
            f"2**32 - 1, got {seed}"
        )

    def test_largest_accepted_seed_solves(self, compile_knapsack):
        compiled = compile_knapsack()
        result = SimulatedBifurcationBackend().solve(
            compiled, SolverPreferences(num_reads=10, seed=2**32 - 1)
        )
        assert result.num_samples == 10

    def test_over_the_variable_cap_is_refused_before_running(self, compile_knapsack):
        compiled = compile_knapsack()
        backend = SimulatedBifurcationBackend(max_variables=compiled.num_variables - 1)
        with pytest.raises(SolverExecutionError) as excinfo:
            backend.solve(compiled, SolverPreferences(num_reads=10, seed=1))
        assert excinfo.value.code == "SB_VARIABLE_LIMIT"
        assert excinfo.value.status == "resource_limit_exceeded"
        # The same wording the service uses from the declaration.
        assert str(excinfo.value) == (
            f"Compiled problem has {compiled.num_variables} variables (including "
            f"internal), exceeding the simulated_bifurcation backend limit of "
            f"{compiled.num_variables - 1}"
        )

    def test_exactly_the_cap_is_accepted(self, compile_knapsack):
        compiled = compile_knapsack()
        backend = SimulatedBifurcationBackend(max_variables=compiled.num_variables)
        result = backend.solve(compiled, SolverPreferences(num_reads=10, seed=1))
        assert result.num_samples == 10

    def test_modes_both_solve_and_differ(self, compile_knapsack):
        compiled = compile_knapsack()
        backend = SimulatedBifurcationBackend()
        discrete = backend.solve(
            compiled,
            SolverPreferences(
                num_reads=50, seed=5, simulated_bifurcation={"mode": "discrete"}
            ),
        )
        ballistic = backend.solve(
            compiled,
            SolverPreferences(
                num_reads=50, seed=5, simulated_bifurcation={"mode": "ballistic"}
            ),
        )
        assert discrete.num_samples == ballistic.num_samples == 50
        # Same seed, same start; a different force law lands elsewhere.
        assert discrete.samples.tolist() != ballistic.samples.tolist()

    def test_no_option_block_means_discrete(self, compile_knapsack):
        compiled = compile_knapsack()
        backend = SimulatedBifurcationBackend()
        implicit = backend.solve(compiled, SolverPreferences(num_reads=30, seed=9))
        explicit = backend.solve(
            compiled,
            SolverPreferences(
                num_reads=30, seed=9, simulated_bifurcation={"mode": "discrete"}
            ),
        )
        assert implicit.samples.tolist() == explicit.samples.tolist()

    @pytest.mark.parametrize("mode", ["discrete", "ballistic"])
    def test_finds_the_optimum_of_a_dense_instance(self, load_knapsack, mode):
        # 14 variables: the exact solver is the oracle. Dense +-1 couplings
        # with a random field is SB's home ground.
        bqm = self.dense_bqm(14, seed=2)
        compiled = self.compiled_from_bqm(bqm, load_knapsack())
        optimum = dimod.ExactSolver().sample(bqm).first.energy
        result = SimulatedBifurcationBackend().solve(
            compiled,
            SolverPreferences(
                num_reads=100, seed=11, simulated_bifurcation={"mode": mode}
            ),
        )
        assert result.energies.min() == pytest.approx(optimum)

    def test_linear_only_model_is_solved_exactly(self, load_knapsack):
        # No couplings: the field is the only scale, and the optimum is the
        # sign of every field. Every read should land on it.
        bqm = dimod.BinaryQuadraticModel(
            {"a": 3.0, "b": -2.0, "c": 0.5, "d": -4.0}, {}, 1.0, dimod.BINARY
        )
        compiled = self.compiled_from_bqm(bqm, load_knapsack())
        result = SimulatedBifurcationBackend().solve(
            compiled, SolverPreferences(num_reads=20, seed=1)
        )
        best = {"a": 0, "b": 1, "c": 0, "d": 1}
        assert result.variables == ["a", "b", "c", "d"]
        assert result.samples.tolist() == [[best[v] for v in result.variables]] * 20
        assert result.energies.tolist() == [-5.0] * 20

    def test_constant_model_solves_with_equal_energies(self, load_knapsack):
        # Neither couplings nor field: every sample is optimal and the
        # energy is the offset. Nothing to optimise, and nothing to raise.
        bqm = dimod.BinaryQuadraticModel({"a": 0.0, "b": 0.0}, {}, 2.5, dimod.BINARY)
        compiled = self.compiled_from_bqm(bqm, load_knapsack())
        result = SimulatedBifurcationBackend().solve(
            compiled, SolverPreferences(num_reads=8, seed=1)
        )
        assert result.samples.shape == (8, 2)
        assert result.energies.tolist() == [2.5] * 8

    def test_batch_layout(self):
        assert _batch_sizes(0) == []
        assert _batch_sizes(1) == [1]
        assert _batch_sizes(AGENTS_PER_BATCH) == [AGENTS_PER_BATCH]
        assert _batch_sizes(AGENTS_PER_BATCH + 1) == [AGENTS_PER_BATCH, 1]
        assert _batch_sizes(3 * AGENTS_PER_BATCH) == [AGENTS_PER_BATCH] * 3

    def test_initial_states_do_not_depend_on_the_batch_layout(self):
        # Drawing 1024 + 7 reads in two batches consumes the generator
        # exactly as drawing 1031 at once: read k's start is a function of
        # (seed, k) alone.
        n = 5
        rng = np.random.default_rng(np.random.SeedSequence(42))
        first, _ = _initial_states(rng, n, AGENTS_PER_BATCH)
        second, _ = _initial_states(rng, n, 7)
        rng = np.random.default_rng(np.random.SeedSequence(42))
        whole, momenta = _initial_states(rng, n, AGENTS_PER_BATCH + 7)

        assert whole.shape == (n, AGENTS_PER_BATCH + 7)
        assert whole.dtype == np.float32
        assert np.array_equal(np.concatenate([first, second], axis=1), whole)
        assert not momenta.any()

    def test_dense_ising_is_the_models_spin_form(self):
        bqm = dimod.BinaryQuadraticModel(
            {"a": 1.0, "b": -2.0}, {("a", "b"): 4.0}, 0.5, dimod.BINARY
        )
        variables, field, couplings, gain = _dense_ising(bqm)
        spin = bqm.change_vartype(dimod.SPIN, inplace=False)
        # dimod's own spin form is the oracle for the binary-to-spin
        # arithmetic; both arrays come back divided by the largest magnitude.
        peak = max(abs(spin.linear["a"]), abs(spin.linear["b"]),
                   abs(spin.quadratic[("a", "b")]))

        assert variables == ["a", "b"]
        assert field.dtype == np.float32 and couplings.dtype == np.float32
        assert field.tolist() == pytest.approx(
            [spin.linear["a"] / peak, spin.linear["b"] / peak]
        )
        # J[i, j] = -q_ij, symmetric, zero diagonal: the force J s - h is
        # minus the gradient of  -1/2 s^T J s + h . s.
        q = spin.quadratic[("a", "b")] / peak
        assert np.allclose(couplings, [[0.0, -q], [-q, 0.0]])
        # c0 = 0.5 / (rms sqrt(N)) on the rescaled couplings: the two
        # off-diagonal entries are +-q, so their RMS is |q|.
        assert gain == pytest.approx(0.5 / (abs(q) * np.sqrt(2)))

    def test_rescaling_makes_huge_biases_safe(self, load_knapsack):
        # Finite float64 biases far outside float32's range (2026-09-17
        # review): uniformly rescaled before the cast, so the dynamics see
        # the same problem and the optimum is still found.
        bqm = dimod.BinaryQuadraticModel(
            {"a": 3e38, "b": -2e38, "c": 5e37}, {("a", "b"): -4e38}, 0.0, dimod.BINARY
        )
        compiled = self.compiled_from_bqm(bqm, load_knapsack())
        optimum = dimod.ExactSolver().sample(bqm).first.energy
        result = SimulatedBifurcationBackend().solve(
            compiled, SolverPreferences(num_reads=50, seed=1)
        )
        assert np.isfinite(result.energies).all()
        assert result.energies.min() == pytest.approx(optimum)

    def test_a_non_finite_state_is_a_solver_error(self, monkeypatch, compile_knapsack):
        # The guard behind the rescaling: a NaN state must never become a
        # silent all-zero sample.
        import annealbridge.solvers.simulated_bifurcation as module

        def nan_run(xp, couplings, field, x, y, steps, c0, discrete):
            x[...] = np.nan
            return x

        monkeypatch.setattr(module, "_run", nan_run)
        with pytest.raises(SolverExecutionError, match="non-finite state"):
            SimulatedBifurcationBackend().solve(
                compile_knapsack(), SolverPreferences(num_reads=5, seed=1)
            )

    def test_declares_its_cap_for_the_service_and_recommend(self, compile_knapsack):
        # The cap is in the declaration, so the service refuses before
        # compiling and the capabilities view reports it as max_variables.
        backend = SimulatedBifurcationBackend(max_variables=37)
        limit = backend.capabilities.compiled_variable_limit
        assert limit is not None
        assert limit.maximum == 37
        assert limit.error_code == "SB_VARIABLE_LIMIT"
        assert SimulatedBifurcationBackend().capabilities.compiled_variable_limit.maximum == 10_000
        # Each instance declares its own; the module declaration is untouched.
        assert _CAPABILITIES.compiled_variable_limit is None

    def test_logs_the_run_parameters(self, compile_knapsack, caplog):
        compiled = compile_knapsack()
        with caplog.at_level(
            logging.INFO, logger="annealbridge.solvers.simulated_bifurcation"
        ):
            SimulatedBifurcationBackend().solve(
                compiled,
                SolverPreferences(
                    num_reads=AGENTS_PER_BATCH + 1,
                    num_sweeps=50,
                    seed=4,
                    simulated_bifurcation={"mode": "ballistic"},
                ),
            )
        [record] = [r for r in caplog.records if "solved problem" in r.getMessage()]
        message = record.getMessage()
        assert "Backend simulated_bifurcation solved problem knapsack" in message
        assert (
            f"num_reads={AGENTS_PER_BATCH + 1}, num_sweeps=50, seed=4, "
            "mode=ballistic, device=cpu, batches=2" in message
        )


class TestSimulatedBifurcationCudaDevice:
    """The ``cuda`` device: PyTorch is imported lazily and only on that
    setting, and a device that is not usable is reported, never replaced
    by the CPU. CI has no GPU, so the torch seen here is a fake built on
    numpy -- which also proves the dynamics routine is device-agnostic:
    the same request gives the same samples on the fake device."""

    @staticmethod
    def _install_fake_torch(monkeypatch, *, cuda_available: bool) -> types.ModuleType:
        class FakeTensor(np.ndarray):
            def to(self, device):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.asarray(self)

        torch = types.ModuleType("torch")
        torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
        torch.sign = np.sign
        torch.abs = np.abs
        torch.matmul = np.matmul
        torch.isfinite = np.isfinite
        torch.from_numpy = lambda array: array.view(FakeTensor)
        torch.device = lambda name: name
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
        monkeypatch.setitem(sys.modules, "torch", torch)
        return torch

    @staticmethod
    def _uninstall_torch(monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, package=None):
            if name == "torch":
                return None
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    def test_cpu_device_never_looks_for_torch(self, monkeypatch, compile_knapsack):
        self._uninstall_torch(monkeypatch)
        backend = SimulatedBifurcationBackend(device="cpu")
        assert backend.is_available() == AvailabilityStatus(category="available")
        result = backend.solve(compile_knapsack(), SolverPreferences(num_reads=5))
        assert result.num_samples == 5

    def test_torch_not_installed_is_reported(self, monkeypatch):
        self._uninstall_torch(monkeypatch)
        status = SimulatedBifurcationBackend(device="cuda").is_available()
        assert status == AvailabilityStatus(
            category="not_installed",
            detail="PyTorch not installed (annealbridge[gpu] extra)",
        )
        assert status.available is False

    def test_no_cuda_device_is_reported(self, monkeypatch):
        self._install_fake_torch(monkeypatch, cuda_available=False)
        status = SimulatedBifurcationBackend(device="cuda").is_available()
        assert status == AvailabilityStatus(
            category="unavailable", detail="no CUDA device visible to PyTorch"
        )

    def test_cuda_device_is_available_and_solves_like_the_cpu(
        self, monkeypatch, compile_knapsack
    ):
        self._install_fake_torch(monkeypatch, cuda_available=True)
        compiled = compile_knapsack()
        preferences = SolverPreferences(num_reads=AGENTS_PER_BATCH + 3, seed=8)
        gpu = SimulatedBifurcationBackend(device="cuda")
        cpu = SimulatedBifurcationBackend(device="cpu")

        assert gpu.is_available() == AvailabilityStatus(category="available")
        on_gpu = gpu.solve(compiled, preferences)
        on_cpu = cpu.solve(compiled, preferences)

        assert on_gpu.samples.dtype == np.int8
        assert on_gpu.samples.tolist() == on_cpu.samples.tolist()
        assert on_gpu.energies.tolist() == on_cpu.energies.tolist()

    def test_a_failing_device_is_a_solver_error(self, monkeypatch, compile_knapsack):
        torch = self._install_fake_torch(monkeypatch, cuda_available=True)

        def broken(array):
            raise RuntimeError("CUDA out of memory")

        torch.from_numpy = broken
        with pytest.raises(SolverExecutionError, match="CUDA out of memory"):
            SimulatedBifurcationBackend(device="cuda").solve(
                compile_knapsack(), SolverPreferences(num_reads=5, seed=1)
            )


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

        assert len(backends) == 8
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
