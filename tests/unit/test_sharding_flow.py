"""The sampling flow the local sampler backends share (code review 2026-11-18, R-14).

``sample_reads`` / ``sample_reads_interruptible`` decide the layout -- one
plain call for one shard, derived seeds for several -- and
``local_sampler_result`` builds the result. These tests drive them with a
scripted ``sample`` so the layout itself is what is checked; the two real
backends are covered in ``test_solvers.py`` and ``test_solver_interrupt.py``.
"""

import dimod
import numpy as np
import pytest

from annealbridge.models import CompiledProblem, OptimizationProblem
from annealbridge.solvers.sharding import (
    READS_PER_SHARD,
    local_sampler_result,
    sample_reads,
    sample_reads_interruptible,
    shard_count,
    shard_seeds,
)

BQM = dimod.BinaryQuadraticModel({"a": 1.0, "b": -1.0}, {("a", "b"): 0.5}, 0.0, dimod.BINARY)


class Recorder:
    """A ``sample`` that records its calls and returns ``num_reads`` rows,
    or only ``short`` of them when set (a sampler stopped part-way)."""

    def __init__(self, short: int | None = None) -> None:
        self.calls: list[tuple[int, int | None]] = []
        self.short = short

    def __call__(self, num_reads: int, seed: int | None) -> dimod.SampleSet:
        self.calls.append((num_reads, seed))
        rows = num_reads if self.short is None else min(self.short, num_reads)
        samples = np.zeros((rows, 2), dtype=np.int8)
        return dimod.SampleSet.from_samples_bqm((samples, ["a", "b"]), BQM)


def accept(seed: int | None) -> None:
    """A seed rule that accepts everything."""


def reject(seed: int | None) -> None:
    raise ValueError("seed rejected")


def never() -> bool:
    return False


class StopFrom:
    """``should_stop`` that answers True from the ``k``-th call on."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.k


class TestSampleReads:
    @pytest.mark.parametrize("num_reads", [1, READS_PER_SHARD])
    def test_one_shard_is_the_plain_call_with_the_seed_untouched(self, num_reads):
        sample = Recorder()

        merged = sample_reads(sample, num_reads, 7, 4, reject)

        # The seed rule is the sampler's own business on this path.
        assert sample.calls == [(num_reads, 7)]
        assert len(merged) == num_reads

    @pytest.mark.parametrize("num_reads", [0, -3])
    def test_an_invalid_count_is_still_the_one_plain_call(self, num_reads):
        # The sampler then reports the invalid count itself; here it is a
        # stand-in that only records what it was asked for.
        calls: list[tuple[int, int | None]] = []
        answer = object()

        def sample(reads: int, seed: int | None) -> object:
            calls.append((reads, seed))
            return answer

        assert sample_reads(sample, num_reads, 7, 4, reject) is answer
        assert calls == [(num_reads, 7)]

    def test_several_shards_use_the_derived_seeds_in_shard_order(self):
        sample = Recorder()

        merged = sample_reads(sample, 60, 7, 1, accept)

        assert sample.calls == list(zip([25, 25, 10], shard_seeds(7, 3)))
        assert len(merged) == 60

    def test_no_seed_stays_no_seed_in_every_shard(self):
        sample = Recorder()

        sample_reads(sample, 60, None, 1, accept)

        assert sample.calls == [(25, None), (25, None), (10, None)]

    def test_the_seed_rule_runs_before_any_shard(self):
        sample = Recorder()

        with pytest.raises(ValueError, match="seed rejected"):
            sample_reads(sample, 60, 7, 1, reject)

        assert sample.calls == []

    @pytest.mark.parametrize("workers", [1, 2, 8])
    def test_the_calls_do_not_depend_on_the_worker_count(self, workers):
        sample = Recorder()

        sample_reads(sample, 110, 3, workers, accept)

        assert sorted(sample.calls) == sorted(zip([25, 25, 25, 25, 10], shard_seeds(3, 5)))


class TestSampleReadsInterruptible:
    @pytest.mark.parametrize("num_reads", [1, 25, 60, 110])
    def test_never_stopped_makes_exactly_the_uninterruptible_calls(self, num_reads):
        plain, stoppable = Recorder(), Recorder()

        sample_reads(plain, num_reads, 7, 1, accept)
        merged, cut = sample_reads_interruptible(stoppable, num_reads, 7, 1, never, accept)

        assert stoppable.calls == plain.calls
        assert merged is not None and len(merged) == num_reads
        assert cut is False

    def test_one_shard_stopped_before_it_starts_returns_nothing(self):
        sample = Recorder()

        merged, cut = sample_reads_interruptible(sample, 10, 7, 1, StopFrom(1), accept)

        assert sample.calls == []
        assert merged is None
        assert cut is True

    def test_one_shard_that_returns_fewer_reads_is_cut(self):
        sample = Recorder(short=4)

        merged, cut = sample_reads_interruptible(sample, 10, 7, 1, never, accept)

        assert merged is not None and len(merged) == 4
        assert cut is True

    def test_shards_not_started_at_the_stop_are_skipped(self):
        sample = Recorder()
        should_stop = StopFrom(2)

        merged, cut = sample_reads_interruptible(sample, 60, 7, 1, should_stop, accept)

        assert sample.calls == [(25, shard_seeds(7, 3)[0])]
        assert merged is not None and len(merged) == 25
        assert cut is True
        assert should_stop.calls == 3  # asked once before each shard

    def test_every_shard_skipped_returns_nothing(self):
        sample = Recorder()

        merged, cut = sample_reads_interruptible(sample, 60, 7, 1, StopFrom(1), accept)

        assert sample.calls == []
        assert merged is None
        assert cut is True

    def test_a_started_shard_that_returns_fewer_reads_is_cut(self):
        # No shard was skipped, but every one came back short.
        sample = Recorder(short=20)

        merged, cut = sample_reads_interruptible(sample, 60, 7, 1, never, accept)

        assert len(sample.calls) == 3
        assert merged is not None and len(merged) == 50
        assert cut is True

    def test_the_seed_rule_runs_before_any_shard(self):
        sample = Recorder()

        with pytest.raises(ValueError, match="seed rejected"):
            sample_reads_interruptible(sample, 60, 7, 1, never, reject)

        assert sample.calls == []

    def test_one_shard_leaves_the_seed_rule_to_the_sampler(self):
        sample = Recorder()

        sample_reads_interruptible(sample, 10, 7, 1, never, reject)

        assert sample.calls == [(10, 7)]


class TestShardCount:
    @pytest.mark.parametrize(
        ("num_reads", "expected"),
        [(-3, 1), (0, 1), (1, 1), (25, 1), (26, 2), (60, 3), (100, 4)],
    )
    def test_counts_the_sampler_calls(self, num_reads, expected):
        assert shard_count(num_reads) == expected


class TestLocalSamplerResult:
    @staticmethod
    def compiled() -> CompiledProblem:
        problem = OptimizationProblem.model_validate(
            {
                "version": "1.0",
                "name": "flow",
                "variables": [
                    {"name": "a", "type": "binary"},
                    {"name": "b", "type": "binary"},
                ],
                "objective": {
                    "direction": "minimize",
                    "linear_terms": [
                        {"variable": "a", "coefficient": 1.0},
                        {"variable": "b", "coefficient": -1.0},
                    ],
                },
                "constraints": [],
            }
        )
        return CompiledProblem(
            model=BQM,
            original_problem=problem,
            internal_variables=set(),
            constraint_trace=[],
            hard_penalty=None,
            objective_scale=1.0,
            num_variables=BQM.num_variables,
            num_interactions=BQM.num_interactions,
        )

    def test_a_sample_set_becomes_the_reads_with_local_metadata(self):
        sampleset = Recorder()(3, None)

        result = local_sampler_result(sampleset, False, self.compiled(), "local", 3)

        assert result.variables == ["a", "b"]
        assert result.samples.shape == (3, 2)
        assert result.backend == "local"
        assert result.interrupted is False
        assert result.metadata.backend == "local"
        assert result.metadata.remote is False
        assert result.metadata.num_reads_requested == 3

    def test_the_interrupted_flag_is_carried(self):
        sampleset = Recorder()(2, None)

        result = local_sampler_result(sampleset, True, self.compiled(), "local", 5)

        assert result.interrupted is True
        assert result.metadata.num_reads_requested == 5

    def test_no_sample_set_is_an_empty_interrupted_result(self):
        # ``interrupted`` is not consulted: nothing completing is itself the cut.
        result = local_sampler_result(None, False, self.compiled(), "local", 5)

        assert result.variables == ["a", "b"]
        assert result.samples.shape == (0, 2)
        assert result.samples.dtype == np.int8
        assert result.interrupted is True
        assert result.metadata.num_reads_requested == 5
