"""Unit tests for the read sharding shared by the local samplers (spec §21).

The module is exercised here without a sampler: ``run_shards`` only has to
call ``run`` once per shard index and concatenate the results in shard
order, whatever the worker count. The backend-level tests in
``test_solvers.py`` cover what that means for a real sampler.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import dimod
import pytest

from annealbridge.solvers import sharding
from annealbridge.solvers.sharding import (
    READS_PER_SHARD,
    SHARD_THREAD_PREFIX,
    run_shards,
    shard_seeds,
    shard_sizes,
)


def _shard_sampleset(shard: int, rows: int = 1) -> dimod.SampleSet:
    """A recognisable sample set: every row's energy names its shard.

    The energies survive ``dimod.concatenate`` unchanged, so the merged
    energy column is a direct read-out of the merged row order.
    """
    return dimod.SampleSet.from_samples(
        [{"x": shard % 2}] * rows,
        vartype=dimod.BINARY,
        energy=[float(shard) + index / 100 for index in range(rows)],
    )


class _RecordingExecutor(ThreadPoolExecutor):
    """A real pool that records its size and every ``shutdown`` call."""

    def __init__(
        self,
        record: dict,
        max_workers: int | None = None,
        thread_name_prefix: str = "",
    ) -> None:
        self.record = record
        record["max_workers"] = max_workers
        record["thread_name_prefix"] = thread_name_prefix
        super().__init__(
            max_workers=max_workers, thread_name_prefix=thread_name_prefix
        )

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.record.setdefault("shutdown", []).append((wait, cancel_futures))
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


@pytest.fixture
def executor_record(monkeypatch):
    """Make ``run_shards`` build a recording pool instead of a plain one."""
    record: dict = {}
    monkeypatch.setattr(
        sharding,
        "ThreadPoolExecutor",
        lambda max_workers, thread_name_prefix="": _RecordingExecutor(
            record, max_workers, thread_name_prefix
        ),
    )
    return record


class TestShardSizes:
    """The split is ``READS_PER_SHARD``-sized shards with the remainder last."""

    def test_no_shards_below_one_read(self):
        # The caller makes the plain single call instead, so the sampler
        # reports the invalid count exactly as the vendor wrote it.
        assert shard_sizes(0) == []
        assert shard_sizes(-1) == []

    @pytest.mark.parametrize(
        ("num_reads", "expected"),
        [
            (1, [1]),
            (25, [25]),
            (26, [25, 1]),
            (50, [25, 25]),
            (51, [25, 25, 1]),
        ],
    )
    def test_full_shards_then_the_remainder(self, num_reads, expected):
        assert shard_sizes(num_reads) == expected

    def test_the_split_is_exact_and_never_empty_shards(self):
        for num_reads in range(1, 120):
            sizes = shard_sizes(num_reads)
            assert sum(sizes) == num_reads
            assert all(1 <= size <= READS_PER_SHARD for size in sizes)


class TestShardSeeds:
    """Derived seeds must fit every local sampler and depend on the user seed only."""

    def test_no_seed_stays_no_seed(self):
        # Each shard then draws its own random seed, as an unseeded plain
        # call would.
        assert shard_seeds(None, 0) == []
        assert shard_seeds(None, 3) == [None, None, None]

    @pytest.mark.parametrize("seed", [0, 1, 2**31 - 1, 2**32 - 1])
    def test_every_derived_seed_fits_the_strictest_sampler(self, seed):
        # Masked to 31 bits so a derived seed is in range for the simulated
        # annealer (< 2**31) as well as the tabu sampler (<= 2**32 - 1).
        seeds = shard_seeds(seed, 64)
        assert len(seeds) == 64
        assert all(0 <= value <= 2**31 - 1 for value in seeds)

    def test_the_same_seed_gives_the_same_shard_seeds(self):
        assert shard_seeds(7, 5) == shard_seeds(7, 5)

    def test_different_seeds_give_different_shard_seeds(self):
        # By construction, not by luck: SeedSequence spreads the user seed
        # rather than offsetting it, so neighbouring seeds do not overlap the
        # way ``seed + k`` would.
        assert shard_seeds(7, 5) != shard_seeds(8, 5)
        assert not set(shard_seeds(7, 5)) & set(shard_seeds(8, 5))


class TestRunShards:
    def test_sequential_run_calls_and_merges_in_shard_order(self):
        calls: list[int] = []

        def run(shard: int) -> dimod.SampleSet:
            calls.append(shard)
            return _shard_sampleset(shard)

        merged = run_shards(run, count=4, workers=1)

        assert calls == [0, 1, 2, 3]
        assert merged.record.energy.tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_every_read_of_every_shard_is_kept(self):
        merged = run_shards(lambda shard: _shard_sampleset(shard, rows=3), 3, workers=1)

        assert len(merged) == 9
        assert merged.record.energy.tolist() == [
            0.0, 0.01, 0.02, 1.0, 1.01, 1.02, 2.0, 2.01, 2.02
        ]

    def test_concurrent_run_merges_in_shard_order_not_completion_order(self):
        # The early shards finish last on purpose: if the merge followed
        # completion order the energies below would come out reversed.
        def run(shard: int) -> dimod.SampleSet:
            time.sleep(0.05 * (3 - shard))
            return _shard_sampleset(shard)

        merged = run_shards(run, count=4, workers=4)

        assert merged.record.energy.tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_a_concurrent_run_really_uses_several_threads(self):
        # Guards the premise of the ordering test above: with workers=1 the
        # pool is never built, so the shards must genuinely overlap here.
        seen: set[int] = set()
        lock = threading.Lock()

        def run(shard: int) -> dimod.SampleSet:
            with lock:
                seen.add(threading.get_ident())
            time.sleep(0.02)
            return _shard_sampleset(shard)

        run_shards(run, count=4, workers=4)

        assert len(seen) > 1

    def test_a_failed_shard_propagates_unchanged(self, executor_record):
        def run(shard: int) -> dimod.SampleSet:
            if shard == 1:
                raise RuntimeError("shard exploded")
            time.sleep(0.01)
            return _shard_sampleset(shard)

        with pytest.raises(RuntimeError) as exc_info:
            run_shards(run, count=6, workers=2)

        # The backend wraps this; run_shards itself must not translate it.
        assert type(exc_info.value) is RuntimeError
        assert str(exc_info.value) == "shard exploded"
        # And the pool is gone before the exception leaves: shut down without
        # waiting for the shards that are still queued. Batch 6 (J): only
        # run_shards_interruptible waits for running shards on a failure;
        # this uninterruptible path keeps raising at once.
        assert executor_record["shutdown"] == [(False, True)]

    def test_a_successful_run_shuts_the_pool_down(self, executor_record):
        run_shards(_shard_sampleset, count=4, workers=2)

        assert executor_record["shutdown"] == [(True, False)]

    def test_the_pool_threads_carry_the_shard_prefix(self, executor_record):
        # Batch 6 (J): named so a test (or a thread dump) can count the
        # shard threads apart from everything else in the process.
        names: list[str] = []
        lock = threading.Lock()

        def run(shard: int) -> dimod.SampleSet:
            with lock:
                names.append(threading.current_thread().name)
            return _shard_sampleset(shard)

        run_shards(run, count=4, workers=2)

        assert executor_record["thread_name_prefix"] == SHARD_THREAD_PREFIX
        assert SHARD_THREAD_PREFIX == "annealbridge-shard"
        assert names and all(
            name.startswith(SHARD_THREAD_PREFIX) for name in names
        )

    @pytest.mark.parametrize(
        ("count", "workers", "expected"),
        [(1, 4, 1), (3, 8, 3), (8, 3, 3), (4, 4, 4)],
    )
    def test_the_pool_is_never_larger_than_the_shard_count(
        self, executor_record, count, workers, expected
    ):
        # Idle threads for shards that do not exist: min(workers, count).
        run_shards(_shard_sampleset, count=count, workers=workers)

        assert executor_record["max_workers"] == expected

    def test_one_worker_builds_no_pool_at_all(self, executor_record):
        run_shards(_shard_sampleset, count=4, workers=1)

        assert executor_record == {}
