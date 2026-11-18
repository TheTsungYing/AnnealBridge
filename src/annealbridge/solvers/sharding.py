"""Read sharding shared by the local samplers (spec §21).

A local sampler's reads are split into fixed-size shards so several can be
sampled at once. The split, the per-shard seeds and the merge live here
because every local backend that samples repeatedly needs exactly the same
rules: the layout must depend on ``(num_reads, seed)`` alone, so a seeded
result is the same whatever the worker count or the machine.

:func:`sample_reads` / :func:`sample_reads_interruptible` are that layout
decision -- one plain call for one shard, derived seeds for several -- and
:func:`local_sampler_result` is the result every such backend builds from
the merged sample set. A backend supplies only what is its own: the sampler
call, its seed rule and its log line.
"""

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import dimod
import numpy as np

from annealbridge.models import CompiledProblem, SolverExecutionMetadata
from annealbridge.solvers.base import (
    RawSolverResult,
    empty_result,
    result_from_sampleset,
)

# Reads are sampled in fixed-size shards so several can run at once. The
# shard size is part of the reproducibility contract: the shard layout and
# every shard's seed depend on ``num_reads`` and ``seed`` alone, never on
# the machine, so the merged sample set is the same whatever the worker
# count. Changing this constant changes every seeded result with more than
# one shard. 25 is still the measured balance between the extra cost on one
# core and the parallel speed-up on eight. That extra cost is a fixed cost
# per sampler call, independent of reads. Measured on the simulated
# annealer, where every shard re-runs the sampler's default beta-range
# estimate (about 15 ms at 300 variables / 13.6k interactions, 110 ms at
# 800 / 96k): with workers=1 and 400 reads (16 shards) it was +9–18 % at
# 1000 sweeps and +52–73 % at 200. The tabu sampler has no such per-call
# setup, so for it the shard size is only the unit of parallelism.
READS_PER_SHARD = 25

# The name prefix of the shard pool's worker threads.
SHARD_THREAD_PREFIX = "annealbridge-shard"

_T = TypeVar("_T")


def default_workers() -> int:
    """CPUs available to this process, never below 1.

    ``os.process_cpu_count`` (3.13+) and ``sched_getaffinity`` (Linux)
    honour the affinity mask; ``os.cpu_count`` is the portable fallback.
    None of them sees a container CPU quota, which is what the per-backend
    worker settings (``ANNEALBRIDGE_SA_WORKERS``,
    ``ANNEALBRIDGE_TABU_WORKERS``) are for.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return max(1, count or 1)


def shard_sizes(num_reads: int) -> list[int]:
    """Split ``num_reads`` into ``READS_PER_SHARD``-sized shards, remainder last.

    Empty for ``num_reads <= 0``; the caller then makes the plain single
    call so the sampler reports the invalid count exactly as before.
    """
    if num_reads <= 0:
        return []
    full, rest = divmod(num_reads, READS_PER_SHARD)
    return [READS_PER_SHARD] * full + ([rest] if rest else [])


def shard_seeds(seed: int | None, count: int) -> list[int | None]:
    """One sampler seed per shard, derived from the user's seed.

    ``numpy.random.SeedSequence`` spreads one seed into independent streams
    deterministically and platform-independently; masking to 31 bits keeps
    every derived value inside the accepted range of *every* local sampler
    at once — the simulated annealer takes ``0 <= seed < 2**31`` and the
    tabu sampler ``0 <= seed <= 2**32 - 1``, and a 31-bit value satisfies
    both. Two distinct user seeds do not systematically share shard seeds
    the way ``seed + k`` would; a coincidence is still possible -- the
    derived values are 31-bit -- but only with negligible probability, not
    excluded by construction. ``None`` stays ``None``: each shard then draws
    its own random seed.
    """
    if seed is None:
        return [None] * count
    states = np.random.SeedSequence(seed).generate_state(count, dtype=np.uint32)
    return [int(value & 0x7FFF_FFFF) for value in states]


def run_shards(
    run: Callable[[int], dimod.SampleSet],
    count: int,
    workers: int,
) -> dimod.SampleSet:
    """Sample every shard, ``workers`` at a time, and concatenate in order.

    ``run`` is called with the shard index and returns that shard's sample
    set; the merged row order is the shard order, whatever the worker count.

    Every worker thread has returned when this does: a failed shard makes
    the shards not yet started return at once, and the failure is raised
    only after the shards already running have ended.
    """
    # With no stop, a shard is only skipped after another one failed, and
    # the failure is then raised; a return therefore has every sample set.
    return dimod.concatenate(_run_guarded(run, count, workers, _never_stop))


def run_shards_interruptible(
    run: Callable[[int], dimod.SampleSet],
    count: int,
    workers: int,
    should_stop: Callable[[], bool],
) -> tuple[dimod.SampleSet | None, bool]:
    """:func:`run_shards` that skips every shard not started before a stop.

    ``should_stop`` is polled as each shard is about to start; a shard that
    has started runs to the end (``run`` itself may stop early -- the
    annealer's per-read callback does -- and reports that through the rows
    it returns). Returns the shards that ran, concatenated in shard order,
    or None when none did, and whether any shard was skipped. With
    ``should_stop`` never true the merged set is exactly
    :func:`run_shards`'s.

    Every worker thread has returned when this does, on every path: a stop
    only makes the remaining shards return at once, and a failure is
    handled exactly as in :func:`run_shards`.
    """
    samplesets = _run_guarded(run, count, workers, should_stop)
    ran = [sampleset for sampleset in samplesets if sampleset is not None]
    merged = dimod.concatenate(ran) if ran else None
    return merged, len(ran) < count


def sample_reads(
    sample: Callable[[int, int | None], dimod.SampleSet],
    num_reads: int,
    seed: int | None,
    workers: int,
    check_seed: Callable[[int | None], None],
) -> dimod.SampleSet:
    """``num_reads`` reads through ``sample(num_reads, seed)``, sharded.

    One shard (or an invalid count) is the plain call with the user's seed
    untouched, so the sampler's own validation and a pre-sharding result
    both stay exactly as they were. Several shards run ``workers`` at a time
    with seeds derived by :func:`shard_seeds`; ``check_seed`` -- the
    backend's copy of its sampler's seed rule, which raises for a seed out
    of range -- runs first, because a derived seed is always in range and
    would otherwise hide a seed the single-shard path rejects.
    """
    sizes = shard_sizes(num_reads)
    if len(sizes) <= 1:
        return sample(num_reads, seed)
    check_seed(seed)
    seeds = shard_seeds(seed, len(sizes))
    return run_shards(
        lambda shard: sample(sizes[shard], seeds[shard]), len(sizes), workers
    )


def sample_reads_interruptible(
    sample: Callable[[int, int | None], dimod.SampleSet],
    num_reads: int,
    seed: int | None,
    workers: int,
    should_stop: Callable[[], bool],
    check_seed: Callable[[int | None], None],
) -> tuple[dimod.SampleSet | None, bool]:
    """:func:`sample_reads`, stoppable; returns the reads that completed
    (None when none did) and whether any read asked for is missing.

    The layout is exactly :func:`sample_reads`'s, so a run ``should_stop``
    never stops returns the same reads. ``should_stop`` is asked before
    each shard starts (before the one call, for one shard); a shard not
    started by then is skipped. A started ``sample`` may itself return
    fewer rows than it was asked for -- a sampler with a per-read interrupt
    callback does -- and that counts as missing reads too.
    """
    sizes = shard_sizes(num_reads)
    if len(sizes) <= 1:
        if should_stop():
            return None, True
        sampleset = sample(num_reads, seed)
        return sampleset, len(sampleset) < num_reads
    check_seed(seed)
    seeds = shard_seeds(seed, len(sizes))
    merged, skipped = run_shards_interruptible(
        lambda shard: sample(sizes[shard], seeds[shard]),
        len(sizes),
        workers,
        should_stop,
    )
    cut = skipped or (merged is not None and len(merged) < sum(sizes))
    return merged, cut


def shard_count(num_reads: int) -> int:
    """Sampler calls a request of ``num_reads`` makes; an invalid count is
    still the one plain call."""
    return max(1, len(shard_sizes(num_reads)))


def local_sampler_result(
    sampleset: dimod.SampleSet | None,
    interrupted: bool,
    compiled_problem: CompiledProblem,
    backend: str,
    num_reads: int,
) -> RawSolverResult:
    """The result of a local sampler run; ``sampleset`` None means an
    interrupt stopped the run before any read completed.

    A local run leaves no vendor facts behind: no solver id, no timing, no
    quota. Reported are the backend, that it ran here, and the read count
    asked of the sampler -- the shard layout is an implementation detail
    and is not part of the output.
    """
    metadata = SolverExecutionMetadata(
        backend=backend,
        remote=False,
        num_reads_requested=num_reads,
    )
    if sampleset is None:
        return empty_result(compiled_problem.model, backend=backend, metadata=metadata)
    result = result_from_sampleset(sampleset, backend=backend, metadata=metadata)
    result.interrupted = interrupted
    return result


def _never_stop() -> bool:
    return False


def _run_guarded(
    run: Callable[[int], dimod.SampleSet],
    count: int,
    workers: int,
    should_stop: Callable[[], bool],
) -> list[dimod.SampleSet | None]:
    """Every shard's sample set, or None for a shard skipped before it started.

    A shard is skipped when ``should_stop`` is true or another shard has
    already failed. The check sits between shards only: a started ``run``
    is never handed anything new, so a seeded shard samples exactly as it
    would alone.
    """
    failed = threading.Event()

    def guarded(shard: int) -> dimod.SampleSet | None:
        if failed.is_set() or should_stop():
            return None
        try:
            return run(shard)
        except BaseException:
            failed.set()
            raise

    return _map_shards(guarded, count, workers)


def _map_shards(
    run: Callable[[int], _T],
    count: int,
    workers: int,
) -> list[_T]:
    """``[run(0), ..., run(count - 1)]``, ``workers`` shards at a time.

    On a failure the shards not yet started are cancelled and the running
    ones are waited for, so no worker thread outlives the call.
    """
    if workers == 1:
        return [run(shard) for shard in range(count)]
    # Named so a test (or an operator reading a thread dump) can tell this
    # pool's threads apart from everything else running in the process.
    executor = ThreadPoolExecutor(
        max_workers=min(workers, count), thread_name_prefix=SHARD_THREAD_PREFIX
    )
    try:
        # ``map`` yields in shard order, so the merged row order is
        # the same as a sequential run.
        results = list(executor.map(run, range(count)))
    except BaseException:
        # A failed shard fails the solve, but only once the shards already
        # running have ended: the service then releases the solve's
        # concurrency slot with no shard of it still using a CPU.
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return results
