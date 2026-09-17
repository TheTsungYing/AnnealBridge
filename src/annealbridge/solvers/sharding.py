"""Read sharding shared by the local samplers (spec §21).

A local sampler's reads are split into fixed-size shards so several can be
sampled at once. The split, the per-shard seeds and the merge live here
because every local backend that samples repeatedly needs exactly the same
rules: the layout must depend on ``(num_reads, seed)`` alone, so a seeded
result is the same whatever the worker count or the machine.
"""

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import dimod
import numpy as np

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
    both. Distinct user seeds never share a shard seed by construction (as
    ``seed + k`` would). ``None`` stays ``None``: each shard then draws its
    own random seed.
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
    """
    if workers == 1:
        samplesets = [run(shard) for shard in range(count)]
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, count))
        try:
            # ``map`` yields in shard order, so the merged row order is
            # the same as a sequential run.
            samplesets = list(executor.map(run, range(count)))
        except BaseException:
            # A failed shard fails the solve; do not wait for the rest.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        executor.shutdown(wait=True)
    return dimod.concatenate(samplesets)
