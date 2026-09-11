"""Simulated annealing solver backend (spec §21)."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import (
    AvailabilityStatus,
    BackendAliases,
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_arrays,
)

logger = logging.getLogger(__name__)

# Reads are sampled in fixed-size shards so several can run at once. The
# shard size is part of the reproducibility contract: the shard layout and
# every shard's seed depend on ``num_reads`` and ``seed`` alone, never on
# the machine, so the merged sample set is the same whatever the worker
# count. Changing this constant changes every seeded result with more than
# one shard. 25 measured as the best balance between the per-call Python
# overhead on one core (+4–15 %) and the parallel speed-up on eight.
READS_PER_SHARD = 25

# ``SimulatedAnnealingSampler`` accepts ``0 <= seed < 2**31``. Checked here
# so a request rejects the same way whether it takes one shard (seed passed
# through) or several (shard seeds derived from it).
_SEED_LIMIT = 2**31

_CAPABILITIES = SolverCapabilities(
    name="simulated_annealing",
    remote=False,
    heuristic=True,
    exhaustive=False,
    supports_seed=True,
    supports_num_reads=True,
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    supports_num_sweeps=True,
    description=(
        "Local heuristic simulated-annealing sampler; scales to larger "
        "problems but does not prove optimality or infeasibility."
    ),
    # 2026-09-09 review (F-07): reads and sweeps bound the CPU time one
    # request can hold a concurrency slot for. Declared under their own
    # keys (``local_reads`` rather than the QPU quota key ``reads``) so the
    # policy can give local and remote sampling different ceilings.
    parameter_limits=[
        ParameterLimit(
            preference="num_reads", limit="local_reads", error_code="LOCAL_READS_LIMIT"
        ),
        ParameterLimit(
            preference="num_sweeps", limit="sweeps", error_code="SWEEPS_LIMIT"
        ),
    ],
)


def default_workers() -> int:
    """CPUs available to this process, never below 1.

    ``os.process_cpu_count`` (3.13+) and ``sched_getaffinity`` (Linux)
    honour the affinity mask; ``os.cpu_count`` is the portable fallback.
    None of them sees a container CPU quota, which is what
    ``ANNEALBRIDGE_SA_WORKERS`` is for.
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
    every derived value inside the sampler's accepted range. Distinct user
    seeds never share a shard seed by construction (as ``seed + k`` would).
    ``None`` stays ``None``: each shard then draws its own random seed.
    """
    if seed is None:
        return [None] * count
    states = np.random.SeedSequence(seed).generate_state(count, dtype=np.uint32)
    return [int(value & 0x7FFF_FFFF) for value in states]


class SimulatedAnnealingBackend(BackendAliases):
    """Samples the compiled problem with D-Wave's simulated annealer.

    All reads are kept (never just ``.first``) so every candidate reaches the
    solution validator. The seed is passed to the sampler only; global random
    state is never touched by this backend (spec §37) — with ``seed=None``
    the sampler draws its own seed, once per shard.

    Reads are sampled in ``READS_PER_SHARD``-sized shards, up to ``workers``
    of them concurrently (the sampler's C++ loop releases the GIL and keeps
    its RNG thread-local). The shard layout and shard seeds are a function of
    ``(num_reads, seed)`` only, so the result is a function of
    ``(problem, num_reads, num_sweeps, seed)`` and identical for any worker
    count; ``workers`` changes wall time and nothing else. A request of at
    most ``READS_PER_SHARD`` reads is one shard and passes the user's seed
    straight through, exactly as before sharding existed.
    """

    def __init__(self, workers: int | None = None) -> None:
        """``workers``: shards sampled at once; ``None`` detects the CPUs
        available to the process. Never clamped: below 1 is a caller bug."""
        if workers is None:
            workers = default_workers()
        elif workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        self._workers = workers
        self._sampler = SimulatedAnnealingSampler()

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    @property
    def workers(self) -> int:
        """Maximum shards sampled concurrently."""
        return self._workers

    def is_available(self) -> AvailabilityStatus:
        """Local backend, always available. No network I/O."""
        return AvailabilityStatus(category="available")

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        """Run simulated annealing, returning all reads as samples."""
        sizes = shard_sizes(preferences.num_reads)
        try:
            if len(sizes) <= 1:
                # One shard (or an invalid count): the plain call, seed
                # untouched, so the sampler's own validation and the
                # pre-sharding results both stay exactly as they were.
                sampleset = self._sample(
                    compiled_problem.model,
                    preferences.num_reads,
                    preferences.num_sweeps,
                    preferences.seed,
                )
            else:
                sampleset = self._sample_sharded(compiled_problem.model, sizes, preferences)
        except Exception as exc:
            raise SolverExecutionError(
                f"Simulated annealing solver failed: {exc}"
            ) from exc

        variables, samples, energies = sampleset_to_arrays(sampleset)
        logger.info(
            "Backend %s solved problem %s: %d variables, %d samples "
            "(num_reads=%d, num_sweeps=%d, seed=%s, shards=%d, workers=%d)",
            self.name,
            compiled_problem.original_problem.name,
            compiled_problem.num_variables,
            len(samples),
            preferences.num_reads,
            preferences.num_sweeps,
            preferences.seed,
            max(1, len(sizes)),
            min(self._workers, max(1, len(sizes))),
        )
        return RawSolverResult(
            variables=variables,
            samples=samples,
            energies=energies,
            backend=self.name,
        )

    def _sample(
        self,
        model: dimod.BinaryQuadraticModel,
        num_reads: int,
        num_sweeps: int,
        seed: int | None,
    ) -> dimod.SampleSet:
        sample_kwargs: dict[str, int] = {"num_reads": num_reads, "num_sweeps": num_sweeps}
        if seed is not None:
            sample_kwargs["seed"] = seed
        return self._sampler.sample(model, **sample_kwargs)

    def _sample_sharded(
        self,
        model: dimod.BinaryQuadraticModel,
        sizes: list[int],
        preferences: SolverPreferences,
    ) -> dimod.SampleSet:
        """Sample every shard, ``workers`` at a time, and concatenate in order."""
        seed = preferences.seed
        if seed is not None and not 0 <= seed < _SEED_LIMIT:
            # The sampler's own rule, applied before derivation so an
            # out-of-range seed fails the same way for any num_reads.
            raise ValueError(
                f"'seed' should be an integer between 0 and {_SEED_LIMIT - 1}: "
                f"value = {seed}"
            )
        seeds = shard_seeds(seed, len(sizes))
        sweeps = preferences.num_sweeps

        def run(shard: int) -> dimod.SampleSet:
            return self._sample(model, sizes[shard], sweeps, seeds[shard])

        if self._workers == 1:
            samplesets = [run(shard) for shard in range(len(sizes))]
        else:
            executor = ThreadPoolExecutor(max_workers=min(self._workers, len(sizes)))
            try:
                # ``map`` yields in shard order, so the merged row order is
                # the same as a sequential run.
                samplesets = list(executor.map(run, range(len(sizes))))
            except BaseException:
                # A failed shard fails the solve; do not wait for the rest.
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            executor.shutdown(wait=True)
        return dimod.concatenate(samplesets)
