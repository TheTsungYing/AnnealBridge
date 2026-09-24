"""Tabu search solver backend (spec §21)."""

import logging

import dimod
from dwave.samplers import TabuSampler

from annealbridge.exceptions import SolverExecutionError
from annealbridge.interrupt import Interrupt
from annealbridge.models import (
    CompiledProblem,
    SolverExecutionMetadata,
    SolverPreferences,
)
from annealbridge.solvers.base import (
    AvailabilityStatus,
    BackendAliases,
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
    empty_result,
    log_solved,
    result_from_sampleset,
)

# The same shard layout, derived shard seeds and bounded fan-out the
# simulated annealer uses; the reproducibility contract they carry is
# documented in that module.
from annealbridge.solvers.sharding import (
    default_workers,
    run_shards,
    run_shards_interruptible,
    shard_seeds,
    shard_sizes,
)

logger = logging.getLogger(__name__)

# ``TabuSampler`` accepts ``0 <= seed <= 2**32 - 1``; this copies that rule
# on purpose. It is *not* the simulated annealer's rule, which stops at
# ``2**31`` -- the two samplers in dwave-samplers disagree, so each backend
# declares its own limit rather than sharing one constant. The sharded path
# derives every shard seed through ``SeedSequence(seed)`` and masks it to 31
# bits, so a derived seed is always in range: unchecked, several shards
# would silently accept a seed that the single-shard path (seed passed
# straight to the sampler) rejects. Checking it first makes both paths
# accept the same range and raise the same exception type with the same
# message (the vendor's own wording, copied so the two paths cannot be told
# apart from the error).
# The seed tests in ``tests/unit/test_solvers.py`` pin both the vendor rule
# and the agreement of the two paths, and turn red if a dwave-samplers
# upgrade changes the rule.
_SEED_LIMIT = 2**32

# ``TabuSampler.sample`` defaults to ``timeout=20`` -- a 20 ms wall clock per
# read -- which is submitted explicitly as None instead. A wall-clock budget
# makes the result a function of the machine's speed and of whatever else is
# running on it: at 2000 variables the first plain tabu search is still cut
# off at 20 ms and returns best energy -25677 where the untimed search
# reaches -25951. Disabling it is what lets this backend promise the same
# result for the same seed, on any machine and at any worker count.
_TIMEOUT_MS = None

# With the wall clock gone, the work per read has to be bounded by a count
# instead, and ``num_restarts=0`` is that bound: one plain tabu search from
# one random starting point, costing the sampler's ``coefficient_z_first``
# variable updates -- ``max(c * n, 500000)`` with ``c`` 10000 up to 500
# variables and 25000 above (the vendor defaults) -- and nothing more. Measured
# single-threaded: 4 ms per read at 30 variables, 12 ms at 200, 88 ms at
# 800, 330 ms at 2000 -- predictable in the problem size, which a timeout is
# not. Restarts are not traded away for quality: on dense +-1 SK instances
# at 300 and 600 variables (five instances each), spending a fixed time
# budget on more restarts and on more reads reached the same best energy.
# Diversification is therefore left to ``num_reads``, which the caller
# controls and the policy caps (``local_reads``), rather than to a restart
# count that neither can see.
_NUM_RESTARTS = 0

_CAPABILITIES = SolverCapabilities(
    name="tabu",
    remote=False,
    heuristic=True,
    exhaustive=False,
    supports_seed=True,
    supports_num_reads=True,
    # ``TabuSampler.sample`` takes no sweeps at all (and its ``**kwargs``
    # would swallow one silently), so the declaration says so and the
    # service emits PARAMETER_IGNORED for a non-default ``num_sweeps``.
    supports_num_sweeps=False,
    # The sampler's ``timeout`` is fixed off by this backend and is never
    # wired to a user preference (see ``_TIMEOUT_MS``), so no time limit is
    # declared: a wall-clock budget must not become a dial on search
    # quality.
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    # The sampler's rule (see ``_SEED_LIMIT``), declared so validation
    # refuses an out-of-range seed before anything runs. The check in
    # ``_sample_sharded`` stays as the last line for a direct caller.
    seed_min=0,
    seed_max=_SEED_LIMIT - 1,
    # Stops between shards only: ``TabuSampler`` has no interrupt callback
    # (its ``timeout`` is a per-read wall clock that would make every
    # result machine-dependent), so a shard that has started runs to its
    # end. docs/backends.md states the resulting overrun.
    supports_interrupt=True,
    # Measured on dense ±1 SK instances (docs/backends.md): the same energy
    # as the annealer in about a third of the time at 1000 variables. No
    # weakness was observed on the shipped hard-constrained examples (their
    # hit rate was not recorded for this backend), so only the strength is
    # declared.
    strong_on_large_dense=True,
    description=(
        "Local multistart tabu search on dwave-samplers' TabuSampler, a "
        "strong dense-QUBO heuristic; general-purpose on small problems too, "
        "and the one to prefer over simulated annealing as models grow large "
        "and dense; honours num_reads and seed."
    ),
    # Reads bound the CPU time one request can hold a concurrency slot for.
    # Declared under the local key (``local_reads`` rather than the QPU
    # quota key ``reads``), like the other local sampler.
    parameter_limits=[
        ParameterLimit(
            preference="num_reads", limit="local_reads", error_code="LOCAL_READS_LIMIT"
        ),
    ],
)


class TabuBackend(BackendAliases):
    """Samples the compiled problem with D-Wave's tabu search sampler.

    All reads are kept (never just ``.first``) so every candidate reaches the
    solution validator. The seed is passed to the sampler only; global random
    state is never touched by this backend (spec §37) — with ``seed=None``
    the sampler draws its own seed, once per shard.

    Reads are sampled in ``READS_PER_SHARD``-sized shards, up to ``workers``
    of them concurrently (each read's C++ search releases the GIL). The shard
    layout and shard seeds are a function of ``(num_reads, seed)`` only, and
    the search itself is bounded by a *count* rather than a wall clock (see
    ``_TIMEOUT_MS`` / ``_NUM_RESTARTS``), so the result is a function of
    ``(problem, num_reads, seed)`` and identical for any worker count;
    ``workers`` changes wall time and nothing else. Measured on the knapsack
    example at 200 reads: 0.39 s at ``workers=1``, 0.15 s at ``workers=4``,
    0.08 s at ``workers=8``, bit-identical throughout. A request of at most
    ``READS_PER_SHARD`` reads is one shard and passes the user's seed
    straight through.
    """

    def __init__(self, workers: int | None = None) -> None:
        """``workers``: shards sampled at once; ``None`` detects the CPUs
        available to the process. Never clamped: below 1 is a caller bug."""
        if workers is None:
            workers = default_workers()
        elif workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        self._workers = workers
        self._sampler = TabuSampler()

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
        *,
        interrupt: Interrupt | None = None,
    ) -> RawSolverResult:
        """Run tabu search, returning all reads as samples.

        With an ``interrupt`` a shard not yet started when it says stop is
        skipped; a started shard -- the whole request when it is one shard
        -- runs to its end, because the sampler cannot be stopped part-way
        (see ``SolverBackend.solve``). Without one the sampler calls are
        exactly the uninterruptible ones.
        """
        sizes = shard_sizes(preferences.num_reads)
        interrupted = False
        try:
            if interrupt is not None:
                sampleset, interrupted = self._sample_interruptible(
                    compiled_problem.model, sizes, preferences, interrupt
                )
            elif len(sizes) <= 1:
                # One shard (or an invalid count): the plain call, seed
                # untouched, so the sampler's own validation reaches the
                # caller exactly as the vendor wrote it.
                sampleset = self._sample(
                    compiled_problem.model,
                    preferences.num_reads,
                    preferences.seed,
                )
            else:
                sampleset = self._sample_sharded(
                    compiled_problem.model, sizes, preferences.seed
                )
        except Exception as exc:
            raise SolverExecutionError(f"Tabu search solver failed: {exc}") from exc

        # A local run leaves no vendor facts behind: no solver id, no
        # timing, no quota. Reported are the backend, that it ran here,
        # and the read count asked of the sampler — the shard layout is
        # an implementation detail and is not part of the output.
        metadata = SolverExecutionMetadata(
            backend=self.name,
            remote=False,
            num_reads_requested=preferences.num_reads,
        )
        if sampleset is None:
            result = empty_result(
                compiled_problem.model, backend=self.name, metadata=metadata
            )
        else:
            result = result_from_sampleset(
                sampleset, backend=self.name, metadata=metadata
            )
            result.interrupted = interrupted
        extra = {"interrupted": True} if result.interrupted else {}
        log_solved(
            logger,
            self.name,
            compiled_problem.original_problem.name,
            compiled_problem.num_variables,
            len(result.samples),
            num_reads=preferences.num_reads,
            seed=preferences.seed,
            shards=max(1, len(sizes)),
            workers=min(self._workers, max(1, len(sizes))),
            **extra,
        )
        return result

    def _sample(
        self,
        model: dimod.BinaryQuadraticModel,
        num_reads: int,
        seed: int | None,
    ) -> dimod.SampleSet:
        # Only these keyword arguments, ever. ``num_reads`` and (when given)
        # ``seed`` are the caller's; ``timeout`` and ``num_restarts`` are this
        # backend's fixed search bound and are submitted explicitly rather
        # than left to the vendor's defaults, because the default
        # ``timeout=20`` would make every result machine- and load-dependent
        # (see the two module constants). They are not user preferences: no
        # ``solver`` field reaches them, which is why the declaration says
        # ``supports_time_limit`` is False.
        #
        # Nothing else is passed. ``TabuSampler.sample`` ends in
        # ``**kwargs``, so an unknown name -- ``num_sweeps`` above all --
        # would be swallowed in silence instead of raising; the declaration
        # says ``supports_num_sweeps`` is False so the service emits
        # PARAMETER_IGNORED instead of the sampler quietly dropping it.
        sample_kwargs: dict[str, int | None] = {
            "num_reads": num_reads,
            "timeout": _TIMEOUT_MS,
            "num_restarts": _NUM_RESTARTS,
        }
        if seed is not None:
            sample_kwargs["seed"] = seed
        return self._sampler.sample(model, **sample_kwargs)

    def _sample_sharded(
        self,
        model: dimod.BinaryQuadraticModel,
        sizes: list[int],
        seed: int | None,
    ) -> dimod.SampleSet:
        """Sample every shard, ``workers`` at a time, and concatenate in order."""
        seeds = self._shard_seeds(seed, len(sizes))
        return run_shards(
            lambda shard: self._sample(model, sizes[shard], seeds[shard]),
            len(sizes),
            self._workers,
        )

    def _sample_interruptible(
        self,
        model: dimod.BinaryQuadraticModel,
        sizes: list[int],
        preferences: SolverPreferences,
        interrupt: Interrupt,
    ) -> tuple[dimod.SampleSet | None, bool]:
        """The sampling above with a check before each shard; returns the
        shards that ran (None when none did) and whether any was skipped.

        The layout -- one plain call for one shard, derived seeds for
        several -- is exactly the uninterruptible one, so a run the
        interrupt never stops returns the same reads.
        """
        if len(sizes) <= 1:
            if interrupt.should_stop():
                return None, True
            return self._sample(model, preferences.num_reads, preferences.seed), False
        seeds = self._shard_seeds(preferences.seed, len(sizes))
        return run_shards_interruptible(
            lambda shard: self._sample(model, sizes[shard], seeds[shard]),
            len(sizes),
            self._workers,
            interrupt.should_stop,
        )

    @staticmethod
    def _shard_seeds(seed: int | None, count: int) -> list[int | None]:
        """Per-shard seeds, the user's seed checked against the sampler's
        rule first."""
        if seed is not None and not 0 <= seed < _SEED_LIMIT:
            # The sampler's own rule (see ``_SEED_LIMIT``), checked before
            # derivation: masked shard seeds are always in range, so without
            # it this path would accept seeds the single-shard path rejects.
            # Both paths now accept the same range, raise ValueError and read
            # the same: this is the vendor's own message, so a caller cannot
            # tell from the error which path ran.
            raise ValueError("Seed must be between 0 and 2**32 - 1")
        return shard_seeds(seed, count)
