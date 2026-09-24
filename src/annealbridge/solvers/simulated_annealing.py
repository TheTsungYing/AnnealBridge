"""Simulated annealing solver backend (spec §21)."""

import logging
from collections.abc import Callable

import dimod
from dwave.samplers import SimulatedAnnealingSampler

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

# The shard layout, the derived shard seeds and the bounded fan-out are
# shared with the other local sampler backends; ``READS_PER_SHARD`` and the
# reproducibility contract it carries live there.
from annealbridge.solvers.sharding import (
    default_workers,
    run_shards,
    run_shards_interruptible,
    shard_seeds,
    shard_sizes,
)

logger = logging.getLogger(__name__)

# ``SimulatedAnnealingSampler`` accepts ``0 <= seed < 2**31``; this copies
# that rule on purpose. The sharded path derives every shard seed through
# ``SeedSequence(seed)`` and masks it to 31 bits, so a derived seed is always
# in range: unchecked, several shards would silently accept a seed that the
# single-shard path (seed passed straight to the sampler) rejects. Checking
# it first makes both paths accept the same range and raise the same
# exception type. The message is ours and states the real bound: the
# dwave-samplers 1.8.0 message says ``2^32 - 1`` but the check is
# ``< 2**31``. The seed tests in ``tests/unit/test_solvers.py``
# (``test_vendor_sampler_accepts_the_largest_seed_the_shard_check_allows``,
# ``test_vendor_sampler_rejects_the_seeds_the_shard_check_rejects``,
# ``test_out_of_range_seed_fails_the_same_way_on_both_paths``) pin both the
# vendor rule and the agreement of the two paths, and turn red if a
# dwave-samplers upgrade changes the rule.
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
    # The sampler's rule (see ``_SEED_LIMIT``), declared so validation
    # refuses an out-of-range seed before anything runs. The check in
    # ``_sample_sharded`` stays as the last line for a direct caller.
    seed_min=0,
    seed_max=_SEED_LIMIT - 1,
    # Stops between shards and, through the sampler's own
    # ``interrupt_function``, after every completed read.
    supports_interrupt=True,
    description=(
        "Local heuristic simulated-annealing sampler; the default and the "
        "safest general-purpose pick, best on small or hard-constrained "
        "problems; scales to larger problems but does not prove optimality "
        "or infeasibility."
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
        *,
        interrupt: Interrupt | None = None,
    ) -> RawSolverResult:
        """Run simulated annealing, returning all reads as samples.

        With an ``interrupt`` the run stops between shards and after any
        completed read once it says stop, and returns only the reads that
        finished their whole schedule (see ``SolverBackend.solve``). Without
        one -- the service passes none unless the solve has a wall-clock
        limit or a cancel token -- the sampler calls are exactly the
        uninterruptible ones.
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
            num_sweeps=preferences.num_sweeps,
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
        num_sweeps: int,
        seed: int | None,
        interrupt_function: Callable[[], bool] | None = None,
    ) -> dimod.SampleSet:
        sample_kwargs: dict[str, object] = {
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
        }
        if seed is not None:
            sample_kwargs["seed"] = seed
        # Only ever passed on the interruptible path, so an uninterrupted
        # solve makes exactly the call it always made.
        if interrupt_function is not None:
            sample_kwargs["interrupt_function"] = interrupt_function
        return self._sampler.sample(model, **sample_kwargs)

    def _sample_sharded(
        self,
        model: dimod.BinaryQuadraticModel,
        sizes: list[int],
        preferences: SolverPreferences,
    ) -> dimod.SampleSet:
        """Sample every shard, ``workers`` at a time, and concatenate in order."""
        seeds = self._shard_seeds(preferences.seed, len(sizes))
        sweeps = preferences.num_sweeps
        return run_shards(
            lambda shard: self._sample(model, sizes[shard], sweeps, seeds[shard]),
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
        """The sampling above, stoppable; returns the reads that completed
        (None when none did) and whether any read asked for was skipped.

        The layout -- one plain call for one shard, derived seeds for
        several -- is exactly the uninterruptible one, so a run the
        interrupt never stops returns the same reads. The sampler calls its
        ``interrupt_function`` after each completed read and then returns
        only the completed ones; a shard not started when the interrupt
        says stop is skipped.
        """
        watch = _InterruptWatch(interrupt)
        sweeps = preferences.num_sweeps
        if len(sizes) <= 1:
            if interrupt.should_stop():
                return None, True
            sampleset = self._sample(
                model, preferences.num_reads, sweeps, preferences.seed, watch
            )
            watch.raise_if_failed()
            return sampleset, len(sampleset) < preferences.num_reads
        seeds = self._shard_seeds(preferences.seed, len(sizes))
        merged, skipped = run_shards_interruptible(
            lambda shard: self._sample(model, sizes[shard], sweeps, seeds[shard], watch),
            len(sizes),
            self._workers,
            # The same wrapper: once a check has failed in one shard, every
            # shard not yet started is skipped instead of run to the end.
            watch,
        )
        watch.raise_if_failed()
        cut = skipped or (merged is not None and len(merged) < sum(sizes))
        return merged, cut

    @staticmethod
    def _shard_seeds(seed: int | None, count: int) -> list[int | None]:
        """Per-shard seeds, the user's seed checked against the sampler's
        rule first."""
        if seed is not None and not 0 <= seed < _SEED_LIMIT:
            # The sampler's own rule (see ``_SEED_LIMIT``), checked before
            # derivation: masked shard seeds are always in range, so without
            # it this path would accept seeds the single-shard path rejects.
            # Both paths now accept the same range and raise ValueError; only
            # the message text differs from the vendor's.
            raise ValueError(
                f"'seed' should be an integer between 0 and {_SEED_LIMIT - 1}: "
                f"value = {seed}"
            )
        return shard_seeds(seed, count)


class _InterruptWatch:
    """The ``interrupt_function`` handed to the sampler.

    The sampler treats an exception raised by its callback as a request to
    stop and swallows it, which would turn a bug into a silently cut run.
    ``Interrupt.should_stop`` never raises, but this wrapper does not rely
    on it: whatever the check raises is kept, the sampler is told to stop,
    and :meth:`raise_if_failed` re-raises it once the sampler has returned,
    so it becomes a solver failure instead.
    """

    def __init__(self, interrupt: Interrupt) -> None:
        self._interrupt = interrupt
        self._error: BaseException | None = None

    def __call__(self) -> bool:
        if self._error is not None:
            return True
        try:
            return self._interrupt.should_stop()
        except BaseException as exc:
            self._error = exc
            return True

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error
