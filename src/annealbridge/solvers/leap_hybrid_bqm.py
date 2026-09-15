"""Leap hybrid BQM solver backend (Phase 2 spec §16).

``dwave.system`` is imported lazily inside the default sampler factory so
this module — and the default registry that instantiates the backend —
imports cleanly without the ``dwave`` extra installed (spec §4).
"""

import logging
from typing import Any, Callable

from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import (
    AvailabilityStatus,
    BackendAliases,
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
    log_solved,
    result_from_sampleset,
)
from annealbridge.solvers.metadata import sanitize_sampleset_info
from annealbridge.solvers.ocean import (
    HYBRID_SAMPLE_EXCEPTION_CODES,
    OCEAN_CREDENTIALS,
    HybridTimeLimitMemo,
    call_ocean,
    dwave_availability,
    ocean_sampler_holder,
    resolve_hybrid_sampler,
    resolved,
)

logger = logging.getLogger(__name__)

_CAPABILITIES = SolverCapabilities(
    name="leap_hybrid_bqm",
    remote=True,
    heuristic=True,
    exhaustive=False,
    supports_seed=False,
    supports_num_reads=False,
    supports_time_limit=True,
    supported_model_types=["bqm"],
    returns_multiple_samples=False,
    credentials=OCEAN_CREDENTIALS,
    parameter_limits=[
        ParameterLimit(
            preference="leap_hybrid_bqm.time_limit_seconds",
            limit="time_seconds",
            error_code="REMOTE_TIME_LIMIT",
        ),
    ],
    description=(
        "D-Wave Leap cloud hybrid (classical + quantum) BQM solver for "
        "large problems. Typically returns a single sample per solve, so "
        "top_k effectively yields at most one solution; this is expected "
        "behaviour, not an error."
    ),
)


def _default_sampler_factory() -> Any:
    """Create a real ``LeapHybridSampler`` (requires the ``dwave`` extra)."""
    from dwave.system import LeapHybridSampler

    return LeapHybridSampler()


class LeapHybridBQMBackend(BackendAliases):
    """Solves the compiled BQM on the Leap hybrid cloud solver.

    ``sampler_factory`` is the single test seam: production uses the default
    (a real ``LeapHybridSampler``), tests inject a fake. The sampler is
    built lazily on first use and cached while the credential sources are
    unchanged (construction fetches solver metadata and starts worker
    threads); a rotated token or edited Ocean config, or an authentication
    failure, makes the next solve rebuild it (review F-21); a failed
    construction is never cached.

    ``num_reads`` / ``num_sweeps`` / ``seed`` are meaningless for the hybrid
    solver and are never forwarded; only ``time_limit`` is passed through.
    """

    def __init__(self, sampler_factory: Callable[[], Any] | None = None) -> None:
        # Review F-03: the holder comes with this backend's credential
        # declaration and the Ocean config-file token source, so a directly
        # constructed backend redacts its token too (see ocean_sampler_holder).
        self._sampler = ocean_sampler_holder(
            sampler_factory, _default_sampler_factory, _CAPABILITIES
        )
        # The ``min_time_limit`` of the service's pre-submission check is
        # reused by the solve of the same attempt (see HybridTimeLimitMemo).
        self._time_limit_memo = HybridTimeLimitMemo()

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        """Installability and credentials, via the shared check. No network I/O."""
        return dwave_availability()

    def _resolve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> tuple[Any, float]:
        """The sampler and the effective ``time_limit``, by the shared hybrid rule."""
        user_time_limit = (
            preferences.leap_hybrid_bqm.time_limit_seconds
            if preferences.leap_hybrid_bqm is not None
            else None
        )
        return resolve_hybrid_sampler(
            self._sampler,
            compiled_problem.model,
            user_time_limit,
            label="Leap hybrid",
            memo=self._time_limit_memo,
        )

    def resolve_time_limit(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> float:
        """Effective ``time_limit`` (seconds) a solve would submit (spec §16).

        The shared hybrid rule (:func:`resolve_hybrid_sampler`): the
        user's value if given, floored at the sampler's
        ``min_time_limit(bqm)``; the sampler minimum alone when the user
        gave none. Nothing is submitted. The service compares this value
        with policy before :meth:`solve` runs; :meth:`solve` resolves
        through the same rule, so both see the same number — and, within
        one attempt, the same sampler and the same memoised minimum.
        """
        return self._resolve(compiled_problem, preferences)[1]

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        """Submit the BQM to the hybrid solver and return all samples.

        The submitted ``time_limit`` is exactly :meth:`resolve_time_limit`
        and is recorded in metadata as ``effective_time_limit_seconds``.
        The policy ceiling is enforced by the service layer, not here.
        """
        bqm = compiled_problem.model
        sampler, effective_time_limit = self._resolve(compiled_problem, preferences)
        sampleset = call_ocean(
            "Leap hybrid solve failed",
            HYBRID_SAMPLE_EXCEPTION_CODES,
            lambda: resolved(sampler.sample(bqm, time_limit=effective_time_limit)),
            holder=self._sampler,
        )

        metadata = sanitize_sampleset_info(
            sampleset.info,
            backend=self.name,
            effective_time_limit_seconds=effective_time_limit,
        )
        result = result_from_sampleset(sampleset, backend=self.name, metadata=metadata)
        log_solved(
            logger,
            self.name,
            compiled_problem.original_problem.name,
            compiled_problem.num_variables,
            len(result.samples),
            effective_time_limit_seconds=effective_time_limit,
        )
        return result
