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
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_arrays,
)
from annealbridge.solvers.metadata import sanitize_sampleset_info
from annealbridge.solvers.ocean import (
    OCEAN_CREDENTIALS,
    HYBRID_SAMPLE_EXCEPTION_CODES,
    SAMPLER_INIT_EXCEPTION_CODES,
    LazySampler,
    call_ocean,
    dwave_availability,
    register_ocean_config_token,
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


class LeapHybridBQMBackend:
    """Solves the compiled BQM on the Leap hybrid cloud solver.

    ``sampler_factory`` is the single test seam: production uses the default
    (a real ``LeapHybridSampler``), tests inject a fake. The sampler is
    built lazily on first use and cached for the lifetime of the backend
    instance (construction fetches solver metadata and starts worker
    threads); a failed construction is never cached.

    ``num_reads`` / ``num_sweeps`` / ``seed`` are meaningless for the hybrid
    solver and are never forwarded; only ``time_limit`` is passed through.
    """

    def __init__(self, sampler_factory: Callable[[], Any] | None = None) -> None:
        self._sampler = LazySampler(sampler_factory, _default_sampler_factory)
        register_ocean_config_token()

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        """Installability and credentials, via the shared check. No network I/O."""
        return dwave_availability()

    @property
    def name(self) -> str:
        """Alias for ``capabilities.name``."""
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        """Alias for ``capabilities.exhaustive``."""
        return self.capabilities.exhaustive

    def resolve_time_limit(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> float:
        """Effective ``time_limit`` (seconds) a solve would submit (spec §16).

        The user's value if given, floored at the sampler's
        ``min_time_limit(bqm)``; the sampler minimum alone when the user
        gave none. Nothing is submitted: ``min_time_limit`` is a local
        interpolation over solver properties fetched at construction. The
        service compares this value with policy before :meth:`solve` runs;
        :meth:`solve` calls this same method so both see the same number.
        """
        bqm = compiled_problem.model
        user_time_limit = (
            preferences.leap_hybrid_bqm.time_limit_seconds
            if preferences.leap_hybrid_bqm is not None
            else None
        )
        sampler = call_ocean(
            "Leap hybrid sampler could not be created",
            SAMPLER_INIT_EXCEPTION_CODES,
            self._sampler.get,
        )
        min_time_limit = call_ocean(
            "Leap hybrid minimum time limit could not be determined",
            HYBRID_SAMPLE_EXCEPTION_CODES,
            lambda: float(sampler.min_time_limit(bqm)),
        )
        if user_time_limit is None:
            return min_time_limit
        return max(float(user_time_limit), min_time_limit)

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
        effective_time_limit = self.resolve_time_limit(compiled_problem, preferences)
        sampler = call_ocean(
            "Leap hybrid sampler could not be created",
            SAMPLER_INIT_EXCEPTION_CODES,
            self._sampler.get,
        )
        sampleset = call_ocean(
            "Leap hybrid solve failed",
            HYBRID_SAMPLE_EXCEPTION_CODES,
            lambda: resolved(sampler.sample(bqm, time_limit=effective_time_limit)),
        )

        variables, samples, energies = sampleset_to_arrays(sampleset)
        metadata = sanitize_sampleset_info(sampleset.info, backend=self.name)
        metadata.effective_time_limit_seconds = effective_time_limit
        logger.info(
            "Backend %s solved problem %s: %d variables, %d samples "
            "(effective_time_limit_seconds=%s)",
            self.name,
            compiled_problem.original_problem.name,
            compiled_problem.num_variables,
            len(samples),
            effective_time_limit,
        )
        return RawSolverResult(
            variables=variables,
            samples=samples,
            energies=energies,
            backend=self.name,
            metadata=metadata,
        )
