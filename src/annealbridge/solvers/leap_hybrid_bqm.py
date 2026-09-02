"""Leap hybrid BQM solver backend (Phase 2 spec §16).

``dwave.system`` is imported lazily inside :meth:`LeapHybridBQMBackend.solve`
so this module — and the default registry that instantiates the backend —
imports cleanly without the ``dwave`` extra installed (spec §4).
"""

import importlib.util
import logging
from typing import Any, Callable

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import (
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_lists,
)
from annealbridge.solvers.metadata import (
    ocean_config_status,
    redact,
    sanitize_sampleset_info,
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
    description=(
        "D-Wave Leap cloud hybrid (classical + quantum) BQM solver for "
        "large problems. Typically returns a single sample per solve, so "
        "top_k effectively yields at most one solution; this is expected "
        "behaviour, not an error."
    ),
)

# Ocean exception class names → catalog error codes. Matched by name across
# the exception's MRO so classification works (and is testable with fakes)
# without dwave-cloud-client installed.
_EXCEPTION_NAME_TO_CODE = {
    "SolverAuthenticationError": "REMOTE_AUTH_FAILED",
    "RequestTimeout": "REMOTE_TIMEOUT",
}


def _dwave_system_installed() -> bool:
    """Return whether ``dwave.system`` is importable, without importing it."""
    try:
        return importlib.util.find_spec("dwave.system") is not None
    except (ImportError, ValueError):
        return False


def _default_sampler_factory() -> Any:
    """Create a real ``LeapHybridSampler`` (requires the ``dwave`` extra)."""
    from dwave.system import LeapHybridSampler

    return LeapHybridSampler()


def _classify_exception(exc: Exception) -> str:
    for klass in type(exc).__mro__:
        code = _EXCEPTION_NAME_TO_CODE.get(klass.__name__)
        if code is not None:
            return code
    return "REMOTE_SOLVER_ERROR"


class LeapHybridBQMBackend:
    """Solves the compiled BQM on the Leap hybrid cloud solver.

    ``sampler_factory`` is the single test seam: production uses the default
    (a real ``LeapHybridSampler``), tests inject a fake. ``num_reads`` /
    ``num_sweeps`` / ``seed`` are meaningless for the hybrid solver and are
    never forwarded; only ``time_limit`` is passed through.
    """

    def __init__(self, sampler_factory: Callable[[], Any] | None = None) -> None:
        self._sampler_factory = sampler_factory

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> tuple[bool, str | None]:
        """Check installability and credentials. No network I/O.

        Reasons are categorical strings only and never contain config
        values (spec §10).
        """
        if not _dwave_system_installed():
            return (False, "dwave-system not installed")
        status = ocean_config_status()
        if status == "invalid":
            return (False, "D-Wave configuration invalid")
        if status == "missing":
            return (False, "D-Wave credentials not configured")
        return (True, None)

    @property
    def name(self) -> str:
        """Alias for ``capabilities.name``."""
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        """Alias for ``capabilities.exhaustive``."""
        return self.capabilities.exhaustive

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        """Submit the BQM to the hybrid solver and return all samples.

        time_limit (spec §16): user value if given, floored at the
        sampler's ``min_time_limit(bqm)``; the effective value is recorded
        in metadata. The policy ceiling is enforced by the service layer,
        not here.
        """
        bqm = compiled_problem.model
        user_time_limit = (
            preferences.leap_hybrid_bqm.time_limit_seconds
            if preferences.leap_hybrid_bqm is not None
            else None
        )

        try:
            factory = self._sampler_factory or _default_sampler_factory
            sampler = factory()
            min_time_limit = float(sampler.min_time_limit(bqm))
            if user_time_limit is None:
                effective_time_limit = min_time_limit
            else:
                effective_time_limit = max(float(user_time_limit), min_time_limit)
            sampleset = sampler.sample(bqm, time_limit=effective_time_limit)
        except Exception as exc:
            code = _classify_exception(exc)
            raise SolverExecutionError(
                redact(f"Leap hybrid solve failed: {exc}"),
                code=code,
            ) from exc

        samples, energies = sampleset_to_lists(sampleset)
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
            samples=samples,
            energies=energies,
            backend=self.name,
            metadata=metadata,
        )
