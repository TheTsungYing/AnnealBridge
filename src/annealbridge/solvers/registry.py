"""Solver backend registry (Phase 2 spec §11)."""

from annealbridge.solvers.base import SolverBackend
from annealbridge.solvers.dwave_qpu import DWaveQPUBackend
from annealbridge.solvers.exact import ExactSolverBackend
from annealbridge.solvers.fujitsu_da import FujitsuDABackend
from annealbridge.solvers.leap_hybrid_bqm import LeapHybridBQMBackend
from annealbridge.solvers.leap_hybrid_cqm import LeapHybridCQMBackend
from annealbridge.solvers.metadata import declare_credentials
from annealbridge.solvers.simulated_annealing import SimulatedAnnealingBackend


class SolverRegistry:
    """Lookup table from backend name to ``SolverBackend`` instance.

    Registering a backend also feeds its credential declaration
    (``capabilities.credentials``) into the shared redaction (review
    F-10): a backend only has to *declare* its env vars / headers, and
    every error message, log line and metadata field that passes through
    ``metadata.redact`` masks them from then on. Since the 2026-09-11
    review (F03) a shipped backend declares the same thing in its own
    ``__init__`` too, so a directly constructed instance is masked as well;
    the registry's declaration is the idempotent second write.
    """

    def __init__(self, backends: dict[str, SolverBackend]) -> None:
        self._backends = dict(backends)
        for name, backend in self._backends.items():
            declare_credentials(name, backend.capabilities.credentials)

    def get(self, name: str) -> SolverBackend:
        """Return the registered backend named ``name``.

        Raises ``KeyError`` for unknown names; the service maps it to an
        ``UNKNOWN_BACKEND`` :class:`SolveError`.
        """
        return self._backends[name]

    def names(self) -> list[str]:
        """Return the registered backend names, in registration order."""
        return list(self._backends)

    @classmethod
    def default(cls, *, sa_workers: int | None = None) -> "SolverRegistry":
        """Build the default registry.

        Registration order is fixed (3a §17.7, 3b §20.9): ``exact``,
        ``simulated_annealing``, ``dwave_qpu``, ``leap_hybrid_bqm``,
        ``leap_hybrid_cqm``, ``fujitsu_da``. The capabilities list, the CLI
        table and the routing tie-break all follow it.

        ``sa_workers`` is handed to :class:`SimulatedAnnealingBackend`
        (``None``: detect the CPUs available to the process). It only
        changes how fast that backend samples, never what it returns.

        Registering the remote backends never imports any D-Wave cloud
        package: each D-Wave backend lazy-imports ``dwave.system`` inside
        its sampler factory only (spec §4), and the Fujitsu backend uses
        the standard library for HTTP (3b §20.4).
        """
        backends = (
            ExactSolverBackend(),
            SimulatedAnnealingBackend(workers=sa_workers),
            DWaveQPUBackend(),
            LeapHybridBQMBackend(),
            LeapHybridCQMBackend(),
            FujitsuDABackend(),
        )
        return cls({backend.name: backend for backend in backends})
