"""Solver backend registry (Phase 2 spec §11)."""

from annealbridge.solvers.base import SolverBackend
from annealbridge.solvers.dwave_qpu import DWaveQPUBackend
from annealbridge.solvers.exact import ExactSolverBackend
from annealbridge.solvers.fujitsu_da import FujitsuDABackend
from annealbridge.solvers.leap_hybrid_bqm import LeapHybridBQMBackend
from annealbridge.solvers.leap_hybrid_cqm import LeapHybridCQMBackend
from annealbridge.solvers.simulated_annealing import SimulatedAnnealingBackend


class SolverRegistry:
    """Lookup table from backend name to ``SolverBackend`` instance."""

    def __init__(self, backends: dict[str, SolverBackend]) -> None:
        self._backends = dict(backends)

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
    def default(cls) -> "SolverRegistry":
        """Build the default registry.

        Registration order is fixed (3a §17.7, 3b §20.9): ``exact``,
        ``simulated_annealing``, ``dwave_qpu``, ``leap_hybrid_bqm``,
        ``leap_hybrid_cqm``, ``fujitsu_da``. The capabilities list, the CLI
        table and the routing tie-break all follow it.

        Registering the remote backends never imports any D-Wave cloud
        package: each D-Wave backend lazy-imports ``dwave.system`` inside
        its sampler factory only (spec §4), and the Fujitsu backend uses
        the standard library for HTTP (3b §20.4).
        """
        backends = (
            ExactSolverBackend(),
            SimulatedAnnealingBackend(),
            DWaveQPUBackend(),
            LeapHybridBQMBackend(),
            LeapHybridCQMBackend(),
            FujitsuDABackend(),
        )
        return cls({backend.name: backend for backend in backends})
