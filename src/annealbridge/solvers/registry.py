"""Solver backend registry (Phase 2 spec §11)."""

from annealbridge.solvers.base import SolverBackend
from annealbridge.solvers.dwave_qpu import DWaveQPUBackend
from annealbridge.solvers.exact import ExactSolverBackend
from annealbridge.solvers.leap_hybrid_bqm import LeapHybridBQMBackend
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

        Registering the remote backends (``dwave_qpu``, ``leap_hybrid_bqm``)
        never imports any D-Wave cloud package: each backend lazy-imports
        ``dwave.system`` inside ``solve()`` only (spec §4).
        """
        exact = ExactSolverBackend()
        annealer = SimulatedAnnealingBackend()
        qpu = DWaveQPUBackend()
        leap_hybrid = LeapHybridBQMBackend()
        return cls(
            {
                exact.name: exact,
                annealer.name: annealer,
                qpu.name: qpu,
                leap_hybrid.name: leap_hybrid,
            }
        )
