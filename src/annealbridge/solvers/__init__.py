"""Solver backends: exact enumeration, simulated annealing and the remote
D-Wave QPU and Leap hybrid BQM solvers (spec §20–§22, Phase 2 §15–§16)."""

from annealbridge.solvers.base import (
    RawSolverResult,
    SolverBackend,
    SolverCapabilities,
)
from annealbridge.solvers.dwave_qpu import DWaveQPUBackend
from annealbridge.solvers.exact import ExactSolverBackend
from annealbridge.solvers.leap_hybrid_bqm import LeapHybridBQMBackend
from annealbridge.solvers.metadata import (
    REASON_CONFIG_INVALID,
    REASON_CREDENTIALS_MISSING,
    REASON_NOT_INSTALLED,
    redact,
    sanitize_sampleset_info,
)
from annealbridge.solvers.registry import SolverRegistry
from annealbridge.solvers.simulated_annealing import SimulatedAnnealingBackend

__all__ = [
    "DWaveQPUBackend",
    "ExactSolverBackend",
    "LeapHybridBQMBackend",
    "REASON_CONFIG_INVALID",
    "REASON_CREDENTIALS_MISSING",
    "REASON_NOT_INSTALLED",
    "RawSolverResult",
    "SimulatedAnnealingBackend",
    "SolverBackend",
    "SolverCapabilities",
    "SolverRegistry",
    "redact",
    "sanitize_sampleset_info",
]
