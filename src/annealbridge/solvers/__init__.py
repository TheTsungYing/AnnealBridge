"""Solver backends: exact enumeration, simulated annealing and the remote
D-Wave QPU, Leap hybrid BQM and Leap hybrid CQM solvers (spec §20–§22,
Phase 2 §15–§16, Phase 3a §17)."""

from annealbridge.solvers.base import (
    AvailabilityStatus,
    RawSolverResult,
    SolverBackend,
    SolverCapabilities,
    assert_samples_within_bounds,
    sampleset_to_arrays,
)
from annealbridge.solvers.dwave_qpu import DWaveQPUBackend
from annealbridge.solvers.exact import ExactSolverBackend
from annealbridge.solvers.leap_hybrid_bqm import LeapHybridBQMBackend
from annealbridge.solvers.leap_hybrid_cqm import LeapHybridCQMBackend
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
    "AvailabilityStatus",
    "DWaveQPUBackend",
    "ExactSolverBackend",
    "LeapHybridBQMBackend",
    "LeapHybridCQMBackend",
    "REASON_CONFIG_INVALID",
    "REASON_CREDENTIALS_MISSING",
    "REASON_NOT_INSTALLED",
    "RawSolverResult",
    "SimulatedAnnealingBackend",
    "SolverBackend",
    "SolverCapabilities",
    "SolverRegistry",
    "assert_samples_within_bounds",
    "redact",
    "sampleset_to_arrays",
    "sanitize_sampleset_info",
]
