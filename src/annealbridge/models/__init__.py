"""Domain models for the optimization middleware."""

from annealbridge.models.capabilities import (
    AvailabilityCategory,
    AvailabilityStatus,
    ModelType,
    ParameterLimit,
    SolverCapabilities,
)
from annealbridge.models.compiled import (
    CompiledProblem,
    ConstraintTrace,
    IntegerEncoding,
)
from annealbridge.models.constraint import Constraint
from annealbridge.models.error_catalog import (
    RECOMMENDED_ACTIONS,
    RETRYABLE_CODES,
    catalog_error,
)
from annealbridge.models.metadata import SolverExecutionMetadata
from annealbridge.models.objective import LinearTerm, Objective, QuadraticTerm
from annealbridge.models.problem import (
    DWaveQPUOptions,
    LeapHybridBQMOptions,
    LeapHybridCQMOptions,
    OptimizationProblem,
    SolverPreferences,
)
from annealbridge.models.solution import (
    ConstraintEvaluation,
    ProblemError,
    Solution,
    SolveAttempt,
    SolveError,
    SolveResult,
    ValidationResult,
)
from annealbridge.models.variable import Variable

__all__ = [
    "catalog_error",
    "AvailabilityCategory",
    "AvailabilityStatus",
    "CompiledProblem",
    "Constraint",
    "ConstraintEvaluation",
    "ConstraintTrace",
    "DWaveQPUOptions",
    "IntegerEncoding",
    "LeapHybridBQMOptions",
    "LeapHybridCQMOptions",
    "LinearTerm",
    "ModelType",
    "Objective",
    "OptimizationProblem",
    "ParameterLimit",
    "ProblemError",
    "QuadraticTerm",
    "RECOMMENDED_ACTIONS",
    "RETRYABLE_CODES",
    "Solution",
    "SolveAttempt",
    "SolveError",
    "SolveResult",
    "SolverCapabilities",
    "SolverExecutionMetadata",
    "SolverPreferences",
    "ValidationResult",
    "Variable",
]
