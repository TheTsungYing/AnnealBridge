"""Domain models for the optimization middleware."""

from annealbridge.models.compiled import CompiledProblem, ConstraintTrace
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
    "CompiledProblem",
    "Constraint",
    "ConstraintEvaluation",
    "ConstraintTrace",
    "DWaveQPUOptions",
    "LeapHybridBQMOptions",
    "LinearTerm",
    "Objective",
    "OptimizationProblem",
    "ProblemError",
    "QuadraticTerm",
    "RECOMMENDED_ACTIONS",
    "RETRYABLE_CODES",
    "Solution",
    "SolveAttempt",
    "SolveError",
    "SolveResult",
    "SolverExecutionMetadata",
    "SolverPreferences",
    "ValidationResult",
    "Variable",
]
