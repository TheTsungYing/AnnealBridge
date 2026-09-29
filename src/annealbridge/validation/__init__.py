"""Problem and solution validation."""

from annealbridge.validation.expansion import (
    DEFAULT_MAX_TEMPLATE_BINDINGS,
    ExpandedProblem,
    expand_problem,
)
from annealbridge.validation.problem_validator import (
    ProblemValidationResult,
    validate_expanded,
    validate_problem,
    validate_problem_full,
)
from annealbridge.validation.recommendation import (
    ADVISORY_TEXT,
    BackendRecommendation,
    BackendRecommendationResult,
)
from annealbridge.validation.solution_validator import BatchValidation, validate_batch
from annealbridge.validation.solution_validator import validate as validate_solution
from annealbridge.validation.tolerance import (
    ABSOLUTE_TOLERANCE,
    EPSILON,
    RELATIVE_TOLERANCE,
    satisfies,
    tolerance,
    tolerance_array,
)

__all__ = [
    "ABSOLUTE_TOLERANCE",
    "ADVISORY_TEXT",
    "BackendRecommendation",
    "BackendRecommendationResult",
    "BatchValidation",
    "DEFAULT_MAX_TEMPLATE_BINDINGS",
    "EPSILON",
    "ExpandedProblem",
    "ProblemValidationResult",
    "RELATIVE_TOLERANCE",
    "expand_problem",
    "satisfies",
    "tolerance",
    "tolerance_array",
    "validate_expanded",
    "validate_problem",
    "validate_problem_full",
    "validate_batch",
    "validate_solution",
]
