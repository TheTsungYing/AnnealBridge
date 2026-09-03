"""Backend recommendation models (Phase 3a spec §23.2).

Pure Pydantic, kept in ``validation`` so the interfaces can import the
result type without touching ``orchestration``. The ranking itself lives
in ``orchestration/routing.py``; this module only describes its output.

The result is *advisory*: nothing anywhere feeds it back into a solve.
``solve_optimization`` always uses ``problem.solver.backend`` exactly as
given (overview principle 5).
"""

from pydantic import BaseModel

from annealbridge.models import ModelType, SolveError

# The fixed advisory sentence every result carries (spec §23.2).
ADVISORY_TEXT = (
    "Advisory only: solve_optimization always uses problem.solver.backend "
    "as given and never substitutes a backend."
)


class BackendRecommendation(BaseModel):
    """One ranked backend (spec §23.2).

    ``usable`` means enabled by policy, available right now and within
    every declared limit for the user's preferences — i.e. a solve on it
    would not be refused before submission. ``reasons`` are the fixed
    routing codes (spec §23.4) in the order they were applied.
    """

    rank: int
    backend: str
    usable: bool
    model_type: ModelType | None  # the compiler path it would take; None = no compiler
    reasons: list[str]
    blocking: list[SolveError] = []  # why a solve now would fail (REMOTE_DISABLED, ...)
    warnings: list[SolveError] = []  # service.validate() warnings for this backend
    estimated_compiled_variables: int | None = None


class BackendRecommendationResult(BaseModel):
    """Ranked backends for one problem (spec §23.2).

    ``valid`` is the problem's own validity; when False ``errors`` holds
    the problem errors and ``recommendations`` is empty.
    """

    valid: bool
    errors: list[SolveError] = []
    recommendations: list[BackendRecommendation] = []
    advisory: str = ADVISORY_TEXT
