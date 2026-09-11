"""Backend recommendation models (Phase 3a spec §23.2).

Pure Pydantic, kept in ``validation`` so the interfaces can import the
result type without touching ``orchestration``. The ranking itself lives
in ``orchestration/routing.py``; this module only describes its output.

The result is *advisory*: nothing anywhere feeds it back into a solve.
``solve_optimization`` always uses ``problem.solver.backend`` exactly as
given (overview principle 5).
"""

from pydantic import BaseModel, Field

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

    rank: int = Field(description="1-based position in the ranking.")
    backend: str = Field(
        description=(
            "The registry name — the value to put in solver.backend to run "
            "on it."
        )
    )
    usable: bool = Field(
        description=(
            "Whether a solve on it right now would get past every gate: "
            "enabled by policy, available, and within every declared limit "
            "for the submitted preferences."
        )
    )
    model_type: ModelType | None = Field(
        description=(
            "The compiler path this backend would take; null when the server "
            "has no compiler for any model type it accepts."
        )
    )
    reasons: list[str] = Field(
        description=(
            "Fixed routing reason codes, in the order they were applied to "
            "reach this rank."
        )
    )
    blocking: list[SolveError] = Field(
        default=[],
        description=(
            "Why a solve now would fail, e.g. REMOTE_DISABLED. Empty exactly "
            "when usable is true."
        ),
    )
    warnings: list[SolveError] = Field(
        default=[],
        description="The validation warnings for this backend's compiler path.",
    )
    estimated_compiled_variables: int | None = Field(
        default=None,
        description=(
            "The compiled size on this backend's path, computed arithmetically "
            "without building a model. Null when there is no compiler path."
        ),
    )


class BackendRecommendationResult(BaseModel):
    """Ranked backends for one problem (spec §23.2).

    ``valid`` is the problem's own validity; when False ``errors`` holds
    the problem errors and ``recommendations`` is empty.
    """

    valid: bool = Field(
        description=(
            "The problem's own validity. When false, recommendations is empty."
        )
    )
    errors: list[SolveError] = Field(
        default=[],
        description="The problem's errors when it is invalid; empty otherwise.",
    )
    recommendations: list[BackendRecommendation] = Field(
        default=[],
        description="Every registered backend, ranked best first.",
    )
    advisory: str = Field(
        default=ADVISORY_TEXT,
        description=(
            "A fixed sentence restating that a solve always uses "
            "problem.solver.backend as given and never substitutes one."
        ),
    )
