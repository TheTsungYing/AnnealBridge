"""Solver execution metadata model (Phase 2 spec §17).

Defined in the models layer so that ``SolveResult`` (models) can reference
it without a models → solvers dependency; the sanitizer and redaction
helpers that produce it live in ``annealbridge.solvers.metadata``.
"""

from pydantic import BaseModel, Field

from annealbridge.models.capabilities import ModelType


class SolverExecutionMetadata(BaseModel):
    """Sanitized execution facts about one solver run.

    Phase 2 spec §17 minus ``logical_variables`` / ``logical_interactions``
    (2026-09-09 review F-18): no backend ever filled those two, so they
    were two always-null fields in the MCP output schema.

    It describes the *last completed attempt*. A local backend reports it
    too, with ``remote: false``, an empty ``timing_us`` and null in every
    vendor-side field.
    """

    backend: str = Field(description="The backend that produced this run.")
    remote: bool = Field(
        description=(
            "Whether the run left this machine. False for the local backends, "
            "which have no vendor side at all."
        )
    )
    solver_id: str | None = Field(
        default=None,
        description=(
            "Vendor-side solver identifier, e.g. fujitsuDA3/v4. Null on a "
            "local backend, or when the vendor reported none."
        ),
    )
    num_reads_requested: int | None = Field(
        default=None,
        description=(
            "Reads actually requested of the sampler. Null on a backend that "
            "does not sample repeatedly."
        ),
    )
    effective_time_limit_seconds: float | None = Field(
        default=None,
        description=(
            "The time limit actually used, after a sampler's own minimum was "
            "applied. Null when the backend takes no time limit."
        ),
    )
    timing_us: dict[str, float] = Field(
        default={},
        description=(
            "Vendor-reported timing facts in microseconds, filtered through a "
            "fixed whitelist of keys so nothing else a vendor reports leaves "
            "the solver layer. Empty on a local backend and whenever the "
            "backend reported none."
        ),
    )
    average_chain_break_fraction: float | None = Field(
        default=None,
        description="QPU only: the mean fraction of broken embedding chains.",
    )
    embedding_max_chain_length: int | None = Field(
        default=None,
        description="QPU only: the longest chain in the embedding that was used.",
    )
    # 3a spec §22: filled in by the service (the backend never knows it).
    model_type: ModelType | None = Field(
        default=None,
        description=(
            "Which compiler path ran. Filled in by the service, because the "
            "backend does not know it."
        ),
    )
    # 3a spec §22: only constraint-model backends report this; it is
    # informational and never feeds feasibility or ranking.
    sampler_reported_feasible: int | None = Field(
        default=None,
        description=(
            "How many samples the sampler itself called feasible. "
            "Informational only: every sample is still independently "
            "re-validated against the original problem, and this number feeds "
            "neither feasibility nor ranking. Only a constraint-model backend "
            "fills it; null everywhere else."
        ),
    )
