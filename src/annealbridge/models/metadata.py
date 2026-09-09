"""Solver execution metadata model (Phase 2 spec §17).

Defined in the models layer so that ``SolveResult`` (models) can reference
it without a models → solvers dependency; the sanitizer and redaction
helpers that produce it live in ``annealbridge.solvers.metadata``.
"""

from pydantic import BaseModel

from annealbridge.models.capabilities import ModelType


class SolverExecutionMetadata(BaseModel):
    """Sanitized execution facts about one solver run.

    Phase 2 spec §17 minus ``logical_variables`` / ``logical_interactions``
    (2026-09-09 review F-18): no backend ever filled those two, so they
    were two always-null fields in the MCP output schema.
    """

    backend: str
    remote: bool
    solver_id: str | None = None
    num_reads_requested: int | None = None
    effective_time_limit_seconds: float | None = None
    timing_us: dict[str, float] = {}
    average_chain_break_fraction: float | None = None
    embedding_max_chain_length: int | None = None
    # 3a spec §22: filled in by the service (the backend never knows it).
    model_type: ModelType | None = None
    # 3a spec §22: only constraint-model backends report this; it is
    # informational and never feeds feasibility or ranking.
    sampler_reported_feasible: int | None = None
