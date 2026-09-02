"""Solver execution metadata model (Phase 2 spec §17).

Defined in the models layer so that ``SolveResult`` (models) can reference
it without a models → solvers dependency; the sanitizer and redaction
helpers that produce it live in ``annealbridge.solvers.metadata``.
"""

from pydantic import BaseModel


class SolverExecutionMetadata(BaseModel):
    """Sanitized execution facts about one solver run."""

    backend: str
    remote: bool
    solver_id: str | None = None
    logical_variables: int | None = None
    logical_interactions: int | None = None
    num_reads_requested: int | None = None
    effective_time_limit_seconds: float | None = None
    timing_us: dict[str, float] = {}
    average_chain_break_fraction: float | None = None
    embedding_max_chain_length: int | None = None
