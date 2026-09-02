"""Execution policy for resource limits and remote access (Phase 2 spec §8).

A pure Pydantic model: it never reads the environment. The composition
root (``annealbridge.config``) builds one from env vars and injects it
into :class:`~annealbridge.orchestration.optimizer.OptimizationService`.
"""

from pydantic import BaseModel, Field


class ExecutionPolicy(BaseModel):
    """Limits the service enforces on every solve (spec §8, §14).

    Every limit has a lower bound: a zero or negative ceiling would reject
    every solve (or, for ``max_concurrent_solves``, break the semaphore the
    service builds from it), so such values are configuration errors.
    """

    allow_remote: bool = False
    allow_remote_retries: bool = False
    exact_max_variables: int = Field(default=24, ge=1)  # compiled variables, slack included
    max_qpu_reads: int = Field(default=1000, ge=1)
    max_qpu_annealing_time_us: float = Field(default=2000.0, gt=0)
    max_remote_time_seconds: int = Field(default=300, ge=1)  # hybrid time_limit upper bound
    max_concurrent_solves: int = Field(default=4, ge=1)
    enabled_backends: set[str] | None = None   # None = all registry backends
