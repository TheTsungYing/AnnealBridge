"""Execution policy for resource limits and remote access (Phase 2 spec §8).

A pure Pydantic model: it never reads the environment. The composition
root (``annealbridge.config``) builds one from env vars and injects it
into :class:`~annealbridge.orchestration.optimizer.OptimizationService`.
"""

from pydantic import BaseModel


class ExecutionPolicy(BaseModel):
    """Limits the service enforces on every solve (spec §8, §14)."""

    allow_remote: bool = False
    allow_remote_retries: bool = False
    exact_max_variables: int = 24          # compiled variables, slack included
    max_qpu_reads: int = 1000
    max_qpu_annealing_time_us: float = 2000.0
    max_remote_time_seconds: int = 300     # hybrid time_limit upper bound
    max_concurrent_solves: int = 4
    enabled_backends: set[str] | None = None   # None = all registry backends
