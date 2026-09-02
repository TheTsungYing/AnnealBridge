"""Execution policy for resource limits and remote access (Phase 2 spec §8, 3a §11).

A pure Pydantic model: it never reads the environment. The composition
root (``annealbridge.config``) builds one from env vars and injects it
into :class:`~annealbridge.orchestration.optimizer.OptimizationService`.
"""

import math

from pydantic import BaseModel, Field, field_validator

from annealbridge.models import SolverCapabilities

# Generic limit key → the Phase 2 field that still carries its value
# (spec §11.1). The fields, their defaults, bounds and env names are the
# compatibility layer; new backends only ever add keys to ``limits``.
COMPATIBILITY_LIMIT_FIELDS: dict[str, str] = {
    "variables": "exact_max_variables",
    "reads": "max_qpu_reads",
    "annealing_time_us": "max_qpu_annealing_time_us",
    "time_seconds": "max_remote_time_seconds",
}


def validate_limits(limits: dict[str, float]) -> dict[str, float]:
    """Reject a ``limits`` mapping that would be ambiguous or unenforceable.

    A key that is also a compatibility key would give one limit two
    sources; a value that is not finite or not positive would either
    reject every solve or defeat every ``value > limit`` comparison.
    Shared by :class:`ExecutionPolicy` and the env-driven settings so the
    same rule applies wherever the mapping enters the process.
    """
    for key, value in limits.items():
        if key in COMPATIBILITY_LIMIT_FIELDS:
            raise ValueError(
                f"limit '{key}' is a built-in key; set "
                f"'{COMPATIBILITY_LIMIT_FIELDS[key]}' instead of limits['{key}']"
            )
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"limit '{key}' must be a finite number greater than 0 (got {value!r})"
            )
    return limits


class ExecutionPolicy(BaseModel):
    """Limits the service enforces on every solve (spec §8, §14, 3a §11).

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
    # Generic limits keyed by the names backends declare in their
    # ``parameter_limits`` (spec §11). Every value is finite and > 0; the
    # four compatibility keys above are refused here so each limit has
    # exactly one source.
    limits: dict[str, float] = Field(default_factory=dict)

    @field_validator("limits")
    @classmethod
    def _validate_limits(cls, value: dict[str, float]) -> dict[str, float]:
        return validate_limits(value)

    def limit(self, key: str) -> float | int | None:
        """Generic lookup: ``limits`` first, then the compatibility field.

        Compatibility keys return the field's own type (``int`` for
        ``variables``, ``reads`` and ``time_seconds``; ``float`` for
        ``annealing_time_us``); ``limits`` values are always ``float``.
        Returns None when the policy has no value for ``key``.
        """
        if key in self.limits:
            return self.limits[key]
        field = COMPATIBILITY_LIMIT_FIELDS.get(key)
        if field is None:
            return None
        return getattr(self, field)

    def limits_for(self, capabilities: SolverCapabilities) -> dict[str, float | int]:
        """The limits this policy applies to a backend, keyed ``max_<limit>``.

        The single source for both the capabilities view and the service
        (spec §12.3): what an agent reads here is exactly what a solve is
        checked against. The exhaustive variable ceiling and the effective
        remote time limit are flag-driven (they need compiled data); the
        rest follows the backend's declared ``parameter_limits``.
        """
        result: dict[str, float | int] = {}
        if capabilities.exhaustive:
            result["max_variables"] = self.limit("variables")
        for declaration in capabilities.parameter_limits:
            result[f"max_{declaration.limit}"] = self.limit(declaration.limit)
        if capabilities.remote and capabilities.supports_time_limit:
            result["max_time_seconds"] = self.limit("time_seconds")
        return result
