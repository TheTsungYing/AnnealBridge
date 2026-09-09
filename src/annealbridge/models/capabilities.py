"""Solver capability declarations and availability status (Phase 3a spec §7, §8).

Pure Pydantic models, kept in ``models`` (not ``solvers``) because both the
validation layer and the orchestration policy need to read a backend's
declaration, and ``validation`` must not import ``solvers`` (spec §4).
``solvers.base`` re-exports these names so
``from annealbridge.solvers import SolverCapabilities`` keeps working.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# A new model type is the core's business (it needs a compiler), not
# something a backend plugin may declare freely -- hence a Literal.
ModelType = Literal["bqm", "cqm"]

AvailabilityCategory = Literal[
    "available",
    "not_installed",
    "credentials_missing",
    "config_invalid",
    "unavailable",
]


class ParameterLimit(BaseModel):
    """One user preference a backend declares as policy-limited (spec §7).

    ``preference`` is the dotted path into ``SolverPreferences`` (e.g.
    ``"num_reads"`` or ``"dwave_qpu.annealing_time_us"``), ``limit`` the
    generic ``ExecutionPolicy`` limit key (spec §11.1) and ``error_code``
    the catalog code reported when the preference exceeds the limit.
    """

    preference: str
    limit: str
    error_code: str


class CredentialDeclaration(BaseModel):
    """What a backend's credential material *looks like*, so the shared
    redaction can mask it (Phase 2 spec §19; 2026-09-09 review F-10).

    Names and shapes only — never values. A backend that declares nothing
    is still covered by the protocol-level fallbacks in ``solvers.metadata``
    (``token=`` query strings, ``Authorization`` headers), but anything
    vendor-specific must be declared here; the shared layer knows no vendor.

    * ``env_vars``: environment variables holding a credential. ``redact()``
      reads each one live on every call and masks its non-empty value.
    * ``header_names``: HTTP header names carrying a credential. Both the
      ``"<Name>: <value>"`` line form and the JSON ``"<Name>": "<value>"``
      form are masked.
    * ``value_patterns``: regular expressions (source strings) matching the
      vendor's token shape, for a value that reaches the text without ever
      having been in this process's environment (a stale token echoed by
      the vendor, a token from another profile).
    """

    env_vars: list[str] = Field(default_factory=list)
    header_names: list[str] = Field(default_factory=list)
    value_patterns: list[str] = Field(default_factory=list)

    @field_validator("env_vars", "header_names")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        for name in value:
            if not name.strip():
                raise ValueError("credential names must be non-empty strings")
        return value

    @field_validator("value_patterns")
    @classmethod
    def _patterns_compile(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid credential value pattern {pattern!r}: {exc}") from exc
        return value

    @property
    def empty(self) -> bool:
        return not (self.env_vars or self.header_names or self.value_patterns)


class AvailabilityStatus(BaseModel):
    """Structured answer of ``SolverBackend.is_available()`` (spec §8).

    ``detail`` is categorical text (e.g. "dwave-system not installed") and
    must never contain configuration values. ``error_code`` lets a backend
    name the catalog code to report; when None the service picks a default
    from ``category`` (spec §8.2).
    """

    category: AvailabilityCategory
    detail: str | None = None
    error_code: str | None = None

    @property
    def available(self) -> bool:
        return self.category == "available"


class SolverCapabilities(BaseModel):
    """Static self-description of a solver backend (Phase 2 §10, 3a §7).

    The Phase 2 fields are unchanged. The 3a additions all default, so a
    backend that declares nothing new behaves as before: it accepts no
    ``num_sweeps``, needs no embedding and has no policy-limited parameters.
    ``credentials`` (review F-10) also defaults to an empty declaration: a
    local backend has nothing to redact, a remote one declares its
    credential env vars / headers here and ``SolverRegistry`` feeds the
    declaration to the shared redaction — no change to the solver layer.
    """

    name: str
    remote: bool
    heuristic: bool
    exhaustive: bool
    supports_seed: bool
    supports_num_reads: bool
    supports_time_limit: bool
    # Preference order; the service takes the first type it has a compiler for.
    supported_model_types: list[ModelType]
    returns_multiple_samples: bool
    description: str
    supports_num_sweeps: bool = False
    requires_embedding: bool = False
    parameter_limits: list[ParameterLimit] = Field(default_factory=list)
    credentials: CredentialDeclaration = Field(default_factory=CredentialDeclaration)

    @field_validator("supported_model_types")
    @classmethod
    def _at_least_one_model_type(cls, value: list[ModelType]) -> list[ModelType]:
        if not value:
            raise ValueError("supported_model_types must contain at least one model type")
        return value

    @property
    def preferred_model_type(self) -> ModelType:
        return self.supported_model_types[0]
