"""Solver capability declarations and availability status (Phase 3a spec §7, §8).

Pure Pydantic models, kept in ``models`` (not ``solvers``) because both the
validation layer and the orchestration policy need to read a backend's
declaration, and ``validation`` must not import ``solvers`` (spec §4).
``solvers.base`` re-exports these names so
``from annealbridge.solvers import SolverCapabilities`` keeps working.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# A new model type is the core's business (it needs a compiler), not
# something a backend plugin may declare freely -- hence a Literal.
ModelType = Literal["bqm", "cqm"]

# Why a backend cannot run. The first four are the only categories the seven
# built-in backends ever report; ``"unavailable"`` is the generic escape
# hatch for a third-party backend that just cannot be reached and has no
# more specific reason to give. ``orchestration/limits.py``'s
# ``AVAILABILITY_MAP`` maps every category to a status plus a default error
# code, and maps ``"unavailable"`` to ``BACKEND_UNAVAILABLE``.
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

    preference: str = Field(
        description=(
            "Dotted path into SolverPreferences, e.g. num_reads or "
            "dwave_qpu.annealing_time_us."
        )
    )
    limit: str = Field(
        description=(
            "The generic ExecutionPolicy limit key this preference is checked "
            "against."
        )
    )
    error_code: str = Field(
        description=(
            "The error-catalog code reported when the preference exceeds the "
            "limit. The value is refused, never clamped."
        )
    )


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

    env_vars: list[str] = Field(
        default_factory=list,
        description=(
            "Names of environment variables holding a credential; each is "
            "read live on every redaction and its non-empty value masked. "
            "Names only — never values."
        ),
    )
    header_names: list[str] = Field(
        default_factory=list,
        description=(
            "Names of HTTP headers carrying a credential; both the "
            "'<Name>: <value>' line form and the JSON form are masked."
        ),
    )
    value_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Regular expressions matching the vendor's token shape, for a "
            "value that reaches the text without ever having been in this "
            "process's environment."
        ),
    )

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

    category: AvailabilityCategory = Field(
        description=(
            "Why the backend can or cannot run: available, not_installed, "
            "credentials_missing, config_invalid, or the generic unavailable."
        )
    )
    detail: str | None = Field(
        default=None,
        description=(
            "Categorical explanation, e.g. 'dwave-system not installed'. "
            "Never contains configuration values; null when there is nothing "
            "to add."
        ),
    )
    error_code: str | None = Field(
        default=None,
        description=(
            "The error-catalog code this backend wants reported. Null lets "
            "the service pick the default for the category."
        ),
    )

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

    name: str = Field(description="The backend's own name for itself.")
    remote: bool = Field(
        description="Whether running it sends the compiled model off this machine."
    )
    heuristic: bool = Field(
        description="Whether it may return a sub-optimal answer."
    )
    exhaustive: bool = Field(
        description=(
            "Whether it enumerates every assignment, so a result can prove "
            "optimality or infeasibility."
        )
    )
    supports_seed: bool = Field(
        description="Whether solver.seed has any effect on this backend."
    )
    supports_num_reads: bool = Field(
        description="Whether it samples repeatedly and so honours solver.num_reads."
    )
    supports_time_limit: bool = Field(
        description="Whether it takes a time budget rather than a read count."
    )
    supported_model_types: list[ModelType] = Field(
        description=(
            "The compiled model types it accepts, in preference order; the "
            "service takes the first type it has a compiler for. At least one."
        )
    )
    returns_multiple_samples: bool = Field(
        description=(
            "Whether one call can yield more than one candidate. False means "
            "the effective top_k is at most 1."
        )
    )
    description: str = Field(description="One-line description of the backend.")
    supports_num_sweeps: bool = Field(
        default=False,
        description="Whether it honours solver.num_sweeps.",
    )
    # strict: a declaration is code, and ``True`` or ``"0"`` read as a bound
    # would be a typo accepted silently.
    seed_min: int | None = Field(
        default=None,
        strict=True,
        description=(
            "The smallest solver.seed the backend accepts, inclusive. Declared "
            "together with seed_max, and only when supports_seed is true; "
            "validation refuses a seed outside the range before anything runs. "
            "None means no range is declared."
        ),
    )
    seed_max: int | None = Field(
        default=None,
        strict=True,
        description=(
            "The largest solver.seed the backend accepts, inclusive. Declared "
            "together with seed_min."
        ),
    )
    requires_embedding: bool = Field(
        default=False,
        description=(
            "Whether the model must be minor-embedded onto hardware before it "
            "can run."
        ),
    )
    parameter_limits: list[ParameterLimit] = Field(
        default_factory=list,
        description=(
            "The user preferences this backend declares as policy-limited. "
            "Empty means no preference of its own is capped."
        ),
    )
    credentials: CredentialDeclaration = Field(
        default_factory=CredentialDeclaration,
        description=(
            "What this backend's credential material looks like, so the "
            "shared redaction can mask it. Shapes and names only, never "
            "values; empty on a local backend."
        ),
    )

    @field_validator("supported_model_types")
    @classmethod
    def _at_least_one_model_type(cls, value: list[ModelType]) -> list[ModelType]:
        if not value:
            raise ValueError("supported_model_types must contain at least one model type")
        return value

    @model_validator(mode="after")
    def _consistent_seed_range(self) -> "SolverCapabilities":
        # A contradictory range is a backend bug: refused at construction,
        # never read as "no range" and so silently unchecked.
        if (self.seed_min is None) != (self.seed_max is None):
            raise ValueError("seed_min and seed_max must be declared together")
        if self.seed_min is None:
            return self
        if not self.supports_seed:
            raise ValueError("a seed range needs supports_seed=True")
        if self.seed_min > self.seed_max:
            raise ValueError(
                f"seed_min ({self.seed_min}) must not exceed seed_max ({self.seed_max})"
            )
        return self

    @property
    def preferred_model_type(self) -> ModelType:
        return self.supported_model_types[0]
