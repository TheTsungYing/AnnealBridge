"""Solver capability declarations and availability status (Phase 3a spec §7, §8).

Pure Pydantic models, kept in ``models`` (not ``solvers``) because both the
validation layer and the orchestration policy need to read a backend's
declaration, and ``validation`` must not import ``solvers`` (spec §4).
``solvers.base`` re-exports these names so
``from annealbridge.solvers import SolverCapabilities`` keeps working.
"""

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

    @field_validator("supported_model_types")
    @classmethod
    def _at_least_one_model_type(cls, value: list[ModelType]) -> list[ModelType]:
        if not value:
            raise ValueError("supported_model_types must contain at least one model type")
        return value

    @property
    def preferred_model_type(self) -> ModelType:
        return self.supported_model_types[0]
