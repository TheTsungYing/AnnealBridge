"""Optimization problem schema — the public JSON API (versions 1.0 and 1.1).

``"1.0"`` problems only have binary variables; ``"1.1"`` (3b spec §7) is a
superset that also allows bounded integer variables. A 1.0 problem carrying
an integer variable is rejected by the problem validator, never silently
upgraded.
"""

from typing import Literal

from pydantic import BaseModel, Field

from annealbridge.models.constraint import Constraint
from annealbridge.models.objective import Objective
from annealbridge.models.variable import Variable


# The schema layer rejects what is not a number at all (NaN / ±inf, which
# would also defeat every `value > limit` policy comparison); the semantic
# "> 0" rule is the problem validator's job so it surfaces as invalid_problem
# with a recommended_action instead of a bare type error.
_FINITE = Field(default=None, allow_inf_nan=False)


class DWaveQPUOptions(BaseModel):
    """Options specific to the D-Wave QPU backend (Phase 2 spec §12)."""

    annealing_time_us: float | None = _FINITE
    chain_strength: float | None = _FINITE  # None → Ocean 預設 (uniform_torque_compensation)
    auto_scale: bool = True


class LeapHybridBQMOptions(BaseModel):
    """Options specific to the Leap hybrid BQM backend (Phase 2 spec §12)."""

    time_limit_seconds: float | None = _FINITE  # None → sampler 最小值


class LeapHybridCQMOptions(BaseModel):
    """Options specific to the Leap hybrid CQM backend (Phase 3a spec §18)."""

    time_limit_seconds: float | None = _FINITE  # None → sampler 最小值


class FujitsuDAOptions(BaseModel):
    """Options specific to the Fujitsu Digital Annealer backend (Phase 3b spec §20.2).

    ``None`` means "do not send the field" so the vendor default applies
    (time_limit_sec 10, num_run 16, num_group 1, num_output_solution 5).
    The ranges are the vendor's documented ranges for QUBO API V4."""

    time_limit_seconds: int | None = Field(default=None, ge=1, le=3600)
    num_run: int | None = Field(default=None, ge=1, le=1024)
    num_group: int | None = Field(default=None, ge=1, le=16)
    num_output_solution: int | None = Field(default=None, ge=1, le=1024)


class SolverPreferences(BaseModel):
    """Caller preferences for solver backend and search parameters.

    Each backend-specific option block is a field named exactly like the
    backend it belongs to (3a §9.3 naming contract): a backend declares its
    limited parameters as dotted paths such as
    ``"leap_hybrid_cqm.time_limit_seconds"`` and the service reads them by
    that path, so the block name and the backend name must agree.
    """

    backend: Literal[
        "simulated_annealing",
        "exact",
        "dwave_qpu",
        "leap_hybrid_bqm",
        "leap_hybrid_cqm",
        "fujitsu_da",
    ] = "simulated_annealing"
    num_reads: int = 100
    num_sweeps: int = 1000
    seed: int | None = None
    top_k: int = 5
    max_retries: int = 3
    # Finite only (2026-09-09 review F-08): ``nan`` would silently disable
    # every hard penalty and ``inf`` would break the compiled model.
    penalty_multiplier: float = Field(default=2.0, allow_inf_nan=False)
    dwave_qpu: DWaveQPUOptions | None = None
    leap_hybrid_bqm: LeapHybridBQMOptions | None = None
    leap_hybrid_cqm: LeapHybridCQMOptions | None = None
    fujitsu_da: FujitsuDAOptions | None = None


class OptimizationProblem(BaseModel):
    """A structured combinatorial optimization problem."""

    version: Literal["1.0", "1.1"] = "1.0"
    name: str
    description: str | None = None
    variables: list[Variable]
    objective: Objective
    constraints: list[Constraint]
    solver: SolverPreferences = Field(default_factory=SolverPreferences)
