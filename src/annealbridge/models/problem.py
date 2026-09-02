"""Optimization problem schema — the public JSON API (version 1.0)."""

from typing import Literal

from pydantic import BaseModel, Field

from annealbridge.models.constraint import Constraint
from annealbridge.models.objective import Objective
from annealbridge.models.variable import Variable


class DWaveQPUOptions(BaseModel):
    """Options specific to the D-Wave QPU backend (Phase 2 spec §12)."""

    annealing_time_us: float | None = None
    chain_strength: float | None = None  # None → Ocean 預設 (uniform_torque_compensation)
    auto_scale: bool = True


class LeapHybridBQMOptions(BaseModel):
    """Options specific to the Leap hybrid BQM backend (Phase 2 spec §12)."""

    time_limit_seconds: float | None = None  # None → sampler 最小值


class SolverPreferences(BaseModel):
    """Caller preferences for solver backend and search parameters."""

    backend: Literal[
        "simulated_annealing", "exact", "dwave_qpu", "leap_hybrid_bqm"
    ] = "simulated_annealing"
    num_reads: int = 100
    num_sweeps: int = 1000
    seed: int | None = None
    top_k: int = 5
    max_retries: int = 3
    penalty_multiplier: float = 2.0
    dwave_qpu: DWaveQPUOptions | None = None
    leap_hybrid_bqm: LeapHybridBQMOptions | None = None


class OptimizationProblem(BaseModel):
    """A structured combinatorial optimization problem."""

    version: Literal["1.0"] = "1.0"
    name: str
    description: str | None = None
    variables: list[Variable]
    objective: Objective
    constraints: list[Constraint]
    solver: SolverPreferences = Field(default_factory=SolverPreferences)
