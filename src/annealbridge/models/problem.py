"""Optimization problem schema — the public JSON API (versions 1.0 and 1.1).

``"1.0"`` problems only have binary variables; ``"1.1"`` (3b spec §7) is a
superset that also allows bounded integer variables. A 1.0 problem carrying
an integer variable is rejected by the problem validator, never silently
upgraded.
"""

from typing import Literal

from pydantic import Field

from annealbridge.models.constraint import Constraint
from annealbridge.models.objective import Objective
from annealbridge.models.quantities import Count, Quantity
from annealbridge.models.strict import InputModel
from annealbridge.models.variable import Variable


# The schema layer rejects what is not a number at all (NaN / ±inf, which
# would also defeat every `value > limit` policy comparison); the semantic
# "> 0" rule is the problem validator's job so it surfaces as invalid_problem
# with a recommended_action instead of a bare type error. The numeric fields
# below are Quantity / Count (models/quantities.py) so a boolean or a string
# is refused rather than coerced (2026-09-09 review F-11).
def _finite(description: str):
    """An optional finite number field carrying ``description``."""
    return Field(default=None, allow_inf_nan=False, description=description)


class DWaveQPUOptions(InputModel):
    """Options specific to the D-Wave QPU backend (Phase 2 spec §12)."""

    annealing_time_us: Quantity | None = _finite(
        "Annealing time per read, in microseconds. Must be positive; null "
        "leaves the QPU's own default in place."
    )
    # None → Ocean 預設 (uniform_torque_compensation)
    chain_strength: Quantity | None = _finite(
        "Strength of the couplings inside an embedding chain. Must be "
        "positive; null uses Ocean's default (uniform_torque_compensation)."
    )
    auto_scale: bool = Field(
        default=True,
        description=(
            "Whether the sampler may rescale the model's biases into the "
            "QPU's own range."
        ),
    )


class LeapHybridBQMOptions(InputModel):
    """Options specific to the Leap hybrid BQM backend (Phase 2 spec §12)."""

    # None → sampler 最小值
    time_limit_seconds: Quantity | None = _finite(
        "Time budget for the hybrid solver, in seconds. Must be positive; "
        "null uses the sampler's own minimum for the model."
    )


class LeapHybridCQMOptions(InputModel):
    """Options specific to the Leap hybrid CQM backend (Phase 3a spec §18)."""

    # None → sampler 最小值
    time_limit_seconds: Quantity | None = _finite(
        "Time budget for the hybrid CQM solver, in seconds. Must be positive; "
        "a value below the sampler's minimum is raised to it and reported in "
        "the result metadata."
    )


class FujitsuDAOptions(InputModel):
    """Options specific to the Fujitsu Digital Annealer backend (Phase 3b spec §20.2).

    ``None`` means "do not send the field" so the vendor default applies
    (time_limit_sec 10, num_run 16, num_group 1, num_output_solution 5).
    The ranges are the vendor's documented ranges for QUBO API V4."""

    time_limit_seconds: Count | None = Field(
        default=None,
        ge=1,
        le=3600,
        description=(
            "Annealing time budget in seconds, within the vendor's documented "
            "range. Null sends nothing, so the vendor default applies."
        ),
    )
    num_run: Count | None = Field(
        default=None,
        ge=1,
        le=1024,
        description=(
            "Number of parallel annealing runs, within the vendor's "
            "documented range. Null sends nothing, so the vendor default "
            "applies."
        ),
    )
    num_group: Count | None = Field(
        default=None,
        ge=1,
        le=16,
        description=(
            "Number of groups per run, within the vendor's documented range. "
            "Null sends nothing, so the vendor default applies."
        ),
    )
    num_output_solution: Count | None = Field(
        default=None,
        ge=1,
        le=1024,
        description=(
            "Number of solutions returned per group, within the vendor's "
            "documented range. Null sends nothing, so the vendor default "
            "applies."
        ),
    )


class SolverPreferences(InputModel):
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
    ] = Field(
        default="simulated_annealing",
        description=(
            "Solver backend to run the problem on: one of the names the "
            "capabilities tool lists, which also says whether it is "
            "available and enabled on this server. Never substituted."
        ),
    )
    num_reads: Count = Field(
        default=100,
        description=(
            "Number of samples to request. Must be positive; ignored (with a "
            "PARAMETER_IGNORED warning when non-default) by backends that do "
            "not sample repeatedly."
        ),
    )
    num_sweeps: Count = Field(
        default=1000,
        description=(
            "Annealing sweeps per read. Must be positive; ignored (with a "
            "PARAMETER_IGNORED warning when non-default) by backends that "
            "take no sweeps."
        ),
    )
    seed: Count | None = Field(
        default=None,
        description=(
            "Random seed for backends that support seeding, giving "
            "reproducible sampling; ignored (with a SEED_IGNORED warning) on "
            "the others."
        ),
    )
    top_k: Count = Field(
        default=5,
        description=(
            "Maximum number of ranked solutions to return. Must be positive; "
            "a backend that returns a single sample yields at most one."
        ),
    )
    max_retries: Count = Field(
        default=3,
        description=(
            "Additional attempts, each with a doubled hard-constraint "
            "penalty, when no feasible solution was found. Zero or more; "
            "ignored on exhaustive backends and on the CQM path."
        ),
    )
    # Finite only (2026-09-09 review F-08): ``nan`` would silently disable
    # every hard penalty and ``inf`` would break the compiled model.
    penalty_multiplier: Quantity = Field(
        default=2.0,
        allow_inf_nan=False,
        description=(
            "Multiplier applied to the server's penalty scale for the first "
            "attempt's hard-constraint penalty. Must be finite and positive; "
            "leave at the default, as hard penalties are sized by the server."
        ),
    )
    dwave_qpu: DWaveQPUOptions | None = Field(
        default=None,
        description=(
            "Options for the dwave_qpu backend. Filling in a block that does "
            "not match the selected backend raises PARAMETER_IGNORED."
        ),
    )
    leap_hybrid_bqm: LeapHybridBQMOptions | None = Field(
        default=None,
        description=(
            "Options for the leap_hybrid_bqm backend. Filling in a block that "
            "does not match the selected backend raises PARAMETER_IGNORED."
        ),
    )
    leap_hybrid_cqm: LeapHybridCQMOptions | None = Field(
        default=None,
        description=(
            "Options for the leap_hybrid_cqm backend. Filling in a block that "
            "does not match the selected backend raises PARAMETER_IGNORED."
        ),
    )
    fujitsu_da: FujitsuDAOptions | None = Field(
        default=None,
        description=(
            "Options for the fujitsu_da backend. Filling in a block that does "
            "not match the selected backend raises PARAMETER_IGNORED."
        ),
    )


class OptimizationProblem(InputModel):
    """A structured combinatorial optimization problem."""

    version: Literal["1.0", "1.1"] = Field(
        default="1.0",
        description=(
            'Problem schema version. "1.1" is a superset of "1.0" and is '
            "required as soon as any variable is an integer."
        ),
    )
    name: str = Field(
        description="Problem name; echoed in reports, logs and results."
    )
    description: str | None = Field(
        default=None,
        description=(
            "Free text describing the problem. For humans only; never sent to "
            "a remote vendor."
        ),
    )
    variables: list[Variable] = Field(
        description=(
            "The decision variables. At least one is required, and every name "
            "referenced by a term must be declared here."
        )
    )
    objective: Objective = Field(
        description="What to minimize or maximize over the declared variables."
    )
    constraints: list[Constraint] = Field(
        description=(
            "Hard and soft constraints over the declared variables. May be "
            "empty."
        )
    )
    solver: SolverPreferences = Field(
        default_factory=SolverPreferences,
        description=(
            "Backend choice and search parameters. Optional: every field has "
            "a default."
        ),
    )
