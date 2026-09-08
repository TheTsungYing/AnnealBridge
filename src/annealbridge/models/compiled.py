"""Compiled problem and constraint trace models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from annealbridge.models.capabilities import ModelType
from annealbridge.models.problem import OptimizationProblem


class ConstraintTrace(BaseModel):
    """Explainability record of how one constraint was compiled."""

    constraint_id: str
    constraint_type: Literal["hard", "soft"]
    operator: Literal["==", "<=", ">="]
    source_description: str | None
    generated_variables: list[str]
    # None when the constraint carries no penalty term (a native CQM hard
    # constraint); soft constraints always record their weight.
    penalty: float | None
    slack_range: int | None
    redundant: bool = False
    # True when the constraint is expressed natively by the model instead
    # of as a penalty term (3a spec §14).
    native: bool = False
    compiler: str


class IntegerEncoding(BaseModel):
    """Binary encoding of one integer variable on the BQM path (3b spec §12).

    ``value = lower + sum(coefficients[k] * bits[k])`` where ``bits`` are
    internal (``__int_<name>_<k>``) variable names and ``coefficients`` come
    from ``compute_slack_coefficients(upper - lower)``. Produced by the BQM
    compiler and consumed only by its ``decode``; the CQM path has none.
    """

    variable: str
    lower: int
    bits: list[str]
    coefficients: list[int]


class CompiledProblem(BaseModel):
    """A compiled solver model plus provenance back to the original problem.

    ``model`` is intentionally untyped (Any); orchestration must not depend
    on its concrete type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Defaults to "bqm" so Phase 1/2 callers keep working unchanged.
    model_type: ModelType = "bqm"
    model: Any
    original_problem: OptimizationProblem
    internal_variables: set[str]
    constraint_trace: list[ConstraintTrace]
    # None for model types that express hard constraints natively (CQM).
    hard_penalty: float | None
    objective_scale: float
    num_variables: int
    # 3b spec §12: filled by the BQM compiler for integer variables (one
    # entry per variable, keyed by its name); empty on the CQM path and for
    # binary-only problems.
    integer_encodings: dict[str, IntegerEncoding] = {}
