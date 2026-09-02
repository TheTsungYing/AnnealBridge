"""Compiled problem and constraint trace models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from annealbridge.models.problem import OptimizationProblem


class ConstraintTrace(BaseModel):
    """Explainability record of how one constraint was compiled."""

    constraint_id: str
    constraint_type: Literal["hard", "soft"]
    operator: Literal["==", "<=", ">="]
    source_description: str | None
    generated_variables: list[str]
    penalty: float
    slack_range: int | None
    redundant: bool = False
    compiler: str


class CompiledProblem(BaseModel):
    """A compiled solver model plus provenance back to the original problem.

    ``model`` is intentionally untyped (Any); orchestration must not depend
    on its concrete type.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    original_problem: OptimizationProblem
    internal_variables: set[str]
    constraint_trace: list[ConstraintTrace]
    hard_penalty: float
    objective_scale: float
    num_variables: int
