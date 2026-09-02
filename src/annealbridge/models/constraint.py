"""Constraint model."""

from typing import Literal

from pydantic import BaseModel

from annealbridge.models.objective import LinearTerm


class Constraint(BaseModel):
    """A linear constraint over binary variables.

    Hard constraints must be satisfied; soft constraints carry a weight
    expressing business preference importance.
    """

    id: str
    description: str | None = None
    type: Literal["hard", "soft"]
    terms: list[LinearTerm]
    operator: Literal["==", "<=", ">="]
    rhs: float
    weight: float | None = None
