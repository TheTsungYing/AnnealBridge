"""Constraint model."""

from typing import Literal

from pydantic import BaseModel

from annealbridge.models.objective import LinearTerm
from annealbridge.models.quantities import Quantity


class Constraint(BaseModel):
    """A linear constraint over the problem's variables (binary or bounded
    integer, IR 1.1).

    Hard constraints must be satisfied; soft constraints carry a weight
    expressing business preference importance.

    ``rhs`` / ``weight`` are ``Quantity`` (models/quantities.py): a boolean or
    a string is refused instead of being coerced (2026-09-09 review F-11).
    """

    id: str
    description: str | None = None
    type: Literal["hard", "soft"]
    terms: list[LinearTerm]
    operator: Literal["==", "<=", ">="]
    rhs: Quantity
    weight: Quantity | None = None
