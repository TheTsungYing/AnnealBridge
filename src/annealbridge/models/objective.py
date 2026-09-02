"""Objective function models."""

from typing import Literal

from pydantic import BaseModel


class LinearTerm(BaseModel):
    """A linear term: coefficient * variable."""

    variable: str
    coefficient: float


class QuadraticTerm(BaseModel):
    """A quadratic term: coefficient * variable1 * variable2."""

    variable1: str
    variable2: str
    coefficient: float


class Objective(BaseModel):
    """The business objective to minimize or maximize."""

    direction: Literal["minimize", "maximize"]
    linear_terms: list[LinearTerm]
    quadratic_terms: list[QuadraticTerm] = []
    constant: float = 0
