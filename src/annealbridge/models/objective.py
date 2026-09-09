"""Objective function models."""

from typing import Literal

from pydantic import BaseModel

from annealbridge.models.quantities import Quantity


class LinearTerm(BaseModel):
    """A linear term: coefficient * variable."""

    variable: str
    # Quantity, not plain float: booleans and strings are refused rather than
    # coerced (see models/quantities.py, 2026-09-09 review F-11).
    coefficient: Quantity


class QuadraticTerm(BaseModel):
    """A quadratic term: coefficient * variable1 * variable2."""

    variable1: str
    variable2: str
    coefficient: Quantity


class Objective(BaseModel):
    """The business objective to minimize or maximize."""

    direction: Literal["minimize", "maximize"]
    linear_terms: list[LinearTerm]
    quadratic_terms: list[QuadraticTerm] = []
    constant: Quantity = 0
