"""Objective function models."""

from typing import Literal

from pydantic import Field

from annealbridge.models.quantities import Quantity
from annealbridge.models.strict import InputModel


class LinearTerm(InputModel):
    """A linear term: coefficient * variable."""

    variable: str = Field(
        description=(
            "Name of a variable declared in the problem's variables list."
        )
    )
    # Quantity, not plain float: booleans and strings are refused rather than
    # coerced (see models/quantities.py, 2026-09-09 review F-11).
    coefficient: Quantity = Field(
        description=(
            "Finite factor multiplying the variable. Must be an integral "
            "value inside a <= / >= constraint, where slack encoding needs an "
            "exact integer range."
        )
    )


class QuadraticTerm(InputModel):
    """A quadratic term: coefficient * variable1 * variable2."""

    variable1: str = Field(
        description=(
            "Name of the first variable in the product; must be declared in "
            "the problem's variables list."
        )
    )
    variable2: str = Field(
        description=(
            "Name of the second variable in the product. May equal variable1 "
            "only for an integer variable (a genuine square); for a binary "
            "variable x·x = x and is rejected."
        )
    )
    coefficient: Quantity = Field(
        description=(
            "Finite factor multiplying the product of the two variables."
        )
    )


class Objective(InputModel):
    """The business objective to minimize or maximize."""

    direction: Literal["minimize", "maximize"] = Field(
        description="Whether the objective value should be minimized or maximized."
    )
    linear_terms: list[LinearTerm] = Field(
        description=(
            "Terms of the form coefficient · variable. May be empty; "
            "duplicate terms are summed with a DUPLICATE_TERM_MERGED warning."
        )
    )
    quadratic_terms: list[QuadraticTerm] = Field(
        default_factory=list,
        description=(
            "Terms of the form coefficient · variable1 · variable2. Optional; "
            "omit for a purely linear objective."
        ),
    )
    constant: Quantity = Field(
        default=0,
        description=(
            "Finite constant added to every objective value. Does not affect "
            "ranking or the objective scale used to size penalties."
        ),
    )
