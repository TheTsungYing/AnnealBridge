"""Constraint model."""

from typing import Literal

from pydantic import Field

from annealbridge.models.objective import LinearTerm
from annealbridge.models.quantities import Quantity
from annealbridge.models.strict import InputModel


class Constraint(InputModel):
    """A linear constraint over the problem's variables (binary or bounded
    integer, IR 1.1).

    Hard constraints must be satisfied; soft constraints carry a weight
    expressing business preference importance.

    ``rhs`` / ``weight`` are ``Quantity`` (models/quantities.py): a boolean or
    a string is refused instead of being coerced (2026-09-09 review F-11).
    """

    id: str = Field(
        description=(
            "Constraint identifier, unique within the problem. Results and "
            "violation reports are traced back to it."
        )
    )
    description: str | None = Field(
        default=None,
        description=(
            "Free text explaining what this constraint expresses. For humans "
            "only; never sent to a remote vendor."
        ),
    )
    type: Literal["hard", "soft"] = Field(
        description=(
            '"hard" must be satisfied by every returned solution; "soft" is a '
            "weighted preference that may be violated at a cost."
        )
    )
    terms: list[LinearTerm] = Field(
        description=(
            "The left-hand side, as a sum of coefficient · variable terms. "
            "Must be non-empty; there is no quadratic constraint form."
        )
    )
    operator: Literal["==", "<=", ">="] = Field(
        description=(
            "How the left-hand side is compared against rhs. Inequalities "
            "require integral coefficients and rhs."
        )
    )
    rhs: Quantity = Field(
        description=(
            "The right-hand side of the comparison. Must be an integral value "
            "for a <= / >= constraint."
        )
    )
    weight: Quantity | None = Field(
        default=None,
        description=(
            "Soft constraints only: the cost of violating the constraint, in "
            "objective-value units (penalty = weight × violation²). Must be "
            "positive, and hard constraints must not carry a weight."
        ),
    )
