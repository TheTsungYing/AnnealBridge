"""Cardinality constraint model (schema 1.2, schema 1.2 spec 2026-09-25 §4-§5).

A cardinality constraint counts how many of a list of *binary* variables
are chosen (take the value 1) and compares that count with ``rhs``:
one-hot is ``"==", 1``, at-most-k ``"<=", k``, at-least-k ``">=", k``. It is
declared in its own top-level list, ``OptimizationProblem.cardinality_constraints``,
so the ``Constraint`` model and every error path of a ``"1.0"`` / ``"1.1"``
document stay exactly as they were (spec §4.4).

Downstream of the validator nothing knows about this shape: every reader
goes through ``OptimizationProblem.all_constraints()``, which lowers each
declaration to a :class:`LoweredCardinalityConstraint` -- an ordinary linear
``Constraint`` whose coefficients are all 1. The subclass is only a marker:
the BQM path encodes a *hard* at-most-one (``"<=", 1`` over two or more
variables) as the pairwise penalty ``λ·Σ x_i x_j`` instead of a slack
variable (``validation.estimates.uses_pairwise_penalty``), and everything
else compiles, estimates and re-validates exactly like the equivalent
linear constraint.

The lowering is rebuilt on every call and never cached: a cached view
survives ``model_copy(update=...)`` and attribute assignment, and a stale
view would make re-validation check a constraint other than the declared
one (spec §5.1, overview principle 2).
"""

from typing import Literal

from pydantic import Field

from annealbridge.models.constraint import Constraint
from annealbridge.models.objective import LinearTerm
from annealbridge.models.quantities import Count, Quantity
from annealbridge.models.strict import InputModel

__all__ = ["CardinalityConstraint", "LoweredCardinalityConstraint"]

# The count is bounded like an integer variable's bounds (validation's
# INTEGER_BOUND_LIMIT): a larger count means nothing, and an unbounded JSON
# integer would overflow the float the lowered rhs is (spec §4.2).
_RHS_LIMIT = 2**31 - 1


class LoweredCardinalityConstraint(Constraint):
    """The linear view of a :class:`CardinalityConstraint`.

    Every coefficient is ``1.0`` and ``rhs`` is a float, as for any parsed
    ``Constraint``. Built only by :meth:`CardinalityConstraint.lowered`,
    never parsed from JSON; the class itself is the marker the BQM
    encoding decision reads.
    """


class CardinalityConstraint(InputModel):
    """Chosen-count constraint over binary variables (schema version 1.2)."""

    id: str = Field(
        description=(
            "Constraint identifier, unique across both constraints and "
            "cardinality_constraints. Results and violation reports are "
            "traced back to it."
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
    variables: list[str] = Field(
        description=(
            "Names of the binary variables being counted; each one that takes "
            "the value 1 counts once. Must be non-empty, list each name at "
            "most once and name binary variables only: a weighted sum, or a "
            "sum over integer variables, is a linear constraint in "
            "constraints."
        )
    )
    operator: Literal["==", "<=", ">="] = Field(
        description=(
            "How the number of chosen variables is compared against rhs: "
            '"==" exactly, "<=" at most, ">=" at least. A hard "<=" with rhs '
            "1 (at most one) needs no slack variable on a BQM backend."
        )
    )
    rhs: Count = Field(
        ge=-_RHS_LIMIT,
        le=_RHS_LIMIT,
        description=(
            "The count the number of chosen variables is compared against, "
            'an integer (for example 1 for "exactly one" or "at most one").'
        ),
    )
    weight: Quantity | None = Field(
        default=None,
        description=(
            "Soft constraints only: the cost of violating the constraint, in "
            "objective-value units (penalty = weight × violation², the "
            "violation being how many variables the count is off by). Must "
            "be positive, and hard constraints must not carry a weight."
        ),
    )

    def lowered(self) -> LoweredCardinalityConstraint:
        """This declaration as a linear constraint with every coefficient 1.

        A new object on every call (no cache, spec §5.1). Never raises for
        an input that passed the schema: ``rhs`` is bounded, so ``float``
        is exact, and nothing else is converted.
        """
        return LoweredCardinalityConstraint.model_construct(
            id=self.id,
            description=self.description,
            type=self.type,
            terms=[
                LinearTerm.model_construct(variable=name, coefficient=1.0)
                for name in self.variables
            ],
            operator=self.operator,
            rhs=float(self.rhs),
            weight=self.weight,
        )
