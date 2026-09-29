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

The lowering is cached per declaration, keyed by the declaration's field
values (schema 1.3 spec §14.14, which replaces the §5.1 rule "never
cached"): a template can generate tens of thousands of small cardinality
constraints, and rebuilding every lowering on each of the dozen
``all_constraints()`` calls of a solve cost nine tenths of its time. The key
is recomputed from the current field values on every call, so
``model_copy(update=...)``, attribute assignment and an in-place edit of
``variables`` all miss the cache and rebuild: a stale view would make
re-validation check a constraint other than the declared one (overview
principle 2).
"""

from operator import attrgetter
from typing import Literal, get_origin

from pydantic import ConfigDict, Field

from annealbridge.models.constraint import Constraint
from annealbridge.models.objective import LinearTerm
from annealbridge.models.quantities import Count, Quantity
from annealbridge.models.strict import InputModel

__all__ = ["CardinalityConstraint", "LoweredCardinalityConstraint"]

# The count is bounded like an integer variable's bounds (validation's
# INTEGER_BOUND_LIMIT): a larger count means nothing, and an unbounded JSON
# integer would overflow the float the lowered rhs is (spec §4.2).
_RHS_LIMIT = 2**31 - 1


# Where a declaration keeps its lowering: a non-field key of the instance
# ``__dict__``, the mechanism ``functools.cached_property`` uses, so it
# never takes part in equality, dumps or the JSON schema.
_LOWERED_CACHE = "_lowered_cache"


class LoweredCardinalityConstraint(Constraint):
    """The linear view of a :class:`CardinalityConstraint`.

    Every coefficient is ``1.0`` and ``rhs`` is a float, as for any parsed
    ``Constraint``. Built only by :meth:`CardinalityConstraint.lowered`,
    never parsed from JSON; the class itself is the marker the BQM
    encoding decision reads. Frozen, because one object is shared by every
    ``all_constraints()`` call while its declaration is unchanged; its
    ``terms`` list is read-only by contract.
    """

    model_config = ConfigDict(frozen=True)


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

    def _lowering_key(self) -> tuple:
        """Every field's current value, lists as tuples (spec §14.14).

        Read through :data:`_KEY_FIELDS`, which is ``model_fields`` itself,
        so a field added later is part of the key without anyone
        remembering to add it. Called on every ``all_constraints()``, hence
        one C-level ``attrgetter`` instead of a loop over the fields.
        """
        values = list(_KEY_GETTER(self))
        for position in _KEY_LIST_POSITIONS:
            values[position] = tuple(values[position])
        return tuple(values)

    def lowered(self) -> LoweredCardinalityConstraint:
        """This declaration as a linear constraint with every coefficient 1.

        The same object while every field keeps its value, a new one as
        soon as any field differs from when it was built (see the module
        docstring). Never raises for an input that passed the schema:
        ``rhs`` is bounded, so ``float`` is exact, and nothing else is
        converted.
        """
        key = self._lowering_key()
        cached = self.__dict__.get(_LOWERED_CACHE)
        if cached is not None and cached[0] == key:
            return cached[1]
        lowered = self._build_lowered()
        self.__dict__[_LOWERED_CACHE] = (key, lowered)
        return lowered

    def _build_lowered(self) -> LoweredCardinalityConstraint:
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


# The lowering cache key (spec §14.14): every field, in declaration order,
# the list-typed ones (``variables``) turned into tuples.
_KEY_FIELDS: tuple[str, ...] = tuple(CardinalityConstraint.model_fields)
_KEY_GETTER = attrgetter(*_KEY_FIELDS)
_KEY_LIST_POSITIONS: tuple[int, ...] = tuple(
    position
    for position, name in enumerate(_KEY_FIELDS)
    if get_origin(CardinalityConstraint.model_fields[name].annotation) is list
)
