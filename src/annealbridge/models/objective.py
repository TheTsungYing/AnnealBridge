"""Objective function models."""

from typing import Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer

from annealbridge.exceptions import TemplatesNotExpandedError
from annealbridge.models.quantities import Quantity
from annealbridge.models.strict import InputModel
from annealbridge.models.templates import LinearTermTemplate, QuadraticTermTemplate

# The objective's two template lists (schema 1.3 spec §14.1), in the order
# they are reported; named with the ``objective.`` prefix everywhere.
_TEMPLATE_FIELDS = ("linear_term_templates", "quadratic_term_templates")


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
    # Schema 1.3 spec §14.5. Optional and empty by default, and left out of
    # every dump while empty, so an older document parses and dumps exactly
    # as before.
    linear_term_templates: list[LinearTermTemplate] = Field(
        default_factory=list,
        description=(
            "Version 1.3 or later: linear terms repeated over index sets. "
            "Optional; may be empty."
        ),
    )
    quadratic_term_templates: list[QuadraticTermTemplate] = Field(
        default_factory=list,
        description=(
            "Version 1.3 or later: quadratic terms repeated over index sets. "
            "Optional; may be empty."
        ),
    )

    @model_serializer(mode="wrap")
    def _omit_empty_templates(self, handler: SerializerFunctionWrapHandler):
        """Leave the empty template lists out of every dump (spec §14.13)."""
        data = handler(self)
        if isinstance(data, dict):
            for name in _TEMPLATE_FIELDS:
                if not getattr(self, name):
                    data.pop(name, None)
        return data

    def template_fields(self) -> list[str]:
        """The non-empty template lists, as ``objective.<name>``."""
        return [f"objective.{name}" for name in _TEMPLATE_FIELDS if getattr(self, name)]

    def has_templates(self) -> bool:
        """Whether any objective template is still unexpanded."""
        return bool(self.linear_term_templates or self.quadratic_term_templates)

    def require_expanded(self, operation: str) -> None:
        """Raise :class:`TemplatesNotExpandedError` if templates remain.

        Every function that evaluates or compiles an objective calls this
        first: an unexpanded template would silently contribute nothing.
        """
        if self.has_templates():
            raise TemplatesNotExpandedError(
                f"{operation} needs an expanded objective, but "
                f"{' and '.join(self.template_fields())} still hold templates; "
                "expand the problem with annealbridge.validation.expand_problem "
                "first"
            )
