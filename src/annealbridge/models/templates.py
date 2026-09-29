"""Index sets, parameters, variable families and templates (schema 1.3).

Schema 1.3 spec 2026-09-25 §14 (v4). A ``"1.3"`` problem may describe
large structured models compactly: an *index set* is an ordered list of
elements, a *parameter* is a table of numbers keyed by elements, a
*variable family* declares one variable per combination of elements, and
every explicit list of the problem has a *template* sibling that repeats
one entry ``for_each`` combination of bound indices, filtered by ``where``
conditions (``variables`` ↔ ``variable_families``, ``linear_terms`` ↔
``linear_term_templates`` and so on).

These models are only the document shape. Templates are expanded into an
ordinary problem as the very first validation step
(``annealbridge.validation.expand_problem``); nothing downstream of that
step ever sees them, and every stage that reads a problem refuses one that
still carries them (``TemplatesNotExpandedError``). The reference and
condition strings follow a two-rule grammar (``validation/template_grammar``)
checked by the expander, not here: this module only rejects what is not
even the right JSON type, so the union fields never report a pydantic
error per union member.
"""

import math
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, StringConstraints, ValidationInfo

from annealbridge.models.quantities import Count, Quantity
from annealbridge.models.strict import InputModel

__all__ = [
    "CardinalityConstraintTemplate",
    "CardinalityMemberTemplate",
    "ConstraintTemplate",
    "IndexSet",
    "LinearTermTemplate",
    "Parameter",
    "ParameterValue",
    "QuadraticTermTemplate",
    "TEMPLATE_TEXT_LIMIT",
    "VariableFamily",
]

# Longest reference, for_each item or where condition (spec §14.5). Every
# such string is parsed once, so this bounds the parser's work per string.
TEMPLATE_TEXT_LIMIT = 256

# A count (cardinality rhs) is bounded like an integer variable's bounds
# (validation's INTEGER_BOUND_LIMIT; models/cardinality.py uses the same).
_COUNT_LIMIT = 2**31 - 1


def _field(info: ValidationInfo) -> str:
    return info.field_name or "value"


def _element(value: object, info: ValidationInfo) -> object:
    """An index set element or a parameter key element: a string or an int.

    Only the JSON type is checked here; the character set, the length, the
    range and the set's own rules are the expander's (INDEX_SET_INVALID /
    PARAMETER_TABLE_INVALID), so every element problem of a document is
    reported in one pass.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("index set elements must be strings or integers")
    return value


def _number_or_reference(value: object, info: ValidationInfo) -> object:
    """A literal number, or a parameter reference string (spec §14.5).

    Refuses, before pydantic tries the union's members, everything that
    would make *both* members fail -- a boolean, another JSON type, a
    non-finite number, an integer too large for a float -- so the error is
    reported once, at the field, never as ``coefficient.float`` plus
    ``coefficient.str``.
    """
    field = _field(info)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number or a parameter reference, not a boolean")
    if isinstance(value, int):
        try:
            float(value)
        except OverflowError:
            raise ValueError(f"{field} is too large to be a finite number") from None
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number")
        return value
    if isinstance(value, str):
        if len(value) > TEMPLATE_TEXT_LIMIT:
            raise ValueError(
                f"{field} must be at most {TEMPLATE_TEXT_LIMIT} characters"
            )
        return value
    raise ValueError(f"{field} must be a number or a parameter reference string")


def _optional_number_or_reference(value: object, info: ValidationInfo) -> object:
    """:func:`_number_or_reference` that lets ``null`` through (weight)."""
    return None if value is None else _number_or_reference(value, info)


def _count_or_reference(value: object, info: ValidationInfo) -> object:
    """A whole number within ±(2^31-1), or a parameter reference string.

    An integral float (``2.0``) is accepted as the integer, like ``Count``.
    Out-of-range integers are refused by comparison only, never formatted.
    """
    field = _field(info)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer or a parameter reference, not a boolean")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field} must be an integer or a parameter reference")
        value = int(value)
    if isinstance(value, int):
        if not -_COUNT_LIMIT <= value <= _COUNT_LIMIT:
            raise ValueError(f"{field} must lie within ±{_COUNT_LIMIT}")
        return value
    if isinstance(value, str):
        if len(value) > TEMPLATE_TEXT_LIMIT:
            raise ValueError(
                f"{field} must be at most {TEMPLATE_TEXT_LIMIT} characters"
            )
        return value
    raise ValueError(f"{field} must be an integer or a parameter reference string")


# A reference, for_each item or where condition.
TemplateText = Annotated[str, StringConstraints(max_length=TEMPLATE_TEXT_LIMIT)]
Element = Annotated[int | str, BeforeValidator(_element)]
NumberOrReference = Annotated[float | str, BeforeValidator(_number_or_reference)]
OptionalNumberOrReference = Annotated[
    float | str | None, BeforeValidator(_optional_number_or_reference)
]
CountOrReference = Annotated[int | str, BeforeValidator(_count_or_reference)]

_NAME = (
    "An identifier (letters, digits and underscores, not starting with a "
    "digit, at most 64 characters)"
)


def _for_each(scope: str):
    return Field(
        default_factory=list,
        description=(
            f'Indices to repeat {scope} over, outermost first, each "index in '
            'set" (at most eight). An inner scope may use the indices bound '
            "outside it but not rebind their names. Optional: without it "
            f"{scope} is generated once."
        ),
    )


def _where():
    return Field(
        default_factory=list,
        description=(
            "Conditions, all of which must hold for an entry to be generated "
            '(at most eight), each comparing two operands with ==, !=, <, <=, '
            '> or >=: two indices of one set (== and != compare the '
            "elements, the others their positions in the set), or numbers "
            "and parameter references such as dist[i,j]. There are no literal "
            "elements: compare a 0/1 parameter instead."
        ),
    )


class IndexSet(InputModel):
    """An ordered set of elements that indices range over (schema 1.3)."""

    name: str = Field(description=f"{_NAME}; must not be the word in.")
    elements: list[Element] = Field(
        description=(
            "The elements in order, each listed once, at least one: all "
            "strings (letters, digits, underscore, dot and hyphen, at most 64 "
            "characters) or all integers. They appear in generated names, "
            "e.g. x[a,0]."
        )
    )
    order: Literal["none", "linear", "cyclic"] = Field(
        default="none",
        description=(
            'Whether an index of this set may be shifted (i+1, i-2): "none" '
            '(the default) forbids shifts; "linear" allows them and an entry '
            "whose shifted index runs past either end is left out (a "
            'constraint entirely); "cyclic" wraps around. A shift moves by '
            "positions in elements, not by numeric value."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Free text for humans; never sent to a remote vendor.",
    )


class ParameterValue(InputModel):
    """One row of a parameter table."""

    key: list[Element] = Field(
        description=(
            "One element of each of the parameter's index sets, in the order "
            "of its indices."
        )
    )
    value: Quantity = Field(description="The finite number stored for this key.")


class Parameter(InputModel):
    """A table of numbers keyed by index set elements (schema 1.3)."""

    name: str = Field(description=f"{_NAME}.")
    indices: list[str] = Field(
        description=(
            "Names of the index sets the keys range over, one to eight; a "
            "reference must give one bound index per entry, e.g. dist[i,j]."
        )
    )
    values: list[ParameterValue] = Field(
        default_factory=list,
        description=(
            "The table as rows, each key at most once. Keys left out take "
            "default; without a default, using one is an error."
        ),
    )
    default: Quantity | None = Field(
        default=None,
        description=(
            "Finite value for every key that has no row. Beware: it also "
            "hides a row left out by mistake."
        ),
    )
    description: str | None = Field(
        default=None,
        description="Free text for humans; never sent to a remote vendor.",
    )


class VariableFamily(InputModel):
    """One variable per combination of index set elements (schema 1.3)."""

    name: str = Field(
        description=(
            f"{_NAME}; the generated variables are named name[e1,e2,...] with "
            'no spaces, e.g. x[a,0]. Must not start with "__" or equal an '
            "explicit variable's name."
        )
    )
    indices: list[str] = Field(
        description=(
            "Names of the index sets, one to eight; one variable is generated "
            "for every combination of their elements, the first set outermost. "
            "A generated variable no objective term or constraint uses is "
            "left out, with an UNUSED_TEMPLATE_VARIABLES warning."
        )
    )
    type: Literal["binary", "integer"] = Field(
        default="binary",
        description='As for a variable: "binary" (the default) or "integer".',
    )
    lower_bound: Count | None = Field(
        default=None,
        description="As for a variable: required for integer, absent for binary.",
    )
    upper_bound: Count | None = Field(
        default=None,
        description="As for a variable: required for integer, absent for binary.",
    )
    description: str | None = Field(
        default=None,
        description=(
            "Free text copied to every generated variable; never sent to a "
            "remote vendor."
        ),
    )


class LinearTermTemplate(InputModel):
    """coefficient · variable, repeated for each binding (schema 1.3)."""

    for_each: list[TemplateText] = _for_each("the term")
    where: list[TemplateText] = _where()
    coefficient: NumberOrReference = Field(
        description=(
            "A JSON number, or a parameter reference string such as "
            '"cost[i]"; a number written as a string is refused.'
        )
    )
    variable: TemplateText = Field(
        description=(
            "A family reference with one bound index per family index, "
            'e.g. "x[i,p+1]", or the name of an explicit variable.'
        )
    )


class QuadraticTermTemplate(InputModel):
    """coefficient · variable1 · variable2, repeated for each binding."""

    for_each: list[TemplateText] = _for_each("the term")
    where: list[TemplateText] = _where()
    coefficient: NumberOrReference = Field(
        description="A JSON number, or a parameter reference string."
    )
    variable1: TemplateText = Field(
        description="A family reference or the name of an explicit variable."
    )
    variable2: TemplateText = Field(
        description="A family reference or the name of an explicit variable."
    )


class ConstraintTemplate(InputModel):
    """A linear constraint repeated for each binding (schema 1.3)."""

    id: str = Field(
        description=(
            f"{_NAME}, unique among the templates and the explicit constraint "
            "ids. A generated constraint is named id[e1,...] by the for_each "
            "elements, or id itself without for_each."
        )
    )
    description: str | None = Field(
        default=None,
        description="Free text copied to every generated constraint.",
    )
    type: Literal["hard", "soft"] = Field(description="As for a constraint.")
    for_each: list[TemplateText] = _for_each("the constraint")
    where: list[TemplateText] = _where()
    terms: list[LinearTermTemplate] = Field(
        description=(
            "The left-hand side: each term template contributes its terms "
            "for each of its own bindings, which may use the constraint's."
        )
    )
    operator: Literal["==", "<=", ">="] = Field(description="As for a constraint.")
    rhs: NumberOrReference = Field(
        description="A JSON number, or a parameter reference string."
    )
    weight: OptionalNumberOrReference = Field(
        default=None,
        description=(
            "Soft only: a positive JSON number or a parameter reference; "
            "hard constraint templates carry none."
        ),
    )


def _member_shorthand(value: object) -> object:
    """A plain string member means ``{"variable": value}`` (spec §14.5)."""
    if isinstance(value, str):
        return {"variable": value}
    if isinstance(value, (dict, CardinalityMemberTemplate)):
        return value
    raise ValueError(
        "each variables entry must be a variable reference string or an object"
    )


class CardinalityMemberTemplate(InputModel):
    """The counted variables of a cardinality template, per binding."""

    for_each: list[TemplateText] = _for_each("the member")
    where: list[TemplateText] = _where()
    variable: TemplateText = Field(
        description=(
            "A reference to a binary family (e.g. \"x[i,p]\") or the name of "
            "an explicit binary variable."
        )
    )


class CardinalityConstraintTemplate(InputModel):
    """A cardinality constraint repeated for each binding (schema 1.3)."""

    id: str = Field(
        description=(
            f"{_NAME}, unique among the templates and the explicit constraint "
            "ids; generated constraints are named as for constraint_templates."
        )
    )
    description: str | None = Field(
        default=None,
        description="Free text copied to every generated constraint.",
    )
    type: Literal["hard", "soft"] = Field(description="As for a constraint.")
    for_each: list[TemplateText] = _for_each("the constraint")
    where: list[TemplateText] = _where()
    variables: list[
        Annotated[
            CardinalityMemberTemplate,
            BeforeValidator(
                _member_shorthand,
                json_schema_input_type=TemplateText | CardinalityMemberTemplate,
            ),
        ]
    ] = Field(
        description=(
            "The counted variables: a string is one variable reference, an "
            "object repeats its variable over its own for_each."
        )
    )
    operator: Literal["==", "<=", ">="] = Field(
        description="As for a cardinality constraint."
    )
    rhs: CountOrReference = Field(
        description=(
            "An integer, or a reference to a parameter whose values are all "
            "whole numbers."
        )
    )
    weight: OptionalNumberOrReference = Field(
        default=None,
        description="Soft only: a positive JSON number or a parameter reference.",
    )
