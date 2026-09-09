"""Numeric field types for the IR (2026-09-09 review F-11).

pydantic's lax mode coerces ``True`` to ``1`` and ``"10"`` to ``10``. A
coefficient, a right-hand side, a weight or a read count is a *quantity*:
a boolean is a flag and a string is text, so either one is a mistake worth
rejecting at the schema instead of silently solving a different problem
(OVERVIEW §五 principles 1 and 5; 3b spec §7 made the same call for
``Variable`` bounds). The check runs *before* pydantic's own parsing, so
everything else keeps lax semantics: an integer is accepted for a float
field (JSON ``2`` → ``2.0``) and an integral float for an integer field
(``10.0`` → ``10``; ``10.5`` is still refused by pydantic).

Annotated types only — no ``model_config`` switch — so a model opts in per
field and the published JSON Schema is unchanged (``BeforeValidator`` has
no schema effect).
"""

from typing import Annotated

from pydantic import BeforeValidator, ValidationInfo

__all__ = ["Count", "Quantity"]


def _reject_flag_or_text(noun: str):
    def check(value: object, info: ValidationInfo) -> object:
        field = info.field_name or "value"
        if isinstance(value, bool):
            raise ValueError(f"{field} must be {noun}, not a boolean")
        if isinstance(value, str):
            raise ValueError(f"{field} must be {noun}, not a string")
        return value

    return check


# A real-valued quantity (coefficient, rhs, weight, time limit, ...).
Quantity = Annotated[float, BeforeValidator(_reject_flag_or_text("a number"))]

# An integer-valued quantity (read count, seed, bound, top_k, ...).
Count = Annotated[int, BeforeValidator(_reject_flag_or_text("an integer"))]
