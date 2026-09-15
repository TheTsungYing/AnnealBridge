"""Typing primitives for reflecting over the IR's field annotations.

Two readers reflect over ``SolverPreferences.model_fields`` and must not
drift apart: the problem validator finds the backend option
blocks and their positive numeric options (3a §9.4), and the limit checks
in orchestration resolve a backend's dotted preference paths to a numeric
leaf (2026-09-09 review, service-layer follow-ups). ``validation`` may not
import ``orchestration``, so the shared pieces live here, in ``models``,
below both.

These are primitives, not the callers' rules. Each caller composes its own
rule from them — an *optional* model versus a model *anywhere* in the
annotation, a single optional number versus a union of numbers — because
the two rules answer different questions and were never meant to agree.
The one judgement made here is what counts as a number (``int`` or
``float``, never ``bool``), which both rules share.
"""

import types
import typing
from typing import Annotated

from pydantic import BaseModel

__all__ = ["is_number_type", "model_class", "strip_annotated", "union_members"]


def union_members(annotation: object) -> tuple[object, ...] | None:
    """The members of a union annotation with ``None`` dropped, else None.

    Both spellings count — ``types.UnionType`` (``X | None``) and
    ``typing.Union`` / ``Optional[X]``. A bare type, a ``Literal``, a
    ``list[...]`` or an ``Annotated[...]`` is not a union and yields None.
    """
    origin = typing.get_origin(annotation)
    if origin is not types.UnionType and origin is not typing.Union:
        return None
    return tuple(arg for arg in typing.get_args(annotation) if arg is not type(None))


def strip_annotated(annotation: object) -> object:
    """The underlying ``T`` of an ``Annotated[T, ...]``, else the annotation.

    The IR's numeric fields are ``Annotated`` aliases (``Quantity`` /
    ``Count`` from models/quantities.py, 2026-09-09 review F-11); the
    validators attached to them are pydantic's business, not the
    reflection's. Only the outer wrapper is stripped: a union or a
    ``list[...]`` inside stays as it is.
    """
    while typing.get_origin(annotation) is Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def model_class(annotation: object) -> type[BaseModel] | None:
    """``annotation`` itself when it is a BaseModel subclass, else None.

    Exactly the class: ``Model | None``, ``list[Model]`` and
    ``Annotated[Model, ...]`` are generic aliases, not classes, and yield
    None. Callers that want to look inside take the annotation apart with
    the other primitives first.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def is_number_type(annotation: object) -> bool:
    """True when ``annotation`` is exactly ``int`` or ``float``.

    ``Annotated`` wrappers are transparent, so ``Count`` and ``Quantity``
    count. ``bool`` does not, although it subclasses ``int``: a flag is
    not a quantity, and neither a positivity check nor a limit applies to
    one. A union is not a number either — a caller decides what to do with
    its members.
    """
    inner = strip_annotated(annotation)
    return inner is int or inner is float
