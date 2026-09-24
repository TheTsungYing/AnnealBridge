"""Parse a submitted problem document into the model, shared by CLI and MCP.

A document that does not fit the problem schema — a field the schema does
not declare, a missing required field, a value of the wrong type — never
becomes an :class:`OptimizationProblem`, so the service and its validator
never see it. This module turns each such schema error into a catalog
:class:`SolveError` (``UNKNOWN_FIELD``, ``MISSING_FIELD``,
``INVALID_FIELD_VALUE``) with the same path notation the validator uses
(``constraints[0].terms[1].coefficient``), all of them at once, and builds
the failure results the MCP tools return for them, shaped exactly like the
service's own results for a semantic error.

The messages carry pydantic's short description only: never the submitted
value (``input_value``) and never a documentation URL. The one piece of the
document a path can carry is the name of an undeclared field — reporting it
is the point of ``UNKNOWN_FIELD`` — and such a name is written in its JSON
spelling whenever it is not a plain identifier, so it can neither look like
further path segments nor break a log or console line.

Pure request/response formatting, no optimization logic; lives outside the
``mcp`` subpackage so the CLI can use it without the ``[mcp]`` extra.
"""

import json
import re
import time
from typing import Any

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from annealbridge.models import OptimizationProblem, SolveError, SolveResult
from annealbridge.models.error_catalog import catalog_error
from annealbridge.validation import (
    BackendRecommendationResult,
    ProblemValidationResult,
)
from annealbridge.version import package_version

__all__ = [
    "invalid_recommendation_result",
    "invalid_solve_result",
    "invalid_validation_result",
    "parse_problem",
]

# pydantic prefixes the message of a ValueError raised in a field validator.
_VALUE_ERROR_PREFIX = "Value error, "

# A key written as ``.name`` in a path; any other key is written ``["..."]``.
# Every field the schema declares matches, so only an undeclared field name
# (the one caller-chosen part of a path) can take the bracketed form.
_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_problem(data: Any) -> OptimizationProblem | list[SolveError]:
    """Validate ``data`` into an :class:`OptimizationProblem`.

    Returns the model, or — when the document does not fit the schema —
    one :class:`SolveError` per schema error, in pydantic's order, never
    empty. Validation is exactly ``OptimizationProblem.model_validate``, so
    a document that parses here parses the same everywhere.
    """
    try:
        return OptimizationProblem.model_validate(data)
    except ValidationError as exc:
        return [_schema_error(detail) for detail in exc.errors()]


def _schema_error(detail: ErrorDetails) -> SolveError:
    """One pydantic error detail as a catalog error, without its input value."""
    path = _format_path(detail["loc"])
    kind = detail["type"]
    if kind == "extra_forbidden":
        # The input models refuse unknown keys (models/strict.py): say so in
        # the problem's own vocabulary instead of pydantic's generic wording.
        return catalog_error(
            "UNKNOWN_FIELD", "unknown field, not in the problem schema", path=path
        )
    if kind == "missing":
        return catalog_error("MISSING_FIELD", detail["msg"], path=path)
    if path is None and kind == "model_type":
        # The document itself is not an object (a string, a list, null).
        return catalog_error(
            "INVALID_FIELD_VALUE", "the problem must be a JSON object"
        )
    message = detail["msg"].removeprefix(_VALUE_ERROR_PREFIX)
    return catalog_error("INVALID_FIELD_VALUE", message, path=path)


def _format_path(loc: tuple[int | str, ...]) -> str | None:
    """``("constraints", 0, "terms")`` → ``constraints[0].terms``; ``()`` → None.

    A key that is not a plain identifier — an undeclared field such as
    ``""``, ``"a.b"`` or one containing a newline — is written as its JSON
    string in brackets (``solver[""]``, ``["a.b"]``), control characters
    escaped.
    """
    if not loc:
        return None
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        elif _PLAIN_KEY.fullmatch(part):
            path += f".{part}" if path else part
        else:
            path += f"[{json.dumps(part, ensure_ascii=False)}]"
    return path


def invalid_solve_result(errors: list[SolveError], started: float) -> SolveResult:
    """The ``invalid_problem`` result for a document that failed the schema.

    Same shape as the service's result for a semantic error: no backend, no
    direction, no solutions or attempts, no warnings, the first error's
    message, the wall clock since ``started`` (a ``time.perf_counter()``
    reading) and the package version.
    """
    return SolveResult(
        status="invalid_problem",
        backend=None,
        objective_direction=None,
        solutions=[],
        attempts=[],
        errors=errors,
        message=errors[0].message,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        annealbridge_version=package_version(),
    )


def invalid_validation_result(errors: list[SolveError]) -> ProblemValidationResult:
    """``valid: false`` with ``errors`` and nothing estimated, as for a semantic error."""
    return ProblemValidationResult(valid=False, errors=errors)


def invalid_recommendation_result(
    errors: list[SolveError],
) -> BackendRecommendationResult:
    """``valid: false`` with ``errors`` and no ranking, as for a semantic error."""
    return BackendRecommendationResult(valid=False, errors=errors, recommendations=[])
