"""Base class for the models that make up the public problem JSON.

Every model an external caller writes — the problem, its variables,
objective terms, constraints and solver preferences — refuses keys it does
not declare (``extra="forbid"``). pydantic's default is to drop unknown keys
silently, and for this project's callers that is the worst possible
outcome: the caller is usually an LLM, and inventing a plausible field name
(``"variable3"`` on a quadratic term, an ``"objective.cubic_terms"`` block,
``"solver.num_restarts"``) is its characteristic mistake. Dropping the key
does not produce an error; it produces a *different problem* that passes
every guarantee — hard constraints satisfied, independently re-validated,
ranked — and answers a question nobody asked. That violates the overview's
principle 5 and the README's "no silent decisions" as badly as clamping a
parameter would.

A forbidden key surfaces on the type layer, like a boolean in a numeric
field (2026-09-09 review F-11): both interfaces report it as ``UNKNOWN_FIELD``
naming the path (``interfaces/problem_input.py``) — the CLI exits 2, the MCP
tools return ``invalid_problem`` / ``valid: false`` — and the published JSON Schema
carries ``additionalProperties: false`` so a schema-aware host refuses the
document before it is sent. Output models (solutions, results, capabilities)
are built by our own code and keep pydantic's default.
"""

from pydantic import BaseModel, ConfigDict

__all__ = ["InputModel"]


class InputModel(BaseModel):
    """A caller-authored model: unknown keys are a validation error."""

    model_config = ConfigDict(extra="forbid")
