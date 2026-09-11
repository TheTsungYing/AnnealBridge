"""Capability models and their construction, shared by CLI and MCP (spec §22, §30).

Pure request/response formatting: capability flags come from each backend's
``SolverCapabilities``, availability from ``is_available()`` (no network I/O),
and limits from the ``ExecutionPolicy``. No optimization logic lives here.
Lives outside the ``mcp`` subpackage so the CLI ``capabilities`` command can
use the same source without the ``[mcp]`` extra installed.
"""

import json
from functools import cache
from typing import get_args

from pydantic import BaseModel, Field

from annealbridge.models import Constraint, OptimizationProblem, Variable
from annealbridge.orchestration import ExecutionPolicy
from annealbridge.solvers import SolverRegistry


class BackendCapability(BaseModel):
    """What one solver backend offers, and whether this server will use it.

    ``name`` is the *registry key*: the value to put in ``solver.backend``
    and the one ``enabled_backends`` is matched against (2026-09-09 review
    F-22). The six built-in backends register under their own
    ``capabilities.name``, so for them the two coincide; a custom registry
    may register a backend under another key, and the key is the only
    name a request can use.
    """

    name: str = Field(
        description=(
            "The registry key: the value to put in solver.backend, and the "
            "one ANNEALBRIDGE_ENABLED_BACKENDS is matched against."
        )
    )
    available: bool = Field(
        description=(
            "Whether the backend can run right now — dependency installed, "
            "credentials present, configuration valid."
        )
    )
    enabled: bool = Field(
        description=(
            "Whether server policy permits it: in the enabled-backends list, "
            "and remote execution allowed if it is remote. Independent of "
            "available; both must be true for a solve to reach it."
        )
    )
    unavailable_reason: str | None = Field(
        description=(
            "Categorical detail when available is false, e.g. 'dwave-system "
            "not installed'. Never contains configuration values; null when "
            "the backend is available."
        )
    )
    remote: bool = Field(description="Whether running it leaves this machine.")
    heuristic: bool = Field(
        description="Whether it may return a sub-optimal answer."
    )
    exhaustive: bool = Field(
        description=(
            "Whether it enumerates every assignment and can therefore prove "
            "optimality or infeasibility."
        )
    )
    supports_seed: bool = Field(
        description="Whether solver.seed has any effect on this backend."
    )
    returns_multiple_samples: bool = Field(
        description=(
            "False means the effective solver.top_k on this backend is at "
            "most 1."
        )
    )
    limits: dict[str, float | int] = Field(
        description=(
            "The server-side policy ceilings that apply to this backend. A "
            "request beyond one of them is refused, never clamped."
        )
    )
    description: str = Field(description="One-line description of the backend.")


class OptimizationCapabilities(BaseModel):
    """Everything an agent needs before formulating and submitting a problem."""

    schema_version: str = Field(
        description=(
            "The newest problem schema version this server accepts; use it "
            "unless an older one is needed."
        )
    )
    schema_versions: list[str] = Field(
        description=(
            "Every accepted problem schema version. Derived from the model, "
            'not hard-coded; "1.1" is a superset of "1.0".'
        )
    )
    supported_variable_types: list[str] = Field(
        description="The variable types a problem may declare."
    )
    supported_constraint_operators: list[str] = Field(
        description="The operators a constraint may use."
    )
    supported_objective_terms: list[str] = Field(
        description="The kinds of objective term a problem may carry."
    )
    inequality_requires_integer_coefficients: bool = Field(
        description=(
            "Whether <= / >= constraints need integral coefficients and rhs, "
            "which the slack encoding requires."
        )
    )
    backends: list[BackendCapability] = Field(
        description="Every registered backend, with its flags and limits."
    )
    problem_json_schema: dict = Field(
        description=(
            "The full OptimizationProblem JSON Schema, identical to what "
            "annealbridge export-schema prints. Its field descriptions "
            "explain the document one level down."
        )
    )


@cache
def _problem_json_schema_json() -> str:
    """The problem JSON schema as text, generated once per process.

    ``model_json_schema()`` walks the whole model tree (~3 ms) and the
    result is a pure function of the model classes, so it is safe to cache
    for the process lifetime. Availability is deliberately *not* cached:
    credentials can change between calls. The cache holds the *serialised*
    form (2026-09-09 review F-22): a cached dict is a shared mutable
    object, and pydantic only shallow-copies the top level, so one
    caller's edit to a nested entry would show up in every later response.
    """
    return json.dumps(OptimizationProblem.model_json_schema())


def _problem_json_schema() -> dict:
    """A fresh, independent copy of the problem JSON schema for one response."""
    return json.loads(_problem_json_schema_json())


def build_capabilities(
    registry: SolverRegistry, policy: ExecutionPolicy
) -> OptimizationCapabilities:
    """Assemble the capabilities response from the registry and policy."""
    backends: list[BackendCapability] = []
    for name in registry.names():
        backend = registry.get(name)
        caps = backend.capabilities
        status = backend.is_available()
        in_set = policy.enabled_backends is None or name in policy.enabled_backends
        # A remote backend the policy refuses to call is not "enabled": reporting
        # it as enabled would steer agents into a guaranteed REMOTE_DISABLED.
        enabled = in_set and (not caps.remote or policy.allow_remote)
        backends.append(
            BackendCapability(
                # The registry key, not caps.name: it is what a request
                # names and what ``enabled`` was just judged by.
                name=name,
                available=status.available,
                enabled=enabled,
                unavailable_reason=status.detail if not status.available else None,
                remote=caps.remote,
                heuristic=caps.heuristic,
                exhaustive=caps.exhaustive,
                supports_seed=caps.supports_seed,
                returns_multiple_samples=caps.returns_multiple_samples,
                limits=policy.limits_for(caps),
                description=caps.description,
            )
        )
    schema_versions = list(
        get_args(OptimizationProblem.model_fields["version"].annotation)
    )
    return OptimizationCapabilities(
        schema_version=schema_versions[-1],
        schema_versions=schema_versions,
        supported_variable_types=list(
            get_args(Variable.model_fields["type"].annotation)
        ),
        supported_constraint_operators=list(
            get_args(Constraint.model_fields["operator"].annotation)
        ),
        supported_objective_terms=["linear", "quadratic"],
        inequality_requires_integer_coefficients=True,
        backends=backends,
        problem_json_schema=_problem_json_schema(),
    )
