"""Capability models and their construction, shared by CLI and MCP (spec §22, §30).

Pure request/response formatting: capability flags come from each backend's
``SolverCapabilities``, availability from ``is_available()`` (no network I/O),
and limits from the ``ExecutionPolicy``. No optimization logic lives here.
Availability is asked through the same guard a solve uses
(``availability_refusal``), so one backend whose check raises or reports an
unknown category is only listed as unavailable itself.
Lives outside the ``mcp`` subpackage so the CLI ``capabilities`` command can
use the same source without the ``[mcp]`` extra installed.
"""

import json
from functools import cache
from typing import get_args

from pydantic import BaseModel, Field

from annealbridge.models import Constraint, OptimizationProblem, Variable
from annealbridge.orchestration import ExecutionPolicy
from annealbridge.orchestration.limits import availability_refusal, policy_gate_errors
from annealbridge.solvers import SolverRegistry
from annealbridge.version import package_version


class BackendCapability(BaseModel):
    """What one solver backend offers, and whether this server will use it.

    ``name`` is the *registry key*: the value to put in ``solver.backend``
    and the one ``enabled_backends`` is matched against (2026-09-09 review
    F-22). ``solver.backend`` is a closed schema that accepts only the eight
    built-in names, so a registry key must equal the backend's own
    ``capabilities.name``: a backend registered under any other key has no
    name a request could ask for.
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
    seed_min: int | None = Field(
        description=(
            "The smallest solver.seed this backend accepts, inclusive. A seed "
            "outside seed_min..seed_max is refused with "
            "INVALID_SOLVER_PREFERENCE before anything runs. Null when the "
            "backend declares no seed range."
        )
    )
    seed_max: int | None = Field(
        description=(
            "The largest solver.seed this backend accepts, inclusive. Null "
            "when the backend declares no seed range."
        )
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
    problem_json_schema: dict | None = Field(
        description=(
            "The full OptimizationProblem JSON Schema, identical to what "
            "annealbridge export-schema prints, when it was requested "
            "(include_schema on the MCP tool); null otherwise. Its field "
            "descriptions explain the document one level down."
        )
    )
    annealbridge_version: str | None = Field(
        default=None,
        description=(
            'The installed package version that produced this view; '
            '"unknown" outside an installed distribution.'
        ),
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
    registry: SolverRegistry, policy: ExecutionPolicy, *, include_schema: bool = True
) -> OptimizationCapabilities:
    """Assemble the capabilities response from the registry and policy.

    ``include_schema=False`` leaves ``problem_json_schema`` null and skips
    generating it: the schema is three quarters of the response, and an
    agent that only needs the backend list should not pay for it on every
    call. The default keeps the full view for the CLI and any direct caller.
    """
    backends: list[BackendCapability] = []
    for name in registry.names():
        backend = registry.get(name)
        caps = backend.capabilities
        # Guarded as in a solve: a check that raises or reports an unknown
        # category marks only this backend unavailable, with a redacted
        # reason, instead of failing the whole view.
        refusal = availability_refusal(backend)
        # The very policy gates a solve runs before its availability check,
        # judged by the registry key: a remote backend the policy refuses to
        # call is not "enabled", since reporting it as enabled would steer
        # agents into a guaranteed REMOTE_DISABLED.
        enabled = policy_gate_errors(name, caps, policy) is None
        backends.append(
            BackendCapability(
                # The registry key, not caps.name: it is what a request
                # names and what ``enabled`` was just judged by.
                name=name,
                available=refusal is None,
                enabled=enabled,
                unavailable_reason=None if refusal is None else refusal[2],
                remote=caps.remote,
                heuristic=caps.heuristic,
                exhaustive=caps.exhaustive,
                supports_seed=caps.supports_seed,
                seed_min=caps.seed_min,
                seed_max=caps.seed_max,
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
        problem_json_schema=_problem_json_schema() if include_schema else None,
        annealbridge_version=package_version(),
    )
