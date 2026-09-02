"""Capability models and their construction, shared by CLI and MCP (spec §22, §30).

Pure request/response formatting: capability flags come from each backend's
``SolverCapabilities``, availability from ``is_available()`` (no network I/O),
and limits from the ``ExecutionPolicy``. No optimization logic lives here.
Lives outside the ``mcp`` subpackage so the CLI ``capabilities`` command can
use the same source without the ``[mcp]`` extra installed.
"""

from functools import cache
from typing import get_args

from pydantic import BaseModel

from annealbridge.models import Constraint, OptimizationProblem, Variable
from annealbridge.orchestration import ExecutionPolicy
from annealbridge.solvers import SolverRegistry


class BackendCapability(BaseModel):
    """What one solver backend offers, and whether this server will use it."""

    name: str
    available: bool
    enabled: bool
    unavailable_reason: str | None
    remote: bool
    heuristic: bool
    exhaustive: bool
    supports_seed: bool
    returns_multiple_samples: bool
    limits: dict[str, float | int]
    description: str


class OptimizationCapabilities(BaseModel):
    """Everything an agent needs before formulating and submitting a problem."""

    schema_version: str
    supported_variable_types: list[str]
    supported_constraint_operators: list[str]
    supported_objective_terms: list[str]
    inequality_requires_integer_coefficients: bool
    backends: list[BackendCapability]
    problem_json_schema: dict


def _policy_limits(name: str, policy: ExecutionPolicy) -> dict[str, float | int]:
    if name == "exact":
        return {"max_variables": policy.exact_max_variables}
    if name == "dwave_qpu":
        return {
            "max_reads": policy.max_qpu_reads,
            "max_annealing_time_us": policy.max_qpu_annealing_time_us,
        }
    if name == "leap_hybrid_bqm":
        return {"max_time_seconds": policy.max_remote_time_seconds}
    return {}


@cache
def _problem_json_schema() -> dict:
    """The problem JSON schema, generated once per process.

    ``model_json_schema()`` walks the whole model tree (~3 ms) and the
    result is a pure function of the model classes, so it is safe to cache
    for the process lifetime. Availability is deliberately *not* cached:
    credentials can change between calls.
    """
    return OptimizationProblem.model_json_schema()


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
                name=caps.name,
                available=status.available,
                enabled=enabled,
                unavailable_reason=status.detail if not status.available else None,
                remote=caps.remote,
                heuristic=caps.heuristic,
                exhaustive=caps.exhaustive,
                supports_seed=caps.supports_seed,
                returns_multiple_samples=caps.returns_multiple_samples,
                limits=_policy_limits(name, policy),
                description=caps.description,
            )
        )
    return OptimizationCapabilities(
        schema_version=get_args(
            OptimizationProblem.model_fields["version"].annotation
        )[0],
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
