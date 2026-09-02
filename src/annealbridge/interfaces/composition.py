"""Composition root shared by the CLI and MCP interfaces (spec §21, §30).

Wires settings → policy → registry → service. Lives outside the ``mcp``
subpackage so the CLI can use it without the ``[mcp]`` extra installed.
Importing it has no side effects: the environment is read only when
``build_state``/``build_service`` is called.
"""

from dataclasses import dataclass

from annealbridge.config import ServerSettings
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import SolverRegistry


@dataclass(frozen=True)
class AppState:
    """The wired-up objects the interface layers operate on."""

    policy: ExecutionPolicy
    registry: SolverRegistry
    service: OptimizationService


def build_state_from_policy(
    policy: ExecutionPolicy, registry: SolverRegistry | None = None
) -> AppState:
    # The registry is kept alongside the service because the capabilities
    # view needs it directly; the service's copy is a private attribute.
    registry = registry if registry is not None else SolverRegistry.default()
    service = OptimizationService(registry=registry, policy=policy)
    return AppState(policy=policy, registry=registry, service=service)


def build_state(settings: ServerSettings | None = None) -> AppState:
    settings = settings if settings is not None else ServerSettings()
    return build_state_from_policy(settings.to_policy())


def build_service(settings: ServerSettings | None = None) -> OptimizationService:
    """Composition root: settings → policy → registry → service."""
    return build_state(settings).service
