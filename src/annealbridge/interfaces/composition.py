"""Composition root shared by the CLI and MCP interfaces (spec §21, §30).

Wires settings → policy → registry → service. Lives outside the ``mcp``
subpackage so the CLI can use it without the ``[mcp]`` extra installed.
Importing it has no side effects: the environment is read only when
``build_state``/``build_service`` is called.
"""

import sys
from dataclasses import dataclass
from typing import NoReturn

from annealbridge.config import ServerSettings, SettingsError, load_settings
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
    try:
        service = OptimizationService(registry=registry, policy=policy)
    except ValueError as exc:
        # A backend declares a limit key the policy has no value for (spec
        # §11.3): a configuration error, reported like any other bad setting.
        raise SettingsError(f"Invalid server settings: {exc}") from None
    return AppState(policy=policy, registry=registry, service=service)


def exit_on_settings_error(exc: SettingsError) -> NoReturn:
    """Report invalid settings on stderr and end the process with exit code 2.

    The one place both entry points turn a :class:`SettingsError` into a
    process exit, so the CLI and the MCP server print the same
    ``Error: ...`` line and return the same exit code. The message is
    already operator-formatted and never echoes a value (review F-20). Only
    the standard library is used, so the MCP server does not depend on
    Typer through it and the CLI does not depend on the ``[mcp]`` extra.
    """
    print(f"Error: {exc}", file=sys.stderr)
    sys.exit(2)


def build_state(settings: ServerSettings | None = None) -> AppState:
    settings = settings if settings is not None else load_settings()
    # ``sa_workers`` is the one setting that is not policy: it tunes a
    # backend's speed, so it goes to the registry, not the service.
    registry = SolverRegistry.default(sa_workers=settings.sa_workers)
    return build_state_from_policy(settings.to_policy(), registry)


def build_service(settings: ServerSettings | None = None) -> OptimizationService:
    """Composition root: settings → policy → registry → service.

    The Phase 2 §21 name, kept as the library-facing entry point. The CLI
    and the MCP server call :func:`build_state` instead because they also
    need the registry for the capabilities view, so this wrapper has no
    production caller inside the package (2026-09-09 review F-18).
    """
    return build_state(settings).service
