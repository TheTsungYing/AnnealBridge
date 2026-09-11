"""MCP server instance and process-wide state (spec §21).

This module owns the ``MCPServer`` instance. The composition root itself
(``build_service`` and friends) lives in ``annealbridge.interfaces.composition``
so the CLI shares it without the ``[mcp]`` extra; it is re-exported here.
Importing this module has no side effects: the environment is read only when
the first tool call (or an explicit ``build_service``) needs it.
"""

import argparse
import json
import logging
import sys
from mcp.server import MCPServer

from annealbridge.config import (
    SettingsError,
    load_settings,
    validate_http_host,
    validate_http_port,
)
from annealbridge.interfaces.composition import (  # noqa: F401  (re-exports)
    AppState,
    build_service,
    build_state,
    build_state_from_policy,
)
from annealbridge.version import package_version

# The smallest complete problem, shown to the host at initialize so an agent
# learns the document shape before its first call instead of from its first
# error. Kept as data so a test can prove it parses against the schema.
EXAMPLE_PROBLEM: dict = {
    "version": "1.0",
    "name": "knapsack",
    "variables": [
        {"name": "item_a", "type": "binary"},
        {"name": "item_b", "type": "binary"},
        {"name": "item_c", "type": "binary"},
    ],
    "objective": {
        "direction": "maximize",
        "linear_terms": [
            {"variable": "item_a", "coefficient": 10},
            {"variable": "item_b", "coefficient": 8},
            {"variable": "item_c", "coefficient": 7},
        ],
    },
    "constraints": [
        {
            "id": "capacity",
            "type": "hard",
            "terms": [
                {"variable": "item_a", "coefficient": 6},
                {"variable": "item_b", "coefficient": 5},
                {"variable": "item_c", "coefficient": 4},
            ],
            "operator": "<=",
            "rhs": 10,
        }
    ],
    "solver": {"backend": "exact"},
}

# Server-level guidance the host shows the agent once, at initialize. It
# complements the per-tool descriptions (which say *when* to call a tool)
# with what only the whole server can say: the call order, the rules a
# first document most often breaks, and one complete example. Like every
# recommended_action it names no configuration values and no limits — those
# come from get_optimization_capabilities.
SERVER_INSTRUCTIONS = (
    """AnnealBridge solves combinatorial optimization problems that you write
as a structured JSON document (binary or bounded-integer variables, a linear
or quadratic objective, hard and soft linear constraints). Every result is
re-validated against the original document; nothing is ever silently
substituted, clamped or dropped.

Call order:
1. get_optimization_capabilities - once, first. It returns the problem JSON
   schema (problem_json_schema), the accepted schema versions and, per
   backend, whether it is available and enabled and which limits apply.
2. validate_optimization_problem - after writing the document, before
   spending anything. It returns every semantic error at once, each with a
   recommended_action, plus advisory warnings and the compiled size estimate.
3. recommend_backend - when the choice of backend is not obvious. Advisory
   only; you still write the backend into solver.backend.
4. solve_optimization - last. Its warnings are the same ones validate gives
   for that backend, followed by any raised during the run: read them before
   trusting an answer that looks weaker than expected.

Rules a first document most often breaks:
- Only the fields in problem_json_schema exist. A field the schema does not
  declare (at any level) is rejected as a tool error naming its path; it is
  never ignored, so an invented field can never silently change the problem.
- An integer variable needs "type": "integer" with both lower_bound and
  upper_bound, and the document must then carry "version": "1.1".
- Inequality constraints (<=, >=) need integer coefficients and an integer
  rhs. Soft constraints need a positive weight in objective-value units;
  hard constraints must not carry one.
- Leave solver.penalty_multiplier at its default: hard-constraint penalties
  are managed by the server.

A minimal complete problem (maximize value under a weight limit):
"""
    + json.dumps(EXAMPLE_PROBLEM, indent=2)
    + "\n"
)


def _package_version() -> str:
    """The installed distribution's version, or ``"unknown"`` outside one."""
    return package_version()


mcp = MCPServer(
    "AnnealBridge",
    instructions=SERVER_INSTRUCTIONS,
    version=_package_version(),
)


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = build_state()
    return _state


def reset_state(state: AppState | None = None) -> None:
    """Replace (or clear) the process-wide state; used by tests to inject policy."""
    global _state
    _state = state


def _host_argument(value: str) -> str:
    """argparse ``type`` for ``--host``: the environment's rule, verbatim.

    Raising :class:`argparse.ArgumentTypeError` keeps argparse's own reason
    (which would only say "invalid value") from replacing ours, and ends the
    process with exit code 2 and one line instead of a traceback.
    """
    try:
        return validate_http_host(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _port_argument(value: str) -> int:
    """argparse ``type`` for ``--port``: an integer in 1–65535, 0 included in
    the refusal (an ephemeral port no host can be pointed at)."""
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "port must be an integer between 1 and 65535"
        ) from None
    try:
        return validate_http_port(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def main() -> None:
    """Entry point for ``annealbridge-mcp`` (spec §25).

    stdio is the default transport. Streamable HTTP binds 127.0.0.1:8000
    unless overridden; exposing it beyond localhost requires an explicit
    ``--host`` — an empty one is refused, never read as "every interface".
    Host and port obey the same rule whether they come from the environment
    or from an override. Tool registration happens at import time via the
    tools module; this function only parses arguments and runs the transport.
    Invalid ``ANNEALBRIDGE_*`` settings are reported on stderr and end the
    process with exit code 2 instead of a traceback.

    The ``annealbridge-mcp`` console script does not point here directly: it
    enters through ``annealbridge.interfaces.mcp_entrypoint``, which imports
    this module lazily so a core-only install (no ``[mcp]`` extra) reports the
    missing dependency instead of raising ``ModuleNotFoundError``. Running
    ``python -m annealbridge.interfaces.mcp.server`` still reaches this
    function unchanged.
    """
    try:
        settings = load_settings()
    except SettingsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    parser = argparse.ArgumentParser(
        prog="annealbridge-mcp",
        description="Run the AnnealBridge MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="Transport protocol (default: stdio).",
    )
    parser.add_argument(
        "--host",
        type=_host_argument,
        default=settings.http_host,
        help=(
            "Bind address for streamable-http (non-empty, no whitespace; "
            f"default: {settings.http_host}; ignored for stdio). Binding "
            "beyond localhost must be requested explicitly; an empty value "
            "is refused rather than treated as every interface."
        ),
    )
    parser.add_argument(
        "--port",
        type=_port_argument,
        default=settings.http_port,
        help=(
            f"Port for streamable-http (1-65535; default: {settings.http_port}; "
            "ignored for stdio)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The settings were validated above; wire the state from them now so
    # the first tool call cannot hit a configuration error.
    reset_state(build_state(settings))
    # Ensure the tools are registered on `mcp` before serving.
    import annealbridge.interfaces.mcp.tools  # noqa: F401

    if args.transport == "streamable-http":
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    # Under ``python -m`` this file runs as the ``__main__`` module, which is a
    # *different* module object from ``annealbridge.interfaces.mcp.server`` —
    # with its own ``mcp`` instance that the tools never register on. Delegate
    # to the canonical module so exactly one server instance exists. Loading
    # this file *by path* (``mcp dev .../server.py``) creates the same toolless
    # duplicate and cannot be delegated away, so point the Inspector at the
    # package's ``__init__.py`` instead.
    from annealbridge.interfaces.mcp.server import main as _canonical_main

    _canonical_main()
