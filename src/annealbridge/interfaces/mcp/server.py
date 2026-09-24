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
from collections.abc import Sequence

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from annealbridge.config import (
    SettingsError,
    load_settings,
    validate_http_host,
    validate_http_port,
)
from annealbridge.interfaces.composition import (  # noqa: F401  (re-exports build_service, build_state_from_policy)
    AppState,
    build_service,
    build_state,
    build_state_from_policy,
    exit_on_settings_error,
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
# with what only the whole server can say: the everyday requests the server
# is for, when each tool is worth a call — a local backend can be solved
# directly, a remote one or a large problem is validated first — how to turn
# a recommendation into the one backend the request carries, including when
# to put that choice to the user instead of deciding alone, the rules a
# first document most often breaks, and one complete example. Like every
# recommended_action it names no configuration values and no limits — those
# come from get_optimization_capabilities, and the reason codes it points at
# are categorical, so nothing here goes stale when a setting changes.
SERVER_INSTRUCTIONS = (
    """AnnealBridge solves combinatorial optimization problems that you write
as a structured JSON document (binary or bounded-integer variables, a linear
or quadratic objective, hard and soft linear constraints). Use it when the
user asks, in whatever words, which items to take within a budget, weight
or capacity; how to assign people or jobs to seats, shifts or machines; in
which order to visit a handful of places; or how to split things into
groups or pick a subset that meets several requirements at once - anything
that can be written as yes/no or bounded-count decisions with a linear or
quadratic score and linear rules, even when the user never says
"optimization". Every result is re-validated against the original
document; nothing is ever silently substituted, clamped or dropped.

When to call each tool:
- get_optimization_capabilities - when you need to know which backends are
  available and enabled and which limits apply, or the accepted schema
  versions. The full problem JSON schema (problem_json_schema) is only
  included when you pass include_schema: true; ask for it when the
  document needs more than the example below shows. For a small binary
  problem on a local backend the example already shows the whole document
  shape.
- validate_optimization_problem - before solving on a remote backend, or
  when the problem is large (many variables, wide integer ranges): it
  returns every semantic error at once, each with a recommended_action,
  plus advisory warnings and the compiled size estimate, before anything
  is spent. On a local backend you may solve directly: an invalid document
  comes back as status "invalid_problem" carrying the same errors and
  recommended_action, so fix the document and solve again.
- recommend_backend - whenever the user did not name a backend. Advisory
  only; you still write the backend into solver.backend.
- solve_optimization - last. Its warnings are the same ones validate gives
  for that backend, followed by any raised during the run: read them before
  trusting an answer that looks weaker than expected.
Local backends are free and make no network request; remote ones spend
vendor quota and need credentials, which is why they get the extra call.

Three prompts (pick_subset, assign, schedule_shifts) walk through turning a
request of one of those shapes into a document, each ending in a complete
example.
The resources under annealbridge://examples/ (knapsack, integer_knapsack,
assignment, tsp) are complete example documents, and annealbridge://schema
is the full problem JSON schema.

Choosing solver.backend:
- If the user named a backend, use it; it is never substituted.
- Otherwise call recommend_backend and read the reasons: an exhaustive
  backend that fits proves optimality; among local heuristics,
  R_DENSE_STRENGTH marks one that reaches the same energy faster on this
  problem's shape and R_PENALTY_WEAKNESS one with a lower hit rate on it.
  Local backends are free; remote ones spend quota and need credentials.
- When more than one local backend is usable and the user did not ask for
  an answer without being consulted, present the top entries with one-line
  reasons and ask which to run. Otherwise run the first usable entry and
  say in the answer which backend ran and why.

Rules a first document most often breaks:
- Only the fields the problem JSON schema declares exist
  (get_optimization_capabilities with include_schema: true returns it as
  problem_json_schema). A field the schema does not declare (at any level)
  is rejected as UNKNOWN_FIELD naming its path, in an invalid_problem
  (valid: false) result like any other error; it is never ignored, so an
  invented field can never silently change the problem.
- An integer variable needs "type": "integer" with both lower_bound and
  upper_bound, and the document must then carry "version": "1.1".
- Inequality constraints (<=, >=) need integer coefficients and an integer
  rhs. Soft constraints need a positive weight in objective-value units;
  hard constraints must not carry one.
- Leave solver.penalty_multiplier at its default: hard-constraint penalties
  are managed by the server.
- solver.postprocess is off ("none") by default. "repair_local_search"
  repairs and locally improves the best few samples on this server's CPU,
  which often helps a heuristic backend whose samples are infeasible or far
  from optimal, at some extra solve time; every such solution is
  re-validated and marked by its source field.

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
    """The process-wide state, built from the environment on first use.

    ``main()`` builds it before serving, so this lazy path is only taken
    when the server is used without ``main()`` (``mcp dev``, an embedding
    host, a test). It runs inside a tool call, where ending the process is
    not an option: invalid settings are raised as a ``ToolError`` carrying
    the same operator-formatted message ``main()`` prints (it never echoes a
    value), so the client reads the reason instead of a generic crash and
    the server logs one line instead of a traceback. The state stays unset,
    so the next call tries again.
    """
    global _state
    if _state is None:
        try:
            _state = build_state()
        except SettingsError as exc:
            raise ToolError(str(exc)) from None
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


def _add_version_argument(parser: argparse.ArgumentParser) -> None:
    """``--version``: print ``annealbridge <version>`` and exit 0, with the
    same text and help as the CLI's ``--version``."""
    parser.add_argument(
        "--version",
        action="version",
        version=f"annealbridge {package_version()}",
        help="Show the version and exit.",
    )


def main(argv: Sequence[str] | None = None, prog: str = "annealbridge-mcp") -> None:
    """Entry point for ``annealbridge-mcp`` (spec §25).

    stdio is the default transport. Streamable HTTP binds 127.0.0.1:8000
    unless overridden; exposing it beyond localhost requires an explicit
    ``--host`` — an empty one is refused, never read as "every interface".
    Host and port obey the same rule whether they come from the environment
    or from an override. Tool registration happens at import time via the
    tools module; this function only parses arguments and runs the transport.
    Invalid ``ANNEALBRIDGE_*`` settings — refused when read, or when the
    service is wired from them — are reported on stderr and end the process
    with exit code 2 instead of a traceback. Logs go to stderr only, in the
    ``%(asctime)s %(levelname)s %(name)s: %(message)s`` format.

    ``--version`` is answered first, before the settings are read: it prints
    ``annealbridge <version>`` and exits 0 even when those settings are
    invalid, and logs no unknown-variable warning.

    The ``annealbridge-mcp`` console script does not point here directly: it
    enters through ``annealbridge.interfaces.mcp_entrypoint``, which imports
    this module lazily so a core-only install (no ``[mcp]`` extra) reports the
    missing dependency instead of raising ``ModuleNotFoundError``. Running
    ``python -m annealbridge.interfaces.mcp.server`` still reaches this
    function unchanged.

    ``argv`` and ``prog`` exist for the ``annealbridge mcp`` subcommand, which
    reaches this function through the same missing-extra guard with the
    arguments Typer did not consume. They default to ``sys.argv[1:]`` and
    ``annealbridge-mcp``, so the console script's behaviour is unchanged.
    """
    # Only --version is recognised here; everything else, --help included,
    # is left to the full parser, which needs the settings for its defaults.
    version_parser = argparse.ArgumentParser(prog=prog, add_help=False)
    _add_version_argument(version_parser)
    version_parser.parse_known_args(argv)
    # Logging is configured before the settings are read, so even the
    # unknown-variable WARNING they may log comes out in this format.
    # ``force=True`` is required: constructing ``MCPServer`` at import time
    # already ran the SDK's own ``basicConfig`` (a rich handler), and a plain
    # ``basicConfig`` does nothing once the root logger has a handler. The
    # replaced handler carried no filter — redaction happens before a message
    # is logged, in the code that logs it — so nothing is lost with it. The
    # one handler writes to stderr: under stdio, stdout is the protocol
    # channel and must carry nothing else.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    try:
        settings = load_settings()
    except SettingsError as exc:
        exit_on_settings_error(exc)
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the AnnealBridge MCP server.",
    )
    # Already answered above; declared again so --help lists it.
    _add_version_argument(parser)
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
    args = parser.parse_args(argv)

    # The settings were validated above; wire the state from them now so
    # the first tool call cannot hit a configuration error. Wiring can still
    # refuse them (a backend declaring a limit the policy has no value for),
    # and that ends the process the same way as an invalid setting.
    try:
        state = build_state(settings)
    except SettingsError as exc:
        exit_on_settings_error(exc)
    reset_state(state)
    # Ensure the tools, prompts and resources are registered on `mcp`
    # before serving (the package import registers all three).
    import annealbridge.interfaces.mcp  # noqa: F401

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
