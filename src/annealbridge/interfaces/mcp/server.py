"""MCP server instance and process-wide state (spec §21).

This module owns the ``MCPServer`` instance. The composition root itself
(``build_service`` and friends) lives in ``annealbridge.interfaces.composition``
so the CLI shares it without the ``[mcp]`` extra; it is re-exported here.
Importing this module has no side effects: the environment is read only when
the first tool call (or an explicit ``build_service``) needs it.
"""

import argparse
import logging
import sys

from mcp.server import MCPServer

from annealbridge.config import ServerSettings
from annealbridge.interfaces.composition import (  # noqa: F401  (re-exports)
    AppState,
    build_service,
    build_state,
    build_state_from_policy,
)

mcp = MCPServer("AnnealBridge")


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


def main() -> None:
    """Entry point for ``annealbridge-mcp`` (spec §25).

    stdio is the default transport. Streamable HTTP binds 127.0.0.1:8000
    unless overridden; exposing it beyond localhost requires an explicit
    ``--host``. Tool registration happens at import time via the tools
    module; this function only parses arguments and runs the transport.
    """
    settings = ServerSettings()
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
        default=settings.http_host,
        help=(
            "Bind address for streamable-http (default: "
            f"{settings.http_host}; ignored for stdio). Binding beyond "
            "localhost must be requested explicitly."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.http_port,
        help=(
            f"Port for streamable-http (default: {settings.http_port}; "
            "ignored for stdio)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
    # to the canonical module so exactly one server instance exists.
    from annealbridge.interfaces.mcp.server import main as _canonical_main

    _canonical_main()
