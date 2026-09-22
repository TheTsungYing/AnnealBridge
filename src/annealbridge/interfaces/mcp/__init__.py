"""MCP interface layer: server instance, composition root, and tool / prompt /
resource registration."""

# The server must be imported before the tools. On a core-only install this is
# the import that fails on the missing ``mcp`` package, which
# ``annealbridge.interfaces.mcp_entrypoint`` reports as exit code 2; importing
# ``tools`` first would fail on ``anyio`` instead and print a traceback.
from annealbridge.interfaces.mcp.server import build_service, mcp

# isort: split
from annealbridge.interfaces.mcp import (  # noqa: F401  (registers them on ``mcp``)
    prompts,
    resources,
    tools,
)

__all__ = ["build_service", "mcp"]
