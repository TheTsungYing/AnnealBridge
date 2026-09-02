"""MCP interface layer: server instance, composition root and tool registration."""

from annealbridge.interfaces.mcp.server import build_service, mcp
from annealbridge.interfaces.mcp import tools  # noqa: F401  (registers the tools)

__all__ = ["build_service", "mcp"]
