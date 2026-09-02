"""MCP-facing capability models and their construction (spec §22).

The models and assembly logic live in ``annealbridge.interfaces.capabilities``
so the CLI ``capabilities`` command shares the same source without the
``[mcp]`` extra; this module re-exports them for the MCP tools.
"""

from annealbridge.interfaces.capabilities import (  # noqa: F401  (re-exports)
    BackendCapability,
    OptimizationCapabilities,
    build_capabilities,
)

__all__ = ["BackendCapability", "OptimizationCapabilities", "build_capabilities"]
