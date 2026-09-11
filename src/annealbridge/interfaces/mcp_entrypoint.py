"""Console-script entry point for ``annealbridge-mcp``.

The core install ships this script even though the MCP server itself lives
behind the ``[mcp]`` extra (2026-09-11 install verification, gap 3). This
module imports nothing from ``mcp`` at module level, so a core-only install
gets one line on stderr and exit code 2 instead of a traceback. Any other
``ModuleNotFoundError`` is re-raised unchanged.

It deliberately lives outside ``annealbridge.interfaces.mcp``: that package's
``__init__`` imports the server eagerly, so a module inside it would trigger
the very ``mcp`` import this script exists to report on.
"""

import sys

MISSING_EXTRA_MESSAGE = (
    'Error: the MCP server needs the optional "mcp" dependency; '
    'install it with: pip install "annealbridge[mcp]"'
)


def _is_missing_mcp(exc: ModuleNotFoundError) -> bool:
    """True only for the third-party ``mcp`` package, not for a module of ours
    that happens to sit under ``annealbridge.interfaces.mcp``."""
    name = exc.name or ""
    return name == "mcp" or name.startswith("mcp.")


def main() -> None:
    try:
        from annealbridge.interfaces.mcp.server import main as run
    except ModuleNotFoundError as exc:
        if _is_missing_mcp(exc):
            print(MISSING_EXTRA_MESSAGE, file=sys.stderr)
            sys.exit(2)
        raise
    run()
