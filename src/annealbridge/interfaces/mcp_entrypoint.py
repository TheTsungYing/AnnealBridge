"""Console-script entry point for ``annealbridge-mcp``.

The core install ships this script even though the MCP server itself lives
behind the ``[mcp]`` extra (2026-09-11 install verification, gap 3). This
module imports nothing from ``mcp`` at module level, so a core-only install
gets one line on stderr and exit code 2 instead of a traceback. Any other
``ModuleNotFoundError`` is re-raised unchanged.

It deliberately lives outside ``annealbridge.interfaces.mcp``: that package's
``__init__`` imports the server eagerly, so a module inside it would trigger
the very ``mcp`` import this script exists to report on.

``run`` is the shared body: the ``annealbridge mcp`` subcommand calls it too,
so both entry points report a missing extra identically.
"""

import sys
from collections.abc import Sequence

MISSING_EXTRA_MESSAGE = (
    'Error: the MCP server needs the optional "mcp" dependency; '
    'install it with: pip install "annealbridge[mcp]"'
)


def _is_missing_mcp(exc: ModuleNotFoundError) -> bool:
    """True only for the third-party ``mcp`` package, not for a module of ours
    that happens to sit under ``annealbridge.interfaces.mcp``."""
    name = exc.name or ""
    return name == "mcp" or name.startswith("mcp.")


def run(argv: Sequence[str] | None = None, prog: str = "annealbridge-mcp") -> None:
    """Start the MCP server, or report the missing ``[mcp]`` extra and exit 2.

    ``argv`` and ``prog`` are passed through to the server's own parser so the
    ``annealbridge mcp`` subcommand shares this guard; their defaults are the
    console script's behaviour.
    """
    try:
        from annealbridge.interfaces.mcp.server import main as server_main
    except ModuleNotFoundError as exc:
        if _is_missing_mcp(exc):
            print(MISSING_EXTRA_MESSAGE, file=sys.stderr)
            sys.exit(2)
        raise
    server_main(argv, prog=prog)


def main() -> None:
    run()
