"""Console-script entry point for ``annealbridge-mcp``.

The core install ships this script even though the MCP server itself lives
behind the ``[mcp]`` extra (2026-09-11 install verification, gap 3). This
module imports nothing from ``mcp`` at module level, so a core-only install
gets one line on stderr and exit code 2 instead of a traceback.

An installed but incompatible ``mcp`` release (an older one lacking a name or
submodule the server imports) is reported the same way, with an upgrade hint
instead of the install hint (2026-11-18 review R-17). Any other import failure,
including one from a module of ours under ``annealbridge.interfaces.mcp`` or
one that names no module, is re-raised unchanged.

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


INCOMPATIBLE_MCP_MESSAGE = (
    'Error: the installed "mcp" package is not compatible with this version '
    'of annealbridge; upgrade it with: pip install --upgrade "annealbridge[mcp]"'
)


def _is_third_party_mcp(name: str | None) -> bool:
    """True only for the third-party ``mcp`` package or one of its
    submodules, not for a module of ours that happens to sit under
    ``annealbridge.interfaces.mcp``, nor for an error naming no module."""
    if name is None:
        return False
    return name == "mcp" or name.startswith("mcp.")


def _mcp_import_problem(exc: ImportError) -> str | None:
    """The message to report for an import failure caused by the ``mcp``
    package, or ``None`` when the failure is someone else's.

    Only ``mcp`` itself missing means the extra is not installed. A missing
    ``mcp`` submodule, or a name ``mcp`` does not provide, means an
    installed release the server cannot use.
    """
    if not _is_third_party_mcp(exc.name):
        return None
    if isinstance(exc, ModuleNotFoundError) and exc.name == "mcp":
        return MISSING_EXTRA_MESSAGE
    return INCOMPATIBLE_MCP_MESSAGE


def run(argv: Sequence[str] | None = None, prog: str = "annealbridge-mcp") -> None:
    """Start the MCP server, or report an unusable ``mcp`` package and exit 2.

    A missing ``[mcp]`` extra gets the install hint, an incompatible ``mcp``
    release the upgrade hint. ``argv`` and ``prog`` are passed through to
    the server's own parser so the ``annealbridge mcp`` subcommand shares
    this guard; their defaults are the console script's behaviour.
    """
    try:
        from annealbridge.interfaces.mcp.server import main as server_main
    except ImportError as exc:
        message = _mcp_import_problem(exc)
        if message is None:
            raise
        print(message, file=sys.stderr)
        sys.exit(2)
    server_main(argv, prog=prog)


def main() -> None:
    run()
