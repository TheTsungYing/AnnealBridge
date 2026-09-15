"""The ``annealbridge-mcp`` stdio entry point, driven as a real subprocess.

This is the only test allowed to spawn a subprocess (spec §26). It covers both
ways the server is started: ``python -m annealbridge.interfaces.mcp.server``,
and the ``annealbridge-mcp`` console script the installed distribution
generates. The latter is the regression test for the 2026-09-11 install
verification gaps 3 and 4 — the script now enters through
``annealbridge.interfaces.mcp_entrypoint``, and nothing else may notice.
Both must serve the same four tools over stdio.
"""

import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

pytestmark = pytest.mark.anyio

EXPECTED_TOOLS = [
    "get_optimization_capabilities",
    "recommend_backend",
    "solve_optimization",
    "validate_optimization_problem",
]


async def test_stdio_entrypoint_serves_the_four_tools():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "annealbridge.interfaces.mcp.server"],
    )
    async with Client(params) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS


def _console_script() -> str:
    """The ``annealbridge-mcp`` script generated next to this interpreter."""
    script = shutil.which("annealbridge-mcp", path=str(Path(sys.executable).parent))
    if script is None:
        pytest.fail(
            "the annealbridge-mcp console script is missing next to "
            f"{sys.executable}; reinstall the project with "
            'pip install -e ".[all,dev]"'
        )
    return script


async def test_console_script_serves_the_four_tools():
    """2026-09-11 install verification (gaps 3 and 4): the generated
    ``annealbridge-mcp`` script — not ``python -m`` — must start the real
    server through the entry-point shim and serve the same four tools."""
    script = _console_script()

    params = StdioServerParameters(command=script, args=[])
    async with Client(params) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS


HOST_REASON = "host must not be empty, blank or contain whitespace"
PORT_REASON = "port must be between 1 and 65535"


@pytest.mark.parametrize(
    ("arguments", "expected_argument", "expected_reason"),
    [
        (["--transport", "streamable-http", "--host", ""], "--host", HOST_REASON),
        (
            ["--transport", "streamable-http", "--host", "127.0.0.1 "],
            "--host",
            HOST_REASON,
        ),
        (["--transport", "streamable-http", "--port", "65536"], "--port", PORT_REASON),
        (["--transport", "streamable-http", "--port", "0"], "--port", PORT_REASON),
        (["--port", "-1"], "--port", PORT_REASON),
        (["--port", "not-a-port"], "--port", "port must be an integer"),
    ],
)
def test_invalid_bind_argument_exits_2_before_binding_a_socket(
    arguments, expected_argument, expected_reason
):
    """2026-09-11 review (F01 / F09): ``--host`` / ``--port`` are validated
    by argparse, so a bad value ends the process with exit code 2 and one line
    — no traceback, and no socket ever opened (the run would otherwise block
    until killed, so returning at all is the proof)."""
    completed = subprocess.run(
        [sys.executable, "-m", "annealbridge.interfaces.mcp.server", *arguments],
        env={**os.environ},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 2
    assert f"argument {expected_argument}: {expected_reason}" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""


def test_invalid_settings_exit_2_with_a_message_and_no_traceback():
    env = {**os.environ, "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES": "0"}

    completed = subprocess.run(
        [sys.executable, "-m", "annealbridge.interfaces.mcp.server"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 2
    assert "Error: Invalid server settings" in completed.stderr
    assert "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize(
    "extra_env",
    [
        {},
        {
            "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES": "0",
            "ANNEALBRIDGE_NOT_A_REAL_SETTING": "1",
        },
        {"ANNEALBRIDGE_NOT_A_REAL_SETTING": "1"},
    ],
    ids=["valid-settings", "invalid-settings", "unknown-variable"],
)
def test_version_exits_0_before_the_settings_are_read(extra_env):
    """2026-09-15 consolidation: ``--version`` is answered before
    ``load_settings()``, so an invalid value cannot block it and an unknown
    variable is not even warned about — stdout carries the version, and
    stderr carries neither a settings error nor the unknown-variable warning.

    Had the settings been read, ``invalid-settings`` would end with
    ``Error: Invalid server settings`` and exit 2 (the invalid value is
    refused before unknown variables are checked), and ``unknown-variable``
    would log a ``WARNING`` record naming the variable. stderr is not
    required to be empty: an unrelated interpreter warning on a developer
    machine (``PYTHONWARNINGS``, say) may legitimately write to it.

    Driven through the console script: under ``python -m`` the interpreter
    itself writes runpy's "found in sys.modules" RuntimeWarning to stderr
    before the server code runs, which is noise this test has no use for."""
    completed = subprocess.run(
        [_console_script(), "--version"],
        env={**os.environ, **extra_env},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0
    assert completed.stdout == f"annealbridge {metadata.version('annealbridge')}\n"
    # The two traces load_settings() leaves: the unknown-variable WARNING
    # log record and the "Error: Invalid server settings" line.
    assert "WARNING" not in completed.stderr
    assert "Error:" not in completed.stderr
    assert "Traceback" not in completed.stderr
