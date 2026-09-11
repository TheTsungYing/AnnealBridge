"""The ``annealbridge-mcp`` stdio entry point, driven as a real subprocess.

This is the only test allowed to spawn a subprocess (spec §26). It launches
the server via ``python -m annealbridge.interfaces.mcp.server`` — equivalent
to the ``annealbridge-mcp`` console script but immune to PATH differences on
Windows and in CI — and asserts the four tools are served over stdio.
"""

import os
import subprocess
import sys

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
