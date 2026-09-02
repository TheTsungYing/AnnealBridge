"""The ``annealbridge-mcp`` stdio entry point, driven as a real subprocess.

This is the only test allowed to spawn a subprocess (spec §26). It launches
the server via ``python -m annealbridge.interfaces.mcp.server`` — equivalent
to the ``annealbridge-mcp`` console script but immune to PATH differences on
Windows and in CI — and asserts the three tools are served over stdio.
"""

import os
import subprocess
import sys

import pytest
from mcp import Client, StdioServerParameters

pytestmark = pytest.mark.anyio

EXPECTED_TOOLS = [
    "get_optimization_capabilities",
    "solve_optimization",
    "validate_optimization_problem",
]


async def test_stdio_entrypoint_serves_the_three_tools():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "annealbridge.interfaces.mcp.server"],
    )
    async with Client(params) as client:
        tools = (await client.list_tools()).tools

    assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS


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
