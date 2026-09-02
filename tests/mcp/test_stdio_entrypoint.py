"""The ``annealbridge-mcp`` stdio entry point, driven as a real subprocess.

This is the only test allowed to spawn a subprocess (spec §26). It launches
the server via ``python -m annealbridge.interfaces.mcp.server`` — equivalent
to the ``annealbridge-mcp`` console script but immune to PATH differences on
Windows and in CI — and asserts the three tools are served over stdio.
"""

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
