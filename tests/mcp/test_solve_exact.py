"""The ``solve_optimization`` MCP tool on the exact backend, via a real client.

The knapsack optimum itself is pinned by tests/scenarios/test_knapsack.py; the
point here is that the same answer survives the MCP round trip, that the SDK
hands back structured content (a dict, not a JSON string) and that slack
variables stay inside the compiler.
"""

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp

# tests/mcp has no __init__.py, so pytest puts this directory on sys.path and
# the sibling conftest is importable as a top-level module.
from conftest import load_example

pytestmark = pytest.mark.anyio

KNAPSACK_OPTIMUM_VALUE = 17.0


async def test_solve_knapsack_on_exact_backend():
    problem = load_example("knapsack.json", backend="exact")

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        content = result.structured_content

    # The SDK must give us a parsed object, not a JSON string the caller would
    # have to decode itself.
    assert isinstance(content, dict)

    assert content["status"] == "success"

    best = content["solutions"][0]
    assert best["objective_value"] == pytest.approx(KNAPSACK_OPTIMUM_VALUE)

    # Slack bits are named with a "__" prefix and must never reach the caller.
    for solution in content["solutions"]:
        assert all(not name.startswith("__") for name in solution["variables"])
