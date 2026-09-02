"""tools/list over the in-memory MCP client (spec §26)."""

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp

pytestmark = pytest.mark.anyio

EXPECTED_TOOLS = [
    "get_optimization_capabilities",
    "solve_optimization",
    "validate_optimization_problem",
]


async def _list_tools():
    async with Client(mcp) as client:
        return (await client.list_tools()).tools


async def test_exactly_three_tools():
    tools = await _list_tools()
    assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS


async def test_solve_input_schema_wraps_problem():
    tools = await _list_tools()
    solve = next(tool for tool in tools if tool.name == "solve_optimization")
    schema = solve.input_schema

    assert set(schema["required"]) == {"problem"}
    problem_ref = schema["properties"]["problem"]["$ref"]
    problem_schema = schema["$defs"][problem_ref.rsplit("/", 1)[-1]]
    assert {"variables", "objective", "constraints", "solver"} <= set(
        problem_schema["properties"]
    )


async def test_solve_output_schema_matches_solve_result():
    tools = await _list_tools()
    solve = next(tool for tool in tools if tool.name == "solve_optimization")
    output = solve.output_schema

    assert output["title"] == "SolveResult"
    assert {"status", "backend", "solutions", "attempts", "errors", "warnings"} <= set(
        output["properties"]
    )
