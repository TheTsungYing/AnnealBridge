"""tools/list over the in-memory MCP client (Phase 2 §26, 3a §26.4)."""

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp

pytestmark = pytest.mark.anyio

EXPECTED_TOOLS = [
    "get_optimization_capabilities",
    "recommend_backend",
    "solve_optimization",
    "validate_optimization_problem",
]


async def _list_tools():
    async with Client(mcp) as client:
        return (await client.list_tools()).tools


async def test_exactly_four_tools():
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


async def test_recommend_output_schema_matches_recommendation_result():
    tools = await _list_tools()
    tool = next(tool for tool in tools if tool.name == "recommend_backend")

    assert set(tool.input_schema["required"]) == {"problem"}
    output = tool.output_schema
    assert output["title"] == "BackendRecommendationResult"
    assert {"valid", "errors", "recommendations", "advisory"} <= set(
        output["properties"]
    )
    entry = output["$defs"]["BackendRecommendation"]
    assert {
        "rank",
        "backend",
        "usable",
        "model_type",
        "reasons",
        "blocking",
        "warnings",
        "estimated_compiled_variables",
    } <= set(entry["properties"])


async def test_solve_docstring_points_at_recommend_backend():
    tools = await _list_tools()
    solve = next(tool for tool in tools if tool.name == "solve_optimization")
    assert "recommend_backend" in solve.description
    recommend = next(tool for tool in tools if tool.name == "recommend_backend")
    assert "Advisory only" in recommend.description
