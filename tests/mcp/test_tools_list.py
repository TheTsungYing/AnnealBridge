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


def _properties_without_description(schema: dict) -> list[str]:
    """``Model.field`` for every schema property missing a description."""
    missing: list[str] = []

    def check(model_name: str, definition: dict) -> None:
        for field, prop in definition.get("properties", {}).items():
            description = prop.get("description")
            if not isinstance(description, str) or not description.strip():
                missing.append(f"{model_name}.{field}")

    check(schema.get("title", "<root>"), schema)
    for name, definition in schema.get("$defs", {}).items():
        check(name, definition)
    return missing


@pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
async def test_every_output_schema_property_has_a_description(tool_name):
    """An agent reading a result must not have to guess what a field means.

    The outputSchema is derived from the tool's return type, so this is the
    interface-level guard behind the per-model checks in
    tests/unit/test_models.py — and the only one that covers
    ``OptimizationCapabilities``, which lives in ``interfaces``.
    """
    tools = await _list_tools()
    tool = next(tool for tool in tools if tool.name == tool_name)

    missing = _properties_without_description(tool.output_schema)

    assert not missing, f"{tool_name}: properties without a description: " + ", ".join(
        sorted(missing)
    )


async def test_solve_docstring_points_at_recommend_backend():
    tools = await _list_tools()
    solve = next(tool for tool in tools if tool.name == "solve_optimization")
    assert "recommend_backend" in solve.description
    recommend = next(tool for tool in tools if tool.name == "recommend_backend")
    assert "Advisory only" in recommend.description
