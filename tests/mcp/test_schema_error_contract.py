"""Contract guards for schema errors over MCP (roadmap batch 5, part B).

A problem document that does not fit the schema is parsed inside the tool
(``interfaces/problem_input.py``) instead of by the SDK, so it comes back in
the same shape as a semantic error. Three things must hold for that to be a
pure improvement:

* the ``input_schema`` a host sees is byte for byte what the bare
  ``problem: OptimizationProblem`` annotation publishes — the model an AI
  writes against does not move;
* a schema-error result has exactly the fields, and the same non-error
  values, as the service's own result for a semantic error;
* a valid document is answered exactly as before.
"""

import copy
import json

import pytest
from mcp import Client
from mcp.server.mcpserver.utilities.func_metadata import func_metadata

from annealbridge.interfaces.composition import build_state_from_policy
from annealbridge.interfaces.mcp import mcp
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy
from tests.conftest import EXAMPLES_DIR

pytestmark = pytest.mark.anyio

PROBLEM_TOOLS = (
    "validate_optimization_problem",
    "recommend_backend",
    "solve_optimization",
)


def _knapsack() -> dict:
    return json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))


def _semantic_error_document() -> dict:
    document = _knapsack()
    document["constraints"][0]["terms"][0]["variable"] = "not_declared"
    return document


def _schema_error_document() -> dict:
    document = _knapsack()
    document["not_a_field"] = 1
    return document


def _bare_annotation_schema(tool_name: str) -> dict:
    """The input schema the SDK derives from ``problem: OptimizationProblem``."""

    async def reference(problem: OptimizationProblem):  # pragma: no cover
        raise NotImplementedError

    reference.__name__ = tool_name
    return func_metadata(reference).arg_model.model_json_schema(by_alias=True)


async def _call(tool: str, problem: object) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool(tool, {"problem": problem})
    assert result.is_error is False, result.content
    return result.structured_content


async def test_input_schema_is_byte_identical_to_the_bare_annotation():
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    for name in PROBLEM_TOOLS:
        published = json.dumps(tools[name].input_schema, sort_keys=True)
        expected = json.dumps(_bare_annotation_schema(name), sort_keys=True)
        assert published == expected, name
        # Still the full model, refusing unknown keys at every level.
        assert tools[name].input_schema["properties"]["problem"] == {
            "$ref": "#/$defs/OptimizationProblem"
        }
        assert tools[name].input_schema["required"] == ["problem"]


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
async def test_schema_error_has_the_shape_of_a_semantic_error(tool):
    schema = await _call(tool, _schema_error_document())
    semantic = await _call(tool, _semantic_error_document())

    assert schema.keys() == semantic.keys()
    assert [error["code"] for error in schema["errors"]] == ["UNKNOWN_FIELD"]
    assert [error["code"] for error in semantic["errors"]] == ["UNKNOWN_VARIABLE"]
    # Every error carries the same keys, a catalog action included.
    assert schema["errors"][0].keys() == semantic["errors"][0].keys()
    assert schema["errors"][0]["recommended_action"]
    assert schema["errors"][0]["retryable"] is False

    varying = {"errors", "message", "elapsed_ms"}
    assert {k: v for k, v in schema.items() if k not in varying} == {
        k: v for k, v in semantic.items() if k not in varying
    }
    if tool == "solve_optimization":
        assert schema["status"] == "invalid_problem"
        assert schema["message"] == schema["errors"][0]["message"]
        assert isinstance(schema["elapsed_ms"], float)
    else:
        assert schema["valid"] is False


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
async def test_a_valid_document_is_answered_as_before(tool):
    """The tool's answer equals the service's for the parsed model."""
    service = build_state_from_policy(ExecutionPolicy()).service
    problem = OptimizationProblem.model_validate(_knapsack())
    got = await _call(tool, _knapsack())

    if tool == "validate_optimization_problem":
        assert got == service.validate(problem).model_dump(mode="json")
    elif tool == "recommend_backend":
        assert got == service.recommend(problem).model_dump(mode="json")
    else:
        expected = service.solve(problem).model_dump(mode="json")
        for result in (got, expected):
            # Wall-clock readings differ on every run; nothing else may.
            result.pop("elapsed_ms")
            for attempt in result["attempts"]:
                for key in [key for key in attempt if key.endswith("_ms")]:
                    attempt.pop(key)
        assert got == expected


async def test_semantic_errors_still_come_back_all_at_once():
    document = _knapsack()
    document["constraints"][0]["terms"][0]["variable"] = "not_declared"
    document["variables"].append(copy.deepcopy(document["variables"][0]))
    result = await _call("solve_optimization", document)
    assert result["status"] == "invalid_problem"
    assert sorted(error["code"] for error in result["errors"]) == [
        "DUPLICATE_VARIABLE",
        "UNKNOWN_VARIABLE",
    ]
