"""Cardinality constraints (schema 1.2) through a real MCP client.

An agent writes ``cardinality_constraints`` because the server instructions,
the tool descriptions and the prompts tell it to; this file holds that the
tools then accept such a document end to end: a version ``"1.2"`` problem
mixing a linear and cardinality constraints solves, each cardinality
constraint comes back in ``constraint_evaluations`` under its own id after
the linear ones, and a malformed or too-old document comes back as a
structured ``invalid_problem`` naming the path inside
``cardinality_constraints``, not as a tool error.
"""

import copy

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp
from annealbridge.models.error_catalog import RECOMMENDED_ACTIONS

pytestmark = pytest.mark.anyio

# Maximize 5a + 4b + 3c + 2d within a weight limit of 6 (weights 3, 3, 2, 1),
# taking at most one of a and b, and preferably at least three items. Worked
# by hand over the 16 assignments: a, c and d weigh 6 and are worth 10 with
# three items chosen; b, c, d is worth 9; every other feasible pick is worth
# less. So the unique optimum is a, c, d with value 10 and no soft cost.
PROBLEM: dict = {
    "version": "1.2",
    "name": "pick_with_counts",
    "variables": [
        {"name": "a", "type": "binary"},
        {"name": "b", "type": "binary"},
        {"name": "c", "type": "binary"},
        {"name": "d", "type": "binary"},
    ],
    "objective": {
        "direction": "maximize",
        "linear_terms": [
            {"variable": "a", "coefficient": 5},
            {"variable": "b", "coefficient": 4},
            {"variable": "c", "coefficient": 3},
            {"variable": "d", "coefficient": 2},
        ],
    },
    "constraints": [
        {
            "id": "weight_limit",
            "type": "hard",
            "terms": [
                {"variable": "a", "coefficient": 3},
                {"variable": "b", "coefficient": 3},
                {"variable": "c", "coefficient": 2},
                {"variable": "d", "coefficient": 1},
            ],
            "operator": "<=",
            "rhs": 6,
        }
    ],
    "cardinality_constraints": [
        {
            "id": "a_or_b",
            "type": "hard",
            "variables": ["a", "b"],
            "operator": "<=",
            "rhs": 1,
        },
        {
            "id": "at_least_three",
            "type": "soft",
            "variables": ["a", "b", "c", "d"],
            "operator": ">=",
            "rhs": 3,
            "weight": 1,
        },
    ],
    "solver": {"backend": "exact"},
}


async def _call(tool: str, problem: dict) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool(tool, {"problem": problem})
    # A structured result, never an SDK tool error.
    assert result.is_error is False, result.content
    return result.structured_content


async def test_a_version_1_2_cardinality_problem_solves():
    content = await _call("solve_optimization", PROBLEM)

    assert content["status"] == "success"
    assert content["optimality_proven"] is True
    best = content["solutions"][0]
    assert best["variables"] == {"a": 1, "b": 0, "c": 1, "d": 1}
    assert best["objective_value"] == 10.0
    assert best["soft_violation_score"] == 0.0
    assert best["hard_constraints_satisfied"] is True
    # The linear constraint first, then the cardinality ones in declaration
    # order, each under its own id.
    evaluations = best["constraint_evaluations"]
    assert [e["constraint_id"] for e in evaluations] == [
        "weight_limit",
        "a_or_b",
        "at_least_three",
    ]
    by_id = {e["constraint_id"]: e for e in evaluations}
    # actual_value of a cardinality constraint is the number chosen.
    assert by_id["a_or_b"]["actual_value"] == 1.0
    assert by_id["a_or_b"]["operator"] == "<="
    assert by_id["a_or_b"]["expected_value"] == 1.0
    assert by_id["a_or_b"]["constraint_type"] == "hard"
    assert by_id["at_least_three"]["actual_value"] == 3.0
    assert by_id["at_least_three"]["satisfied"] is True
    assert all(e["satisfied"] for e in evaluations)


async def test_validate_counts_no_slack_bit_for_the_hard_at_most_one():
    content = await _call("validate_optimization_problem", PROBLEM)

    assert content["valid"] is True
    assert content["errors"] == []
    assert content["model_type"] == "bqm"
    # 4 binaries + 3 slack bits for the weight limit (slack 0..6) + 1 for
    # the soft ">= 3" over four variables (slack 0..1); the hard "<= 1" is a
    # pairwise penalty with none.
    assert content["estimated_compiled_variables"] == 8


async def test_a_malformed_cardinality_constraint_is_a_structured_schema_error():
    document = copy.deepcopy(PROBLEM)
    document["cardinality_constraints"][0]["rhs"] = "1"

    solved = await _call("solve_optimization", document)
    validated = await _call("validate_optimization_problem", document)

    assert solved["status"] == "invalid_problem"
    assert solved["solutions"] == []
    assert validated["valid"] is False
    for content in (solved, validated):
        assert [(e["code"], e["path"]) for e in content["errors"]] == [
            ("INVALID_FIELD_VALUE", "cardinality_constraints[0].rhs")
        ]
        error = content["errors"][0]
        assert error["retryable"] is False
        assert error["recommended_action"] == RECOMMENDED_ACTIONS["INVALID_FIELD_VALUE"]
        # A schema error never echoes the submitted value.
        assert '"1"' not in error["message"]


async def test_cardinality_constraints_below_version_1_2_name_the_version():
    document = copy.deepcopy(PROBLEM)
    document["version"] = "1.1"

    content = await _call("solve_optimization", document)

    assert content["status"] == "invalid_problem"
    assert [(e["code"], e["path"]) for e in content["errors"]] == [
        ("FEATURE_REQUIRES_NEWER_VERSION", "version")
    ]
    error = content["errors"][0]
    assert '"1.2" or later' in error["message"]
    assert error["recommended_action"] == RECOMMENDED_ACTIONS[
        "FEATURE_REQUIRES_NEWER_VERSION"
    ]
