"""The ``validate_optimization_problem`` MCP tool, driven through a real client.

These exercise the adapter end to end: the problem dict is parsed by the SDK
into ``OptimizationProblem``, the tool returns ``ProblemValidationResult`` and
the SDK turns it into structured content. The validator's own rules are
covered by tests/unit/test_problem_validator*.py; here we only assert that the
errors and warnings survive the round trip.
"""

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp

pytestmark = pytest.mark.anyio


# The constraint references "item_z", which is not in `variables`.
UNKNOWN_VARIABLE_PROBLEM = {
    "version": "1.0",
    "name": "unknown-variable",
    "variables": [
        {"name": "item_a", "type": "binary"},
        {"name": "item_b", "type": "binary"},
    ],
    "objective": {
        "direction": "maximize",
        "linear_terms": [
            {"variable": "item_a", "coefficient": 3},
            {"variable": "item_b", "coefficient": 2},
        ],
    },
    "constraints": [
        {
            "id": "capacity",
            "type": "hard",
            "terms": [{"variable": "item_z", "coefficient": 1}],
            "operator": "<=",
            "rhs": 1,
        }
    ],
}

# Valid, but the soft weight (0.1) is far below 1% of the objective scale
# (|100| + |100| = 200, so the threshold is 2.0).
SMALL_SOFT_WEIGHT_PROBLEM = {
    "version": "1.0",
    "name": "small-soft-weight",
    "variables": [
        {"name": "item_a", "type": "binary"},
        {"name": "item_b", "type": "binary"},
    ],
    "objective": {
        "direction": "maximize",
        "linear_terms": [
            {"variable": "item_a", "coefficient": 100},
            {"variable": "item_b", "coefficient": 100},
        ],
    },
    "constraints": [
        {
            "id": "prefer_a",
            "type": "soft",
            "terms": [{"variable": "item_a", "coefficient": 1}],
            "operator": "==",
            "rhs": 1,
            "weight": 0.1,
        }
    ],
}


async def validate(problem: dict) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "validate_optimization_problem", {"problem": problem}
        )
        assert result.is_error is False
        return result.structured_content


async def test_unknown_variable_is_reported_as_an_error():
    content = await validate(UNKNOWN_VARIABLE_PROBLEM)

    assert content["valid"] is False
    codes = [error["code"] for error in content["errors"]]
    assert "UNKNOWN_VARIABLE" in codes


async def test_unknown_variable_error_carries_path():
    content = await validate(UNKNOWN_VARIABLE_PROBLEM)

    error = next(e for e in content["errors"] if e["code"] == "UNKNOWN_VARIABLE")
    assert error["path"] == "constraints[0].terms[0]"
    assert error["retryable"] is False


async def test_small_soft_weight_is_a_warning_not_an_error():
    content = await validate(SMALL_SOFT_WEIGHT_PROBLEM)

    assert content["valid"] is True
    assert content["errors"] == []
    codes = [warning["code"] for warning in content["warnings"]]
    assert "SOFT_WEIGHT_SMALL" in codes


async def test_valid_problem_reports_estimates():
    content = await validate(SMALL_SOFT_WEIGHT_PROBLEM)

    estimated = content["estimated_compiled_variables"]
    assert isinstance(estimated, int)
    assert estimated > 0
    # An equality constraint needs no slack bits, so only the two binaries.
    assert estimated == 2
    assert content["objective_scale"] == pytest.approx(200.0)


async def test_errors_carry_recommended_action():
    # tools.py documents that validation errors come back "each with a
    # recommended_action"; the adapter must not drop the catalog text.
    content = await validate(UNKNOWN_VARIABLE_PROBLEM)

    action = content["errors"][0]["recommended_action"]
    assert isinstance(action, str)
    assert action.strip()
