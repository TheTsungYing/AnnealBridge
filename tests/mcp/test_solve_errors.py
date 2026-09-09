"""Error paths of the ``solve_optimization`` MCP tool, via a real client.

These tests show that the two error channels are both observable through the
MCP round trip and that they stay distinct:

* an SDK *tool error* (``result.is_error is True``) for the type layer, i.e. a
  payload Pydantic cannot even parse into ``OptimizationProblem``; and
* a *structured status* (``result.is_error is False`` plus a ``status`` other
  than ``"success"``) for the semantic layer, i.e. a well-typed problem that
  the validator or the backend gate rejects.

A caller must therefore check both: a clean ``is_error`` does not mean the
solve succeeded.
"""

import copy

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp

pytestmark = pytest.mark.anyio


# The constraint references "item_z", which is not in `variables`. Well-typed,
# so the SDK parses it happily; only the validator can reject it.
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


async def test_remote_backend_is_refused_without_a_fallback(load_example):
    # The autouse fixture injects the default policy, where allow_remote is
    # False; the gate fires before any availability probe.
    problem = load_example("knapsack.json", backend="dwave_qpu")

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        content = result.structured_content

    assert content["status"] == "backend_unavailable"
    codes = [error["code"] for error in content["errors"]]
    assert "REMOTE_DISABLED" in codes
    # Refusal must be explicit: no silent downgrade to a local backend.
    assert content["solutions"] == []


async def test_wrong_type_in_payload_is_an_sdk_tool_error(load_example):
    problem = copy.deepcopy(load_example("knapsack.json", backend="exact"))
    problem["objective"]["linear_terms"][0]["coefficient"] = "abc"

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})

    # Pydantic rejects the payload before the tool body runs, so this surfaces
    # as a tool error rather than as a SolveResult.
    assert result.is_error is True


async def test_unknown_variable_is_a_structured_invalid_problem():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "solve_optimization", {"problem": UNKNOWN_VARIABLE_PROBLEM}
        )
        # Semantic failure: the call itself succeeded.
        assert result.is_error is False
        content = result.structured_content

    assert content["status"] == "invalid_problem"
    assert content["solutions"] == []


# --- 2026-09-09 review (F-02 / F-07 / F-08): parameter ceilings -------------


async def test_top_k_over_the_ceiling_is_a_structured_resource_limit(load_example):
    # top_k is a service-level ceiling that applies to every backend; the
    # value is refused, never clamped down to the ceiling.
    problem = load_example("knapsack.json", backend="exact", top_k=1001)

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        content = result.structured_content

    assert content["status"] == "resource_limit_exceeded"
    codes = [error["code"] for error in content["errors"]]
    assert "TOP_K_LIMIT" in codes
    assert content["solutions"] == []


async def test_max_retries_over_the_local_ceiling_is_refused(load_example):
    # simulated_annealing is a local backend, so max_retries is bounded by
    # ANNEALBRIDGE_MAX_LOCAL_RETRIES (default 10) — 11 is over it.
    problem = load_example(
        "knapsack.json", backend="simulated_annealing", max_retries=11
    )

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        content = result.structured_content

    assert content["status"] == "resource_limit_exceeded"
    codes = [error["code"] for error in content["errors"]]
    assert "RETRY_LIMIT" in codes
    assert content["solutions"] == []


@pytest.mark.parametrize(
    "value", [float("inf"), float("nan")], ids=["inf", "nan"]
)
async def test_non_finite_penalty_multiplier_never_reaches_the_solver(
    load_example, value
):
    # Blocked by the MCP type layer, not by the validator: JSON has no
    # representation for inf/nan, so the value arrives as null and the tool's
    # argument model (`penalty_multiplier: float`) refuses to parse it. The
    # solver-side guard (`allow_inf_nan=False` on the field, plus the
    # INVALID_SOLVER_PREFERENCE validator rule) covers the in-process callers
    # that bypass this transport; over MCP the request never gets that far.
    problem = load_example(
        "knapsack.json", backend="exact", penalty_multiplier=value
    )

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})

    assert result.is_error is True
    assert "penalty_multiplier" in result.content[0].text
