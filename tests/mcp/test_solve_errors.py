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

from annealbridge.interfaces.mcp import mcp, server
from annealbridge.orchestration import ExecutionPolicy
from annealbridge.solvers import SolverRegistry
from tests.fakes.declared_backend import FAKE_LIMIT_KEY, FakeDeclaredBackend

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


# --- 2026-09-09 review F-11: booleans and strings in numeric fields ---------
#
# Type errors and semantic errors keep using different channels: a payload
# Pydantic cannot parse is an SDK *tool error* (the tool body never runs), a
# well-typed but nonsensical problem is a structured ``invalid_problem``
# result. F-11 does not move anything between the two channels — it only makes
# sure the type-layer message names the offending field and says *why* it was
# refused, so an agent can fix the payload without guessing.


def _set_path(problem: dict, path: str, value) -> dict:
    """Set e.g. ``objective.linear_terms[0].coefficient`` on a payload copy."""
    node = problem
    keys = path.split(".")
    for key in keys[:-1]:
        if key.endswith("]"):
            key, _, index = key[:-1].partition("[")
            node = node[key][int(index)]
        else:
            node = node[key]
    node[keys[-1]] = value
    return problem


@pytest.mark.parametrize("tool", ["validate_optimization_problem", "solve_optimization"])
@pytest.mark.parametrize(
    "path, value, field, noun",
    [
        ("objective.linear_terms[0].coefficient", True, "coefficient", "boolean"),
        ("objective.linear_terms[0].coefficient", "2", "coefficient", "string"),
        ("constraints[0].rhs", True, "rhs", "boolean"),
        ("solver.num_reads", True, "num_reads", "boolean"),
        ("solver.top_k", "3", "top_k", "string"),
    ],
    ids=[
        "coefficient-bool",
        "coefficient-string",
        "rhs-bool",
        "num_reads-bool",
        "top_k-string",
    ],
)
async def test_boolean_or_string_numeric_field_is_a_readable_tool_error(
    load_example, tool, path, value, field, noun
):
    """Type errors go through the SDK tool error, semantic ones through the
    structured ``invalid_problem`` result; F-11 only guarantees the type-layer
    message is readable — it adds no new structured channel."""
    problem = _set_path(
        copy.deepcopy(load_example("knapsack.json", backend="exact")), path, value
    )

    async with Client(mcp) as client:
        result = await client.call_tool(tool, {"problem": problem})

    assert result.is_error is True
    message = result.content[0].text
    assert field in message
    assert noun in message


# --- Unknown fields (models/strict.py) ---------------------------------------
#
# Same channel as the type errors above: an invented field name is the
# LLM caller's characteristic mistake, and dropping it silently would solve
# a different problem. The SDK's message names the offending path.


@pytest.mark.parametrize("tool", ["validate_optimization_problem", "solve_optimization"])
@pytest.mark.parametrize(
    "path, key",
    [
        ("", "minimize_secondary"),
        ("objective", "cubic_terms"),
        ("objective.linear_terms[0]", "variable3"),
        ("constraints[0]", "penalty"),
        ("solver", "num_restarts"),
    ],
    ids=["top-level", "objective", "term", "constraint", "solver"],
)
async def test_unknown_field_is_a_tool_error_naming_the_path(
    load_example, tool, path, key
):
    problem = copy.deepcopy(load_example("knapsack.json", backend="exact"))
    _set_path(problem, f"{path}.{key}" if path else key, 1)

    async with Client(mcp) as client:
        result = await client.call_tool(tool, {"problem": problem})

    assert result.is_error is True
    message = result.content[0].text
    assert key in message
    assert "not permitted" in message


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


# --- infeasibility diagnostics ----------------------------------------------
#
# An `infeasible` status is a structured result, not an error, and it now
# carries the reason: the candidate that came closest to feasibility and how
# often each hard constraint failed. Both come from re-validating the last
# attempt's candidates against the original problem, so the agent can say
# *which* requirement is the binding one without a second round trip.

# Neither constraint is unsatisfiable on its own, so the problem validator
# passes this; only enumeration can show the pair has no solution.
JOINTLY_INFEASIBLE_PROBLEM = {
    "version": "1.0",
    "name": "jointly-infeasible",
    "variables": [{"name": "x1"}, {"name": "x2"}, {"name": "x3"}],
    "objective": {
        "direction": "maximize",
        "linear_terms": [
            {"variable": "x1", "coefficient": 1},
            {"variable": "x2", "coefficient": 1},
            {"variable": "x3", "coefficient": 1},
        ],
    },
    "constraints": [
        {
            "id": "all_three",
            "type": "hard",
            "terms": [
                {"variable": "x1", "coefficient": 1},
                {"variable": "x2", "coefficient": 1},
                {"variable": "x3", "coefficient": 1},
            ],
            "operator": ">=",
            "rhs": 3,
        },
        {
            "id": "budget",
            "type": "hard",
            "terms": [
                {"variable": "x1", "coefficient": 3},
                {"variable": "x2", "coefficient": 2},
                {"variable": "x3", "coefficient": 1},
            ],
            "operator": "<=",
            "rhs": 1,
        },
    ],
    "solver": {"backend": "exact"},
}


async def test_infeasible_result_carries_structured_diagnostics():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "solve_optimization", {"problem": JOINTLY_INFEASIBLE_PROBLEM}
        )
        # Infeasible is an answer, not a transport failure.
        assert result.is_error is False
        content = result.structured_content

    assert content["status"] == "infeasible"
    assert content["infeasibility_proven"] is True
    assert content["solutions"] == []

    diagnostics = content["infeasibility"]
    closest = diagnostics["closest_candidate"]
    assert closest["variables"] == {"x1": 0, "x2": 0, "x3": 1}
    assert closest["hard_violation_total"] == 2.0
    assert [
        (e["constraint_id"], e["satisfied"], e["violation_amount"])
        for e in closest["constraint_evaluations"]
    ] == [("all_three", False, 2.0), ("budget", True, 0.0)]

    rates = diagnostics["hard_violation_rates"]
    assert rates
    assert [
        (r["constraint_id"], r["violated_candidates"], r["candidates"])
        for r in rates
    ] == [("all_three", 7, 8), ("budget", 6, 8)]
    assert all(
        rate["candidates"] == content["attempts"][-1]["unique_samples"]
        for rate in rates
    )


async def test_success_carries_no_diagnostics(load_example):
    problem = load_example("knapsack.json", backend="exact")

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        content = result.structured_content

    assert content["status"] == "success"
    assert content["infeasibility"] is None


class VendorCrashBackend(FakeDeclaredBackend):
    """A backend whose ``solve()`` raises a bare vendor exception.

    2026-09-09 review F-03: before the service-level fallback this escaped
    ``solve_optimization`` as an SDK tool error, so the agent lost the
    structured result (and the raw text was never redacted).
    """

    def solve(self, compiled_problem, preferences):
        self.solve_calls += 1
        raise RuntimeError("vendor sdk blew up")


async def test_backend_crash_is_a_structured_solver_error(load_example):
    # Registered under a shipped name: `solver.backend` is a Literal, so the
    # SDK's pydantic layer would refuse an unknown key before the tool runs.
    backend = VendorCrashBackend()
    server.reset_state(
        server.build_state_from_policy(
            ExecutionPolicy(allow_remote=True, limits={FAKE_LIMIT_KEY: 1000}),
            SolverRegistry({"simulated_annealing": backend}),
        )
    )
    problem = load_example("knapsack.json", backend="simulated_annealing")

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        content = result.structured_content

    assert backend.solve_calls == 1
    assert content["status"] == "solver_error"
    assert content["solutions"] == []
    (error,) = content["errors"]
    assert error["code"] == "SOLVER_ERROR"
    assert "RuntimeError" in error["message"]
