"""Error paths of the problem tools over MCP, via a real client.

A problem the tools refuse comes back as a *structured result*
(``result.is_error is False``), never as an SDK tool error, whichever layer
refuses it:

* the schema layer — a document that does not fit ``OptimizationProblem``
  (an unknown field, a missing field, a value of the wrong type, a problem
  that is not even an object) — is parsed inside the tool
  (``interfaces/problem_input.py``) and answered with ``status:
  invalid_problem`` (``valid: false`` for validate / recommend) carrying
  ``UNKNOWN_FIELD``, ``MISSING_FIELD`` or ``INVALID_FIELD_VALUE`` with the
  offending path, every such error at once;
* the semantic layer — a well-typed problem the validator or the backend gate
  rejects — keeps its own status and codes.

A caller must therefore read ``status`` (or ``valid``): a clean ``is_error``
does not mean the solve succeeded. The one remaining SDK tool error is a call
without any ``problem`` argument at all, which the SDK refuses before the tool
runs.
"""

import copy
import json

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


# --- Schema errors: a document that does not fit the problem schema ---------
#
# Parsed inside the tool (interfaces/problem_input.py), so it comes back in the
# same shape as a semantic error: ``is_error`` False, ``status:
# invalid_problem`` for solve and ``valid: false`` for validate / recommend,
# with catalog codes and the path of each offending field. The message is
# pydantic's short description only: never the submitted value, never a
# documentation URL.

PROBLEM_TOOLS = (
    "validate_optimization_problem",
    "recommend_backend",
    "solve_optimization",
)


async def _call(tool: str, arguments: dict):
    async with Client(mcp) as client:
        return await client.call_tool(tool, arguments)


def _schema_errors(tool: str, result) -> list[dict]:
    """The errors of a schema-error answer, after checking its envelope."""
    assert result.is_error is False, result.content
    content = result.structured_content
    if tool == "solve_optimization":
        assert content["status"] == "invalid_problem"
        assert content["solutions"] == []
        assert content["attempts"] == []
        assert content["message"] == content["errors"][0]["message"]
    else:
        assert content["valid"] is False
    if tool == "recommend_backend":
        assert content["recommendations"] == []
    for error in content["errors"]:
        assert "input_value" not in error["message"]
        assert "pydantic.dev" not in error["message"]
        assert error["retryable"] is False
        assert error["recommended_action"]
    return content["errors"]


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


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
async def test_wrong_type_in_payload_is_a_structured_invalid_field_value(
    load_example, tool
):
    problem = copy.deepcopy(load_example("knapsack.json", backend="exact"))
    problem["objective"]["linear_terms"][0]["coefficient"] = "SECRET_SENTINEL_123"

    result = await _call(tool, {"problem": problem})

    (error,) = _schema_errors(tool, result)
    assert error["code"] == "INVALID_FIELD_VALUE"
    assert error["path"] == "objective.linear_terms[0].coefficient"
    # The submitted value is never echoed back, anywhere in the answer.
    assert "SECRET_SENTINEL_123" not in json.dumps(result.structured_content)


# --- 2026-09-09 review F-11: booleans and strings in numeric fields ---------
#
# F-11 made sure a boolean or a string in a numeric field is refused with a
# message that names the field and says *why*, so an agent can fix the
# payload without guessing. The refusal is now a structured
# INVALID_FIELD_VALUE at the field's path instead of an SDK tool error; the
# readable message is what F-11 guarantees and it must survive the move.


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
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
async def test_boolean_or_string_numeric_field_is_a_readable_invalid_field_value(
    load_example, tool, path, value, field, noun
):
    problem = _set_path(
        copy.deepcopy(load_example("knapsack.json", backend="exact")), path, value
    )

    result = await _call(tool, {"problem": problem})

    (error,) = _schema_errors(tool, result)
    assert error["code"] == "INVALID_FIELD_VALUE"
    assert error["path"] == path
    message = error["message"]
    assert field in message
    assert noun in message
    # pydantic's "Value error, " prefix is stripped.
    assert not message.startswith("Value error")


# --- Unknown fields (models/strict.py) ---------------------------------------
#
# An invented field name is the LLM caller's characteristic mistake, and
# dropping it silently would solve a different problem. It is refused as
# UNKNOWN_FIELD naming the offending path, at every level of the document.


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
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
async def test_unknown_field_is_a_structured_unknown_field_naming_the_path(
    load_example, tool, path, key
):
    problem = copy.deepcopy(load_example("knapsack.json", backend="exact"))
    field_path = f"{path}.{key}" if path else key
    _set_path(problem, field_path, 1)

    result = await _call(tool, {"problem": problem})

    (error,) = _schema_errors(tool, result)
    assert error["code"] == "UNKNOWN_FIELD"
    assert error["path"] == field_path
    assert error["message"] == "unknown field, not in the problem schema"


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
async def test_every_schema_error_comes_back_at_once(load_example, tool):
    problem = copy.deepcopy(load_example("knapsack.json", backend="exact"))
    del problem["objective"]
    problem["constraints"][0]["operator"] = "<"
    problem["not_a_field"] = 1

    result = await _call(tool, {"problem": problem})

    errors = _schema_errors(tool, result)
    assert [(error["code"], error["path"]) for error in errors] == [
        ("MISSING_FIELD", "objective"),
        ("INVALID_FIELD_VALUE", "constraints[0].operator"),
        ("UNKNOWN_FIELD", "not_a_field"),
    ]


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
@pytest.mark.parametrize(
    "problem", ["a knapsack please", [], None], ids=["string", "list", "null"]
)
async def test_a_problem_that_is_not_an_object_has_no_path(tool, problem):
    result = await _call(tool, {"problem": problem})

    (error,) = _schema_errors(tool, result)
    assert error["code"] == "INVALID_FIELD_VALUE"
    assert error["path"] is None
    assert error["message"] == "the problem must be a JSON object"


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
async def test_a_call_without_the_problem_argument_is_still_an_sdk_tool_error(tool):
    """The one edge case left on the SDK's channel: with no ``problem``
    argument at all the SDK refuses the call before the tool body runs, so
    there is nothing to parse and no structured result to return."""
    result = await _call(tool, {})

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
    # Blocked by the schema layer, not by the validator: JSON has no
    # representation for inf/nan, so the value arrives as null and parsing the
    # document (`penalty_multiplier: float`) refuses it as INVALID_FIELD_VALUE.
    # The solver-side guard (`allow_inf_nan=False` on the field, plus the
    # INVALID_SOLVER_PREFERENCE validator rule) covers the in-process callers
    # that bypass this transport; over MCP the request never gets that far.
    # A spy backend under the requested name proves the solver never runs.
    backend = FakeDeclaredBackend()
    server.reset_state(
        server.build_state_from_policy(
            ExecutionPolicy(allow_remote=True, limits={FAKE_LIMIT_KEY: 1000}),
            SolverRegistry({"simulated_annealing": backend}),
        )
    )
    problem = load_example(
        "knapsack.json", backend="simulated_annealing", penalty_multiplier=value
    )

    result = await _call("solve_optimization", {"problem": problem})

    (error,) = _schema_errors("solve_optimization", result)
    assert error["code"] == "INVALID_FIELD_VALUE"
    assert error["path"] == "solver.penalty_multiplier"
    assert backend.solve_calls == 0


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
