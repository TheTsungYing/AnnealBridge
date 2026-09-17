"""The ``solve_optimization`` MCP tool on the exact backend, via a real client.

The knapsack optimum itself is pinned by tests/scenarios/test_knapsack.py; the
point here is that the same answer survives the MCP round trip, that the SDK
hands back structured content (a dict, not a JSON string) and that slack
variables stay inside the compiler.
"""

from importlib.metadata import version

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp

pytestmark = pytest.mark.anyio

KNAPSACK_OPTIMUM_VALUE = 17.0

INTEGER_KNAPSACK_OPTIMUM_VALUE = 34.0
INTEGER_KNAPSACK_OPTIMUM = {"item_a": 0, "item_b": 1, "item_c": 1, "item_d": 3}


async def test_solve_knapsack_on_exact_backend(load_example):
    problem = load_example("knapsack.json", backend="exact")

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        content = result.structured_content

    # The SDK must give us a parsed object, not a JSON string the caller would
    # have to decode itself.
    assert isinstance(content, dict)

    assert content["status"] == "success"
    assert content["optimality_proven"] is True
    # The one-line summary travels with the payload and states the proof.
    assert isinstance(content["message"], str)
    assert "proved optimality" in content["message"]
    assert content["annealbridge_version"] == version("annealbridge")
    assert content["elapsed_ms"] >= 0
    attempt = content["attempts"][0]
    assert attempt["compiled_variables"] == 8  # 4 items + 4 slack bits
    assert attempt["compiled_interactions"] > 0
    assert all(attempt[key] >= 0 for key in ("compile_ms", "solve_ms", "validate_ms"))

    best = content["solutions"][0]
    assert best["objective_value"] == pytest.approx(KNAPSACK_OPTIMUM_VALUE)

    # Slack bits are named with a "__" prefix and must never reach the caller.
    for solution in content["solutions"]:
        assert all(not name.startswith("__") for name in solution["variables"])

    # A local backend reports execution metadata too: which backend ran,
    # that it ran here, and the model type the service stamped on it.
    metadata = content["metadata"]
    assert metadata is not None
    assert metadata["backend"] == "exact"
    assert metadata["remote"] is False
    assert metadata["model_type"] == "bqm"
    # No vendor facts on a local run, and this backend takes no read count.
    assert metadata["timing_us"] == {}
    assert metadata["solver_id"] is None
    assert metadata["num_reads_requested"] is None


async def test_solve_integer_knapsack_on_exact_backend(load_example):
    # 3b: the integer variables are binary-encoded for a bqm backend, but the
    # caller must see decoded integers in [lower_bound, upper_bound] — never
    # the encoding bits.
    problem = load_example("integer_knapsack.json", backend="exact")

    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        content = result.structured_content

    assert content["status"] == "success"

    best = content["solutions"][0]
    assert best["objective_value"] == pytest.approx(INTEGER_KNAPSACK_OPTIMUM_VALUE)
    assert best["variables"] == INTEGER_KNAPSACK_OPTIMUM

    for solution in content["solutions"]:
        for name, value in solution["variables"].items():
            assert not name.startswith("__")
            assert isinstance(value, int)
            assert 0 <= value <= 3


async def test_solve_reports_the_validators_warnings(load_example):
    # The same advice validate_optimization_problem gives for this backend
    # travels with the solve result, so an agent that skipped validate still
    # learns that the exact backend ignored its seed.
    problem = load_example("knapsack.json", backend="exact", seed=7)

    async with Client(mcp) as client:
        solved = await client.call_tool("solve_optimization", {"problem": problem})
        validated = await client.call_tool(
            "validate_optimization_problem", {"problem": problem}
        )

    assert solved.is_error is False
    content = solved.structured_content
    assert content["status"] == "success"
    assert [warning["code"] for warning in content["warnings"]] == ["SEED_IGNORED"]
    assert content["warnings"] == validated.structured_content["warnings"]
