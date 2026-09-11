"""The ``solve_optimization`` MCP tool on the simulated annealing backend.

Unlike the exact backend, SA is a sampler: we do not pin the objective value
here, only that a seeded run returns a feasible best solution and that the same
seed produces byte-identical solutions across two separate client sessions.
"""

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp

pytestmark = pytest.mark.anyio


async def solve(problem: dict) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        return result.structured_content


async def test_seeded_sa_run_returns_a_feasible_solution(load_example):
    problem = load_example(
        "knapsack.json", backend="simulated_annealing", seed=42, num_reads=40
    )

    content = await solve(problem)

    assert content["status"] == "success"
    assert content["solutions"][0]["hard_constraints_satisfied"] is True

    # A local backend reports execution metadata too, the reads it asked
    # the sampler for included; the vendor-side fields stay empty.
    metadata = content["metadata"]
    assert metadata is not None
    assert metadata["backend"] == "simulated_annealing"
    assert metadata["remote"] is False
    assert metadata["model_type"] == "bqm"
    assert metadata["num_reads_requested"] == problem["solver"]["num_reads"]
    assert metadata["timing_us"] == {}
    assert metadata["solver_id"] is None


async def test_same_seed_gives_identical_solutions(load_example):
    problem = load_example(
        "knapsack.json", backend="simulated_annealing", seed=42
    )

    # Two independent client sessions, same payload: the seed must make the
    # whole ranked solution list reproducible.
    first = await solve(problem)
    second = await solve(problem)

    assert first["solutions"] == second["solutions"]
