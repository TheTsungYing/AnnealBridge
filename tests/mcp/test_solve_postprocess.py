"""``solve_optimization`` with post-processing on, through a real MCP client.

Batch 4 (G), postprocess spec 2026-09-23 §5 and §9 item 10: the structured
output carries ``source`` on every solution and the attempt's
``postprocess`` statistics; with post-processing off every solution is
``"solver"`` and the statistics are null.
"""

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp

pytestmark = pytest.mark.anyio

SOURCES = {"solver", "repaired", "local_search", "repaired_local_search"}
STATS_KEYS = {
    "candidates_selected",
    "repair_attempted",
    "repair_succeeded",
    "local_search_started",
    "local_search_improved",
    "new_candidates",
    "feasible_added",
    "limit_reached",
}


async def solve(problem: dict) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": problem})
        assert result.is_error is False
        return result.structured_content


async def test_postprocessed_solve_reports_sources_and_statistics(load_example):
    problem = load_example(
        "knapsack.json",
        backend="simulated_annealing",
        seed=3,
        num_reads=10,
        postprocess="repair_local_search",
        postprocess_candidates=5,
    )

    content = await solve(problem)

    assert content["status"] == "success"
    assert content["solutions"]
    for solution in content["solutions"]:
        assert solution["source"] in SOURCES
        if solution["source"] != "solver":
            assert solution["energy"] is None
            assert solution["sample_count"] == 0
    attempt = content["attempts"][0]
    assert isinstance(attempt["postprocess"], dict)
    assert set(attempt["postprocess"]) == STATS_KEYS
    assert 1 <= attempt["postprocess"]["candidates_selected"] <= 5
    assert attempt["postprocess_ms"] is not None


async def test_default_solve_marks_every_solution_as_the_solvers(load_example):
    content = await solve(
        load_example("knapsack.json", backend="simulated_annealing", seed=3)
    )

    assert content["status"] == "success"
    assert {solution["source"] for solution in content["solutions"]} == {"solver"}
    assert content["attempts"][0]["postprocess"] is None
    assert content["attempts"][0]["postprocess_ms"] is None
