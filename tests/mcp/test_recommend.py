"""The ``recommend_backend`` MCP tool through the in-memory client (3a §26.4).

The tool is a one-line delegation to ``service.recommend()``; here we only
check the round trip: structured content is a dict shaped like
``BackendRecommendationResult``, the knapsack ranking survives, and no
backend was solved or asked for a time limit along the way.
"""

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp, server
from annealbridge.validation import BackendRecommendationResult

pytestmark = pytest.mark.anyio


async def recommend(problem: dict) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool("recommend_backend", {"problem": problem})
        assert result.is_error is False
        return result.structured_content


async def test_knapsack_ranks_exact_first(load_example):
    content = await recommend(load_example("knapsack.json"))

    assert isinstance(content, dict)
    assert content["valid"] is True
    assert content["errors"] == []
    assert content["recommendations"][0]["backend"] == "exact"
    assert content["recommendations"][0]["rank"] == 1
    assert content["recommendations"][0]["usable"] is True


async def test_remote_backends_are_unusable_under_the_default_policy(load_example):
    content = await recommend(load_example("knapsack.json"))
    registry = server.get_state().registry

    remote = [n for n in registry.names() if registry.get(n).capabilities.remote]
    assert len(remote) == 3
    entries = {e["backend"]: e for e in content["recommendations"]}
    for name in remote:
        assert entries[name]["usable"] is False
        assert "REMOTE_DISABLED" in [b["code"] for b in entries[name]["blocking"]]


async def test_structured_content_parses_as_the_result_model(load_example):
    content = await recommend(load_example("knapsack.json"))
    parsed = BackendRecommendationResult.model_validate(content)
    assert [e.rank for e in parsed.recommendations] == [1, 2, 3, 4, 5]
    assert parsed.advisory.startswith("Advisory only")


async def test_nothing_is_solved(load_example, monkeypatch):
    registry = server.get_state().registry

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("recommend_backend must not solve or resolve time limits")

    for name in registry.names():
        backend = registry.get(name)
        monkeypatch.setattr(backend, "solve", _forbidden)
        monkeypatch.setattr(backend, "resolve_time_limit", _forbidden)

    content = await recommend(load_example("knapsack.json"))
    assert content["valid"] is True


async def test_invalid_problem_has_errors_and_no_recommendations(load_example):
    problem = load_example("knapsack.json")
    problem["constraints"][0]["terms"][0]["variable"] = "ghost"

    content = await recommend(problem)

    assert content["valid"] is False
    assert "UNKNOWN_VARIABLE" in [e["code"] for e in content["errors"]]
    assert content["recommendations"] == []
