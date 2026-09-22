"""The problem tools run the core off the event loop (2026-09-09 review F-15).

``validate`` / ``recommend`` / ``solve`` are all CPU-bound: run directly on
the event loop, one large problem freezes the whole server for every other
client of the session. Each of the three tools must therefore hand the call
to ``anyio.to_thread.run_sync``, so the service never observes the loop's own
thread.

The assertions are on where the call lands — the thread it runs on, and the
absence of a running event loop there — not on elapsed time: exact, and never
flaky on a loaded machine.
"""

import asyncio
import threading

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp
from annealbridge.interfaces.mcp.server import get_state

pytestmark = pytest.mark.anyio


# Two binaries and one hard constraint: valid, and solved by `exact` at once.
TINY_PROBLEM = {
    "version": "1.0",
    "name": "offload-probe",
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
            "id": "at_most_one",
            "type": "hard",
            "terms": [
                {"variable": "item_a", "coefficient": 1},
                {"variable": "item_b", "coefficient": 1},
            ],
            "operator": "<=",
            "rhs": 1,
        }
    ],
    "solver": {"backend": "exact"},
}


def running_loop() -> bool:
    """Is this thread currently driving an event loop?"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def record_calls(monkeypatch, method: str) -> list[tuple[int, bool]]:
    """Wrap ``service.<method>`` so every call records where it ran.

    Each entry is ``(thread ident, is there a running event loop here)``.
    The conftest fixture injects a fresh state per test and monkeypatch
    restores the real method afterwards, so the wrapper never leaks.
    """
    service = get_state().service
    original = getattr(service, method)
    calls: list[tuple[int, bool]] = []

    # ``solve`` is called with ``on_progress=`` as well; the wrapper passes
    # every keyword through so it observes, never narrows, the real call.
    def wrapper(problem, **kwargs):
        calls.append((threading.get_ident(), running_loop()))
        return original(problem, **kwargs)

    monkeypatch.setattr(service, method, wrapper)
    return calls


async def call_tool(name: str) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(name, {"problem": TINY_PROBLEM})
        assert result.is_error is False


@pytest.mark.parametrize(
    ("tool", "method"),
    [
        ("validate_optimization_problem", "validate"),
        ("recommend_backend", "recommend"),
        ("solve_optimization", "solve"),
    ],
    ids=["validate", "recommend", "solve"],
)
async def test_tool_runs_the_service_in_a_worker_thread(monkeypatch, tool, method):
    # The in-memory client shares this coroutine's loop and thread, so the
    # tool body itself runs right here; anything the tool offloads does not.
    loop_ident = threading.get_ident()
    assert running_loop()
    calls = record_calls(monkeypatch, method)

    await call_tool(tool)

    # The tool really went through the service (a wrong method name here
    # would otherwise pass silently).
    assert calls
    for ident, on_a_loop in calls:
        assert ident != loop_ident
        assert on_a_loop is False
