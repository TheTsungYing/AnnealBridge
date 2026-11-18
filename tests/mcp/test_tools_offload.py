"""The problem tools run the core off the event loop (2026-09-09 review F-15).

``validate`` / ``recommend`` / ``solve`` are all CPU-bound: run directly on
the event loop, one large problem freezes the whole server for every other
client of the session. Each of the three tools must therefore hand the call
to ``anyio.to_thread.run_sync``, so the service never observes the loop's own
thread.

The same holds for what a tool does before it reaches the service (2026-11-18
review R-05): parsing the submitted document into the model is a full
``model_validate``, and the capabilities view asks every backend whether it
is available. Neither may run on the event loop either.

The assertions are on where the call lands — the thread it runs on, and the
absence of a running event loop there — not on elapsed time: exact, and never
flaky on a loaded machine.
"""

import asyncio
import threading

import anyio
import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp, tools
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


def record_function(monkeypatch, name: str) -> list[tuple[int, bool]]:
    """Wrap the module-level ``tools.<name>`` so every call records where it ran.

    The tools look these names up in their own module at call time, so the
    wrapper is what they call. Entries are shaped as in ``record_calls``.
    """
    original = getattr(tools, name)
    calls: list[tuple[int, bool]] = []

    def wrapper(*args, **kwargs):
        calls.append((threading.get_ident(), running_loop()))
        return original(*args, **kwargs)

    monkeypatch.setattr(tools, name, wrapper)
    return calls


async def call_tool(name: str, arguments: dict | None = None):
    if arguments is None:
        arguments = {"problem": TINY_PROBLEM}
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments)
        assert result.is_error is False
        return result


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


PROBLEM_TOOLS = [
    "validate_optimization_problem",
    "recommend_backend",
    "solve_optimization",
]


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
async def test_tool_parses_the_problem_in_a_worker_thread(monkeypatch, tool):
    """2026-11-18 review R-05: the document is parsed off the event loop."""
    loop_ident = threading.get_ident()
    calls = record_function(monkeypatch, "parse_problem")

    await call_tool(tool)

    assert len(calls) == 1
    ident, on_a_loop = calls[0]
    assert ident != loop_ident
    assert on_a_loop is False


@pytest.mark.parametrize("tool", PROBLEM_TOOLS)
async def test_a_document_that_fails_the_schema_is_parsed_in_a_worker_thread_too(
    monkeypatch, tool
):
    """The rejection is the same structured result as before (not a tool
    error), produced without the service ever being called."""
    loop_ident = threading.get_ident()
    calls = record_function(monkeypatch, "parse_problem")
    service = get_state().service
    for method in ("validate", "recommend", "solve"):
        monkeypatch.setattr(
            service, method, lambda *args, **kwargs: pytest.fail("service called")
        )

    result = await call_tool(tool, {"problem": {**TINY_PROBLEM, "bogus": 1}})

    assert len(calls) == 1
    ident, on_a_loop = calls[0]
    assert ident != loop_ident
    assert on_a_loop is False
    content = result.structured_content
    assert [error["code"] for error in content["errors"]] == ["UNKNOWN_FIELD"]
    if tool == "solve_optimization":
        assert content["status"] == "invalid_problem"
    else:
        assert content["valid"] is False


@pytest.mark.parametrize("include_schema", [False, True])
async def test_capabilities_are_assembled_in_a_worker_thread(
    monkeypatch, include_schema
):
    """2026-11-18 review R-05: so is the capabilities view."""
    loop_ident = threading.get_ident()
    calls = record_function(monkeypatch, "build_capabilities")

    result = await call_tool(
        "get_optimization_capabilities", {"include_schema": include_schema}
    )

    assert len(calls) == 1
    ident, on_a_loop = calls[0]
    assert ident != loop_ident
    assert on_a_loop is False
    schema = result.structured_content["problem_json_schema"]
    assert (schema is not None) is include_schema


async def test_a_request_cancelled_while_parsing_never_starts_the_solve(monkeypatch):
    """The parse thread is awaited to its end, then the cancellation takes
    effect before any solve is started: nothing is left running for a
    request nobody is waiting on, and the handler returns no result."""
    solve_calls = record_calls(monkeypatch, "solve")
    parsing = threading.Event()
    release = threading.Event()
    original = tools.parse_problem

    def held_parse(data):
        parsing.set()
        assert release.wait(30)
        return original(data)

    monkeypatch.setattr(tools, "parse_problem", held_parse)
    returned: list[object] = []

    async def handle() -> None:
        # The handler itself: the context is only used once a solve reports
        # progress, which this request never gets to.
        returned.append(await tools.solve_optimization(TINY_PROBLEM, None))

    async with anyio.create_task_group() as group:
        group.start_soon(handle)
        assert await anyio.to_thread.run_sync(parsing.wait, 30)
        group.cancel_scope.cancel()
        release.set()

    assert returned == []
    assert solve_calls == []
