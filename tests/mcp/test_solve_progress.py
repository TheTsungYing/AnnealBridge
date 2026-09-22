"""Progress notifications from ``solve_optimization``, at the MCP boundary.

The core's own progress semantics (when an event is emitted, what it says)
are pinned by tests/unit/test_solve_progress.py. What is tested here is the
adapter: that a real client asking for progress receives one notification per
stage of every attempt with the core's wording, that asking for progress
changes nothing about the result, and that ``_progress_reporter`` keeps the
solve alive -- logging once and going quiet -- when the client cannot be
reached any more.
"""

import logging

import anyio
import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp
from annealbridge.interfaces.mcp.tools import _progress_reporter
from annealbridge.orchestration import SolveProgress

pytestmark = pytest.mark.anyio

TOOLS_LOGGER = "annealbridge.interfaces.mcp.tools"

# Two binaries, one hard constraint: valid, and solved by `exact` at once.
TINY_PROBLEM = {
    "version": "1.0",
    "name": "progress-probe",
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

# One binary asked to be both 0 and 1. Each constraint is satisfiable on its
# own -- x = 0 satisfies the first, x = 1 the second -- so neither is
# TRIVIALLY_INFEASIBLE and the problem passes validation; together they admit
# no assignment, so every candidate fails independent re-validation and the
# whole retry ladder is walked. max_retries 2 makes that three attempts.
CONTRADICTORY_PROBLEM = {
    "version": "1.0",
    "name": "progress-infeasible",
    "variables": [{"name": "x", "type": "binary"}],
    "objective": {
        "direction": "maximize",
        "linear_terms": [{"variable": "x", "coefficient": 1}],
    },
    "constraints": [
        {
            "id": "x_is_zero",
            "type": "hard",
            "terms": [{"variable": "x", "coefficient": 1}],
            "operator": "<=",
            "rhs": 0,
        },
        {
            "id": "x_is_one",
            "type": "hard",
            "terms": [{"variable": "x", "coefficient": 1}],
            "operator": ">=",
            "rhs": 1,
        },
    ],
    "solver": {"backend": "simulated_annealing", "seed": 1, "max_retries": 2},
}


def collector() -> tuple[list[tuple[float, float | None, str | None]], object]:
    """A progress_callback plus the list it appends every call to."""
    events: list[tuple[float, float | None, str | None]] = []

    async def callback(
        progress: float, total: float | None = None, message: str | None = None
    ) -> None:
        events.append((progress, total, message))

    return events, callback


def without_timings(content: dict) -> dict:
    """``content`` minus every wall-clock field, which no two runs share.

    The service stamps ``elapsed_ms`` on the result and ``compile_ms`` /
    ``solve_ms`` / ``validate_ms`` on each attempt; everything else the
    exact backend produces is deterministic.
    """
    stripped = dict(content)
    stripped.pop("elapsed_ms", None)
    stripped["attempts"] = [
        {key: value for key, value in attempt.items() if not key.endswith("_ms")}
        for attempt in content["attempts"]
    ]
    return stripped


class DeadClient:
    """A ctx whose ``report_progress`` always fails, counting its calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def report_progress(
        self, progress: float, total: float | None = None, message: str | None = None
    ) -> None:
        self.calls += 1
        raise RuntimeError("client gone")


class RecordingClient:
    """A ctx that records the arguments of every ``report_progress``."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float | None, str | None]] = []

    async def report_progress(
        self, progress: float, total: float | None = None, message: str | None = None
    ) -> None:
        self.calls.append((progress, total, message))


async def test_exact_solve_reports_one_notification_per_stage():
    events, callback = collector()

    async with Client(mcp) as client:
        result = await client.call_tool(
            "solve_optimization",
            {"problem": TINY_PROBLEM},
            progress_callback=callback,
        )

    assert result.is_error is False
    assert result.structured_content["status"] == "success"
    # One attempt, three stages, in execution order, with the core's wording.
    assert events == [
        (0.0, 3.0, "attempt 1 of 1: compiling on exact"),
        (1.0, 3.0, "attempt 1 of 1: solving on exact"),
        (2.0, 3.0, "attempt 1 of 1: validating on exact"),
    ]


async def test_asking_for_progress_does_not_change_the_result():
    _, callback = collector()

    async with Client(mcp) as client:
        watched = await client.call_tool(
            "solve_optimization",
            {"problem": TINY_PROBLEM},
            progress_callback=callback,
        )
        unwatched = await client.call_tool(
            "solve_optimization", {"problem": TINY_PROBLEM}
        )

    assert watched.is_error is False
    assert unwatched.is_error is False
    assert without_timings(watched.structured_content) == without_timings(
        unwatched.structured_content
    )


async def test_every_attempt_of_the_retry_ladder_is_reported():
    events, callback = collector()

    async with Client(mcp) as client:
        result = await client.call_tool(
            "solve_optimization",
            {"problem": CONTRADICTORY_PROBLEM},
            progress_callback=callback,
        )

    assert result.is_error is False
    content = result.structured_content
    # The problem passed validation (no invalid_problem) and no candidate
    # survived re-validation, so all three attempts really ran.
    assert content["status"] == "infeasible"
    assert len(content["attempts"]) == 3

    messages = [
        f"attempt {attempt} of 3: {verb} on simulated_annealing"
        for attempt in (1, 2, 3)
        for verb in ("compiling", "solving", "validating")
    ]
    assert events == [
        (float(step), 9.0, message) for step, message in enumerate(messages)
    ]


async def test_reporter_forwards_each_event_to_the_context():
    fake = RecordingClient()
    reporter = _progress_reporter(fake)
    first = SolveProgress(
        attempt=1, max_attempts=2, stage="compile", backend="exact"
    )
    second = SolveProgress(
        attempt=2, max_attempts=2, stage="validate", backend="exact"
    )

    # ``anyio.from_thread.run`` only works from a worker thread anyio itself
    # started, which is exactly where the service calls the callback.
    await anyio.to_thread.run_sync(
        lambda: [reporter(event) for event in (first, second)]
    )

    assert fake.calls == [
        (float(first.step), float(first.total_steps), first.message),
        (float(second.step), float(second.total_steps), second.message),
    ]


async def test_reporter_goes_quiet_after_one_undeliverable_notification(caplog):
    caplog.set_level(logging.INFO, logger=TOOLS_LOGGER)
    fake = DeadClient()
    reporter = _progress_reporter(fake)
    first = SolveProgress(
        attempt=1, max_attempts=2, stage="compile", backend="exact"
    )
    second = SolveProgress(attempt=1, max_attempts=2, stage="solve", backend="exact")

    # The failure must not escape: the solve has to finish and return.
    await anyio.to_thread.run_sync(
        lambda: [reporter(event) for event in (first, second)]
    )

    # The second event was skipped, not retried against a dead client.
    assert fake.calls == 1
    records = [record for record in caplog.records if record.name == TOOLS_LOGGER]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "progress notification dropped" in records[0].getMessage()


async def test_the_context_parameter_stays_out_of_the_input_schema():
    async with Client(mcp) as client:
        tools = (await client.list_tools()).tools

    solve = next(tool for tool in tools if tool.name == "solve_optimization")
    schema = solve.input_schema
    # ``ctx`` is injected by the server, so a client must not be asked for it.
    assert schema["required"] == ["problem"]
    assert list(schema["properties"]) == ["problem"]
