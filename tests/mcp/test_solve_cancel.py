"""Cancelling ``solve_optimization`` and its wall-clock limit, over MCP.

Batch 6 (J), time-limit spec 2026-09-24 §2.3, §6.2 and §9 items 2, 4 and 6.
Before this batch a cancelled request kept its solve running to the end on
a worker thread (every shard thread busy, the result thrown away), and the
default in-process client even received the full result. Now the handler
gives the solve a ``CancelToken`` and cancels it when the request is
cancelled; the solve stops at its next checkpoint and raises
``SolveCancelled``, which the handler turns back into the request's own
cancellation (no response).

Pinned here, through a real MCP client on both in-memory paths -- the
default in-process dispatcher and the legacy JSON-RPC session:

* the client gets no result; the server's solve ends with SolveCancelled
  well before a full run would have, the SA backend reports an
  interrupted, partial read set, the handler never outlives the solve
  (no orphan), and no shard thread is left behind;
* nothing is reported as progress after the cancellation;
* a token that never fires changes nothing: the MCP result equals the
  library's (no token) for SA single / multi shard, tabu, SB and SA with
  post-processing, timings and the package version aside;
* ``solver.wall_clock_limit_seconds`` over MCP yields a flagged partial
  result, and on ``exact`` (no ``supports_interrupt``) is refused as
  WALL_CLOCK_LIMIT_UNSUPPORTED by solve, validate and recommend.

The time bounds are loose on purpose (a full run of the long problem takes
several seconds; a cancelled one must end within ``STOP_BOUND`` of the
cancel), so a loaded machine does not make them flaky.
"""

import random
import threading
import time
from dataclasses import dataclass, field

import anyio
import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp, tools
from annealbridge.interfaces.mcp.server import get_state
from annealbridge.interfaces.mcp.tools import _progress_reporter, _send_progress
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import CancelToken, SolveCancelled, SolveProgress
from annealbridge.solvers.sharding import SHARD_THREAD_PREFIX

pytestmark = pytest.mark.anyio

# The client gives up on the request this long after sending it.
CANCEL_AFTER = 0.5
# A cancelled solve must have ended this soon after the cancel. A full run
# of LONG_READS reads takes several seconds (about 5 s on 12 threads).
STOP_BOUND = 2.0
# How long a test waits, at most, for the server side to settle; generous,
# so a regression shows up as a failed timing assertion, not a hang.
SETTLE_TIMEOUT = 30.0

SA = "simulated_annealing"
LONG_READS = 400

MODES = [
    pytest.param("auto", id="in-process"),
    pytest.param("legacy", id="legacy-jsonrpc"),
]


def long_problem(**solver_overrides) -> dict:
    """60 binaries, a random quadratic objective and one cardinality cap.

    400 reads of 20 000 sweeps: 16 shards of 25 reads, several seconds of
    simulated annealing in total, while a single read takes milliseconds
    -- so a stop is prompt and a full run is unmistakably longer.
    """
    rng = random.Random(7)
    names = [f"x{i}" for i in range(60)]
    quadratic = [
        {
            "variable1": names[i],
            "variable2": names[j],
            "coefficient": rng.randint(-5, 5) or 1,
        }
        for i in range(60)
        for j in range(i + 1, 60)
        if rng.random() < 0.2
    ]
    return {
        "version": "1.0",
        "name": "cancel-probe",
        "variables": [{"name": name, "type": "binary"} for name in names],
        "objective": {
            "direction": "maximize",
            "linear_terms": [
                {"variable": name, "coefficient": rng.randint(1, 20)} for name in names
            ],
            "quadratic_terms": quadratic,
        },
        "constraints": [
            {
                "id": "cap",
                "type": "hard",
                "terms": [{"variable": name, "coefficient": 1} for name in names],
                "operator": "<=",
                "rhs": 20,
            }
        ],
        "solver": {
            "backend": SA,
            "seed": 1,
            "num_reads": LONG_READS,
            "num_sweeps": 20000,
            "max_retries": 0,
            **solver_overrides,
        },
    }


# Two binaries, one hard constraint: solved by ``exact`` at once.
TINY_PROBLEM = {
    "version": "1.0",
    "name": "after-cancel",
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


def shard_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(SHARD_THREAD_PREFIX)
    ]


@dataclass
class ServerSide:
    """What the server did with the request, recorded by thin wrappers."""

    solve_done: threading.Event = field(default_factory=threading.Event)
    solve_ended: float | None = None
    solve_outcome: object = None
    solve_kwargs: dict = field(default_factory=dict)
    backend_rows: list[int] = field(default_factory=list)
    backend_interrupted: list[bool] = field(default_factory=list)
    handler_done: threading.Event = field(default_factory=threading.Event)
    handler_ended: float | None = None
    handler_outcome: object = None
    shard_threads_at_solve_end: int | None = None


def instrument(monkeypatch) -> ServerSide:
    """Wrap the service's ``solve``, the SA backend and the handler's runner.

    Every wrapper passes its call through unchanged and only records; the
    conftest fixture injects a fresh state per test and monkeypatch
    restores the originals afterwards.
    """
    side = ServerSide()
    state = get_state()
    service = state.service
    original_solve = service.solve

    def solve(problem, **kwargs):
        side.solve_kwargs = kwargs
        try:
            side.solve_outcome = original_solve(problem, **kwargs)
            return side.solve_outcome
        except BaseException as exc:
            side.solve_outcome = exc
            raise
        finally:
            side.solve_ended = time.perf_counter()
            side.shard_threads_at_solve_end = len(shard_threads())
            side.solve_done.set()

    monkeypatch.setattr(service, "solve", solve)

    backend = state.registry.get(SA)
    original_backend_solve = backend.solve

    def backend_solve(compiled, preferences, **kwargs):
        raw = original_backend_solve(compiled, preferences, **kwargs)
        side.backend_rows.append(len(raw.samples))
        side.backend_interrupted.append(raw.interrupted)
        return raw

    monkeypatch.setattr(backend, "solve", backend_solve)

    original_runner = tools._run_cancellable

    async def runner(solve_call, cancel):
        try:
            side.handler_outcome = await original_runner(solve_call, cancel)
            return side.handler_outcome
        except BaseException as exc:
            side.handler_outcome = exc
            raise
        finally:
            side.handler_ended = time.perf_counter()
            side.handler_done.set()

    monkeypatch.setattr(tools, "_run_cancellable", runner)
    return side


async def wait_for(event: threading.Event) -> None:
    with anyio.fail_after(SETTLE_TIMEOUT):
        while not event.is_set():
            await anyio.sleep(0.01)


def timed_collector():
    """A progress_callback recording ``(perf_counter, message)`` per call."""
    events: list[tuple[float, str | None]] = []

    async def callback(progress, total=None, message=None) -> None:
        events.append((time.perf_counter(), message))

    return events, callback


def without_run_facts(content: dict) -> dict:
    """``content`` minus what legitimately differs between two runs.

    The wall-clock fields (``elapsed_ms``; ``compile_ms`` / ``solve_ms`` /
    ``validate_ms`` / ``postprocess_ms`` per attempt) and the package
    version stamp. Everything else -- solutions, energies, attempt counts,
    metadata, warnings -- must match exactly.
    """
    stripped = {
        key: value
        for key, value in content.items()
        if key not in {"elapsed_ms", "annealbridge_version"}
    }
    stripped["attempts"] = [
        {key: value for key, value in attempt.items() if not key.endswith("_ms")}
        for attempt in content["attempts"]
    ]
    return stripped


# ---------------------------------------------------------------------------
# §9 item 4: a cancelled request stops its solve.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_cancelling_the_request_stops_the_solve(monkeypatch, mode):
    assert shard_threads() == [], "a previous test left shard threads running"
    side = instrument(monkeypatch)
    events, callback = timed_collector()
    received = None

    async with Client(mcp, mode=mode) as client:
        sent = time.perf_counter()
        with anyio.move_on_after(CANCEL_AFTER) as scope:
            received = await client.call_tool(
                "solve_optimization",
                {"problem": long_problem()},
                progress_callback=callback,
            )
        cancelled_at = sent + CANCEL_AFTER

        # The client gave up and got nothing back.
        assert scope.cancelled_caught
        assert received is None

        await wait_for(side.solve_done)
        await wait_for(side.handler_done)

        # The server stopped the solve: it ended in SolveCancelled, soon
        # after the cancel, and the backend really cut its reads short
        # rather than finishing and being discarded afterwards.
        assert isinstance(side.solve_outcome, SolveCancelled)
        assert isinstance(side.solve_kwargs.get("cancel"), CancelToken)
        assert side.solve_ended - cancelled_at < STOP_BOUND
        assert side.backend_interrupted == [True]
        (rows,) = side.backend_rows
        assert rows < LONG_READS

        # No orphan: the handler finished only after the solve had, and
        # every shard thread had already returned by then.
        assert side.solve_ended <= side.handler_ended
        assert side.handler_ended - cancelled_at < STOP_BOUND
        assert side.shard_threads_at_solve_end == 0
        assert shard_threads() == []

        # Nothing was reported after the cancellation: the solve never got
        # to its validate stage.
        assert [message for _, message in events] == [
            f"attempt 1 of 1: compiling on {SA}",
            f"attempt 1 of 1: solving on {SA}",
        ]
        assert all(when < cancelled_at for when, _ in events)

        # The session survives the cancellation and serves the next call.
        follow_up = await client.call_tool(
            "solve_optimization", {"problem": TINY_PROBLEM}
        )
        assert follow_up.is_error is False
        assert follow_up.structured_content["status"] == "success"

    assert shard_threads() == []


# ---------------------------------------------------------------------------
# §6.2: the progress reporter goes quiet once the token is cancelled. The
# live test above cannot hit the race (the solve stops before its next
# stage), so the adapter's own guard is pinned directly.
# ---------------------------------------------------------------------------


class RecordingContext:
    """A ctx that records every ``report_progress`` call."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def report_progress(self, progress, total=None, message=None) -> None:
        self.calls.append(message)


async def test_reporter_sends_nothing_once_the_token_is_cancelled():
    ctx = RecordingContext()
    cancel = CancelToken()
    reporter = _progress_reporter(ctx, cancel)
    before = SolveProgress(attempt=1, max_attempts=1, stage="compile", backend=SA)
    after = SolveProgress(attempt=1, max_attempts=1, stage="validate", backend=SA)

    def run() -> None:
        reporter(before)
        cancel.cancel()
        reporter(after)
        reporter(after)

    await anyio.to_thread.run_sync(run)

    assert ctx.calls == [before.message]


async def test_reporter_without_a_token_still_forwards_everything():
    ctx = RecordingContext()
    reporter = _progress_reporter(ctx)
    event = SolveProgress(attempt=1, max_attempts=1, stage="solve", backend=SA)

    await anyio.to_thread.run_sync(lambda: [reporter(event), reporter(event)])

    assert ctx.calls == [event.message, event.message]


class ArgumentRecordingContext:
    """A ctx that records the full arguments of every ``report_progress``."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float | None, str | None]] = []

    async def report_progress(self, progress, total=None, message=None) -> None:
        self.calls.append((progress, total, message))


async def test_send_progress_skips_a_cancelled_request():
    # The second check, on the event loop where the watcher sets the token:
    # a notification scheduled before the cancel but run after it is dropped.
    ctx = ArgumentRecordingContext()
    cancel = CancelToken()
    cancel.cancel()

    await _send_progress(ctx, cancel, 1.0, 3.0, "attempt 1 of 1: solving on exact")

    assert ctx.calls == []


@pytest.mark.parametrize(
    "cancel",
    [pytest.param(CancelToken(), id="live-token"), pytest.param(None, id="no-token")],
)
async def test_send_progress_forwards_the_arguments_otherwise(cancel):
    ctx = ArgumentRecordingContext()

    await _send_progress(ctx, cancel, 2.0, 3.0, "attempt 1 of 1: validating on exact")

    assert ctx.calls == [(2.0, 3.0, "attempt 1 of 1: validating on exact")]


# ---------------------------------------------------------------------------
# §9 item 2: a token that never fires changes nothing.
# ---------------------------------------------------------------------------

BIT_IDENTICAL_CASES = [
    pytest.param("knapsack.json", {"backend": SA, "seed": 11, "num_reads": 10}, id="sa-one-shard"),
    pytest.param("knapsack.json", {"backend": SA, "seed": 11, "num_reads": 120}, id="sa-many-shards"),
    pytest.param("knapsack.json", {"backend": "tabu", "seed": 11, "num_reads": 40}, id="tabu-many-shards"),
    pytest.param(
        "knapsack.json",
        {"backend": "simulated_bifurcation", "seed": 11, "num_reads": 50},
        id="sb",
    ),
    pytest.param(
        "knapsack.json",
        {
            "backend": SA,
            "seed": 11,
            "num_reads": 60,
            "postprocess": "repair_local_search",
            "postprocess_candidates": 5,
        },
        id="sa-many-shards-postprocess",
    ),
]


@pytest.mark.parametrize(("example", "solver"), BIT_IDENTICAL_CASES)
async def test_mcp_result_equals_the_library_result(load_example, example, solver):
    payload = load_example(example, **solver)
    service = get_state().service

    library = service.solve(OptimizationProblem.model_validate(payload))
    async with Client(mcp) as client:
        result = await client.call_tool("solve_optimization", {"problem": payload})

    assert result.is_error is False
    via_mcp = result.structured_content
    assert via_mcp["status"] == "success"
    assert via_mcp["wall_clock_limit_reached"] is False
    assert without_run_facts(via_mcp) == without_run_facts(
        library.model_dump(mode="json")
    )


# ---------------------------------------------------------------------------
# §9 items 3 and 6 at the MCP boundary: the wall-clock limit.
# ---------------------------------------------------------------------------


async def test_wall_clock_limit_yields_a_flagged_partial_result(monkeypatch):
    side = instrument(monkeypatch)

    async with Client(mcp) as client:
        result = await client.call_tool(
            "solve_optimization",
            {"problem": long_problem(wall_clock_limit_seconds=0.5)},
        )

    assert result.is_error is False
    content = result.structured_content
    assert content["wall_clock_limit_reached"] is True
    assert [attempt["wall_clock_limit_reached"] for attempt in content["attempts"]] == [
        True
    ]
    assert "WALL_CLOCK_LIMIT_REACHED" in [w["code"] for w in content["warnings"]]
    assert content["status"] in {"success", "infeasible"}
    # Measured from entering the service, like the limit itself.
    assert 450 <= content["elapsed_ms"] <= 2500
    assert side.backend_interrupted == [True]
    assert side.backend_rows[0] < LONG_READS
    assert shard_threads() == []


async def test_exact_refuses_a_wall_clock_limit_on_every_tool(load_example):
    problem = load_example(
        "knapsack.json", backend="exact", wall_clock_limit_seconds=5
    )

    async with Client(mcp) as client:
        solved = await client.call_tool("solve_optimization", {"problem": problem})
        validated = await client.call_tool(
            "validate_optimization_problem", {"problem": problem}
        )
        recommended = await client.call_tool("recommend_backend", {"problem": problem})

    solve_content = solved.structured_content
    assert solve_content["status"] == "invalid_problem"
    assert [e["code"] for e in solve_content["errors"]] == [
        "WALL_CLOCK_LIMIT_UNSUPPORTED"
    ]
    assert solve_content["errors"][0]["path"] == "solver.wall_clock_limit_seconds"
    assert solve_content["errors"][0]["recommended_action"]
    assert solve_content["attempts"] == []

    validate_content = validated.structured_content
    assert validate_content["valid"] is False
    assert [e["code"] for e in validate_content["errors"]] == [
        "WALL_CLOCK_LIMIT_UNSUPPORTED"
    ]

    (entry,) = [
        e
        for e in recommended.structured_content["recommendations"]
        if e["backend"] == "exact"
    ]
    assert entry["usable"] is False
    assert [b["code"] for b in entry["blocking"]] == ["WALL_CLOCK_LIMIT_UNSUPPORTED"]
