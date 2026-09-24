"""The wall-clock limit and cancellation end to end (batch 6 J).

Time-limit spec 2026-09-24 §4.2 and §9 items 2, 3 and 5, through
``OptimizationService.solve`` on real backends:

* **a limit that does not fire changes nothing**: with a generous
  ``wall_clock_limit_seconds`` and/or an uncancelled token, the result --
  solutions, attempts and every other field except the timings -- is
  exactly the result without either, on the annealer with one shard and
  several (in parallel), tabu with one shard and several, simulated
  bifurcation, and the annealer with post-processing;
* **a limit that has already passed** when attempt 1 reaches its backend
  leaves zero reads on every interruptible backend (a fake clock pushed
  forward from the progress callback);
* **real time**: a solve of about 5 s cut at 0.5 s returns within the
  documented overrun, marked, with every solution re-validated; a library
  caller's ``cancel()`` from another thread ends the solve promptly with
  ``SolveCancelled`` and leaves no shard thread behind.
"""

import random
import threading
import time

import pytest

from annealbridge.exceptions import SolveCancelled
from annealbridge.interrupt import CancelToken
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.candidates import evaluate_objective
from annealbridge.solvers import (
    SimulatedAnnealingBackend,
    SimulatedBifurcationBackend,
    SolverRegistry,
    TabuBackend,
)
from annealbridge.solvers.sharding import SHARD_THREAD_PREFIX
from annealbridge.validation import validate_solution
from tests.fakes.interruptible_backend import FakeClock

ATTEMPT_TIMINGS = ("compile_ms", "solve_ms", "validate_ms", "postprocess_ms")
LIMIT_REACHED = "WALL_CLOCK_LIMIT_REACHED"


def registry(workers: int = 4) -> SolverRegistry:
    """The interruptible backends, the sharded ones with several workers so
    the parallel shard path is the one exercised."""
    return SolverRegistry(
        {
            "simulated_annealing": SimulatedAnnealingBackend(workers=workers),
            "tabu": TabuBackend(workers=workers),
            "simulated_bifurcation": SimulatedBifurcationBackend(),
        }
    )


def without_timings(result) -> dict:
    """The result as JSON, minus the fields that measure time."""
    data = result.model_dump(mode="json")
    assert data.pop("elapsed_ms") is not None
    for attempt in data["attempts"]:
        for name in ATTEMPT_TIMINGS:
            attempt.pop(name)
    return data


def shard_threads() -> list[str]:
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(SHARD_THREAD_PREFIX)
    ]


def assert_revalidated(problem: OptimizationProblem, result) -> None:
    """Every returned solution is the re-validation's own verdict."""
    minimize = problem.objective.direction == "minimize"
    names = {variable.name for variable in problem.variables}
    for solution in result.solutions:
        assert set(solution.variables) == names
        check = validate_solution(problem, solution.variables)
        objective = evaluate_objective(problem.objective, solution.variables)
        assert check.feasible is True
        assert solution.hard_constraints_satisfied is True
        assert solution.objective_value == objective
        assert solution.soft_violation_score == check.soft_violation_score
        expected = (
            objective + check.soft_violation_score
            if minimize
            else objective - check.soft_violation_score
        )
        assert solution.ranking_score == expected
    scores = [solution.ranking_score for solution in result.solutions]
    assert scores == sorted(scores, reverse=not minimize)


# --------------------------------------------------------------------------
# §9 item 2: bit-identical when the interrupt never fires
# --------------------------------------------------------------------------

UNFIRED_CASES = [
    pytest.param("knapsack.json", {"backend": "simulated_annealing", "num_reads": 20}, id="sa-one-shard"),
    pytest.param("knapsack.json", {"backend": "simulated_annealing", "num_reads": 130}, id="sa-sharded"),
    pytest.param("knapsack.json", {"backend": "tabu", "num_reads": 10}, id="tabu-one-shard"),
    pytest.param("knapsack.json", {"backend": "tabu", "num_reads": 60}, id="tabu-sharded"),
    pytest.param("knapsack.json", {"backend": "simulated_bifurcation"}, id="sb"),
    pytest.param(
        "assignment.json",
        {"backend": "simulated_annealing", "num_reads": 20, "postprocess": "repair_local_search"},
        id="sa-postprocess",
    ),
    pytest.param(
        "assignment.json",
        {"backend": "simulated_annealing", "num_reads": 130, "postprocess": "repair_local_search"},
        id="sa-sharded-postprocess",
    ),
    # A too-weak penalty: several attempts, so the retry ladder runs with
    # the interrupt in place (tests/scenarios/test_retry.py pins the ladder).
    pytest.param(
        "knapsack.json",
        {"backend": "simulated_annealing", "num_reads": 100, "penalty_multiplier": 0.01},
        id="sa-retries",
    ),
]


class TestAnUnfiredLimitChangesNothing:
    @pytest.mark.parametrize(("example", "solver"), UNFIRED_CASES)
    @pytest.mark.parametrize(
        ("limit", "with_token"),
        [(1000, False), (None, True), (1000, True)],
        ids=["limit", "token", "limit+token"],
    )
    def test_same_result_as_without(self, load_example, example, solver, limit, with_token):
        service = OptimizationService(registry=registry())
        plain_problem = OptimizationProblem.model_validate(
            load_example(example, seed=11, **solver)
        )
        overrides = {} if limit is None else {"wall_clock_limit_seconds": limit}
        limited_problem = OptimizationProblem.model_validate(
            load_example(example, seed=11, **solver, **overrides)
        )

        plain = service.solve(plain_problem)
        limited = service.solve(
            limited_problem, cancel=CancelToken() if with_token else None
        )

        assert limited.wall_clock_limit_reached is False
        assert LIMIT_REACHED not in {warning.code for warning in limited.warnings}
        assert without_timings(limited) == without_timings(plain)
        assert plain.status == "success"

    def test_the_retry_case_really_retries(self, load_example):
        problem = OptimizationProblem.model_validate(
            load_example(
                "knapsack.json",
                backend="simulated_annealing",
                seed=11,
                num_reads=100,
                penalty_multiplier=0.01,
                wall_clock_limit_seconds=1000,
            )
        )
        result = OptimizationService(registry=registry()).solve(problem)

        assert len(result.attempts) > 1


# --------------------------------------------------------------------------
# §9 item 3: attempt 1 after the deadline, on the real backends
# --------------------------------------------------------------------------


class TestADeadlineAlreadyPassed:
    @pytest.mark.parametrize(
        "solver",
        [
            {"backend": "simulated_annealing", "num_reads": 10},
            {"backend": "simulated_annealing", "num_reads": 130},
            {"backend": "tabu", "num_reads": 10},
            {"backend": "tabu", "num_reads": 60},
            {"backend": "simulated_bifurcation"},
        ],
        ids=["sa-one-shard", "sa-sharded", "tabu-one-shard", "tabu-sharded", "sb"],
    )
    def test_attempt_one_runs_and_returns_no_reads(self, load_example, solver):
        clock = FakeClock()

        def on_progress(progress):
            if progress.attempt == 1 and progress.stage == "compile":
                clock.advance(60)

        problem = OptimizationProblem.model_validate(
            load_example(
                "knapsack.json", seed=11, max_retries=2, wall_clock_limit_seconds=5, **solver
            )
        )
        service = OptimizationService(registry=registry(), clock=clock)

        result = service.solve(problem, on_progress=on_progress)

        assert result.status == "infeasible"
        assert result.solutions == []
        (attempt,) = result.attempts
        assert attempt.samples_received == 0
        assert attempt.wall_clock_limit_reached is True
        assert result.wall_clock_limit_reached is True
        assert result.metadata is not None
        assert result.metadata.backend == solver["backend"]
        assert result.metadata.num_reads_requested == problem.solver.num_reads
        warnings = [w for w in result.warnings if w.code == LIMIT_REACHED]
        assert len(warnings) == 1
        assert "no further retry was started" in warnings[0].message
        assert result.message.startswith(
            "No feasible solution found before solver.wall_clock_limit_seconds ran out"
        )
        assert shard_threads() == []


# --------------------------------------------------------------------------
# §9 items 3 and 5: real time
# --------------------------------------------------------------------------

# 60 binary variables, a sparse random QUBO and one hard "pick 3 of the
# first 10" constraint. 400 reads x 25000 sweeps on two workers take about
# 5 s uninterrupted (measured 2026-09-24 on a 12-thread Windows machine:
# 5.2 s); a read is about 25 ms, so a stop is seen within one read per
# worker, and the tests below never run the uninterrupted solve.
LIVE_SWEEPS = 25_000
LIVE_READS = 400


def live_problem(limit: float | None) -> OptimizationProblem:
    rng = random.Random(11)
    names = [f"x{index}" for index in range(60)]
    solver = {
        "backend": "simulated_annealing",
        "num_reads": LIVE_READS,
        "num_sweeps": LIVE_SWEEPS,
        "seed": 5,
    }
    if limit is not None:
        solver["wall_clock_limit_seconds"] = limit
    return OptimizationProblem.model_validate(
        {
            "name": "wall-clock-live",
            "variables": [{"name": name} for name in names],
            "objective": {
                "direction": "minimize",
                "linear_terms": [
                    {"variable": name, "coefficient": rng.randint(-3, 3)} for name in names
                ],
                "quadratic_terms": [
                    {
                        "variable1": names[i],
                        "variable2": names[j],
                        "coefficient": rng.choice([-2, -1, 1, 2]),
                    }
                    for i in range(60)
                    for j in range(i + 1, 60)
                    if rng.random() < 0.1
                ],
            },
            "constraints": [
                {
                    "id": "pick_three",
                    "type": "hard",
                    "terms": [{"variable": name, "coefficient": 1} for name in names[:10]],
                    "operator": "==",
                    "rhs": 3,
                }
            ],
            "solver": solver,
        }
    )


def live_service() -> OptimizationService:
    return OptimizationService(registry=registry(workers=2))


class TestRealTime:
    def test_a_half_second_limit_cuts_a_multi_second_solve(self):
        problem = live_problem(0.5)

        result = live_service().solve(problem)

        # The lower bound is loose for Windows' coarse clock; the upper one
        # allows the uninterruptible stages plus one read per worker.
        assert 450 <= result.elapsed_ms <= 2500, result.elapsed_ms
        assert result.wall_clock_limit_reached is True
        assert result.attempts[0].wall_clock_limit_reached is True
        assert 0 <= result.attempts[0].samples_received < LIVE_READS
        assert LIMIT_REACHED in {warning.code for warning in result.warnings}
        assert result.status in {"success", "infeasible"}
        if result.status == "success":
            assert result.message.endswith(" The wall-clock limit cut the search short.")
        assert_revalidated(problem, result)
        assert shard_threads() == []

    def test_a_library_cancel_ends_the_solve_promptly(self):
        token = CancelToken()
        cancelled_at: list[float] = []

        def cancel() -> None:
            cancelled_at.append(time.perf_counter())
            token.cancel()

        timer = threading.Timer(0.5, cancel)
        started = time.perf_counter()
        timer.start()
        try:
            with pytest.raises(SolveCancelled):
                live_service().solve(live_problem(None), cancel=token)
            raised_at = time.perf_counter()
        finally:
            timer.cancel()
            timer.join()

        assert cancelled_at, "the solve ended before the cancel was sent"
        assert raised_at - cancelled_at[0] <= 2.0, raised_at - cancelled_at[0]
        # Well short of the uninterrupted 3-5 s.
        assert raised_at - started <= 2.5
        assert shard_threads() == []
