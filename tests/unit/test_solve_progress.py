"""``OptimizationService.solve(on_progress=...)`` and the ``SolveProgress`` event.

The core side of progress reporting: which events a solve emits, in which
order, and that a listener can never change the result. The MCP adapter that
turns these events into progress notifications has its own tests under
``tests/mcp/``.
"""

import inspect
import logging

import pytest

from annealbridge.models import OptimizationProblem, SolveAttempt
from annealbridge.orchestration import (
    ExecutionPolicy,
    OptimizationService,
    ProgressCallback,
    SolveProgress,
)
from annealbridge.orchestration.progress import STAGES
from annealbridge.solvers import SolverRegistry

# Two binaries, one hard constraint: solved by ``exact`` in one attempt.
FEASIBLE = {
    "version": "1.0",
    "name": "progress-feasible",
    "variables": [
        {"name": "x", "type": "binary"},
        {"name": "y", "type": "binary"},
    ],
    "objective": {
        "direction": "maximize",
        "linear_terms": [
            {"variable": "x", "coefficient": 3},
            {"variable": "y", "coefficient": 2},
        ],
    },
    "constraints": [
        {
            "id": "at_most_one",
            "type": "hard",
            "terms": [
                {"variable": "x", "coefficient": 1},
                {"variable": "y", "coefficient": 1},
            ],
            "operator": "<=",
            "rhs": 1,
        }
    ],
    "solver": {"backend": "exact"},
}

# Each constraint is satisfiable on its own — so the validator lets the
# problem through — but together they admit no assignment, so the penalty
# ladder on a heuristic backend runs every attempt it was given.
JOINTLY_INFEASIBLE = {
    **FEASIBLE,
    "name": "progress-infeasible",
    "constraints": [
        {
            "id": "both",
            "type": "hard",
            "terms": [
                {"variable": "x", "coefficient": 1},
                {"variable": "y", "coefficient": 1},
            ],
            "operator": ">=",
            "rhs": 2,
        },
        FEASIBLE["constraints"][0],
    ],
    "solver": {"backend": "simulated_annealing", "seed": 1, "max_retries": 2},
}


def make_service(policy: ExecutionPolicy | None = None) -> OptimizationService:
    return OptimizationService(
        registry=SolverRegistry.default(), policy=policy or ExecutionPolicy()
    )


def as_tuples(events: list[SolveProgress]) -> list[tuple[int, int, str, str]]:
    return [(e.attempt, e.max_attempts, e.stage, e.backend) for e in events]


# Everything in a result except the wall-clock fields two runs never share:
# the result's own ``elapsed_ms`` and every ``*_ms`` field of an attempt.
TIMING_FREE = {
    "elapsed_ms": True,
    "attempts": {
        "__all__": {name for name in SolveAttempt.model_fields if name.endswith("_ms")}
    },
}


class TestSolveProgressEvent:
    def test_message_is_deterministic_and_names_no_setting(self):
        event = SolveProgress(
            attempt=2, max_attempts=3, stage="solve", backend="simulated_annealing"
        )

        assert event.message == "attempt 2 of 3: solving on simulated_annealing"
        assert event.step == 4
        assert event.total_steps == 9

    @pytest.mark.parametrize(
        ("stage", "verb"),
        [("compile", "compiling"), ("solve", "solving"), ("validate", "validating")],
    )
    def test_every_stage_has_a_verb(self, stage, verb):
        event = SolveProgress(attempt=1, max_attempts=1, stage=stage, backend="b")

        assert event.message == f"attempt 1 of 1: {verb} on b"
        assert event.step == STAGES.index(stage)

    def test_event_is_immutable(self):
        event = SolveProgress(attempt=1, max_attempts=1, stage="compile", backend="b")

        with pytest.raises(AttributeError):
            event.attempt = 2  # type: ignore[misc]


class TestSolveCallback:
    def test_on_progress_is_keyword_only_and_optional(self):
        parameter = inspect.signature(OptimizationService.solve).parameters[
            "on_progress"
        ]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    def test_one_attempt_emits_the_three_stages_in_order(self):
        events: list[SolveProgress] = []
        problem = OptimizationProblem.model_validate(FEASIBLE)

        result = make_service().solve(problem, on_progress=events.append)

        assert result.status == "success"
        assert as_tuples(events) == [
            (1, 1, "compile", "exact"),
            (1, 1, "solve", "exact"),
            (1, 1, "validate", "exact"),
        ]
        assert [e.message for e in events] == [
            "attempt 1 of 1: compiling on exact",
            "attempt 1 of 1: solving on exact",
            "attempt 1 of 1: validating on exact",
        ]

    def test_every_attempt_of_the_penalty_ladder_is_reported(self):
        events: list[SolveProgress] = []
        problem = OptimizationProblem.model_validate(JOINTLY_INFEASIBLE)

        result = make_service().solve(problem, on_progress=events.append)

        assert result.status == "infeasible"
        assert len(result.attempts) == 3
        assert as_tuples(events) == [
            (attempt, 3, stage, "simulated_annealing")
            for attempt in (1, 2, 3)
            for stage in STAGES
        ]
        assert [e.step for e in events] == list(range(9))
        assert {e.total_steps for e in events} == {9}

    def test_nothing_is_emitted_when_no_attempt_starts(self):
        # A variable limit of 1 stops the solve in ``_prepare_attempts``,
        # before any compile: there is no attempt to report on.
        events: list[SolveProgress] = []
        problem = OptimizationProblem.model_validate(FEASIBLE)
        service = make_service(ExecutionPolicy(exact_max_variables=1))

        result = service.solve(problem, on_progress=events.append)

        assert result.status == "resource_limit_exceeded"
        assert events == []

    def test_an_invalid_problem_emits_nothing(self):
        events: list[SolveProgress] = []
        problem = OptimizationProblem.model_validate(
            {
                **FEASIBLE,
                "constraints": [
                    {**FEASIBLE["constraints"][0], "operator": ">=", "rhs": 3}
                ],
            }
        )

        result = make_service().solve(problem, on_progress=events.append)

        assert result.status == "invalid_problem"
        assert events == []

    def test_a_raising_callback_changes_nothing_but_the_log(self, caplog):
        problem = OptimizationProblem.model_validate(FEASIBLE)
        service = make_service()
        calls = 0

        def broken(event: SolveProgress) -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError(f"listener gone at {event.stage}")

        quiet = service.solve(problem)
        with caplog.at_level(logging.WARNING, logger="annealbridge.orchestration"):
            noisy = service.solve(problem, on_progress=broken)

        # Every stage still called the listener, and the solve reached the
        # same result — not ``solver_error``, not a shorter attempt list.
        assert calls == 3
        assert noisy.status == "success"
        assert noisy.model_dump(exclude=TIMING_FREE) == quiet.model_dump(
            exclude=TIMING_FREE
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 3
        assert "progress callback raised RuntimeError" in warnings[0].getMessage()
        assert "solving continues" in warnings[0].getMessage()

    def test_callback_type_alias_accepts_a_plain_function(self):
        # Documentation-level guard: the alias is a one-argument callable
        # returning None, so ``list.append`` and a def both satisfy it.
        events: list[SolveProgress] = []
        callback: ProgressCallback = events.append

        make_service().solve(
            OptimizationProblem.model_validate(FEASIBLE), on_progress=callback
        )

        assert len(events) == 3
