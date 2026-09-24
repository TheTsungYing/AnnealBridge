"""The service side of the wall-clock limit and cancellation (batch 6 J).

Time-limit spec 2026-09-24 §4, §6.3 and §9 items 1, 3, 5 and 6, pinned
without a real clock: the service gets a :class:`FakeClock` through
``OptimizationService(clock=...)`` and the scripted backend moves it (or
cancels the token) at the point a test chooses, so every outcome below is
deterministic.

* who receives ``interrupt=``: only a backend declaring
  ``supports_interrupt``, and only when the solve has a limit or a token;
* the deadline is measured on the service's clock from entering ``solve``;
* a retry is not started after the deadline, while attempt 1 always
  starts and its backend decides how much it can do (possibly nothing);
* the flag, the one WALL_CLOCK_LIMIT_REACHED warning and the messages,
  and "only the uninterruptible stages ran past it" leaves no flag;
* post-processing cut by the limit is reported apart from its ceilings;
* cancellation raises ``SolveCancelled`` from the checkpoints, wins over
  the deadline, releases the concurrency slot and never pre-empts a
  validation error;
* a limit on a backend that cannot stop part-way is refused with
  WALL_CLOCK_LIMIT_UNSUPPORTED by validate, solve and recommend alike.
"""

import json

import pytest
from pydantic import ValidationError

from annealbridge.exceptions import SolveCancelled
from annealbridge.interrupt import CancelToken, Interrupt
from annealbridge.models import OptimizationProblem, SolverPreferences
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import validate_problem
from tests.conftest import EXAMPLES_DIR
from tests.fakes.declared_backend import (
    FAKE_DECLARED_NAME,
    FAKE_LIMIT_KEY,
    FakeDeclaredBackend,
)
from tests.fakes.interruptible_backend import (
    SCRIPTED_INTERRUPT_NAME,
    FakeClock,
    ScriptedInterruptBackend,
    route,
)

LIMIT_REACHED = "WALL_CLOCK_LIMIT_REACHED"
UNSUPPORTED = "WALL_CLOCK_LIMIT_UNSUPPORTED"
CUT_SUFFIX = " The wall-clock limit cut the search short."
INFEASIBLE_CUT_PREFIX = (
    "No feasible solution found before solver.wall_clock_limit_seconds ran out"
)
ON = "repair_local_search"


def knapsack_problem() -> OptimizationProblem:
    """``examples/knapsack.json``: optimum {item_a, item_c}, value 17."""
    payload = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
    return OptimizationProblem.model_validate(payload)


def knapsack(*items: str) -> dict[str, int]:
    row = dict.fromkeys(["item_a", "item_b", "item_c", "item_d"], 0)
    for item in items:
        row[f"item_{item}"] = 1
    return row


FEASIBLE = [knapsack("a", "c"), knapsack("b", "d")]
INFEASIBLE = [knapsack("a", "b", "c", "d")]  # weight 18 > 10


def scripted_service(
    rows, *, script=None, supports_interrupt=True, clock=None, **policy
) -> tuple[ScriptedInterruptBackend, OptimizationService]:
    backend = ScriptedInterruptBackend(
        rows, supports_interrupt=supports_interrupt, script=script
    )
    service = OptimizationService(
        registry=SolverRegistry({backend.name: backend}),
        policy=ExecutionPolicy(**policy),
        clock=clock if clock is not None else FakeClock(),
    )
    return backend, service


def on_scripted(**preferences) -> OptimizationProblem:
    return route(knapsack_problem(), SCRIPTED_INTERRUPT_NAME, **preferences)


def codes(errors) -> list[str]:
    return [error.code for error in errors]


def the_warning(result):
    (warning,) = [w for w in result.warnings if w.code == LIMIT_REACHED]
    return warning


# --------------------------------------------------------------------------
# §9 item 1: who receives the interrupt
# --------------------------------------------------------------------------


class TestWhoReceivesTheInterrupt:
    def test_no_limit_and_no_token_passes_nothing(self):
        backend, service = scripted_service(FEASIBLE)

        result = service.solve(on_scripted())

        assert result.status == "success"
        assert backend.received_interrupt == [False]
        assert backend.calls == [{}]
        assert result.wall_clock_limit_reached is False
        assert LIMIT_REACHED not in codes(result.warnings)

    def test_a_limit_passes_an_interrupt_measured_from_entering_solve(self):
        clock = FakeClock(100.0)
        seen: list[tuple[bool, bool, bool]] = []

        def script(call, interrupt):
            before = interrupt.should_stop()
            clock.advance(4.999)
            almost = interrupt.deadline_passed()
            clock.advance(0.001)
            seen.append((before, almost, interrupt.deadline_passed()))
            clock.now = 100.0  # leave the rest of the solve inside the limit
            return "full"

        backend, service = scripted_service(FEASIBLE, script=script, clock=clock)

        result = service.solve(on_scripted(wall_clock_limit_seconds=5))

        assert backend.received_interrupt == [True]
        assert isinstance(backend.calls[0]["interrupt"], Interrupt)
        # The deadline is the service clock at entry (100) plus the limit.
        assert seen == [(False, False, True)]
        assert result.wall_clock_limit_reached is False

    def test_a_token_alone_passes_an_interrupt_without_a_deadline(self):
        token = CancelToken()
        clock = FakeClock()
        states: list[tuple[bool, bool]] = []

        def script(call, interrupt):
            clock.advance(1e9)
            states.append((interrupt.cancelled, interrupt.deadline_passed()))
            return "full"

        backend, service = scripted_service(FEASIBLE, script=script, clock=clock)

        result = service.solve(on_scripted(), cancel=token)

        assert result.status == "success"
        assert backend.received_interrupt == [True]
        assert states == [(False, False)]
        assert result.wall_clock_limit_reached is False

    def test_every_attempt_gets_the_same_interrupt(self):
        backend, service = scripted_service(INFEASIBLE)

        result = service.solve(
            on_scripted(wall_clock_limit_seconds=1000, max_retries=2),
            cancel=CancelToken(),
        )

        assert result.status == "infeasible"
        assert backend.received_interrupt == [True, True, True]
        interrupts = {id(call["interrupt"]) for call in backend.calls}
        assert len(interrupts) == 1
        assert result.wall_clock_limit_reached is False
        # The ladder ran out, not the clock: the ordinary message.
        assert not result.message.startswith(INFEASIBLE_CUT_PREFIX)

    def test_an_undeclared_backend_never_receives_it_even_with_a_token(self):
        backend, service = scripted_service(FEASIBLE, supports_interrupt=False)

        result = service.solve(on_scripted(), cancel=CancelToken())

        assert result.status == "success"
        assert backend.received_interrupt == [False]
        assert backend.calls == [{}]

    def test_an_undeclared_backend_with_a_limit_is_refused_before_it_runs(self):
        backend, service = scripted_service(FEASIBLE, supports_interrupt=False)

        result = service.solve(on_scripted(wall_clock_limit_seconds=5))

        assert result.status == "invalid_problem"
        assert codes(result.errors) == [UNSUPPORTED]
        assert result.errors[0].path == "solver.wall_clock_limit_seconds"
        assert backend.calls == []


# --------------------------------------------------------------------------
# §9 item 3: the deadline, deterministically
# --------------------------------------------------------------------------


class TestRetriesAfterTheDeadline:
    def test_no_retry_starts_once_the_deadline_has_passed(self):
        clock = FakeClock()

        def script(call, interrupt):
            clock.advance(20)  # a full, uninterrupted attempt 1 that overran
            return "full"

        backend, service = scripted_service(INFEASIBLE, script=script, clock=clock)

        result = service.solve(on_scripted(wall_clock_limit_seconds=10, max_retries=2))

        assert result.status == "infeasible"
        assert len(backend.calls) == 1
        (attempt,) = result.attempts
        assert attempt.samples_received == 1
        # Attempt 1 itself completed; what the limit cut was the ladder.
        assert attempt.wall_clock_limit_reached is False
        assert result.wall_clock_limit_reached is True
        warning = the_warning(result)
        assert "no further retry was started" in warning.message
        assert "stopped with work left" not in warning.message
        assert warning.path is None or "wall_clock" in warning.path
        assert result.message.startswith(INFEASIBLE_CUT_PREFIX + " (1 attempt(s))")
        # elapsed_ms is read from the same injected clock as the deadline.
        assert result.elapsed_ms == 20000.0

    def test_retries_continue_while_the_deadline_has_not_passed(self):
        clock = FakeClock()

        def script(call, interrupt):
            clock.advance(3)
            return "full"

        backend, service = scripted_service(INFEASIBLE, script=script, clock=clock)

        result = service.solve(on_scripted(wall_clock_limit_seconds=10, max_retries=2))

        assert len(backend.calls) == 3
        assert result.wall_clock_limit_reached is False
        assert LIMIT_REACHED not in codes(result.warnings)
        assert not result.message.startswith(INFEASIBLE_CUT_PREFIX)

    def test_the_deadline_passing_during_attempt_two_skips_attempt_three(self):
        clock = FakeClock()

        def script(call, interrupt):
            if call == 2:
                clock.advance(11)
                return "partial"
            return "full"

        backend, service = scripted_service(INFEASIBLE, script=script, clock=clock)

        result = service.solve(on_scripted(wall_clock_limit_seconds=10, max_retries=2))

        assert len(backend.calls) == 2
        assert [a.wall_clock_limit_reached for a in result.attempts] == [False, True]
        assert result.wall_clock_limit_reached is True
        warning = the_warning(result)
        assert "attempt 2 stopped with work left" in warning.message
        assert "no further retry was started" in warning.message
        assert result.message.startswith(INFEASIBLE_CUT_PREFIX + " (2 attempt(s))")

    def test_attempt_one_starts_even_after_the_deadline(self):
        # The deadline passes while attempt 1 compiles (the progress
        # callback runs as that stage starts); the backend is still called
        # and, following the contract, returns zero reads, interrupted.
        clock = FakeClock()

        def on_progress(progress):
            if progress.attempt == 1 and progress.stage == "compile":
                clock.advance(100)

        def script(call, interrupt):
            return "empty" if interrupt.should_stop() else "full"

        backend, service = scripted_service(FEASIBLE, script=script, clock=clock)

        result = service.solve(
            on_scripted(wall_clock_limit_seconds=10, max_retries=2),
            on_progress=on_progress,
        )

        assert len(backend.calls) == 1
        assert result.status == "infeasible"
        assert result.solutions == []
        (attempt,) = result.attempts
        assert attempt.samples_received == 0
        assert attempt.unique_samples == 0
        assert attempt.wall_clock_limit_reached is True
        assert result.wall_clock_limit_reached is True
        assert result.infeasibility_proven is False
        warning = the_warning(result)
        assert "attempt 1 stopped with work left" in warning.message
        assert "no further retry was started" in warning.message
        assert result.message.startswith(INFEASIBLE_CUT_PREFIX + " (1 attempt(s))")


class TestOnlyTheUninterruptibleStagesRanOver:
    """Running past the limit without skipping any work sets no flag."""

    def test_success_after_the_deadline_is_not_marked(self):
        clock = FakeClock()

        def script(call, interrupt):
            clock.advance(20)  # validation then starts after the deadline
            return "full"

        _, service = scripted_service(FEASIBLE, script=script, clock=clock)
        _, reference = scripted_service(FEASIBLE)

        result = service.solve(on_scripted(wall_clock_limit_seconds=10))
        plain = reference.solve(on_scripted())

        assert result.status == "success"
        assert result.elapsed_ms == 20000.0  # the limit was overrun...
        assert result.wall_clock_limit_reached is False  # ...but nothing was cut
        assert result.attempts[0].wall_clock_limit_reached is False
        assert LIMIT_REACHED not in codes(result.warnings)
        assert result.message == plain.message
        assert not result.message.endswith(CUT_SUFFIX)

    def test_infeasible_without_a_retry_left_is_not_marked(self):
        clock = FakeClock()

        def script(call, interrupt):
            clock.advance(20)
            return "full"

        _, service = scripted_service(INFEASIBLE, script=script, clock=clock)
        _, reference = scripted_service(INFEASIBLE)

        result = service.solve(on_scripted(wall_clock_limit_seconds=10, max_retries=0))
        plain = reference.solve(on_scripted(max_retries=0))

        assert result.status == "infeasible"
        assert result.wall_clock_limit_reached is False
        assert LIMIT_REACHED not in codes(result.warnings)
        assert result.message == plain.message


class TestACutResult:
    def test_success_from_a_cut_attempt_says_so_three_ways(self):
        backend, service = scripted_service(
            FEASIBLE, script=lambda call, interrupt: "partial"
        )
        _, reference = scripted_service(FEASIBLE)

        result = service.solve(on_scripted(wall_clock_limit_seconds=10))
        plain = reference.solve(on_scripted())

        assert result.status == "success"
        assert result.solutions == plain.solutions
        assert result.wall_clock_limit_reached is True
        assert result.attempts[0].wall_clock_limit_reached is True
        assert result.message == plain.message + CUT_SUFFIX
        warning = the_warning(result)
        assert "attempt 1 stopped with work left" in warning.message
        assert "no further retry was started" not in warning.message
        assert "may differ between runs" in warning.message
        assert warning.recommended_action

    def test_postprocessing_cut_by_the_limit_is_not_a_ceiling(self):
        clock = FakeClock()

        def script(call, interrupt):
            clock.advance(20)  # the deadline passes before post-processing
            return "full"

        _, service = scripted_service(FEASIBLE, script=script, clock=clock)

        result = service.solve(
            on_scripted(wall_clock_limit_seconds=10, postprocess=ON)
        )

        assert result.status == "success"
        (attempt,) = result.attempts
        stats = attempt.postprocess
        assert stats is not None
        assert stats.wall_clock_limit_reached is True
        assert stats.limit_reached == []
        assert stats.candidates_selected == 0  # stopped before the first start
        assert attempt.wall_clock_limit_reached is True
        assert result.wall_clock_limit_reached is True
        assert "POSTPROCESS_LIMIT_REACHED" not in codes(result.warnings)
        assert codes(result.warnings).count(LIMIT_REACHED) == 1
        assert result.message.endswith(CUT_SUFFIX)

    def test_postprocessing_the_limit_never_stops_is_unchanged(self):
        _, limited = scripted_service(FEASIBLE)
        _, reference = scripted_service(FEASIBLE)

        result = limited.solve(
            on_scripted(wall_clock_limit_seconds=1000, postprocess=ON),
            cancel=CancelToken(),
        )
        plain = reference.solve(on_scripted(postprocess=ON))

        assert result.attempts[0].postprocess == plain.attempts[0].postprocess
        assert result.attempts[0].postprocess.wall_clock_limit_reached is False
        assert result.solutions == plain.solutions
        assert result.message == plain.message
        assert result.wall_clock_limit_reached is False


# --------------------------------------------------------------------------
# §9 items 3 and 5: cancellation
# --------------------------------------------------------------------------


class TestCancellation:
    def test_an_already_cancelled_token_stops_at_the_first_checkpoint(self):
        token = CancelToken()
        token.cancel()
        backend, service = scripted_service(FEASIBLE)

        with pytest.raises(SolveCancelled):
            service.solve(on_scripted(), cancel=token)
        assert backend.calls == []

    def test_cancelled_while_compiling_never_reaches_the_backend(self):
        token = CancelToken()

        def on_progress(progress):
            if progress.stage == "compile":
                token.cancel()

        backend, service = scripted_service(FEASIBLE)

        with pytest.raises(SolveCancelled):
            service.solve(on_scripted(), cancel=token, on_progress=on_progress)
        assert backend.calls == []

    def test_cancelled_during_the_backend_raises_after_it_returns(self):
        token = CancelToken()

        def script(call, interrupt):
            token.cancel()
            return "full"

        backend, service = scripted_service(FEASIBLE, script=script)

        with pytest.raises(SolveCancelled):
            service.solve(on_scripted(), cancel=token)
        assert len(backend.calls) == 1

    def test_cancelled_while_validating_raises_after_the_candidates(self):
        token = CancelToken()

        def on_progress(progress):
            if progress.stage == "validate":
                token.cancel()

        backend, service = scripted_service(FEASIBLE)

        with pytest.raises(SolveCancelled):
            service.solve(
                on_scripted(postprocess=ON), cancel=token, on_progress=on_progress
            )
        assert len(backend.calls) == 1

    def test_cancellation_wins_over_the_deadline(self):
        token = CancelToken()
        clock = FakeClock()

        def script(call, interrupt):
            clock.advance(100)
            token.cancel()
            return "partial"

        _, service = scripted_service(FEASIBLE, script=script, clock=clock)

        with pytest.raises(SolveCancelled):
            service.solve(on_scripted(wall_clock_limit_seconds=10), cancel=token)

    def test_cancelled_on_a_retry_returns_no_partial_result(self):
        token = CancelToken()

        def script(call, interrupt):
            if call == 2:
                token.cancel()
            return "full"

        backend, service = scripted_service(INFEASIBLE, script=script)

        with pytest.raises(SolveCancelled):
            service.solve(on_scripted(max_retries=3), cancel=token)
        assert len(backend.calls) == 2

    def test_an_undeclared_backend_is_cancelled_at_the_service_checkpoints(self):
        token = CancelToken()

        def script(call, interrupt):
            token.cancel()
            return "full"

        backend, service = scripted_service(
            INFEASIBLE, script=script, supports_interrupt=False
        )

        with pytest.raises(SolveCancelled):
            service.solve(on_scripted(max_retries=3), cancel=token)
        # It never saw the interrupt, ran to its end, and no retry started.
        assert backend.calls == [{}]

    def test_the_concurrency_slot_is_released(self):
        token = CancelToken()
        token.cancel()
        backend, service = scripted_service(FEASIBLE, max_concurrent_solves=1)

        with pytest.raises(SolveCancelled):
            service.solve(on_scripted(), cancel=token)
        # With one slot, a leaked one would make both of these
        # CONCURRENCY_LIMIT.
        first = service.solve(on_scripted())
        second = service.solve(on_scripted(), cancel=CancelToken())

        assert first.status == second.status == "success"

    def test_a_cancelled_solve_mid_run_also_releases_the_slot(self):
        token = CancelToken()

        def script(call, interrupt):
            if call == 1:
                token.cancel()
            return "full"

        _, service = scripted_service(FEASIBLE, script=script, max_concurrent_solves=1)

        with pytest.raises(SolveCancelled):
            service.solve(on_scripted(), cancel=token)
        assert service.solve(on_scripted()).status == "success"

    def test_a_validation_error_is_returned_even_when_cancelled(self):
        token = CancelToken()
        token.cancel()
        backend, service = scripted_service(FEASIBLE)

        result = service.solve(on_scripted(wall_clock_limit_seconds=-1), cancel=token)

        assert result.status == "invalid_problem"
        assert codes(result.errors) == ["INVALID_SOLVER_PREFERENCE"]
        assert backend.calls == []

    def test_a_gate_failure_is_returned_even_when_cancelled(self):
        # A read count over the policy ceiling is refused in dispatch,
        # before the attempts (and their checkpoints) begin.
        token = CancelToken()
        token.cancel()
        service = OptimizationService(policy=ExecutionPolicy(max_local_reads=10))
        problem = knapsack_problem().model_copy(
            update={
                "solver": SolverPreferences(
                    backend="simulated_annealing", num_reads=11
                )
            }
        )

        result = service.solve(problem, cancel=token)

        assert result.status == "resource_limit_exceeded"
        assert codes(result.errors) == ["LOCAL_READS_LIMIT"]


# --------------------------------------------------------------------------
# §9 item 6 (validator part): an unsupported backend and invalid values
# --------------------------------------------------------------------------


def with_limit(backend: str, limit) -> OptimizationProblem:
    return knapsack_problem().model_copy(
        update={
            "solver": SolverPreferences.model_construct(
                backend=backend, wall_clock_limit_seconds=limit
            )
        }
    )


def declared_service() -> tuple[FakeDeclaredBackend, OptimizationService]:
    """The default backends plus the remote-looking fake, remote allowed so
    only the wall-clock rule can block it."""
    defaults = SolverRegistry.default()
    backends = {name: defaults.get(name) for name in defaults.names()}
    fake = FakeDeclaredBackend()
    backends[FAKE_DECLARED_NAME] = fake
    service = OptimizationService(
        registry=SolverRegistry(backends),
        policy=ExecutionPolicy(allow_remote=True, limits={FAKE_LIMIT_KEY: 1000}),
    )
    return fake, service


class TestUnsupportedBackend:
    @pytest.mark.parametrize("backend_name", ["exact", FAKE_DECLARED_NAME])
    def test_validate_refuses_it(self, backend_name):
        _, service = declared_service()

        result = service.validate(with_limit(backend_name, 5))

        assert result.valid is False
        assert codes(result.errors) == [UNSUPPORTED]
        error = result.errors[0]
        assert error.path == "solver.wall_clock_limit_seconds"
        assert backend_name in error.message
        assert error.recommended_action

    @pytest.mark.parametrize("backend_name", ["exact", FAKE_DECLARED_NAME])
    def test_solve_refuses_it_before_the_backend_runs(self, backend_name):
        fake, service = declared_service()

        result = service.solve(with_limit(backend_name, 5))

        assert result.status == "invalid_problem"
        assert codes(result.errors) == [UNSUPPORTED]
        assert fake.solve_calls == 0

    def test_without_a_limit_they_are_accepted(self):
        fake, service = declared_service()

        for backend_name in ("exact", FAKE_DECLARED_NAME):
            assert service.validate(with_limit(backend_name, None)).valid is True
        assert service.solve(with_limit(FAKE_DECLARED_NAME, None)).status == "success"
        assert fake.solve_calls == 1

    @pytest.mark.parametrize(
        "backend_name", ["simulated_annealing", "tabu", "simulated_bifurcation"]
    )
    def test_a_declaring_backend_is_accepted(self, backend_name):
        _, service = declared_service()

        result = service.validate(with_limit(backend_name, 5))

        assert result.valid is True, result.errors
        assert UNSUPPORTED not in codes(result.warnings)

    def test_recommend_blocks_exactly_the_backends_that_cannot_stop(self):
        _, service = declared_service()

        result = service.recommend(with_limit("simulated_annealing", 5))

        assert result.valid is True
        by_name = {entry.backend: entry for entry in result.recommendations}
        for name in ("exact", FAKE_DECLARED_NAME):
            entry = by_name[name]
            assert entry.usable is False
            assert codes(entry.blocking) == [UNSUPPORTED]
            assert "R_UNUSABLE" in entry.reasons
        for name in ("simulated_annealing", "tabu", "simulated_bifurcation"):
            entry = by_name[name]
            assert entry.usable is True, entry.blocking
            assert UNSUPPORTED not in codes(entry.blocking)
        # The unusable ones sort behind every usable one.
        ranks = [entry.usable for entry in result.recommendations]
        assert ranks == sorted(ranks, reverse=True)

    def test_recommend_without_a_limit_does_not_block_them(self):
        _, service = declared_service()

        result = service.recommend(with_limit("simulated_annealing", None))

        by_name = {entry.backend: entry for entry in result.recommendations}
        for name in ("exact", FAKE_DECLARED_NAME):
            assert UNSUPPORTED not in codes(by_name[name].blocking)
            assert by_name[name].usable is True


class TestInvalidLimitValues:
    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_the_schema_refuses_non_finite_values(self, value):
        with pytest.raises(ValidationError, match="wall_clock_limit_seconds"):
            SolverPreferences(wall_clock_limit_seconds=value)

    def test_the_schema_refuses_them_in_a_document_too(self):
        payload = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
        payload["solver"] = {"backend": "simulated_annealing", "wall_clock_limit_seconds": "inf"}
        with pytest.raises(ValidationError, match="wall_clock_limit_seconds"):
            OptimizationProblem.model_validate(payload)

    @pytest.mark.parametrize("value", [0, 0.0, -1, -0.5])
    def test_zero_and_negative_are_invalid_preferences(self, value):
        problem = with_limit("simulated_annealing", value)
        # The schema lets them through; the validator names the rule.
        SolverPreferences(wall_clock_limit_seconds=value)

        errors = validate_problem(problem)
        result = OptimizationService().solve(problem)

        assert [(e.code, e.path) for e in errors] == [
            ("INVALID_SOLVER_PREFERENCE", "solver.wall_clock_limit_seconds")
        ]
        assert errors[0].message == (
            f"solver.wall_clock_limit_seconds must be a finite number > 0, got {value}"
        )
        assert result.status == "invalid_problem"
        assert codes(result.errors) == ["INVALID_SOLVER_PREFERENCE"]

    @pytest.mark.parametrize("value", [float("inf"), float("nan")])
    def test_a_model_built_without_validation_is_still_refused(self, value):
        errors = validate_problem(with_limit("simulated_annealing", value))

        assert codes(errors) == ["INVALID_SOLVER_PREFERENCE"]

    def test_a_small_positive_limit_is_valid(self):
        assert validate_problem(with_limit("simulated_annealing", 0.001)) == []
        assert SolverPreferences().wall_clock_limit_seconds is None
