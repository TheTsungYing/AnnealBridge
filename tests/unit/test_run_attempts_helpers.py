"""``_run_attempts``'s helpers, extracted from it in the 2026-09-11 batch 5
refactor: ``_infeasible_message`` is a pure function and ``_max_attempts``
now returns an ``_AttemptBudget``, so each wording and each budget cut can
be pinned directly instead of through a whole solve.

``_success_message`` is pure for the same reason: its wording is pinned
here word for word, without a backend, a compiler and a policy behind it.
"""

import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import Solution, SolveAttempt, SolverPreferences
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.orchestration.optimizer import (
    _exact_number,
    _infeasible_message,
    _success_message,
)
from annealbridge.solvers import SolverRegistry
from tests.fakes.declared_backend import (
    FAKE_DECLARED_NAME,
    FAKE_LIMIT_KEY,
    FakeDeclaredBackend,
)

PROVEN = (
    "No feasible solution exists: the exhaustive backend enumerated every assignment"
)
EXHAUSTIVE_NO_SAMPLES = (
    "No feasible solution found: the exhaustive backend returned no samples, "
    "so infeasibility is not proven"
)
NO_HARD_PENALTY = (
    "No feasible solution found: the constraint-model backend returned no sample "
    "satisfying the hard constraints under independent validation; infeasibility "
    "is not proven"
)
REMOTE_BLOCKED = (
    "No feasible solution found in 1 attempt; server policy disables automatic "
    "remote retries"
)
REMOTE_BLOCKED_WARNING = (
    "1 attempt was made; server policy disables automatic remote retries to "
    "protect quota"
)


class TestInfeasibleMessage:
    @pytest.mark.parametrize(
        "cut_reason",
        [None, "exhaustive", "no_hard_penalty", "remote_retries_disabled"],
    )
    def test_proven_wins_over_every_cut_reason(self, cut_reason):
        message, warnings = _infeasible_message(True, cut_reason, 1)

        assert message == PROVEN
        assert warnings == []

    def test_exhaustive_without_samples_is_not_proven(self):
        message, warnings = _infeasible_message(False, "exhaustive", 1)

        assert message == EXHAUSTIVE_NO_SAMPLES
        assert warnings == []

    def test_no_hard_penalty_has_nothing_to_retune(self):
        message, warnings = _infeasible_message(False, "no_hard_penalty", 1)

        assert message == NO_HARD_PENALTY
        assert warnings == []

    def test_remote_retries_disabled_warns(self):
        message, warnings = _infeasible_message(False, "remote_retries_disabled", 1)

        assert message == REMOTE_BLOCKED
        (warning,) = warnings
        assert warning.code == "REMOTE_RETRIES_DISABLED"
        assert warning.message == REMOTE_BLOCKED_WARNING

    def test_uncut_budget_reports_the_attempts_actually_made(self):
        message, warnings = _infeasible_message(False, None, 3)

        assert message == (
            "No feasible solution found in 3 attempt(s); the problem may still be "
            "feasible under a different solver configuration"
        )
        assert "in 3 attempt(s)" in message
        assert warnings == []


def solution(objective: float, soft: float = 0.0, rank: int = 1) -> Solution:
    """One ranked solution; only the fields the message reads matter here."""
    return Solution(
        rank=rank,
        variables={"x1": 1},
        objective_value=objective,
        soft_violation_score=soft,
        ranking_score=objective - soft,
        energy=None,
        sample_count=1,
        hard_constraints_satisfied=True,
        constraint_evaluations=[],
    )


def solutions(objective: float, soft: float = 0.0, count: int = 5) -> list[Solution]:
    """``count`` solutions whose rank 1 carries the given objective."""
    return [solution(objective, soft, rank) for rank in range(1, count + 1)]


def attempt(number: int = 1, unique: int = 16, feasible: int = 10) -> SolveAttempt:
    """One recorded attempt; the timing and size fields keep their defaults."""
    return SolveAttempt(
        attempt=number,
        penalty=62.0,
        samples_received=256,
        unique_samples=unique,
        feasible_samples=feasible,
    )


class TestExactNumber:
    def test_integral_value_drops_the_trailing_zero(self):
        assert _exact_number(17.0) == "17"

    def test_fractional_value_keeps_its_digits(self):
        assert _exact_number(17.5) == "17.5"

    def test_large_fractional_value_is_not_rounded(self):
        # The CLI's ``:g`` would print 1.23457e+06; a quoted message must not.
        assert _exact_number(1234567.5) == "1234567.5"


class TestSuccessMessage:
    def test_proven_on_the_first_attempt(self):
        message = _success_message(
            "exact", "maximize", True, attempt(), solutions(17.0)
        )

        assert message == (
            "exact proved optimality: rank 1 has objective 17 (maximize); "
            "10 of 16 distinct candidates were feasible, 5 returned."
        )

    def test_not_proven_says_so_explicitly(self):
        message = _success_message(
            "simulated_annealing", "maximize", False, attempt(), solutions(17.0)
        )

        assert message == (
            "simulated_annealing found 16 distinct candidates (10 feasible), "
            "5 returned: rank 1 has objective 17 (maximize); optimality is "
            "not proven."
        )

    def test_not_proven_names_the_attempt_it_succeeded_on(self):
        # Only the heuristic wording can carry an attempt number: an
        # exhaustive backend, the only kind that proves optimality, is given
        # a single attempt by ``_max_attempts``.
        message = _success_message(
            "simulated_annealing", "maximize", False, attempt(2), solutions(17.0)
        )

        assert message == (
            "simulated_annealing found 16 distinct candidates (10 feasible), "
            "5 returned on attempt 2: rank 1 has objective 17 (maximize); "
            "optimality is not proven."
        )

    def test_soft_violation_is_reported_beside_the_objective(self):
        message = _success_message(
            "exact", "maximize", True, attempt(), solutions(17.0, 2.5)
        )

        assert message == (
            "exact proved optimality: rank 1 has objective 17 with soft "
            "violation 2.5 (maximize); 10 of 16 distinct candidates were "
            "feasible, 5 returned."
        )

    def test_fractional_objective_keeps_its_digits(self):
        message = _success_message(
            "exact", "maximize", True, attempt(), solutions(17.5)
        )

        assert message == (
            "exact proved optimality: rank 1 has objective 17.5 (maximize); "
            "10 of 16 distinct candidates were feasible, 5 returned."
        )

    def test_minimize_direction_is_echoed(self):
        message = _success_message(
            "simulated_annealing", "minimize", False, attempt(), solutions(17.0)
        )

        assert message == (
            "simulated_annealing found 16 distinct candidates (10 feasible), "
            "5 returned: rank 1 has objective 17 (minimize); optimality is "
            "not proven."
        )


def make_service(**policy_overrides) -> OptimizationService:
    policy = ExecutionPolicy(
        allow_remote=True, limits={FAKE_LIMIT_KEY: 1000}, **policy_overrides
    )
    return OptimizationService(
        registry=SolverRegistry({FAKE_DECLARED_NAME: FakeDeclaredBackend()}),
        policy=policy,
    )


def prefs(max_retries: int) -> SolverPreferences:
    return SolverPreferences(max_retries=max_retries)


class TestMaxAttempts:
    def test_exhaustive_backend_never_retries(self):
        service = make_service()
        backend = SolverRegistry.default().get("exact")

        assert service._max_attempts(backend, BQMCompiler(), prefs(2)) == (
            1,
            "exhaustive",
        )

    def test_native_constraint_model_has_no_lever_to_turn(self):
        service = make_service()
        backend = SolverRegistry.default().get("simulated_annealing")

        assert service._max_attempts(backend, CQMCompiler(), prefs(2)) == (
            1,
            "no_hard_penalty",
        )

    def test_remote_retries_blocked_by_policy_is_a_cut(self):
        service = make_service(allow_remote_retries=False)
        backend = FakeDeclaredBackend()

        assert service._max_attempts(backend, BQMCompiler(), prefs(2)) == (
            1,
            "remote_retries_disabled",
        )

    def test_no_cut_when_the_user_asked_for_no_retry(self):
        service = make_service(allow_remote_retries=False)
        backend = FakeDeclaredBackend()

        assert service._max_attempts(backend, BQMCompiler(), prefs(0)) == (1, None)

    def test_local_hard_penalty_backend_uses_the_full_ladder(self):
        service = make_service()
        backend = SolverRegistry.default().get("simulated_annealing")

        assert service._max_attempts(backend, BQMCompiler(), prefs(2)) == (3, None)

    def test_remote_ladder_when_policy_allows_retries(self):
        service = make_service(allow_remote_retries=True)
        backend = FakeDeclaredBackend()

        assert service._max_attempts(backend, BQMCompiler(), prefs(2)) == (3, None)
