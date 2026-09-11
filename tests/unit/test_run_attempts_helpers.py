"""``_run_attempts``'s helpers, extracted from it in the 2026-09-11 batch 5
refactor: ``_infeasible_message`` is a pure function and ``_max_attempts``
now returns an ``_AttemptBudget``, so each wording and each budget cut can
be pinned directly instead of through a whole solve.
"""

import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import SolverPreferences
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.orchestration.optimizer import _infeasible_message
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
