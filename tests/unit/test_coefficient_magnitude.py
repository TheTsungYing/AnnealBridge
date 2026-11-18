"""COEFFICIENT_MAGNITUDE_TOO_LARGE and the zero-attempt PENALTY_OVERFLOW wording.

2026-11-18 review R-11. A problem whose every number is finite can still
span a range float64 cannot hold. Before this change the service reported
that as PENALTY_OVERFLOW "after 0 attempt(s); no feasible solution was
found", for a problem that may have no hard constraint at all. Now:

* the validator rejects a non-finite objective scale (any path) and a
  non-finite penalty scale (BQM path) as ``invalid_problem``;
* PENALTY_OVERFLOW keeps meaning "the hard penalty left the float range",
  and a solve that never completed an attempt is worded as such;
* the retry ladder's own overflow is unchanged
  (``tests/unit/test_parameter_ceilings.py``).
"""

import math

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.penalty.strategy import ScaledPenaltyStrategy
from annealbridge.validation import validate_problem, validate_problem_full
from annealbridge.validation.estimates import (
    compute_objective_scale,
    compute_penalty_scale,
    compute_soft_energy_total,
    variable_bounds,
)
from tests.builders import hard, lin, soft

CODE = "COEFFICIENT_MAGNITUDE_TOO_LARGE"


def problem(coefficients, constraints=(), *, backend="exact", **solver):
    return OptimizationProblem.model_validate(
        {
            "version": "1.0",
            "name": "magnitude",
            "variables": [{"name": f"x{i}"} for i in range(len(coefficients))],
            "objective": {
                "direction": "maximize",
                "linear_terms": [lin(f"x{i}", c) for i, c in enumerate(coefficients)],
            },
            "constraints": list(constraints),
            "solver": {"backend": backend, **solver},
        }
    )


def huge_objective(**kwargs):
    """The review's problem: two binaries, both coefficients 1e308, no constraint."""
    return problem([1e308, 1e308], **kwargs)


def huge_soft(**kwargs):
    """Finite numbers whose soft energy bound ``1e300 * (1e10)**2`` is not."""
    return problem(
        [1.0, 1.0], [soft("s", "==", 1e10, [lin("x0", 1e10)], weight=1e300)], **kwargs
    )


def error_codes(result) -> list[str]:
    return [error.code for error in result.errors]


class TestValidator:
    @pytest.mark.parametrize("model_type", ["bqm", "cqm"])
    def test_an_overflowing_objective_is_rejected_on_every_path(self, model_type):
        result = validate_problem_full(huge_objective(), model_type=model_type)

        assert result.valid is False
        assert error_codes(result) == [CODE]
        error = result.errors[0]
        assert error.path == "objective"
        assert "objective" in error.message
        assert error.retryable is False
        assert error.recommended_action
        assert result.warnings == []
        assert result.objective_scale is None

    def test_an_overflowing_soft_bound_is_rejected_on_the_bqm_path(self):
        result = validate_problem_full(huge_soft(), model_type="bqm")

        assert result.valid is False
        assert error_codes(result) == [CODE]
        assert result.errors[0].path == "constraints"
        assert "soft" in result.errors[0].message

    def test_the_cqm_path_derives_no_penalty_scale(self):
        # Its compiler reports an overflowing soft term itself
        # (test_service_cqm_flow.TestNonFiniteModelNeverReachesTheBackend).
        assert validate_problem_full(huge_soft(), model_type="cqm").valid is True

    def test_one_error_when_both_scales_overflow(self):
        both = problem(
            [1e308, 1e308], [soft("s", "==", 1e10, [lin("x0", 1e10)], weight=1e300)]
        )
        result = validate_problem_full(both, model_type="bqm")
        assert [(e.code, e.path) for e in result.errors] == [(CODE, "objective")]

    @pytest.mark.parametrize("coefficients", [[1e200, 1e200], [8e307, 8e307], [1.0, 2.0]])
    def test_a_finite_scale_is_accepted(self, coefficients):
        result = validate_problem_full(problem(coefficients), model_type="bqm")
        assert result.valid is True
        assert math.isfinite(result.objective_scale)

    def test_earlier_errors_are_reported_alone(self):
        # The scales are only computed once the error pass is clean.
        broken = problem([1e308, 1e308], [hard("h", "==", 1, [lin("missing", 1)])])
        result = validate_problem_full(broken, model_type="bqm")
        assert error_codes(result) == ["UNKNOWN_VARIABLE"]

    def test_the_objective_verdict_belongs_to_the_error_pass(self):
        # Backend-independent, so the errors-only entry point reports it.
        assert [e.code for e in validate_problem(huge_objective())] == [CODE]

    def test_the_soft_verdict_depends_on_the_path_and_is_not_in_the_error_pass(self):
        assert validate_problem(huge_soft()) == []


class TestSharedArithmetic:
    def test_penalty_scale_is_objective_scale_plus_the_soft_total(self):
        subject = problem(
            [3.0, 0.1, 7.5],
            [
                soft("a", "==", 1, [lin("x0", 1), lin("x1", 1)], weight=0.3),
                soft("b", "<=", 1, [lin("x1", 1), lin("x2", 1)], weight=1.7),
                hard("c", ">=", 1, [lin("x0", 1), lin("x2", 1)]),
            ],
        )
        bounds = variable_bounds(subject)
        expected = compute_objective_scale(
            subject.objective, bounds
        ) + compute_soft_energy_total(subject, bounds)

        # Bit for bit: the validator's check and the penalty strategy must
        # judge the very same float.
        assert compute_penalty_scale(subject) == expected
        assert ScaledPenaltyStrategy().penalty_scale(subject) == expected

    def test_hard_constraints_add_nothing_to_the_soft_total(self):
        subject = problem([1.0, 1.0], [hard("h", "==", 1, [lin("x0", 1), lin("x1", 1)])])
        assert compute_soft_energy_total(subject, variable_bounds(subject)) == 0


class TestService:
    def test_the_review_problem_is_an_invalid_problem(self):
        service = OptimizationService()

        validation = service.validate(huge_objective())
        assert validation.valid is False
        assert error_codes(validation) == [CODE]

        result = service.solve(huge_objective())
        assert result.status == "invalid_problem"
        assert error_codes(result) == [CODE]
        assert result.attempts == []
        assert result.solutions == []
        assert "penalty" not in result.message.lower()
        assert "feasible" not in result.message

    def test_recommend_reports_it_too(self):
        result = OptimizationService().recommend(huge_objective())
        assert result.valid is False
        assert error_codes(result) == [CODE]

    def test_recommend_blocks_only_the_bqm_backends_for_a_soft_overflow(self):
        result = OptimizationService().recommend(huge_soft())
        assert result.valid is True
        for entry in result.recommendations:
            blocked = CODE in [error.code for error in entry.blocking]
            assert blocked is (entry.model_type == "bqm"), entry.backend
            if blocked:
                assert entry.usable is False

    def test_a_heuristic_bqm_backend_gets_the_same_verdict(self):
        result = OptimizationService().solve(huge_soft(backend="simulated_annealing"))
        assert result.status == "invalid_problem"
        assert error_codes(result) == [CODE]

    def test_a_multiplier_that_overflows_a_finite_scale_is_a_penalty_overflow(self):
        # Scale 1.6e308 is finite; the default multiplier takes it past the
        # range. The hard penalty is what overflowed, so the code stays, but
        # nothing was attempted and the message must not claim otherwise.
        subject = problem([8e307, 8e307])
        assert OptimizationService().validate(subject).valid is True

        result = OptimizationService().solve(subject)
        assert result.status == "resource_limit_exceeded"
        assert error_codes(result) == ["PENALTY_OVERFLOW"]
        assert result.attempts == []
        message = result.errors[0].message
        assert message.startswith("Initial hard penalty inf")
        assert "solver.penalty_multiplier" in message
        assert "attempt(s)" not in message
        assert "feasible" not in message

    def test_a_first_compile_that_overflows_is_worded_for_zero_attempts(self):
        # Penalty 4.0 is finite; 4.0 * (1e160)**2 is not.
        subject = problem([1.0, 1.0], [hard("h", "==", 1e160, [lin("x0", 1e160)])])
        assert OptimizationService().validate(subject).valid is True

        result = OptimizationService().solve(subject)
        assert result.status == "resource_limit_exceeded"
        assert error_codes(result) == ["PENALTY_OVERFLOW"]
        assert result.attempts == []
        message = result.errors[0].message
        assert "initial hard penalty 4.0" in message
        assert "non-finite bias" in message
        assert "attempt(s)" not in message
        assert "feasible" not in message


class InfiniteScaleStrategy(ScaledPenaltyStrategy):
    """A strategy whose scale the validator cannot know about."""

    def penalty_scale(self, problem):
        return math.inf


class TestServiceDefence:
    def test_a_strategy_with_a_non_finite_scale_is_a_problem_side_error(self):
        service = OptimizationService(penalty_strategy=InfiniteScaleStrategy())
        subject = problem([1.0, 2.0])
        assert service.validate(subject).valid is True

        result = service.solve(subject)
        assert result.status == "invalid_problem"
        assert error_codes(result) == [CODE]
        assert result.attempts == []
        assert "penalty scale" in result.errors[0].message
