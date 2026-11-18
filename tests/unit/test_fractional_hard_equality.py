"""FRACTIONAL_HARD_EQUALITY: a hard equality a BQM penalty may not enforce.

2026-11-18 review R-01, stage A. The hard penalty is derived for violations
of at least one unit; a hard ``==`` with fractional coefficients or rhs can
be violated by less, and then a heuristic backend on the BQM path may return
no feasible sample at all. ``validate`` now says so. Nothing about
compilation changes (the golden recordings hold that), so these tests pin
the judgement and where the warning is and is not raised.
"""

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.penalty.strategy import ScaledPenaltyStrategy
from annealbridge.validation import validate_problem_full
from annealbridge.validation.estimates import equality_violation_step_below_one
from tests.builders import hard, lin, soft

CODE = "FRACTIONAL_HARD_EQUALITY"


def equality(rhs: float, *coefficients: float | tuple[str, float]):
    """A hard ``==`` over ``x0, x1, ...`` (or over the named variables)."""
    terms = [
        lin(*item) if isinstance(item, tuple) else lin(f"x{index}", item)
        for index, item in enumerate(coefficients)
    ]
    document = {
        "name": "fractional",
        "variables": [{"name": name} for name in dict.fromkeys(t["variable"] for t in terms)],
        "objective": {"direction": "maximize", "linear_terms": []},
        "constraints": [hard("eq", "==", rhs, terms)],
    }
    return OptimizationProblem.model_validate(document).constraints[0]


def pick_three(scale: float, *, backend: str = "simulated_annealing", **solver):
    """Review R-01's problem: maximize ``sum((10+i) x_i)``, exactly three chosen.

    ``scale`` multiplies both sides of the equality, so every ``scale``
    describes the same feasible set.
    """
    count = 30
    return OptimizationProblem.model_validate(
        {
            "version": "1.0",
            "name": "pick three",
            "variables": [{"name": f"x{i}"} for i in range(count)],
            "objective": {
                "direction": "maximize",
                "linear_terms": [lin(f"x{i}", 10 + i) for i in range(count)],
            },
            "constraints": [
                hard("pick3", "==", 3 * scale, [lin(f"x{i}", scale) for i in range(count)])
            ],
            "solver": {"backend": backend, **solver},
        }
    )


def warning_codes(result) -> list[str]:
    return [warning.code for warning in result.warnings]


class TestViolationStep:
    """``equality_violation_step_below_one``: gcd of coefficients and rhs < 1."""

    @pytest.mark.parametrize(
        ("rhs", "coefficients"),
        [
            (3.0, (1.0, 1.0, 1.0, 1.0)),
            (0.0, (1.0, -1.0)),
            (6.0, (2.0, 4.0)),
            # Fractions whose common step is still at least one unit.
            (5.0, (2.5, 2.5)),
            (4.5, (1.5, 3.0)),
            # The rhs is off the lattice, but every violation is >= 2.
            (2.0, (4.0, 4.0)),
            # Repeated variables accumulate before the judgement.
            (1.0, (("x", 0.5), ("x", 0.5), ("y", 1.0))),
        ],
    )
    def test_a_step_of_at_least_one_is_not_flagged(self, rhs, coefficients):
        assert equality_violation_step_below_one(equality(rhs, *coefficients)) is False

    @pytest.mark.parametrize(
        ("rhs", "coefficients"),
        [
            (0.03, (0.01, 0.01, 0.01, 0.01)),
            (0.3, (0.1, 0.1, 0.1)),
            (1.5, (0.5, 0.5, 0.5)),
            # Integer coefficients, fractional rhs.
            (1.5, (1.0, 1.0)),
            # One fractional coefficient among integers.
            (2.0, (1.0, 1.0, 0.25)),
            (1e-300, (1e-300, 1e-300)),
            # Conservative: no 0/1 assignment is off by less than 1.5 here,
            # but the common step is 0.5 and the test is a sufficient one.
            (4.0, (1.5, 2.5)),
        ],
    )
    def test_a_step_below_one_is_flagged(self, rhs, coefficients):
        assert equality_violation_step_below_one(equality(rhs, *coefficients)) is True

    def test_coefficients_that_cancel_leave_a_constant_constraint(self):
        constraint = equality(0.0, ("x", 0.5), ("x", -0.5))
        assert equality_violation_step_below_one(constraint) is False


class TestValidatorWarning:
    def test_the_review_problem_gets_the_warning(self):
        result = validate_problem_full(pick_three(0.01), model_type="bqm")

        assert result.valid is True
        assert warning_codes(result) == [CODE]
        warning = result.warnings[0]
        assert warning.path == "constraints[0]"
        assert "pick3" in warning.message
        assert "BQM" in warning.message
        assert warning.retryable is False
        assert "integers" in warning.recommended_action

    @pytest.mark.parametrize("scale", [0.5, 0.1, 0.01])
    def test_every_fractional_scale_is_flagged(self, scale):
        result = validate_problem_full(pick_three(scale), model_type="bqm")
        assert warning_codes(result) == [CODE]

    @pytest.mark.parametrize("scale", [1.0, 2.0, 100.0])
    def test_integer_coefficients_are_not_flagged(self, scale):
        result = validate_problem_full(pick_three(scale), model_type="bqm")
        assert result.warnings == []

    def test_the_default_path_without_capabilities_is_bqm(self):
        assert warning_codes(validate_problem_full(pick_three(0.01))) == [CODE]

    def test_the_cqm_path_is_not_flagged(self):
        result = validate_problem_full(pick_three(0.01), model_type="cqm")
        assert result.valid is True
        assert result.warnings == []

    def test_a_soft_equality_is_not_flagged(self):
        problem = OptimizationProblem.model_validate(
            {
                "name": "fractional",
                "variables": [{"name": "x"}, {"name": "y"}],
                "objective": {
                    "direction": "maximize",
                    "linear_terms": [lin("x", 1), lin("y", 1)],
                },
                "constraints": [
                    soft("half", "==", 0.5, [lin("x", 0.5), lin("y", 0.5)], weight=10.0)
                ],
            }
        )
        result = validate_problem_full(problem, model_type="bqm")
        assert CODE not in warning_codes(result)

    def test_only_the_fractional_constraint_is_named(self):
        problem = OptimizationProblem.model_validate(
            {
                "name": "fractional",
                "variables": [{"name": "x"}, {"name": "y"}, {"name": "z"}],
                "objective": {"direction": "maximize", "linear_terms": [lin("x", 1)]},
                "constraints": [
                    hard("whole", "==", 1, [lin("x", 1), lin("y", 1)]),
                    hard("half", "==", 0.5, [lin("y", 0.5), lin("z", 0.5)]),
                    hard("cap", "<=", 2, [lin("x", 1), lin("y", 1), lin("z", 1)]),
                ],
            }
        )
        result = validate_problem_full(problem, model_type="bqm")
        flagged = [w for w in result.warnings if w.code == CODE]
        assert [w.path for w in flagged] == ["constraints[1]"]
        assert "half" in flagged[0].message

    def test_a_cardinality_constraint_is_never_flagged(self):
        problem = OptimizationProblem.model_validate(
            {
                "version": "1.2",
                "name": "fractional",
                "variables": [{"name": "x"}, {"name": "y"}, {"name": "z"}],
                "objective": {"direction": "maximize", "linear_terms": [lin("x", 1)]},
                "constraints": [],
                "cardinality_constraints": [
                    {
                        "id": "two",
                        "type": "hard",
                        "variables": ["x", "y", "z"],
                        "operator": "==",
                        "rhs": 2,
                    }
                ],
            }
        )
        result = validate_problem_full(problem, model_type="bqm")
        assert CODE not in warning_codes(result)


class TestThroughTheService:
    def test_validate_warns_for_a_heuristic_bqm_backend(self):
        result = OptimizationService().validate(pick_three(0.01))
        assert result.valid is True
        assert CODE in warning_codes(result)

    def test_an_exhaustive_backend_is_not_warned(self):
        # It enumerates every assignment and each one is re-validated
        # against the original problem, whatever the penalty is.
        problem = OptimizationProblem.model_validate(
            {
                "name": "fractional",
                "variables": [{"name": "x"}, {"name": "y"}, {"name": "z"}],
                "objective": {
                    "direction": "maximize",
                    "linear_terms": [lin("x", 3), lin("y", 2), lin("z", 1)],
                },
                "constraints": [
                    hard("one", "==", 0.01, [lin("x", 0.01), lin("y", 0.01), lin("z", 0.01)])
                ],
                "solver": {"backend": "exact"},
            }
        )
        service = OptimizationService()

        assert CODE not in warning_codes(service.validate(problem))
        result = service.solve(problem)
        assert result.status == "success"
        assert result.solutions[0].variables == {"x": 1, "y": 0, "z": 0}
        assert CODE not in warning_codes(result)

    def test_solve_carries_the_warning(self):
        # Whatever the heuristic finds, the advisory travels with the result
        # (the outcome itself depends on the sampler and is not pinned here).
        result = OptimizationService().solve(pick_three(0.01, seed=1, max_retries=0))
        assert result.status in ("success", "infeasible")
        assert CODE in warning_codes(result)


class TestCompilationIsUntouched:
    def test_the_scaled_problem_compiles_as_the_formula_says(self):
        # Stage A is advisory only: the penalty still comes from the
        # objective scale alone and is not rescaled by the coefficients.
        problem = pick_three(0.01)
        penalty = ScaledPenaltyStrategy().initial_penalty(problem)
        assert penalty == ScaledPenaltyStrategy().initial_penalty(pick_three(1.0))

        compiled = BQMCompiler().compile(problem, penalty)
        assert compiled.hard_penalty == penalty
        assert compiled.num_variables == 30
