"""Tests for validate_problem_full and the estimate helpers (Phase 2 spec §20).

Covers the advisory layer on top of ``validate_problem``: result shape,
``estimated_compiled_variables`` against what the compiler actually builds,
and one triggering plus one non-triggering case per warning code.
"""

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.validation import validate_problem, validate_problem_full
from annealbridge.validation.estimates import (
    analyze_inequality,
    compute_objective_scale,
    count_slack_bits,
    estimate_compiled_variables,
)


def lin(variable: str, coefficient: float) -> dict:
    return {"variable": variable, "coefficient": coefficient}


def quad(variable1: str, variable2: str, coefficient: float) -> dict:
    return {"variable1": variable1, "variable2": variable2, "coefficient": coefficient}


def hard(constraint_id: str, operator: str, rhs: float, terms: list[dict]) -> dict:
    return {
        "id": constraint_id,
        "type": "hard",
        "terms": terms,
        "operator": operator,
        "rhs": rhs,
    }


def soft(
    constraint_id: str, operator: str, rhs: float, terms: list[dict], weight: float
) -> dict:
    return {
        "id": constraint_id,
        "type": "soft",
        "terms": terms,
        "operator": operator,
        "rhs": rhs,
        "weight": weight,
    }


def make_problem(
    *,
    variables: tuple[str, ...] = ("x1", "x2", "x3"),
    direction: str = "minimize",
    linear: list[dict] | None = None,
    quadratic: list[dict] | None = None,
    constant: float = 0,
    constraints: list[dict] | None = None,
    solver: dict | None = None,
) -> OptimizationProblem:
    """Build a legal OptimizationProblem with sensible defaults."""
    payload: dict = {
        "name": "full validator test problem",
        "variables": [{"name": name} for name in variables],
        "objective": {
            "direction": direction,
            "linear_terms": linear if linear is not None else [lin("x1", 10)],
            "quadratic_terms": quadratic or [],
            "constant": constant,
        },
        "constraints": constraints or [],
    }
    if solver is not None:
        payload["solver"] = solver
    return OptimizationProblem.model_validate(payload)


def warning_codes(result) -> set[str]:
    return {warning.code for warning in result.warnings}


def error_codes(errors) -> set[str]:
    return {error.code for error in errors}


class TestResultShape:
    def test_valid_problem(self):
        problem = make_problem(
            linear=[lin("x1", 3), lin("x2", 4)],
            quadratic=[quad("x1", "x2", -2)],
            constraints=[hard("cap", "<=", 1, [lin("x1", 1), lin("x2", 1)])],
        )
        result = validate_problem_full(problem)
        assert result.valid is True
        assert result.errors == []
        assert isinstance(result.estimated_compiled_variables, int)
        assert result.objective_scale == compute_objective_scale(problem.objective)

    def test_problem_with_errors_reports_no_advisory_data(self):
        problem = make_problem(linear=[lin("ghost", 1)])
        result = validate_problem_full(problem)
        assert result.valid is False
        assert error_codes(result.errors) == error_codes(validate_problem(problem))
        assert error_codes(result.errors) == {"UNKNOWN_VARIABLE"}
        assert result.warnings == []
        assert result.estimated_compiled_variables is None
        assert result.objective_scale is None

    def test_non_finite_coefficient_does_not_raise(self):
        problem = make_problem(linear=[lin("x1", float("inf"))])
        result = validate_problem_full(problem)
        assert result.valid is False
        assert "NON_FINITE_COEFFICIENT" in error_codes(result.errors)


class TestEstimatedCompiledVariables:
    def make_mixed_problem(self) -> OptimizationProblem:
        # 4 variables + 1 slack bit (<= 1) + 2 slack bits (>= 2, slack range 3)
        # + 0 (redundant <= 5) + 0 (equality) == 7 compiled variables.
        return make_problem(
            variables=("x1", "x2", "x3", "x4"),
            linear=[lin("x1", 3), lin("x2", 4)],
            constraints=[
                hard("cap", "<=", 1, [lin("x1", 1), lin("x2", 1)]),
                hard("cover", ">=", 2, [lin("x1", 2), lin("x2", 3)]),
                hard("loose", "<=", 5, [lin("x1", 1), lin("x2", 1)]),
                hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)]),
            ],
        )

    def test_mixed_problem_exact_estimate(self):
        problem = self.make_mixed_problem()
        result = validate_problem_full(problem)
        assert result.valid is True
        assert result.estimated_compiled_variables == 7

    def test_estimate_matches_compilation(self):
        problem = self.make_mixed_problem()
        result = validate_problem_full(problem)
        compiled = BQMCompiler().compile(problem, hard_penalty=2.0)
        assert compiled.num_variables == result.estimated_compiled_variables

    def test_slack_bits_per_constraint(self):
        problem = self.make_mixed_problem()
        assert [count_slack_bits(c) for c in problem.constraints] == [1, 2, 0, 0]
        assert analyze_inequality(problem.constraints[2]).redundant is True

    @pytest.mark.parametrize("rhs", [3, 1])
    def test_duplicate_constraint_terms_match_compilation(self, rhs):
        # x1 appears twice with coefficient 1: the accumulated coefficient is 2,
        # so rhs=3 is redundant while rhs=1 still needs one slack bit.
        problem = make_problem(
            variables=("x1", "x2"),
            linear=[lin("x1", 1)],
            constraints=[hard("dup", "<=", rhs, [lin("x1", 1), lin("x1", 1)])],
        )
        result = validate_problem_full(problem)
        compiled = BQMCompiler().compile(problem, hard_penalty=2.0)
        assert result.estimated_compiled_variables == compiled.num_variables

    def test_no_constraints_estimate_equals_variable_count(self):
        problem = make_problem(variables=("x1", "x2", "x3", "x4"))
        result = validate_problem_full(problem)
        assert result.estimated_compiled_variables == 4
        assert estimate_compiled_variables(problem) == 4


class TestSoftWeightSmall:
    def make(self, weight: float) -> OptimizationProblem:
        # objective scale 100 -> warning threshold 1.0
        return make_problem(
            linear=[lin("x1", 100)],
            constraints=[soft("prefer", "==", 1, [lin("x2", 1)], weight)],
        )

    def test_tiny_weight_warns(self):
        problem = self.make(0.5)
        assert compute_objective_scale(problem.objective) == 100.0
        assert "SOFT_WEIGHT_SMALL" in warning_codes(validate_problem_full(problem))

    def test_weight_exactly_at_threshold_does_not_warn(self):
        problem = self.make(1.0)
        assert "SOFT_WEIGHT_SMALL" not in warning_codes(validate_problem_full(problem))


class TestLargeSlackRange:
    def make(self, rhs: int) -> OptimizationProblem:
        return make_problem(
            constraints=[hard("big", "<=", rhs, [lin("x1", 2000)])],
        )

    def test_eleven_slack_bits_warns(self):
        problem = self.make(1500)  # slack range 1500 -> 11 bits
        assert count_slack_bits(problem.constraints[0]) == 11
        assert "LARGE_SLACK_RANGE" in warning_codes(validate_problem_full(problem))

    def test_ten_slack_bits_does_not_warn(self):
        problem = self.make(1023)  # slack range 1023 -> 10 bits
        assert count_slack_bits(problem.constraints[0]) == 10
        assert "LARGE_SLACK_RANGE" not in warning_codes(validate_problem_full(problem))


class TestRedundantConstraint:
    def test_always_satisfied_le_warns(self):
        problem = make_problem(
            variables=("x1", "x2"),
            constraints=[hard("loose", "<=", 5, [lin("x1", 1), lin("x2", 1)])],
        )
        result = validate_problem_full(problem)
        assert result.valid is True  # always-true is not TRIVIALLY_INFEASIBLE
        assert "REDUNDANT_CONSTRAINT" in warning_codes(result)

    def test_binding_le_does_not_warn(self):
        problem = make_problem(
            variables=("x1", "x2"),
            constraints=[hard("cap", "<=", 1, [lin("x1", 1), lin("x2", 1)])],
        )
        assert "REDUNDANT_CONSTRAINT" not in warning_codes(validate_problem_full(problem))

    def test_always_satisfied_ge_warns(self):
        problem = make_problem(
            variables=("x1", "x2"),
            constraints=[hard("nonneg", ">=", 0, [lin("x1", 1), lin("x2", 1)])],
        )
        result = validate_problem_full(problem)
        assert result.valid is True
        assert "REDUNDANT_CONSTRAINT" in warning_codes(result)


class TestExactBackendLimits:
    def make(self, num_variables: int, *, backend: str = "exact") -> OptimizationProblem:
        names = tuple(f"x{index}" for index in range(1, num_variables + 1))
        return make_problem(
            variables=names,
            linear=[lin(names[0], 1)],
            solver={"backend": backend},
        )

    @pytest.mark.parametrize(
        "num_variables,expected",
        [
            (9, {"EXACT_NEAR_LIMIT"}),
            (10, {"EXACT_NEAR_LIMIT"}),
            (11, {"EXACT_OVER_LIMIT"}),
            (8, set()),
        ],
    )
    def test_limit_boundaries(self, num_variables, expected):
        problem = self.make(num_variables)
        result = validate_problem_full(problem, exact_max_variables=10)
        assert result.estimated_compiled_variables == num_variables
        assert warning_codes(result) & {
            "EXACT_NEAR_LIMIT",
            "EXACT_OVER_LIMIT",
        } == expected

    def test_near_and_over_are_mutually_exclusive(self):
        for num_variables in (9, 10, 11):
            result = validate_problem_full(
                self.make(num_variables), exact_max_variables=10
            )
            codes = warning_codes(result)
            assert not {"EXACT_NEAR_LIMIT", "EXACT_OVER_LIMIT"} <= codes

    def test_without_limit_no_warning(self):
        result = validate_problem_full(self.make(11))
        assert not warning_codes(result) & {"EXACT_NEAR_LIMIT", "EXACT_OVER_LIMIT"}

    def test_other_backend_ignores_limit(self):
        problem = self.make(11, backend="simulated_annealing")
        result = validate_problem_full(problem, exact_max_variables=10)
        assert not warning_codes(result) & {"EXACT_NEAR_LIMIT", "EXACT_OVER_LIMIT"}


class TestDenseForQPU:
    def make(
        self,
        num_variables: int,
        *,
        backend: str = "dwave_qpu",
        wide_constraint: bool = False,
    ) -> OptimizationProblem:
        names = tuple(f"x{index}" for index in range(1, num_variables + 1))
        constraints = []
        if wide_constraint:
            # 31 variables, sum >= 1: slack range 30 -> 5 bits.
            covered = names[:31]
            constraints.append(
                hard("wide", ">=", 1, [lin(name, 1) for name in covered])
            )
        return make_problem(
            variables=names,
            linear=[lin(names[0], 1)],
            constraints=constraints,
            solver={"backend": backend},
        )

    def test_many_variables_warns(self):
        result = validate_problem_full(self.make(151))
        assert result.estimated_compiled_variables == 151
        assert "DENSE_FOR_QPU" in warning_codes(result)

    def test_wide_constraint_alone_warns(self):
        problem = self.make(31, wide_constraint=True)
        result = validate_problem_full(problem)
        assert result.estimated_compiled_variables == 36  # 31 + 5 slack bits
        assert "DENSE_FOR_QPU" in warning_codes(result)

    def test_below_thresholds_does_not_warn(self):
        result = validate_problem_full(self.make(150))
        assert result.estimated_compiled_variables == 150
        assert "DENSE_FOR_QPU" not in warning_codes(result)

    def test_other_backend_does_not_warn(self):
        result = validate_problem_full(self.make(151, backend="simulated_annealing"))
        assert "DENSE_FOR_QPU" not in warning_codes(result)

    def test_both_conditions_warn_only_once(self):
        problem = self.make(151, wide_constraint=True)
        result = validate_problem_full(problem)
        dense = [w for w in result.warnings if w.code == "DENSE_FOR_QPU"]
        assert len(dense) == 1


class TestSeedIgnored:
    @pytest.mark.parametrize("backend", ["dwave_qpu", "leap_hybrid_bqm"])
    def test_remote_backends_warn(self, backend):
        problem = make_problem(solver={"backend": backend, "seed": 42})
        assert "SEED_IGNORED" in warning_codes(validate_problem_full(problem))

    def test_hybrid_seed_only_warns_once_without_parameter_ignored(self):
        problem = make_problem(solver={"backend": "leap_hybrid_bqm", "seed": 42})
        codes = warning_codes(validate_problem_full(problem))
        assert "SEED_IGNORED" in codes
        assert "PARAMETER_IGNORED" not in codes

    def test_local_backend_does_not_warn(self):
        problem = make_problem(
            solver={"backend": "simulated_annealing", "seed": 42}
        )
        assert "SEED_IGNORED" not in warning_codes(validate_problem_full(problem))


class TestParameterIgnored:
    def test_hybrid_num_reads_warns_once(self):
        problem = make_problem(
            solver={"backend": "leap_hybrid_bqm", "num_reads": 200}
        )
        ignored = [
            w
            for w in validate_problem_full(problem).warnings
            if w.code == "PARAMETER_IGNORED"
        ]
        assert len(ignored) == 1
        assert ignored[0].path == "solver.num_reads"

    def test_hybrid_two_parameters_warn_twice(self):
        problem = make_problem(
            solver={
                "backend": "leap_hybrid_bqm",
                "num_reads": 200,
                "num_sweeps": 500,
            }
        )
        ignored = [
            w
            for w in validate_problem_full(problem).warnings
            if w.code == "PARAMETER_IGNORED"
        ]
        assert len(ignored) == 2
        assert {w.path for w in ignored} == {"solver.num_reads", "solver.num_sweeps"}

    def test_hybrid_defaults_do_not_warn(self):
        problem = make_problem(solver={"backend": "leap_hybrid_bqm"})
        assert "PARAMETER_IGNORED" not in warning_codes(validate_problem_full(problem))

    def test_local_backend_does_not_warn(self):
        problem = make_problem(
            solver={"backend": "simulated_annealing", "num_reads": 200}
        )
        assert "PARAMETER_IGNORED" not in warning_codes(validate_problem_full(problem))


class TestDuplicateTermMerged:
    def test_duplicate_linear_term_warns_but_stays_valid(self):
        problem = make_problem(linear=[lin("x1", 2), lin("x1", 3)])
        result = validate_problem_full(problem)
        assert result.valid is True
        assert "DUPLICATE_TERM_MERGED" in warning_codes(result)
        assert validate_problem(problem) == []  # legacy behaviour unchanged

    def test_duplicate_quadratic_pair_across_orderings_warns(self):
        problem = make_problem(
            quadratic=[quad("x1", "x2", 3), quad("x2", "x1", 4)],
        )
        result = validate_problem_full(problem)
        warnings = [
            w for w in result.warnings if w.code == "DUPLICATE_TERM_MERGED"
        ]
        assert len(warnings) == 1
        assert warnings[0].path == "objective.quadratic_terms"

    def test_unique_terms_do_not_warn(self):
        problem = make_problem(
            linear=[lin("x1", 2), lin("x2", 3)],
            quadratic=[quad("x1", "x2", 1)],
        )
        assert "DUPLICATE_TERM_MERGED" not in warning_codes(validate_problem_full(problem))


class TestWarningPayload:
    def all_warning_results(self) -> list:
        return [
            validate_problem_full(
                make_problem(
                    linear=[lin("x1", 100)],
                    constraints=[soft("prefer", "==", 1, [lin("x2", 1)], 0.5)],
                )
            ),
            validate_problem_full(
                make_problem(constraints=[hard("big", "<=", 1500, [lin("x1", 2000)])])
            ),
            validate_problem_full(
                make_problem(
                    variables=("x1", "x2"),
                    constraints=[
                        hard("loose", "<=", 5, [lin("x1", 1), lin("x2", 1)])
                    ],
                )
            ),
            validate_problem_full(  # 3 variables, limit 3 -> near limit
                make_problem(solver={"backend": "exact"}), exact_max_variables=3
            ),
            validate_problem_full(  # 3 variables, limit 2 -> over limit
                make_problem(solver={"backend": "exact"}), exact_max_variables=2
            ),
            validate_problem_full(
                make_problem(
                    variables=tuple(f"x{index}" for index in range(1, 152)),
                    solver={"backend": "dwave_qpu"},
                )
            ),
            validate_problem_full(
                make_problem(
                    solver={
                        "backend": "leap_hybrid_bqm",
                        "seed": 42,
                        "num_reads": 200,
                    }
                )
            ),
            validate_problem_full(make_problem(linear=[lin("x1", 2), lin("x1", 3)])),
        ]

    def test_every_warning_code_is_covered(self):
        seen: set[str] = set()
        for result in self.all_warning_results():
            seen |= warning_codes(result)
        assert seen == {
            "SOFT_WEIGHT_SMALL",
            "LARGE_SLACK_RANGE",
            "REDUNDANT_CONSTRAINT",
            "EXACT_NEAR_LIMIT",
            "EXACT_OVER_LIMIT",
            "DENSE_FOR_QPU",
            "SEED_IGNORED",
            "PARAMETER_IGNORED",
            "DUPLICATE_TERM_MERGED",
        }

    def test_warnings_are_not_retryable_and_carry_an_action(self):
        for result in self.all_warning_results():
            assert result.warnings  # every fixture above triggers something
            for warning in result.warnings:
                assert warning.retryable is False
                assert isinstance(warning.recommended_action, str)
                assert warning.recommended_action.strip()
