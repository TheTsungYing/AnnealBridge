"""Tests for validate_problem_full and the estimate helpers (Phase 2 §20, 3a §9).

Covers the advisory layer on top of ``validate_problem``: result shape,
``estimated_compiled_variables`` against what the compiler actually builds,
and one triggering plus one non-triggering case per warning code. Since
3a §9 every backend-dependent warning is driven by an injected
``SolverCapabilities`` declaration, so the table in §9.3 is exercised with
synthetic declarations (one flag at a time) plus the real registry ones
where the spec pins behaviour (drift 2: exact + seed).
"""

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import CompilationError
from annealbridge.models import OptimizationProblem, SolverCapabilities
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import validate_problem, validate_problem_full
from annealbridge.validation import problem_validator
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
    """Build a legal OptimizationProblem with sensible defaults.

    ``variables`` entries are names (binary) or full variable dicts (see
    :func:`integer`); a problem with any integer variable is version 1.1.
    """
    declared = [entry if isinstance(entry, dict) else {"name": entry} for entry in variables]
    payload: dict = {
        "name": "full validator test problem",
        "variables": declared,
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
    if any(entry.get("type") == "integer" for entry in declared):
        payload["version"] = "1.1"
    return OptimizationProblem.model_validate(payload)


def integer(name: str, lower: int, upper: int) -> dict:
    return {"name": name, "type": "integer", "lower_bound": lower, "upper_bound": upper}


def warning_codes(result) -> set[str]:
    return {warning.code for warning in result.warnings}


def warnings_with(result, code: str) -> list:
    return [warning for warning in result.warnings if warning.code == code]


def caps(**overrides) -> SolverCapabilities:
    """A permissive synthetic declaration; override one flag per test."""
    base = dict(
        name="fake_backend",
        remote=False,
        heuristic=True,
        exhaustive=False,
        supports_seed=True,
        supports_num_reads=True,
        supports_time_limit=False,
        supported_model_types=["bqm"],
        returns_multiple_samples=True,
        supports_num_sweeps=True,
        requires_embedding=False,
        description="synthetic declaration for validator tests",
    )
    return SolverCapabilities(**{**base, **overrides})


_REGISTRY = SolverRegistry.default()


def registry_caps(name: str) -> SolverCapabilities:
    return _REGISTRY.get(name).capabilities


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


class TestSoftAlwaysViolated:
    """Review F-04 / F-12: a soft constraint no assignment can satisfy warns.

    It is a legal problem (the weight is simply always paid), so this is a
    warning, never an error; the judgement uses the same hybrid tolerance
    as ``TRIVIALLY_INFEASIBLE`` does for hard constraints.
    """

    def test_zero_coefficients_with_nonzero_rhs_warns(self):
        # F-04's reproduction: x:+1, x:-1 == 5, weight 2.
        problem = make_problem(
            constraints=[soft("c1", "==", 5, [lin("x1", 1), lin("x1", -1)], 2.0)]
        )
        result = validate_problem_full(problem)
        assert result.valid is True
        assert result.errors == []
        (warning,) = warnings_with(result, "SOFT_ALWAYS_VIOLATED")
        assert warning.path == "constraints[0]"
        assert "c1" in warning.message
        assert "== 5.0" in warning.message
        assert "2.0" in warning.message
        assert warning.recommended_action
        assert warning.retryable is False

    def test_f12_pair_reports_both_sides_symmetrically(self):
        problem = make_problem(
            variables=("x", "y"),
            linear=[lin("x", 1)],
            constraints=[
                soft("impossible", "<=", -5, [lin("x", -1), lin("y", -1)], 10.0),
                soft("always", "<=", 3, [lin("x", 1)], 1.0),
            ],
        )
        result = validate_problem_full(problem)
        assert result.valid is True
        assert {(w.code, w.path) for w in result.warnings} == {
            ("SOFT_ALWAYS_VIOLATED", "constraints[0]"),
            ("REDUNDANT_CONSTRAINT", "constraints[1]"),
        }
        (warning,) = warnings_with(result, "SOFT_ALWAYS_VIOLATED")
        assert "impossible" in warning.message
        assert "[-2.0, 0.0]" in warning.message
        assert "<= -5.0" in warning.message

    def test_equality_whose_range_misses_the_rhs_warns(self):
        problem = make_problem(
            variables=("x1", "x2"),
            constraints=[soft("three", "==", 3, [lin("x1", 1), lin("x2", 1)], 1.0)],
        )
        (warning,) = warnings_with(validate_problem_full(problem), "SOFT_ALWAYS_VIOLATED")
        assert warning.path == "constraints[0]"

    def test_greater_equal_over_integer_bounds_warns(self):
        problem = make_problem(
            variables=(integer("n", -3, 2), "x1"),
            constraints=[soft("big", ">=", 4, [lin("n", 1), lin("x1", 1)], 1.0)],
        )
        (warning,) = warnings_with(validate_problem_full(problem), "SOFT_ALWAYS_VIOLATED")
        assert "[-3.0, 3.0]" in warning.message

    @pytest.mark.parametrize("rhs", [0, 1e-9, 9e-9])
    def test_zero_coefficients_with_rhs_within_tolerance_is_redundant(self, rhs):
        problem = make_problem(
            constraints=[soft("noop", "==", rhs, [lin("x1", 1), lin("x1", -1)], 2.0)]
        )
        result = validate_problem_full(problem)
        assert warnings_with(result, "SOFT_ALWAYS_VIOLATED") == []
        (warning,) = warnings_with(result, "REDUNDANT_CONSTRAINT")
        assert warning.path == "constraints[0]"
        assert "noop" in warning.message

    @pytest.mark.parametrize("rhs", [1.5e-8, 1e-7, -1e-7])
    def test_zero_coefficients_with_rhs_beyond_tolerance_warns(self, rhs):
        problem = make_problem(
            constraints=[soft("off", "==", rhs, [lin("x1", 1), lin("x1", -1)], 2.0)]
        )
        result = validate_problem_full(problem)
        assert warnings_with(result, "REDUNDANT_CONSTRAINT") == []
        assert len(warnings_with(result, "SOFT_ALWAYS_VIOLATED")) == 1

    def test_hard_zero_coefficient_equality_that_holds_stays_silent(self):
        # The Phase 3a golden recording pins the validator output of a 1.0
        # problem with exactly this constraint (no warning), so the soft
        # REDUNDANT_CONSTRAINT above is not extended to hard equalities.
        problem = make_problem(
            constraints=[hard("noop_eq", "==", 0, [lin("x1", 1), lin("x1", -1)])]
        )
        result = validate_problem_full(problem)
        assert result.valid is True
        assert result.warnings == []

    def test_hard_constraint_is_an_error_never_the_soft_warning(self):
        problem = make_problem(
            constraints=[hard("c1", "==", 5, [lin("x1", 1), lin("x1", -1)])]
        )
        result = validate_problem_full(problem)
        assert result.valid is False
        assert error_codes(result.errors) == {"TRIVIALLY_INFEASIBLE"}
        assert "SOFT_ALWAYS_VIOLATED" not in warning_codes(result)

    def test_satisfiable_soft_constraints_do_not_warn(self):
        problem = make_problem(
            variables=("x1", "x2"),
            constraints=[
                soft("eq", "==", 1, [lin("x1", 1), lin("x2", 1)], 1.0),
                soft("le", "<=", 0, [lin("x1", 1)], 1.0),
                soft("ge", ">=", 2, [lin("x1", 1), lin("x2", 1)], 1.0),
            ],
        )
        assert "SOFT_ALWAYS_VIOLATED" not in warning_codes(validate_problem_full(problem))


class TestExactBackendLimits:
    """§9.3 rows EXACT_OVER_LIMIT / EXACT_NEAR_LIMIT: ``caps.exhaustive``."""

    def make(self, num_variables: int) -> OptimizationProblem:
        names = tuple(f"x{index}" for index in range(1, num_variables + 1))
        return make_problem(variables=names, linear=[lin(names[0], 1)])

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
        result = validate_problem_full(
            self.make(num_variables),
            capabilities=caps(exhaustive=True),
            max_compiled_variables=10,
        )
        assert result.estimated_compiled_variables == num_variables
        assert warning_codes(result) & {
            "EXACT_NEAR_LIMIT",
            "EXACT_OVER_LIMIT",
        } == expected

    def test_near_and_over_are_mutually_exclusive(self):
        for num_variables in (9, 10, 11):
            result = validate_problem_full(
                self.make(num_variables),
                capabilities=caps(exhaustive=True),
                max_compiled_variables=10,
            )
            assert not {"EXACT_NEAR_LIMIT", "EXACT_OVER_LIMIT"} <= warning_codes(result)

    def test_without_limit_no_warning(self):
        # max_compiled_variables=None: the exhaustive rows need a ceiling.
        result = validate_problem_full(self.make(11), capabilities=caps(exhaustive=True))
        assert not warning_codes(result) & {"EXACT_NEAR_LIMIT", "EXACT_OVER_LIMIT"}

    def test_non_exhaustive_backend_ignores_limit(self):
        result = validate_problem_full(
            self.make(11), capabilities=caps(exhaustive=False), max_compiled_variables=10
        )
        assert not warning_codes(result) & {"EXACT_NEAR_LIMIT", "EXACT_OVER_LIMIT"}

    def test_real_exact_declaration_triggers_the_rows(self):
        result = validate_problem_full(
            self.make(11), capabilities=registry_caps("exact"), max_compiled_variables=10
        )
        assert "EXACT_OVER_LIMIT" in warning_codes(result)

    def test_messages_describe_capability_not_backend_name(self):
        # 3a §9.3 / §13.2: code names stay, wording becomes capability-based.
        over = validate_problem_full(
            self.make(11), capabilities=caps(exhaustive=True), max_compiled_variables=10
        )
        near = validate_problem_full(
            self.make(9), capabilities=caps(exhaustive=True), max_compiled_variables=10
        )
        for result in (over, near):
            (warning,) = [
                w for w in result.warnings if w.code.startswith("EXACT_")
            ]
            assert "exhaustive backend limit" in warning.message
            assert "exact solver" not in warning.message
            for name in ("exact", "simulated_annealing", "dwave_qpu", "leap_hybrid_bqm"):
                assert name not in warning.recommended_action


class TestDenseForQPU:
    """§9.3 row DENSE_FOR_QPU: ``caps.requires_embedding``."""

    def make(
        self, num_variables: int, *, wide_constraint: bool = False
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
            variables=names, linear=[lin(names[0], 1)], constraints=constraints
        )

    def test_many_variables_warns(self):
        result = validate_problem_full(
            self.make(151), capabilities=caps(requires_embedding=True)
        )
        assert result.estimated_compiled_variables == 151
        assert "DENSE_FOR_QPU" in warning_codes(result)

    def test_wide_constraint_alone_warns(self):
        result = validate_problem_full(
            self.make(31, wide_constraint=True),
            capabilities=caps(requires_embedding=True),
        )
        assert result.estimated_compiled_variables == 36  # 31 + 5 slack bits
        assert "DENSE_FOR_QPU" in warning_codes(result)

    def test_below_thresholds_does_not_warn(self):
        result = validate_problem_full(
            self.make(150), capabilities=caps(requires_embedding=True)
        )
        assert result.estimated_compiled_variables == 150
        assert "DENSE_FOR_QPU" not in warning_codes(result)

    def test_backend_without_embedding_does_not_warn(self):
        result = validate_problem_full(
            self.make(151), capabilities=caps(requires_embedding=False)
        )
        assert "DENSE_FOR_QPU" not in warning_codes(result)

    def test_real_qpu_declaration_warns(self):
        result = validate_problem_full(
            self.make(151), capabilities=registry_caps("dwave_qpu")
        )
        assert "DENSE_FOR_QPU" in warning_codes(result)

    def test_both_conditions_warn_only_once(self):
        result = validate_problem_full(
            self.make(151, wide_constraint=True),
            capabilities=caps(requires_embedding=True),
        )
        assert len(warnings_with(result, "DENSE_FOR_QPU")) == 1

    def test_message_describes_embedding_not_backend_name(self):
        result = validate_problem_full(
            self.make(151), capabilities=caps(requires_embedding=True)
        )
        (warning,) = warnings_with(result, "DENSE_FOR_QPU")
        assert "requires minor-embedding" in warning.message
        for name in ("dwave_qpu", "leap_hybrid_bqm", "simulated_annealing"):
            assert name not in warning.recommended_action


class TestSeedIgnored:
    """§9.3 row SEED_IGNORED: ``solver.seed`` set and ``not caps.supports_seed``."""

    def test_unsupported_seed_warns(self):
        problem = make_problem(solver={"seed": 42})
        result = validate_problem_full(problem, capabilities=caps(supports_seed=False))
        (warning,) = warnings_with(result, "SEED_IGNORED")
        assert warning.path == "solver.seed"

    def test_supported_seed_does_not_warn(self):
        problem = make_problem(solver={"seed": 42})
        result = validate_problem_full(problem, capabilities=caps(supports_seed=True))
        assert "SEED_IGNORED" not in warning_codes(result)

    def test_no_seed_does_not_warn(self):
        result = validate_problem_full(make_problem(), capabilities=caps(supports_seed=False))
        assert "SEED_IGNORED" not in warning_codes(result)

    @pytest.mark.parametrize("backend", ["exact", "dwave_qpu", "leap_hybrid_bqm"])
    def test_registry_backends_that_declare_no_seed_warn(self, backend):
        # 3a §9.3 drift 2: exact declares supports_seed=False and now warns
        # like the remote backends do, instead of being silently exempt.
        problem = make_problem(solver={"backend": backend, "seed": 42})
        result = validate_problem_full(problem, capabilities=registry_caps(backend))
        assert "SEED_IGNORED" in warning_codes(result)

    def test_seed_alone_does_not_also_produce_parameter_ignored(self):
        problem = make_problem(solver={"seed": 42})
        result = validate_problem_full(
            problem, capabilities=caps(supports_seed=False, supports_num_reads=False)
        )
        assert "SEED_IGNORED" in warning_codes(result)
        assert "PARAMETER_IGNORED" not in warning_codes(result)

    def test_simulated_annealing_declaration_does_not_warn(self):
        problem = make_problem(solver={"backend": "simulated_annealing", "seed": 42})
        result = validate_problem_full(
            problem, capabilities=registry_caps("simulated_annealing")
        )
        assert "SEED_IGNORED" not in warning_codes(result)


class TestParameterIgnored:
    """§9.3 PARAMETER_IGNORED rows, one capability condition each."""

    def test_num_reads_without_support_warns_once(self):
        problem = make_problem(solver={"num_reads": 200})
        result = validate_problem_full(
            problem, capabilities=caps(supports_num_reads=False)
        )
        (warning,) = warnings_with(result, "PARAMETER_IGNORED")
        assert warning.path == "solver.num_reads"

    def test_num_sweeps_without_support_warns_once(self):
        problem = make_problem(solver={"num_sweeps": 500})
        result = validate_problem_full(
            problem, capabilities=caps(supports_num_sweeps=False)
        )
        (warning,) = warnings_with(result, "PARAMETER_IGNORED")
        assert warning.path == "solver.num_sweeps"

    def test_two_unsupported_parameters_warn_twice(self):
        problem = make_problem(solver={"num_reads": 200, "num_sweeps": 500})
        result = validate_problem_full(
            problem,
            capabilities=caps(supports_num_reads=False, supports_num_sweeps=False),
        )
        ignored = warnings_with(result, "PARAMETER_IGNORED")
        assert {w.path for w in ignored} == {"solver.num_reads", "solver.num_sweeps"}

    def test_defaults_never_warn(self):
        result = validate_problem_full(
            make_problem(),
            capabilities=caps(
                supports_num_reads=False, supports_num_sweeps=False, exhaustive=True
            ),
        )
        assert "PARAMETER_IGNORED" not in warning_codes(result)

    def test_supported_parameters_do_not_warn(self):
        problem = make_problem(solver={"num_reads": 200, "num_sweeps": 500})
        result = validate_problem_full(
            problem, capabilities=caps(supports_num_reads=True, supports_num_sweeps=True)
        )
        assert "PARAMETER_IGNORED" not in warning_codes(result)

    def test_exhaustive_backend_ignores_max_retries(self):
        # §9.3: an exhaustive backend always makes exactly one attempt (§16.3).
        problem = make_problem(solver={"max_retries": 5})
        result = validate_problem_full(problem, capabilities=caps(exhaustive=True))
        (warning,) = warnings_with(result, "PARAMETER_IGNORED")
        assert warning.path == "solver.max_retries"

    def test_heuristic_backend_keeps_max_retries(self):
        problem = make_problem(solver={"max_retries": 5})
        result = validate_problem_full(problem, capabilities=caps(exhaustive=False))
        assert "PARAMETER_IGNORED" not in warning_codes(result)

    def test_real_exact_declaration_warns_for_reads_sweeps_and_retries(self):
        # 3a §9.3 (intentional behaviour change): exact + non-default
        # num_reads / num_sweeps / max_retries now warn.
        problem = make_problem(
            solver={
                "backend": "exact",
                "num_reads": 200,
                "num_sweeps": 500,
                "max_retries": 1,
            }
        )
        result = validate_problem_full(problem, capabilities=registry_caps("exact"))
        assert {w.path for w in warnings_with(result, "PARAMETER_IGNORED")} == {
            "solver.num_reads",
            "solver.num_sweeps",
            "solver.max_retries",
        }

    def test_real_hybrid_declaration_warns_for_reads_and_sweeps(self):
        problem = make_problem(
            solver={"backend": "leap_hybrid_bqm", "num_reads": 200, "num_sweeps": 500}
        )
        result = validate_problem_full(
            problem, capabilities=registry_caps("leap_hybrid_bqm")
        )
        assert {w.path for w in warnings_with(result, "PARAMETER_IGNORED")} == {
            "solver.num_reads",
            "solver.num_sweeps",
        }

    def test_option_block_for_another_backend_warns(self):
        # §9.3 naming contract: block field name must equal caps.name.
        problem = make_problem(
            solver={"backend": "exact", "dwave_qpu": {"annealing_time_us": 20}}
        )
        result = validate_problem_full(problem, capabilities=registry_caps("exact"))
        (warning,) = warnings_with(result, "PARAMETER_IGNORED")
        assert warning.path == "solver.dwave_qpu"

    def test_option_block_for_the_selected_backend_does_not_warn(self):
        problem = make_problem(
            solver={"backend": "dwave_qpu", "dwave_qpu": {"annealing_time_us": 20}}
        )
        result = validate_problem_full(problem, capabilities=registry_caps("dwave_qpu"))
        assert "PARAMETER_IGNORED" not in warning_codes(result)

    def test_option_block_compares_against_caps_name_not_registry_key(self):
        # A custom registry may register the same declaration under another
        # key; the block check follows the declaration's own name.
        problem = make_problem(
            solver={"backend": "dwave_qpu", "dwave_qpu": {"annealing_time_us": 20}}
        )
        result = validate_problem_full(problem, capabilities=caps(name="dwave_qpu"))
        assert "PARAMETER_IGNORED" not in warning_codes(result)
        result = validate_problem_full(problem, capabilities=caps(name="other"))
        assert [w.path for w in warnings_with(result, "PARAMETER_IGNORED")] == [
            "solver.dwave_qpu"
        ]

    def test_messages_and_actions_carry_no_backend_name_constants(self):
        problem = make_problem(solver={"num_reads": 200})
        result = validate_problem_full(
            problem, capabilities=caps(supports_num_reads=False)
        )
        (warning,) = warnings_with(result, "PARAMETER_IGNORED")
        assert "fake_backend" in warning.message  # the declaration's own name
        assert "leap_hybrid_bqm" not in warning.message


class TestCQMModelType:
    """§9.2 estimate and the two ``model_type == "cqm"`` rows of §9.3.

    No real backend takes the CQM path yet (3a step 3), so a synthetic
    declaration stands in; the same rows are re-asserted in step 7.
    """

    def make(self, **solver) -> OptimizationProblem:
        # 3 variables + one <= 1 inequality: BQM would add 1 slack bit.
        return make_problem(
            constraints=[hard("cap", "<=", 1, [lin("x1", 1), lin("x2", 1)])],
            solver=solver or None,
        )

    def test_cqm_estimate_is_the_plain_variable_count(self):
        result = validate_problem_full(
            self.make(), capabilities=caps(supported_model_types=["cqm"])
        )
        assert result.model_type == "cqm"
        assert result.estimated_compiled_variables == 3

    def test_bqm_estimate_includes_slack(self):
        result = validate_problem_full(self.make(), capabilities=caps())
        assert result.model_type == "bqm"
        assert result.estimated_compiled_variables == 4

    def test_explicit_model_type_overrides_preferred(self):
        # A backend accepting both types: the service tells the validator
        # which compiler it will actually use (§9.1).
        both = caps(supported_model_types=["bqm", "cqm"])
        assert validate_problem_full(self.make(), capabilities=both).model_type == "bqm"
        forced = validate_problem_full(self.make(), capabilities=both, model_type="cqm")
        assert forced.model_type == "cqm"
        assert forced.estimated_compiled_variables == 3

    def test_model_type_without_capabilities_defaults_to_bqm(self):
        assert validate_problem_full(self.make()).model_type == "bqm"
        assert validate_problem_full(self.make(), model_type="cqm").model_type == "cqm"

    def test_invalid_problem_records_no_model_type(self):
        problem = make_problem(linear=[lin("ghost", 1)])
        assert validate_problem_full(problem, capabilities=caps()).model_type is None

    def test_cqm_ignores_penalty_multiplier(self):
        result = validate_problem_full(
            self.make(penalty_multiplier=3.0),
            capabilities=caps(supported_model_types=["cqm"]),
        )
        (warning,) = warnings_with(result, "PARAMETER_IGNORED")
        assert warning.path == "solver.penalty_multiplier"

    def test_cqm_ignores_max_retries(self):
        result = validate_problem_full(
            self.make(max_retries=1), capabilities=caps(supported_model_types=["cqm"])
        )
        (warning,) = warnings_with(result, "PARAMETER_IGNORED")
        assert warning.path == "solver.max_retries"

    def test_bqm_keeps_penalty_multiplier_and_retries(self):
        result = validate_problem_full(
            self.make(penalty_multiplier=3.0, max_retries=1), capabilities=caps()
        )
        assert "PARAMETER_IGNORED" not in warning_codes(result)

    def test_large_slack_range_still_reported_on_cqm(self):
        # §21.1: the warning describes the BQM cost, useful either way.
        problem = make_problem(constraints=[hard("big", "<=", 1500, [lin("x1", 2000)])])
        result = validate_problem_full(
            problem, capabilities=caps(supported_model_types=["cqm"])
        )
        (warning,) = warnings_with(result, "LARGE_SLACK_RANGE")
        assert "on a BQM backend" in warning.message


class TestWithoutCapabilities:
    """§9.1: ``capabilities=None`` runs only backend-independent checks."""

    def test_no_backend_warnings_at_all(self):
        problem = make_problem(
            variables=tuple(f"x{index}" for index in range(1, 152)),
            solver={
                "backend": "exact",
                "seed": 42,
                "num_reads": 200,
                "num_sweeps": 500,
                "max_retries": 1,
                "dwave_qpu": {"annealing_time_us": 20},
            },
        )
        result = validate_problem_full(problem, max_compiled_variables=10)
        assert result.valid is True
        assert not warning_codes(result) & {
            "EXACT_NEAR_LIMIT",
            "EXACT_OVER_LIMIT",
            "DENSE_FOR_QPU",
            "SEED_IGNORED",
            "PARAMETER_IGNORED",
        }

    def test_backend_independent_warnings_still_run(self):
        problem = make_problem(
            linear=[lin("x1", 100), lin("x1", 1)],
            constraints=[
                soft("prefer", "==", 1, [lin("x2", 1)], 0.5),
                hard("big", "<=", 1500, [lin("x3", 2000)]),
                hard("loose", "<=", 5, [lin("x1", 1), lin("x2", 1)]),
            ],
        )
        assert warning_codes(validate_problem_full(problem)) == {
            "SOFT_WEIGHT_SMALL",
            "LARGE_SLACK_RANGE",
            "REDUNDANT_CONSTRAINT",
            "DUPLICATE_TERM_MERGED",
        }


class TestOptionBlockReflection:
    """§9.4: option blocks and their positive numeric fields come from the model."""

    def test_blocks_are_the_optional_model_fields(self):
        assert set(problem_validator._option_blocks()) == {
            "dwave_qpu",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
            "fujitsu_da",
        }

    def test_positive_fields_exclude_bool(self):
        assert set(problem_validator._POSITIVE_OPTION_FIELDS) == {
            ("dwave_qpu", "annealing_time_us"),
            ("dwave_qpu", "chain_strength"),
            ("leap_hybrid_bqm", "time_limit_seconds"),
            ("leap_hybrid_cqm", "time_limit_seconds"),
            ("fujitsu_da", "time_limit_seconds"),
            ("fujitsu_da", "num_run"),
            ("fujitsu_da", "num_group"),
            ("fujitsu_da", "num_output_solution"),
        }

    def test_seed_is_not_a_block(self):
        # ``seed: int | None`` shares the union spelling but is a leaf.
        assert "seed" not in problem_validator._option_blocks()

    def test_typing_union_spelling_is_recognised(self):
        from typing import Optional, Union

        from pydantic import BaseModel

        class Block(BaseModel):
            value: Optional[float] = None
            count: Union[int, None] = None
            flag: bool = True

        assert problem_validator._optional_model(Optional[Block]) is Block
        assert problem_validator._optional_model(Union[Block, None]) is Block
        assert problem_validator._optional_model(Block | None) is Block
        assert problem_validator._optional_model(Block) is None
        assert problem_validator._optional_model(Optional[int]) is None
        assert problem_validator._optional_number(Optional[float]) is True
        assert problem_validator._optional_number(Union[int, None]) is True
        assert problem_validator._optional_number(bool) is False
        assert problem_validator._optional_number(Optional[Block]) is False

    @pytest.mark.parametrize(
        "block,field,value",
        [
            ("dwave_qpu", "annealing_time_us", 0),
            ("dwave_qpu", "chain_strength", -1.0),
            ("leap_hybrid_bqm", "time_limit_seconds", 0),
            ("leap_hybrid_cqm", "time_limit_seconds", 0),
        ],
    )
    def test_non_positive_option_is_rejected(self, block, field, value):
        problem = make_problem(solver={block: {field: value}})
        result = validate_problem_full(problem)
        assert result.valid is False
        (error,) = result.errors
        assert error.code == "INVALID_SOLVER_PREFERENCE"
        assert error.path == f"solver.{block}.{field}"


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
                make_problem(), capabilities=caps(exhaustive=True), max_compiled_variables=3
            ),
            validate_problem_full(  # 3 variables, limit 2 -> over limit
                make_problem(), capabilities=caps(exhaustive=True), max_compiled_variables=2
            ),
            validate_problem_full(
                make_problem(variables=tuple(f"x{index}" for index in range(1, 152))),
                capabilities=caps(requires_embedding=True),
            ),
            validate_problem_full(
                make_problem(solver={"seed": 42, "num_reads": 200}),
                capabilities=caps(supports_seed=False, supports_num_reads=False),
            ),
            validate_problem_full(make_problem(linear=[lin("x1", 2), lin("x1", 3)])),
            validate_problem_full(  # 3b: 21 encoding bits on the BQM path
                make_problem(variables=(integer("n", 0, 2**20), "x1"))
            ),
            validate_problem_full(  # 3b: two 31-bit integers coupled by a constraint
                make_problem(
                    variables=(integer("n", 0, 2**31 - 1), integer("m", 0, 2**31 - 1), "x1"),
                    constraints=[hard("cap", "<=", 1000, [lin("n", 1), lin("m", 1)])],
                )
            ),
            validate_problem_full(  # review F-04 / F-12: a soft constraint never satisfiable
                make_problem(
                    constraints=[soft("c1", "==", 5, [lin("x1", 1), lin("x1", -1)], 2.0)]
                )
            ),
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
            "LARGE_INTEGER_RANGE",
            "INTEGER_QUADRATIC_BLOWUP",
            "SOFT_ALWAYS_VIOLATED",
        }

    def test_warnings_are_not_retryable_and_carry_an_action(self):
        for result in self.all_warning_results():
            assert result.warnings  # every fixture above triggers something
            for warning in result.warnings:
                assert warning.retryable is False
                assert isinstance(warning.recommended_action, str)
                assert warning.recommended_action.strip()

    def test_recommended_actions_name_no_backend(self):
        # 3a §13.2: the validator's guidance describes capabilities only.
        for action in problem_validator._WARNING_RECOMMENDED_ACTIONS.values():
            for name in ("exact", "simulated_annealing", "dwave_qpu", "leap_hybrid"):
                assert name not in action


class TestValidateCompileConsistency:
    """Whatever passes validate_problem must compile, and vice versa.

    Inequalities with repeated variables are the historical mismatch: the
    validator used per-term ranges while the compiler sums coefficients.
    Both now judge on accumulated coefficients, so the two verdicts agree
    on every case below and the estimate matches the compiled size.
    """

    CASES = [
        # (operator, rhs, terms) — repeated variables with mixed signs
        (">=", 1, [lin("x1", 1), lin("x1", -1)]),  # sums to 0: infeasible
        ("<=", -1, [lin("x1", 1), lin("x1", -1)]),  # sums to 0: infeasible
        ("<=", -2, [lin("x1", 1), lin("x1", -2)]),  # sums to -1: infeasible
        ("<=", -1, [lin("x1", 1), lin("x1", -2)]),  # sums to -1: feasible
        ("<=", 0, [lin("x1", 2), lin("x1", -1)]),  # sums to +1: feasible
        (">=", 1, [lin("x1", 2), lin("x1", -1)]),  # sums to +1: feasible
        (">=", 2, [lin("x1", 2), lin("x1", -1)]),  # sums to +1: infeasible
        ("<=", 1, [lin("x1", 1), lin("x1", 1)]),  # sums to 2: feasible
        ("<=", 3, [lin("x1", 1), lin("x1", 1)]),  # redundant: feasible
        (">=", 3, [lin("x1", 1), lin("x1", 1)]),  # sums to 2: infeasible
        (">=", 0, [lin("x1", 1), lin("x1", -1), lin("x2", 1)]),  # feasible
        (">=", 2, [lin("x1", 1), lin("x1", -1), lin("x2", 1)]),  # infeasible
        ("<=", -1, [lin("x1", 1), lin("x1", -1), lin("x2", -1)]),  # feasible
        ("<=", 0, [lin("x1", 3), lin("x1", -3), lin("x2", 0)]),  # all-zero: redundant
    ]

    @pytest.mark.parametrize(("operator", "rhs", "terms"), CASES)
    def test_validator_and_compiler_agree(self, operator, rhs, terms):
        problem = make_problem(
            variables=("x1", "x2"),
            constraints=[hard("dup", operator, rhs, terms)],
        )
        result = validate_problem_full(problem)
        if result.valid:
            compiled = BQMCompiler().compile(problem, hard_penalty=2.0)
            assert compiled.num_variables == result.estimated_compiled_variables
        else:
            assert [e.code for e in result.errors] == ["TRIVIALLY_INFEASIBLE"]
            with pytest.raises(CompilationError):
                BQMCompiler().compile(problem, hard_penalty=2.0)

    def test_cases_cover_both_verdicts(self):
        verdicts = {
            validate_problem_full(
                make_problem(
                    variables=("x1", "x2"),
                    constraints=[hard("dup", operator, rhs, terms)],
                )
            ).valid
            for operator, rhs, terms in self.CASES
        }
        assert verdicts == {True, False}


class TestCountSlackBitsNeverClampsHard:
    def test_hard_infeasible_raises_instead_of_clamping(self):
        problem = make_problem(
            constraints=[hard("dup", ">=", 1, [lin("x1", 1), lin("x1", -1)])],
        )
        with pytest.raises(ValueError, match="trivially infeasible"):
            count_slack_bits(problem.constraints[0])

    def test_soft_infeasible_counts_zero_like_the_compiler(self):
        problem = make_problem(
            constraints=[
                soft("dup", ">=", 1, [lin("x1", 1), lin("x1", -1)], 1.0),
            ],
        )
        assert count_slack_bits(problem.constraints[0]) == 0
        compiled = BQMCompiler().compile(problem, hard_penalty=2.0)
        assert compiled.num_variables == estimate_compiled_variables(problem)


# ---------------------------------------------------------------------------
# Phase 3b: integer variables (spec §9.3, §9.4)
# ---------------------------------------------------------------------------


class TestLargeIntegerRange:
    """§9.3 row LARGE_INTEGER_RANGE: more than the bit threshold, BQM path only."""

    def make(self, upper: int) -> OptimizationProblem:
        return make_problem(variables=(integer("n", 0, upper), "x1"))

    def test_eleven_bits_warns(self):
        result = validate_problem_full(self.make(1024))  # 11 bits
        warnings = warnings_with(result, "LARGE_INTEGER_RANGE")
        assert len(warnings) == 1
        assert warnings[0].path == "variables[0]"
        assert "11 encoding bits" in warnings[0].message

    def test_ten_bits_does_not_warn(self):
        result = validate_problem_full(self.make(1023))  # 10 bits
        assert "LARGE_INTEGER_RANGE" not in warning_codes(result)

    def test_cqm_path_does_not_warn(self):
        result = validate_problem_full(self.make(1024), model_type="cqm")
        assert "LARGE_INTEGER_RANGE" not in warning_codes(result)

    def test_negative_lower_bound_counts_the_span(self):
        problem = make_problem(variables=(integer("n", -1024, 1023), "x1"))  # span 2047
        assert "LARGE_INTEGER_RANGE" in warning_codes(validate_problem_full(problem))


class TestIntegerQuadraticBlowup:
    """§9.3 row INTEGER_QUADRATIC_BLOWUP: estimated interactions above the threshold."""

    def coupled(self, upper: int) -> OptimizationProblem:
        return make_problem(
            variables=(integer("n", 0, upper), integer("m", 0, upper), "x1"),
            constraints=[hard("cap", "<=", 1000, [lin("n", 1), lin("m", 1)])],
        )

    def test_wide_integers_in_one_constraint_warn(self):
        # 31 + 31 bits plus 10 slack bits: 72 * 71 / 2 = 2556 interactions.
        result = validate_problem_full(self.coupled(2**31 - 1))
        warnings = warnings_with(result, "INTEGER_QUADRATIC_BLOWUP")
        assert len(warnings) == 1
        assert warnings[0].path == "variables"
        assert "2556" in warnings[0].message

    def test_small_integers_do_not_warn(self):
        result = validate_problem_full(self.coupled(7))
        assert "INTEGER_QUADRATIC_BLOWUP" not in warning_codes(result)

    def test_cqm_path_does_not_warn(self):
        result = validate_problem_full(self.coupled(2**31 - 1), model_type="cqm")
        assert "INTEGER_QUADRATIC_BLOWUP" not in warning_codes(result)

    def test_all_binary_problem_never_warns_however_dense(self):
        names = tuple(f"x{index}" for index in range(1, 101))
        problem = make_problem(
            variables=names,
            constraints=[hard("one", "==", 1, [lin(name, 1) for name in names])],
        )
        result = validate_problem_full(problem)
        assert "INTEGER_QUADRATIC_BLOWUP" not in warning_codes(result)

    def test_integer_warning_texts_name_no_backend_and_no_number(self):
        for code in ("LARGE_INTEGER_RANGE", "INTEGER_QUADRATIC_BLOWUP"):
            action = problem_validator._WARNING_RECOMMENDED_ACTIONS[code]
            assert not any(char.isdigit() for char in action)


class TestIntegerEstimates:
    """§9.4: the estimate follows the model type."""

    def make(self, *constraints: dict) -> OptimizationProblem:
        return make_problem(
            variables=(integer("x", 0, 7), integer("y", 0, 3), "b"),
            linear=[lin("x", 1)],
            constraints=list(constraints),
        )

    def test_bqm_estimate_counts_encoding_and_slack_bits(self):
        # 3 + 2 + 1 encoding bits, plus 3 slack bits for x + y <= 5.
        result = validate_problem_full(
            self.make(hard("cap", "<=", 5, [lin("x", 1), lin("y", 1)]))
        )
        assert result.model_type == "bqm"
        assert result.estimated_compiled_variables == 9

    @pytest.mark.parametrize(
        "constraint, extra",
        [
            # soft inequality on an integer with a positive slack range: +1
            (soft("cap", "<=", 5, [lin("x", 1), lin("y", 1)], 2.0), 1),
            (soft("cover", ">=", 2, [lin("x", 1)], 2.0), 1),
            # hard constraints are native, no slack
            (hard("cap", "<=", 5, [lin("x", 1), lin("y", 1)]), 0),
            # soft equality: objective form without a slack
            (soft("eq", "==", 5, [lin("x", 1)], 2.0), 0),
            # soft inequality over binary variables only: native dimod form
            (soft("bin", "<=", 0, [lin("b", 1)], 2.0), 0),
            # redundant: nothing is added
            (soft("loose", "<=", 100, [lin("x", 1)], 2.0), 0),
            # trivially infeasible soft (clamped): no slack variable
            (soft("never", ">=", 100, [lin("x", 1)], 2.0), 0),
            # slack range exactly zero: no slack variable
            (soft("tight", "<=", 0, [lin("x", 1)], 2.0), 0),
        ],
    )
    def test_cqm_estimate_adds_integer_soft_slacks(self, constraint, extra):
        result = validate_problem_full(self.make(constraint), model_type="cqm")
        assert result.model_type == "cqm"
        assert result.estimated_compiled_variables == 3 + extra

    def test_cqm_estimate_for_all_binary_is_the_variable_count(self):
        problem = make_problem(
            constraints=[soft("cap", "<=", 1, [lin("x1", 1), lin("x2", 1)], 2.0)]
        )
        assert validate_problem_full(problem, model_type="cqm").estimated_compiled_variables == 3


class TestDenseForQPUCountsBits:
    """§9.3: the constraint-width condition counts compiled bits, not variables."""

    def test_one_wide_integer_in_a_constraint_warns(self):
        # 31 encoding bits + 3 slack bits for n <= 5: 34 bits > 30 in one
        # constraint (the estimate also counts the untouched binary x1).
        problem = make_problem(
            variables=(integer("n", 0, 2**31 - 1), "x1"),
            constraints=[hard("cap", "<=", 5, [lin("n", 1)])],
        )
        result = validate_problem_full(problem, capabilities=caps(requires_embedding=True))
        assert result.estimated_compiled_variables == 35
        (warning,) = warnings_with(result, "DENSE_FOR_QPU")
        assert "34 compiled bits" in warning.message

    def test_narrow_integer_does_not_warn(self):
        problem = make_problem(
            variables=(integer("n", 0, 7), "x1"),
            constraints=[hard("cap", "<=", 5, [lin("n", 1)])],
        )
        result = validate_problem_full(problem, capabilities=caps(requires_embedding=True))
        assert "DENSE_FOR_QPU" not in warning_codes(result)


class TestBoundsAwareObjectiveScale:
    def test_objective_scale_uses_the_integer_ranges(self):
        problem = make_problem(
            variables=(integer("n", -5, 3), "x1"),
            linear=[lin("n", 2), lin("x1", 1)],
            quadratic=[quad("n", "x1", 1)],
        )
        result = validate_problem_full(problem)
        # Range formula (review F-06): 2 * (3 - (-5)) = 16 for the linear n term,
        # 1 * (1 - 0) = 1 for x1, and the product n * x1 spans -5..3 over the
        # corners {0, 0, -5, 3} -> 8; total 25.
        assert result.objective_scale == 25.0
        assert compute_objective_scale(problem.objective) == 4.0  # binary reading
