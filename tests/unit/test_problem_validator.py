"""Tests for the pre-compilation Problem Validator (spec §12)."""

import logging

import math

import pytest
from pydantic import ValidationError

from annealbridge.models import OptimizationProblem
from annealbridge.validation import validate_problem, validate_problem_full


def make_problem_dict() -> dict:
    """A complete, legal OptimizationProblem payload."""
    return {
        "name": "validator test problem",
        "variables": [
            {"name": "x1"},
            {"name": "x2"},
            {"name": "x3"},
        ],
        "objective": {
            "direction": "maximize",
            "linear_terms": [
                {"variable": "x1", "coefficient": 10},
                {"variable": "x2", "coefficient": 8},
            ],
            "quadratic_terms": [
                {"variable1": "x1", "variable2": "x2", "coefficient": -3},
            ],
        },
        "constraints": [
            {
                "id": "cap",
                "type": "hard",
                "terms": [
                    {"variable": "x1", "coefficient": 6},
                    {"variable": "x2", "coefficient": 5},
                ],
                "operator": "<=",
                "rhs": 10,
            },
            {
                "id": "prefer_x3",
                "type": "soft",
                "terms": [{"variable": "x3", "coefficient": 1}],
                "operator": "==",
                "rhs": 1,
                "weight": 4.0,
            },
        ],
    }


def validate_dict(data: dict):
    return validate_problem(OptimizationProblem.model_validate(data))


def codes(errors) -> list[str]:
    return [error.code for error in errors]


class TestValidProblem:
    def test_legal_problem_has_no_errors(self):
        assert validate_dict(make_problem_dict()) == []

    def test_equality_constraint_allows_non_integer_values(self):
        data = make_problem_dict()
        data["constraints"][1]["terms"][0]["coefficient"] = 0.5
        data["constraints"][1]["rhs"] = 0.5
        assert validate_dict(data) == []


class TestDuplicateVariable:
    def test_duplicate_variable_reported(self):
        data = make_problem_dict()
        data["variables"].append({"name": "x1"})
        errors = validate_dict(data)
        assert codes(errors) == ["DUPLICATE_VARIABLE"]
        assert errors[0].path == "variables[3]"
        assert "x1" in errors[0].message


class TestReservedVariableName:
    def test_dunder_prefix_reported(self):
        data = make_problem_dict()
        data["variables"].append({"name": "__slack_cap_0"})
        errors = validate_dict(data)
        assert codes(errors) == ["RESERVED_VARIABLE_NAME"]
        assert errors[0].path == "variables[3]"


class TestDuplicateConstraintId:
    def test_duplicate_id_reported(self):
        data = make_problem_dict()
        data["constraints"].append(dict(data["constraints"][0]))
        errors = validate_dict(data)
        assert codes(errors) == ["DUPLICATE_CONSTRAINT_ID"]
        assert errors[0].path == "constraints[2]"


class TestUnknownVariable:
    def test_unknown_in_objective_linear(self):
        data = make_problem_dict()
        data["objective"]["linear_terms"][1]["variable"] = "ghost"
        errors = validate_dict(data)
        assert codes(errors) == ["UNKNOWN_VARIABLE"]
        assert errors[0].path == "objective.linear_terms[1]"
        assert "ghost" in errors[0].message

    def test_unknown_in_objective_quadratic(self):
        data = make_problem_dict()
        data["objective"]["quadratic_terms"][0]["variable2"] = "ghost"
        errors = validate_dict(data)
        assert codes(errors) == ["UNKNOWN_VARIABLE"]
        assert errors[0].path == "objective.quadratic_terms[0]"

    def test_unknown_in_constraint_terms(self):
        data = make_problem_dict()
        data["constraints"][1]["terms"][0]["variable"] = "employee_xyz"
        errors = validate_dict(data)
        assert codes(errors) == ["UNKNOWN_VARIABLE"]
        assert errors[0].path == "constraints[1].terms[0]"


class TestSelfQuadraticTerm:
    def test_same_variable_twice_reported(self):
        data = make_problem_dict()
        data["objective"]["quadratic_terms"][0]["variable2"] = "x1"
        errors = validate_dict(data)
        assert codes(errors) == ["SELF_QUADRATIC_TERM"]
        assert errors[0].path == "objective.quadratic_terms[0]"


class TestNonFiniteCoefficient:
    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_objective_linear_coefficient(self, bad):
        data = make_problem_dict()
        data["objective"]["linear_terms"][0]["coefficient"] = bad
        errors = validate_dict(data)
        assert codes(errors) == ["NON_FINITE_COEFFICIENT"]
        assert errors[0].path == "objective.linear_terms[0]"

    def test_objective_quadratic_coefficient(self):
        data = make_problem_dict()
        data["objective"]["quadratic_terms"][0]["coefficient"] = float("inf")
        errors = validate_dict(data)
        assert codes(errors) == ["NON_FINITE_COEFFICIENT"]
        assert errors[0].path == "objective.quadratic_terms[0]"

    def test_objective_constant(self):
        data = make_problem_dict()
        data["objective"]["constant"] = float("nan")
        errors = validate_dict(data)
        assert codes(errors) == ["NON_FINITE_COEFFICIENT"]
        assert errors[0].path == "objective.constant"

    def test_constraint_rhs(self):
        data = make_problem_dict()
        data["constraints"][1]["rhs"] = float("inf")
        errors = validate_dict(data)
        assert codes(errors) == ["NON_FINITE_COEFFICIENT"]
        assert errors[0].path == "constraints[1].rhs"

    def test_soft_weight(self):
        data = make_problem_dict()
        data["constraints"][1]["weight"] = float("inf")
        errors = validate_dict(data)
        assert codes(errors) == ["NON_FINITE_COEFFICIENT"]
        assert errors[0].path == "constraints[1].weight"

    def test_inequality_non_finite_coefficient_not_double_reported(self):
        data = make_problem_dict()
        data["constraints"][0]["terms"][0]["coefficient"] = float("nan")
        errors = validate_dict(data)
        assert codes(errors) == ["NON_FINITE_COEFFICIENT"]


class TestEmptyConstraint:
    def test_empty_terms_reported(self):
        data = make_problem_dict()
        data["constraints"][0]["terms"] = []
        errors = validate_dict(data)
        assert codes(errors) == ["EMPTY_CONSTRAINT"]
        assert errors[0].path == "constraints[0]"


class TestNonIntegerInequality:
    @pytest.mark.parametrize("operator", ["<=", ">="])
    def test_non_integer_coefficient_reported(self, operator):
        data = make_problem_dict()
        data["constraints"][0]["operator"] = operator
        data["constraints"][0]["rhs"] = 5  # reachable for both operators
        data["constraints"][0]["terms"][1]["coefficient"] = 2.5
        errors = validate_dict(data)
        assert codes(errors) == ["NON_INTEGER_INEQUALITY"]
        assert errors[0].path == "constraints[0].terms[1]"

    def test_non_integer_rhs_reported(self):
        data = make_problem_dict()
        data["constraints"][0]["rhs"] = 10.5
        errors = validate_dict(data)
        assert codes(errors) == ["NON_INTEGER_INEQUALITY"]
        assert errors[0].path == "constraints[0].rhs"

    def test_integer_valued_floats_allowed(self):
        data = make_problem_dict()
        data["constraints"][0]["terms"][0]["coefficient"] = 6.0
        data["constraints"][0]["rhs"] = 10.0
        assert validate_dict(data) == []

    def test_equality_not_restricted(self):
        data = make_problem_dict()
        data["constraints"][1]["terms"][0]["coefficient"] = 1.5
        data["constraints"][1]["rhs"] = 1.5
        assert validate_dict(data) == []


class TestHardConstraintHasWeight:
    def test_hard_with_weight_reported(self):
        data = make_problem_dict()
        data["constraints"][0]["weight"] = 3.0
        errors = validate_dict(data)
        assert codes(errors) == ["HARD_CONSTRAINT_HAS_WEIGHT"]
        assert errors[0].path == "constraints[0].weight"


class TestSoftConstraintMissingWeight:
    def test_missing_weight_reported(self):
        data = make_problem_dict()
        del data["constraints"][1]["weight"]
        errors = validate_dict(data)
        assert codes(errors) == ["SOFT_CONSTRAINT_MISSING_WEIGHT"]
        assert errors[0].path == "constraints[1].weight"

    @pytest.mark.parametrize("bad_weight", [0, -1.5])
    def test_non_positive_weight_reported(self, bad_weight):
        data = make_problem_dict()
        data["constraints"][1]["weight"] = bad_weight
        errors = validate_dict(data)
        assert codes(errors) == ["SOFT_CONSTRAINT_MISSING_WEIGHT"]
        assert errors[0].path == "constraints[1].weight"


class TestInvalidSolverPreference:
    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("top_k", 0),
            ("num_reads", 0),
            ("num_sweeps", -1),
            ("max_retries", -1),
            ("penalty_multiplier", 0.0),
        ],
    )
    def test_bad_preference_reported(self, field, bad_value):
        data = make_problem_dict()
        data["solver"] = {field: bad_value}
        errors = validate_dict(data)
        assert codes(errors) == ["INVALID_SOLVER_PREFERENCE"]
        assert errors[0].path == f"solver.{field}"

    def test_max_retries_zero_is_legal(self):
        data = make_problem_dict()
        data["solver"] = {"max_retries": 0}
        assert validate_dict(data) == []


class TestTriviallyInfeasible:
    def test_le_with_negative_rhs_below_reach(self):
        # all-positive coefficients: lhs_min = 0 > rhs = -1
        data = make_problem_dict()
        data["constraints"][0]["rhs"] = -1
        errors = validate_dict(data)
        assert codes(errors) == ["TRIVIALLY_INFEASIBLE"]
        assert errors[0].path == "constraints[0]"

    def test_ge_with_rhs_above_reach(self):
        # lhs_max = 6 + 5 = 11 < rhs = 12
        data = make_problem_dict()
        data["constraints"][0]["operator"] = ">="
        data["constraints"][0]["rhs"] = 12
        errors = validate_dict(data)
        assert codes(errors) == ["TRIVIALLY_INFEASIBLE"]

    def test_eq_with_rhs_outside_range(self):
        # with a negative coefficient: lhs range is [-4, 6], rhs = 7 unreachable
        data = make_problem_dict()
        data["constraints"][0] = {
            "id": "cap",
            "type": "hard",
            "terms": [
                {"variable": "x1", "coefficient": 6},
                {"variable": "x2", "coefficient": -4},
            ],
            "operator": "==",
            "rhs": 7,
        }
        errors = validate_dict(data)
        assert codes(errors) == ["TRIVIALLY_INFEASIBLE"]

    def test_negative_coefficients_extend_feasible_range(self):
        # lhs_min = -4, so rhs = -1 is reachable for <=
        data = make_problem_dict()
        data["constraints"][0]["terms"][1]["coefficient"] = -4
        data["constraints"][0]["rhs"] = -1
        assert validate_dict(data) == []

    def test_soft_constraint_not_checked(self):
        data = make_problem_dict()
        data["constraints"][1]["rhs"] = 99  # unreachable, but soft
        assert validate_dict(data) == []

    def test_repeated_variable_is_judged_on_accumulated_coefficients(self):
        # x1 appears as +1 and -1: per-term ranges say [-1, 1] (feasible) but
        # the compiler sums them to 0, so ">= 1" can never hold.
        data = make_problem_dict()
        data["constraints"][0] = {
            "id": "dup",
            "type": "hard",
            "terms": [
                {"variable": "x1", "coefficient": 1},
                {"variable": "x1", "coefficient": -1},
            ],
            "operator": ">=",
            "rhs": 1,
        }
        errors = validate_dict(data)
        assert codes(errors) == ["TRIVIALLY_INFEASIBLE"]
        assert errors[0].path == "constraints[0]"
        assert errors[0].recommended_action

    def test_message_reports_the_users_operator_and_rhs(self):
        data = make_problem_dict()
        data["constraints"][0] = {
            "id": "dup",
            "type": "hard",
            "terms": [
                {"variable": "x1", "coefficient": 1},
                {"variable": "x1", "coefficient": -1},
            ],
            "operator": ">=",
            "rhs": 1,
        }
        message = validate_dict(data)[0].message
        assert ">= 1" in message
        assert "-1" not in message  # never the compiler's normalized form

    def test_repeated_variable_with_reachable_accumulated_range_passes(self):
        # x1: +2 and -1 accumulate to +1, so "<= 0" is satisfiable (x1 = 0).
        data = make_problem_dict()
        data["constraints"][0] = {
            "id": "dup",
            "type": "hard",
            "terms": [
                {"variable": "x1", "coefficient": 2},
                {"variable": "x1", "coefficient": -1},
            ],
            "operator": "<=",
            "rhs": 0,
        }
        assert validate_dict(data) == []

    def test_equality_uses_accumulated_coefficients_too(self):
        # +1 and -1 sum to 0, so "== 1" is unreachable even though the raw
        # per-term range [-1, 1] contains 1.
        data = make_problem_dict()
        data["constraints"][0] = {
            "id": "dup",
            "type": "hard",
            "terms": [
                {"variable": "x1", "coefficient": 1},
                {"variable": "x1", "coefficient": -1},
            ],
            "operator": "==",
            "rhs": 1,
        }
        assert codes(validate_dict(data)) == ["TRIVIALLY_INFEASIBLE"]


class TestErrorCollection:
    def test_all_errors_collected_in_one_pass(self):
        data = make_problem_dict()
        data["variables"].append({"name": "x1"})  # DUPLICATE_VARIABLE
        data["objective"]["linear_terms"][0]["variable"] = "ghost"  # UNKNOWN_VARIABLE
        data["constraints"][0]["weight"] = 1.0  # HARD_CONSTRAINT_HAS_WEIGHT
        data["constraints"][1]["weight"] = -2  # SOFT_CONSTRAINT_MISSING_WEIGHT
        data["solver"] = {"top_k": 0}  # INVALID_SOLVER_PREFERENCE
        found = codes(validate_dict(data))
        assert set(found) == {
            "DUPLICATE_VARIABLE",
            "UNKNOWN_VARIABLE",
            "HARD_CONSTRAINT_HAS_WEIGHT",
            "SOFT_CONSTRAINT_MISSING_WEIGHT",
            "INVALID_SOLVER_PREFERENCE",
        }
        assert len(found) == 5


class TestDuplicateObjectiveTermWarning:
    """Spec §8: duplicates are merged, never rejected; one log line each."""

    def test_duplicate_linear_term_warns_but_passes(self, caplog):
        data = make_problem_dict()
        data["objective"]["linear_terms"].append({"variable": "x1", "coefficient": 2})
        with caplog.at_level(logging.WARNING, logger="annealbridge.validation"):
            errors = validate_dict(data)
        assert errors == []
        assert any("x1" in record.getMessage() for record in caplog.records)

    def test_duplicate_quadratic_pair_warns_but_passes(self, caplog):
        data = make_problem_dict()
        data["objective"]["quadratic_terms"].append(
            {"variable1": "x2", "variable2": "x1", "coefficient": 1}
        )
        with caplog.at_level(logging.WARNING, logger="annealbridge.validation"):
            errors = validate_dict(data)
        assert errors == []
        assert any("x1" in record.getMessage() for record in caplog.records)

    def test_full_validation_logs_each_duplicate_exactly_once(self, caplog):
        # One helper feeds both the structured warning and the log line, so
        # the full entry point must not log the same duplicate twice.
        data = make_problem_dict()
        data["objective"]["linear_terms"].append({"variable": "x1", "coefficient": 2})
        problem = OptimizationProblem.model_validate(data)
        with caplog.at_level(logging.WARNING, logger="annealbridge.validation"):
            result = validate_problem_full(problem)
        merged = [w for w in result.warnings if w.code == "DUPLICATE_TERM_MERGED"]
        assert len(merged) == 1
        logged = [r for r in caplog.records if "x1" in r.getMessage()]
        assert len(logged) == 1
        assert logged[0].getMessage() == merged[0].message


class TestNoVariables:
    def test_empty_variables_is_rejected(self):
        data = make_problem_dict()
        data["variables"] = []
        data["objective"] = {"direction": "minimize", "linear_terms": []}
        data["constraints"] = []
        errors = validate_dict(data)
        assert codes(errors) == ["NO_VARIABLES"]
        assert errors[0].path == "variables"
        assert errors[0].recommended_action

    def test_no_variables_is_collected_alongside_other_errors(self):
        data = make_problem_dict()
        data["variables"] = []
        errors = validate_dict(data)
        assert "NO_VARIABLES" in codes(errors)
        assert "UNKNOWN_VARIABLE" in codes(errors)


class TestBackendOptionRanges:
    """annealing_time_us / chain_strength / time_limit_seconds must be finite > 0.

    The Pydantic layer rejects NaN / ±inf (they are not numbers at all and
    would defeat every ``value > limit`` policy comparison); the validator
    owns the semantic ``> 0`` rule so it surfaces as invalid_problem with a
    recommended_action rather than a bare type error.
    """

    OPTIONS = [
        ("dwave_qpu", "annealing_time_us"),
        ("dwave_qpu", "chain_strength"),
        ("leap_hybrid_bqm", "time_limit_seconds"),
    ]

    @pytest.mark.parametrize(("options_field", "field"), OPTIONS)
    @pytest.mark.parametrize("bad_value", [-5, 0])
    def test_non_positive_is_invalid_solver_preference(
        self, options_field, field, bad_value
    ):
        data = make_problem_dict()
        data["solver"] = {options_field: {field: bad_value}}
        errors = validate_dict(data)
        assert codes(errors) == ["INVALID_SOLVER_PREFERENCE"]
        assert errors[0].path == f"solver.{options_field}.{field}"
        assert str(bad_value) in errors[0].message
        assert errors[0].recommended_action

    @pytest.mark.parametrize(("options_field", "field"), OPTIONS)
    @pytest.mark.parametrize("bad_value", [math.nan, math.inf])
    def test_non_finite_is_rejected_by_the_schema(
        self, options_field, field, bad_value
    ):
        data = make_problem_dict()
        data["solver"] = {options_field: {field: bad_value}}
        with pytest.raises(ValidationError) as excinfo:
            OptimizationProblem.model_validate(data)
        assert excinfo.value.errors()[0]["type"] == "finite_number"

    @pytest.mark.parametrize(("options_field", "field"), OPTIONS)
    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_json_literal_is_rejected_by_the_schema(
        self, options_field, field, literal
    ):
        # A raw JSON payload (the MCP path) cannot smuggle NaN/Infinity in
        # either: nan > limit is False, so it would bypass policy limits.
        data = make_problem_dict()
        data["solver"] = {options_field: {field: 1}}
        import json

        payload = json.dumps(data).replace(f'"{field}": 1', f'"{field}": {literal}')
        assert literal in payload
        with pytest.raises(ValidationError):
            OptimizationProblem.model_validate_json(payload)

    @pytest.mark.parametrize(("options_field", "field"), OPTIONS)
    def test_positive_value_is_legal(self, options_field, field):
        data = make_problem_dict()
        data["solver"] = {options_field: {field: 2.5}}
        assert validate_dict(data) == []

    def test_omitted_options_are_legal(self):
        data = make_problem_dict()
        data["solver"] = {"backend": "dwave_qpu", "dwave_qpu": {}}
        assert validate_dict(data) == []


class TestRecommendedAction:
    def test_every_collected_error_carries_a_recommended_action(self):
        data = make_problem_dict()
        data["variables"].append({"name": "x1"})
        data["objective"]["linear_terms"][0]["variable"] = "ghost"
        data["constraints"][0]["weight"] = 1.0
        data["constraints"][1]["weight"] = -2
        data["solver"] = {"top_k": 0}
        errors = validate_dict(data)
        assert len(errors) == 5
        for error in errors:
            assert isinstance(error.recommended_action, str)
            assert error.recommended_action.strip()
            assert error.retryable is False


# ---------------------------------------------------------------------------
# Phase 3b: integer variables (spec §7, §9.1, §9.2)
# ---------------------------------------------------------------------------


def integer_problem_dict(version: str = "1.1", **overrides) -> dict:
    """The legal payload with ``x1`` turned into an integer in ``[0, 3]``."""
    data = make_problem_dict()
    data["version"] = version
    data["variables"][0] = {
        "name": "x1",
        "type": "integer",
        "lower_bound": 0,
        "upper_bound": 3,
        **overrides,
    }
    return data


class TestIntegerBoundsErrors:
    def test_legal_integer_problem_passes(self):
        assert validate_dict(integer_problem_dict()) == []

    def test_version_1_1_with_only_binary_variables_is_legal(self):
        data = make_problem_dict()
        data["version"] = "1.1"
        assert validate_dict(data) == []

    @pytest.mark.parametrize(
        "overrides",
        [
            {"upper_bound": None},
            {"lower_bound": None},
            {"lower_bound": None, "upper_bound": None},
        ],
    )
    def test_integer_bounds_missing(self, overrides):
        errors = validate_dict(integer_problem_dict(**overrides))
        assert codes(errors) == ["INTEGER_BOUNDS_MISSING"]
        assert errors[0].path == "variables[0]"
        assert errors[0].recommended_action == (
            "An integer variable needs both lower_bound and upper_bound; add "
            "them, or make the variable binary."
        )

    @pytest.mark.parametrize("lower, upper", [(3, 3), (4, 3), (-1, -5)])
    def test_integer_bounds_invalid(self, lower, upper):
        errors = validate_dict(integer_problem_dict(lower_bound=lower, upper_bound=upper))
        assert codes(errors) == ["INTEGER_BOUNDS_INVALID"]
        assert errors[0].path == "variables[0]"
        assert errors[0].recommended_action.startswith(
            "upper_bound must be greater than lower_bound"
        )

    @pytest.mark.parametrize(
        "bounds",
        [
            {"lower_bound": 0},
            {"upper_bound": 1},
            {"lower_bound": 0, "upper_bound": 1},
        ],
    )
    def test_bounds_on_binary(self, bounds):
        data = make_problem_dict()
        data["version"] = "1.1"
        data["variables"][1] = {"name": "x2", **bounds}
        errors = validate_dict(data)
        assert codes(errors) == ["BOUNDS_ON_BINARY"]
        assert errors[0].path == "variables[1]"
        assert errors[0].recommended_action.startswith(
            "Binary variables are 0/1 and take no bounds"
        )

    @pytest.mark.parametrize(
        "lower, upper",
        [(0, 2**31), (-(2**31), 0), (-(2**40), 2**40)],
    )
    def test_integer_range_too_large(self, lower, upper):
        errors = validate_dict(integer_problem_dict(lower_bound=lower, upper_bound=upper))
        assert codes(errors) == ["INTEGER_RANGE_TOO_LARGE"]
        assert errors[0].path == "variables[0]"
        assert errors[0].recommended_action == (
            "Integer bounds must lie within ±(2^31-1); tighten the bounds or "
            "rescale the variable's unit."
        )

    def test_bounds_exactly_at_the_limit_are_legal(self):
        limit = 2**31 - 1
        assert validate_dict(integer_problem_dict(lower_bound=-limit, upper_bound=limit)) == []

    def test_invalid_and_too_large_are_reported_together(self):
        errors = validate_dict(integer_problem_dict(lower_bound=0, upper_bound=-(2**31)))
        assert sorted(codes(errors)) == ["INTEGER_BOUNDS_INVALID", "INTEGER_RANGE_TOO_LARGE"]

    def test_integer_requires_version_1_1(self):
        errors = validate_dict(integer_problem_dict(version="1.0"))
        assert codes(errors) == ["INTEGER_REQUIRES_VERSION_1_1"]
        assert errors[0].path == "version"
        assert errors[0].recommended_action == (
            'Integer variables require schema version 1.1; set version to "1.1".'
        )

    def test_version_error_is_collected_with_bounds_errors(self):
        errors = validate_dict(integer_problem_dict(version="1.0", upper_bound=None))
        assert sorted(codes(errors)) == [
            "INTEGER_BOUNDS_MISSING",
            "INTEGER_REQUIRES_VERSION_1_1",
        ]


class TestSelfQuadraticTermByType:
    def test_binary_square_is_rejected(self):
        data = make_problem_dict()
        data["objective"]["quadratic_terms"] = [
            {"variable1": "x2", "variable2": "x2", "coefficient": 1}
        ]
        assert codes(validate_dict(data)) == ["SELF_QUADRATIC_TERM"]

    def test_integer_square_is_legal(self):
        data = integer_problem_dict()
        data["objective"]["quadratic_terms"] = [
            {"variable1": "x1", "variable2": "x1", "coefficient": 1}
        ]
        assert validate_dict(data) == []

    def test_unknown_variable_square_keeps_the_binary_verdict(self):
        data = make_problem_dict()
        data["objective"]["quadratic_terms"] = [
            {"variable1": "ghost", "variable2": "ghost", "coefficient": 1}
        ]
        assert sorted(codes(validate_dict(data))) == [
            "SELF_QUADRATIC_TERM",
            "UNKNOWN_VARIABLE",
            "UNKNOWN_VARIABLE",
        ]


class TestTriviallyInfeasibleWithBounds:
    def _with_constraint(self, operator: str, rhs: float, coefficient: float = 2, **bounds):
        data = integer_problem_dict(**bounds)
        data["constraints"][0] = {
            "id": "range",
            "type": "hard",
            "terms": [{"variable": "x1", "coefficient": coefficient}],
            "operator": operator,
            "rhs": rhs,
        }
        return data

    def test_le_reachable_within_integer_range(self):
        # x1 in [0, 3]: 2 * x1 <= 7 holds for x1 <= 3.
        assert validate_dict(self._with_constraint("<=", 7)) == []

    def test_ge_beyond_integer_range_is_infeasible(self):
        # 2 * x1 >= 7 needs x1 >= 3.5, above the upper bound 3.
        errors = validate_dict(self._with_constraint(">=", 7))
        assert codes(errors) == ["TRIVIALLY_INFEASIBLE"]
        assert errors[0].path == "constraints[0]"
        assert "[0.0, 6.0]" in errors[0].message
        assert ">= 7" in errors[0].message

    def test_ge_reachable_when_binary_would_not_be(self):
        # For a binary x1 "2 * x1 >= 4" is impossible; for x1 in [0, 3] it holds.
        assert validate_dict(self._with_constraint(">=", 4)) == []

    def test_negative_lower_bound_extends_the_range_downwards(self):
        assert validate_dict(
            self._with_constraint("<=", -3, lower_bound=-2, upper_bound=3)
        ) == []
        errors = validate_dict(
            self._with_constraint("<=", -5, lower_bound=-2, upper_bound=3)
        )
        assert codes(errors) == ["TRIVIALLY_INFEASIBLE"]
        assert "[-4.0, 6.0]" in errors[0].message

    def test_equality_uses_the_integer_range(self):
        assert validate_dict(self._with_constraint("==", 6)) == []
        assert codes(validate_dict(self._with_constraint("==", 8))) == [
            "TRIVIALLY_INFEASIBLE"
        ]

    def test_constraint_on_a_variable_with_illegal_bounds_is_skipped(self):
        # x1 has no bounds: it is reported once, and no range judgement is
        # made on a constraint that mentions it (3b §9.2).
        data = self._with_constraint("<=", -1, upper_bound=None)
        assert codes(validate_dict(data)) == ["INTEGER_BOUNDS_MISSING"]

    def test_other_constraints_are_still_judged(self):
        data = self._with_constraint("<=", -1, upper_bound=None)
        data["constraints"].append(
            {
                "id": "impossible",
                "type": "hard",
                "terms": [{"variable": "x2", "coefficient": 1}],
                "operator": ">=",
                "rhs": 2,
            }
        )
        assert sorted(codes(validate_dict(data))) == [
            "INTEGER_BOUNDS_MISSING",
            "TRIVIALLY_INFEASIBLE",
        ]
