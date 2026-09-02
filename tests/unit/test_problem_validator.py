"""Tests for the pre-compilation Problem Validator (spec §12)."""

import logging

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.validation import validate_problem


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
