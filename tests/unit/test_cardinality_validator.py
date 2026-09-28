"""Problem validation of ``cardinality_constraints`` (schema 1.2 spec §6, §11).

Spec 2026-09-25 §13 item 8. Every error and warning code a cardinality
constraint can produce gets a triggering and a non-triggering case, with
the path pointing at what the user wrote (``cardinality_constraints[i]``
and ``...variables[j]``) and the cardinality message template of §6.5:

* new errors: FEATURE_REQUIRES_NEWER_VERSION (three message variants),
  CARDINALITY_VARIABLE_NOT_BINARY, DUPLICATE_CARDINALITY_VARIABLE;
* reused errors: EMPTY_CONSTRAINT, UNKNOWN_VARIABLE,
  HARD_CONSTRAINT_HAS_WEIGHT, SOFT_CONSTRAINT_MISSING_WEIGHT,
  NON_FINITE_COEFFICIENT, DUPLICATE_CONSTRAINT_ID (across both lists),
  TRIVIALLY_INFEASIBLE (only without structural errors);
* reused warnings: SOFT_ALWAYS_VIOLATED, REDUNDANT_CONSTRAINT,
  SOFT_WEIGHT_SMALL;
* the new warning CARDINALITY_FORM_AVAILABLE and its place in the list;
* an out-of-range ``rhs`` is a schema error (INVALID_FIELD_VALUE), never an
  exception.
"""

import json
import math

import pytest

from annealbridge.interfaces.problem_input import parse_problem
from annealbridge.models import OptimizationProblem, SolverCapabilities
from annealbridge.models.error_catalog import RECOMMENDED_ACTIONS
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import validate_problem, validate_problem_full
from annealbridge.validation.problem_validator import _WARNING_RECOMMENDED_ACTIONS

INTEGER_K = {"name": "k", "type": "integer", "lower_bound": 0, "upper_bound": 3}


def lin(variable: str, coefficient: float) -> dict:
    return {"variable": variable, "coefficient": coefficient}


def card(
    variables: list[str],
    operator: str = "<=",
    rhs=2,
    *,
    constraint_id: str = "c",
    kind: str = "hard",
    **extra,
) -> dict:
    return {
        "id": constraint_id,
        "type": kind,
        "variables": variables,
        "operator": operator,
        "rhs": rhs,
        **extra,
    }


def payload(
    cardinality: list[dict] | None = None,
    *,
    constraints: list[dict] | None = None,
    variables: list | None = None,
    version: str | None = "1.2",
    linear: list[dict] | None = None,
) -> dict:
    declared = variables if variables is not None else ["x0", "x1", "x2", "x3", INTEGER_K]
    data: dict = {
        "name": "cardinality validator",
        "variables": [entry if isinstance(entry, dict) else {"name": entry} for entry in declared],
        "objective": {
            "direction": "minimize",
            "linear_terms": linear if linear is not None else [lin("x0", 10)],
        },
        "constraints": constraints or [],
    }
    if cardinality is not None:
        data["cardinality_constraints"] = cardinality
    if version is not None:
        data["version"] = version
    return data


def build(*args, **kwargs) -> OptimizationProblem:
    return OptimizationProblem.model_validate(payload(*args, **kwargs))


def findings(items, code: str) -> list:
    return [item for item in items if item.code == code]


def codes(items) -> list[str]:
    return [item.code for item in items]


def warnings_of(problem: OptimizationProblem, **kwargs) -> list:
    result = validate_problem_full(problem, **kwargs)
    assert result.valid is True, result.errors
    return result.warnings


# --------------------------------------------------------------------------
# Structural errors
# --------------------------------------------------------------------------


class TestEmptyConstraint:
    def test_no_variables(self):
        errors = validate_problem(build([card([], "==", 1, constraint_id="empty")]))
        assert codes(errors) == ["EMPTY_CONSTRAINT"]
        (error,) = errors
        assert error.path == "cardinality_constraints[0]"
        assert error.message == "Cardinality constraint empty lists no variables"
        # §6.5: the recommended action is the catalog's, unchanged.
        assert error.recommended_action == RECOMMENDED_ACTIONS["EMPTY_CONSTRAINT"]

    def test_one_variable_is_not_empty(self):
        assert validate_problem(build([card(["x0"], "<=", 1)])) == []


class TestUnknownVariable:
    def test_undeclared_name(self):
        errors = validate_problem(build([card(["x0", "ghost"], "<=", 1)]))
        assert codes(errors) == ["UNKNOWN_VARIABLE"]
        (error,) = errors
        assert error.path == "cardinality_constraints[0].variables[1]"
        assert error.message == (
            "Cardinality constraint c counts variable 'ghost', which is not declared in variables"
        )
        assert error.recommended_action == RECOMMENDED_ACTIONS["UNKNOWN_VARIABLE"]

    def test_index_of_the_constraint_is_in_the_path(self):
        errors = validate_problem(
            build([card(["x0"], constraint_id="a"), card(["x1", "x2", "nope"], constraint_id="b")])
        )
        assert [(e.code, e.path) for e in errors] == [
            ("UNKNOWN_VARIABLE", "cardinality_constraints[1].variables[2]")
        ]

    def test_declared_names_pass(self):
        assert validate_problem(build([card(["x0", "x1", "x2"], "<=", 2)])) == []


class TestVariableNotBinary:
    def test_integer_variable(self):
        errors = validate_problem(build([card(["x0", "k"], "<=", 1, constraint_id="mixed")]))
        assert codes(errors) == ["CARDINALITY_VARIABLE_NOT_BINARY"]
        (error,) = errors
        assert error.path == "cardinality_constraints[0].variables[1]"
        assert "'k'" in error.message
        assert "integer variable" in error.message
        assert "only binary" in error.message
        assert error.recommended_action == RECOMMENDED_ACTIONS["CARDINALITY_VARIABLE_NOT_BINARY"]

    def test_binary_variables_pass(self):
        assert findings(validate_problem(build([card(["x0", "x1"])])), "CARDINALITY_VARIABLE_NOT_BINARY") == []


class TestDuplicateVariable:
    def test_second_occurrence_is_reported(self):
        errors = validate_problem(build([card(["x0", "x1", "x0"], "<=", 1, constraint_id="dup")]))
        assert codes(errors) == ["DUPLICATE_CARDINALITY_VARIABLE"]
        (error,) = errors
        assert error.path == "cardinality_constraints[0].variables[2]"
        assert "'x0'" in error.message
        assert "more than once" in error.message
        assert error.recommended_action == RECOMMENDED_ACTIONS["DUPLICATE_CARDINALITY_VARIABLE"]

    def test_every_later_occurrence_is_reported(self):
        errors = validate_problem(build([card(["x1", "x1", "x2", "x1"])]))
        assert [(e.code, e.path) for e in errors] == [
            ("DUPLICATE_CARDINALITY_VARIABLE", "cardinality_constraints[0].variables[1]"),
            ("DUPLICATE_CARDINALITY_VARIABLE", "cardinality_constraints[0].variables[3]"),
        ]

    def test_distinct_names_pass(self):
        assert validate_problem(build([card(["x0", "x1", "x2", "x3"])])) == []

    def test_the_same_variable_in_two_constraints_is_fine(self):
        assert validate_problem(build([card(["x0", "x1"], constraint_id="a"), card(["x0", "x2"], constraint_id="b")])) == []


class TestWeights:
    def test_hard_with_weight(self):
        errors = validate_problem(build([card(["x0", "x1"], weight=2)]))
        assert [(e.code, e.path) for e in errors] == [
            ("HARD_CONSTRAINT_HAS_WEIGHT", "cardinality_constraints[0].weight")
        ]
        assert "Hard constraint c" in errors[0].message

    def test_hard_without_weight_passes(self):
        assert validate_problem(build([card(["x0", "x1"])])) == []

    @pytest.mark.parametrize("weight", [None, 0, -1.5], ids=["missing", "zero", "negative"])
    def test_soft_without_a_positive_weight(self, weight):
        extra = {} if weight is None else {"weight": weight}
        errors = validate_problem(build([card(["x0", "x1"], kind="soft", **extra)]))
        assert [(e.code, e.path) for e in errors] == [
            ("SOFT_CONSTRAINT_MISSING_WEIGHT", "cardinality_constraints[0].weight")
        ]
        assert "Soft constraint c requires a weight > 0" in errors[0].message

    def test_soft_with_a_positive_weight_passes(self):
        assert validate_problem(build([card(["x0", "x1"], kind="soft", weight=0.5)])) == []

    @pytest.mark.parametrize("weight", [math.nan, math.inf], ids=["nan", "inf"])
    def test_non_finite_weight(self, weight):
        errors = validate_problem(build([card(["x0", "x1"], kind="soft", weight=weight)]))
        assert [(e.code, e.path) for e in errors] == [
            ("NON_FINITE_COEFFICIENT", "cardinality_constraints[0].weight")
        ]


class TestDuplicateConstraintId:
    def test_across_the_two_lists(self):
        problem = build(
            [card(["x0", "x1"], constraint_id="shared")],
            constraints=[
                {"id": "shared", "type": "hard", "terms": [lin("x2", 1)], "operator": "<=", "rhs": 1}
            ],
        )
        errors = validate_problem(problem)
        assert [(e.code, e.path) for e in errors] == [
            ("DUPLICATE_CONSTRAINT_ID", "cardinality_constraints[0]")
        ]
        assert errors[0].message == "Constraint id shared is used more than once"

    def test_within_the_cardinality_list(self):
        errors = validate_problem(
            build([card(["x0"], constraint_id="twice"), card(["x1"], constraint_id="twice")])
        )
        assert [(e.code, e.path) for e in errors] == [
            ("DUPLICATE_CONSTRAINT_ID", "cardinality_constraints[1]")
        ]

    def test_distinct_ids_pass(self):
        problem = build(
            [card(["x0", "x1"], constraint_id="card")],
            constraints=[
                {"id": "lin", "type": "hard", "terms": [lin("x2", 1)], "operator": "<=", "rhs": 1}
            ],
        )
        assert validate_problem(problem) == []


# --------------------------------------------------------------------------
# Range verdicts
# --------------------------------------------------------------------------


class TestTriviallyInfeasible:
    @pytest.mark.parametrize(
        "operator, rhs",
        [("==", 5), ("==", -1), (">=", 4), ("<=", -1)],
    )
    def test_hard_count_out_of_range(self, operator, rhs):
        errors = validate_problem(build([card(["x0", "x1", "x2"], operator, rhs, constraint_id="h")]))
        assert [(e.code, e.path) for e in errors] == [
            ("TRIVIALLY_INFEASIBLE", "cardinality_constraints[0]")
        ]
        assert errors[0].message == (
            f"Hard cardinality constraint h counts 3 variables, so between 0 and 3 "
            f"are chosen, but requires {operator} {rhs}"
        )

    @pytest.mark.parametrize(
        "operator, rhs", [("==", 3), ("==", 0), (">=", 3), ("<=", 0), (">=", 0), ("<=", 3)]
    )
    def test_hard_count_in_range_passes(self, operator, rhs):
        assert validate_problem(build([card(["x0", "x1", "x2"], operator, rhs)])) == []

    @pytest.mark.parametrize(
        "variables",
        [["x0", "ghost"], ["x0", "x0"], ["x0", "k"], []],
        ids=["unknown", "duplicate", "integer", "empty"],
    )
    def test_not_reported_alongside_a_structural_error(self, variables):
        errors = validate_problem(build([card(variables, "==", 5)]))
        assert len(errors) == 1
        assert "TRIVIALLY_INFEASIBLE" not in codes(errors)

    def test_not_reported_alongside_a_weight_error(self):
        errors = validate_problem(build([card(["x0", "x1"], "==", 5, weight=1)]))
        assert codes(errors) == ["HARD_CONSTRAINT_HAS_WEIGHT"]

    def test_soft_out_of_range_is_no_error(self):
        assert validate_problem(build([card(["x0", "x1"], "==", 5, kind="soft", weight=2)])) == []


class TestSoftAlwaysViolated:
    @pytest.mark.parametrize("operator, rhs", [("==", 5), (">=", 4), ("<=", -1)])
    def test_soft_count_out_of_range(self, operator, rhs):
        problem = build([card(["x0", "x1", "x2"], operator, rhs, constraint_id="s", kind="soft", weight=2)])
        (warning,) = findings(warnings_of(problem), "SOFT_ALWAYS_VIOLATED")
        assert warning.path == "cardinality_constraints[0]"
        assert warning.message == (
            "Soft cardinality constraint s can never be satisfied: it counts 3 "
            "variables, so between 0 and 3 are chosen, but requires "
            f"{operator} {rhs}; every solution pays weight 2.0"
        )
        assert warning.recommended_action == _WARNING_RECOMMENDED_ACTIONS["SOFT_ALWAYS_VIOLATED"]

    def test_soft_count_in_range_is_silent(self):
        problem = build([card(["x0", "x1", "x2"], "==", 3, kind="soft", weight=2)])
        assert findings(warnings_of(problem), "SOFT_ALWAYS_VIOLATED") == []


class TestRedundant:
    @pytest.mark.parametrize(
        "operator, rhs, kind",
        [("<=", 3, "hard"), ("<=", 7, "hard"), (">=", 0, "hard"), (">=", -2, "soft")],
    )
    def test_always_holds(self, operator, rhs, kind):
        extra = {"weight": 1} if kind == "soft" else {}
        problem = build([card(["x0", "x1", "x2"], operator, rhs, constraint_id="r", kind=kind, **extra)])
        (warning,) = findings(warnings_of(problem), "REDUNDANT_CONSTRAINT")
        assert warning.path == "cardinality_constraints[0]"
        message = warning.message
        assert message.startswith("Cardinality constraint r counts 3 variables")
        assert f"requires {operator} {rhs}, which always holds" in message
        assert "lhs maximum" not in message

    @pytest.mark.parametrize("operator, rhs", [("<=", 2), (">=", 1), ("==", 3)])
    def test_binding_constraint_is_not_redundant(self, operator, rhs):
        problem = build([card(["x0", "x1", "x2"], operator, rhs)])
        assert findings(warnings_of(problem), "REDUNDANT_CONSTRAINT") == []

    def test_single_variable_at_most_one_is_redundant(self):
        problem = build([card(["x0"], "<=", 1)])
        (warning,) = findings(warnings_of(problem), "REDUNDANT_CONSTRAINT")
        assert warning.path == "cardinality_constraints[0]"
        # One counted variable reads in the singular.
        assert "counts 1 variable, so between 0 and 1 are chosen" in warning.message

    def test_pairwise_at_most_one_never_needs_slack(self):
        problem = build([card(["x0", "x1", "x2", "x3"], "<=", 1)])
        warnings = warnings_of(problem)
        assert findings(warnings, "REDUNDANT_CONSTRAINT") == []
        assert findings(warnings, "LARGE_SLACK_RANGE") == []


class TestSoftWeightSmall:
    def test_weight_below_the_ratio(self):
        # objective scale 10, ratio 0.01: anything below 0.1 is small.
        problem = build([card(["x0", "x1"], kind="soft", weight=0.05, constraint_id="tiny")])
        (warning,) = findings(warnings_of(problem), "SOFT_WEIGHT_SMALL")
        assert warning.path == "cardinality_constraints[0].weight"
        assert warning.message.startswith("Soft constraint tiny has weight 0.05")

    def test_weight_at_scale_is_silent(self):
        problem = build([card(["x0", "x1"], kind="soft", weight=1)])
        assert findings(warnings_of(problem), "SOFT_WEIGHT_SMALL") == []

    def test_linear_paths_come_first(self):
        problem = build(
            [card(["x0", "x1"], kind="soft", weight=0.05, constraint_id="card")],
            constraints=[
                {
                    "id": "lin",
                    "type": "soft",
                    "weight": 0.01,
                    "terms": [lin("x2", 1)],
                    "operator": "<=",
                    "rhs": 0,
                }
            ],
        )
        assert [w.path for w in findings(warnings_of(problem), "SOFT_WEIGHT_SMALL")] == [
            "constraints[0].weight",
            "cardinality_constraints[0].weight",
        ]


# --------------------------------------------------------------------------
# Version gate (§6.4)
# --------------------------------------------------------------------------


class TestFeatureRequiresNewerVersion:
    def test_explicit_1_1(self):
        errors = validate_problem(build([card(["x0", "x1"])], version="1.1"))
        assert [(e.code, e.path) for e in errors] == [("FEATURE_REQUIRES_NEWER_VERSION", "version")]
        assert errors[0].message == (
            "cardinality_constraints requires version \"1.2\" or later, but version is '1.1'"
        )
        assert errors[0].recommended_action == RECOMMENDED_ACTIONS["FEATURE_REQUIRES_NEWER_VERSION"]

    def test_explicit_1_0_without_integers(self):
        problem = build([card(["x0", "x1"])], variables=["x0", "x1"], version="1.0")
        errors = validate_problem(problem)
        assert codes(errors) == ["FEATURE_REQUIRES_NEWER_VERSION"]
        assert errors[0].message == (
            "cardinality_constraints requires version \"1.2\" or later, but version is '1.0'"
        )

    def test_omitted_version(self):
        problem = build([card(["x0", "x1"])], variables=["x0", "x1"], version=None)
        assert problem.version == "1.0"
        errors = validate_problem(problem)
        assert codes(errors) == ["FEATURE_REQUIRES_NEWER_VERSION"]
        assert 'version defaults to "1.0" when omitted' in errors[0].message
        assert "also allows the integer variables" not in errors[0].message

    def test_1_0_with_integers_keeps_the_integer_error(self):
        errors = validate_problem(build([card(["x0", "x1"])], version="1.0"))
        assert codes(errors) == ["INTEGER_REQUIRES_VERSION_1_1", "FEATURE_REQUIRES_NEWER_VERSION"]
        message = errors[1].message
        assert '"1.2" also allows the integer variables' in message
        assert "defaults to" not in message

    def test_omitted_version_with_integers_carries_both_notes(self):
        errors = validate_problem(build([card(["x0", "x1"])], version=None))
        assert codes(errors) == ["INTEGER_REQUIRES_VERSION_1_1", "FEATURE_REQUIRES_NEWER_VERSION"]
        message = errors[1].message
        assert 'version defaults to "1.0" when omitted' in message
        assert '"1.2" also allows the integer variables' in message

    def test_1_1_with_integers_has_no_integer_note(self):
        errors = validate_problem(build([card(["x0", "x1"])], version="1.1"))
        assert "integer" not in errors[0].message

    def test_1_2_is_accepted(self):
        assert validate_problem(build([card(["x0", "x1"])], version="1.2")) == []

    @pytest.mark.parametrize("version", ["1.0", "1.1", None])
    def test_an_empty_list_needs_no_newer_version(self, version):
        problem = build([], variables=["x0", "x1"], version=version)
        assert validate_problem(problem) == []

    def test_the_other_errors_are_still_reported(self):
        errors = validate_problem(build([card(["x0", "ghost"])], version="1.1"))
        assert codes(errors) == ["FEATURE_REQUIRES_NEWER_VERSION", "UNKNOWN_VARIABLE"]


# --------------------------------------------------------------------------
# CARDINALITY_FORM_AVAILABLE (§6.3)
# --------------------------------------------------------------------------


def linear_amo(terms: list[dict], *, rhs=1, kind="hard", constraint_id="amo", **extra) -> dict:
    return {
        "id": constraint_id,
        "type": kind,
        "terms": terms,
        "operator": "<=",
        "rhs": rhs,
        **extra,
    }


def form_advice(problem: OptimizationProblem, **kwargs) -> list:
    return findings(warnings_of(problem, **kwargs), "CARDINALITY_FORM_AVAILABLE")


def exact_capabilities() -> SolverCapabilities:
    return SolverRegistry.default().get("exact").capabilities


class TestCardinalityFormAvailable:
    def test_linear_at_most_one_in_a_1_2_document(self):
        problem = build([], constraints=[linear_amo([lin("x0", 1), lin("x1", 1), lin("x2", 1)])])
        (warning,) = form_advice(problem)
        assert warning.path == "constraints[0]"
        assert "Hard constraint amo is an at-most-one over 3 binary variables" in warning.message
        assert "cardinality_constraints" in warning.message
        assert "without a slack variable" in warning.message
        assert warning.recommended_action == _WARNING_RECOMMENDED_ACTIONS[
            "CARDINALITY_FORM_AVAILABLE"
        ]

    def test_on_a_bqm_backend_declaration(self):
        problem = build([], constraints=[linear_amo([lin("x0", 1), lin("x1", 1)])])
        assert len(form_advice(problem, capabilities=exact_capabilities())) == 1

    def test_path_counts_every_linear_constraint(self):
        problem = build(
            [card(["x2", "x3"], "<=", 1)],
            constraints=[
                linear_amo([lin("x0", 2), lin("x1", 1)], constraint_id="weighted"),
                linear_amo([lin("x0", 1), lin("x1", 1)], constraint_id="plain"),
            ],
        )
        assert [w.path for w in form_advice(problem)] == ["constraints[1]"]

    @pytest.mark.parametrize(
        "case",
        [
            "version_1_1",
            "cqm_path",
            "coefficient_2",
            "repeated_variable",
            "integer_variable",
            "rhs_2",
            "soft",
            "single_variable",
            "equality",
        ],
    )
    def test_not_advised(self, case):
        terms = [lin("x0", 1), lin("x1", 1)]
        constraint = linear_amo(terms)
        version = "1.2"
        kwargs: dict = {}
        if case == "version_1_1":
            version = "1.1"
        elif case == "cqm_path":
            kwargs["model_type"] = "cqm"
        elif case == "coefficient_2":
            constraint = linear_amo([lin("x0", 2), lin("x1", 1)])
        elif case == "repeated_variable":
            constraint = linear_amo([lin("x0", 1), lin("x0", 1), lin("x1", 1)])
        elif case == "integer_variable":
            constraint = linear_amo([lin("x0", 1), lin("k", 1)])
        elif case == "rhs_2":
            constraint = linear_amo([lin("x0", 1), lin("x1", 1), lin("x2", 1)], rhs=2)
        elif case == "soft":
            constraint = linear_amo(terms, kind="soft", weight=1)
        elif case == "single_variable":
            constraint = linear_amo([lin("x0", 1)])
        elif case == "equality":
            constraint = {**linear_amo(terms), "operator": "=="}
        problem = build([], constraints=[constraint], version=version)
        assert form_advice(problem, **kwargs) == []

    def test_declared_at_most_one_is_not_advised(self):
        problem = build([card(["x0", "x1"], "<=", 1)])
        assert form_advice(problem) == []

    def test_placed_after_the_range_warnings_and_before_integer_encoding(self):
        big = {"name": "big", "type": "integer", "lower_bound": 0, "upper_bound": 5000}
        problem = build(
            [],
            variables=["x0", "x1", "x2", big],
            constraints=[
                # _warn_constraint_ranges: SOFT_ALWAYS_VIOLATED.
                {
                    "id": "impossible",
                    "type": "soft",
                    "weight": 1,
                    "terms": [lin("x2", 1)],
                    "operator": ">=",
                    "rhs": 2,
                },
                linear_amo([lin("x0", 1), lin("x1", 1)]),
            ],
        )
        warning_codes = codes(warnings_of(problem))
        assert warning_codes == [
            "SOFT_ALWAYS_VIOLATED",
            "CARDINALITY_FORM_AVAILABLE",
            "LARGE_INTEGER_RANGE",
        ]


# --------------------------------------------------------------------------
# rhs bounds at the schema (§4.2)
# --------------------------------------------------------------------------


def with_rhs(rhs) -> dict:
    return payload([card(["x0", "x1"], "<=", rhs)])


class TestRhsBounds:
    @pytest.mark.parametrize(
        "rhs",
        [10**400, -(10**400), 2**31, -(2**31)],
        ids=["401_digits", "minus_401_digits", "2^31", "-2^31"],
    )
    def test_out_of_range_is_invalid_field_value(self, rhs):
        result = parse_problem(with_rhs(rhs))
        assert isinstance(result, list)
        assert [(e.code, e.path) for e in result] == [
            ("INVALID_FIELD_VALUE", "cardinality_constraints[0].rhs")
        ]
        # Never the submitted value (interfaces/problem_input.py).
        assert str(rhs) not in result[0].message

    def test_401_digit_literal_in_json_text(self):
        text = json.dumps(with_rhs(0)).replace('"rhs": 0', '"rhs": 1' + "0" * 400)
        assert "0" * 400 in text
        result = parse_problem(json.loads(text))
        assert [(e.code, e.path) for e in result] == [
            ("INVALID_FIELD_VALUE", "cardinality_constraints[0].rhs")
        ]

    @pytest.mark.parametrize(
        "rhs, fragment",
        [(True, "boolean"), (False, "boolean"), ("1", "string"), (1.5, "integer")],
        ids=["true", "false", "string", "fraction"],
    )
    def test_non_integer_is_invalid_field_value(self, rhs, fragment):
        result = parse_problem(with_rhs(rhs))
        assert [(e.code, e.path) for e in result] == [
            ("INVALID_FIELD_VALUE", "cardinality_constraints[0].rhs")
        ]
        assert fragment in result[0].message

    @pytest.mark.parametrize("rhs", [2**31 - 1, -(2**31 - 1)])
    def test_the_limits_are_accepted_and_lower_exactly(self, rhs):
        problem = parse_problem(with_rhs(rhs))
        assert isinstance(problem, OptimizationProblem)
        (lowered,) = problem.all_constraints()
        assert lowered.rhs == float(rhs)
        assert int(lowered.rhs) == rhs
        # And the validator judges it without raising.
        assert codes(validate_problem(problem)) in (["TRIVIALLY_INFEASIBLE"], [])

    def test_terms_instead_of_variables_is_a_schema_error(self):
        data = payload([{"id": "c", "type": "hard", "terms": [lin("x0", 1)], "operator": "<=", "rhs": 1}])
        result = parse_problem(data)
        assert sorted((e.code, e.path) for e in result) == [
            ("MISSING_FIELD", "cardinality_constraints[0].variables"),
            ("UNKNOWN_FIELD", "cardinality_constraints[0].terms"),
        ]
