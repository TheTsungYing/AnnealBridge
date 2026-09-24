"""``parse_problem`` and the schema-error results (interfaces/problem_input.py).

The CLI and the MCP tools parse a submitted document through this one module,
so a document that does not fit the problem schema is reported the same way
on both: catalog codes (``UNKNOWN_FIELD``, ``MISSING_FIELD``,
``INVALID_FIELD_VALUE``), the validator's path notation, every error at once
and never the submitted value.
"""

import copy
import json
import time

import pytest
from pydantic import ValidationError

from annealbridge.interfaces.problem_input import (
    invalid_recommendation_result,
    invalid_solve_result,
    invalid_validation_result,
    parse_problem,
)
from annealbridge.models import (
    RECOMMENDED_ACTIONS,
    OptimizationProblem,
    SolveError,
    catalog_error,
)
from annealbridge.version import package_version
from tests.conftest import EXAMPLES_DIR

SENTINEL = "SECRET_SENTINEL_123"


def _knapsack() -> dict:
    return json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))


def _errors(document) -> list[SolveError]:
    parsed = parse_problem(document)
    assert isinstance(parsed, list), parsed
    assert parsed, "a rejected document must carry at least one error"
    return parsed


def _pydantic_errors(document) -> list[dict]:
    with pytest.raises(ValidationError) as excinfo:
        OptimizationProblem.model_validate(document)
    return excinfo.value.errors()


def _notation(loc: tuple) -> str:
    """``("a", 0, "b")`` → ``a[0].b``, written independently of the module."""
    parts: list[str] = []
    for part in loc:
        if isinstance(part, int):
            parts[-1] += f"[{part}]"
        else:
            parts.append(part)
    return ".".join(parts)


class TestValidDocument:
    def test_returns_the_model_equal_to_model_validate(self):
        document = _knapsack()
        parsed = parse_problem(document)
        assert isinstance(parsed, OptimizationProblem)
        assert parsed == OptimizationProblem.model_validate(document)

    def test_the_document_is_not_modified(self):
        document = _knapsack()
        before = copy.deepcopy(document)
        parse_problem(document)
        assert document == before


class TestErrorKinds:
    def test_unknown_field_is_unknown_field_with_a_fixed_message(self):
        document = _knapsack()
        document["objective"]["cubic_terms"] = [{"coefficient": 1}]

        (error,) = _errors(document)

        assert error.code == "UNKNOWN_FIELD"
        assert error.path == "objective.cubic_terms"
        assert error.message == "unknown field, not in the problem schema"

    def test_missing_field_is_missing_field_with_pydantic_message(self):
        document = _knapsack()
        del document["variables"][1]["name"]

        (error,) = _errors(document)

        assert error.code == "MISSING_FIELD"
        assert error.path == "variables[1].name"
        assert error.message == "Field required"

    def test_wrong_literal_is_invalid_field_value(self):
        document = _knapsack()
        document["constraints"][0]["operator"] = "<"

        (error,) = _errors(document)

        assert error.code == "INVALID_FIELD_VALUE"
        assert error.path == "constraints[0].operator"
        assert error.message == "Input should be '==', '<=' or '>='"

    def test_null_in_a_numeric_field_is_invalid_field_value(self):
        document = _knapsack()
        document["solver"]["penalty_multiplier"] = None

        (error,) = _errors(document)

        assert error.code == "INVALID_FIELD_VALUE"
        assert error.path == "solver.penalty_multiplier"

    @pytest.mark.parametrize(
        "document", ["a knapsack", [], None, 5], ids=["string", "list", "null", "int"]
    )
    def test_a_document_that_is_not_an_object_has_no_path(self, document):
        (error,) = _errors(document)

        assert error.code == "INVALID_FIELD_VALUE"
        assert error.path is None
        assert error.message == "the problem must be a JSON object"

    def test_value_error_prefix_is_stripped(self):
        document = _knapsack()
        document["objective"]["linear_terms"][0]["coefficient"] = "2"
        (raw,) = _pydantic_errors(document)
        assert raw["msg"].startswith("Value error, ")

        (error,) = _errors(document)

        assert error.code == "INVALID_FIELD_VALUE"
        assert error.message == raw["msg"].removeprefix("Value error, ")
        assert error.message == "coefficient must be a number, not a string"


class TestPaths:
    @pytest.mark.parametrize(
        "path",
        [
            "constraints[0].terms[1].coefficient",
            "objective.linear_terms[3].coefficient",
            "variables[2].type",
            "solver.top_k",
            "version",
        ],
    )
    def test_path_uses_the_validator_notation(self, path):
        document = _knapsack()
        node = document
        keys = path.split(".")
        for key in keys[:-1]:
            if key.endswith("]"):
                key, _, index = key[:-1].partition("[")
                node = node[key][int(index)]
            else:
                node = node[key]
        node[keys[-1]] = [SENTINEL]  # the wrong type for every field above

        (error,) = _errors(document)

        assert error.path == path

    @pytest.mark.parametrize(
        ("where", "key", "path"),
        [
            # An empty name still yields a path: null is reserved for a
            # document that is not an object at all.
            ((), "", '[""]'),
            (("solver",), "", 'solver[""]'),
            # A dot or a bracket in a name cannot pose as further segments.
            ((), "a.b[0]", '["a.b[0]"]'),
            # A newline cannot start a forged line in a log or on stderr.
            ((), "x\n  [MISSING_FIELD] objective", '["x\\n  [MISSING_FIELD] objective"]'),
            (("variables", 0), "note", "variables[0].note"),
        ],
    )
    def test_an_undeclared_name_that_is_not_an_identifier_is_quoted(
        self, where, key, path
    ):
        """2026-09-24 independent review: the name of an undeclared field is
        the one caller-chosen part of a path."""
        document = _knapsack()
        node = document
        for step in where:
            node = node[step]
        node[key] = 1

        (error,) = _errors(document)

        assert error.code == "UNKNOWN_FIELD"
        assert error.path == path
        assert "\n" not in error.path


class TestAllErrorsAtOnce:
    def test_every_error_in_pydantic_order(self):
        document = _knapsack()
        del document["objective"]
        document["constraints"][0]["operator"] = "<"
        document["constraints"][0]["terms"][1]["coefficient"] = True
        document["foo"] = 1

        errors = _errors(document)
        raw = _pydantic_errors(document)

        assert len(errors) == len(raw) == 4
        assert [(error.code, error.path) for error in errors] == [
            ("MISSING_FIELD", "objective"),
            ("INVALID_FIELD_VALUE", "constraints[0].terms[1].coefficient"),
            ("INVALID_FIELD_VALUE", "constraints[0].operator"),
            ("UNKNOWN_FIELD", "foo"),
        ]
        # Same order as pydantic reports them.
        assert [error.path for error in errors] == [_notation(item["loc"]) for item in raw]


class TestNoInputValueLeaks:
    def _leaky_document(self) -> dict:
        document = _knapsack()
        document["version"] = SENTINEL
        document["variables"][0]["type"] = SENTINEL
        document["objective"]["linear_terms"][0]["coefficient"] = SENTINEL
        document["constraints"][0]["operator"] = SENTINEL
        document["solver"]["backend"] = SENTINEL
        document["solver"]["num_reads"] = SENTINEL
        document["unknown"] = SENTINEL
        return document

    def test_the_sentinel_is_in_pydantic_own_error_text(self):
        # Proves the check below is meaningful: pydantic would echo it.
        with pytest.raises(ValidationError) as excinfo:
            OptimizationProblem.model_validate(self._leaky_document())
        assert SENTINEL in str(excinfo.value)

    def test_no_message_or_serialised_error_carries_the_value(self):
        errors = _errors(self._leaky_document())

        assert len(errors) == 7
        for error in errors:
            assert SENTINEL not in error.message
            assert SENTINEL not in error.model_dump_json()
            assert "input_value" not in error.message
            assert "pydantic.dev" not in error.message

    def test_no_result_built_from_them_carries_the_value(self):
        errors = _errors(self._leaky_document())
        for result in (
            invalid_solve_result(errors, time.perf_counter()),
            invalid_validation_result(errors),
            invalid_recommendation_result(errors),
        ):
            assert SENTINEL not in result.model_dump_json()


class TestCatalogFields:
    @pytest.mark.parametrize(
        "mutate, code",
        [
            (lambda d: d.update(foo=1), "UNKNOWN_FIELD"),
            (lambda d: d.pop("objective"), "MISSING_FIELD"),
            (lambda d: d.update(version="9.9"), "INVALID_FIELD_VALUE"),
        ],
        ids=["unknown", "missing", "invalid"],
    )
    def test_recommended_action_comes_from_the_catalog(self, mutate, code):
        document = _knapsack()
        mutate(document)

        (error,) = _errors(document)

        assert error.code == code
        assert error.recommended_action == RECOMMENDED_ACTIONS[code]
        assert error.retryable is False


class TestInvalidResults:
    @staticmethod
    def _errors() -> list[SolveError]:
        return [
            catalog_error("MISSING_FIELD", "Field required", path="objective"),
            catalog_error(
                "UNKNOWN_FIELD", "unknown field, not in the problem schema", path="foo"
            ),
        ]

    def test_invalid_solve_result(self):
        errors = self._errors()
        started = time.perf_counter() - 0.05

        result = invalid_solve_result(errors, started)

        assert result.status == "invalid_problem"
        assert result.backend is None
        assert result.objective_direction is None
        assert result.solutions == []
        assert result.attempts == []
        assert result.warnings == []
        assert result.errors == errors
        assert result.message == "Field required"
        assert isinstance(result.elapsed_ms, float)
        assert result.elapsed_ms >= 50
        assert result.annealbridge_version == package_version()

    def test_invalid_validation_result(self):
        errors = self._errors()

        result = invalid_validation_result(errors)

        assert result.valid is False
        assert result.errors == errors
        assert result.warnings == []
        assert result.estimated_compiled_variables is None
        assert result.objective_scale is None
        assert result.model_type is None

    def test_invalid_recommendation_result(self):
        errors = self._errors()

        result = invalid_recommendation_result(errors)

        assert result.valid is False
        assert result.errors == errors
        assert result.recommendations == []
