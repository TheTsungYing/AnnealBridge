"""The schema layer of schema 1.3 templates (models/templates.py).

Schema 1.3 spec 2026-09-25 §14.3-§14.5, §14.13 and §14.16 item 10, through
``interfaces/problem_input.parse_problem`` (the one parser the CLI and the
MCP tools share):

* a union field -- ``coefficient`` / ``rhs`` / ``weight`` (number or
  reference) and a cardinality ``rhs`` (count or reference) -- refuses a
  boolean, ``null`` (except ``weight``), another JSON type, NaN / ±inf, an
  integer too large for a float (or outside ±(2^31-1) for a count) and a
  string over 256 characters with exactly one ``INVALID_FIELD_VALUE`` at the
  field, never one error per union member (``rhs.float`` / ``rhs.str``);
* a cardinality template member is a reference string or an object; any
  other JSON type is one clear ``INVALID_FIELD_VALUE``;
* ``for_each`` / ``where`` items and variable references are at most 256
  characters, and an index set element or parameter key element is a
  string or an integer, never a boolean or a float;
* the JSON Schema has no free-form object: every ``$defs`` entry is
  ``additionalProperties: false`` and every property has a description;
* a 1.0 document dumps without any template key, and a 1.3 document
  round-trips through dump and parse unchanged.
"""

import json
import math

import pytest

from annealbridge.interfaces.problem_input import parse_problem
from annealbridge.models import OptimizationProblem, SolveError
from annealbridge.models.templates import (
    CardinalityConstraintTemplate,
    CardinalityMemberTemplate,
    ConstraintTemplate,
    IndexSet,
    LinearTermTemplate,
    Parameter,
    ParameterValue,
    QuadraticTermTemplate,
    VariableFamily,
)
from tests.conftest import EXAMPLES_DIR

TEMPLATE_KEYS = (
    "index_sets",
    "parameters",
    "variable_families",
    "constraint_templates",
    "cardinality_constraint_templates",
)
OBJECTIVE_TEMPLATE_KEYS = ("linear_term_templates", "quadratic_term_templates")


def document(**fields) -> dict:
    data = {
        "version": "1.3",
        "name": "templates",
        "variables": [],
        "objective": {"direction": "minimize", "linear_terms": []},
        "constraints": [],
    }
    data.update(fields)
    return data


def linear_template(**overrides) -> dict:
    data = {"for_each": ["i in s"], "coefficient": 1, "variable": "x[i]"}
    data.update(overrides)
    return document(
        objective={
            "direction": "minimize",
            "linear_terms": [],
            "linear_term_templates": [data],
        }
    )


def constraint_template(**overrides) -> dict:
    data = {
        "id": "c",
        "type": "soft",
        "terms": [{"coefficient": 1, "variable": "x"}],
        "operator": "==",
        "rhs": 1,
        "weight": 1,
    }
    data.update(overrides)
    return document(constraint_templates=[data])


def cardinality_template(**overrides) -> dict:
    data = {
        "id": "k",
        "type": "soft",
        "variables": ["x"],
        "operator": "==",
        "rhs": 1,
        "weight": 1,
    }
    data.update(overrides)
    return document(cardinality_constraint_templates=[data])


def parsed(data: dict) -> OptimizationProblem:
    problem = parse_problem(data)
    assert isinstance(problem, OptimizationProblem), problem
    return problem


def errors(data: dict) -> list[SolveError]:
    result = parse_problem(data)
    assert isinstance(result, list), result
    return result


def only_error(data: dict) -> tuple[str, str | None, str]:
    found = errors(data)
    assert len(found) == 1, [(e.code, e.path, e.message) for e in found]
    return found[0].code, found[0].path, found[0].message


# (document builder, path, the field name the message starts with, count?)
UNION_FIELDS = [
    (
        lambda v: linear_template(coefficient=v),
        "objective.linear_term_templates[0].coefficient",
        "coefficient",
        False,
    ),
    (
        lambda v: document(
            objective={
                "direction": "minimize",
                "linear_terms": [],
                "quadratic_term_templates": [
                    {"coefficient": v, "variable1": "x", "variable2": "y"}
                ],
            }
        ),
        "objective.quadratic_term_templates[0].coefficient",
        "coefficient",
        False,
    ),
    (
        lambda v: constraint_template(
            terms=[{"coefficient": v, "variable": "x"}]
        ),
        "constraint_templates[0].terms[0].coefficient",
        "coefficient",
        False,
    ),
    (lambda v: constraint_template(rhs=v), "constraint_templates[0].rhs", "rhs", False),
    (
        lambda v: constraint_template(weight=v),
        "constraint_templates[0].weight",
        "weight",
        False,
    ),
    (
        lambda v: cardinality_template(rhs=v),
        "cardinality_constraint_templates[0].rhs",
        "rhs",
        True,
    ),
    (
        lambda v: cardinality_template(weight=v),
        "cardinality_constraint_templates[0].weight",
        "weight",
        False,
    ),
]
UNION_IDS = [path for _, path, _, _ in UNION_FIELDS]


def _expected_message(value, field: str, count: bool) -> str:
    """The one message each refused value gets (models/templates.py)."""
    noun = "an integer" if count else "a number"
    if isinstance(value, bool):
        return f"{field} must be {noun} or a parameter reference, not a boolean"
    if isinstance(value, str):
        return f"{field} must be at most 256 characters"
    if isinstance(value, float):
        if count:
            return f"{field} must be an integer or a parameter reference"
        return f"{field} must be a finite number"
    if isinstance(value, int):
        if count:
            return f"{field} must lie within ±2147483647"
        return f"{field} is too large to be a finite number"
    return f"{field} must be {noun} or a parameter reference string"


REFUSED_VALUES = [
    True,
    False,
    [1],
    {"value": 1},
    math.nan,
    math.inf,
    -math.inf,
    10**400,
    "p" * 257,
]
REFUSED_IDS = ["true", "false", "list", "object", "nan", "inf", "-inf", "10**400", "257 chars"]


class TestUnionFields:
    """A union field reports one error at the field itself (spec §14.5)."""

    @pytest.mark.parametrize("value", REFUSED_VALUES, ids=REFUSED_IDS)
    @pytest.mark.parametrize("build, path, field, count", UNION_FIELDS, ids=UNION_IDS)
    def test_one_error_without_a_branch_label(self, build, path, field, count, value):
        code, error_path, message = only_error(build(value))
        assert code == "INVALID_FIELD_VALUE"
        assert error_path == path
        assert message == _expected_message(value, field, count)

    @pytest.mark.parametrize(
        "build, path, field, count",
        [entry for entry in UNION_FIELDS if entry[2] != "weight"],
        ids=[path for _, path, field, _ in UNION_FIELDS if field != "weight"],
    )
    def test_null_is_refused_except_for_weight(self, build, path, field, count):
        code, error_path, message = only_error(build(None))
        assert (code, error_path) == ("INVALID_FIELD_VALUE", path)
        noun = "an integer" if count else "a number"
        assert message == f"{field} must be {noun} or a parameter reference string"

    def test_weight_may_be_null(self):
        assert parsed(constraint_template(weight=None)).constraint_templates[0].weight is None
        assert (
            parsed(cardinality_template(weight=None)).cardinality_constraint_templates[0].weight
            is None
        )

    @pytest.mark.parametrize("build, path, field, count", UNION_FIELDS, ids=UNION_IDS)
    def test_no_path_carries_a_union_branch_label(self, build, path, field, count):
        for value in [*REFUSED_VALUES, None]:
            result = parse_problem(build(value))
            if isinstance(result, OptimizationProblem):
                continue  # null weight
            for error in result:
                assert error.path == path
                assert not error.path.endswith(
                    (".float", ".str", ".int", ".function-before[_number_or_reference()]")
                )

    def test_a_number_is_a_float_and_a_reference_stays_a_string(self):
        template = parsed(linear_template(coefficient=3)).objective.linear_term_templates[0]
        assert type(template.coefficient) is float and template.coefficient == 3.0
        template = parsed(linear_template(coefficient="cost[i]")).objective.linear_term_templates[0]
        assert template.coefficient == "cost[i]"
        # A reference of exactly 256 characters is still accepted.
        text = "p" * 256
        assert parsed(constraint_template(rhs=text)).constraint_templates[0].rhs == text

    def test_a_count_accepts_an_integral_float_and_the_bounds(self):
        rhs = parsed(cardinality_template(rhs=2.0)).cardinality_constraint_templates[0].rhs
        assert type(rhs) is int and rhs == 2
        for bound in (2**31 - 1, -(2**31 - 1)):
            assert parsed(cardinality_template(rhs=bound)).cardinality_constraint_templates[0].rhs == bound
        for outside in (2**31, -(2**31)):
            assert only_error(cardinality_template(rhs=outside)) == (
                "INVALID_FIELD_VALUE",
                "cardinality_constraint_templates[0].rhs",
                "rhs must lie within ±2147483647",
            )

    def test_a_count_refuses_a_fraction(self):
        assert only_error(cardinality_template(rhs=1.5)) == (
            "INVALID_FIELD_VALUE",
            "cardinality_constraint_templates[0].rhs",
            "rhs must be an integer or a parameter reference",
        )

    def test_the_submitted_value_is_never_echoed(self):
        sentinel = "SECRET_SENTINEL_" + "q" * 250
        for build, _, _, _ in UNION_FIELDS:
            for error in errors(build(sentinel)):
                assert "SECRET_SENTINEL" not in error.message


class TestCardinalityMembers:
    """``variables``: a reference string or ``{for_each?, where?, variable}``."""

    def test_the_string_shorthand(self):
        member = parsed(cardinality_template(variables=["x[i]"])).cardinality_constraint_templates[0].variables[0]
        assert member == CardinalityMemberTemplate(variable="x[i]")
        assert member.for_each == [] and member.where == []

    def test_the_object_form(self):
        data = {"for_each": ["i in s"], "where": ["w[i] == 1"], "variable": "x[i]"}
        member = parsed(cardinality_template(variables=[data])).cardinality_constraint_templates[0].variables[0]
        assert member == CardinalityMemberTemplate(**data)

    def test_both_forms_mix(self):
        template = parsed(
            cardinality_template(variables=["y", {"for_each": ["i in s"], "variable": "x[i]"}])
        ).cardinality_constraint_templates[0]
        assert [member.variable for member in template.variables] == ["y", "x[i]"]
        assert template.variables[1].for_each == ["i in s"]

    @pytest.mark.parametrize(
        "member", [1, 1.5, None, True, ["x"]], ids=["int", "float", "null", "bool", "list"]
    )
    def test_any_other_member_is_one_clear_error(self, member):
        assert only_error(cardinality_template(variables=[member])) == (
            "INVALID_FIELD_VALUE",
            "cardinality_constraint_templates[0].variables[0]",
            "each variables entry must be a variable reference string or an object",
        )

    def test_an_object_member_is_checked_field_by_field(self):
        assert only_error(cardinality_template(variables=[{"variable": 1}])) == (
            "INVALID_FIELD_VALUE",
            "cardinality_constraint_templates[0].variables[0].variable",
            "Input should be a valid string",
        )
        found = errors(cardinality_template(variables=[{"varible": "x"}]))
        assert [(e.code, e.path) for e in found] == [
            ("MISSING_FIELD", "cardinality_constraint_templates[0].variables[0].variable"),
            ("UNKNOWN_FIELD", "cardinality_constraint_templates[0].variables[0].varible"),
        ]

    def test_the_shorthand_is_limited_like_the_object(self):
        assert only_error(cardinality_template(variables=["x" * 257])) == (
            "INVALID_FIELD_VALUE",
            "cardinality_constraint_templates[0].variables[0].variable",
            "String should have at most 256 characters",
        )


LONG = "i in s" + " " * 251  # 257 characters


class TestTemplateTextLength:
    """for_each / where items and references: at most 256 characters."""

    @pytest.mark.parametrize(
        "data, path",
        [
            (linear_template(for_each=[LONG]), "objective.linear_term_templates[0].for_each[0]"),
            (linear_template(where=[LONG]), "objective.linear_term_templates[0].where[0]"),
            (linear_template(variable=LONG), "objective.linear_term_templates[0].variable"),
            (
                document(
                    objective={
                        "direction": "minimize",
                        "linear_terms": [],
                        "quadratic_term_templates": [
                            {"coefficient": 1, "variable1": "x", "variable2": LONG}
                        ],
                    }
                ),
                "objective.quadratic_term_templates[0].variable2",
            ),
            (
                constraint_template(
                    terms=[{"for_each": ["i in s", LONG], "coefficient": 1, "variable": "x"}]
                ),
                "constraint_templates[0].terms[0].for_each[1]",
            ),
            (constraint_template(where=[LONG]), "constraint_templates[0].where[0]"),
            (
                cardinality_template(variables=[{"where": [LONG], "variable": "x"}]),
                "cardinality_constraint_templates[0].variables[0].where[0]",
            ),
            (cardinality_template(for_each=[LONG]), "cardinality_constraint_templates[0].for_each[0]"),
        ],
    )
    def test_over_the_limit(self, data, path):
        assert only_error(data) == (
            "INVALID_FIELD_VALUE",
            path,
            "String should have at most 256 characters",
        )

    def test_at_the_limit(self):
        text = "i in s" + " " * 250  # 256 characters
        assert parsed(linear_template(for_each=[text])).objective.linear_term_templates[0].for_each == [text]


class TestElements:
    """Index set elements and parameter key elements: strings or integers."""

    @pytest.mark.parametrize(
        "element", [True, False, 1.5, 1.0, None, [1], {"a": 1}],
        ids=["true", "false", "1.5", "1.0", "null", "list", "object"],
    )
    def test_index_set_elements(self, element):
        assert only_error(document(index_sets=[{"name": "s", "elements": ["a", element]}])) == (
            "INVALID_FIELD_VALUE",
            "index_sets[0].elements[1]",
            "index set elements must be strings or integers",
        )

    @pytest.mark.parametrize(
        "element", [True, 1.5, 1.0, None], ids=["true", "1.5", "1.0", "null"]
    )
    def test_parameter_key_elements(self, element):
        data = document(
            parameters=[
                {"name": "p", "indices": ["s"], "values": [{"key": [element], "value": 1}]}
            ]
        )
        assert only_error(data) == (
            "INVALID_FIELD_VALUE",
            "parameters[0].values[0].key[0]",
            "index set elements must be strings or integers",
        )

    def test_strings_and_integers_are_kept_as_given(self):
        problem = parsed(document(index_sets=[{"name": "s", "elements": ["a", 1, -2]}]))
        elements = problem.index_sets[0].elements
        assert elements == ["a", 1, -2]
        assert [type(element) for element in elements] == [str, int, int]

    def test_a_parameter_value_is_a_number(self):
        data = document(
            parameters=[{"name": "p", "indices": ["s"], "values": [{"key": ["a"], "value": "1"}]}]
        )
        assert only_error(data) == (
            "INVALID_FIELD_VALUE",
            "parameters[0].values[0].value",
            "value must be a number, not a string",
        )


def _walk(node, where="schema"):
    """Every (location, dict) in a JSON Schema, depth first."""
    if isinstance(node, dict):
        yield where, node
        for key, value in node.items():
            yield from _walk(value, f"{where}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{where}[{index}]")


class TestJsonSchema:
    """No free-form object anywhere (spec §14.4, §14.13, §14.16 item 10)."""

    @pytest.fixture
    def schema(self) -> dict:
        return OptimizationProblem.model_json_schema()

    def test_no_additional_properties_true_anywhere(self, schema):
        offenders = [
            where
            for where, node in _walk(schema)
            if node.get("additionalProperties") not in (None, False)
        ]
        assert offenders == []
        assert '"additionalProperties": true' not in json.dumps(schema)

    def test_every_object_is_closed(self, schema):
        assert schema["additionalProperties"] is False
        for name, definition in schema["$defs"].items():
            assert definition.get("additionalProperties") is False, name
        for where, node in _walk(schema):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, where

    def test_every_property_has_a_description(self, schema):
        missing = [
            f"{where}.{name}"
            for where, node in _walk(schema)
            if isinstance(node.get("properties"), dict)
            for name, prop in node["properties"].items()
            if not (isinstance(prop, dict) and prop.get("description"))
        ]
        assert missing == []

    def test_the_template_models_are_in_defs(self, schema):
        for model in (
            IndexSet,
            Parameter,
            ParameterValue,
            VariableFamily,
            LinearTermTemplate,
            QuadraticTermTemplate,
            ConstraintTemplate,
            CardinalityConstraintTemplate,
            CardinalityMemberTemplate,
        ):
            assert model.__name__ in schema["$defs"]

    def test_union_fields_keep_their_shape(self, schema):
        defs = schema["$defs"]
        assert defs["ConstraintTemplate"]["properties"]["rhs"]["anyOf"] == [
            {"type": "number"},
            {"type": "string"},
        ]
        assert defs["ConstraintTemplate"]["properties"]["weight"]["anyOf"] == [
            {"type": "number"},
            {"type": "string"},
            {"type": "null"},
        ]
        assert defs["CardinalityConstraintTemplate"]["properties"]["rhs"]["anyOf"] == [
            {"type": "integer"},
            {"type": "string"},
        ]
        assert defs["IndexSet"]["properties"]["elements"]["items"]["anyOf"] == [
            {"type": "integer"},
            {"type": "string"},
        ]

    def test_members_list_both_forms(self, schema):
        items = schema["$defs"]["CardinalityConstraintTemplate"]["properties"]["variables"]["items"]
        assert items["anyOf"] == [
            {"maxLength": 256, "type": "string"},
            {"$ref": "#/$defs/CardinalityMemberTemplate"},
        ]

    def test_template_text_is_limited_in_the_schema(self, schema):
        defs = schema["$defs"]
        assert defs["LinearTermTemplate"]["properties"]["for_each"]["items"]["maxLength"] == 256
        assert defs["LinearTermTemplate"]["properties"]["where"]["items"]["maxLength"] == 256
        assert defs["LinearTermTemplate"]["properties"]["variable"]["maxLength"] == 256
        assert defs["QuadraticTermTemplate"]["properties"]["variable1"]["maxLength"] == 256
        assert defs["CardinalityMemberTemplate"]["properties"]["variable"]["maxLength"] == 256

    def test_version_lists_1_3(self, schema):
        assert schema["properties"]["version"]["enum"] == ["1.0", "1.1", "1.2", "1.3"]


def _example(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


class TestDump:
    """Older dumps are unchanged; a 1.3 dump parses back equal (spec §14.13)."""

    @pytest.mark.parametrize(
        "name",
        [
            "knapsack.json",
            "assignment.json",
            "shift_scheduling.json",
            "tsp.json",
            "integer_knapsack.json",
            "exam_timetabling.json",
        ],
    )
    def test_an_older_document_dumps_no_template_key(self, name):
        problem = parsed(_example(name))
        for mode in ("python", "json"):
            dumped = problem.model_dump(mode=mode)
            assert not set(dumped) & set(TEMPLATE_KEYS)
            assert not set(dumped["objective"]) & set(OBJECTIVE_TEMPLATE_KEYS)
        text = problem.model_dump_json()
        for key in (*TEMPLATE_KEYS, *OBJECTIVE_TEMPLATE_KEYS):
            assert f'"{key}"' not in text

    def test_a_minimal_1_0_document(self):
        problem = parsed(
            {
                "version": "1.0",
                "name": "minimal",
                "variables": [{"name": "a"}],
                "objective": {"direction": "minimize", "linear_terms": []},
                "constraints": [],
            }
        )
        dumped = problem.model_dump()
        assert list(dumped) == [
            "version",
            "name",
            "description",
            "variables",
            "objective",
            "constraints",
            "solver",
        ]
        assert list(dumped["objective"]) == [
            "direction",
            "linear_terms",
            "quadratic_terms",
            "constant",
        ]

    def test_empty_template_lists_are_left_out_too(self):
        data = document(
            variables=[{"name": "a"}],
            index_sets=[],
            parameters=[],
            variable_families=[],
            constraint_templates=[],
            cardinality_constraint_templates=[],
        )
        data["objective"]["linear_term_templates"] = []
        data["objective"]["quadratic_term_templates"] = []
        dumped = parsed(data).model_dump()
        assert not set(dumped) & set(TEMPLATE_KEYS)
        assert not set(dumped["objective"]) & set(OBJECTIVE_TEMPLATE_KEYS)

    def test_the_template_example_round_trips(self):
        problem = parsed(_example("tsp_template.json"))
        assert problem.has_templates()
        again = OptimizationProblem.model_validate(problem.model_dump())
        assert again == problem
        assert again.model_dump() == problem.model_dump()
        from_json = OptimizationProblem.model_validate_json(problem.model_dump_json())
        assert from_json == problem
        assert from_json.model_dump_json() == problem.model_dump_json()

    def test_every_template_field_round_trips(self):
        data = document(
            variables=[{"name": "v"}],
            index_sets=[
                {"name": "s", "elements": ["a", "b"], "description": "s"},
                {"name": "t", "elements": [0, 1], "order": "cyclic"},
            ],
            parameters=[
                {
                    "name": "w",
                    "indices": ["s", "t"],
                    "values": [{"key": ["a", 0], "value": 2.5}],
                    "default": 1,
                    "description": "weights",
                }
            ],
            variable_families=[
                {"name": "x", "indices": ["s", "t"], "description": "x"},
                {"name": "n", "indices": ["s"], "type": "integer", "lower_bound": 0, "upper_bound": 3},
            ],
            constraint_templates=[
                {
                    "id": "c",
                    "description": "c",
                    "type": "soft",
                    "for_each": ["i in s"],
                    "where": [],
                    "terms": [
                        {"for_each": ["p in t"], "where": ["w[i,p] >= 1"], "coefficient": "w[i,p]", "variable": "x[i,p+1]"}
                    ],
                    "operator": "<=",
                    "rhs": 2,
                    "weight": "w[i,p]",
                }
            ],
            cardinality_constraint_templates=[
                {
                    "id": "k",
                    "type": "hard",
                    "for_each": ["p in t"],
                    "variables": ["v", {"for_each": ["i in s"], "variable": "x[i,p]"}],
                    "operator": "==",
                    "rhs": 1,
                }
            ],
        )
        data["objective"]["linear_term_templates"] = [
            {"for_each": ["i in s"], "coefficient": -1, "variable": "n[i]"}
        ]
        data["objective"]["quadratic_term_templates"] = [
            {"for_each": ["i in s", "p in t"], "coefficient": "w[i,p]", "variable1": "x[i,p]", "variable2": "v"}
        ]
        problem = parsed(data)
        dumped = problem.model_dump()
        assert OptimizationProblem.model_validate(dumped) == problem
        assert OptimizationProblem.model_validate_json(problem.model_dump_json()) == problem
        # The shorthand member dumps as its object form and parses back equal.
        assert dumped["cardinality_constraint_templates"][0]["variables"][0] == {
            "for_each": [],
            "where": [],
            "variable": "v",
        }
        assert set(TEMPLATE_KEYS) <= set(dumped)
        assert set(OBJECTIVE_TEMPLATE_KEYS) <= set(dumped["objective"])
