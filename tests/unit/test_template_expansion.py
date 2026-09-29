"""Static checks and expansion rules of ``expand_problem`` (validation/expansion).

Schema 1.3 spec 2026-09-25 §14.3-§14.11 and §14.16 items 2, 3, 5 and 8 (v4),
with the user's choices of §15 (5 = B: unused generated variables are left
out with UNUSED_TEMPLATE_VARIABLES):

* every expansion error code has at least one triggering case whose path
  and message are asserted exactly: FEATURE_REQUIRES_NEWER_VERSION (§14.10,
  with the notes of §6.4), TEMPLATE_REFERENCE_INVALID (each trigger of the
  §14.10 table, including the "every bound index is used" rule of §14.5),
  INDEX_SET_INVALID (§14.3), PARAMETER_TABLE_INVALID and
  PARAMETER_VALUE_MISSING (§14.4), DUPLICATE_TEMPLATE_NAME (§14.5, §14.7),
  and the reused codes the expander raises statically (§14.10);
* every branch of the evaluation order of §14.7 -- where short-circuits, a
  missing where value is an error even for an entry that would then be
  skipped, an out-of-bounds objective term is left out alone, an
  out-of-bounds constraint is left out whole without looking up its
  parameters, bounds are only judged for bindings that pass where, cyclic
  wrap-around, a shift moves by position, ``<`` compares positions and
  numbers compare exactly;
* names (§14.7): ``family[e1,e2]`` without spaces, negative integers, the
  generated constraint id with and without an outer for_each, the order of
  the expanded lists, and a property test that names are injective;
* the four expansion warnings (§14.7, §14.10, §14.14 item 2);
* generated items carry exactly the field types ``model_validate`` would
  give them (§14.8), recursively;
* a problem without templates is returned as it is (§14.8);
* malicious input never raises and only yields the codes of §14.16 item 8.
"""

import builtins
import itertools
import json
import random

import pytest
from pydantic import BaseModel

from annealbridge.interfaces.problem_input import parse_problem
from annealbridge.models import (
    CardinalityConstraint,
    Constraint,
    LinearTerm,
    OptimizationProblem,
    QuadraticTerm,
    Variable,
)
from annealbridge.validation import ExpandedProblem, expand_problem
from annealbridge.validation.issues import _WARNING_RECOMMENDED_ACTIONS
from tests.conftest import EXAMPLES_DIR

S = {"name": "s", "elements": ["a", "b", "c"]}
T = {"name": "t", "elements": [0, 1, 2], "order": "linear"}
X = {"name": "x", "indices": ["s"]}
Y = {"name": "y", "indices": ["t"]}
NAME_RULE = (
    "an identifier: letters, digits and underscores, not starting with a "
    "digit, at most 64 characters"
)
LITERAL_HINT = (
    "literal elements are not supported: bind an index with for_each and "
    "compare a 0/1 parameter such as is_first[p] == 1 instead"
)
UNUSED_INDEX = (
    "is bound but no reference uses it, so every entry would be repeated "
    "once per element; use it in a reference (or in an inner where), or "
    "remove it"
)


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


def objective(linear=(), quadratic=(), terms=()) -> dict:
    data = {"direction": "minimize", "linear_terms": list(terms)}
    if linear:
        data["linear_term_templates"] = list(linear)
    if quadratic:
        data["quadratic_term_templates"] = list(quadratic)
    return data


def term(variable="x[i]", coefficient=1, for_each=("i in s",), where=()) -> dict:
    data = {"coefficient": coefficient, "variable": variable}
    if for_each:
        data["for_each"] = list(for_each)
    if where:
        data["where"] = list(where)
    return data


def pair(variable1, variable2, coefficient=1, for_each=(), where=()) -> dict:
    data = {"coefficient": coefficient, "variable1": variable1, "variable2": variable2}
    if for_each:
        data["for_each"] = list(for_each)
    if where:
        data["where"] = list(where)
    return data


def linear_constraint(**overrides) -> dict:
    data = {
        "id": "c",
        "type": "hard",
        "for_each": ["i in s"],
        "terms": [{"coefficient": 1, "variable": "x[i]"}],
        "operator": "<=",
        "rhs": 1,
    }
    data.update(overrides)
    return data


def cardinality(**overrides) -> dict:
    data = {
        "id": "k",
        "type": "hard",
        "for_each": ["i in s"],
        "variables": ["x[i]"],
        "operator": "<=",
        "rhs": 1,
    }
    data.update(overrides)
    return data


def parameter(name, indices, rows=(), default=None) -> dict:
    data = {
        "name": name,
        "indices": list(indices),
        "values": [{"key": list(key), "value": value} for key, value in rows],
    }
    if default is not None:
        data["default"] = default
    return data


def expand(data: dict, **kwargs) -> ExpandedProblem:
    return expand_problem(OptimizationProblem.model_validate(data), **kwargs)


def triples(items) -> list[tuple[str, str | None, str]]:
    return [(item.code, item.path, item.message) for item in items]


def failed(data: dict, **kwargs) -> list[tuple[str, str | None, str]]:
    """The expansion errors of ``data``; there must be some, and nothing else."""
    result = expand(data, **kwargs)
    assert result.errors, "expected expansion errors"
    assert result.problem is None
    assert result.warnings == ()
    assert result.limit_exceeded is False
    return triples(result.errors)


def only_error(data: dict, **kwargs) -> tuple[str, str | None, str]:
    found = failed(data, **kwargs)
    assert len(found) == 1, found
    return found[0]


def expanded(data: dict, **kwargs) -> ExpandedProblem:
    result = expand(data, **kwargs)
    assert result.errors == (), triples(result.errors)
    assert result.problem is not None
    return result


def names(result: ExpandedProblem) -> list[str]:
    return [variable.name for variable in result.problem.variables]


def linear_pairs(result: ExpandedProblem) -> list[tuple[str, float]]:
    return [(t.variable, t.coefficient) for t in result.problem.objective.linear_terms]


def constraint_rows(result: ExpandedProblem):
    return [
        (c.id, [(t.variable, t.coefficient) for t in c.terms], c.operator, c.rhs, c.weight)
        for c in result.problem.constraints
    ]


def warning_codes(result: ExpandedProblem) -> list[str]:
    return [warning.code for warning in result.warnings]


def _example(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# No templates
# ---------------------------------------------------------------------------


class TestNoTemplates:
    """A problem without template fields is returned untouched (§14.8)."""

    @pytest.mark.parametrize(
        "name", ["knapsack.json", "integer_knapsack.json", "exam_timetabling.json", "tsp.json"]
    )
    def test_the_problem_itself(self, name):
        problem = OptimizationProblem.model_validate(_example(name))
        result = expand_problem(problem)
        assert result.source is problem
        assert result.problem is problem
        assert result.errors == ()
        assert result.warnings == ()
        assert result.limit_exceeded is False
        assert result.templated is False

    def test_a_1_3_problem_with_empty_template_lists(self):
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
        problem = OptimizationProblem.model_validate(data)
        result = expand_problem(problem, max_template_bindings=1)
        assert result.problem is problem
        assert (result.errors, result.warnings, result.limit_exceeded) == ((), (), False)


# ---------------------------------------------------------------------------
# FEATURE_REQUIRES_NEWER_VERSION
# ---------------------------------------------------------------------------


def _all_template_fields(version: str | None) -> dict:
    data = document(
        index_sets=[S],
        parameters=[parameter("w", ["s"], default=1)],
        variable_families=[X],
        objective=objective(
            linear=[term(coefficient="w[i]")],
            quadratic=[pair("x[i]", "x[j]", for_each=["i in s", "j in s"], where=["i < j"])],
        ),
        constraint_templates=[linear_constraint()],
        cardinality_constraint_templates=[cardinality(id="k")],
    )
    if version is None:
        del data["version"]
    else:
        data["version"] = version
    return data


class TestVersionGate:
    """Template fields need version "1.3" (§14.10, notes as in §6.4)."""

    @pytest.mark.parametrize("version", ["1.0", "1.1", "1.2"])
    def test_every_field_in_declaration_order(self, version):
        assert only_error(_all_template_fields(version)) == (
            "FEATURE_REQUIRES_NEWER_VERSION",
            "version",
            "index_sets, parameters, variable_families, "
            "objective.linear_term_templates, objective.quadratic_term_templates, "
            "constraint_templates and cardinality_constraint_templates require "
            f"version \"1.3\" or later, but version is '{version}'",
        )

    def test_1_0_one_field(self):
        data = document(version="1.0", index_sets=[S])
        assert only_error(data) == (
            "FEATURE_REQUIRES_NEWER_VERSION",
            "version",
            "index_sets requires version \"1.3\" or later, but version is '1.0'",
        )

    def test_1_1_two_fields(self):
        data = document(
            version="1.1",
            index_sets=[S],
            variable_families=[X],
            objective=objective(quadratic=[pair("x[i]", "x[j]", for_each=["i in s", "j in s"], where=["i != j"])]),
        )
        assert only_error(data) == (
            "FEATURE_REQUIRES_NEWER_VERSION",
            "version",
            "index_sets, variable_families and objective.quadratic_term_templates "
            "require version \"1.3\" or later, but version is '1.1'",
        )

    def test_1_2_objective_field_carries_its_prefix(self):
        data = document(
            version="1.2",
            index_sets=[S],
            variable_families=[X],
            objective=objective(linear=[term()]),
        )
        assert only_error(data) == (
            "FEATURE_REQUIRES_NEWER_VERSION",
            "version",
            "index_sets, variable_families and objective.linear_term_templates "
            "require version \"1.3\" or later, but version is '1.2'",
        )

    def test_omitted_version_names_its_default(self):
        data = document(index_sets=[S])
        del data["version"]
        assert only_error(data) == (
            "FEATURE_REQUIRES_NEWER_VERSION",
            "version",
            "index_sets requires version \"1.3\" or later, but version is '1.0'; "
            'version defaults to "1.0" when omitted',
        )

    @pytest.mark.parametrize("version", ["1.0", "1.1"])
    def test_cardinality_constraints_are_mentioned_below_1_2(self, version):
        data = document(
            version=version,
            variables=[{"name": "v"}],
            index_sets=[S],
            cardinality_constraints=[
                {"id": "one", "type": "hard", "variables": ["v"], "operator": "<=", "rhs": 1}
            ],
        )
        assert only_error(data) == (
            "FEATURE_REQUIRES_NEWER_VERSION",
            "version",
            f"index_sets requires version \"1.3\" or later, but version is '{version}'; "
            '"1.3" also allows cardinality_constraints',
        )

    def test_cardinality_constraints_are_not_mentioned_at_1_2(self):
        data = document(
            version="1.2",
            variables=[{"name": "v"}],
            index_sets=[S],
            cardinality_constraints=[
                {"id": "one", "type": "hard", "variables": ["v"], "operator": "<=", "rhs": 1}
            ],
        )
        assert only_error(data)[2] == (
            "index_sets requires version \"1.3\" or later, but version is '1.2'"
        )

    def test_an_explicit_integer_variable_is_mentioned_at_1_0(self):
        data = document(
            version="1.0",
            variables=[{"name": "n", "type": "integer", "lower_bound": 0, "upper_bound": 3}],
            index_sets=[S],
        )
        assert only_error(data)[2] == (
            "index_sets requires version \"1.3\" or later, but version is '1.0'; "
            '"1.3" also allows the integer variables'
        )

    def test_an_integer_family_is_mentioned_at_1_0(self):
        data = document(
            version="1.0",
            index_sets=[S],
            variable_families=[
                {"name": "n", "indices": ["s"], "type": "integer", "lower_bound": 0, "upper_bound": 3}
            ],
        )
        assert only_error(data)[2] == (
            "index_sets and variable_families require version \"1.3\" or later, "
            "but version is '1.0'; \"1.3\" also allows the integer variables"
        )

    def test_integers_are_not_mentioned_at_1_1(self):
        data = document(
            version="1.1",
            variables=[{"name": "n", "type": "integer", "lower_bound": 0, "upper_bound": 3}],
            index_sets=[S],
        )
        assert only_error(data)[2] == (
            "index_sets requires version \"1.3\" or later, but version is '1.1'"
        )

    def test_every_note_in_order(self):
        data = document(
            variables=[
                {"name": "v"},
                {"name": "n", "type": "integer", "lower_bound": 0, "upper_bound": 3},
            ],
            index_sets=[S],
            cardinality_constraints=[
                {"id": "one", "type": "hard", "variables": ["v"], "operator": "<=", "rhs": 1}
            ],
        )
        del data["version"]
        assert only_error(data)[2] == (
            "index_sets requires version \"1.3\" or later, but version is '1.0'; "
            'version defaults to "1.0" when omitted; "1.3" also allows '
            'cardinality_constraints; "1.3" also allows the integer variables'
        )

    def test_other_static_errors_are_reported_with_it(self):
        data = document(
            version="1.2",
            index_sets=[S, {"name": "e", "elements": []}],
            variable_families=[{"name": "x", "indices": ["nope"]}],
        )
        assert failed(data) == [
            (
                "FEATURE_REQUIRES_NEWER_VERSION",
                "version",
                "index_sets and variable_families require version \"1.3\" or later, "
                "but version is '1.2'",
            ),
            (
                "INDEX_SET_INVALID",
                "index_sets[1].elements",
                'Index set "e" has no elements; list at least one',
            ),
            (
                "TEMPLATE_REFERENCE_INVALID",
                "variable_families[0].indices[0]",
                'Variable family "x" names "nope", which is not a declared (valid) index set',
            ),
        ]

    def test_1_3_is_accepted(self):
        result = expanded(document(index_sets=[S], variable_families=[X], objective=objective(linear=[term()])))
        assert result.problem.version == "1.3"


# ---------------------------------------------------------------------------
# TEMPLATE_REFERENCE_INVALID
# ---------------------------------------------------------------------------


def _linear(*templates, **fields) -> dict:
    return document(
        index_sets=fields.pop("index_sets", [S, T]),
        variable_families=fields.pop("variable_families", [X, {"name": "z", "indices": ["s", "t"]}]),
        objective=objective(linear=templates),
        **fields,
    )


class TestTemplateReferenceInvalid:
    """Every trigger of the §14.10 table."""

    @pytest.mark.parametrize(
        "fields, path, message",
        [
            (
                {"index_sets": [{"name": "1s", "elements": ["a"]}]},
                "index_sets[0].name",
                f'The name of index set "1s" must be {NAME_RULE}, and not the word "in"',
            ),
            (
                {"index_sets": [{"name": "n" * 65, "elements": ["a"]}]},
                "index_sets[0].name",
                f'The name of index set "{"n" * 64}"... must be {NAME_RULE}, and not the word "in"',
            ),
            (
                {"index_sets": [{"name": "in", "elements": ["a"]}]},
                "index_sets[0].name",
                f'The name of index set "in" must be {NAME_RULE}, and not the word "in"',
            ),
            (
                {"index_sets": [S], "parameters": [parameter("in", ["s"], default=1)]},
                "parameters[0].name",
                f'The name of parameter "in" must be {NAME_RULE}, and not the word "in"',
            ),
            (
                {"index_sets": [S], "variable_families": [{"name": "x-y", "indices": ["s"]}]},
                "variable_families[0].name",
                f'The name of variable family "x-y" must be {NAME_RULE}, and not the word "in"',
            ),
            (
                {"index_sets": [S], "variable_families": [X], "constraint_templates": [linear_constraint(id="1c")]},
                "constraint_templates[0].id",
                f'The template id "1c" must be {NAME_RULE}',
            ),
            (
                {"index_sets": [S], "variable_families": [X], "cardinality_constraint_templates": [cardinality(id="in")]},
                "cardinality_constraint_templates[0].id",
                f'The template id "in" must be {NAME_RULE}',
            ),
        ],
        ids=["digit", "too long", "in", "parameter in", "family", "template id", "template id in"],
    )
    def test_declared_names(self, fields, path, message):
        assert only_error(document(**fields)) == ("TEMPLATE_REFERENCE_INVALID", path, message)

    def test_too_many_for_each_items(self):
        data = _linear(term(for_each=[f"i{k} in s" for k in range(9)]))
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].for_each",
            "for_each has 9 items; at most 8 are allowed",
        )

    def test_too_many_where_conditions(self):
        data = _linear(term(where=["i == i"] * 9))
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].where",
            "where has 9 conditions; at most 8 are allowed",
        )

    def test_eight_of_each_are_allowed(self):
        loops = [f"i{k} in s" for k in range(8)]
        data = document(
            index_sets=[S],
            variable_families=[{"name": "x", "indices": ["s"] * 8}],
            objective=objective(
                linear=[
                    term(
                        variable="x[" + ",".join(f"i{k}" for k in range(8)) + "]",
                        for_each=loops,
                        where=["i0 == i1"] * 8,
                    )
                ]
            ),
        )
        assert len(expanded(data).problem.objective.linear_terms) == 3**7

    @pytest.mark.parametrize(
        "declaration, path, owner, count",
        [
            ({"variable_families": [{"name": "x", "indices": []}]}, "variable_families[0]", 'Variable family "x"', 0),
            ({"variable_families": [{"name": "x", "indices": ["s"] * 9}]}, "variable_families[0]", 'Variable family "x"', 9),
            ({"parameters": [parameter("p", [], default=1)]}, "parameters[0]", 'Parameter "p"', 0),
            ({"parameters": [parameter("p", ["s"] * 9, default=1)]}, "parameters[0]", 'Parameter "p"', 9),
        ],
        ids=["family 0", "family 9", "parameter 0", "parameter 9"],
    )
    def test_one_to_eight_indices(self, declaration, path, owner, count):
        assert only_error(document(index_sets=[S], **declaration)) == (
            "TEMPLATE_REFERENCE_INVALID",
            f"{path}.indices",
            f"{owner} needs one to 8 index sets in indices, got {count}",
        )

    @pytest.mark.parametrize(
        "template, path, message",
        [
            (term(variable="x[i"), "variable", 'expected "," or "]" at character 4, found the end of the text'),
            (term(for_each=["i on s"]), "for_each[0]", 'expected "in" (a for_each item is "index in set") at character 3, found "o"'),
            (term(where=["i = i"]), "where[0]", 'use "==" to compare (character 3 is a single "=")'),
            (term(coefficient="w[i"), "coefficient", 'expected "," or "]" at character 4, found the end of the text'),
        ],
        ids=["variable", "for_each", "where", "coefficient"],
    )
    def test_syntax_errors(self, template, path, message):
        assert only_error(_linear(template)) == (
            "TEMPLATE_REFERENCE_INVALID",
            f"objective.linear_term_templates[0].{path}",
            message,
        )

    @pytest.mark.parametrize(
        "template, path, message",
        [
            (term(variable="q[i]"), "variable", "q is not a declared variable family"),
            (term(coefficient="q[i]"), "coefficient", "q is not a declared parameter"),
            (term(for_each=["i in u"]), "for_each[0]", "u is not a declared (valid) index set"),
            (term(variable="x[k]"), "variable", f"k is not an index bound by for_each; {LITERAL_HINT}"),
            (term(where=["k == i"]), "where[0]", f"k is not an index bound by for_each; {LITERAL_HINT}"),
            (term(where=["q[i] == 1"]), "where[0]", "q is not a declared parameter"),
        ],
        ids=["family", "parameter", "set", "index", "where index", "where parameter"],
    )
    def test_unknown_names(self, template, path, message):
        assert only_error(_linear(template)) == (
            "TEMPLATE_REFERENCE_INVALID",
            f"objective.linear_term_templates[0].{path}",
            message,
        )

    @pytest.mark.parametrize(
        "template, path, message",
        [
            (term(variable="s[i]"), "variable", "s is an index set, not a variable family"),
            (term(variable="w[i]"), "variable", "w is a parameter, not a variable family"),
            (term(coefficient="x[i]"), "coefficient", "x is a variable family, not a parameter"),
            (term(coefficient="s[i]"), "coefficient", "s is an index set, not a parameter"),
            (term(coefficient="w"), "coefficient", "Parameter w needs its indices in brackets, e.g. w[i]"),
            (term(where=["w == 1"]), "where[0]", "w needs its indices in brackets, e.g. w[i]"),
            (term(where=["s == i"]), "where[0]", 's is an index set, not an index; bind an index with for_each ("i in s") and use that'),
            (term(variable="x"), "variable", "Variable family x needs its indices, e.g. x[i]"),
        ],
        ids=["set as family", "parameter as family", "family as parameter", "set as parameter", "bare parameter", "bare parameter in where", "set as index", "bare family"],
    )
    def test_names_of_the_wrong_kind(self, template, path, message):
        data = _linear(template, parameters=[parameter("w", ["s"], default=1)])
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            f"objective.linear_term_templates[0].{path}",
            message,
        )

    def test_a_family_whose_indices_do_not_resolve(self):
        data = _linear(
            term(variable="b[i]"),
            variable_families=[X, {"name": "b", "indices": ["nope"]}],
        )
        assert failed(data) == [
            (
                "TEMPLATE_REFERENCE_INVALID",
                "variable_families[1].indices[0]",
                'Variable family "b" names "nope", which is not a declared (valid) index set',
            ),
            (
                "TEMPLATE_REFERENCE_INVALID",
                "objective.linear_term_templates[0].variable",
                "b is a variable family whose declaration has errors",
            ),
        ]

    def test_a_parameter_whose_indices_do_not_resolve(self):
        data = _linear(term(coefficient="w[i]"), parameters=[parameter("w", ["nope"], default=1)])
        assert failed(data) == [
            (
                "TEMPLATE_REFERENCE_INVALID",
                "parameters[0].indices[0]",
                'Parameter "w" names "nope", which is not a declared (valid) index set',
            ),
            (
                "TEMPLATE_REFERENCE_INVALID",
                "objective.linear_term_templates[0].coefficient",
                "w is a parameter whose declaration has errors",
            ),
        ]

    def test_wrong_index_count(self):
        assert only_error(_linear(term(variable="z[i]"))) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].variable",
            "Variable family z has 2 indices (s, t), but the reference gives 1",
        )

    def test_wrong_index_set(self):
        assert only_error(_linear(term(variable="x[p]", for_each=["p in t"]))) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].variable",
            "Index p ranges over t, but position 0 of variable family x is s",
        )

    def test_wrong_index_set_of_a_parameter(self):
        data = _linear(term(coefficient="w[p]", variable="z[i,p]", for_each=["i in s", "p in t"]), parameters=[parameter("w", ["s"], default=1)])
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].coefficient",
            "Index p ranges over t, but position 0 of parameter w is s",
        )

    def test_shift_on_a_set_without_order(self):
        assert only_error(_linear(term(variable="x[i+1]"))) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].variable",
            'Index i is shifted, but index set s has no order; declare its order '
            '"linear" or "cyclic" to allow shifts',
        )

    def test_shift_inside_where(self):
        data = _linear(
            term(variable="z[i,p]", for_each=["i in s", "p in t"], where=["w[p+1] == 1"]),
            parameters=[parameter("w", ["t"], default=1)],
        )
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].where[0]",
            "the index at character 3 is shifted, but a where condition cannot "
            "shift an index; shifts are only allowed in the generated entry's own "
            "references",
        )

    def test_index_compared_with_a_number(self):
        assert only_error(_linear(term(where=["i == 0"]))) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].where[0]",
            f"The condition compares an index with a number; {LITERAL_HINT}",
        )

    def test_index_compared_with_a_parameter(self):
        data = _linear(term(where=["w[i] < i"]), parameters=[parameter("w", ["s"], default=1)])
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].where[0]",
            f"The condition compares an index with a number; {LITERAL_HINT}",
        )

    def test_indices_of_different_sets_compared(self):
        data = _linear(term(variable="z[i,p]", for_each=["i in s", "p in t"], where=["i == p"]))
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].where[0]",
            "The condition compares an index of s with an index of t; both sides "
            "must range over the same index set",
        )

    def test_a_literal_element_name(self):
        assert only_error(_linear(term(variable="x[a]", for_each=()))) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].variable",
            f"a is not an index bound by for_each; {LITERAL_HINT}",
        )

    def test_a_literal_integer_element(self):
        assert only_error(_linear(term(variable="z[i,0]"))) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].variable",
            "expected an index name (literal elements such as a or 0 are not "
            "supported; bind an index with for_each) at character 5, found \"0\"",
        )

    @pytest.mark.parametrize("text", ["2", " -1.5 ", "1e999", "0"])
    def test_a_number_written_as_a_string(self, text):
        assert only_error(_linear(term(coefficient=text))) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].coefficient",
            "Write the number as a JSON number, not as a string",
        )

    def test_a_number_written_as_a_string_in_rhs_and_weight(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            constraint_templates=[linear_constraint(type="soft", rhs="1", weight="2")],
        )
        message = "Write the number as a JSON number, not as a string"
        assert failed(data) == [
            ("TEMPLATE_REFERENCE_INVALID", "constraint_templates[0].rhs", message),
            ("TEMPLATE_REFERENCE_INVALID", "constraint_templates[0].weight", message),
        ]


class TestEveryBoundIndexIsUsed:
    """§14.5: a bound index appears in a reference or an inner where."""

    def test_an_index_no_reference_uses(self):
        assert only_error(_linear(term(for_each=["i in s", "j in s"]))) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].for_each[1]",
            f"Index j {UNUSED_INDEX}",
        )

    def test_an_index_only_its_own_where_uses(self):
        data = _linear(term(for_each=["i in s", "j in s"], where=["i != j"]))
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "objective.linear_term_templates[0].for_each[1]",
            f"Index j {UNUSED_INDEX}",
        )

    def test_an_outer_index_only_the_outer_where_uses(self):
        data = document(
            index_sets=[S, {"name": "g", "elements": ["g1", "g2"]}],
            parameters=[parameter("open", ["g"], default=1)],
            variable_families=[X],
            cardinality_constraint_templates=[
                cardinality(for_each=["h in g"], where=["open[h] == 1"], variables=[{"for_each": ["i in s"], "variable": "x[i]"}])
            ],
        )
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "cardinality_constraint_templates[0].for_each[0]",
            f"Index h {UNUSED_INDEX}",
        )

    def test_an_outer_index_an_inner_where_uses(self):
        """The exam_timetabling pattern: members listed per group (§14.5 v4)."""
        data = document(
            index_sets=[
                {"name": "groups", "elements": ["g1", "g2"]},
                {"name": "exams", "elements": ["e1", "e2", "e3"]},
            ],
            parameters=[
                parameter(
                    "in_group",
                    ["groups", "exams"],
                    rows=[(("g1", "e1"), 1), (("g1", "e2"), 1), (("g2", "e3"), 1)],
                    default=0,
                )
            ],
            variable_families=[{"name": "x", "indices": ["exams"]}],
            cardinality_constraint_templates=[
                cardinality(
                    id="one_per_group",
                    for_each=["g in groups"],
                    variables=[
                        {"for_each": ["e in exams"], "where": ["in_group[g,e] == 1"], "variable": "x[e]"}
                    ],
                )
            ],
        )
        result = expanded(data)
        assert [(c.id, c.variables) for c in result.problem.cardinality_constraints] == [
            ("one_per_group[g1]", ["x[e1]", "x[e2]"]),
            ("one_per_group[g2]", ["x[e3]"]),
        ]

    def test_an_inner_index_only_its_own_where_uses(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("w", ["s"], default=1)],
            variable_families=[X],
            cardinality_constraint_templates=[
                cardinality(for_each=[], variables=[{"for_each": ["i in s", "j in s"], "where": ["w[j] == 1"], "variable": "x[i]"}])
            ],
        )
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "cardinality_constraint_templates[0].variables[0].for_each[1]",
            f"Index j {UNUSED_INDEX}",
        )

    def test_an_outer_index_the_rhs_uses(self):
        data = document(
            index_sets=[S, {"name": "g", "elements": ["g1", "g2"]}],
            parameters=[parameter("r", ["g"], default=1)],
            variable_families=[X],
            cardinality_constraint_templates=[
                cardinality(for_each=["h in g"], rhs="r[h]", variables=[{"for_each": ["i in s"], "variable": "x[i]"}])
            ],
        )
        assert [c.id for c in expanded(data).problem.cardinality_constraints] == ["k[g1]", "k[g2]"]

    def test_an_outer_index_a_member_coefficient_uses(self):
        data = document(
            index_sets=[S, {"name": "g", "elements": ["g1", "g2"]}],
            parameters=[parameter("a", ["g", "s"], default=1)],
            variable_families=[X],
            constraint_templates=[
                linear_constraint(for_each=["h in g"], terms=[{"for_each": ["i in s"], "coefficient": "a[h,i]", "variable": "x[i]"}])
            ],
        )
        assert [c.id for c in expanded(data).problem.constraints] == ["c[g1]", "c[g2]"]

    def test_an_outer_index_no_member_uses(self):
        data = document(
            index_sets=[S, {"name": "g", "elements": ["g1", "g2"]}],
            variable_families=[X],
            constraint_templates=[
                linear_constraint(for_each=["h in g"], terms=[{"for_each": ["i in s"], "coefficient": 1, "variable": "x[i]"}])
            ],
        )
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "constraint_templates[0].for_each[0]",
            f"Index h {UNUSED_INDEX}",
        )

    def test_no_knock_on_report_when_a_reference_fails(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            constraint_templates=[linear_constraint(terms=[{"coefficient": 1, "variable": "q[i]"}])],
        )
        assert only_error(data) == (
            "TEMPLATE_REFERENCE_INVALID",
            "constraint_templates[0].terms[0].variable",
            "q is not a declared variable family",
        )

    def test_sibling_members_may_reuse_an_index_name(self):
        data = document(
            index_sets=[S, T],
            variable_families=[X, Y],
            constraint_templates=[
                linear_constraint(
                    for_each=[],
                    terms=[
                        {"for_each": ["i in s"], "coefficient": 1, "variable": "x[i]"},
                        {"for_each": ["i in t"], "coefficient": -1, "variable": "y[i]"},
                    ],
                )
            ],
        )
        assert constraint_rows(expanded(data)) == [
            (
                "c",
                [("x[a]", 1.0), ("x[b]", 1.0), ("x[c]", 1.0), ("y[0]", -1.0), ("y[1]", -1.0), ("y[2]", -1.0)],
                "<=",
                1.0,
                None,
            )
        ]


# ---------------------------------------------------------------------------
# INDEX_SET_INVALID
# ---------------------------------------------------------------------------


class TestIndexSetInvalid:
    """§14.3: non-empty, one type, no repeats, ASCII strings, bounded ints."""

    def test_an_empty_set(self):
        assert only_error(document(index_sets=[{"name": "s", "elements": []}])) == (
            "INDEX_SET_INVALID",
            "index_sets[0].elements",
            'Index set "s" has no elements; list at least one',
        )

    @pytest.mark.parametrize("elements", [["a", 1, "b"], [1, "a"]], ids=["string first", "integer first"])
    def test_mixed_types(self, elements):
        assert only_error(document(index_sets=[{"name": "s", "elements": elements}])) == (
            "INDEX_SET_INVALID",
            "index_sets[0].elements[1]",
            'Index set "s" mixes strings and integers; its elements must be all '
            "strings or all integers",
        )

    @pytest.mark.parametrize(
        "elements, shown", [(["a", "b", "a"], '"a"'), ([1, 2, 1], "1")], ids=["string", "integer"]
    )
    def test_a_repeated_element(self, elements, shown):
        assert only_error(document(index_sets=[{"name": "s", "elements": elements}])) == (
            "INDEX_SET_INVALID",
            "index_sets[0].elements[2]",
            f'Element {shown} is listed more than once in index set "s"',
        )

    def test_string_characters_and_length(self):
        bad = ["a b", "a,b", "a[b]", "é", "a\n", "", "x" * 65, "١", "a　"]
        shown = ['"a b"', '"a,b"', '"a[b]"', '"\\u00e9"', '"a\\n"', '""', f'"{"x" * 64}"...', '"\\u0661"', '"a\\u3000"']
        data = document(index_sets=[{"name": "s", "elements": [*bad, "ok.-_9", "x" * 64]}])
        assert failed(data) == [
            (
                "INDEX_SET_INVALID",
                f"index_sets[0].elements[{j}]",
                f'Element {text} of index set "s" must be 1 to 64 letters, digits, '
                "underscores, dots or hyphens",
            )
            for j, text in enumerate(shown)
        ]

    def test_integer_range(self):
        data = document(
            index_sets=[{"name": "s", "elements": [2**31, -(2**31 - 1), 2**31 - 1, -(2**31), 10**400]}]
        )
        assert failed(data) == [
            (
                "INDEX_SET_INVALID",
                f"index_sets[0].elements[{j}]",
                f'The element at position {j} of index set "s" lies outside ±2147483647',
            )
            for j in (0, 3, 4)
        ]

    def test_valid_sets_expand(self):
        data = document(
            index_sets=[
                {"name": "s", "elements": ["A.b-c_9", "0", "-1"]},
                {"name": "t", "elements": [2**31 - 1, -(2**31 - 1)]},
            ],
            variable_families=[{"name": "x", "indices": ["s", "t"]}],
            objective=objective(linear=[term(variable="x[i,p]", for_each=["i in s", "p in t"])]),
        )
        assert names(expanded(data)) == [
            "x[A.b-c_9,2147483647]",
            "x[A.b-c_9,-2147483647]",
            "x[0,2147483647]",
            "x[0,-2147483647]",
            "x[-1,2147483647]",
            "x[-1,-2147483647]",
        ]


# ---------------------------------------------------------------------------
# PARAMETER_TABLE_INVALID / PARAMETER_VALUE_MISSING
# ---------------------------------------------------------------------------


class TestParameterTableInvalid:
    """§14.4: one row per key, keys of the declared sets, finite values."""

    def _table(self, *parameters) -> dict:
        return document(index_sets=[S, T], parameters=list(parameters))

    @pytest.mark.parametrize(
        "indices, key, message",
        [
            (["s", "t"], ("a",), "is indexed by 2 index sets, but row 0 has a key of 1 element"),
            (["s"], ("a", "b"), "is indexed by 1 index set, but row 0 has a key of 2 elements"),
            (["s", "t"], ("a", 0, 1), "is indexed by 2 index sets, but row 0 has a key of 3 elements"),
            (["s"], (), "is indexed by 1 index set, but row 0 has a key of 0 elements"),
        ],
        ids=["short", "long", "longer", "empty"],
    )
    def test_key_length(self, indices, key, message):
        data = self._table(parameter("p", indices, rows=[(key, 1)]))
        assert only_error(data) == (
            "PARAMETER_TABLE_INVALID",
            "parameters[0].values[0].key",
            f'Parameter "p" {message}',
        )

    def test_key_elements_outside_their_sets(self):
        data = self._table(parameter("p", ["s", "t"], rows=[(("z", 0), 1), (("a", 5), 1)]))
        assert failed(data) == [
            (
                "PARAMETER_TABLE_INVALID",
                "parameters[0].values[0].key[0]",
                '"z" is not an element of index set s, which position 0 of '
                'Parameter "p"\'s key ranges over',
            ),
            (
                "PARAMETER_TABLE_INVALID",
                "parameters[0].values[1].key[1]",
                '5 is not an element of index set t, which position 1 of '
                'Parameter "p"\'s key ranges over',
            ),
        ]

    @pytest.mark.parametrize(
        "sets, key, shown",
        [
            ([T], "0", '"0"'),
            ([{"name": "t", "elements": ["1", "2"]}], 1, "1"),
        ],
        ids=["string for an integer set", "integer for a string set"],
    )
    def test_key_elements_of_the_wrong_type(self, sets, key, shown):
        data = document(index_sets=sets, parameters=[parameter("p", ["t"], rows=[((key,), 1)])])
        assert only_error(data) == (
            "PARAMETER_TABLE_INVALID",
            "parameters[0].values[0].key[0]",
            f'{shown} is not an element of index set t, which position 0 of '
            'Parameter "p"\'s key ranges over',
        )

    def test_a_long_or_huge_key_element_is_shown_bounded(self):
        data = self._table(
            parameter("p", ["s"], rows=[(("Q" * 100,), 1)]),
            parameter("q", ["t"], rows=[((10**400,), 1)]),
        )
        assert failed(data) == [
            (
                "PARAMETER_TABLE_INVALID",
                "parameters[0].values[0].key[0]",
                f'"{"Q" * 64}"... is not an element of index set s, which position 0 '
                'of Parameter "p"\'s key ranges over',
            ),
            (
                "PARAMETER_TABLE_INVALID",
                "parameters[1].values[0].key[0]",
                "<an integer of 401 digits> is not an element of index set t, "
                'which position 0 of Parameter "q"\'s key ranges over',
            ),
        ]

    def test_a_repeated_key(self):
        data = self._table(parameter("p", ["s", "t"], rows=[(("a", 0), 1), (("a", 0), 2)]))
        assert only_error(data) == (
            "PARAMETER_TABLE_INVALID",
            "parameters[0].values[1].key",
            'Parameter "p" has more than one row for the key (a,0)',
        )

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"])
    def test_a_value_that_is_not_finite(self, value):
        data = self._table(parameter("p", ["s"], rows=[(("a",), 1), (("b",), value)]))
        assert only_error(data) == (
            "PARAMETER_TABLE_INVALID",
            "parameters[0].values[1].value",
            'Parameter "p" has a value that is not a finite number in row 1',
        )

    def test_a_default_that_is_not_finite(self):
        data = self._table(parameter("p", ["s"], default=float("inf")))
        assert only_error(data) == (
            "PARAMETER_TABLE_INVALID",
            "parameters[0].default",
            'Parameter "p" has a default that is not a finite number',
        )

    def test_rows_are_checked_in_order(self):
        data = self._table(
            parameter("p", ["s"], rows=[(("z",), 1), (("a",), 1), (("a",), 1), (("b", "c"), 1)])
        )
        assert [path for _, path, _ in failed(data)] == [
            "parameters[0].values[0].key[0]",
            "parameters[0].values[2].key",
            "parameters[0].values[3].key",
        ]

    def test_a_sparse_table_is_fine(self):
        """Rows left out are no error while no generated entry uses them."""
        data = document(
            index_sets=[S],
            parameters=[
                parameter("has", ["s"], rows=[(("a",), 0), (("b",), 1), (("c",), 0)]),
                parameter("w", ["s"], rows=[(("b",), 4)]),
            ],
            variable_families=[X],
            objective=objective(linear=[term(coefficient="w[i]", where=["has[i] == 1"])]),
        )
        assert linear_pairs(expanded(data)) == [("x[b]", 4.0)]

    def test_a_cardinality_rhs_parameter_holds_whole_numbers(self):
        data = document(
            index_sets=[S],
            parameters=[
                parameter("r", ["s"], rows=[(("a",), 1), (("b",), 1.5), (("c",), 3e9)], default=0.5)
            ],
            variable_families=[X],
            cardinality_constraint_templates=[cardinality(rhs="r[i]")],
        )
        prefix = (
            "Parameter r is used as a cardinality rhs, so every value must be a "
            "whole number within ±2147483647, but it holds"
        )
        assert failed(data) == [
            ("PARAMETER_TABLE_INVALID", "parameters[0].values[1].value", f"{prefix} 1.5"),
            ("PARAMETER_TABLE_INVALID", "parameters[0].values[2].value", f"{prefix} 3000000000.0"),
            ("PARAMETER_TABLE_INVALID", "parameters[0].default", f"{prefix} 0.5"),
        ]

    def test_a_linear_rhs_parameter_may_hold_fractions(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("r", ["s"], default=0.5)],
            variable_families=[X],
            constraint_templates=[linear_constraint(operator="==", rhs="r[i]")],
        )
        assert [c.rhs for c in expanded(data).problem.constraints] == [0.5, 0.5, 0.5]

    def test_the_bound_values_of_a_cardinality_rhs_parameter(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("r", ["s"], rows=[(("a",), 2**31 - 1), (("b",), -(2**31 - 1))], default=2.0)],
            variable_families=[X],
            cardinality_constraint_templates=[cardinality(rhs="r[i]")],
        )
        rhs = [c.rhs for c in expanded(data).problem.cardinality_constraints]
        assert rhs == [2**31 - 1, -(2**31 - 1), 2]
        assert all(type(value) is int for value in rhs)


class TestParameterValueMissing:
    """§14.4 / §14.7: a key used without a row and without a default."""

    def test_a_coefficient(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("w", ["s"], rows=[(("a",), 1)])],
            variable_families=[X],
            objective=objective(linear=[term(coefficient="w[i]")]),
        )
        assert failed(data) == [
            ("PARAMETER_VALUE_MISSING", "objective.linear_term_templates[0].coefficient", f"Parameter w has no value for ({key}) and no default")
            for key in ("b", "c")
        ]

    def test_a_where_lookup(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("f", ["s"], rows=[(("a",), 0)])],
            variable_families=[X],
            objective=objective(linear=[term(where=["f[i] == 1"])]),
        )
        assert failed(data) == [
            ("PARAMETER_VALUE_MISSING", "objective.linear_term_templates[0].where[0]", f"Parameter f has no value for ({key}) and no default")
            for key in ("b", "c")
        ]

    def test_a_two_dimensional_key(self):
        data = document(
            index_sets=[S, T],
            parameters=[parameter("d", ["s", "t"], rows=[(("a", 0), 1)])],
            variable_families=[{"name": "z", "indices": ["s", "t"]}],
            objective=objective(linear=[term(variable="z[i,p]", coefficient="d[i,p]", for_each=["i in s", "p in t"], where=["i == i"])]),
        )
        found = failed(data)
        assert len(found) == 8
        assert found[0] == (
            "PARAMETER_VALUE_MISSING",
            "objective.linear_term_templates[0].coefficient",
            "Parameter d has no value for (a,1) and no default",
        )

    def test_constraint_rhs_weight_and_coefficient(self):
        data = document(
            index_sets=[S],
            parameters=[
                parameter("r", ["s"], rows=[(("b",), 1), (("c",), 1)]),
                parameter("w", ["s"], rows=[(("a",), 1), (("c",), 1)]),
                parameter("k", ["s"], rows=[(("a",), 1), (("b",), 1)]),
            ],
            variable_families=[X],
            constraint_templates=[
                linear_constraint(
                    type="soft",
                    rhs="r[i]",
                    weight="w[i]",
                    terms=[{"coefficient": "k[i]", "variable": "x[i]"}],
                )
            ],
        )
        assert failed(data) == [
            ("PARAMETER_VALUE_MISSING", "constraint_templates[0].rhs", "Parameter r has no value for (a) and no default"),
            ("PARAMETER_VALUE_MISSING", "constraint_templates[0].weight", "Parameter w has no value for (b) and no default"),
            ("PARAMETER_VALUE_MISSING", "constraint_templates[0].terms[0].coefficient", "Parameter k has no value for (c) and no default"),
        ]

    def test_a_default_fills_the_gap(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("w", ["s"], rows=[(("a",), 5)], default=-2)],
            variable_families=[X],
            objective=objective(linear=[term(coefficient="w[i]")]),
        )
        assert linear_pairs(expanded(data)) == [("x[a]", 5.0), ("x[b]", -2.0), ("x[c]", -2.0)]

    def test_a_zero_value_is_a_value(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("w", ["s"], rows=[(("a",), 0), (("b",), 0), (("c",), 0)])],
            variable_families=[X],
            objective=objective(linear=[term(coefficient="w[i]")]),
        )
        assert linear_pairs(expanded(data)) == [("x[a]", 0.0), ("x[b]", 0.0), ("x[c]", 0.0)]


# ---------------------------------------------------------------------------
# DUPLICATE_TEMPLATE_NAME
# ---------------------------------------------------------------------------


class TestDuplicateTemplateName:
    """§14.5 / §14.10: one namespace per kind of name."""

    @pytest.mark.parametrize(
        "fields, path, name, first",
        [
            ({"index_sets": [S, {"name": "s", "elements": ["q"]}]}, "index_sets[1]", "s", "index_sets[0]"),
            ({"index_sets": [S], "parameters": [parameter("s", ["s"], default=1)]}, "parameters[0]", "s", "index_sets[0]"),
            ({"index_sets": [S], "variable_families": [{"name": "s", "indices": ["s"]}]}, "variable_families[0]", "s", "index_sets[0]"),
            (
                {"index_sets": [S], "parameters": [parameter("p", ["s"], default=1)], "variable_families": [{"name": "p", "indices": ["s"]}]},
                "variable_families[0]",
                "p",
                "parameters[0]",
            ),
            (
                {"index_sets": [S], "parameters": [parameter("p", ["s"], default=1), parameter("p", ["s"], default=2)]},
                "parameters[1]",
                "p",
                "parameters[0]",
            ),
        ],
        ids=["set set", "set parameter", "set family", "parameter family", "parameter parameter"],
    )
    def test_declared_names_share_one_namespace(self, fields, path, name, first):
        assert only_error(document(**fields)) == (
            "DUPLICATE_TEMPLATE_NAME",
            f"{path}.name",
            f"The name {name} is already declared at {first}; index sets, "
            "parameters and variable families need distinct names",
        )

    def test_a_family_named_like_an_explicit_variable(self):
        data = document(index_sets=[S], variables=[{"name": "x"}], variable_families=[X])
        assert only_error(data) == (
            "DUPLICATE_TEMPLATE_NAME",
            "variable_families[0].name",
            "Variable family x has the name of an explicit variable, which templates "
            "could then no longer reference; rename one of them",
        )

    def test_template_ids_within_one_list(self):
        data = document(index_sets=[S], variable_families=[X], constraint_templates=[linear_constraint(), linear_constraint()])
        assert only_error(data) == (
            "DUPLICATE_TEMPLATE_NAME",
            "constraint_templates[1].id",
            "Template id c is already used by constraint_templates[0]; constraint "
            "template ids must be unique",
        )

    def test_template_ids_across_the_two_lists(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            constraint_templates=[linear_constraint()],
            cardinality_constraint_templates=[cardinality(id="c")],
        )
        assert only_error(data) == (
            "DUPLICATE_TEMPLATE_NAME",
            "cardinality_constraint_templates[0].id",
            "Template id c is already used by constraint_templates[0]; constraint "
            "template ids must be unique",
        )

    @pytest.mark.parametrize(
        "explicit",
        [
            {"constraints": [{"id": "c", "type": "hard", "terms": [{"variable": "v", "coefficient": 1}], "operator": "<=", "rhs": 1}]},
            {"cardinality_constraints": [{"id": "c", "type": "hard", "variables": ["v"], "operator": "<=", "rhs": 1}]},
        ],
        ids=["linear", "cardinality"],
    )
    @pytest.mark.parametrize("list_name", ["constraint_templates", "cardinality_constraint_templates"])
    def test_a_template_id_equal_to_an_explicit_id(self, explicit, list_name):
        template = linear_constraint() if list_name == "constraint_templates" else cardinality(id="c")
        data = document(index_sets=[S], variables=[{"name": "v"}], variable_families=[X], **{list_name: [template]}, **explicit)
        assert only_error(data) == (
            "DUPLICATE_TEMPLATE_NAME",
            f"{list_name}[0].id",
            "Template id c is also the id of an explicit constraint; template ids "
            "share the constraint id namespace",
        )

    def test_an_index_bound_twice_in_one_for_each(self):
        assert only_error(_linear(term(for_each=["i in s", "i in s"]))) == (
            "DUPLICATE_TEMPLATE_NAME",
            "objective.linear_term_templates[0].for_each[1]",
            "Index i is already bound in this scope; use a different name",
        )

    def test_an_inner_scope_rebinding_an_outer_index(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            constraint_templates=[linear_constraint(terms=[{"for_each": ["i in s"], "coefficient": 1, "variable": "x[i]"}])],
        )
        assert only_error(data) == (
            "DUPLICATE_TEMPLATE_NAME",
            "constraint_templates[0].terms[0].for_each[0]",
            "Index i is already bound in this scope; use a different name",
        )

    @pytest.mark.parametrize(
        "index, declared",
        [("s", "index_sets[0]"), ("w", "parameters[0]"), ("x", "variable_families[0]")],
        ids=["set", "parameter", "family"],
    )
    def test_an_index_named_like_a_declaration(self, index, declared):
        data = _linear(term(variable=f"x[{index}]", for_each=[f"{index} in s"]), parameters=[parameter("w", ["s"], default=1)])
        assert only_error(data) == (
            "DUPLICATE_TEMPLATE_NAME",
            "objective.linear_term_templates[0].for_each[0]",
            f"Index {index} has the name of the declaration at {declared}; an "
            "index name must differ from every declared name",
        )


# ---------------------------------------------------------------------------
# Reused codes raised statically
# ---------------------------------------------------------------------------


class TestReusedCodes:
    """Codes of the ordinary validator, raised once per template (§14.10)."""

    def test_reserved_family_name(self):
        data = document(index_sets=[S], variable_families=[{"name": "__x", "indices": ["s"]}])
        assert only_error(data) == (
            "RESERVED_VARIABLE_NAME",
            "variable_families[0].name",
            "Variable family name __x is reserved: names starting with '__' are for "
            "internal variables",
        )

    @pytest.mark.parametrize(
        "family, code, message",
        [
            (
                {"lower_bound": 0, "upper_bound": 1},
                "BOUNDS_ON_BINARY",
                "Binary variable x declares bounds [0, 1]; binary variables are always 0/1",
            ),
            (
                {"type": "integer", "lower_bound": 0},
                "INTEGER_BOUNDS_MISSING",
                "Integer variable x needs both lower_bound and upper_bound, got "
                "lower_bound=0, upper_bound=None",
            ),
            (
                {"type": "integer", "lower_bound": 3, "upper_bound": 3},
                "INTEGER_BOUNDS_INVALID",
                "Integer variable x has upper_bound 3 <= lower_bound 3; a variable "
                "needs at least two values",
            ),
            (
                {"type": "integer", "lower_bound": 0, "upper_bound": 2**31},
                "INTEGER_RANGE_TOO_LARGE",
                "Integer variable x has bounds [0, 2147483648] outside the supported "
                "range of ±2147483647",
            ),
        ],
        ids=["bounds on binary", "missing", "invalid", "too large"],
    )
    def test_family_bounds_checked_once(self, family, code, message):
        data = document(
            index_sets=[S],
            variable_families=[{"name": "x", "indices": ["s"], **family}],
            objective=objective(linear=[term()]),
        )
        # Once, at the family: the template using it adds no knock-on error.
        assert only_error(data) == (code, "variable_families[0]", message)

    def test_unknown_explicit_variable(self):
        assert only_error(_linear(term(variable="nope", for_each=()))) == (
            "UNKNOWN_VARIABLE",
            "objective.linear_term_templates[0].variable",
            "Variable nope does not exist: a name without brackets must be an "
            "explicit variable declared in variables",
        )

    def test_an_explicit_variable_may_be_referenced(self):
        data = _linear(term(variable="v", for_each=()), variables=[{"name": "v"}])
        assert linear_pairs(expanded(data)) == [("v", 1.0)]

    def test_a_cardinality_member_of_an_integer_family(self):
        data = document(
            index_sets=[S],
            variable_families=[{"name": "n", "indices": ["s"], "type": "integer", "lower_bound": 0, "upper_bound": 3}],
            cardinality_constraint_templates=[cardinality(variables=["n[i]"])],
        )
        assert only_error(data) == (
            "CARDINALITY_VARIABLE_NOT_BINARY",
            "cardinality_constraint_templates[0].variables[0]",
            "Cardinality constraint template k counts n[i], which is an integer "
            "variable; only binary variables can be counted",
        )

    def test_a_cardinality_member_naming_an_integer_explicit_variable(self):
        data = document(
            index_sets=[S],
            variables=[{"name": "v", "type": "integer", "lower_bound": 0, "upper_bound": 2}],
            variable_families=[X],
            cardinality_constraint_templates=[cardinality(variables=["x[i]", "v"])],
        )
        assert only_error(data) == (
            "CARDINALITY_VARIABLE_NOT_BINARY",
            "cardinality_constraint_templates[0].variables[1]",
            "Cardinality constraint template k counts v, which is an integer "
            "variable; only binary variables can be counted",
        )

    @pytest.mark.parametrize("second", ["x[i]", " x[ i ] "], ids=["same text", "same reference"])
    def test_self_quadratic_term(self, second):
        data = document(
            index_sets=[S],
            variable_families=[X],
            objective=objective(quadratic=[pair("x[i]", second, for_each=["i in s"])]),
        )
        assert only_error(data) == (
            "SELF_QUADRATIC_TERM",
            "objective.quadratic_term_templates[0]",
            "Quadratic term template multiplies x[i] by itself; for binary variables "
            "x*x = x, use a linear term template instead",
        )

    def test_self_quadratic_term_of_an_explicit_binary_variable(self):
        data = document(variables=[{"name": "v"}], index_sets=[S], objective=objective(quadratic=[pair("v", "v")]))
        assert only_error(data) == (
            "SELF_QUADRATIC_TERM",
            "objective.quadratic_term_templates[0]",
            "Quadratic term template multiplies v by itself; for binary variables "
            "x*x = x, use a linear term template instead",
        )

    def test_an_integer_square_is_fine(self):
        data = document(
            index_sets=[S],
            variable_families=[{"name": "n", "indices": ["s"], "type": "integer", "lower_bound": 0, "upper_bound": 3}],
            objective=objective(quadratic=[pair("n[i]", "n[i]", for_each=["i in s"])]),
        )
        terms = expanded(data).problem.objective.quadratic_terms
        assert [(t.variable1, t.variable2) for t in terms] == [("n[a]", "n[a]"), ("n[b]", "n[b]"), ("n[c]", "n[c]")]

    def test_a_partly_equal_pair_is_left_to_the_validator(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            objective=objective(quadratic=[pair("x[i]", "x[j]", for_each=["i in s", "j in s"])]),
        )
        terms = expanded(data).problem.objective.quadratic_terms
        assert len(terms) == 9
        assert ("x[a]", "x[a]") in [(t.variable1, t.variable2) for t in terms]

    @pytest.mark.parametrize("list_name", ["constraint_templates", "cardinality_constraint_templates"])
    def test_hard_template_with_a_weight(self, list_name):
        template = linear_constraint(weight=2) if list_name == "constraint_templates" else cardinality(id="c", weight=2)
        data = document(index_sets=[S], variable_families=[X], **{list_name: [template]})
        assert only_error(data) == (
            "HARD_CONSTRAINT_HAS_WEIGHT",
            f"{list_name}[0].weight",
            "Hard constraint c must not carry a weight; hard penalty strength is "
            "chosen by the penalty strategy",
        )

    @pytest.mark.parametrize(
        "weight, shown", [(None, "None"), (0, "0.0"), (-1.5, "-1.5")], ids=["missing", "zero", "negative"]
    )
    @pytest.mark.parametrize("list_name", ["constraint_templates", "cardinality_constraint_templates"])
    def test_soft_template_without_a_positive_weight(self, list_name, weight, shown):
        overrides = {"type": "soft"}
        if weight is not None:
            overrides["weight"] = weight
        template = linear_constraint(**overrides) if list_name == "constraint_templates" else cardinality(id="c", **overrides)
        data = document(index_sets=[S], variable_families=[X], **{list_name: [template]})
        assert only_error(data) == (
            "SOFT_CONSTRAINT_MISSING_WEIGHT",
            f"{list_name}[0].weight",
            f"Soft constraint c requires a weight > 0, got {shown}",
        )

    def test_a_weight_parameter_is_checked_later(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("w", ["s"], default=-1)],
            variable_families=[X],
            constraint_templates=[linear_constraint(type="soft", weight="w[i]")],
        )
        assert [c.weight for c in expanded(data).problem.constraints] == [-1.0, -1.0, -1.0]

    def test_non_integer_literal_rhs(self):
        data = document(index_sets=[S], variable_families=[X], constraint_templates=[linear_constraint(rhs=1.5)])
        assert only_error(data) == (
            "NON_INTEGER_INEQUALITY",
            "constraint_templates[0].rhs",
            "Inequality constraint c requires integer coefficients and rhs for slack "
            "encoding, got 1.5",
        )

    def test_non_integer_literal_coefficient(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            constraint_templates=[linear_constraint(operator=">=", terms=[{"coefficient": 0.5, "variable": "x[i]"}])],
        )
        assert only_error(data) == (
            "NON_INTEGER_INEQUALITY",
            "constraint_templates[0].terms[0].coefficient",
            "Inequality constraint c requires integer coefficients and rhs for slack "
            "encoding, got 0.5",
        )

    def test_fractions_are_fine_in_an_equality(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            constraint_templates=[linear_constraint(operator="==", rhs=1.5, terms=[{"coefficient": 0.5, "variable": "x[i]"}])],
        )
        assert constraint_rows(expanded(data))[0] == ("c[a]", [("x[a]", 0.5)], "==", 1.5, None)


class TestNoKnockOnErrors:
    """A declaration with errors is reported once, not again at every use.

    §14.6 (implementation rule): one pass reports every error it can, and
    none of them as a consequence of another.
    """

    USE = objective(linear=[term(coefficient="w[i]")])

    def _problem(self, **fields) -> dict:
        data = {
            "index_sets": [{"name": "s", "elements": ["a", "b"]}],
            "parameters": [parameter("w", ["s"], default=1)],
            "variable_families": [X],
            "objective": self.USE,
        }
        data.update(fields)
        return document(**data)

    def test_an_index_set_with_bad_elements(self):
        data = self._problem(index_sets=[{"name": "s", "elements": ["a", "b c", "a"]}])
        assert failed(data) == [
            (
                "INDEX_SET_INVALID",
                "index_sets[0].elements[1]",
                'Element "b c" of index set "s" must be 1 to 64 letters, digits, '
                "underscores, dots or hyphens",
            ),
            (
                "INDEX_SET_INVALID",
                "index_sets[0].elements[2]",
                'Element "a" is listed more than once in index set "s"',
            ),
        ]

    def test_a_parameter_with_a_bad_row(self):
        data = self._problem(
            parameters=[parameter("w", ["s"], rows=[(("z",), 1)], default=1)]
        )
        assert only_error(data) == (
            "PARAMETER_TABLE_INVALID",
            "parameters[0].values[0].key[0]",
            '"z" is not an element of index set s, which position 0 of Parameter '
            "\"w\"'s key ranges over",
        )

    def test_a_parameter_with_a_bad_default(self):
        data = self._problem(parameters=[parameter("w", ["s"], default=float("inf"))])
        assert only_error(data) == (
            "PARAMETER_TABLE_INVALID",
            "parameters[0].default",
            'Parameter "w" has a default that is not a finite number',
        )

    def test_a_reserved_family_name(self):
        data = self._problem(
            variable_families=[{"name": "__x", "indices": ["s"]}],
            objective=objective(linear=[term(variable="__x[i]", coefficient="w[i]")]),
        )
        assert only_error(data)[0] == "RESERVED_VARIABLE_NAME"

    def test_a_family_named_like_an_explicit_variable(self):
        data = self._problem(variables=[{"name": "x"}])
        assert only_error(data)[0] == "DUPLICATE_TEMPLATE_NAME"

    def test_a_repeated_index_set_name(self):
        data = self._problem(
            index_sets=[{"name": "s", "elements": ["a", "b"]}, {"name": "s", "elements": ["q"]}]
        )
        assert only_error(data) == (
            "DUPLICATE_TEMPLATE_NAME",
            "index_sets[1].name",
            "The name s is already declared at index_sets[0]; index sets, "
            "parameters and variable families need distinct names",
        )


# ---------------------------------------------------------------------------
# Evaluation order (§14.7)
# ---------------------------------------------------------------------------


class TestEvaluationOrder:
    """Every branch of the §14.7 order."""

    def test_where_short_circuits(self):
        """g has no row for a, but f[a] == 1 is false first: no error."""
        data = document(
            index_sets=[S],
            parameters=[
                parameter("f", ["s"], rows=[(("a",), 0), (("b",), 1), (("c",), 1)]),
                parameter("g", ["s"], rows=[(("b",), 1), (("c",), 1)]),
            ],
            variable_families=[X],
            objective=objective(linear=[term(where=["f[i] == 1", "g[i] == 1"])]),
        )
        assert linear_pairs(expanded(data)) == [("x[b]", 1.0), ("x[c]", 1.0)]

    def test_where_missing_value_is_reported_in_order(self):
        data = document(
            index_sets=[S],
            parameters=[
                parameter("f", ["s"], rows=[(("a",), 1), (("b",), 1), (("c",), 1)]),
                parameter("g", ["s"], rows=[(("b",), 1)]),
            ],
            variable_families=[X],
            objective=objective(linear=[term(where=["f[i] == 1", "g[i] == 1"])]),
        )
        assert failed(data) == [
            ("PARAMETER_VALUE_MISSING", "objective.linear_term_templates[0].where[1]", f"Parameter g has no value for ({key}) and no default")
            for key in ("a", "c")
        ]

    def test_where_missing_value_before_the_bounds_check(self):
        """The constraint for p=2 would be skipped (y[3]), but where runs first."""
        data = document(
            index_sets=[T],
            parameters=[parameter("g", ["t"], rows=[((0,), 1), ((1,), 1)])],
            variable_families=[Y],
            constraint_templates=[
                linear_constraint(
                    for_each=["p in t"],
                    where=["g[p] == 1"],
                    terms=[{"coefficient": 1, "variable": "y[p]"}, {"coefficient": 1, "variable": "y[p+1]"}],
                )
            ],
        )
        assert only_error(data) == (
            "PARAMETER_VALUE_MISSING",
            "constraint_templates[0].where[0]",
            "Parameter g has no value for (2) and no default",
        )

    def test_where_missing_value_in_a_member_before_its_bounds_check(self):
        data = document(
            index_sets=[T],
            parameters=[parameter("g", ["t"], rows=[((0,), 1), ((1,), 1)])],
            variable_families=[Y],
            constraint_templates=[
                linear_constraint(
                    for_each=[],
                    terms=[{"for_each": ["q in t"], "where": ["g[q] == 1"], "coefficient": 1, "variable": "y[q+1]"}],
                )
            ],
        )
        assert only_error(data) == (
            "PARAMETER_VALUE_MISSING",
            "constraint_templates[0].terms[0].where[0]",
            "Parameter g has no value for (2) and no default",
        )

    def test_an_objective_term_out_of_bounds_is_left_out_alone(self):
        data = document(
            index_sets=[T],
            variable_families=[Y],
            objective=objective(linear=[term(variable="y[p+1]", for_each=["p in t"])]),
        )
        result = expanded(data)
        assert linear_pairs(result) == [("y[1]", 1.0), ("y[2]", 1.0)]
        assert triples(result.warnings)[-1] == (
            "TEMPLATE_BOUNDARY_SKIPPED",
            "objective.linear_term_templates[0]",
            "objective.linear_term_templates[0] left out 1 term whose shifted index "
            "runs past the end of a linear index set",
        )

    def test_an_out_of_bounds_coefficient_leaves_the_term_out(self):
        data = document(
            index_sets=[T],
            parameters=[parameter("w", ["t"], rows=[((0,), 5), ((1,), 6), ((2,), 7)])],
            variable_families=[Y],
            objective=objective(linear=[term(variable="y[p]", coefficient="w[p-1]", for_each=["p in t"])]),
        )
        assert linear_pairs(expanded(data)) == [("y[1]", 5.0), ("y[2]", 6.0)]

    def test_an_objective_term_out_of_bounds_skips_its_lookup(self):
        """w has no row for p=2, but y[3] leaves that term out before lookup."""
        data = document(
            index_sets=[T],
            parameters=[parameter("w", ["t"], rows=[((0,), 5), ((1,), 6)])],
            variable_families=[Y],
            objective=objective(linear=[term(variable="y[p+1]", coefficient="w[p]", for_each=["p in t"])]),
        )
        assert linear_pairs(expanded(data)) == [("y[1]", 5.0), ("y[2]", 6.0)]

    def test_a_constraint_out_of_bounds_is_left_out_whole_without_lookups(self):
        """rhs, weight and coefficient all lack p=2; y[3] skips it first."""
        rows = [((0,), 2), ((1,), 3)]
        data = document(
            index_sets=[T],
            parameters=[parameter("r", ["t"], rows=rows), parameter("k", ["t"], rows=rows), parameter("w", ["t"], rows=rows)],
            variable_families=[Y],
            constraint_templates=[
                linear_constraint(
                    type="soft",
                    weight="w[p]",
                    for_each=["p in t"],
                    terms=[{"coefficient": "k[p]", "variable": "y[p]"}, {"coefficient": 1, "variable": "y[p+1]"}],
                    rhs="r[p]",
                )
            ],
        )
        result = expanded(data)
        assert constraint_rows(result) == [
            ("c[0]", [("y[0]", 2.0), ("y[1]", 1.0)], "<=", 2.0, 2.0),
            ("c[1]", [("y[1]", 3.0), ("y[2]", 1.0)], "<=", 3.0, 3.0),
        ]
        assert triples(result.warnings) == [
            (
                "TEMPLATE_BOUNDARY_SKIPPED",
                "constraint_templates[0]",
                "constraint_templates[0] left out 1 constraint whose shifted index "
                "runs past the end of a linear index set",
            )
        ]

    def test_a_member_of_a_later_member_template_out_of_bounds(self):
        """The first member collected fine; the second one skips it all."""
        data = document(
            index_sets=[T],
            variable_families=[Y],
            cardinality_constraint_templates=[
                cardinality(for_each=["p in t"], variables=["y[p]", "y[p+1]"])
            ],
        )
        result = expanded(data)
        assert [(c.id, c.variables) for c in result.problem.cardinality_constraints] == [
            ("k[0]", ["y[0]", "y[1]"]),
            ("k[1]", ["y[1]", "y[2]"]),
        ]
        assert warning_codes(result) == ["TEMPLATE_BOUNDARY_SKIPPED"]

    def test_an_out_of_bounds_rhs_leaves_the_constraint_out(self):
        data = document(
            index_sets=[T],
            parameters=[parameter("r", ["t"], default=1)],
            variable_families=[Y],
            constraint_templates=[linear_constraint(for_each=["p in t"], terms=[{"coefficient": 1, "variable": "y[p]"}], rhs="r[p+1]")],
        )
        result = expanded(data)
        assert [c.id for c in result.problem.constraints] == ["c[0]", "c[1]"]
        assert "TEMPLATE_BOUNDARY_SKIPPED" in warning_codes(result)

    def test_bounds_are_only_judged_for_bindings_that_pass_where(self):
        data = document(
            index_sets=[T],
            parameters=[parameter("last", ["t"], rows=[((0,), 0), ((1,), 0), ((2,), 1)])],
            variable_families=[Y],
            constraint_templates=[
                linear_constraint(
                    for_each=[],
                    terms=[{"for_each": ["q in t"], "where": ["last[q] == 0"], "coefficient": 1, "variable": "y[q+1]"}],
                )
            ],
        )
        result = expanded(data)
        assert constraint_rows(result) == [("c", [("y[1]", 1.0), ("y[2]", 1.0)], "<=", 1.0, None)]
        assert "TEMPLATE_BOUNDARY_SKIPPED" not in warning_codes(result)

    def test_without_the_where_the_whole_constraint_is_left_out(self):
        data = document(
            index_sets=[T],
            variable_families=[Y],
            constraint_templates=[
                linear_constraint(for_each=[], terms=[{"for_each": ["q in t"], "coefficient": 1, "variable": "y[q+1]"}])
            ],
        )
        result = expanded(data)
        assert result.problem.constraints == []
        assert warning_codes(result) == [
            "UNUSED_TEMPLATE_VARIABLES",
            "TEMPLATE_BOUNDARY_SKIPPED",
            "EMPTY_TEMPLATE_EXPANSION",
        ]

    def test_cyclic_sets_wrap_around(self):
        cyclic = {"name": "c", "elements": [0, 1, 2], "order": "cyclic"}
        data = document(
            index_sets=[cyclic],
            variable_families=[{"name": "y", "indices": ["c"]}],
            objective=objective(
                linear=[
                    term(variable="y[p+1]", coefficient=1, for_each=["p in c"]),
                    term(variable="y[p-1]", coefficient=2, for_each=["p in c"]),
                    term(variable="y[p+4]", coefficient=3, for_each=["p in c"]),
                    term(variable="y[p-7]", coefficient=4, for_each=["p in c"]),
                ]
            ),
        )
        result = expanded(data)
        assert linear_pairs(result) == [
            ("y[1]", 1.0), ("y[2]", 1.0), ("y[0]", 1.0),
            ("y[2]", 2.0), ("y[0]", 2.0), ("y[1]", 2.0),
            ("y[1]", 3.0), ("y[2]", 3.0), ("y[0]", 3.0),
            ("y[2]", 4.0), ("y[0]", 4.0), ("y[1]", 4.0),
        ]
        assert result.warnings == ()

    def test_a_shift_moves_by_position_not_by_value(self):
        data = document(
            index_sets=[{"name": "t", "elements": [0, 10, 20], "order": "linear"}],
            variable_families=[Y],
            objective=objective(linear=[term(variable="y[p+1]", for_each=["p in t"])]),
        )
        assert linear_pairs(expanded(data)) == [("y[10]", 1.0), ("y[20]", 1.0)]

    def test_a_negative_shift_on_an_unsorted_integer_set(self):
        data = document(
            index_sets=[{"name": "t", "elements": [7, -3, 5], "order": "linear"}],
            variable_families=[Y],
            objective=objective(linear=[term(variable="y[p-1]", for_each=["p in t"])]),
        )
        assert linear_pairs(expanded(data)) == [("y[7]", 1.0), ("y[-3]", 1.0)]

    @pytest.mark.parametrize(
        "operator, expected",
        [
            ("<", [("20", "10"), ("20", "0"), ("10", "0")]),
            ("<=", [("20", "20"), ("20", "10"), ("20", "0"), ("10", "10"), ("10", "0"), ("0", "0")]),
            (">", [("10", "20"), ("0", "20"), ("0", "10")]),
            ("==", [("20", "20"), ("10", "10"), ("0", "0")]),
            ("!=", [("20", "10"), ("20", "0"), ("10", "20"), ("10", "0"), ("0", "20"), ("0", "10")]),
        ],
    )
    def test_index_comparisons_use_positions(self, operator, expected):
        """``i < j`` on [20, 10, 0] follows the list, not the numbers."""
        data = document(
            index_sets=[{"name": "t", "elements": [20, 10, 0]}],
            variable_families=[Y],
            objective=objective(linear=[term(variable="y[i]", coefficient="d[i,j]", for_each=["i in t", "j in t"], where=[f"i {operator} j"])]),
            parameters=[parameter("d", ["t", "t"], default=1)],
        )
        found = [t.variable for t in expanded(data).problem.objective.linear_terms]
        assert found == [f"y[{i}]" for i, _ in expected]

    def test_string_elements_compare_by_position(self):
        data = document(
            index_sets=[{"name": "s", "elements": ["c", "a", "b"]}],
            variable_families=[X],
            objective=objective(quadratic=[pair("x[i]", "x[j]", for_each=["i in s", "j in s"], where=["i < j"])]),
        )
        terms = expanded(data).problem.objective.quadratic_terms
        assert [(t.variable1, t.variable2) for t in terms] == [("x[c]", "x[a]"), ("x[c]", "x[b]"), ("x[a]", "x[b]")]

    def test_numbers_compare_exactly(self):
        """No solution tolerance: 1.000000000001 is not 1 (§14.6)."""
        data = document(
            index_sets=[S],
            parameters=[parameter("v", ["s"], rows=[(("a",), 1), (("b",), 1.000000000001), (("c",), 0.9999999999999)])],
            variable_families=[X],
            objective=objective(linear=[term(where=["v[i] == 1"])]),
        )
        assert linear_pairs(expanded(data)) == [("x[a]", 1.0)]

    @pytest.mark.parametrize(
        "condition, kept",
        [
            ("v[i] < 2", ["a"]),
            ("v[i] <= 2", ["a", "b"]),
            ("v[i] > 2", ["c"]),
            ("v[i] >= 2", ["b", "c"]),
            ("v[i] != 2", ["a", "c"]),
            ("2 == v[i]", ["b"]),
            ("v[i] < w[i]", ["a"]),
        ],
    )
    def test_numeric_comparisons(self, condition, kept):
        data = document(
            index_sets=[S],
            parameters=[
                parameter("v", ["s"], rows=[(("a",), 1), (("b",), 2), (("c",), 3)]),
                parameter("w", ["s"], default=2),
            ],
            variable_families=[X],
            objective=objective(linear=[term(where=[condition])]),
        )
        assert [v for v, _ in linear_pairs(expanded(data))] == [f"x[{k}]" for k in kept]


# ---------------------------------------------------------------------------
# Names and order
# ---------------------------------------------------------------------------


class TestNames:
    """§14.7: generated names and ids, and the order of the lists."""

    def test_family_names_have_no_spaces_and_follow_the_indices(self):
        data = document(
            index_sets=[{"name": "t", "elements": [-1, 0, 5]}, S],
            variable_families=[{"name": "x", "indices": ["s", "t"]}],
            objective=objective(linear=[term(variable="x[ i , p ]", for_each=["i in s", "p in t"])]),
        )
        assert names(expanded(data)) == [
            "x[a,-1]", "x[a,0]", "x[a,5]",
            "x[b,-1]", "x[b,0]", "x[b,5]",
            "x[c,-1]", "x[c,0]", "x[c,5]",
        ]

    def test_the_first_for_each_item_is_outermost(self):
        data = document(
            index_sets=[S, T],
            variable_families=[{"name": "z", "indices": ["s", "t"]}],
            objective=objective(linear=[term(variable="z[i,p]", for_each=["p in t", "i in s"])]),
        )
        assert [v for v, _ in linear_pairs(expanded(data))] == [
            "z[a,0]", "z[b,0]", "z[c,0]",
            "z[a,1]", "z[b,1]", "z[c,1]",
            "z[a,2]", "z[b,2]", "z[c,2]",
        ]

    def test_generated_ids(self):
        data = document(
            index_sets=[T, S],
            variable_families=[{"name": "z", "indices": ["s", "t"]}],
            constraint_templates=[
                linear_constraint(for_each=["p in t", "i in s"], terms=[{"coefficient": 1, "variable": "z[i,p]"}]),
                linear_constraint(id="d", for_each=[], terms=[{"for_each": ["i in s", "p in t"], "coefficient": 1, "variable": "z[i,p]"}]),
            ],
            cardinality_constraint_templates=[cardinality(id="e", for_each=["i in s"], variables=[{"for_each": ["p in t"], "variable": "z[i,p]"}])],
        )
        problem = expanded(data).problem
        assert [c.id for c in problem.constraints] == [
            "c[0,a]", "c[0,b]", "c[0,c]",
            "c[1,a]", "c[1,b]", "c[1,c]",
            "c[2,a]", "c[2,b]", "c[2,c]",
            "d",
        ]
        assert [c.id for c in problem.cardinality_constraints] == ["e[a]", "e[b]", "e[c]"]

    def test_explicit_entries_come_first_then_templates_in_order(self):
        data = document(
            variables=[{"name": "v"}],
            index_sets=[S],
            variable_families=[X, {"name": "w", "indices": ["s"]}],
            objective=objective(
                terms=[{"variable": "v", "coefficient": 9}],
                linear=[term(variable="w[i]", coefficient=2), term(variable="x[i]", coefficient=3)],
            ),
            constraints=[{"id": "e", "type": "hard", "terms": [{"variable": "v", "coefficient": 1}], "operator": "<=", "rhs": 1}],
            constraint_templates=[linear_constraint(id="c")],
            cardinality_constraints=[{"id": "ke", "type": "hard", "variables": ["v"], "operator": "<=", "rhs": 1}],
            cardinality_constraint_templates=[cardinality(id="k", variables=["w[i]"])],
        )
        result = expanded(data)
        assert names(result) == ["v", "x[a]", "x[b]", "x[c]", "w[a]", "w[b]", "w[c]"]
        assert linear_pairs(result) == [
            ("v", 9.0),
            ("w[a]", 2.0), ("w[b]", 2.0), ("w[c]", 2.0),
            ("x[a]", 3.0), ("x[b]", 3.0), ("x[c]", 3.0),
        ]
        assert [c.id for c in result.problem.constraints] == ["e", "c[a]", "c[b]", "c[c]"]
        assert [c.id for c in result.problem.cardinality_constraints] == ["ke", "k[a]", "k[b]", "k[c]"]

    def test_members_follow_member_templates_then_their_bindings(self):
        data = document(
            index_sets=[S, T],
            variable_families=[X, Y],
            constraint_templates=[
                linear_constraint(
                    for_each=[],
                    terms=[
                        {"for_each": ["p in t"], "coefficient": 2, "variable": "y[p]"},
                        {"for_each": ["i in s"], "coefficient": 1, "variable": "x[i]"},
                    ],
                )
            ],
        )
        assert [t.variable for t in expanded(data).problem.constraints[0].terms] == [
            "y[0]", "y[1]", "y[2]", "x[a]", "x[b]", "x[c]",
        ]

    def test_the_expanded_problem_keeps_everything_else(self):
        data = document(
            name="kept",
            description="d",
            index_sets=[S],
            variable_families=[X],
            objective=objective(linear=[term()]),
            solver={"backend": "exact", "top_k": 3},
        )
        data["objective"]["direction"] = "maximize"
        data["objective"]["constant"] = 4
        source = OptimizationProblem.model_validate(data)
        result = expand_problem(source)
        problem = result.problem
        assert result.source is source
        assert (problem.version, problem.name, problem.description) == ("1.3", "kept", "d")
        assert problem.solver == source.solver
        assert (problem.objective.direction, problem.objective.constant) == ("maximize", 4.0)
        assert problem.template_fields() == []
        assert not problem.has_templates()
        assert "version" in problem.model_fields_set
        assert source.has_templates()  # the source itself is untouched


_STRING_POOL = ["a", "b", "a.b", "a-b", "a_b", "-1", "1", "01", "1.0", "1e3", "A", "_", ".", "-", "a..b", "0", "x", "10"]
_INTEGER_POOL = [-10, -1, 0, 1, 10, 100, 2**31 - 1, -(2**31 - 1), 11, 101]


class TestNamesAreInjective:
    """§14.7 / §14.16 item 5: different (family, elements) never share a name."""

    @pytest.mark.parametrize("seed", range(12))
    def test_random_sets_and_families(self, seed):
        rng = random.Random(20260928 + seed)
        sets = []
        for k in range(rng.randint(1, 3)):
            pool = _STRING_POOL if rng.random() < 0.6 else _INTEGER_POOL
            sets.append({"name": f"s{k}", "elements": rng.sample(pool, rng.randint(1, 5))})
        families = []
        templates = []
        constraint_templates = []
        for f in range(rng.randint(1, 3)):
            indices = [rng.choice(sets)["name"] for _ in range(rng.randint(1, 3))]
            name = rng.choice(["x", "x_", "X", "x1", "xx"]) + str(f)
            families.append({"name": name, "indices": indices})
            loops = [f"i{d} in {set_name}" for d, set_name in enumerate(indices)]
            reference = f"{name}[" + ",".join(f"i{d}" for d in range(len(indices))) + "]"
            templates.append(term(variable=reference, for_each=loops))
            constraint_templates.append(
                linear_constraint(id=f"c{name}", for_each=loops, terms=[{"coefficient": 1, "variable": reference}])
            )
        data = document(
            index_sets=sets,
            variable_families=families,
            objective=objective(linear=templates),
            constraint_templates=constraint_templates,
        )
        problem = expanded(data).problem
        by_name = {s["name"]: s["elements"] for s in sets}

        expected_names = []
        for family in families:
            for combo in itertools.product(*(by_name[i] for i in family["indices"])):
                expected_names.append((family["name"], combo))
        generated = [variable.name for variable in problem.variables]
        assert len(generated) == len(expected_names)
        assert len(set(generated)) == len(generated)
        # Each name reads back to exactly its (family, elements).
        for name, (family, combo) in zip(generated, expected_names):
            head, _, inner = name.partition("[")
            assert head == family
            assert inner.endswith("]")
            assert inner[:-1].split(",") == [str(element) for element in combo]
        ids = [constraint.id for constraint in problem.constraints]
        assert len(set(ids)) == len(ids) == len(expected_names)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    """The expansion's own advisories (§14.7, §14.10, §14.14 item 2)."""

    def test_every_warning_carries_its_recommended_action(self):
        data = document(
            index_sets=[S, T],
            parameters=[parameter("f", ["s"], default=0)],
            variable_families=[X, Y],
            objective=objective(linear=[term(where=["f[i] == 1"]), term(variable="y[p+1]", for_each=["p in t"])]),
            constraint_templates=[linear_constraint(terms=[{"coefficient": 1, "variable": "x[i]"}, {"coefficient": 1, "variable": "x[i]"}])],
        )
        result = expanded(data)
        assert set(warning_codes(result)) == {
            "UNUSED_TEMPLATE_VARIABLES",
            "EMPTY_TEMPLATE_EXPANSION",
            "TEMPLATE_BOUNDARY_SKIPPED",
            "TEMPLATE_TERMS_MERGED",
        }
        for warning in result.warnings:
            assert warning.recommended_action == _WARNING_RECOMMENDED_ACTIONS[warning.code]
            assert warning.retryable is False

    def test_empty_template_expansion(self):
        data = document(
            index_sets=[S],
            parameters=[parameter("f", ["s"], default=0)],
            variable_families=[X],
            objective=objective(linear=[term(where=["f[i] == 1"])]),
            constraints=[{"id": "e", "type": "hard", "terms": [{"variable": "x[a]", "coefficient": 1}], "operator": "<=", "rhs": 1}],
            cardinality_constraint_templates=[cardinality(where=["f[i] == 1"])],
        )
        result = expanded(data)
        assert triples(result.warnings) == [
            (
                "UNUSED_TEMPLATE_VARIABLES",
                "variable_families[0]",
                "2 of the 3 variables of family x appear in no objective term or "
                "constraint and were left out: x[b], x[c]",
            ),
            ("EMPTY_TEMPLATE_EXPANSION", "objective.linear_term_templates[0]", "objective.linear_term_templates[0] generated no terms"),
            ("EMPTY_TEMPLATE_EXPANSION", "cardinality_constraint_templates[0]", "cardinality_constraint_templates[0] generated no constraints"),
        ]
        assert result.problem.objective.linear_terms == []
        assert result.problem.cardinality_constraints == []

    def test_a_constraint_whose_members_are_all_filtered_is_still_generated(self):
        """An empty ``== 1`` must reach the validator (EMPTY_CONSTRAINT), §14.7."""
        data = document(
            index_sets=[S],
            parameters=[parameter("f", ["s"], default=0)],
            variable_families=[X],
            cardinality_constraint_templates=[
                cardinality(for_each=[], operator="==", variables=[{"for_each": ["i in s"], "where": ["f[i] == 1"], "variable": "x[i]"}])
            ],
        )
        result = expanded(data)
        assert [(c.id, c.variables) for c in result.problem.cardinality_constraints] == [("k", [])]
        assert "EMPTY_TEMPLATE_EXPANSION" not in warning_codes(result)

    def test_boundary_skipped_counts(self):
        data = document(
            index_sets=[T, {"name": "u", "elements": [0, 1, 2, 3], "order": "linear"}],
            variable_families=[Y, {"name": "z", "indices": ["u"]}],
            objective=objective(linear=[term(variable="z[q+2]", for_each=["q in u"])]),
            constraint_templates=[
                linear_constraint(for_each=["p in t"], terms=[{"coefficient": 1, "variable": "y[p-2]"}]),
            ],
        )
        result = expanded(data)
        assert [w for w in triples(result.warnings) if w[0] == "TEMPLATE_BOUNDARY_SKIPPED"] == [
            (
                "TEMPLATE_BOUNDARY_SKIPPED",
                "objective.linear_term_templates[0]",
                "objective.linear_term_templates[0] left out 2 terms whose shifted "
                "index runs past the end of a linear index set",
            ),
            (
                "TEMPLATE_BOUNDARY_SKIPPED",
                "constraint_templates[0]",
                "constraint_templates[0] left out 2 constraints whose shifted index "
                "runs past the end of a linear index set",
            ),
        ]

    def test_terms_merged_keeps_every_term(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            constraint_templates=[
                linear_constraint(terms=[{"coefficient": 1, "variable": "x[i]"}, {"coefficient": 2, "variable": "x[i]"}])
            ],
        )
        result = expanded(data)
        assert constraint_rows(result) == [
            (f"c[{e}]", [(f"x[{e}]", 1.0), (f"x[{e}]", 2.0)], "<=", 1.0, None) for e in "abc"
        ]
        assert triples(result.warnings) == [
            (
                "TEMPLATE_TERMS_MERGED",
                "constraint_templates[0]",
                "3 constraints generated by constraint_templates[0] name the same "
                "variable more than once, and the compiler sums its coefficients; "
                "for example c[a] names x[a] more than once",
            )
        ]

    def test_terms_merged_by_a_cyclic_shift(self):
        data = document(
            index_sets=[{"name": "c", "elements": [0, 1], "order": "cyclic"}],
            variable_families=[{"name": "y", "indices": ["c"]}],
            constraint_templates=[
                linear_constraint(
                    id="ring",
                    for_each=["p in c"],
                    operator="==",
                    rhs=0,
                    terms=[{"coefficient": 1, "variable": "y[p]"}, {"coefficient": -1, "variable": "y[p+2]"}],
                )
            ],
        )
        result = expanded(data)
        assert constraint_rows(result)[0] == ("ring[0]", [("y[0]", 1.0), ("y[0]", -1.0)], "==", 0.0, None)
        assert warning_codes(result) == ["TEMPLATE_TERMS_MERGED"]
        assert "for example ring[0] names y[0] more than once" in result.warnings[0].message

    def test_one_merged_constraint_is_singular(self):
        data = document(
            index_sets=[{"name": "c", "elements": [0], "order": "cyclic"}],
            variable_families=[{"name": "y", "indices": ["c"]}],
            constraint_templates=[
                linear_constraint(
                    for_each=["p in c"],
                    operator="==",
                    rhs=0,
                    terms=[{"coefficient": 1, "variable": "y[p]"}, {"coefficient": -1, "variable": "y[p+1]"}],
                )
            ],
        )
        assert triples(expanded(data).warnings) == [
            (
                "TEMPLATE_TERMS_MERGED",
                "constraint_templates[0]",
                "1 constraint generated by constraint_templates[0] names the same "
                "variable more than once, and the compiler sums its coefficients; "
                "for example c[0] names y[0] more than once",
            )
        ]

    def test_terms_merged_is_not_raised_for_explicit_constraints(self):
        data = document(
            variables=[{"name": "v"}],
            index_sets=[S],
            variable_families=[X],
            objective=objective(linear=[term()]),
            constraints=[{"id": "e", "type": "hard", "terms": [{"variable": "v", "coefficient": 1}, {"variable": "v", "coefficient": 1}], "operator": "<=", "rhs": 1}],
        )
        assert expanded(data).warnings == ()

    def test_terms_merged_is_not_raised_for_cardinality_templates(self):
        data = document(
            index_sets=[S],
            variable_families=[X],
            cardinality_constraint_templates=[cardinality(variables=["x[i]", "x[i]"])],
        )
        result = expanded(data)
        assert result.warnings == ()
        assert result.problem.cardinality_constraints[0].variables == ["x[a]", "x[a]"]

    def test_unused_variables_are_left_out(self):
        data = document(
            index_sets=[S, T],
            parameters=[parameter("keep", ["t"], rows=[((0,), 1), ((1,), 1), ((2,), 0)])],
            variable_families=[X, {"name": "z", "indices": ["s", "t"]}],
            objective=objective(
                linear=[term(variable="z[i,p]", for_each=["i in s", "p in t"], where=["keep[p] == 1"])]
            ),
        )
        result = expanded(data)
        assert names(result) == ["z[a,0]", "z[a,1]", "z[b,0]", "z[b,1]", "z[c,0]", "z[c,1]"]
        assert triples(result.warnings) == [
            (
                "UNUSED_TEMPLATE_VARIABLES",
                "variable_families[0]",
                "3 of the 3 variables of family x appear in no objective term or "
                "constraint and were left out: x[a], x[b], x[c]",
            ),
            (
                "UNUSED_TEMPLATE_VARIABLES",
                "variable_families[1]",
                "3 of the 9 variables of family z appear in no objective term or "
                "constraint and were left out: z[a,2], z[b,2], z[c,2]",
            ),
        ]

    def test_one_unused_variable_is_singular(self):
        data = document(
            index_sets=[T],
            variable_families=[Y],
            objective=objective(linear=[term(variable="y[p+1]", for_each=["p in t"])]),
        )
        assert triples(expanded(data).warnings)[0] == (
            "UNUSED_TEMPLATE_VARIABLES",
            "variable_families[0]",
            "1 of the 3 variables of family y appears in no objective term or "
            "constraint and was left out: y[0]",
        )

    def test_at_most_five_names_are_listed(self):
        data = document(
            variables=[{"name": "v"}],
            index_sets=[{"name": "s", "elements": list(range(8))}],
            variable_families=[X],
            objective=objective(terms=[{"variable": "v", "coefficient": 1}]),
        )
        result = expanded(data)
        assert names(result) == ["v"]
        assert triples(result.warnings) == [
            (
                "UNUSED_TEMPLATE_VARIABLES",
                "variable_families[0]",
                "8 of the 8 variables of family x appear in no objective term or "
                "constraint and were left out: x[0], x[1], x[2], x[3], x[4] and 3 more",
            )
        ]

    def test_exactly_five_names_need_no_more(self):
        data = document(
            variables=[{"name": "v"}],
            index_sets=[{"name": "s", "elements": list(range(5))}],
            variable_families=[X],
            objective=objective(terms=[{"variable": "v", "coefficient": 1}]),
        )
        assert expanded(data).warnings[0].message.endswith("left out: x[0], x[1], x[2], x[3], x[4]")

    def test_a_variable_named_like_an_explicit_one_is_kept(self):
        """Kept so the validator still reports DUPLICATE_VARIABLE (§14.7)."""
        data = document(index_sets=[S], variables=[{"name": "x[a]"}], variable_families=[X])
        result = expanded(data)
        assert names(result) == ["x[a]", "x[a]"]
        assert triples(result.warnings) == [
            (
                "UNUSED_TEMPLATE_VARIABLES",
                "variable_families[0]",
                "2 of the 3 variables of family x appear in no objective term or "
                "constraint and were left out: x[b], x[c]",
            )
        ]

    def test_variables_used_anywhere_are_kept(self):
        data = document(
            index_sets=[S],
            variable_families=[
                {"name": "a", "indices": ["s"]},
                {"name": "b", "indices": ["s"]},
                {"name": "c", "indices": ["s"]},
                {"name": "d", "indices": ["s"]},
            ],
            objective=objective(
                linear=[term(variable="a[i]")],
                quadratic=[pair("b[i]", "b[j]", for_each=["i in s", "j in s"], where=["i < j"])],
            ),
            constraint_templates=[linear_constraint(terms=[{"coefficient": 1, "variable": "c[i]"}])],
            cardinality_constraint_templates=[cardinality(variables=["d[i]"])],
        )
        result = expanded(data)
        assert len(result.problem.variables) == 12
        assert result.warnings == ()

    def test_warning_order(self):
        """Families first, then templates in order; per template: boundary, merged, empty."""
        data = document(
            index_sets=[S, T],
            parameters=[parameter("f", ["t"], default=0)],
            variable_families=[X, Y],
            objective=objective(linear=[term(variable="y[p+1]", for_each=["p in t"], where=["f[p] == 1"])]),
            constraint_templates=[
                linear_constraint(
                    id="c",
                    for_each=["p in t"],
                    terms=[{"coefficient": 1, "variable": "y[p]"}, {"coefficient": 1, "variable": "y[p]"}, {"coefficient": 1, "variable": "y[p+1]"}],
                )
            ],
        )
        result = expanded(data)
        assert [(w.code, w.path) for w in result.warnings] == [
            ("UNUSED_TEMPLATE_VARIABLES", "variable_families[0]"),
            ("EMPTY_TEMPLATE_EXPANSION", "objective.linear_term_templates[0]"),
            ("TEMPLATE_BOUNDARY_SKIPPED", "constraint_templates[0]"),
            ("TEMPLATE_TERMS_MERGED", "constraint_templates[0]"),
        ]


# ---------------------------------------------------------------------------
# Field types of generated items (§14.8)
# ---------------------------------------------------------------------------


def _assert_same_types(built, validated, where: str) -> None:
    assert type(built) is type(validated), (where, type(built), type(validated))
    if isinstance(built, BaseModel):
        for name in type(built).model_fields:
            _assert_same_types(getattr(built, name), getattr(validated, name), f"{where}.{name}")
    elif isinstance(built, list):
        assert len(built) == len(validated), where
        for k, (a, b) in enumerate(zip(built, validated)):
            _assert_same_types(a, b, f"{where}[{k}]")


def _typed_document() -> dict:
    return document(
        variables=[{"name": "v"}],
        index_sets=[S, T],
        parameters=[
            parameter("w", ["s"], rows=[(("a",), 2)], default=3),
            parameter("r", ["s"], default=1),
            parameter("big", ["t"], default=2.0),
        ],
        variable_families=[
            {"name": "x", "indices": ["s"], "description": "chosen"},
            {"name": "n", "indices": ["t"], "type": "integer", "lower_bound": -1, "upper_bound": 4.0},
        ],
        objective=objective(
            terms=[{"variable": "v", "coefficient": 1}],
            linear=[term(coefficient="w[i]"), term(variable="n[p]", coefficient=2, for_each=["p in t"])],
            quadratic=[
                pair("x[i]", "x[j]", coefficient=3, for_each=["i in s", "j in s"], where=["i < j"]),
                pair("x[i]", "n[p]", coefficient="w[i]", for_each=["i in s", "p in t"]),
            ],
        ),
        constraint_templates=[
            linear_constraint(id="hard_int", rhs=1, terms=[{"coefficient": 1, "variable": "x[i]"}, {"coefficient": "r[i]", "variable": "v"}]),
            linear_constraint(id="soft_param", type="soft", operator="==", rhs="w[i]", weight="r[i]", description="d"),
            linear_constraint(id="ints", for_each=["p in t"], terms=[{"coefficient": 2, "variable": "n[p]"}], rhs="big[p]"),
        ],
        cardinality_constraint_templates=[
            cardinality(id="hard_card", for_each=[], variables=[{"for_each": ["i in s"], "variable": "x[i]"}, "v"], operator="==", rhs=1),
            cardinality(id="soft_card", type="soft", weight=2, rhs="r[i]", description="k"),
            cardinality(id="float_rhs", rhs=1.0, operator=">="),
        ],
    )


class TestGeneratedTypes:
    """Each generated field has the type ``model_validate`` gives it (§14.8)."""

    def test_every_generated_item(self):
        problem = expanded(_typed_document()).problem
        items = [
            *problem.variables,
            *problem.objective.linear_terms,
            *problem.objective.quadratic_terms,
            *problem.constraints,
            *problem.cardinality_constraints,
        ]
        kinds = {type(item) for item in items}
        assert kinds == {Variable, LinearTerm, QuadraticTerm, Constraint, CardinalityConstraint}
        for k, item in enumerate(items):
            validated = type(item).model_validate(item.model_dump())
            _assert_same_types(item, validated, f"{type(item).__name__}#{k}")
            assert validated == item

    def test_the_counted_types(self):
        problem = expanded(_typed_document()).problem
        integers = [v for v in problem.variables if v.type == "integer"]
        assert integers and all(
            type(v.lower_bound) is int and type(v.upper_bound) is int for v in integers
        )
        assert all(type(c.rhs) is int for c in problem.cardinality_constraints)
        assert all(type(c.rhs) is float for c in problem.constraints)
        assert all(
            type(t.coefficient) is float for c in problem.constraints for t in c.terms
        )
        soft = [c for c in problem.constraints if c.type == "soft"]
        assert soft and all(type(c.weight) is float for c in soft)

    def test_the_whole_problem_revalidates_equal(self):
        problem = expanded(_typed_document()).problem
        again = OptimizationProblem.model_validate(problem.model_dump())
        assert again == problem
        assert again.model_dump_json() == problem.model_dump_json()


# ---------------------------------------------------------------------------
# Malicious input (§14.16 item 8)
# ---------------------------------------------------------------------------

ALLOWED_CODES = {
    "TEMPLATE_REFERENCE_INVALID",
    "PARAMETER_TABLE_INVALID",
    "INDEX_SET_INVALID",
    "INVALID_FIELD_VALUE",
}
PYTHON = "__import__('os').system('echo pwned')"


def _malicious_documents():
    long_text = "q" * 10_000
    yield "long set name", document(index_sets=[{"name": long_text, "elements": ["a"]}])
    yield "long element", document(index_sets=[{"name": "s", "elements": [long_text]}])
    yield "long reference", _linear(term(variable="x[" + "i," * 200 + "i]"))
    yield "text over 256", _linear(term(variable=long_text))
    yield "many rows", document(
        index_sets=[S],
        parameters=[parameter("p", ["s"], rows=[((f"k{r}",), r) for r in range(5000)])],
    )
    yield "python in variable", _linear(term(variable=PYTHON))
    yield "python in coefficient", _linear(term(coefficient=PYTHON))
    yield "python in where", _linear(term(where=[PYTHON]))
    yield "python in for_each", _linear(term(for_each=[PYTHON]))
    yield "python as element", document(index_sets=[{"name": "s", "elements": [PYTHON]}])
    yield "python as template id", document(index_sets=[S], variable_families=[X], constraint_templates=[linear_constraint(id=PYTHON)])
    yield "20-digit shift", document(index_sets=[T], variable_families=[Y], objective=objective(linear=[term(variable="y[p+12345678901234567890]", for_each=["p in t"])]))
    yield "unicode digit shift", document(index_sets=[T], variable_families=[Y], objective=objective(linear=[term(variable="y[p+١]", for_each=["p in t"])]))
    yield "unicode letters", _linear(term(variable="х[i]"))
    yield "full-width space", _linear(term(for_each=["i　in s"]))
    yield "unicode digit element", document(index_sets=[{"name": "s", "elements": ["١"]}])
    yield "control characters", _linear(term(where=["i == i\x00"]))
    yield "trailing newline", _linear(term(variable="x[i]\n"))
    yield "newline element", document(index_sets=[{"name": "s", "elements": ["a\n"]}])
    yield "1e999 in where", _linear(term(where=["1e999 > 0"]))
    yield "1e999 as coefficient text", _linear(term(coefficient="1e999"))
    yield "Infinity as coefficient", _linear(term(coefficient=float("inf")))
    yield "Infinity as text", _linear(term(coefficient="Infinity"))
    yield "400-digit coefficient", _linear(term(coefficient=10**400))
    yield "5000-digit coefficient", _linear(term(coefficient=10**5000))
    yield "400-digit element", document(index_sets=[{"name": "s", "elements": [10**400]}])
    yield "5000-digit element", document(index_sets=[{"name": "s", "elements": [-(10**5000)]}])
    yield "5000-digit key", document(index_sets=[T], parameters=[parameter("p", ["t"], rows=[((10**5000,), 1)])])
    yield "5000-digit cardinality rhs", document(index_sets=[S], variable_families=[X], cardinality_constraint_templates=[cardinality(rhs=10**5000)])
    yield "deep brackets", _linear(term(variable="x" + "[" * 120 + "i" + "]" * 120))


class TestMaliciousInput:
    """Nothing raises and nothing is evaluated (§14.16 item 8)."""

    @pytest.mark.parametrize(
        "data", [data for _, data in _malicious_documents()], ids=[name for name, _ in _malicious_documents()]
    )
    def test_only_the_listed_codes(self, data):
        parsed = parse_problem(data)
        if isinstance(parsed, list):
            codes = {error.code for error in parsed}
        else:
            result = expand_problem(parsed)
            assert result.problem is None
            codes = {error.code for error in result.errors}
            assert len(result.errors) <= 100
        assert codes and codes <= ALLOWED_CODES, codes

    def test_nothing_is_evaluated(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("the expander must not evaluate anything")

        documents = [data for _, data in _malicious_documents()]
        problems = [p for p in map(parse_problem, documents) if not isinstance(p, list)]
        monkeypatch.setattr(builtins, "eval", refuse)
        monkeypatch.setattr(builtins, "exec", refuse)
        monkeypatch.setattr(builtins, "compile", refuse)
        monkeypatch.setattr(builtins, "__import__", refuse)
        for problem in problems:
            assert expand_problem(problem).errors

    def test_many_rows_are_capped(self):
        data = dict(_malicious_documents())["many rows"]
        found = failed(data)
        assert len(found) == 20
        assert found[-1][2].endswith(
            "; 4980 more errors from parameters[0] are not listed "
            "(PARAMETER_TABLE_INVALID ×4980)"
        )

    def test_a_5000_digit_family_bound_does_not_raise(self):
        """Expansion never raises (§14.1 principle 6, §14.16 item 8).

        The family's bounds are checked with the explicit variables' rule
        and wording, whose message must not format a bound too long for
        ``str()`` (Python refuses more than 4,300 digits).
        """
        data = document(
            index_sets=[S],
            variable_families=[
                {"name": "n", "indices": ["s"], "type": "integer", "lower_bound": 0, "upper_bound": 10**5000}
            ],
        )
        result = expand(data)
        assert [(e.code, e.path) for e in result.errors] == [
            ("INTEGER_RANGE_TOO_LARGE", "variable_families[0]")
        ]
