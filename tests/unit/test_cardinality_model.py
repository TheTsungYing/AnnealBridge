"""The ``CardinalityConstraint`` model, its lowering and the dump contract.

Schema 1.2 spec 2026-09-25 §4.2-§5.3, §13 items 8 and 9:

* ``lowered()`` is a fresh linear view on every call (never cached), so
  ``model_copy(update=...)`` and attribute assignment are seen by
  ``all_constraints()`` and therefore by every reader;
* ``all_constraints()`` is the linear list's own objects, then the lowered
  cardinality constraints, in declaration order;
* an empty ``cardinality_constraints`` is left out of every dump, so a 1.0 /
  1.1 problem dumps exactly as before the field existed, and every dump
  validates back to an equal problem.
"""

import json

import pytest

from annealbridge.models import (
    CardinalityConstraint,
    Constraint,
    LinearTerm,
    LoweredCardinalityConstraint,
    OptimizationProblem,
)
from tests.conftest import EXAMPLES_DIR

KEY = "cardinality_constraints"


def declaration(**overrides) -> CardinalityConstraint:
    data = {
        "id": "pick",
        "description": "at most two",
        "type": "hard",
        "variables": ["a", "b", "c"],
        "operator": "<=",
        "rhs": 2,
    }
    return CardinalityConstraint.model_validate({**data, **overrides})


def problem_12(**cardinality_overrides) -> OptimizationProblem:
    return OptimizationProblem.model_validate(
        {
            "version": "1.2",
            "name": "cardinality model",
            "variables": [{"name": name} for name in ("a", "b", "c", "d")],
            "objective": {
                "direction": "maximize",
                "linear_terms": [{"variable": "a", "coefficient": 1}],
            },
            "constraints": [
                {
                    "id": "lin",
                    "type": "hard",
                    "terms": [{"variable": "d", "coefficient": 2}],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
            KEY: [
                declaration(**cardinality_overrides).model_dump(),
                {
                    "id": "soft_one",
                    "type": "soft",
                    "weight": 1.5,
                    "variables": ["c", "d"],
                    "operator": "==",
                    "rhs": 1,
                },
            ],
        }
    )


class TestLowering:
    def test_lowered_is_a_linear_constraint_with_unit_coefficients(self):
        lowered = declaration().lowered()
        assert type(lowered) is LoweredCardinalityConstraint
        assert isinstance(lowered, Constraint)
        assert lowered.id == "pick"
        assert lowered.description == "at most two"
        assert lowered.type == "hard"
        assert lowered.operator == "<="
        assert lowered.weight is None
        assert lowered.terms == [
            LinearTerm(variable=name, coefficient=1.0) for name in ("a", "b", "c")
        ]
        assert all(type(term.coefficient) is float for term in lowered.terms)
        assert lowered.rhs == 2.0
        assert type(lowered.rhs) is float
        # Downstream calls float methods on the rhs (spec §5.1).
        assert lowered.rhs.is_integer()

    def test_soft_weight_is_carried(self):
        lowered = declaration(type="soft", weight=2.5).lowered()
        assert lowered.type == "soft"
        assert lowered.weight == 2.5

    def test_every_call_builds_a_new_object(self):
        source = declaration()
        assert source.lowered() is not source.lowered()
        assert source.lowered() == source.lowered()

    def test_model_copy_of_the_declaration_is_seen(self):
        source = declaration()
        changed = source.model_copy(update={"rhs": 1, "variables": ["b", "c"]})
        assert changed.lowered().rhs == 1.0
        assert [t.variable for t in changed.lowered().terms] == ["b", "c"]
        # The original is unchanged.
        assert source.lowered().rhs == 2.0
        assert [t.variable for t in source.lowered().terms] == ["a", "b", "c"]


class TestAllConstraints:
    def test_linear_first_then_cardinality_in_declaration_order(self):
        problem = problem_12()
        constraints = problem.all_constraints()
        assert [c.id for c in constraints] == ["lin", "pick", "soft_one"]
        assert constraints[0] is problem.constraints[0]
        assert [type(c) for c in constraints[1:]] == [LoweredCardinalityConstraint] * 2

    def test_without_cardinality_it_is_the_linear_list_itself(self):
        problem = problem_12().model_copy(update={KEY: []})
        constraints = problem.all_constraints()
        assert len(constraints) == len(problem.constraints)
        assert all(a is b for a, b in zip(constraints, problem.constraints))

    def test_rebuilt_on_every_call(self):
        problem = problem_12()
        assert problem.all_constraints()[1] is not problem.all_constraints()[1]
        assert problem.all_constraints() is not problem.all_constraints()

    @pytest.mark.parametrize(
        "update, expected_rhs, expected_variables",
        [
            ({"rhs": 1}, 1.0, ["a", "b", "c"]),
            ({"variables": ["a", "d"]}, 2.0, ["a", "d"]),
            ({"rhs": 0, "variables": ["b"]}, 0.0, ["b"]),
        ],
        ids=["rhs", "variables", "both"],
    )
    def test_model_copy_update_is_reflected(self, update, expected_rhs, expected_variables):
        problem = problem_12()
        first, second = problem.cardinality_constraints
        changed = problem.model_copy(update={KEY: [first.model_copy(update=update), second]})
        lowered = changed.all_constraints()[1]
        assert lowered.rhs == expected_rhs
        assert [term.variable for term in lowered.terms] == expected_variables
        # The source problem still lowers its own declaration.
        assert problem.all_constraints()[1].rhs == 2.0

    def test_attribute_assignment_is_reflected(self):
        problem = problem_12()
        problem.all_constraints()  # a first read must not pin anything
        problem.cardinality_constraints[0].rhs = 3
        problem.cardinality_constraints[1].variables = ["a", "b", "c"]
        constraints = problem.all_constraints()
        assert constraints[1].rhs == 3.0
        assert [term.variable for term in constraints[2].terms] == ["a", "b", "c"]

    def test_validator_paths_line_up_with_all_constraints(self):
        # The validator builds its own (path, constraint) list; it must be
        # all_constraints() in the same order, each at its declared path.
        from annealbridge.validation.problem_validator import _constraint_paths

        problem = problem_12()
        entries = _constraint_paths(problem)
        assert [c.id for _, c in entries] == [c.id for c in problem.all_constraints()]
        assert [path for path, _ in entries] == [
            "constraints[0]",
            "cardinality_constraints[0]",
            "cardinality_constraints[1]",
        ]

    def test_rhs_limit_is_the_integer_bound_limit(self):
        # models/cardinality.py cannot import the validator (models is the
        # lowest layer), so the two constants are kept equal by this test.
        from annealbridge.models.cardinality import _RHS_LIMIT
        from annealbridge.validation.problem_validator import INTEGER_BOUND_LIMIT

        assert _RHS_LIMIT == INTEGER_BOUND_LIMIT


# --------------------------------------------------------------------------
# Dump / validate round trip (§4.3)
# --------------------------------------------------------------------------


def load_example(name: str) -> OptimizationProblem:
    return OptimizationProblem.model_validate(
        json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))
    )


OLD_PROBLEMS = {
    "knapsack_1_0": lambda: load_example("knapsack.json"),
    "shift_scheduling_1_0": lambda: load_example("shift_scheduling.json"),
    "integer_knapsack_1_1": lambda: load_example("integer_knapsack.json"),
    "explicit_empty_list_1_1": lambda: OptimizationProblem.model_validate(
        {
            "version": "1.1",
            "name": "explicit empty",
            "variables": [{"name": "x"}],
            "objective": {"direction": "minimize", "linear_terms": []},
            "constraints": [],
            KEY: [],
        }
    ),
}


class TestDump:
    @pytest.mark.parametrize("name", sorted(OLD_PROBLEMS))
    def test_older_problems_dump_without_the_key(self, name):
        problem = OLD_PROBLEMS[name]()
        assert problem.version in ("1.0", "1.1")
        assert KEY not in problem.model_dump()
        assert KEY not in problem.model_dump(mode="json")
        assert KEY not in json.loads(problem.model_dump_json())
        assert f'"{KEY}"' not in problem.model_dump_json()

    @pytest.mark.parametrize("name", sorted(OLD_PROBLEMS))
    def test_older_problems_round_trip(self, name):
        problem = OLD_PROBLEMS[name]()
        assert OptimizationProblem.model_validate(problem.model_dump()) == problem
        assert OptimizationProblem.model_validate_json(problem.model_dump_json()) == problem

    def test_the_dump_is_the_pre_1_2_shape(self):
        """Every field of the dump is one the 1.1 schema already had."""
        problem = OLD_PROBLEMS["integer_knapsack_1_1"]()
        assert set(problem.model_dump()) == set(OptimizationProblem.model_fields) - {KEY}

    def test_cardinality_problem_dumps_the_key_and_round_trips(self):
        problem = problem_12()
        dumped = problem.model_dump()
        assert [entry["id"] for entry in dumped[KEY]] == ["pick", "soft_one"]
        assert [entry["id"] for entry in json.loads(problem.model_dump_json())[KEY]] == [
            "pick",
            "soft_one",
        ]
        assert OptimizationProblem.model_validate(dumped) == problem
        assert OptimizationProblem.model_validate_json(problem.model_dump_json()) == problem
        restored = OptimizationProblem.model_validate(dumped)
        assert [c.id for c in restored.all_constraints()] == ["lin", "pick", "soft_one"]

    def test_version_1_2_without_cardinality_omits_the_key(self):
        problem = problem_12().model_copy(update={KEY: []})
        assert KEY not in problem.model_dump()
        assert OptimizationProblem.model_validate(problem.model_dump()) == problem

    def test_exclude_and_include_still_work(self):
        problem = problem_12()
        assert set(problem.model_dump(include={"name", KEY})) == {"name", KEY}
        assert KEY not in problem.model_dump(exclude={KEY})

    def test_the_input_schema_still_lists_the_field(self):
        """Validation-mode JSON Schema is unaffected by the serializer."""
        schema = OptimizationProblem.model_json_schema()
        assert KEY in schema["properties"]
        assert KEY not in schema.get("required", [])
        definition = schema["$defs"]["CardinalityConstraint"]
        assert definition["additionalProperties"] is False
        assert set(definition["required"]) == {"id", "type", "variables", "operator", "rhs"}
        rhs = definition["properties"]["rhs"]
        assert rhs["type"] == "integer"
        assert rhs["maximum"] == 2**31 - 1
        assert rhs["minimum"] == -(2**31 - 1)
