"""The ``CardinalityConstraint`` model, its lowering and the dump contract.

Schema 1.2 spec 2026-09-25 §4.2-§5.3, §13 items 8 and 9, with the lowering
rule of §5.1 replaced by schema 1.3 spec §14.14 item 3:

* ``lowered()`` is cached per declaration, keyed by the declaration's field
  values: the same object while no field changes, a new one reflecting the
  new values as soon as any field differs, whether through
  ``model_copy(update=...)``, attribute assignment or an in-place edit of
  ``variables``. ``all_constraints()`` and therefore every reader see the
  change. The shared ``LoweredCardinalityConstraint`` is frozen, and the
  cache takes no part in equality, dumps, deep copies or pickling;
* ``all_constraints()`` is the linear list's own objects, then the lowered
  cardinality constraints, in declaration order;
* an empty ``cardinality_constraints`` (and, since schema 1.3, every empty
  template list) is left out of every dump, so a 1.0 / 1.1 problem dumps
  exactly as before the field existed, and every dump validates back to an
  equal problem.
"""

import copy
import json
import pickle

import pytest
from pydantic import ValidationError

from annealbridge.models import (
    CardinalityConstraint,
    Constraint,
    LinearTerm,
    LoweredCardinalityConstraint,
    Objective,
    OptimizationProblem,
)
from annealbridge.models.cardinality import _LOWERED_CACHE
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

    def test_unchanged_declaration_returns_the_same_object(self):
        """Schema 1.3 spec §14.14 item 3 replaces §5.1's "never cached"."""
        source = declaration()
        assert source.lowered() is source.lowered()

    def test_model_copy_update_rebuilds_the_lowering(self):
        source = declaration()
        old = source.lowered()
        changed = source.model_copy(update={"rhs": 1})
        new = changed.lowered()
        assert new is not old
        assert new.rhs == 1.0
        # The copy carried the source's cache over; the source keeps its own.
        assert source.lowered() is old
        assert old.rhs == 2.0

    def test_attribute_assignment_rebuilds_the_lowering(self):
        source = declaration()
        old = source.lowered()
        source.rhs = 1
        new = source.lowered()
        assert new is not old
        assert new.rhs == 1.0
        assert source.lowered() is new

    def test_in_place_edit_of_variables_rebuilds_the_lowering(self):
        source = declaration()
        old = source.lowered()
        source.variables.append("d")
        new = source.lowered()
        assert new is not old
        assert [term.variable for term in new.terms] == ["a", "b", "c", "d"]
        assert [term.variable for term in old.terms] == ["a", "b", "c"]

    def test_the_lowered_object_is_frozen(self):
        # One object is shared by every all_constraints() call while its
        # declaration is unchanged, so nobody may assign to it.
        lowered = declaration().lowered()
        with pytest.raises(ValidationError):
            lowered.rhs = 5.0
        with pytest.raises(ValidationError):
            lowered.terms = []
        assert lowered.rhs == 2.0

    def test_model_copy_of_the_declaration_is_seen(self):
        source = declaration()
        changed = source.model_copy(update={"rhs": 1, "variables": ["b", "c"]})
        assert changed.lowered().rhs == 1.0
        assert [t.variable for t in changed.lowered().terms] == ["b", "c"]
        # The original is unchanged.
        assert source.lowered().rhs == 2.0
        assert [t.variable for t in source.lowered().terms] == ["a", "b", "c"]


# A value different from ``full_declaration()``'s for every field of the
# model. A field added to CardinalityConstraint fails the coverage test below
# until it is listed here, so the tests keep proving that the cache key
# covers every field (schema 1.3 spec §14.14 item 3).
CHANGED_VALUES = {
    "id": "other",
    "description": "changed",
    "type": "hard",
    "variables": ["a", "b"],
    "operator": "==",
    "rhs": 1,
    "weight": 3.0,
}


def full_declaration() -> CardinalityConstraint:
    """A declaration with every field set, so each one can be changed."""
    return declaration(type="soft", weight=2.5)


class TestCacheKeyCoversEveryField:
    def test_every_field_has_a_changed_value(self):
        assert set(CHANGED_VALUES) == set(CardinalityConstraint.model_fields)

    @pytest.mark.parametrize("how", ["model_copy", "assignment"])
    @pytest.mark.parametrize("field", sorted(CardinalityConstraint.model_fields))
    def test_changing_the_field_rebuilds_the_lowering(self, field, how):
        assert field in CHANGED_VALUES, f"add a changed value for {field}"
        source = full_declaration()
        old = source.lowered()
        new_value = CHANGED_VALUES[field]
        assert getattr(source, field) != new_value
        if how == "model_copy":
            changed = source.model_copy(update={field: new_value})
        else:
            changed = source
            setattr(changed, field, new_value)
        new = changed.lowered()
        assert new is not old
        # The rebuilt lowering is exactly what a fresh, never-cached
        # declaration with the new value lowers to.
        fresh = CardinalityConstraint.model_validate(changed.model_dump())
        assert _LOWERED_CACHE not in fresh.__dict__
        assert new == fresh.lowered()


class TestCacheIsInvisible:
    """Schema 1.3 spec §14.14 item 3: the cache lives in a non-field key of
    the instance ``__dict__`` (as ``functools.cached_property`` does) and
    changes nothing observable. No version guard: it runs on every supported
    pydantic, including the lowest-direct CI's 2.10."""

    def test_a_declaration_with_a_cache_behaves_like_one_without(self):
        cached = declaration()
        cached.lowered()
        fresh = declaration()
        # Otherwise this test would not exercise the cache at all.
        assert _LOWERED_CACHE in cached.__dict__
        assert _LOWERED_CACHE not in fresh.__dict__

        assert cached == fresh
        assert fresh == cached
        assert cached != declaration(rhs=1)
        assert cached.model_dump() == fresh.model_dump()
        assert cached.model_dump(mode="json") == fresh.model_dump(mode="json")
        assert cached.model_dump_json() == fresh.model_dump_json()
        assert repr(cached) == repr(fresh)

        deep = copy.deepcopy(cached)
        assert deep == fresh
        assert deep.model_dump_json() == fresh.model_dump_json()
        assert deep.lowered() == fresh.lowered()
        # The deep copy shares nothing: changing it leaves the original's
        # lowering alone.
        assert deep.lowered() is not cached.lowered()
        deep.rhs = 1
        assert deep.lowered().rhs == 1.0
        assert cached.lowered().rhs == 2.0

        restored = pickle.loads(pickle.dumps(cached))
        assert restored == fresh
        assert restored.model_dump_json() == fresh.model_dump_json()
        assert restored.lowered() == fresh.lowered()

    def test_a_problem_with_cached_lowerings_behaves_like_one_without(self):
        cached = problem_12()
        cached.all_constraints()
        fresh = problem_12()
        assert all(
            _LOWERED_CACHE in constraint.__dict__
            for constraint in cached.cardinality_constraints
        )

        assert cached == fresh
        assert cached.model_dump() == fresh.model_dump()
        assert cached.model_dump_json() == fresh.model_dump_json()
        assert copy.deepcopy(cached) == fresh
        restored = pickle.loads(pickle.dumps(cached))
        assert restored == fresh
        assert restored.all_constraints() == fresh.all_constraints()


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

    def test_reused_while_unchanged_and_rebuilt_after_a_change(self):
        """Schema 1.3 spec §14.14 item 3: cached by field values, never stale."""
        problem = problem_12()
        first = problem.all_constraints()
        second = problem.all_constraints()
        # A new list each call, holding the same lowered objects.
        assert first is not second
        assert all(a is b for a, b in zip(first, second))
        problem.cardinality_constraints[0].rhs = 1
        third = problem.all_constraints()
        assert third[1] is not first[1]
        assert third[1].rhs == 1.0
        # The untouched declaration keeps its lowering.
        assert third[2] is first[2]

    def test_model_copy_update_is_seen_through_all_constraints(self):
        problem = problem_12()
        old = problem.all_constraints()[1]
        first, second = problem.cardinality_constraints
        changed = problem.model_copy(
            update={KEY: [first.model_copy(update={"rhs": 1}), second]}
        )
        new = changed.all_constraints()[1]
        assert new is not old
        assert new.rhs == 1.0
        assert problem.all_constraints()[1] is old

    def test_attribute_assignment_is_seen_through_all_constraints(self):
        problem = problem_12()
        old = problem.all_constraints()[1]
        problem.cardinality_constraints[0].rhs = 1
        new = problem.all_constraints()[1]
        assert new is not old
        assert new.rhs == 1.0

    def test_in_place_edit_of_variables_is_seen_through_all_constraints(self):
        problem = problem_12()
        old = problem.all_constraints()[1]
        problem.cardinality_constraints[0].variables.append("d")
        new = problem.all_constraints()[1]
        assert new is not old
        assert [term.variable for term in new.terms] == ["a", "b", "c", "d"]

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
        """Every field of the dump is one the 1.1 schema already had.

        Schema 1.3 spec §14.13: the five template lists of the problem and
        the two of the objective are left out while empty, like
        ``cardinality_constraints``.
        """
        problem = OLD_PROBLEMS["integer_knapsack_1_1"]()
        omitted = {
            KEY,
            "index_sets",
            "parameters",
            "variable_families",
            "constraint_templates",
            "cardinality_constraint_templates",
        }
        objective_omitted = {"linear_term_templates", "quadratic_term_templates"}
        assert omitted <= set(OptimizationProblem.model_fields)
        assert objective_omitted <= set(Objective.model_fields)

        dumped = problem.model_dump()
        assert set(dumped) == set(OptimizationProblem.model_fields) - omitted
        assert set(dumped["objective"]) == set(Objective.model_fields) - objective_omitted
        from_json = json.loads(problem.model_dump_json())
        assert set(from_json) == set(dumped)
        assert set(from_json["objective"]) == set(dumped["objective"])

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
