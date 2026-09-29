"""Templates expand to exactly the problem written out by hand (schema 1.3).

Schema 1.3 spec 2026-09-25 §14 (v4):

* §14.16 item 2 -- every template document under ``tests/fixtures/templates``
  (and ``examples/tsp_template.json``) is paired with an explicit problem
  written *by hand* from the ordering and naming rules of §14.7 (explicit
  entries first, then each template in list order, bindings nested with the
  first ``for_each`` outermost, members in member-template order; generated
  names ``family[e1,...]`` and ids ``id[e1,...]``), and from the pruning of
  unused generated variables the user chose in §15 (choice 5 = B). The
  oracles are never produced by the expander. Each pair must agree on
  ``model_dump``, on the field types (§14.8: generated entries are
  normalised to what ``model_validate`` produces), on
  ``snapshot_problem`` (BQM, CQM, estimates, penalty scale, validator
  output) and on the exact optimum (assignment 8, TSP 8, roster 7).
* §14.2 -- ``tsp_template.json`` against ``tsp.json`` with ``x_i_p`` renamed
  to ``x[i,p]``: the compiled BQM agrees bit for bit (variable insertion
  order, every linear bias, the quadratic items in iteration order, the
  offset) at the initial penalty, 0.1 and 1e300; the CQM agrees on the
  objective and the constraint contents (only the labels differ).
* §14.16 item 9 / §7.1 -- a cardinality template's hard ``<= 1`` compiles to
  the pairwise penalty on the BQM path: no slack variable.
"""

import json
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.penalty.strategy import ScaledPenaltyStrategy
from annealbridge.validation import expand_problem, validate_problem_full
from tests.conftest import EXAMPLES_DIR
from tests.golden.record_phase3a_golden import snapshot_problem

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "templates"

# name -> (template document, hand-written oracle, exact optimum)
PAIRS = {
    "tsp": (EXAMPLES_DIR / "tsp_template.json", FIXTURES / "tsp_explicit.json", 8.0),
    "assignment": (
        FIXTURES / "assignment_template.json",
        FIXTURES / "assignment_explicit.json",
        8.0,
    ),
    "shift": (FIXTURES / "shift_template.json", FIXTURES / "shift_explicit.json", 7.0),
    # Linear index set: the shifted terms and constraints past the last day
    # are left out (TEMPLATE_BOUNDARY_SKIPPED).
    "window": (FIXTURES / "window_template.json", FIXTURES / "window_explicit.json", 12.0),
    # Explicit entries naming generated variables, an integer family, a soft
    # template with parameter rhs / weight, a parameter cardinality rhs.
    "mixed": (FIXTURES / "mixed_template.json", FIXTURES / "mixed_explicit.json", 12.0),
}


def load(path: Path) -> OptimizationProblem:
    return OptimizationProblem.model_validate(json.loads(path.read_text(encoding="utf-8")))


def expanded(path: Path) -> OptimizationProblem:
    expansion = expand_problem(load(path))
    assert expansion.errors == ()
    assert expansion.problem is not None
    return expansion.problem


def field_types(value: object, path: str = "") -> list[tuple[str, str]]:
    """``(path, type name)`` of every field value, recursing into models and lists.

    Reads the attributes themselves, not a dump: ``model_dump`` and ``==``
    cannot tell ``1`` from ``1.0`` (spec §14.8).
    """
    if isinstance(value, BaseModel):
        found = [(path, type(value).__name__)]
        for name in type(value).model_fields:
            found += field_types(getattr(value, name), f"{path}.{name}")
        return found
    if isinstance(value, list):
        found = [(path, "list")]
        for index, item in enumerate(value):
            found += field_types(item, f"{path}[{index}]")
        return found
    return [(path, type(value).__name__)]


def best(result) -> float:
    assert result.status == "success", result.errors
    return result.solutions[0].objective_value


# --------------------------------------------------------------------------
# Template == hand-written oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_expansion_equals_the_hand_written_oracle(name):
    template, oracle, _ = PAIRS[name]
    assert expanded(template).model_dump() == load(oracle).model_dump()


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_generated_field_types_equal_model_validate(name):
    template, oracle, _ = PAIRS[name]
    assert field_types(expanded(template)) == field_types(load(oracle))


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_snapshot_equals_the_oracle_snapshot(name):
    template, oracle, _ = PAIRS[name]
    ours = json.loads(json.dumps(snapshot_problem(expanded(template))))
    theirs = json.loads(json.dumps(snapshot_problem(load(oracle))))
    assert ours == theirs


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_exact_optimum_equals_the_oracle_optimum(name):
    template, oracle, optimum = PAIRS[name]
    service = OptimizationService()
    from_template = service.solve(load(template))
    from_oracle = service.solve(load(oracle))
    assert best(from_template) == best(from_oracle) == optimum
    assert [s.variables for s in from_template.solutions] == [
        s.variables for s in from_oracle.solutions
    ]
    assert [s.ranking_score for s in from_template.solutions] == [
        s.ranking_score for s in from_oracle.solutions
    ]


def test_forbidden_assignment_variable_is_left_out_with_a_warning():
    """``where allowed[w,t] == 1`` leaves x[alice,drive] unused (§14.14 item 2, B)."""
    template = PAIRS["assignment"][0]
    expansion = expand_problem(load(template))
    assert [(w.code, w.path) for w in expansion.warnings] == [
        ("UNUSED_TEMPLATE_VARIABLES", "variable_families[0]")
    ]
    assert "x[alice,drive]" in expansion.warnings[0].message
    assert "x[alice,drive]" not in {v.name for v in expansion.problem.variables}

    result = OptimizationService().solve(load(template))
    assert result.status == "success"
    assert [w.code for w in result.warnings] == ["UNUSED_TEMPLATE_VARIABLES"]
    for solution in result.solutions:
        assert "x[alice,drive]" not in solution.variables
    assert result.solutions[0].variables == {
        "x[alice,clean]": 0,
        "x[alice,cook]": 1,
        "x[bob,clean]": 1,
        "x[bob,cook]": 0,
        "x[bob,drive]": 0,
        "x[carol,clean]": 0,
        "x[carol,cook]": 0,
        "x[carol,drive]": 1,
    }


def test_linear_boundary_skips_are_reported_per_template():
    expansion = expand_problem(load(PAIRS["window"][0]))
    assert [(w.code, w.path) for w in expansion.warnings] == [
        ("TEMPLATE_BOUNDARY_SKIPPED", "objective.quadratic_term_templates[0]"),
        ("TEMPLATE_BOUNDARY_SKIPPED", "constraint_templates[0]"),
    ]
    assert "left out 4 terms" in expansion.warnings[0].message
    assert "left out 2 constraints" in expansion.warnings[1].message


# --------------------------------------------------------------------------
# tsp_template.json == tsp.json renamed, bit for bit (spec §14.2)
# --------------------------------------------------------------------------


def renamed_tsp() -> OptimizationProblem:
    """``examples/tsp.json`` with every ``x_i_p`` renamed to ``x[i,p]``."""
    text = (EXAMPLES_DIR / "tsp.json").read_text(encoding="utf-8")
    renamed, count = re.subn(r'"x_([a-d])_([0-3])"', r'"x[\1,\2]"', text)
    assert count == 16 + 2 * 48 + 8 * 4  # declarations, term ends, members
    return OptimizationProblem.model_validate(json.loads(renamed))


def tsp_template() -> OptimizationProblem:
    return expanded(EXAMPLES_DIR / "tsp_template.json")


def test_tsp_initial_penalties_agree():
    strategy = ScaledPenaltyStrategy()
    assert strategy.initial_penalty(tsp_template()) == strategy.initial_penalty(renamed_tsp())


def _bits(value: float) -> tuple[float, str]:
    return float(value), float(value).hex()


@pytest.mark.parametrize("penalty", ["initial", 0.1, 1e300])
def test_tsp_bqm_is_bit_identical(penalty):
    explicit, template = renamed_tsp(), tsp_template()
    if penalty == "initial":
        penalty = ScaledPenaltyStrategy().initial_penalty(explicit)
    ours = BQMCompiler().compile(template, penalty).model
    theirs = BQMCompiler().compile(explicit, penalty).model

    # Variable insertion order.
    assert list(ours.variables) == list(theirs.variables)
    # Every linear bias, compared with == and as IEEE-754 hex.
    for variable in theirs.variables:
        assert ours.linear[variable] == theirs.linear[variable]
        assert _bits(ours.linear[variable]) == _bits(theirs.linear[variable])
    # The quadratic items in iteration order, keys and biases.
    our_items = list(ours.quadratic.items())
    their_items = list(theirs.quadratic.items())
    assert [key for key, _ in our_items] == [key for key, _ in their_items]
    assert [bias for _, bias in our_items] == [bias for _, bias in their_items]
    assert [_bits(bias) for _, bias in our_items] == [_bits(bias) for _, bias in their_items]
    # The offset.
    assert ours.offset == theirs.offset
    assert _bits(ours.offset) == _bits(theirs.offset)


def _expression(expression) -> dict:
    return {
        "linear": [(str(v), _bits(b)) for v, b in expression.linear.items()],
        "quadratic": [
            ((str(u), str(v)), _bits(b)) for (u, v), b in expression.quadratic.items()
        ],
        "offset": _bits(expression.offset),
    }


def test_tsp_cqm_objective_and_constraints_agree_apart_from_labels():
    ours = CQMCompiler().compile(tsp_template(), None).model
    theirs = CQMCompiler().compile(renamed_tsp(), None).model
    assert list(ours.variables) == list(theirs.variables)
    assert _expression(ours.objective) == _expression(theirs.objective)
    our_labels = list(ours.constraint_labels)
    their_labels = list(theirs.constraint_labels)
    assert our_labels == [f"city_once[{c}]" for c in "abcd"] + [
        f"position_once[{p}]" for p in range(4)
    ]
    assert len(our_labels) == len(their_labels)
    for our_label, their_label in zip(our_labels, their_labels):
        our = ours.constraints[our_label]
        their = theirs.constraints[their_label]
        assert _expression(our.lhs) == _expression(their.lhs)
        assert our.sense == their.sense
        assert _bits(our.rhs) == _bits(their.rhs)


def test_tsp_exact_top_five_is_the_same_as_tsp_json():
    service = OptimizationService()
    ours = service.solve(OptimizationProblem.model_validate(
        json.loads((EXAMPLES_DIR / "tsp_template.json").read_text(encoding="utf-8"))
    ))
    theirs = service.solve(renamed_tsp())
    assert best(ours) == best(theirs) == 8.0
    assert [(s.variables, s.objective_value) for s in ours.solutions] == [
        (s.variables, s.objective_value) for s in theirs.solutions
    ]


# --------------------------------------------------------------------------
# A cardinality template's hard <= 1 takes the pairwise encoding (§7.1)
# --------------------------------------------------------------------------

AT_MOST_ONE = {
    "version": "1.3",
    "name": "at_most_one_template",
    "index_sets": [
        {"name": "item", "elements": ["a", "b", "c"]},
        {"name": "group", "elements": ["g1", "g2"]},
    ],
    "parameters": [
        {
            "name": "in_group",
            "indices": ["group", "item"],
            "default": 0,
            "values": [
                {"key": ["g1", "a"], "value": 1},
                {"key": ["g1", "b"], "value": 1},
                {"key": ["g2", "b"], "value": 1},
                {"key": ["g2", "c"], "value": 1},
            ],
        },
        {"name": "one", "indices": ["group"], "default": 1},
    ],
    "variable_families": [{"name": "x", "indices": ["item"]}],
    "variables": [],
    "objective": {
        "direction": "maximize",
        "linear_terms": [],
        "linear_term_templates": [
            {"for_each": ["i in item"], "coefficient": 1, "variable": "x[i]"}
        ],
    },
    "constraints": [],
    "cardinality_constraint_templates": [
        {
            "id": "literal_rhs",
            "type": "hard",
            "for_each": ["g in group"],
            "variables": [
                {"for_each": ["i in item"], "where": ["in_group[g,i] == 1"], "variable": "x[i]"}
            ],
            "operator": "<=",
            "rhs": 1,
        },
        {
            "id": "parameter_rhs",
            "type": "hard",
            "for_each": ["g in group"],
            "variables": [
                {"for_each": ["i in item"], "where": ["in_group[g,i] == 1"], "variable": "x[i]"}
            ],
            "operator": "<=",
            "rhs": "one[g]",
        },
    ],
    "solver": {"backend": "exact"},
}


def test_cardinality_template_at_most_one_compiles_pairwise_without_slack():
    problem = expanded_from(AT_MOST_ONE)
    assert [c.id for c in problem.cardinality_constraints] == [
        "literal_rhs[g1]",
        "literal_rhs[g2]",
        "parameter_rhs[g1]",
        "parameter_rhs[g2]",
    ]
    penalty = ScaledPenaltyStrategy().initial_penalty(problem)
    compiled = BQMCompiler().compile(problem, penalty)
    assert not any(str(v).startswith("__slack_") for v in compiled.model.variables)
    assert list(compiled.model.variables) == ["x[a]", "x[b]", "x[c]"]
    assert not compiled.internal_variables
    assert len(compiled.constraint_trace) == 4
    for trace in compiled.constraint_trace:
        assert trace.generated_variables == []
        assert trace.slack_range is None


def expanded_from(document: dict) -> OptimizationProblem:
    expansion = expand_problem(OptimizationProblem.model_validate(document))
    assert expansion.errors == ()
    assert expansion.problem is not None
    assert validate_problem_full(expansion.problem).valid
    return expansion.problem
