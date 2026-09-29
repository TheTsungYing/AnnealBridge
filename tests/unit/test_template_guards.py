"""Nothing downstream of the expansion accepts an unexpanded problem.

Schema 1.3 spec 2026-09-25 §14 (v4):

* §14.1 principle 5 and §14.12 -- a problem (or objective) that still holds
  any template field is refused with ``TemplatesNotExpandedError`` by every
  public function of §14.0 item 6: ``all_constraints()`` and the thirteen
  functions that do not go through it (``BQMCompiler.prepare`` /
  ``compile``, ``CQMCompiler.compile``, ``validate_solution``,
  ``validate_batch``, ``process_candidates`` -- also with no candidate at
  all --, the five that take an ``Objective``, ``variable_bounds``,
  ``encode_integer_variables`` and ``estimate_interaction_density`` -- also
  with fewer than two explicit variables), plus those covered through
  ``all_constraints()``: ``estimate_model_variables``,
  ``compute_penalty_scale``, ``ScaledPenaltyStrategy.initial_penalty``,
  ``postprocess_costs`` and ``run_postprocess``. Otherwise a template would
  silently contribute nothing and an infeasible sample could pass.
* §14.12 -- ``has_templates()`` is true as soon as any one of the seven
  template fields is non-empty; the message names the field(s) and points
  at ``expand_problem``. ``TemplatesNotExpandedError`` is a ``ValueError``
  re-exported from ``annealbridge.models``.
"""

import json

import numpy as np
import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.compiler.integer_encoding import encode_integer_variables
from annealbridge.compiler.objective import build_objective_bqm, build_objective_qm
from annealbridge.exceptions import TemplatesNotExpandedError
from annealbridge.models import Objective, OptimizationProblem
from annealbridge.orchestration.candidates import (
    evaluate_objective,
    evaluate_objective_batch,
    process_candidates,
)
from annealbridge.orchestration.postprocess import (
    PostprocessRequest,
    postprocess_costs,
    run_postprocess,
)
from annealbridge.penalty.strategy import ScaledPenaltyStrategy
from annealbridge.solvers import RawSolverResult
from annealbridge.validation import expand_problem, validate_batch, validate_solution
from annealbridge.validation.estimates import (
    compute_objective_scale,
    compute_penalty_scale,
    estimate_interaction_density,
    estimate_model_variables,
    variable_bounds,
)
from tests.conftest import EXAMPLES_DIR


def tsp_template() -> OptimizationProblem:
    """The shipped 1.3 example: no explicit variable, everything templated."""
    return OptimizationProblem.model_validate(
        json.loads((EXAMPLES_DIR / "tsp_template.json").read_text(encoding="utf-8"))
    )


def with_explicit_variables(count: int) -> OptimizationProblem:
    """``tsp_template`` plus ``count`` explicit variables and a term on each."""
    data = json.loads((EXAMPLES_DIR / "tsp_template.json").read_text(encoding="utf-8"))
    data["variables"] = [{"name": f"u{index}"} for index in range(count)]
    data["objective"]["linear_terms"] = [
        {"variable": f"u{index}", "coefficient": 1} for index in range(count)
    ]
    return OptimizationProblem.model_validate(data)


TEMPLATED = tsp_template()
EXPANDED = expand_problem(TEMPLATED).problem
NAMES = [variable.name for variable in EXPANDED.variables]
# One valid tour (a-b-c-d) over the expanded variables, as a dict and a row.
TOUR = {name: int(name in {"x[a,0]", "x[b,1]", "x[c,2]", "x[d,3]"}) for name in NAMES}
ROWS = np.array([[TOUR[name] for name in NAMES]], dtype=np.int8)
TEMPLATED_OBJECTIVE = Objective.model_validate(
    {
        "direction": "minimize",
        "linear_terms": [],
        "linear_term_templates": [
            {"for_each": ["i in item"], "coefficient": 1, "variable": "x[i]"}
        ],
    }
)


def raw(rows: np.ndarray) -> RawSolverResult:
    return RawSolverResult(
        variables=NAMES,
        samples=rows,
        energies=np.zeros(rows.shape[0]),
        backend="exact",
    )


PROBLEM_CALLS = {
    "all_constraints": lambda p: p.all_constraints(),
    "BQMCompiler.prepare": lambda p: BQMCompiler().prepare(p),
    "BQMCompiler.compile": lambda p: BQMCompiler().compile(p, 1.0),
    "CQMCompiler.compile": lambda p: CQMCompiler().compile(p, None),
    "validate_solution": lambda p: validate_solution(p, TOUR),
    "validate_batch": lambda p: validate_batch(p, NAMES, ROWS),
    "process_candidates": lambda p: process_candidates(p, raw(ROWS)),
    "process_candidates without candidates": lambda p: process_candidates(
        p, raw(np.zeros((0, len(NAMES)), dtype=np.int8))
    ),
    "variable_bounds": variable_bounds,
    "encode_integer_variables": encode_integer_variables,
    "estimate_interaction_density": estimate_interaction_density,
    # Through all_constraints():
    "estimate_model_variables bqm": lambda p: estimate_model_variables(p, "bqm"),
    "estimate_model_variables cqm": lambda p: estimate_model_variables(p, "cqm"),
    "compute_penalty_scale": compute_penalty_scale,
    "ScaledPenaltyStrategy.initial_penalty": (
        lambda p: ScaledPenaltyStrategy().initial_penalty(p)
    ),
    "postprocess_costs": lambda p: postprocess_costs(p, 10**9),
}


def postprocess(problem: OptimizationProblem, names: list[str], rows: np.ndarray):
    count = rows.shape[0]
    return run_postprocess(
        problem,
        names,
        rows,
        np.ones(count, dtype=bool),
        np.zeros(count),
        np.zeros(count),
        PostprocessRequest(candidates=1, max_evaluations=10**6),
    )


OBJECTIVE_CALLS = {
    "evaluate_objective": lambda o: evaluate_objective(o, TOUR),
    "evaluate_objective_batch": lambda o: evaluate_objective_batch(o, NAMES, ROWS),
    "build_objective_bqm": build_objective_bqm,
    "build_objective_qm": lambda o: build_objective_qm(o, EXPANDED.variables),
    "compute_objective_scale": compute_objective_scale,
}


@pytest.mark.parametrize("name", list(PROBLEM_CALLS))
def test_every_problem_function_refuses_an_unexpanded_problem(name):
    call = PROBLEM_CALLS[name]
    with pytest.raises(TemplatesNotExpandedError, match="expand_problem"):
        call(TEMPLATED)
    # The expanded problem is accepted by the very same call.
    call(EXPANDED)


@pytest.mark.parametrize("name", list(OBJECTIVE_CALLS))
def test_every_objective_function_refuses_an_unexpanded_objective(name):
    call = OBJECTIVE_CALLS[name]
    for objective in (TEMPLATED_OBJECTIVE, TEMPLATED.objective):
        with pytest.raises(
            TemplatesNotExpandedError, match=r"objective\.\w+_term_templates"
        ):
            call(objective)
    call(EXPANDED.objective)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_interaction_density_refuses_before_its_early_return(count):
    # With fewer than two explicit variables the density is 0.0 without
    # looking at anything else; the guard still comes first (spec §14.12).
    candidate = with_explicit_variables(count)
    assert len(candidate.variables) == count
    with pytest.raises(TemplatesNotExpandedError):
        estimate_interaction_density(candidate)


def test_run_postprocess_refuses_an_unexpanded_problem():
    # Columns naming explicit variables: the neighbourhood reaches
    # all_constraints(), which refuses.
    candidate = with_explicit_variables(3)
    names = ["u0", "u1", "u2"]
    with pytest.raises(TemplatesNotExpandedError, match="expand_problem"):
        postprocess(candidate, names, np.array([[1, 0, 1]], dtype=np.int8))
    postprocess(EXPANDED, NAMES, ROWS)


def test_run_postprocess_refuses_an_unexpanded_problem_with_generated_columns():
    # The natural misuse: the columns of the expanded problem (generated
    # names) with the problem as submitted. Spec §14.12 requires
    # TemplatesNotExpandedError; the neighbourhood reads problem.variables
    # before all_constraints(), so the guard is never reached.
    with pytest.raises(TemplatesNotExpandedError, match="expand_problem"):
        postprocess(TEMPLATED, NAMES, ROWS)


def test_process_candidates_refuses_even_with_postprocess_and_no_sample():
    empty = raw(np.zeros((0, len(NAMES)), dtype=np.int8))
    with pytest.raises(TemplatesNotExpandedError):
        process_candidates(
            TEMPLATED,
            empty,
            postprocess=PostprocessRequest(candidates=1, max_evaluations=10**6),
        )


# --------------------------------------------------------------------------
# has_templates / template_fields / require_expanded, field by field
# --------------------------------------------------------------------------

BASE = {
    "version": "1.3",
    "name": "one template field",
    "variables": [{"name": "u"}],
    "objective": {
        "direction": "minimize",
        "linear_terms": [{"variable": "u", "coefficient": 1}],
    },
    "constraints": [],
}

# Each is the only non-empty template field of its problem. Shape only: the
# expander would reject most of them, the guard must not care.
ONE_FIELD = {
    "index_sets": {"index_sets": [{"name": "item", "elements": ["a"]}]},
    "parameters": {"parameters": [{"name": "p", "indices": ["item"], "default": 1}]},
    "variable_families": {"variable_families": [{"name": "x", "indices": ["item"]}]},
    "objective.linear_term_templates": {
        "objective": {
            "linear_term_templates": [{"coefficient": 1, "variable": "u"}]
        }
    },
    "objective.quadratic_term_templates": {
        "objective": {
            "quadratic_term_templates": [
                {"coefficient": 1, "variable1": "u", "variable2": "u"}
            ]
        }
    },
    "constraint_templates": {
        "constraint_templates": [
            {"id": "c", "type": "hard", "terms": [{"coefficient": 1, "variable": "u"}],
             "operator": "<=", "rhs": 1}
        ]
    },
    "cardinality_constraint_templates": {
        "cardinality_constraint_templates": [
            {"id": "k", "type": "hard", "variables": ["u"], "operator": "<=", "rhs": 1}
        ]
    },
}  # fmt: skip


def one_field_problem(field: str) -> OptimizationProblem:
    data = json.loads(json.dumps(BASE))
    for key, value in ONE_FIELD[field].items():
        if key == "objective":
            data["objective"].update(value)
        else:
            data[key] = value
    return OptimizationProblem.model_validate(data)


def test_a_problem_without_template_fields_has_none():
    plain = OptimizationProblem.model_validate(BASE)
    assert not plain.has_templates()
    assert plain.template_fields() == []
    plain.require_expanded("anything")
    assert plain.all_constraints() == []
    assert not plain.objective.has_templates()
    assert not EXPANDED.has_templates()


@pytest.mark.parametrize("field", list(ONE_FIELD))
def test_any_single_template_field_counts(field):
    candidate = one_field_problem(field)
    assert candidate.has_templates()
    assert candidate.template_fields() == [field]
    assert candidate.objective.has_templates() == field.startswith("objective.")
    with pytest.raises(TemplatesNotExpandedError) as raised:
        candidate.all_constraints()
    message = str(raised.value)
    assert field in message
    assert "expand_problem" in message
    assert "all_constraints()" in message
    with pytest.raises(TemplatesNotExpandedError) as raised:
        candidate.require_expanded("some operation")
    assert str(raised.value).startswith("some operation needs an expanded problem")


@pytest.mark.parametrize(
    "field", ["objective.linear_term_templates", "objective.quadratic_term_templates"]
)
def test_objective_guard_names_its_field(field):
    objective = one_field_problem(field).objective
    assert objective.template_fields() == [field]
    with pytest.raises(TemplatesNotExpandedError) as raised:
        objective.require_expanded("evaluate")
    message = str(raised.value)
    assert message.startswith("evaluate needs an expanded objective")
    assert field in message
    assert "expand_problem" in message


def test_all_template_fields_are_named_in_declaration_order():
    data = json.loads(json.dumps(BASE))
    for field in ONE_FIELD:
        for key, value in ONE_FIELD[field].items():
            if key == "objective":
                data["objective"].update(value)
            else:
                data[key] = value
    candidate = OptimizationProblem.model_validate(data)
    assert candidate.template_fields() == list(ONE_FIELD)
    with pytest.raises(TemplatesNotExpandedError) as raised:
        candidate.require_expanded("compile")
    assert ", ".join(ONE_FIELD) in str(raised.value)


def test_the_error_is_a_value_error():
    assert issubclass(TemplatesNotExpandedError, ValueError)


def test_the_error_is_re_exported_by_models():
    # Spec §14.12: defined in the leaf module annealbridge.exceptions and
    # re-exported by annealbridge.models.
    import annealbridge.models as models

    assert getattr(models, "TemplatesNotExpandedError", None) is TemplatesNotExpandedError
