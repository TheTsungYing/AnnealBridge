"""Scenario: schema 1.3 template documents through ``OptimizationService``.

JSON text -> ``OptimizationProblem`` -> ``OptimizationService`` -> result,
the route an MCP or CLI caller takes. Schema 1.3 spec 2026-09-25 §14 (v4):

* §14.8 -- validate, recommend and solve all expand first; the service's
  validate equals ``validate_problem_full`` of the same document with the
  same backend declaration; solve's warnings equal validate's (the
  expansion's own warnings first); solutions carry the generated names and
  ``constraint_evaluations`` the generated ids, in ``all_constraints()``
  order; an expansion error is ``invalid_problem`` without warnings.
* §14.16 items 6-7 -- re-validation against the expansion: a sample that
  breaks a generated constraint is infeasible.
* §14.16 item 9 -- post-processing, the wall-clock limit and the BQM retry
  cache (``PreparedBQM``, prepared once for several attempts) all run on
  the expanded problem.
* §14.9 -- TEMPLATE_EXPANSION_LIMIT is ``resource_limit_exceeded`` for solve
  and ``valid: false`` for validate and recommend, alone and without
  warnings; the expansion gate (sized ``max_concurrent_solves``) answers
  CONCURRENCY_LIMIT to template documents only while it is full.
* §14.0 item 3 / §14.8 -- recommend has no top-level warnings, so each
  backend's warnings start with the expansion's.
"""

import json
from pathlib import Path

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import (
    expand_problem,
    validate_problem_full,
    validate_solution,
)
from tests.conftest import EXAMPLES_DIR

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "templates"
TSP_TEMPLATE_JSON = (EXAMPLES_DIR / "tsp_template.json").read_text(encoding="utf-8")
CITY_IDS = [f"city_once[{c}]" for c in "abcd"]
POSITION_IDS = [f"position_once[{p}]" for p in range(4)]
GENERATED_NAMES = {f"x[{c},{p}]" for c in "abcd" for p in range(4)}


def from_json(text: str, **solver) -> OptimizationProblem:
    data = json.loads(text)
    if solver:
        data["solver"] = {**data.get("solver", {}), **solver}
    return OptimizationProblem.model_validate(data)


def fixture(name: str, **solver) -> OptimizationProblem:
    return from_json((FIXTURES / name).read_text(encoding="utf-8"), **solver)


def tsp_template(**solver) -> OptimizationProblem:
    return from_json(TSP_TEMPLATE_JSON, **solver)


def exact_capabilities():
    return SolverRegistry.default().get("exact").capabilities


# --------------------------------------------------------------------------
# validate / recommend / solve on tsp_template
# --------------------------------------------------------------------------


def test_validate_matches_validate_problem_full():
    problem = tsp_template()
    result = OptimizationService().validate(problem)
    assert result.valid
    assert result.errors == []
    assert result.estimated_compiled_variables == 16
    assert result.model_type == "bqm"
    direct = validate_problem_full(
        problem,
        capabilities=exact_capabilities(),
        max_compiled_variables=int(ExecutionPolicy().required_limit("variables")),
        model_type="bqm",
    )
    assert result == direct


def test_recommend_assesses_the_expanded_problem():
    result = OptimizationService().recommend(tsp_template())
    assert result.valid
    assert result.errors == []
    by_name = {entry.backend: entry for entry in result.recommendations}
    assert set(by_name) == set(SolverRegistry.default().names())
    exact = by_name["exact"]
    assert exact.usable
    assert exact.model_type == "bqm"
    assert exact.estimated_compiled_variables == 16
    assert [entry.rank for entry in result.recommendations] == list(
        range(1, len(result.recommendations) + 1)
    )


def test_solve_returns_generated_names_and_ids():
    problem = tsp_template()
    expanded = expand_problem(problem).problem
    result = OptimizationService().solve(problem)
    assert result.status == "success"
    assert result.errors == []
    assert result.solutions[0].objective_value == 8.0
    ids = [constraint.id for constraint in expanded.all_constraints()]
    assert ids == CITY_IDS + POSITION_IDS
    for solution in result.solutions:
        assert set(solution.variables) == GENERATED_NAMES
        assert solution.hard_constraints_satisfied
        # constraint_evaluations use the generated ids, in all_constraints order.
        assert [e.constraint_id for e in solution.constraint_evaluations] == ids
        # Re-validated against the expansion, not taken from the solver.
        assert validate_solution(expanded, solution.variables).feasible


@pytest.mark.parametrize(
    ("name", "codes"),
    [
        ("assignment_template.json", ["UNUSED_TEMPLATE_VARIABLES"]),
        (
            "window_template.json",
            [
                "TEMPLATE_BOUNDARY_SKIPPED",
                "TEMPLATE_BOUNDARY_SKIPPED",
                "CARDINALITY_FORM_AVAILABLE",
            ],
        ),
    ],
)
def test_solve_warnings_equal_validate_warnings(name, codes):
    problem = fixture(name)
    service = OptimizationService()
    validation = service.validate(problem)
    result = service.solve(problem)
    assert result.status == "success"
    assert [w.code for w in validation.warnings] == codes
    assert result.warnings == validation.warnings
    # The expansion's own warnings lead.
    expansion_warnings = list(expand_problem(problem).warnings)
    assert validation.warnings[: len(expansion_warnings)] == expansion_warnings


def test_a_sample_breaking_a_generated_constraint_is_infeasible():
    expanded = expand_problem(tsp_template()).problem
    tour = {name: 0 for name in GENERATED_NAMES}
    for city, position in zip("abcd", range(4)):
        tour[f"x[{city},{position}]"] = 1
    assert validate_solution(expanded, tour).feasible

    # a visited twice, d never: breaks city_once[a], city_once[d] and
    # position_once[3].
    broken = dict(tour, **{"x[a,3]": 1, "x[d,3]": 0})
    verdict = validate_solution(expanded, broken)
    assert not verdict.feasible
    assert [e.constraint_id for e in verdict.hard_violations] == [
        "city_once[a]",
        "city_once[d]",
    ]
    assert [e.constraint_id for e in verdict.evaluations] == CITY_IDS + POSITION_IDS


# --------------------------------------------------------------------------
# Post-processing, wall clock and the BQM retry cache on the expansion
# --------------------------------------------------------------------------


def test_sa_with_postprocess_and_wall_clock_limit_solves_the_template():
    problem = tsp_template(
        backend="simulated_annealing",
        seed=1,
        num_reads=20,
        postprocess="repair_local_search",
        wall_clock_limit_seconds=60,
    )
    expanded = expand_problem(problem).problem
    result = OptimizationService().solve(problem)
    assert result.status == "success", result.errors
    assert not result.wall_clock_limit_reached
    assert result.attempts[0].postprocess is not None
    assert result.attempts[0].postprocess.candidates_selected > 0
    best = result.solutions[0]
    assert best.hard_constraints_satisfied
    assert set(best.variables) == GENERATED_NAMES
    assert validate_solution(expanded, best.variables).feasible
    assert best.objective_value >= 8.0


KNAPSACK_TEMPLATE = """
{
  "version": "1.3",
  "name": "knapsack_template",
  "description": "examples/knapsack.json with templates: capacity 10, optimum take a and c (value 17).",
  "index_sets": [{"name": "item", "elements": ["a", "b", "c", "d"]}],
  "parameters": [
    {"name": "value", "indices": ["item"], "values": [
      {"key": ["a"], "value": 10}, {"key": ["b"], "value": 8},
      {"key": ["c"], "value": 7}, {"key": ["d"], "value": 6}]},
    {"name": "weight", "indices": ["item"], "values": [
      {"key": ["a"], "value": 6}, {"key": ["b"], "value": 5},
      {"key": ["c"], "value": 4}, {"key": ["d"], "value": 3}]}
  ],
  "variable_families": [{"name": "take", "indices": ["item"]}],
  "variables": [],
  "objective": {
    "direction": "maximize",
    "linear_terms": [],
    "linear_term_templates": [
      {"for_each": ["i in item"], "coefficient": "value[i]", "variable": "take[i]"}
    ]
  },
  "constraints": [],
  "constraint_templates": [
    {"id": "capacity", "type": "hard",
     "terms": [{"for_each": ["i in item"], "coefficient": "weight[i]", "variable": "take[i]"}],
     "operator": "<=", "rhs": 10}
  ],
  "solver": {"backend": "simulated_annealing", "seed": 0, "num_reads": 100,
             "penalty_multiplier": 0.01}
}
"""


def test_bqm_retries_reuse_one_prepared_model_of_the_expansion(monkeypatch):
    # With a hundredth of the usual penalty the first SA attempt finds only
    # over-weight subsets (tests/scenarios/test_retry.py), so the service
    # retries with a doubled penalty, compiling from one PreparedBQM.
    prepared = []
    original = BQMCompiler.prepare

    def spy(self, problem):
        prepared.append(problem)
        return original(self, problem)

    monkeypatch.setattr(BQMCompiler, "prepare", spy)
    result = OptimizationService().solve(from_json(KNAPSACK_TEMPLATE))
    assert result.status == "success", result.errors
    assert len(result.attempts) > 1
    assert result.attempts[0].feasible_samples == 0
    assert result.attempts[-1].feasible_samples > 0
    penalties = [attempt.penalty for attempt in result.attempts]
    assert penalties[0] == pytest.approx(0.31)
    for previous, current in zip(penalties, penalties[1:]):
        assert current == pytest.approx(previous * 2.0)
    # Prepared once, from the expanded problem.
    assert len(prepared) == 1
    assert not prepared[0].has_templates()
    assert [c.id for c in prepared[0].all_constraints()] == ["capacity"]
    best = result.solutions[0]
    assert best.objective_value == 17.0
    assert best.variables == {"take[a]": 1, "take[b]": 0, "take[c]": 1, "take[d]": 0}
    assert [e.constraint_id for e in best.constraint_evaluations] == ["capacity"]


# --------------------------------------------------------------------------
# Expansion ceiling, expansion errors and the expansion gate
# --------------------------------------------------------------------------


def test_expansion_limit_is_the_only_error_on_every_path():
    service = OptimizationService(policy=ExecutionPolicy(max_template_bindings=10))
    problem = fixture("assignment_template.json")

    solved = service.solve(problem)
    assert solved.status == "resource_limit_exceeded"
    assert [e.code for e in solved.errors] == ["TEMPLATE_EXPANSION_LIMIT"]
    assert solved.warnings == []
    assert solved.solutions == []
    assert "max_template_bindings" in solved.errors[0].message

    validated = service.validate(problem)
    assert not validated.valid
    assert [e.code for e in validated.errors] == ["TEMPLATE_EXPANSION_LIMIT"]
    assert validated.warnings == []

    recommended = service.recommend(problem)
    assert not recommended.valid
    assert [e.code for e in recommended.errors] == ["TEMPLATE_EXPANSION_LIMIT"]
    assert recommended.recommendations == []

    assert solved.errors == validated.errors == recommended.errors


def test_expansion_error_is_invalid_problem_without_warnings():
    # dist has no row for (a,b) and no default; the pair is used once per
    # tour position, and the four identical errors are reported once.
    data = json.loads(TSP_TEMPLATE_JSON)
    rows = data["parameters"][0]["values"]
    data["parameters"][0]["values"] = [row for row in rows if row["key"] != ["a", "b"]]
    problem = OptimizationProblem.model_validate(data)
    service = OptimizationService()

    solved = service.solve(problem)
    assert solved.status == "invalid_problem"
    assert solved.warnings == []
    assert [(e.code, e.path, e.message) for e in solved.errors] == [
        (
            "PARAMETER_VALUE_MISSING",
            "objective.quadratic_term_templates[0].coefficient",
            "Parameter dist has no value for (a,b) and no default",
        )
    ]

    validated = service.validate(problem)
    assert not validated.valid
    assert validated.warnings == []
    assert validated.errors == solved.errors


def test_full_expansion_gate_refuses_template_documents_only():
    policy = ExecutionPolicy(max_concurrent_solves=2)
    service = OptimizationService(policy=policy)
    templated = tsp_template()
    plain = OptimizationProblem.model_validate(
        json.loads((EXAMPLES_DIR / "assignment.json").read_text(encoding="utf-8"))
    )
    held = 0
    try:
        for _ in range(policy.max_concurrent_solves):
            assert service._expansion_slots.acquire(blocking=False)
            held += 1

        validated = service.validate(templated)
        assert not validated.valid
        assert [e.code for e in validated.errors] == ["CONCURRENCY_LIMIT"]
        assert validated.errors[0].retryable
        assert validated.errors[0].message == (
            "Too many concurrent template expansions: the server allows at most 2"
        )
        assert validated.warnings == []

        recommended = service.recommend(templated)
        assert not recommended.valid
        assert [e.code for e in recommended.errors] == ["CONCURRENCY_LIMIT"]
        assert recommended.recommendations == []

        solved = service.solve(templated)
        assert solved.status == "resource_limit_exceeded"
        assert [e.code for e in solved.errors] == ["CONCURRENCY_LIMIT"]
        assert solved.warnings == []

        # A document without templates never touches the gate.
        assert service.validate(plain).valid
        assert service.recommend(plain).valid
        plain_result = service.solve(plain)
        assert plain_result.status == "success"
        assert plain_result.solutions[0].objective_value == 8.0
    finally:
        for _ in range(held):
            service._expansion_slots.release()

    # Released: template documents are admitted again.
    assert service.validate(templated).valid
    assert service.recommend(templated).valid
    assert service.solve(templated).status == "success"


def test_the_gate_is_released_after_every_path():
    policy = ExecutionPolicy(max_concurrent_solves=1)
    service = OptimizationService(policy=policy)
    problem = tsp_template()
    for _ in range(3):
        assert service.validate(problem).valid
        assert service.recommend(problem).valid
        assert service.solve(problem).status == "success"
    assert service._expansion_slots.acquire(blocking=False)
    service._expansion_slots.release()


# --------------------------------------------------------------------------
# recommend: expansion warnings lead every backend's warnings
# --------------------------------------------------------------------------


def test_recommend_warnings_start_with_the_expansion_warnings():
    problem = fixture("window_template.json")
    expansion_warnings = list(expand_problem(problem).warnings)
    assert [w.code for w in expansion_warnings] == [
        "TEMPLATE_BOUNDARY_SKIPPED",
        "TEMPLATE_BOUNDARY_SKIPPED",
    ]
    result = OptimizationService().recommend(problem)
    assert result.valid
    assert result.recommendations
    for entry in result.recommendations:
        assert entry.warnings[: len(expansion_warnings)] == expansion_warnings, entry.backend
