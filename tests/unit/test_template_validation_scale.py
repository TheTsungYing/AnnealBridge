"""Validation cost of an expanded problem stays linear (schema 1.3 spec §14.1 item 8).

A template turns a few lines into tens of thousands of constraints, so any
per-constraint step of the error pass that touches every variable becomes
O(variables x constraints). The 2026-09-28 batch 8 diff review found one:
``_check_trivially_infeasible`` rebuilt a bounds dict over *all* variables
for each hard constraint, so a 3 KB template document took minutes to
validate. These tests pin the fix: the dict holds the constraint's own
variables only (same entries ``lhs_bounds`` reads, so the verdict is
unchanged), and a large generated problem validates in seconds.
"""

import time

from annealbridge.models import Constraint, LinearTerm, OptimizationProblem
from annealbridge.validation import problem_validator, validate_problem_full


def _hard(names: list[str]) -> Constraint:
    return Constraint(
        id="c",
        type="hard",
        terms=[LinearTerm(variable=name, coefficient=1) for name in names],
        operator="<=",
        rhs=1,
    )


def test_the_range_check_only_reads_the_constraints_own_variables(monkeypatch):
    seen: list[int] = []
    original = problem_validator.lhs_bounds

    def spy(coefficients, bounds=None):
        seen.append(len(bounds))
        return original(coefficients, bounds)

    monkeypatch.setattr(problem_validator, "lhs_bounds", spy)
    safe_bounds = {f"v{i}": (0, 1) for i in range(10_000)}
    errors: list = []
    problem_validator._check_trivially_infeasible(
        _hard(["v1", "v2", "v2"]), "constraints[0]", safe_bounds, errors
    )
    assert seen == [2]
    assert errors == []


def test_the_verdict_is_unchanged_for_an_unknown_variable(monkeypatch):
    # An undeclared name has no entry either way (it carries its own
    # UNKNOWN_VARIABLE error); the range treats it as binary as before.
    errors: list = []
    problem_validator._check_trivially_infeasible(
        Constraint(
            id="c",
            type="hard",
            terms=[LinearTerm(variable="ghost", coefficient=1)],
            operator=">=",
            rhs=2,
        ),
        "constraints[0]",
        {"v": (0, 1)},
        errors,
    )
    assert [error.code for error in errors] == ["TRIVIALLY_INFEASIBLE"]


def test_a_large_generated_problem_validates_in_seconds():
    n = 120  # 14,400 variables and 14,400 generated hard constraints
    problem = OptimizationProblem.model_validate(
        {
            "version": "1.3",
            "name": "scale",
            "index_sets": [
                {"name": "r", "elements": list(range(n))},
                {"name": "c", "elements": list(range(n))},
            ],
            "variable_families": [{"name": "x", "indices": ["r", "c"]}],
            "variables": [],
            "objective": {
                "direction": "minimize",
                "linear_terms": [],
                "linear_term_templates": [
                    {"for_each": ["i in r", "j in c"], "coefficient": 1, "variable": "x[i,j]"}
                ],
            },
            "constraints": [],
            "constraint_templates": [
                {
                    "id": "cell",
                    "type": "hard",
                    "for_each": ["i in r", "j in c"],
                    "terms": [{"coefficient": 1, "variable": "x[i,j]"}],
                    "operator": ">=",
                    "rhs": 0,
                }
            ],
        }
    )
    started = time.perf_counter()
    result = validate_problem_full(problem)
    elapsed = time.perf_counter() - started
    assert result.valid
    # About a second once linear; the quadratic version took minutes.
    assert elapsed < 20, elapsed
