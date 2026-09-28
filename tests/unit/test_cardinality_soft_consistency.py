"""Soft cardinality constraints cost what re-validation says (principle 6).

Schema 1.2 spec 2026-09-25 §7.2 and §13 item 6. A soft cardinality
constraint never takes the pairwise encoding (``w·C(k, 2)`` would under-charge
``k >= 3``); it is lowered to a linear constraint and paid through the slack
encoding on the BQM path and as a native soft constraint on the CQM path.
Enumerated over every operator, every ``rhs`` from ``-1`` to ``n + 1`` (so
the always-violated and the redundant cases are included) and ``n = 1..4``:

* BQM: for each business assignment, the model energy minimised over the
  slack bits, minus ``sign · objective``, equals
  ``validate_solution(...).soft_violation_score``;
* CQM: every row ``dimod.ExactCQMSolver`` enumerates (the model has no
  slack for an all-binary soft constraint) has ``energy − sign · objective``
  equal to the same score.
"""

import itertools

import dimod
import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration.candidates import evaluate_objective
from annealbridge.validation import validate_problem, validate_solution
from annealbridge.validation.estimates import (
    compute_penalty_scale,
    uses_pairwise_penalty,
)

WEIGHT = 2.5
OBJECTIVE_COEFFICIENTS = [3.0, -2.0, 0.5, -1.25]


def soft_problem(count: int, operator: str, rhs: int, direction: str) -> OptimizationProblem:
    names = [f"x{index}" for index in range(count)]
    return OptimizationProblem.model_validate(
        {
            "version": "1.2",
            "name": "soft cardinality consistency",
            "variables": [{"name": name} for name in names],
            "objective": {
                "direction": direction,
                "linear_terms": [
                    {"variable": name, "coefficient": coefficient}
                    for name, coefficient in zip(names, OBJECTIVE_COEFFICIENTS)
                ],
            },
            "constraints": [],
            "cardinality_constraints": [
                {
                    "id": "pref",
                    "type": "soft",
                    "weight": WEIGHT,
                    "variables": names,
                    "operator": operator,
                    "rhs": rhs,
                }
            ],
        }
    )


CASES = [
    pytest.param(count, operator, rhs, direction, id=f"n{count}-{operator}{rhs}-{direction[:3]}")
    for count in range(1, 5)
    for operator in ("==", "<=", ">=")
    for rhs in range(-1, count + 2)
    for direction in ("minimize", "maximize")
]


def sign(problem: OptimizationProblem) -> float:
    return 1.0 if problem.objective.direction == "minimize" else -1.0


def expected_score(operator: str, rhs: int, chosen: int) -> float:
    """Independent of the code under test: ``w · violation²`` from the count."""
    if operator == "==":
        violation = abs(chosen - rhs)
    elif operator == "<=":
        violation = max(0, chosen - rhs)
    else:
        violation = max(0, rhs - chosen)
    return WEIGHT * violation * violation


@pytest.mark.parametrize("count, operator, rhs, direction", CASES)
def test_bqm_minimum_over_slack_is_the_soft_score(count, operator, rhs, direction):
    problem = soft_problem(count, operator, rhs, direction)
    assert validate_problem(problem) == []
    (lowered,) = problem.all_constraints()
    assert not uses_pairwise_penalty(lowered)

    compiled = BQMCompiler().compile(problem, 2.0 * compute_penalty_scale(problem))
    names = [variable.name for variable in problem.variables]
    slack_bits = sorted(compiled.internal_variables)
    assert set(str(v) for v in compiled.model.variables) == set(names) | set(slack_bits)

    for values in itertools.product((0, 1), repeat=count):
        sample = dict(zip(names, values))
        minimum = min(
            compiled.model.energy({**sample, **dict(zip(slack_bits, bits))})
            for bits in itertools.product((0, 1), repeat=len(slack_bits))
        )
        objective = evaluate_objective(problem.objective, sample)
        score = validate_solution(problem, sample).soft_violation_score
        assert score == expected_score(operator, rhs, sum(values))
        assert minimum - sign(problem) * objective == pytest.approx(score, abs=1e-9), sample


@pytest.mark.parametrize("count, operator, rhs, direction", CASES)
def test_cqm_row_energy_is_the_soft_score(count, operator, rhs, direction):
    problem = soft_problem(count, operator, rhs, direction)
    compiled = CQMCompiler().compile(problem, None)
    assert compiled.internal_variables == set()

    sampleset = dimod.ExactCQMSolver().sample_cqm(compiled.model)
    columns = [str(variable) for variable in sampleset.variables]
    names = [variable.name for variable in problem.variables]
    assert sorted(columns) == sorted(names)
    rows = 0
    # The record, never samples(): that one sorts by energy (see
    # test_cqm_compiler_integer.min_energy_per_assignment).
    for row, energy in zip(sampleset.record.sample, sampleset.record.energy):
        sample = {name: int(row[columns.index(name)]) for name in names}
        objective = evaluate_objective(problem.objective, sample)
        score = validate_solution(problem, sample).soft_violation_score
        assert float(energy) - sign(problem) * objective == pytest.approx(score, abs=1e-9), sample
        rows += 1
    assert rows == 2**count


def test_the_corpus_covers_out_of_range_and_redundant_cases():
    """rhs below 0, above n, and a redundant ``<= n`` / ``>= 0`` are all in."""
    covered = {(count, operator, rhs) for count, operator, rhs, _ in (c.values for c in CASES)}
    assert (1, "<=", -1) in covered
    assert (4, ">=", 5) in covered
    assert (3, "==", 4) in covered
    assert (4, "<=", 4) in covered
    assert (2, ">=", 0) in covered
    assert len(CASES) == 2 * 3 * sum(count + 3 for count in range(1, 5))
