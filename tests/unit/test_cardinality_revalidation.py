"""Re-validation judges cardinality constraints on the original problem.

Schema 1.2 spec 2026-09-25 §9 and §13 item 7. ``validate_solution`` and
``validate_batch`` read ``all_constraints()``, so every cardinality
constraint produces one ``ConstraintEvaluation`` after the linear ones
(spec §5.3): the declared id, type and operator, the chosen count as
``actual_value``, the rhs as ``expected_value`` and, for a soft one,
``weight × violation²``. The same order runs through the infeasibility
diagnosis of a solve and through post-processing.
"""

import itertools

import numpy as np
import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.validation import validate_batch, validate_problem, validate_solution
from tests.unit.test_postprocess import (
    ON,
    assert_revalidated,
    brute_force_best,
    solve_fixed,
)


def lin(variable: str, coefficient: float) -> dict:
    return {"variable": variable, "coefficient": coefficient}


def mixed_problem() -> OptimizationProblem:
    """Linear hard and soft constraints, then four cardinality constraints."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.2",
            "name": "cardinality re-validation",
            "variables": [
                *({"name": name} for name in "abcde"),
                {"name": "k", "type": "integer", "lower_bound": 0, "upper_bound": 3},
            ],
            "objective": {
                "direction": "minimize",
                "linear_terms": [lin("a", 1), lin("b", 2), lin("k", -1)],
            },
            "constraints": [
                {
                    "id": "lin_hard",
                    "type": "hard",
                    "terms": [lin("a", 1), lin("k", 2)],
                    "operator": "<=",
                    "rhs": 5,
                },
                {
                    "id": "lin_soft",
                    "type": "soft",
                    "weight": 1.5,
                    "terms": [lin("b", 1), lin("k", 1)],
                    "operator": ">=",
                    "rhs": 2,
                },
            ],
            "cardinality_constraints": [
                {
                    "id": "card_one",
                    "type": "hard",
                    "variables": ["a", "b", "c"],
                    "operator": "==",
                    "rhs": 1,
                },
                {
                    "id": "card_amo",
                    "type": "hard",
                    "variables": ["c", "d", "e"],
                    "operator": "<=",
                    "rhs": 1,
                },
                {
                    "id": "card_soft",
                    "type": "soft",
                    "weight": 3,
                    "variables": ["a", "d", "e"],
                    "operator": ">=",
                    "rhs": 2,
                },
                {
                    "id": "card_soft_none",
                    "type": "soft",
                    "weight": 0.5,
                    "variables": ["b", "c"],
                    "operator": "<=",
                    "rhs": 0,
                },
            ],
        }
    )


ALL_IDS = ["lin_hard", "lin_soft", "card_one", "card_amo", "card_soft", "card_soft_none"]
HARD_IDS = ["lin_hard", "card_one", "card_amo"]


def sample(**values: int) -> dict[str, int]:
    return {name: values.get(name, 0) for name in [*"abcde", "k"]}


class TestValidateSolution:
    def test_problem_is_valid(self):
        assert validate_problem(mixed_problem()) == []

    def test_cardinality_rows_follow_the_linear_rows(self):
        result = validate_solution(mixed_problem(), sample(a=1, k=1))
        assert [e.constraint_id for e in result.evaluations] == ALL_IDS
        assert [e.constraint_type for e in result.evaluations] == [
            "hard",
            "soft",
            "hard",
            "hard",
            "soft",
            "soft",
        ]
        assert [e.operator for e in result.evaluations] == ["<=", ">=", "==", "<=", ">=", "<="]

    def test_violating_a_cardinality_constraint_is_infeasible(self):
        # Two of a, b, c chosen: card_one wants exactly one.
        result = validate_solution(mixed_problem(), sample(a=1, b=1))
        assert result.feasible is False
        assert [e.constraint_id for e in result.hard_violations] == ["card_one"]
        card_one = result.evaluations[2]
        assert card_one.satisfied is False
        assert card_one.actual_value == 2.0
        assert card_one.expected_value == 1.0
        assert card_one.violation_amount == 1.0
        assert card_one.weighted_penalty is None

    def test_the_at_most_one_is_judged_on_the_count(self):
        result = validate_solution(mixed_problem(), sample(a=1, c=0, d=1, e=1))
        assert result.feasible is False
        card_amo = result.evaluations[3]
        assert (card_amo.constraint_id, card_amo.actual_value, card_amo.expected_value) == (
            "card_amo",
            2.0,
            1.0,
        )
        assert card_amo.violation_amount == 1.0

    def test_soft_cardinality_pays_weight_times_violation_squared(self):
        # card_soft counts a, d, e = 1 of the >= 2 wanted (violation 1);
        # card_soft_none counts b, c = 1 of the <= 0 allowed (violation 1).
        result = validate_solution(mixed_problem(), sample(b=1, d=1, k=1))
        assert result.feasible is True
        card_soft, card_soft_none = result.evaluations[4], result.evaluations[5]
        assert card_soft.actual_value == 1.0
        assert card_soft.expected_value == 2.0
        assert card_soft.violation_amount == 1.0
        assert card_soft.weighted_penalty == 3.0 * 1.0**2
        assert card_soft_none.actual_value == 1.0
        assert card_soft_none.weighted_penalty == 0.5 * 1.0**2
        # lin_soft: b + k = 2 >= 2 holds.
        assert result.soft_violation_score == 3.0 + 0.5

    def test_a_larger_violation_is_squared(self):
        # card_soft: none of a, d, e chosen, 2 short; card_soft_none: b, c
        # both chosen, 2 over.
        result = validate_solution(mixed_problem(), sample(b=1, c=1, k=1))
        assert result.evaluations[4].weighted_penalty == 3.0 * 2.0**2
        assert result.evaluations[5].weighted_penalty == 0.5 * 2.0**2

    def test_evaluations_are_rebuilt_from_the_current_declaration(self):
        problem = mixed_problem()
        declaration = problem.cardinality_constraints[0]
        changed = problem.model_copy(
            update={
                "cardinality_constraints": [
                    declaration.model_copy(update={"rhs": 2}),
                    *problem.cardinality_constraints[1:],
                ]
            }
        )
        assert validate_solution(problem, sample(a=1, b=1)).feasible is False
        assert validate_solution(changed, sample(a=1, b=1)).feasible is True
        assert validate_solution(changed, sample(a=1, b=1)).evaluations[2].expected_value == 2.0


class TestValidateBatch:
    def test_batch_equals_the_scalar_kernel_bit_for_bit(self):
        problem = mixed_problem()
        names = [variable.name for variable in problem.variables]
        rows = [
            dict(zip(names, (*bits, k)))
            for bits in itertools.product((0, 1), repeat=5)
            for k in range(4)
        ]
        matrix = np.array([[row[name] for name in names] for row in rows], dtype=np.int64)
        batch = validate_batch(problem, names, matrix)

        assert batch.hard_constraint_ids == HARD_IDS
        scalar = [validate_solution(problem, row) for row in rows]
        assert batch.feasible.tolist() == [result.feasible for result in scalar]
        assert batch.soft_violation_score.tolist() == [
            result.soft_violation_score for result in scalar
        ]
        totals = []
        for result in scalar:
            total = 0.0
            for evaluation in result.evaluations:
                if evaluation.constraint_type == "hard":
                    total += evaluation.violation_amount
            totals.append(total)
        assert batch.hard_violation_total.tolist() == totals
        assert batch.hard_violated_counts.tolist() == [
            sum(
                1
                for result in scalar
                if not next(e for e in result.evaluations if e.constraint_id == hard).satisfied
            )
            for hard in HARD_IDS
        ]
        # The corpus exercises every verdict.
        assert any(batch.feasible) and not all(batch.feasible)
        assert all(count > 0 for count in batch.hard_violated_counts.tolist())


# --------------------------------------------------------------------------
# Infeasibility diagnosis through the service
# --------------------------------------------------------------------------


def jointly_infeasible_problem() -> OptimizationProblem:
    """Exactly one and exactly two of the same three: never both.

    Each constraint alone passes the TRIVIALLY_INFEASIBLE check (the count
    ranges over [0, 3]), so only re-validation can tell.
    """
    return OptimizationProblem.model_validate(
        {
            "version": "1.2",
            "name": "cardinality infeasible",
            "variables": [{"name": name} for name in ("x", "y", "z")],
            "objective": {"direction": "minimize", "linear_terms": [lin("x", 1)]},
            "constraints": [
                {
                    "id": "some",
                    "type": "hard",
                    "terms": [lin("x", 1), lin("y", 1), lin("z", 1)],
                    "operator": ">=",
                    "rhs": 1,
                }
            ],
            "cardinality_constraints": [
                {
                    "id": "exactly_one",
                    "type": "hard",
                    "variables": ["x", "y", "z"],
                    "operator": "==",
                    "rhs": 1,
                },
                {
                    "id": "exactly_two",
                    "type": "hard",
                    "variables": ["x", "y", "z"],
                    "operator": "==",
                    "rhs": 2,
                },
            ],
            "solver": {"backend": "exact"},
        }
    )


class TestInfeasibleDiagnosis:
    def test_each_constraint_alone_passes_the_validator(self):
        problem = jointly_infeasible_problem()
        assert validate_problem(problem) == []
        for keep in range(2):
            alone = problem.model_copy(
                update={"cardinality_constraints": [problem.cardinality_constraints[keep]]}
            )
            assert validate_problem(alone) == []

    @pytest.fixture
    def result(self):
        return OptimizationService().solve(jointly_infeasible_problem())

    def test_the_exhaustive_solve_proves_infeasibility(self, result):
        assert result.status == "infeasible"
        assert result.infeasibility_proven is True
        assert result.solutions == []

    def test_violation_rates_cover_the_cardinality_constraints_in_order(self, result):
        rates = result.infeasibility.hard_violation_rates
        assert [rate.constraint_id for rate in rates] == ["some", "exactly_one", "exactly_two"]
        by_id = {rate.constraint_id: rate for rate in rates}
        candidates = by_id["some"].candidates
        assert candidates == 8
        # Of the eight assignments, the count is 1 for three and 2 for three.
        assert by_id["some"].violated_candidates == 1
        assert by_id["exactly_one"].violated_candidates == 5
        assert by_id["exactly_two"].violated_candidates == 5

    def test_the_closest_candidate_reports_the_cardinality_rows(self, result):
        closest = result.infeasibility.closest_candidate
        assert [e.constraint_id for e in closest.constraint_evaluations] == [
            "some",
            "exactly_one",
            "exactly_two",
        ]
        assert closest.hard_violation_total == 1.0
        actual = closest.constraint_evaluations[1].actual_value
        assert actual in (1.0, 2.0)
        assert validate_solution(jointly_infeasible_problem(), closest.variables).feasible is False


# --------------------------------------------------------------------------
# Post-processing (spec §8.3)
# --------------------------------------------------------------------------

WORKERS = ("w0", "w1", "w2")


def staffing_problem() -> OptimizationProblem:
    """Two tasks, three workers: one worker per task, each worker at most once.

    Every rule is a cardinality constraint; the costs make w0 the best
    choice for both tasks, which the at-most-one forbids.
    """
    costs = {"t1": (1, 2, 5), "t2": (1, 3, 5)}
    return OptimizationProblem.model_validate(
        {
            "version": "1.2",
            "name": "cardinality postprocess",
            "variables": [{"name": f"{task}_{w}"} for task in costs for w in WORKERS],
            "objective": {
                "direction": "minimize",
                "linear_terms": [
                    lin(f"{task}_{w}", cost)
                    for task, row in costs.items()
                    for w, cost in zip(WORKERS, row)
                ],
            },
            "constraints": [],
            "cardinality_constraints": [
                *(
                    {
                        "id": f"{task}_staffed",
                        "type": "hard",
                        "variables": [f"{task}_{w}" for w in WORKERS],
                        "operator": "==",
                        "rhs": 1,
                    }
                    for task in costs
                ),
                *(
                    {
                        "id": f"{w}_once",
                        "type": "hard",
                        "variables": [f"{task}_{w}" for task in costs],
                        "operator": "<=",
                        "rhs": 1,
                    }
                    for w in WORKERS
                ),
            ],
        }
    )


STAFFING_IDS = ["t1_staffed", "t2_staffed", "w0_once", "w1_once", "w2_once"]


class TestPostprocess:
    def test_infeasible_samples_are_repaired_against_the_cardinality_rules(self):
        problem = staffing_problem()
        assert validate_problem(problem) == []
        nobody = {variable.name: 0 for variable in problem.variables}
        both_on_w0 = {**nobody, "t1_w0": 1, "t2_w0": 1}
        assert validate_solution(problem, nobody).feasible is False
        assert validate_solution(problem, both_on_w0).feasible is False

        _, result = solve_fixed(problem, [nobody, both_on_w0], postprocess=ON)

        assert result.status == "success", result.errors
        assert_revalidated(problem, result)
        assert result.solutions
        assert all(solution.source != "solver" for solution in result.solutions)
        for solution in result.solutions:
            assert [e.constraint_id for e in solution.constraint_evaluations] == STAFFING_IDS
            assert all(e.satisfied for e in solution.constraint_evaluations)
        # Best: w0 for one task, w1 for the other (1 + 3 or 2 + 1).
        assert result.solutions[0].ranking_score == brute_force_best(problem) == 3.0
        attempt = result.attempts[0]
        assert attempt.postprocess is not None
        assert attempt.postprocess.repair_attempted == 2
        assert attempt.postprocess.repair_succeeded >= 1
