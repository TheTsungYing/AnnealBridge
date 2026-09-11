"""Unit tests for candidate processing and ranking (spec §25, §25.1, §33)."""

import numpy as np
import pytest

from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    Variable,
)
from annealbridge.orchestration import evaluate_objective, process_candidates
from annealbridge.solvers import RawSolverResult
from annealbridge.validation import validate_solution


def make_problem(
    direction: str = "minimize",
    coefficients: dict[str, float] | None = None,
    constraints: list[Constraint] | None = None,
) -> OptimizationProblem:
    coefficients = coefficients if coefficients is not None else {"x": 1.0, "y": 2.0}
    return OptimizationProblem(
        name="ranking-test",
        variables=[Variable(name=name) for name in coefficients],
        objective=Objective(
            direction=direction,
            linear_terms=[
                LinearTerm(variable=name, coefficient=value)
                for name, value in coefficients.items()
            ],
        ),
        constraints=constraints or [],
    )


def raw(samples: list[dict[str, int]], energies: list[float]) -> RawSolverResult:
    return RawSolverResult.from_dicts(samples, energies, "exact")


# --------------------------------------------------------------------------
# helpers for the "top-k equals the full sort" comparisons
# --------------------------------------------------------------------------

# Deliberately not in alphabetical order and mixing cases, so the name-sorted
# tie-break tuple is *not* the column order of the sample matrix.
_WIDE_NAMES = [
    "v13", "a", "zz", "m2", "q", "b7", "kk", "c", "x1", "n",
    "Z9", "p3", "d", "yy", "e2", "w", "g10", "h", "t4", "s",
]


def wide_problem(direction: str) -> OptimizationProblem:
    """20 binary variables of which only three carry an objective coefficient.

    With eight reachable objective values and a single soft constraint,
    thousands of candidates share a ranking score *and* an objective value,
    so the last tie-break -- the name-sorted assignment tuple -- is what
    actually decides the order. The hard constraint is loose enough that
    almost every candidate survives the feasibility filter.
    """
    return OptimizationProblem(
        name="wide-ranking",
        variables=[Variable(name=name) for name in _WIDE_NAMES],
        objective=Objective(
            direction=direction,
            linear_terms=[
                LinearTerm(variable="a", coefficient=1.0),
                LinearTerm(variable="zz", coefficient=2.0),
                LinearTerm(variable="m2", coefficient=4.0),
            ],
        ),
        constraints=[
            # Not one of the objective's variables: two candidates can share
            # an objective value and still rank apart.
            Constraint(
                id="prefer_q",
                type="soft",
                terms=[LinearTerm(variable="q", coefficient=1)],
                operator=">=",
                rhs=1,
                weight=0.5,
            ),
            Constraint(
                id="budget",
                type="hard",
                terms=[
                    LinearTerm(variable=name, coefficient=1) for name in _WIDE_NAMES
                ],
                operator="<=",
                rhs=15,
            ),
        ],
    )


def bit_matrix(values: list[int], n_variables: int) -> np.ndarray:
    """Big-endian bit expansion of ``values`` -- one 0/1 row each."""
    matrix = np.zeros((len(values), n_variables), dtype=np.int8)
    for row, value in enumerate(values):
        for column in range(n_variables):
            matrix[row, column] = (value >> (n_variables - 1 - column)) & 1
    return matrix


def distinct_rows_with_neighbours(
    rng: np.random.Generator, n_rows: int, n_variables: int
) -> np.ndarray:
    """``n_rows`` pairwise-distinct 0/1 rows, including one-bit neighbours.

    Rows differing in a single position are where an order decided by the
    assignment tuple is most fragile, so a good share of the rows are such
    neighbours of earlier ones rather than independent draws.
    """
    values: dict[int, None] = {}
    target = (n_rows * 2) // 3
    while len(values) < target:
        for value in rng.integers(
            0, 1 << n_variables, size=n_rows, dtype=np.int64
        ).tolist():
            values.setdefault(value, None)
            if len(values) >= target:
                break
    base = list(values)
    index = 0
    while len(values) < n_rows and index < 100 * n_rows:
        value = base[index % len(base)]
        values.setdefault(value ^ (1 << int(rng.integers(0, n_variables))), None)
        index += 1
    while len(values) < n_rows:  # pragma: no cover - only if flips collided
        values.setdefault(int(rng.integers(0, 1 << n_variables)), None)
    return bit_matrix(list(values)[:n_rows], n_variables)


def reference_ranking(
    problem: OptimizationProblem, matrix: np.ndarray, names: list[str]
) -> list[dict[str, int]]:
    """Rank every row with the row-by-row §25.1 rules, without any shortlist.

    The oracle for the top-k: score each candidate on its own, sort the
    whole feasible set, and return the assignments in order.
    """
    minimize = problem.objective.direction == "minimize"
    sign = 1.0 if minimize else -1.0
    scored: list[tuple[dict[str, int], float, float]] = []
    for row in range(matrix.shape[0]):
        sample = dict(zip(names, matrix[row].tolist()))
        validation = validate_solution(problem, sample)
        if not validation.feasible:
            continue
        objective_value = evaluate_objective(problem.objective, sample)
        score = (
            objective_value + validation.soft_violation_score
            if minimize
            else objective_value - validation.soft_violation_score
        )
        scored.append((sample, objective_value, score))
    scored.sort(
        key=lambda item: (
            sign * item[2],
            sign * item[1],
            tuple(item[0][name] for name in sorted(item[0])),
        )
    )
    return [sample for sample, _, _ in scored]


class TestDeduplication:
    def test_same_business_solution_kept_once_with_min_energy(self):
        problem = make_problem()
        # Two samples differ only in the internal slack bit -> one business
        # solution; the kept energy must be the minimum of the duplicates.
        result = raw(
            [
                {"x": 1, "y": 0, "__slack_c_0": 0},
                {"x": 1, "y": 0, "__slack_c_0": 1},
                {"x": 0, "y": 1, "__slack_c_0": 0},
            ],
            [5.0, 3.0, 4.0],
        )
        processed = process_candidates(problem, result, {"__slack_c_0"}, top_k=10)
        solutions = processed.solutions

        assert processed.unique_samples == 2
        assert processed.feasible_samples == 2
        assert len(solutions) == 2
        by_vars = {tuple(sorted(s.variables.items())): s for s in solutions}
        merged = by_vars[(("x", 1), ("y", 0))]
        assert merged.energy == 3.0
        # Both slack values collapsed into this one business assignment.
        assert merged.sample_count == 2
        assert by_vars[(("x", 0), ("y", 1))].sample_count == 1

    def test_internal_variables_never_appear_in_solutions(self):
        problem = make_problem()
        result = raw([{"x": 1, "y": 1, "__slack_c_0": 1}], [0.0])
        solutions = process_candidates(
            problem, result, {"__slack_c_0"}, top_k=5
        ).solutions
        assert solutions[0].variables == {"x": 1, "y": 1}


class TestSoftViolationRanking:
    def test_soft_violation_can_outweigh_better_objective(self):
        # maximize 5a + 4b with soft "a <= 0" (weight 2): a=1 scores
        # 5 - 2 = 3, b=1 scores 4 - 0 = 4, so b=1 must rank first even
        # though its raw objective is lower.
        soft = Constraint(
            id="avoid_a",
            type="soft",
            terms=[LinearTerm(variable="a", coefficient=1)],
            operator="<=",
            rhs=0,
            weight=2.0,
        )
        problem = make_problem(
            direction="maximize",
            coefficients={"a": 5.0, "b": 4.0},
            constraints=[soft],
        )
        result = raw([{"a": 1, "b": 0}, {"a": 0, "b": 1}], [-5.0, -4.0])

        processed = process_candidates(problem, result, set(), top_k=5)
        solutions = processed.solutions

        assert processed.feasible_samples == 2
        assert solutions[0].variables == {"a": 0, "b": 1}
        assert solutions[0].ranking_score == pytest.approx(4.0)
        assert solutions[0].soft_violation_score == pytest.approx(0.0)
        assert solutions[1].variables == {"a": 1, "b": 0}
        assert solutions[1].ranking_score == pytest.approx(3.0)
        assert solutions[1].soft_violation_score == pytest.approx(2.0)
        assert solutions[1].objective_value == pytest.approx(5.0)
        assert [s.rank for s in solutions] == [1, 2]


class TestTieBreak:
    def test_equal_scores_break_by_assignment_tuple(self):
        # minimize a + b: both single-selection solutions score 1.0; the
        # name-sorted assignment tuple (0, 1) < (1, 0) puts a=0,b=1 first.
        problem = make_problem(coefficients={"a": 1.0, "b": 1.0})
        samples = [{"a": 1, "b": 0}, {"a": 0, "b": 1}]
        energies = [1.0, 1.0]

        forward = process_candidates(
            problem, raw(samples, energies), set(), top_k=5
        ).solutions
        reverse = process_candidates(
            problem, raw(samples[::-1], energies[::-1]), set(), top_k=5
        ).solutions

        expected_order = [{"a": 0, "b": 1}, {"a": 1, "b": 0}]
        assert [s.variables for s in forward] == expected_order
        assert [s.variables for s in reverse] == expected_order

    def test_tie_break_prefers_objective_before_assignment(self):
        soft = Constraint(
            id="want_x",
            type="soft",
            terms=[LinearTerm(variable="x", coefficient=1)],
            operator=">=",
            rhs=1,
            weight=1.0,
        )
        problem = make_problem(
            direction="minimize",
            coefficients={"x": 2.0, "y": 1.0},
            constraints=[soft],
        )
        # x=1,y=0: obj 2, soft 0 -> score 2; x=0,y=1: obj 1, soft 1 -> score 2.
        # Equal ranking_score, so the lower objective (minimize) ranks first.
        result = raw([{"x": 1, "y": 0}, {"x": 0, "y": 1}], [0.0, 0.0])
        solutions = process_candidates(problem, result, set(), top_k=5).solutions

        assert solutions[0].ranking_score == pytest.approx(2.0)
        assert solutions[1].ranking_score == pytest.approx(2.0)
        assert solutions[0].variables == {"x": 0, "y": 1}
        assert solutions[0].objective_value == pytest.approx(1.0)


class TestFeasibilityFilterAndTopK:
    def test_infeasible_candidates_are_dropped(self):
        hard = Constraint(
            id="pick_one",
            type="hard",
            terms=[
                LinearTerm(variable="x", coefficient=1),
                LinearTerm(variable="y", coefficient=1),
            ],
            operator="==",
            rhs=1,
        )
        problem = make_problem(constraints=[hard])
        result = raw(
            [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
            [0.0, 1.0, 3.0],
        )
        processed = process_candidates(problem, result, set(), top_k=5)
        solutions = processed.solutions

        assert processed.unique_samples == 3
        assert processed.feasible_samples == 1
        assert len(solutions) == 1
        assert solutions[0].variables == {"x": 1, "y": 0}
        assert solutions[0].hard_constraints_satisfied is True

    def test_all_infeasible_is_diagnosed(self):
        # x + y >= 1 and x + y <= 0 cannot both hold. {x:1,y:0} misses the
        # second by 1 and {x:0,y:0} misses the first by 1, so the smallest
        # total is a tie and the first candidate read wins it.
        terms = [
            LinearTerm(variable="x", coefficient=1),
            LinearTerm(variable="y", coefficient=1),
        ]
        problem = make_problem(
            constraints=[
                Constraint(
                    id="at_least_one", type="hard", terms=terms, operator=">=", rhs=1
                ),
                Constraint(
                    id="none_at_all", type="hard", terms=terms, operator="<=", rhs=0
                ),
            ]
        )
        result = raw(
            [{"x": 1, "y": 0}, {"x": 0, "y": 0}, {"x": 1, "y": 1}],
            [0.0, -9.0, 1.0],
        )
        processed = process_candidates(problem, result, set(), top_k=5)

        assert processed.solutions == []
        assert processed.unique_samples == 3
        assert processed.feasible_samples == 0

        diagnostics = processed.infeasibility
        assert diagnostics is not None
        # First-seen, not lowest energy: {x:0,y:0} is the cheaper row.
        assert diagnostics.closest_candidate.variables == {"x": 1, "y": 0}
        assert diagnostics.closest_candidate.hard_violation_total == pytest.approx(1.0)
        assert [
            (rate.constraint_id, rate.violated_candidates, rate.candidates)
            for rate in diagnostics.hard_violation_rates
        ] == [("at_least_one", 1, 3), ("none_at_all", 2, 3)]
        assert diagnostics.hard_violation_rates[0].violated_fraction == pytest.approx(
            1 / 3
        )

    def test_top_k_limits_and_ranks_from_one(self):
        problem = make_problem(coefficients={"x": 1.0, "y": 2.0})
        result = raw(
            [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}, {"x": 1, "y": 1}],
            [0.0, 1.0, 2.0, 3.0],
        )
        processed = process_candidates(problem, result, set(), top_k=2)
        solutions = processed.solutions

        assert processed.unique_samples == 4
        assert processed.feasible_samples == 4
        assert [s.rank for s in solutions] == [1, 2]
        assert solutions[0].variables == {"x": 0, "y": 0}
        assert solutions[1].variables == {"x": 1, "y": 0}


class TestTopKMatchesFullSort:
    """§25.1 on thousands of candidates: whatever the pipeline does to avoid
    ordering the whole set, the top-k it returns must be the prefix of the
    complete row-by-row ranking -- same assignments, same order."""

    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    @pytest.mark.parametrize("top_k", [1, 5, 10_000])
    @pytest.mark.parametrize("seed", range(3))
    def test_top_k_is_the_prefix_of_the_full_ranking(self, direction, top_k, seed):
        rng = np.random.default_rng(4242 + seed)
        n_variables = len(_WIDE_NAMES)
        matrix = distinct_rows_with_neighbours(rng, 3000, n_variables)
        problem = wide_problem(direction)
        # Energy is deliberately unrelated to the ranking: it must not leak
        # into the order at any point (overview principle 2).
        energies = rng.normal(size=matrix.shape[0])
        result = RawSolverResult(
            variables=_WIDE_NAMES,
            samples=matrix,
            energies=energies,
            backend="exact",
        )

        processed = process_candidates(problem, result, set(), top_k=top_k)
        expected = reference_ranking(problem, matrix, _WIDE_NAMES)

        # The rows are pairwise distinct, so deduplication keeps them all.
        assert processed.unique_samples == matrix.shape[0]
        assert processed.feasible_samples == len(expected)
        assert 0 < len(expected) < matrix.shape[0]  # the filter did fire
        assert [s.variables for s in processed.solutions] == expected[:top_k]
        assert [s.rank for s in processed.solutions] == list(
            range(1, len(expected[:top_k]) + 1)
        )

    @pytest.mark.parametrize("top_k", [1, 5, 250, 256, 257])
    def test_shortlist_keeps_every_candidate_tied_at_the_kth_place(self, top_k):
        # 320 candidates over ten variables, only one of which is in the
        # objective: the 256 candidates with a=0 all share ranking_score 0.0
        # *and* objective_value 0.0, so every cut-off tested here falls
        # inside one block that only the assignment tuple separates.
        names = ["m", "a", "q9", "B", "c2", "zz", "d", "e", "f", "g"]
        rows = 320
        problem = OptimizationProblem(
            name="tie-block",
            variables=[Variable(name=name) for name in names],
            objective=Objective(
                direction="minimize",
                linear_terms=[LinearTerm(variable="a", coefficient=1.0)],
            ),
            constraints=[],
        )
        matrix = bit_matrix(list(range(rows)), len(names))
        result = RawSolverResult(
            variables=names,
            samples=matrix,
            # Descending energies, so a shortlist that fell back on energy
            # would return the rows in the opposite order.
            energies=-np.arange(rows, dtype=np.float64),
            backend="exact",
        )

        processed = process_candidates(problem, result, set(), top_k=top_k)
        expected = reference_ranking(problem, matrix, names)

        assert processed.feasible_samples == rows
        # The tie block really is wider than the cut-offs under test.
        assert sum(1 for sample in expected if sample["a"] == 0) == 256
        assert [s.variables for s in processed.solutions] == expected[:top_k]
        assert [s.rank for s in processed.solutions] == list(
            range(1, len(expected[:top_k]) + 1)
        )
