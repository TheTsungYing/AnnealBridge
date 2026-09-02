"""Array-based candidate processing: raw results, deduplication and the
two-layer validation (spec §21, §23, §25; overview principle 2).

The fast paths must be *semantically identical* to the row-by-row
reference, not merely close: every candidate is still re-validated against
the original problem, and the feasibility / soft-violation / objective
numbers feeding the ranking are the very same floats the full validator
reports for the top-k.
"""

import itertools
import random

import numpy as np
import pytest

from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    QuadraticTerm,
    Variable,
)
from annealbridge.orchestration import (
    deduplicate_samples,
    evaluate_objective,
    evaluate_objective_batch,
    process_candidates,
)
from annealbridge.orchestration.optimizer import _lexsort, _pack_rows, _words
from annealbridge.solvers import RawSolverResult
from annealbridge.validation import validate_batch, validate_solution

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def reference_deduplicate(
    raw: RawSolverResult, internal_variables: set[str]
) -> list[tuple[dict[str, int], float]]:
    """The original row-by-row §25 deduplication, kept as the oracle."""
    best: dict[tuple[int, ...], tuple[dict[str, int], float]] = {}
    for sample, energy in zip(raw.as_dicts(), raw.energies.tolist()):
        business = {
            name: value for name, value in sample.items() if name not in internal_variables
        }
        key = tuple(business[name] for name in sorted(business))
        kept = best.get(key)
        if kept is None or energy < kept[1]:
            best[key] = (business, energy)
    return list(best.values())


def random_problem(rng: random.Random, n_variables: int) -> OptimizationProblem:
    """A random linear problem with awkward (non-integer) coefficients.

    Fractions such as 0.1 and 0.3 do not sum exactly in binary, so any
    difference in summation order or association between the two paths
    would show up as a last-bit mismatch.
    """
    names = [f"v{index}" for index in range(n_variables)]
    coefficient_pool = [0.1, 0.2, 0.3, 0.7, 1.0, 1.5, 2.25, -0.1, -0.3, -1.0, 3.0, 7.0]

    def terms(count: int) -> list[LinearTerm]:
        chosen = rng.sample(names, k=count)
        return [
            LinearTerm(variable=name, coefficient=rng.choice(coefficient_pool))
            for name in chosen
        ]

    constraints: list[Constraint] = []
    for index in range(rng.randint(1, 4)):
        operator = rng.choice(["==", "<=", ">="])
        kind = rng.choice(["hard", "soft", "soft"])
        constraint_terms = terms(rng.randint(1, n_variables))
        # Pick rhs so that some assignments hit it exactly (== is reachable)
        # and others miss on both sides.
        lhs_values = [
            sum(t.coefficient * bit for t, bit in zip(constraint_terms, bits))
            for bits in itertools.product((0, 1), repeat=len(constraint_terms))
        ]
        rhs = rng.choice(lhs_values) if rng.random() < 0.7 else rng.choice(
            coefficient_pool
        )
        constraints.append(
            Constraint(
                id=f"c{index}",
                type=kind,
                terms=constraint_terms,
                operator=operator,
                rhs=rhs,
                weight=rng.choice([0.5, 1.0, 1.3, 2.0]) if kind == "soft" else None,
            )
        )

    quadratic: list[QuadraticTerm] = []
    if n_variables >= 2 and rng.random() < 0.6:
        for _ in range(rng.randint(1, 3)):
            first, second = rng.sample(names, k=2)
            quadratic.append(
                QuadraticTerm(
                    variable1=first,
                    variable2=second,
                    coefficient=rng.choice(coefficient_pool),
                )
            )
    return OptimizationProblem(
        name="random",
        variables=[Variable(name=name) for name in names],
        objective=Objective(
            direction=rng.choice(["minimize", "maximize"]),
            linear_terms=terms(rng.randint(1, n_variables)),
            quadratic_terms=quadratic,
            constant=rng.choice([0.0, 0.1, -2.5, 4.0]),
        ),
        constraints=constraints,
    )


def all_assignments(n_variables: int) -> np.ndarray:
    return np.array(
        list(itertools.product((0, 1), repeat=n_variables)), dtype=np.int8
    ).reshape(-1, n_variables)


# --------------------------------------------------------------------------
# RawSolverResult
# --------------------------------------------------------------------------


class TestRawSolverResult:
    def test_from_dicts_round_trips_through_as_dicts(self):
        samples = [{"a": 1, "b": 0, "__s": 1}, {"a": 0, "b": 1, "__s": 0}]
        raw = RawSolverResult.from_dicts(samples, [1.5, -2.0], "exact")

        assert raw.variables == ["a", "b", "__s"]
        assert raw.samples.dtype == np.int8
        assert raw.samples.shape == (2, 3)
        assert raw.energies.dtype == np.float64
        assert raw.num_samples == 2
        assert raw.as_dicts() == samples
        assert raw.energies.tolist() == [1.5, -2.0]

    def test_lists_are_accepted_and_coerced(self):
        raw = RawSolverResult(
            variables=["x", "y"], samples=[[0, 1], [1, 1]], energies=[0.0, 1.0], backend="b"
        )
        assert raw.samples.dtype == np.int8
        assert raw.as_dicts() == [{"x": 0, "y": 1}, {"x": 1, "y": 1}]

    def test_empty_result_keeps_variable_count(self):
        raw = RawSolverResult(variables=["x", "y"], samples=[], energies=[], backend="b")
        assert raw.samples.shape == (0, 2)
        assert raw.num_samples == 0
        assert raw.as_dicts() == []

    def test_shape_mismatches_are_rejected(self):
        with pytest.raises(ValueError):
            RawSolverResult(
                variables=["x"], samples=[[0, 1]], energies=[0.0], backend="b"
            )
        with pytest.raises(ValueError):
            RawSolverResult(
                variables=["x", "y"], samples=[[0, 1]], energies=[0.0, 1.0], backend="b"
            )

    def test_from_dicts_rejects_inconsistent_keys(self):
        with pytest.raises(ValueError):
            RawSolverResult.from_dicts([{"a": 1}, {"b": 1}], [0.0, 0.0], "b")


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


class TestDeduplicationMatchesReference:
    @pytest.mark.parametrize("seed", range(25))
    def test_same_candidates_min_energy_and_first_seen_order(self, seed):
        rng = random.Random(seed)
        n_business = rng.randint(1, 6)
        n_internal = rng.randint(0, 3)
        variables = [f"x{i}" for i in range(n_business)] + [
            f"__slack_{i}" for i in range(n_internal)
        ]
        rng.shuffle(variables)
        internal = {name for name in variables if name.startswith("__")}
        rows = rng.randint(1, 80)
        # Many duplicates on purpose (few variables, many rows) and many
        # equal energies so the "first read wins on ties" rule is exercised.
        samples = [
            {name: rng.randint(0, 1) for name in variables} for _ in range(rows)
        ]
        energies = [float(rng.choice([-3, -2, -1, 0, 1, 2])) for _ in range(rows)]
        raw = RawSolverResult.from_dicts(samples, energies, "exact")

        expected = reference_deduplicate(raw, internal)
        candidates = deduplicate_samples(raw, internal)

        assert candidates.variables == [v for v in variables if v not in internal]
        assert candidates.samples.dtype == np.int8
        assert candidates.as_pairs() == expected

    def test_energy_ties_keep_the_earliest_read(self):
        raw = RawSolverResult.from_dicts(
            [{"x": 1, "__s": 0}, {"x": 1, "__s": 1}, {"x": 0, "__s": 0}],
            [2.0, 2.0, 5.0],
            "exact",
        )
        candidates = deduplicate_samples(raw, {"__s"})
        assert candidates.as_pairs() == [({"x": 1}, 2.0), ({"x": 0}, 5.0)]

    def test_empty_input(self):
        raw = RawSolverResult(variables=["x", "__s"], samples=[], energies=[], backend="b")
        candidates = deduplicate_samples(raw, {"__s"})
        assert len(candidates) == 0
        assert candidates.variables == ["x"]
        assert candidates.as_pairs() == []

    def test_more_than_64_business_variables(self):
        # The packed key spans several 64-bit words here; rows that agree on
        # the first 64 columns but differ later must still be distinct.
        n = 150
        variables = [f"v{i:03d}" for i in range(n)]
        base = {name: 0 for name in variables}
        rows = [
            dict(base),
            {**base, variables[0]: 1},
            {**base, variables[63]: 1},
            {**base, variables[64]: 1},
            {**base, variables[149]: 1},
            {**base, variables[64]: 1},  # duplicate of row 3 with lower energy
            dict(base),  # duplicate of row 0 with higher energy
        ]
        energies = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 9.0]
        raw = RawSolverResult.from_dicts(rows, energies, "exact")

        candidates = deduplicate_samples(raw, set())

        assert _pack_rows(candidates.samples).shape == (5, 3)
        assert candidates.as_pairs() == reference_deduplicate(raw, set())
        assert [energy for _, energy in candidates.as_pairs()] == [5.0, 4.0, 3.0, 0.5, 1.0]


class TestPackRows:
    def test_word_order_matches_tuple_order(self):
        rng = random.Random(7)
        for columns in (1, 7, 8, 63, 64, 65, 130):
            matrix = np.array(
                [[rng.randint(0, 1) for _ in range(columns)] for _ in range(40)],
                dtype=np.int8,
            )
            packed = _pack_rows(matrix)
            assert packed.shape == (40, -(-columns // 64))
            expected = sorted(range(40), key=lambda r: tuple(matrix[r].tolist()))
            got = _lexsort(_words(packed))
            # Equal rows may come in either order from sorted(); compare keys.
            assert [tuple(matrix[r].tolist()) for r in got] == [
                tuple(matrix[r].tolist()) for r in expected
            ]


# --------------------------------------------------------------------------
# two-layer validation: batch path == full validator, bit for bit
# --------------------------------------------------------------------------


class TestBatchValidationConsistency:
    @pytest.mark.parametrize("seed", range(60))
    def test_every_candidate_agrees_with_validate_solution(self, seed):
        rng = random.Random(seed)
        n_variables = rng.randint(1, 7)
        problem = random_problem(rng, n_variables)
        variables = [v.name for v in problem.variables]
        rng.shuffle(variables)  # column order need not be the problem's order
        samples = all_assignments(n_variables)

        batch = validate_batch(problem, variables, samples)
        objective = evaluate_objective_batch(problem.objective, variables, samples)

        assert batch.feasible.dtype == bool
        assert batch.soft_violation_score.dtype == np.float64
        for row in range(samples.shape[0]):
            sample = dict(zip(variables, samples[row].tolist()))
            full = validate_solution(problem, sample)
            assert bool(batch.feasible[row]) is full.feasible, (seed, sample)
            # Exact float equality on purpose: the ranking is computed from
            # the batch numbers and reported from the full ones.
            assert float(batch.soft_violation_score[row]) == full.soft_violation_score
            assert float(objective[row]) == evaluate_objective(problem.objective, sample)

    def test_epsilon_boundary_is_shared(self):
        # 0.1 + 0.2 != 0.3 in binary; both paths must accept it as "==" 0.3
        # through the shared EPSILON and report identical violation scores
        # for the soft constraint that misses by more than EPSILON.
        problem = OptimizationProblem(
            name="eps",
            variables=[Variable(name="a"), Variable(name="b")],
            objective=Objective(
                direction="minimize",
                linear_terms=[LinearTerm(variable="a", coefficient=1.0)],
            ),
            constraints=[
                Constraint(
                    id="sum",
                    type="hard",
                    terms=[
                        LinearTerm(variable="a", coefficient=0.1),
                        LinearTerm(variable="b", coefficient=0.2),
                    ],
                    operator="==",
                    rhs=0.3,
                ),
                Constraint(
                    id="near",
                    type="soft",
                    terms=[LinearTerm(variable="a", coefficient=1.0)],
                    operator="<=",
                    rhs=1.0 - 2e-8,
                    weight=1.3,
                ),
            ],
        )
        variables = ["a", "b"]
        samples = all_assignments(2)
        batch = validate_batch(problem, variables, samples)
        for row in range(4):
            sample = dict(zip(variables, samples[row].tolist()))
            full = validate_solution(problem, sample)
            assert bool(batch.feasible[row]) is full.feasible
            assert float(batch.soft_violation_score[row]) == full.soft_violation_score
        assert batch.feasible.tolist() == [False, False, False, True]
        assert batch.soft_violation_score[3] > 0.0

    def test_missing_variable_raises_like_the_validator(self):
        problem = OptimizationProblem(
            name="missing",
            variables=[Variable(name="v0"), Variable(name="v1"), Variable(name="v2")],
            objective=Objective(
                direction="minimize",
                linear_terms=[LinearTerm(variable="v0", coefficient=1.0)],
            ),
            constraints=[
                Constraint(
                    id="uses_v2",
                    type="hard",
                    terms=[LinearTerm(variable="v2", coefficient=1.0)],
                    operator="<=",
                    rhs=1.0,
                )
            ],
        )
        with pytest.raises(KeyError):
            validate_solution(problem, {"v0": 0, "v1": 0})
        with pytest.raises(KeyError):
            validate_batch(problem, ["v0", "v1"], np.zeros((1, 2), dtype=np.int8))

    def test_shape_mismatch_is_rejected(self):
        problem = random_problem(random.Random(1), 3)
        with pytest.raises(ValueError):
            validate_batch(problem, ["v0", "v1", "v2"], np.zeros((1, 2), dtype=np.int8))


class TestProcessCandidatesMatchesRowByRow:
    """End-to-end: the array pipeline must rank exactly like the original
    per-row pipeline (dedup → validate each → rank → top-k)."""

    @staticmethod
    def reference(problem, raw, internal, top_k):
        deduped = reference_deduplicate(raw, internal)
        minimize = problem.objective.direction == "minimize"
        scored = []
        for sample, energy in deduped:
            validation = validate_solution(problem, sample)
            if not validation.feasible:
                continue
            objective_value = evaluate_objective(problem.objective, sample)
            score = (
                objective_value + validation.soft_violation_score
                if minimize
                else objective_value - validation.soft_violation_score
            )
            scored.append((sample, energy, validation, objective_value, score))
        sign = 1.0 if minimize else -1.0
        scored.sort(
            key=lambda item: (
                sign * item[4],
                sign * item[3],
                tuple(item[0][name] for name in sorted(item[0])),
            )
        )
        return scored[:top_k], len(deduped), len(scored)

    @pytest.mark.parametrize("seed", range(40))
    def test_same_solutions_same_order_same_numbers(self, seed):
        rng = random.Random(1000 + seed)
        n_variables = rng.randint(1, 6)
        problem = random_problem(rng, n_variables)
        business = [v.name for v in problem.variables]
        internal = {f"__slack_{i}" for i in range(rng.randint(0, 2))}
        variables = business + sorted(internal)
        rng.shuffle(variables)
        rows = rng.randint(1, 60)
        samples = [{name: rng.randint(0, 1) for name in variables} for _ in range(rows)]
        energies = [rng.choice([-2.0, -1.0, 0.0, 0.5]) for _ in range(rows)]
        raw = RawSolverResult.from_dicts(samples, energies, "exact")
        top_k = rng.randint(1, 8)

        solutions, unique, feasible = process_candidates(problem, raw, internal, top_k)
        expected, expected_unique, expected_feasible = self.reference(
            problem, raw, internal, top_k
        )

        assert unique == expected_unique
        assert feasible == expected_feasible
        assert len(solutions) == len(expected)
        for solution, (sample, energy, validation, objective_value, score) in zip(
            solutions, expected
        ):
            assert solution.variables == sample
            assert list(solution.variables) == [v for v in variables if v in business]
            assert solution.energy == energy
            assert solution.objective_value == objective_value
            assert solution.ranking_score == score
            assert solution.soft_violation_score == validation.soft_violation_score
            assert solution.hard_constraints_satisfied is True
            assert solution.constraint_evaluations == validation.evaluations
        assert [s.rank for s in solutions] == list(range(1, len(solutions) + 1))

    def test_solutions_carry_full_evaluations_only_for_top_k(self):
        problem = random_problem(random.Random(3), 4)
        raw = RawSolverResult(
            variables=[v.name for v in problem.variables],
            samples=all_assignments(4),
            energies=np.zeros(16),
            backend="exact",
        )
        solutions, unique, feasible = process_candidates(problem, raw, set(), top_k=2)
        assert unique == 16
        assert len(solutions) <= 2
        for solution in solutions:
            assert len(solution.constraint_evaluations) == len(problem.constraints)
