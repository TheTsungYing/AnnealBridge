"""Array-based candidate processing: raw results, deduplication and the
two-layer validation (spec §21, §23, §25; overview principle 2).

The fast paths must be *semantically identical* to the row-by-row
reference, not merely close: every candidate is still re-validated against
the original problem, and the feasibility / soft-violation / objective
numbers feeding the ranking are the very same floats the full validator
reports for the top-k.
"""

import itertools
import math
import random
from collections import Counter

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
    diagnose_infeasibility,
    evaluate_objective,
    evaluate_objective_batch,
    process_candidates,
)
from annealbridge.orchestration.candidates import (
    _lexsort,
    _pack_integer_rows,
    _pack_rows,
    _row_keys,
    _words,
)
from annealbridge.solvers import RawSolverResult
from annealbridge.validation import tolerance, validate_batch, validate_solution

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


def assignment_key(business: dict[str, int]) -> tuple[int, ...]:
    """The name-sorted value tuple identifying a business assignment."""
    return tuple(business[name] for name in sorted(business))


def reference_tally(
    raw: RawSolverResult, internal_variables: set[str]
) -> Counter[tuple[int, ...]]:
    """Raw rows per business assignment, tallied row by row.

    The oracle for ``CandidateSet.counts`` / ``Solution.sample_count``: a
    plain tally keyed by the assignment, so the array path cannot pass by
    reusing its own grouping. ``Counter`` keeps first-seen key order, the
    same order the deduplication returns candidates in.
    """
    tally: Counter[tuple[int, ...]] = Counter()
    for sample in raw.as_dicts():
        business = {
            name: value for name, value in sample.items() if name not in internal_variables
        }
        tally[assignment_key(business)] += 1
    return tally


def reference_counts(raw: RawSolverResult, internal_variables: set[str]) -> list[int]:
    """The tally above, in first-seen order."""
    return list(reference_tally(raw, internal_variables).values())


def hard_ids(problem: OptimizationProblem) -> list[str]:
    """The hard constraint ids in the problem's own order."""
    return [c.id for c in problem.constraints if c.type == "hard"]


def reference_hard_total(validation) -> float:
    """Σ ``violation_amount`` over the hard constraints, in problem order.

    The oracle for ``BatchValidation.hard_violation_total`` and for
    ``ClosestCandidate.hard_violation_total``: the same values the full
    validator reports, accumulated in the same order, so a difference in
    association would show up as a last-bit mismatch.
    """
    total = 0.0
    for evaluation in validation.evaluations:
        if evaluation.constraint_type == "hard":
            total += evaluation.violation_amount
    return total


def reference_hard_violations(validation) -> list[bool]:
    """Per hard constraint, whether this candidate violated it."""
    return [
        not evaluation.satisfied
        for evaluation in validation.evaluations
        if evaluation.constraint_type == "hard"
    ]


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


def column_range(column: int) -> tuple[int, int]:
    """An inclusive value range that differs per column and dips negative.

    Used by the wide integer fixtures: giving every column its own width
    means a key path that assumed one common range -- or that packed the
    columns with a single shared offset -- would misorder the rows.
    """
    return -(column % 7), (column % 13) + 1


def distinct_binary_rows(
    rng: np.random.Generator, count: int, n_variables: int
) -> np.ndarray:
    """``count`` pairwise-distinct 0/1 ``int8`` rows.

    Falls back to the full enumeration when there are not that many
    assignments (``n_variables`` = 1 asks for far more rows than exist).
    """
    if n_variables < 20 and count >= (1 << n_variables):
        return all_assignments(n_variables)
    collected = np.empty((0, n_variables), dtype=np.int8)
    while collected.shape[0] < count:
        draw = rng.integers(0, 2, size=(count * 2, n_variables), dtype=np.int8)
        collected = np.unique(np.vstack([collected, draw]), axis=0)
    rng.shuffle(collected, axis=0)  # np.unique sorts; do not feed sorted rows
    return np.ascontiguousarray(collected[:count])


def distinct_integer_rows(
    rng: np.random.Generator, count: int, n_variables: int
) -> np.ndarray:
    """``count`` pairwise-distinct ``int64`` rows, one range per column."""
    collected = np.empty((0, n_variables), dtype=np.int64)
    while collected.shape[0] < count:
        draw = np.empty((count * 2, n_variables), dtype=np.int64)
        for column in range(n_variables):
            low, high = column_range(column)
            draw[:, column] = rng.integers(low, high + 1, size=draw.shape[0])
        collected = np.unique(np.vstack([collected, draw]), axis=0)
    rng.shuffle(collected, axis=0)
    return np.ascontiguousarray(collected[:count])


# Row counts for the "at scale" deduplication comparisons below. The oracle
# walks every raw row in Python, so these are the largest sizes that still
# keep the whole module in the low seconds.
_LARGE_BINARY_ROWS = 8_000
_LARGE_INTEGER_ROWS = 3_000
_LARGE_INTEGER_VARIABLES = 50


def shuffled_names(rng: np.random.Generator, prefix: str, count: int) -> list[str]:
    """``count`` variable names whose alphabetical order is not the column
    order, so anything keyed by name order cannot coincide with the layout.
    """
    return [f"{prefix}{index:03d}" for index in rng.permutation(count).tolist()]


# Magnitudes on both sides of the hybrid tolerance's crossover (1e4): at
# scale 1 the tolerance is still the absolute 1e-8, at 1e12 it is 1.0.
_SCALES = (1.0, 1e8, 1e10, 1e12)
_SMALL_COEFFICIENTS = (0.1, 0.2, 0.3, -0.1, -0.3)
# Offsets from a reachable lhs value, in units of the tolerance at that
# magnitude and in absolute units, so satisfied and violated both occur.
_BAND_OFFSETS = ("zero", "5e-9", "2e-8", "-2e-8", "0.5tol", "1.5tol", "-1.5tol", "one")


def band_constraint(
    rng: random.Random,
    variables: list[str],
    samples: np.ndarray,
    index: int,
    kind: str | None = None,
    weight: float | None = None,
) -> Constraint:
    """A constraint with huge coefficients whose rhs sits on the edge of the
    hybrid tolerance band (review F-05).

    ``tol = max(1e-8, 1e-12 * magnitude)``, so the band is 1e-8 wide at scale
    1 and 1.0 wide at scale 1e12. Putting the rhs a fraction of a band away
    from an lhs value the assignments can actually reach is what makes the
    scalar and the numpy kernel disagree the moment they drift apart.
    """
    scale = rng.choice(_SCALES)
    chosen = rng.sample(variables, k=rng.randint(2, len(variables)))
    terms = [
        LinearTerm(variable=name, coefficient=scale * rng.uniform(-1.0, 1.0))
        for name in chosen
    ]
    # A couple of small coefficients as well: their low bits are exactly what
    # a large-magnitude sum loses.
    for name in rng.sample(chosen, k=min(2, len(chosen))):
        terms.append(
            LinearTerm(variable=name, coefficient=rng.choice(_SMALL_COEFFICIENTS))
        )

    assignment = dict(zip(variables, samples[rng.randrange(samples.shape[0])].tolist()))
    lhs = 0.0
    for term in terms:
        lhs += term.coefficient * assignment[term.variable]
    tol = tolerance(lhs, lhs)
    offsets = {
        "zero": 0.0,
        "5e-9": 5e-9,
        "2e-8": 2e-8,
        "-2e-8": -2e-8,
        "0.5tol": 0.5 * tol,
        "1.5tol": 1.5 * tol,
        "-1.5tol": -1.5 * tol,
        "one": 1.0,
    }
    delta = offsets[rng.choice(_BAND_OFFSETS)]

    if kind is None:
        kind = rng.choice(["hard", "soft", "soft"])
    if kind == "soft":
        weight = rng.uniform(0.5, 3.0) if weight is None else weight
    else:
        weight = None
    return Constraint(
        id=f"c{index}",
        type=kind,
        terms=terms,
        operator=rng.choice(["==", "<=", ">="]),
        rhs=lhs + delta,
        weight=weight,
    )


def band_problem(
    rng: random.Random, variables: list[str], constraints: list[Constraint]
) -> OptimizationProblem:
    return OptimizationProblem(
        name="hybrid tolerance",
        variables=[Variable(name=name) for name in sorted(variables)],
        objective=Objective(
            direction="minimize",
            linear_terms=[
                LinearTerm(variable=rng.choice(variables), coefficient=1.0)
            ],
        ),
        constraints=constraints,
    )


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


class TestRawSolverResultDtypes:
    """3b §11: any numpy integer dtype is kept as given, non-integer ones
    are rejected rather than silently cast."""

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64])
    def test_ndarray_dtype_is_preserved(self, dtype):
        samples = np.array([[0, 3], [-2, 1]], dtype=dtype)
        raw = RawSolverResult(
            variables=["x", "y"], samples=samples, energies=[0.0, 1.0], backend="b"
        )
        assert raw.samples.dtype == dtype
        assert raw.as_dicts() == [{"x": 0, "y": 3}, {"x": -2, "y": 1}]

    def test_from_dicts_dtype_keyword_round_trips_integer_values(self):
        samples = [{"a": -3, "b": 0, "__s": 7}, {"a": 5, "b": 2, "__s": -1}]
        raw = RawSolverResult.from_dicts(samples, [1.5, -2.0], "cqm", dtype=np.int64)

        assert raw.samples.dtype == np.int64
        assert raw.variables == ["a", "b", "__s"]
        assert raw.as_dicts() == samples

    def test_from_dicts_defaults_to_int8(self):
        raw = RawSolverResult.from_dicts([{"a": 1, "b": 0}], [0.0], "exact")
        assert raw.samples.dtype == np.int8

    def test_list_wider_than_int8_stays_int64(self):
        raw = RawSolverResult(
            variables=["x", "y"],
            samples=[[1000, 0], [-1000, 1]],
            energies=[0.0, 1.0],
            backend="b",
        )
        assert raw.samples.dtype == np.int64
        assert raw.as_dicts() == [{"x": 1000, "y": 0}, {"x": -1000, "y": 1}]

    def test_list_within_int8_range_is_narrowed(self):
        raw = RawSolverResult(
            variables=["x", "y"], samples=[[-5, 7]], energies=[0.0], backend="b"
        )
        assert raw.samples.dtype == np.int8
        assert raw.as_dicts() == [{"x": -5, "y": 7}]

    @pytest.mark.parametrize(
        "samples",
        [
            np.array([[0.0, 1.0]]),
            np.array([[0.5, 1.0]], dtype=np.float32),
            np.array([[True, False]]),
            np.array([[0, 1]], dtype=object),
            [[0.5, 1.0]],
        ],
    )
    def test_non_integer_dtypes_are_rejected(self, samples):
        with pytest.raises(ValueError):
            RawSolverResult(
                variables=["x", "y"], samples=samples, energies=[0.0], backend="b"
            )

    @pytest.mark.parametrize(
        "samples",
        [[], np.empty((0, 2), dtype=np.int64), np.array([])],
    )
    def test_empty_input_is_an_int8_matrix_with_the_variable_count(self, samples):
        raw = RawSolverResult(
            variables=["x", "y"], samples=samples, energies=[], backend="b"
        )
        assert raw.samples.dtype == np.int8
        assert raw.samples.shape == (0, 2)
        assert raw.num_samples == 0
        assert raw.as_dicts() == []


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
        assert candidates.counts.dtype == np.int64
        assert candidates.counts.tolist() == reference_counts(raw, internal)
        assert int(candidates.counts.sum()) == raw.num_samples

    def test_energy_ties_keep_the_earliest_read(self):
        raw = RawSolverResult.from_dicts(
            [{"x": 1, "__s": 0}, {"x": 1, "__s": 1}, {"x": 0, "__s": 0}],
            [2.0, 2.0, 5.0],
            "exact",
        )
        candidates = deduplicate_samples(raw, {"__s"})
        assert candidates.as_pairs() == [({"x": 1}, 2.0), ({"x": 0}, 5.0)]
        # x=1 came back twice (same energy, different slack), x=0 once.
        assert candidates.counts.tolist() == [2, 1]

    def test_counts_follow_the_first_seen_reordering(self):
        # The groups are built in assignment order (x=0 first), but the
        # candidates come back in first-seen order (x=1 first). ``counts``
        # must be permuted along with the rows, not left in group order.
        raw = RawSolverResult.from_dicts(
            [{"x": 1}, {"x": 0}, {"x": 1}, {"x": 1}],
            [4.0, 3.0, 1.0, 2.0],
            "exact",
        )
        candidates = deduplicate_samples(raw, set())
        assert candidates.as_pairs() == [({"x": 1}, 1.0), ({"x": 0}, 3.0)]
        assert candidates.counts.tolist() == [3, 1]
        assert int(candidates.counts.sum()) == raw.num_samples

    def test_duplicate_rows_with_differing_energies_count_once_each(self):
        raw = RawSolverResult.from_dicts(
            [{"x": 0, "y": 1}] * 5 + [{"x": 1, "y": 1}],
            [3.0, 1.0, 2.0, 1.0, 7.0, 0.0],
            "exact",
        )
        candidates = deduplicate_samples(raw, set())
        assert candidates.as_pairs() == [({"x": 0, "y": 1}, 1.0), ({"x": 1, "y": 1}, 0.0)]
        assert candidates.counts.tolist() == [5, 1]

    def test_empty_input(self):
        raw = RawSolverResult(variables=["x", "__s"], samples=[], energies=[], backend="b")
        candidates = deduplicate_samples(raw, {"__s"})
        assert len(candidates) == 0
        assert candidates.variables == ["x"]
        assert candidates.as_pairs() == []
        assert candidates.counts.shape == (0,)
        assert candidates.counts.dtype == np.int64

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
        # Rows 0 and 6 are the same assignment, so are rows 3 and 5.
        assert candidates.counts.tolist() == [2, 1, 1, 2, 1]
        assert candidates.counts.tolist() == reference_counts(raw, set())

    @pytest.mark.parametrize("n_variables", [1, 20, 64, 65])
    @pytest.mark.parametrize("seed", range(5))
    def test_large_binary_matrices_match_the_row_by_row_reference(
        self, n_variables, seed
    ):
        # The cases above are tens of rows; a grouped array path only shows
        # its seams at scale, and around the 64-bit word boundary (1 word,
        # 2 words) with a few thousand duplicates per assignment.
        rng = np.random.default_rng(7_000 + 100 * n_variables + seed)
        rows = _LARGE_BINARY_ROWS
        distinct = distinct_binary_rows(rng, max(1, rows // 5), n_variables)
        matrix = np.ascontiguousarray(
            distinct[rng.integers(0, distinct.shape[0], size=rows)]
        )
        # -0.0 and 0.0 on purpose: they compare equal, so the minimum-energy
        # pick has to fall back to "earliest read wins" for them.
        energy_pool = np.array([-1.0, -0.0, 0.0, 1.0, 2.0])
        energies = energy_pool[rng.integers(0, len(energy_pool), size=rows)]
        variables = shuffled_names(rng, "b", n_variables)
        raw = RawSolverResult(
            variables=variables, samples=matrix, energies=energies, backend="exact"
        )

        expected = reference_deduplicate(raw, set())
        candidates = deduplicate_samples(raw, set())

        assert candidates.samples.dtype == np.int8
        assert candidates.as_pairs() == expected
        assert candidates.counts.tolist() == reference_counts(raw, set())
        assert int(candidates.counts.sum()) == raw.num_samples
        assert len(candidates) < rows  # duplicates really were collapsed
        # ``as_pairs()`` compares energies as floats, where -0.0 == 0.0, so
        # the signed zero has to be checked separately: picking the wrong
        # read of a tie would otherwise pass unnoticed.
        assert [math.copysign(1.0, energy) for _, energy in candidates.as_pairs()] == [
            math.copysign(1.0, energy) for _, energy in expected
        ]


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


class TestRowKeys:
    """3b §11: 0/1 int8 keeps the packed bit keys of 3a; anything else gets
    one int64 key per column. Both order rows lexicographically."""

    @pytest.mark.parametrize("columns", [1, 7, 64, 65, 130])
    def test_binary_int8_still_takes_the_packbits_path(self, columns):
        rng = random.Random(11 + columns)
        matrix = np.array(
            [[rng.randint(0, 1) for _ in range(columns)] for _ in range(40)],
            dtype=np.int8,
        )
        keys = _row_keys(matrix)
        expected = _words(_pack_rows(matrix))

        assert len(keys) == len(expected)
        for key, word in zip(keys, expected):
            assert key.dtype == word.dtype
            assert np.array_equal(key, word)

    @pytest.mark.parametrize("dtype", [np.int8, np.int64])
    def test_non_binary_values_take_the_integer_packing_path(self, dtype):
        rng = random.Random(29)
        columns = 5
        rows = 40
        matrix = np.array(
            [[rng.randint(-1, 2) for _ in range(columns)] for _ in range(rows)],
            dtype=dtype,
        )
        # Make sure the values that packbits would flatten are present.
        matrix[0] = 2
        matrix[1] = -1
        if dtype == np.int64:
            matrix[2] = 300

        keys = _row_keys(matrix)
        packed = _words(_pack_integer_rows(matrix))
        # Five narrow columns share one word: fewer keys than columns.
        assert len(keys) == len(packed) == 1
        assert all(key.dtype == np.uint64 for key in keys)
        assert all(np.array_equal(key, word) for key, word in zip(keys, packed))

        got = _lexsort(keys)
        expected = sorted(range(rows), key=lambda r: tuple(matrix[r].tolist()))
        assert [tuple(matrix[r].tolist()) for r in got] == [
            tuple(matrix[r].tolist()) for r in expected
        ]

    def test_empty_matrix_is_accepted(self):
        keys = _row_keys(np.zeros((0, 5), dtype=np.int8))
        assert all(len(key) == 0 for key in keys)

    @pytest.mark.parametrize("columns", [1, 5, 40, 200])
    def test_integer_columns_of_mixed_width_order_like_tuples(self, columns):
        # Every column carries its own (partly negative) range and one of
        # them is constant, so the keys cannot be built from a single shared
        # range, and a constant column must not shift the ordering either.
        rng = np.random.default_rng(17 + columns)
        rows = 300
        matrix = np.empty((rows, columns), dtype=np.int64)
        for column in range(columns):
            low, high = column_range(column)
            matrix[:, column] = rng.integers(low, high + 1, size=rows)
        if columns >= 2:
            matrix[:, columns // 2] = -3

        got = _lexsort(_row_keys(matrix))
        expected = sorted(range(rows), key=lambda r: tuple(matrix[r].tolist()))

        # Equal rows may come back in either order from either sort, so the
        # comparison is on the row tuples, not on the row indices.
        assert [tuple(matrix[r].tolist()) for r in got] == [
            tuple(matrix[r].tolist()) for r in expected
        ]

    def test_integer_extreme_values_order_like_tuples(self):
        # The int64 saturation points together with 0 and -1: a key path
        # that offset or shifted the values would wrap around here and stop
        # ordering the rows the way tuple comparison does.
        rng = np.random.default_rng(23)
        rows = 50
        columns = 4
        matrix = rng.integers(-2, 3, size=(rows, columns)).astype(np.int64)
        extremes = np.array(
            [np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0, -1], dtype=np.int64
        )
        matrix[:, 1] = extremes[rng.integers(0, len(extremes), size=rows)]
        # Make sure each extreme really occurs whatever the draw did.
        matrix[:4, 1] = extremes

        got = _lexsort(_row_keys(matrix))
        expected = sorted(range(rows), key=lambda r: tuple(matrix[r].tolist()))

        assert set(matrix[:, 1].tolist()) == set(extremes.tolist())
        assert [tuple(matrix[r].tolist()) for r in got] == [
            tuple(matrix[r].tolist()) for r in expected
        ]

    def test_integer_matrix_with_no_rows_is_accepted(self):
        # Same as the int8 case above, on the integer path: no rows must not
        # trip the min/max scan or the key construction.
        keys = _row_keys(np.zeros((0, 5), dtype=np.int64))
        assert all(len(key) == 0 for key in keys)


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
        # The hard tallies feed the infeasibility diagnosis, so they must be
        # the validator's own numbers too, not merely close ones.
        assert batch.hard_constraint_ids == hard_ids(problem)
        assert batch.hard_violation_total.dtype == np.float64
        assert batch.hard_violated_counts.dtype == np.int64
        assert batch.hard_violated_counts.shape == (len(hard_ids(problem)),)
        violated_counts = [0] * len(batch.hard_constraint_ids)

        for row in range(samples.shape[0]):
            sample = dict(zip(variables, samples[row].tolist()))
            full = validate_solution(problem, sample)
            assert bool(batch.feasible[row]) is full.feasible, (seed, sample)
            # Exact float equality on purpose: the ranking is computed from
            # the batch numbers and reported from the full ones.
            assert float(batch.soft_violation_score[row]) == full.soft_violation_score
            assert float(objective[row]) == evaluate_objective(problem.objective, sample)
            assert float(batch.hard_violation_total[row]) == reference_hard_total(
                full
            ), (seed, sample)
            for position, violated in enumerate(reference_hard_violations(full)):
                violated_counts[position] += int(violated)

        assert batch.hard_violated_counts.tolist() == violated_counts

    def test_epsilon_boundary_is_shared(self):
        # 0.1 + 0.2 != 0.3 in binary; both paths must accept it as "==" 0.3
        # through the shared hybrid tolerance and report identical violation
        # scores for the soft constraint that misses by more than that
        # tolerance.
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

    def test_hybrid_tolerance_paths_agree_at_large_scale(self):
        """Review F-05: the tolerance now depends on the magnitude of the
        numbers compared, so the two kernels must still agree once the band
        is 1e-3 or 1.0 wide instead of 1e-8 — including on the rhs values
        that sit right on its edge."""
        rng = random.Random(20260909)
        seen_feasible = False
        seen_infeasible = False
        seen_soft_violation = False

        for problem_index in range(40):
            n_variables = rng.randint(4, 6)
            variables = [f"v{index}" for index in range(n_variables)]
            rng.shuffle(variables)
            samples = all_assignments(n_variables)
            constraints = [
                band_constraint(rng, variables, samples, index)
                for index in range(rng.randint(1, 3))
            ]
            problem = band_problem(rng, variables, constraints)

            batch = validate_batch(problem, variables, samples)
            assert batch.hard_constraint_ids == hard_ids(problem)
            assert batch.hard_violation_total.dtype == np.float64
            assert batch.hard_violated_counts.dtype == np.int64
            assert batch.hard_violated_counts.shape == (len(hard_ids(problem)),)
            violated_counts = [0] * len(batch.hard_constraint_ids)

            for row in range(samples.shape[0]):
                sample = dict(zip(variables, samples[row].tolist()))
                full = validate_solution(problem, sample)
                assert bool(batch.feasible[row]) is full.feasible, (
                    problem_index,
                    sample,
                )
                # Exact equality: the ranking uses the batch number and the
                # report shows the full one.
                assert float(batch.soft_violation_score[row]) == (
                    full.soft_violation_score
                ), (problem_index, sample)
                # Same for the hard total the diagnosis reports: the
                # tolerance decides, so a within-band residual is a zero on
                # both paths.
                assert float(batch.hard_violation_total[row]) == (
                    reference_hard_total(full)
                ), (problem_index, sample)
                for position, violated in enumerate(reference_hard_violations(full)):
                    violated_counts[position] += int(violated)
                seen_feasible = seen_feasible or full.feasible
                seen_infeasible = seen_infeasible or not full.feasible
                seen_soft_violation = (
                    seen_soft_violation or full.soft_violation_score > 0.0
                )

            assert batch.hard_violated_counts.tolist() == violated_counts

        # A single soft constraint of weight 1: the batch score is then the
        # squared violation amount itself, so violation_amount is pinned to
        # the last bit too, not just the aggregate score.
        seen_squared_violation = False
        for problem_index in range(40):
            n_variables = rng.randint(4, 6)
            variables = [f"v{index}" for index in range(n_variables)]
            rng.shuffle(variables)
            samples = all_assignments(n_variables)
            problem = band_problem(
                rng,
                variables,
                [band_constraint(rng, variables, samples, 0, kind="soft", weight=1.0)],
            )

            batch = validate_batch(problem, variables, samples)
            for row in range(samples.shape[0]):
                sample = dict(zip(variables, samples[row].tolist()))
                full = validate_solution(problem, sample)
                violation = full.evaluations[0].violation_amount
                assert float(batch.soft_violation_score[row]) == violation * violation, (
                    problem_index,
                    sample,
                )
                seen_squared_violation = seen_squared_violation or violation > 0.0

        assert seen_feasible
        assert seen_infeasible
        assert seen_soft_violation
        assert seen_squared_violation

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

        processed = process_candidates(problem, raw, internal, top_k)
        solutions = processed.solutions
        expected, expected_unique, expected_feasible = self.reference(
            problem, raw, internal, top_k
        )
        # Tally the raw rows per business assignment independently of the
        # pipeline's own grouping.
        tally = reference_tally(raw, internal)

        assert processed.unique_samples == expected_unique
        assert processed.feasible_samples == expected_feasible
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
            assert solution.sample_count == tally[assignment_key(sample)]
        assert [s.rank for s in solutions] == list(range(1, len(solutions) + 1))
        assert sum(tally.values()) == raw.num_samples

    def test_solutions_carry_full_evaluations_only_for_top_k(self):
        problem = random_problem(random.Random(3), 4)
        raw = RawSolverResult(
            variables=[v.name for v in problem.variables],
            samples=all_assignments(4),
            energies=np.zeros(16),
            backend="exact",
        )
        processed = process_candidates(problem, raw, set(), top_k=2)
        assert processed.unique_samples == 16
        assert len(processed.solutions) <= 2
        for solution in processed.solutions:
            assert len(solution.constraint_evaluations) == len(problem.constraints)
            # Every assignment was enumerated exactly once, no internals.
            assert solution.sample_count == 1


# --------------------------------------------------------------------------
# infeasibility diagnostics
# --------------------------------------------------------------------------


def mutually_exclusive_hard_constraints(names: list[str]) -> list[Constraint]:
    """Two hard constraints no assignment can satisfy together.

    ``Σ v >= 1`` and ``Σ v <= 0``: the all-zero assignment misses the first
    by exactly 1 and every single-one assignment misses the second by
    exactly 1, so the smallest total is reached by more than one candidate
    and the first-seen tie-break is what the choice actually turns on.
    """
    terms = [LinearTerm(variable=name, coefficient=1.0) for name in names]
    return [
        Constraint(id="at_least_one", type="hard", terms=terms, operator=">=", rhs=1),
        Constraint(id="none_at_all", type="hard", terms=terms, operator="<=", rhs=0),
    ]


class TestInfeasibilityDiagnostics:
    """``process_candidates`` explains an attempt where nothing was feasible:
    the closest candidate and each hard constraint's violation rate, both
    recomputed by the validator from the original problem."""

    @staticmethod
    def reference(problem: OptimizationProblem, raw: RawSolverResult, internal: set):
        """Row-by-row oracle over the deduplicated candidates.

        ``reference_deduplicate`` keeps first-seen order, and ``min``
        returns the first minimal element, so this reproduces the
        "smallest total, earliest on a tie" rule independently of
        ``np.argmin``.
        """
        deduped = reference_deduplicate(raw, internal)
        totals = []
        counts = [0] * len(hard_ids(problem))
        for sample, _energy in deduped:
            validation = validate_solution(problem, sample)
            totals.append(reference_hard_total(validation))
            for position, violated in enumerate(reference_hard_violations(validation)):
                counts[position] += int(violated)
        best = min(range(len(totals)), key=lambda index: totals[index])
        return deduped, totals, counts, best

    def assert_matches_reference(self, problem, raw, internal, processed):
        deduped, totals, counts, best = self.reference(problem, raw, internal)
        diagnostics = processed.infeasibility
        assert diagnostics is not None
        assert processed.solutions == []
        assert processed.unique_samples == len(deduped)
        assert processed.feasible_samples == 0

        expected_sample = deduped[best][0]
        closest = diagnostics.closest_candidate
        assert closest.variables == expected_sample
        # Exact equality: the reported total is the validator's own sum.
        assert closest.hard_violation_total == totals[best]
        assert closest.hard_violation_total == min(totals)
        assert closest.constraint_evaluations == (
            validate_solution(problem, expected_sample).evaluations
        )

        rates = diagnostics.hard_violation_rates
        assert [rate.constraint_id for rate in rates] == hard_ids(problem)
        assert [rate.violated_candidates for rate in rates] == counts
        for rate in rates:
            assert rate.candidates == len(deduped)
            assert rate.violated_fraction == rate.violated_candidates / len(deduped)

    @pytest.mark.parametrize("seed", range(30))
    def test_closest_candidate_and_rates_match_the_row_by_row_oracle(self, seed):
        rng = random.Random(7000 + seed)
        n_variables = rng.randint(1, 5)
        problem = random_problem(rng, n_variables)
        names = [variable.name for variable in problem.variables]
        # The random problem's own constraints stay: they contribute to the
        # totals, so the oracle is not comparing against a single term.
        problem = problem.model_copy(
            update={
                "constraints": [
                    *problem.constraints,
                    *mutually_exclusive_hard_constraints(names),
                ]
            }
        )
        internal = {f"__slack_{i}" for i in range(rng.randint(0, 2))}
        variables = names + sorted(internal)
        rng.shuffle(variables)
        rows = rng.randint(1, 40)
        samples = [{name: rng.randint(0, 1) for name in variables} for _ in range(rows)]
        energies = [rng.choice([-2.0, -1.0, 0.0, 0.5]) for _ in range(rows)]
        raw = RawSolverResult.from_dicts(samples, energies, "exact")

        processed = process_candidates(problem, raw, internal, top_k=5)

        self.assert_matches_reference(problem, raw, internal, processed)

    def test_ties_are_broken_by_first_seen_order(self):
        # Both rows miss by exactly 1 (all-zero violates at_least_one,
        # a single one violates none_at_all), so only the order of the raw
        # output can decide which is reported.
        names = ["a", "b"]
        problem = OptimizationProblem(
            name="tie",
            variables=[Variable(name=name) for name in names],
            objective=Objective(
                direction="minimize",
                linear_terms=[LinearTerm(variable="a", coefficient=1.0)],
            ),
            constraints=mutually_exclusive_hard_constraints(names),
        )
        rows = [{"a": 1, "b": 0}, {"a": 0, "b": 0}]
        energies = [0.0, -5.0]  # the later row is the *better* energy

        forward = process_candidates(
            problem, RawSolverResult.from_dicts(rows, energies, "exact"), set(), top_k=5
        )
        reverse = process_candidates(
            problem,
            RawSolverResult.from_dicts(rows[::-1], energies[::-1], "exact"),
            set(),
            top_k=5,
        )

        # Energy plays no part: the first row read wins each way round.
        assert forward.infeasibility.closest_candidate.variables == {"a": 1, "b": 0}
        assert reverse.infeasibility.closest_candidate.variables == {"a": 0, "b": 0}
        for processed in (forward, reverse):
            diagnostics = processed.infeasibility
            assert diagnostics.closest_candidate.hard_violation_total == 1.0
            assert [
                (rate.constraint_id, rate.violated_candidates, rate.candidates)
                for rate in diagnostics.hard_violation_rates
            ] == [("at_least_one", 1, 2), ("none_at_all", 1, 2)]

    def test_diagnose_infeasibility_agrees_with_process_candidates(self):
        names = ["a", "b", "c"]
        problem = OptimizationProblem(
            name="direct",
            variables=[Variable(name=name) for name in names],
            objective=Objective(
                direction="minimize",
                linear_terms=[LinearTerm(variable="a", coefficient=1.0)],
            ),
            constraints=mutually_exclusive_hard_constraints(names),
        )
        raw = RawSolverResult(
            variables=names,
            samples=all_assignments(3),
            energies=np.zeros(8),
            backend="exact",
        )

        candidates = deduplicate_samples(raw, set())
        verdict = validate_batch(problem, candidates.variables, candidates.samples)
        direct = diagnose_infeasibility(problem, candidates, verdict)
        processed = process_candidates(problem, raw, set(), top_k=5)

        assert processed.infeasibility == direct
        # 8 assignments: only the all-zero one satisfies none_at_all, and
        # only the seven others satisfy at_least_one.
        assert [
            (rate.violated_candidates, rate.candidates)
            for rate in direct.hard_violation_rates
        ] == [(1, 8), (7, 8)]
        self.assert_matches_reference(problem, raw, set(), processed)

    def test_a_feasible_candidate_leaves_no_diagnosis(self):
        # Only the second constraint of the exclusive pair, so the all-zero
        # assignment is feasible and 15 of 16 candidates are not.
        names = ["a", "b", "c", "d"]
        problem = OptimizationProblem(
            name="one reachable hard constraint",
            variables=[Variable(name=name) for name in names],
            objective=Objective(
                direction="minimize",
                linear_terms=[LinearTerm(variable="a", coefficient=1.0)],
            ),
            constraints=mutually_exclusive_hard_constraints(names)[1:],
        )
        raw = RawSolverResult(
            variables=names,
            samples=all_assignments(4),
            energies=np.zeros(16),
            backend="exact",
        )
        processed = process_candidates(problem, raw, set(), top_k=3)

        assert processed.feasible_samples == 1
        # A single feasible candidate is enough: the diagnosis is for an
        # attempt that found none at all.
        assert processed.infeasibility is None

    def test_empty_raw_output_has_nothing_to_diagnose(self):
        problem = random_problem(random.Random(13), 3)
        raw = RawSolverResult(
            variables=[v.name for v in problem.variables],
            samples=[],
            energies=[],
            backend="exact",
        )
        processed = process_candidates(problem, raw, set(), top_k=3)

        assert processed.solutions == []
        assert processed.unique_samples == 0
        assert processed.feasible_samples == 0
        # Zero candidates is not a diagnosis: there is nothing to be closest.
        assert processed.infeasibility is None

    def test_problem_without_hard_constraints_has_empty_tallies(self):
        problem = OptimizationProblem(
            name="soft only",
            variables=[Variable(name="a"), Variable(name="b")],
            objective=Objective(
                direction="minimize",
                linear_terms=[LinearTerm(variable="a", coefficient=1.0)],
            ),
            constraints=[
                Constraint(
                    id="prefer_a",
                    type="soft",
                    terms=[LinearTerm(variable="a", coefficient=1.0)],
                    operator="<=",
                    rhs=0,
                    weight=1.0,
                )
            ],
        )
        samples = all_assignments(2)
        batch = validate_batch(problem, ["a", "b"], samples)

        assert batch.hard_constraint_ids == []
        assert batch.hard_violated_counts.shape == (0,)
        assert batch.hard_violated_counts.dtype == np.int64
        assert batch.hard_violation_total.tolist() == [0.0] * 4
        # Without a hard constraint every candidate is feasible, so the
        # diagnosis never runs.
        assert batch.feasible.all()


# --------------------------------------------------------------------------
# integer-valued rows (IR 1.1 / CQM backends, 3b spec §11)
# --------------------------------------------------------------------------


def random_integer_problem(rng: random.Random, n_variables: int) -> OptimizationProblem:
    """An IR 1.1 problem mixing binary and bounded-integer variables.

    Same awkward coefficients as :func:`random_problem` -- the point is
    still last-bit agreement -- but the variables now carry bounds that
    include negative values, so the candidate rows are genuine integers
    rather than bits. Enumerating every assignment is no longer cheap, so a
    constraint's rhs is taken from the lhs of one random assignment
    (accumulated in the validator's term order) to keep "==" reachable.
    """
    names = [f"v{index}" for index in range(n_variables)]
    coefficient_pool = [0.1, 0.2, 0.3, 0.7, 1.0, 1.5, 2.25, -0.1, -0.3, -1.0, 3.0, 7.0]

    variables: list[Variable] = []
    for name in names:
        if rng.random() < 0.6:
            lower = rng.randint(-4, 1)
            variables.append(
                Variable(
                    name=name,
                    type="integer",
                    lower_bound=lower,
                    upper_bound=lower + rng.randint(1, 6),
                )
            )
        else:
            variables.append(Variable(name=name))
    bounds = {variable.name: variable.bounds() for variable in variables}

    def terms(count: int) -> list[LinearTerm]:
        chosen = rng.sample(names, k=count)
        return [
            LinearTerm(variable=name, coefficient=rng.choice(coefficient_pool))
            for name in chosen
        ]

    def random_assignment() -> dict[str, int]:
        return {name: rng.randint(*bounds[name]) for name in names}

    constraints: list[Constraint] = []
    for index in range(rng.randint(1, 4)):
        operator = rng.choice(["<=", ">=", "<=", ">=", "=="])
        kind = rng.choice(["hard", "soft", "soft"])
        constraint_terms = terms(rng.randint(1, n_variables))
        if rng.random() < 0.7:
            assignment = random_assignment()
            rhs = 0.0
            for term in constraint_terms:
                rhs += term.coefficient * assignment[term.variable]
        else:
            rhs = rng.choice(coefficient_pool)
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
        version="1.1",
        name="random_integer",
        variables=variables,
        objective=Objective(
            direction=rng.choice(["minimize", "maximize"]),
            linear_terms=terms(rng.randint(1, n_variables)),
            quadratic_terms=quadratic,
            constant=rng.choice([0.0, 0.1, -2.5, 4.0]),
        ),
        constraints=constraints,
    )


class TestIntegerRowsMatchReference:
    """Deduplication of integer-valued rows: minimum energy, first-seen
    order and grouping must equal the row-by-row oracle on either key path.
    """

    @pytest.mark.parametrize("seed", range(25))
    def test_int64_rows_match_the_row_by_row_reference(self, seed):
        rng = random.Random(2000 + seed)
        n_business = rng.randint(1, 6)
        n_internal = rng.randint(0, 3)
        variables = [f"x{i}" for i in range(n_business)] + [
            f"__slack_{i}" for i in range(n_internal)
        ]
        rng.shuffle(variables)
        internal = {name for name in variables if name.startswith("__")}
        rows = rng.randint(1, 80)
        # Few variables, many rows and only a handful of energies: plenty of
        # duplicates and plenty of ties, exactly as in the binary case.
        samples = [
            {name: rng.randint(-5, 7) for name in variables} for _ in range(rows)
        ]
        energies = [float(rng.choice([-3, -2, -1, 0, 1, 2])) for _ in range(rows)]
        raw = RawSolverResult.from_dicts(samples, energies, "cqm", dtype=np.int64)

        expected = reference_deduplicate(raw, internal)
        candidates = deduplicate_samples(raw, internal)

        assert candidates.variables == [v for v in variables if v not in internal]
        assert candidates.samples.dtype == np.int64
        assert candidates.as_pairs() == expected
        assert candidates.counts.dtype == np.int64
        assert candidates.counts.tolist() == reference_counts(raw, internal)

    @pytest.mark.parametrize("seed", range(25))
    def test_heavily_duplicated_int64_rows_keep_min_energy_and_first_seen(self, seed):
        # The wide-range case above rarely repeats an assignment; here the
        # value range is tiny on purpose so nearly every row is a duplicate
        # and the "minimum energy, earliest read wins ties" rule is what the
        # comparison is actually about.
        rng = random.Random(3000 + seed)
        variables = [f"x{i}" for i in range(rng.randint(1, 3))] + ["__slack_0"]
        rng.shuffle(variables)
        internal = {"__slack_0"}
        rows = rng.randint(2, 80)
        samples = [
            {name: rng.randint(-1, 1) for name in variables} for _ in range(rows)
        ]
        energies = [float(rng.choice([-1, 0, 1])) for _ in range(rows)]
        raw = RawSolverResult.from_dicts(samples, energies, "cqm", dtype=np.int64)

        candidates = deduplicate_samples(raw, internal)

        assert candidates.samples.dtype == np.int64
        assert candidates.as_pairs() == reference_deduplicate(raw, internal)
        assert candidates.counts.tolist() == reference_counts(raw, internal)
        assert int(candidates.counts.sum()) == raw.num_samples

    def test_int8_rows_holding_2_and_minus_1_do_not_take_the_bit_path(self):
        # np.packbits would flatten 2 and -1 to 1; _row_keys must notice the
        # values are not 0/1 even though the dtype is the bit path's int8.
        rng = random.Random(4242)
        variables = ["a", "b", "__s"]
        rows = 60
        samples = [
            {name: rng.choice([-1, 0, 1, 2]) for name in variables}
            for _ in range(rows)
        ]
        energies = [float(rng.choice([-1, 0, 1])) for _ in range(rows)]
        raw = RawSolverResult.from_dicts(samples, energies, "exact")

        assert raw.samples.dtype == np.int8
        candidates = deduplicate_samples(raw, {"__s"})

        assert candidates.samples.dtype == np.int8
        assert sorted(np.unique(candidates.samples).tolist()) == [-1, 0, 1, 2]
        assert candidates.as_pairs() == reference_deduplicate(raw, {"__s"})
        assert candidates.counts.tolist() == reference_counts(raw, {"__s"})

    def test_more_than_64_integer_columns(self):
        # One int64 lexsort key per column, well past the single 64-bit word
        # the bit path would have packed these rows into.
        rng = random.Random(99)
        n = 70
        variables = [f"v{i:03d}" for i in range(n)]
        base = [rng.randint(0, 3) for _ in range(n)]
        rows = 50
        samples = []
        for row in range(rows):
            assignment = dict(zip(variables, base))
            if row % 3:  # every third row repeats the base assignment
                for name in rng.sample(variables, k=rng.randint(1, 5)):
                    assignment[name] = rng.randint(0, 3)
            samples.append(assignment)
        energies = [float(rng.choice([-2, -1, 0, 1])) for _ in range(rows)]
        raw = RawSolverResult.from_dicts(samples, energies, "cqm", dtype=np.int64)

        candidates = deduplicate_samples(raw, set())

        assert candidates.samples.shape[1] == n
        assert candidates.samples.dtype == np.int64
        assert len(candidates) < rows
        assert candidates.as_pairs() == reference_deduplicate(raw, set())
        assert candidates.counts.tolist() == reference_counts(raw, set())
        assert int(candidates.counts.sum()) == rows

    @pytest.mark.parametrize("seed", range(6))
    def test_large_integer_matrices_match_the_row_by_row_reference(self, seed):
        # 50 columns of genuinely different widths, thousands of rows and
        # only four distinct energies: every assignment repeats several
        # times and nearly every repeat is an energy tie.
        rng = np.random.default_rng(9_000 + seed)
        n_variables = _LARGE_INTEGER_VARIABLES
        rows = _LARGE_INTEGER_ROWS
        distinct = distinct_integer_rows(rng, rows // 4, n_variables)
        matrix = np.ascontiguousarray(
            distinct[rng.integers(0, distinct.shape[0], size=rows)]
        )
        energy_pool = np.array([-2.0, -1.0, 0.0, 1.0])
        energies = energy_pool[rng.integers(0, len(energy_pool), size=rows)]
        variables = shuffled_names(rng, "iv", n_variables)
        raw = RawSolverResult(
            variables=variables, samples=matrix, energies=energies, backend="cqm"
        )

        assert raw.samples.dtype == np.int64
        expected = reference_deduplicate(raw, set())
        candidates = deduplicate_samples(raw, set())

        assert candidates.samples.dtype == np.int64
        assert candidates.as_pairs() == expected
        assert candidates.counts.tolist() == reference_counts(raw, set())
        assert int(candidates.counts.sum()) == raw.num_samples
        assert len(candidates) < rows


class TestIntegerProcessCandidatesMatchesRowByRow:
    """The array pipeline on integer rows must rank exactly like the
    row-by-row reference -- same solutions, same order, same numbers."""

    @pytest.mark.parametrize("seed", range(40))
    def test_same_solutions_same_order_same_numbers(self, seed):
        rng = random.Random(5000 + seed)
        n_variables = rng.randint(1, 6)
        problem = random_integer_problem(rng, n_variables)
        bounds = {variable.name: variable.bounds() for variable in problem.variables}
        business = [variable.name for variable in problem.variables]
        internal = {f"__slack_{i}" for i in range(rng.randint(0, 2))}
        variables = business + sorted(internal)
        rng.shuffle(variables)
        rows = rng.randint(1, 60)
        samples = [
            {
                name: (
                    rng.randint(*bounds[name]) if name in bounds else rng.randint(0, 1)
                )
                for name in variables
            }
            for _ in range(rows)
        ]
        energies = [rng.choice([-2.0, -1.0, 0.0, 0.5]) for _ in range(rows)]
        raw = RawSolverResult.from_dicts(samples, energies, "cqm", dtype=np.int64)
        top_k = rng.randint(1, 8)

        assert raw.samples.dtype == np.int64

        processed = process_candidates(problem, raw, internal, top_k)
        solutions = processed.solutions
        expected, expected_unique, expected_feasible = (
            TestProcessCandidatesMatchesRowByRow.reference(
                problem, raw, internal, top_k
            )
        )
        tally = reference_tally(raw, internal)

        assert processed.unique_samples == expected_unique
        assert processed.feasible_samples == expected_feasible
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
            assert solution.sample_count == tally[assignment_key(sample)]
        assert [s.rank for s in solutions] == list(range(1, len(solutions) + 1))
        assert sum(tally.values()) == raw.num_samples
