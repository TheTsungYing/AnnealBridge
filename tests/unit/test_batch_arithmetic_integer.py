"""Batch arithmetic on integer-valued candidates (3b spec §26.1, §11).

Phase 3a proved the two validation kernels agree on 0/1 matrices
(``tests/unit/test_candidate_arrays.py::TestBatchValidationConsistency``).
IR 1.1 widens the candidate matrix to arbitrary integers -- negative values
included -- and widens its dtype from ``int8`` to ``int64``, so the same
proof is repeated here on integer rows: for every row of a random matrix

* ``validate_batch(...).feasible[row]`` is the very same verdict
  ``validate_solution`` reports for that row's dict, and
* ``soft_violation_score`` / ``evaluate_objective_batch`` are *exactly* the
  floats the scalar functions return -- bit for bit, no tolerance, because
  the ranking is computed from the batch numbers and reported from the
  scalar ones (3a §32.1).

The coefficients are deliberately awkward (0.1, 0.3, 2.25, -0.7 ...): they
do not sum exactly in binary, so any difference in summation order or
association between the two kernels shows up as a last-bit mismatch rather
than hiding inside a tolerance. The objective carries ``x*x`` terms, which
IR 1.1 allows for integer variables only, so the quadratic path is
exercised with values other than 0 and 1.

The last class pins the dtype independence: the same equality holds for an
``int8`` matrix that carries 2 and -1, values ``np.packbits`` would
silently fold into 1 (3b §11).
"""

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
from annealbridge.orchestration import evaluate_objective, evaluate_objective_batch
from annealbridge.validation import validate_batch, validate_solution
from annealbridge.validation.estimates import variable_bounds

# Fractions that do not sum exactly in binary, mixed with exact ones.
COEFFICIENTS = [0.1, 0.3, -0.7, 2.25, 1.0, -1.5, 0.2, 3.0, -0.1, 7.0]
WEIGHTS = [0.5, 1.0, 1.3, 2.0]


# --------------------------------------------------------------------------
# Random IR 1.1 problems
# --------------------------------------------------------------------------


def random_variables(rng: random.Random, count: int) -> list[Variable]:
    """A mix of binary and bounded-integer variables; bounds span zero."""
    variables: list[Variable] = []
    for index in range(count):
        name = f"v{index}"
        if rng.random() < 0.4:
            variables.append(Variable(name=name))
        else:
            lower = rng.randint(-6, 2)
            upper = lower + rng.randint(1, 9)
            variables.append(
                Variable(
                    name=name, type="integer", lower_bound=lower, upper_bound=upper
                )
            )
    # At least one integer variable, or the test degenerates into the 3a case.
    if all(variable.type == "binary" for variable in variables):
        index = rng.randrange(count)
        lower = rng.randint(-6, 2)
        variables[index] = Variable(
            name=variables[index].name,
            type="integer",
            lower_bound=lower,
            upper_bound=lower + rng.randint(1, 9),
        )
    return variables


def random_terms(
    rng: random.Random, names: list[str], count: int
) -> list[LinearTerm]:
    # Repeated variables on purpose: term-by-term accumulation, in term
    # order, is part of the contract both kernels must share.
    return [
        LinearTerm(variable=rng.choice(names), coefficient=rng.choice(COEFFICIENTS))
        for _ in range(count)
    ]


def random_objective(rng: random.Random, variables: list[Variable]) -> Objective:
    names = [variable.name for variable in variables]
    integer_names = [v.name for v in variables if v.type == "integer"]
    quadratic: list[QuadraticTerm] = []
    for _ in range(rng.randint(0, 3)):
        first = rng.choice(names)
        # x*x is legal for an integer variable only (3b §9.2).
        if first in integer_names and rng.random() < 0.5:
            second = first
        else:
            second = rng.choice(names)
            if second == first and first not in integer_names:
                continue
        quadratic.append(
            QuadraticTerm(
                variable1=first, variable2=second, coefficient=rng.choice(COEFFICIENTS)
            )
        )
    return Objective(
        direction=rng.choice(["minimize", "maximize"]),
        linear_terms=random_terms(rng, names, rng.randint(1, len(names) + 1)),
        quadratic_terms=quadratic,
        constant=rng.choice([0.0, 0.1, -2.5, 4.0]),
    )


def random_constraints(
    rng: random.Random,
    variables: list[Variable],
    bounds: dict[str, tuple[int, int]],
) -> list[Constraint]:
    names = [variable.name for variable in variables]
    constraints: list[Constraint] = []
    for index in range(rng.randint(1, 4)):
        terms = random_terms(rng, names, rng.randint(1, len(names) + 1))
        kind = rng.choice(["hard", "soft", "soft"])
        # Pick the rhs off a real assignment most of the time so "==" is
        # reachable and the feasible / infeasible split is non-degenerate.
        if rng.random() < 0.7:
            assignment = {
                name: rng.randint(*bounds[name]) for name in names
            }
            rhs = sum(term.coefficient * assignment[term.variable] for term in terms)
        else:
            rhs = rng.choice(COEFFICIENTS) * rng.randint(-3, 3)
        constraints.append(
            Constraint(
                id=f"c{index}",
                type=kind,
                terms=terms,
                operator=rng.choice(["==", "<=", ">="]),
                rhs=rhs,
                weight=rng.choice(WEIGHTS) if kind == "soft" else None,
            )
        )
    return constraints


def random_problem(rng: random.Random) -> OptimizationProblem:
    variables = random_variables(rng, rng.randint(1, 6))
    bounds = {variable.name: variable.bounds() for variable in variables}
    return OptimizationProblem(
        version="1.1",
        name="random-integer",
        variables=variables,
        objective=random_objective(rng, variables),
        constraints=random_constraints(rng, variables, bounds),
    )


def random_matrix(
    rng: random.Random,
    problem: OptimizationProblem,
    order: list[str],
    rows: int,
) -> np.ndarray:
    """An ``int64`` matrix whose column ``j`` holds values of ``order[j]``."""
    bounds = variable_bounds(problem)
    return np.array(
        [[rng.randint(*bounds[name]) for name in order] for _ in range(rows)],
        dtype=np.int64,
    ).reshape(rows, len(order))


def assert_row_by_row(
    problem: OptimizationProblem, order: list[str], samples: np.ndarray, seed: object
) -> None:
    """The shared assertion: batch == scalar, exactly, on every row."""
    batch = validate_batch(problem, order, samples)
    objective = evaluate_objective_batch(problem.objective, order, samples)

    assert batch.feasible.dtype == bool
    assert batch.soft_violation_score.dtype == np.float64
    assert objective.dtype == np.float64
    assert batch.feasible.shape == (samples.shape[0],)

    for row in range(samples.shape[0]):
        sample = dict(zip(order, samples[row].tolist()))
        full = validate_solution(problem, sample)
        assert bool(batch.feasible[row]) is full.feasible, (seed, row, sample)
        # Exact float equality on purpose: no pytest.approx anywhere here.
        assert float(batch.soft_violation_score[row]) == full.soft_violation_score, (
            seed,
            row,
            sample,
        )
        assert float(objective[row]) == evaluate_objective(
            problem.objective, sample
        ), (seed, row, sample)


# --------------------------------------------------------------------------
# The proof
# --------------------------------------------------------------------------


class TestIntegerBatchMatchesRowByRow:
    @pytest.mark.parametrize("seed", range(60))
    def test_every_candidate_agrees_with_the_scalar_validator(self, seed):
        rng = random.Random(seed)
        problem = random_problem(rng)
        order = [variable.name for variable in problem.variables]
        rng.shuffle(order)  # column order need not be the problem's order
        samples = random_matrix(rng, problem, order, rng.randint(1, 40))

        assert samples.dtype == np.int64
        assert_row_by_row(problem, order, samples, seed)

    def test_the_corpus_is_not_degenerate(self):
        """Guards on the generator itself.

        Without these the equality above could hold vacuously: on an
        all-binary corpus it would only repeat the 3a proof, on an
        all-non-negative one it would say nothing about signed arithmetic,
        and on an all-feasible one the ``feasible`` half would be untested.
        """
        verdicts: set[bool] = set()
        integer_variables = 0
        negative_values = 0
        values_above_one = 0
        for seed in range(60):
            rng = random.Random(seed)
            problem = random_problem(rng)
            integer_variables += sum(
                1 for v in problem.variables if v.type == "integer"
            )
            order = [variable.name for variable in problem.variables]
            samples = random_matrix(rng, problem, order, 20)
            negative_values += int((samples < 0).sum())
            values_above_one += int((samples > 1).sum())
            verdicts.update(validate_batch(problem, order, samples).feasible.tolist())

        assert verdicts == {True, False}, verdicts
        assert integer_variables >= 60, integer_variables
        assert negative_values > 0, "no negative candidate value was generated"
        assert values_above_one > 0, "no candidate value above 1 was generated"


class TestDtypeIndependence:
    """3b §11: the batch arithmetic must not depend on the sample dtype.

    ``int8`` is the BQM backends' bit dtype, but a decoded matrix may be
    ``int8`` and still carry values ``np.packbits`` would fold into 1.
    """

    @staticmethod
    def _problem() -> OptimizationProblem:
        return OptimizationProblem(
            version="1.1",
            name="int8-with-2-and-minus-1",
            variables=[
                Variable(name="x", type="integer", lower_bound=-1, upper_bound=2),
                Variable(name="y", type="integer", lower_bound=-1, upper_bound=2),
                Variable(name="b"),
            ],
            objective=Objective(
                direction="minimize",
                linear_terms=[
                    LinearTerm(variable="x", coefficient=0.1),
                    LinearTerm(variable="y", coefficient=-0.7),
                    LinearTerm(variable="b", coefficient=2.25),
                ],
                quadratic_terms=[
                    QuadraticTerm(variable1="x", variable2="x", coefficient=0.3),
                    QuadraticTerm(variable1="x", variable2="y", coefficient=-1.5),
                ],
                constant=0.1,
            ),
            constraints=[
                Constraint(
                    id="hard-sum",
                    type="hard",
                    terms=[
                        LinearTerm(variable="x", coefficient=0.1),
                        LinearTerm(variable="y", coefficient=0.2),
                    ],
                    operator="<=",
                    rhs=0.3,
                ),
                Constraint(
                    id="soft-eq",
                    type="soft",
                    terms=[
                        LinearTerm(variable="x", coefficient=1.0),
                        LinearTerm(variable="b", coefficient=-0.7),
                    ],
                    operator="==",
                    rhs=0.3,
                    weight=1.3,
                ),
            ],
        )

    @staticmethod
    def _rows() -> np.ndarray:
        return np.array(
            [
                [2, -1, 1],
                [-1, 2, 0],
                [0, 0, 1],
                [2, 2, 1],
                [-1, -1, 0],
                [1, -1, 1],
                [2, -1, 1],  # duplicate row on purpose
            ],
            dtype=np.int8,
        )

    def test_int8_matrix_with_2_and_minus_1(self):
        problem = self._problem()
        order = ["x", "y", "b"]
        samples = self._rows()

        assert samples.dtype == np.int8
        assert samples.min() < 0 and samples.max() > 1
        assert_row_by_row(problem, order, samples, "int8")

    def test_int8_and_int64_agree_bit_for_bit(self):
        problem = self._problem()
        order = ["x", "y", "b"]
        narrow = self._rows()
        wide = narrow.astype(np.int64)

        narrow_batch = validate_batch(problem, order, narrow)
        wide_batch = validate_batch(problem, order, wide)

        assert narrow_batch.feasible.tolist() == wide_batch.feasible.tolist()
        assert (
            narrow_batch.soft_violation_score.tolist()
            == wide_batch.soft_violation_score.tolist()
        )
        assert (
            evaluate_objective_batch(problem.objective, order, narrow).tolist()
            == evaluate_objective_batch(problem.objective, order, wide).tolist()
        )

    def test_shuffled_columns_give_the_same_verdicts(self):
        problem = self._problem()
        samples = self._rows()
        straight = validate_batch(problem, ["x", "y", "b"], samples)
        shuffled = validate_batch(
            problem, ["b", "y", "x"], samples[:, [2, 1, 0]].copy()
        )

        assert straight.feasible.tolist() == shuffled.feasible.tolist()
        assert (
            straight.soft_violation_score.tolist()
            == shuffled.soft_violation_score.tolist()
        )
