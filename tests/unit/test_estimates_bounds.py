"""Bounds-aware estimates against brute force (3b spec §8, §26.1).

Two families of evidence:

* **Brute force.** On random tiny problems (at most three variables, integer
  ranges of at most four, negative lower bounds, binary and integer mixed)
  every assignment is enumerated and the closed-form numbers are checked
  against it: ``lhs_bounds`` is attained, ``compute_objective_scale`` bounds
  the objective's range ``max - min`` (review F-06, 2026-09-09: the range
  is what the Phase 1 §18 penalty derivation needs, not ``max|objective|``)
  and ``compute_soft_energy_bound`` bounds (indeed equals) the largest soft
  energy the BQM compiler's squared form can produce.
* **Binary degeneracy.** On random all-binary problems every function
  returns the very same value with ``bounds`` passed as without, so the
  Phase 1/2/3a callers that never pass bounds see no change at all.
"""

import itertools
import math
import random

import pytest

from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    QuadraticTerm,
    Variable,
)
from annealbridge.validation import estimates
from annealbridge.validation.estimates import (
    accumulate_terms,
    analyze_inequality,
    compute_objective_scale,
    compute_slack_coefficients,
    compute_soft_energy_bound,
    constraint_bit_count,
    count_slack_bits,
    estimate_compiled_variables,
    estimate_cqm_variables,
    estimate_encoded_interactions,
    integer_encoding_bits,
    lhs_bounds,
    variable_bounds,
)

# --------------------------------------------------------------------------
# Random problem generation
# --------------------------------------------------------------------------

MAX_VARIABLES = 3
MAX_RANGE = 4


def random_variables(rng: random.Random, *, all_binary: bool) -> list[Variable]:
    count = rng.randint(1, MAX_VARIABLES)
    variables = []
    for index in range(count):
        name = f"v{index}"
        if all_binary or rng.random() < 0.3:
            variables.append(Variable(name=name))
        else:
            lower = rng.randint(-4, 2)
            upper = lower + rng.randint(1, MAX_RANGE)
            variables.append(
                Variable(name=name, type="integer", lower_bound=lower, upper_bound=upper)
            )
    return variables


def random_coefficient(rng: random.Random) -> int:
    return rng.choice([-3, -2, -1, 1, 2, 3])


def random_terms(rng: random.Random, names: list[str]) -> list[LinearTerm]:
    # Repeated variables on purpose: accumulation is part of the contract.
    count = rng.randint(1, len(names) + 1)
    return [
        LinearTerm(variable=rng.choice(names), coefficient=random_coefficient(rng))
        for _ in range(count)
    ]


def random_objective(rng: random.Random, variables: list[Variable]) -> Objective:
    names = [variable.name for variable in variables]
    integer_names = [v.name for v in variables if v.type == "integer"]
    quadratic = []
    for _ in range(rng.randint(0, 2)):
        first = rng.choice(names)
        # x*x is legal for an integer variable (3b §9.2), never for a binary.
        second = first if first in integer_names and rng.random() < 0.4 else rng.choice(names)
        if first == second and first not in integer_names:
            continue
        quadratic.append(
            QuadraticTerm(variable1=first, variable2=second, coefficient=random_coefficient(rng))
        )
    return Objective(
        direction=rng.choice(["minimize", "maximize"]),
        linear_terms=random_terms(rng, names) if rng.random() < 0.9 else [],
        quadratic_terms=quadratic,
        constant=rng.randint(-3, 3),
    )


def random_constraint(
    rng: random.Random, names: list[str], identifier: str, *, soft: bool | None = None
) -> Constraint:
    is_soft = rng.random() < 0.5 if soft is None else soft
    operator = rng.choice(["==", "<=", ">="])
    return Constraint(
        id=identifier,
        type="soft" if is_soft else "hard",
        terms=random_terms(rng, names),
        operator=operator,
        rhs=rng.randint(-6, 6),
        weight=float(rng.randint(1, 5)) if is_soft else None,
    )


def random_problem(rng: random.Random, *, all_binary: bool) -> OptimizationProblem:
    variables = random_variables(rng, all_binary=all_binary)
    names = [variable.name for variable in variables]
    constraints = [
        random_constraint(rng, names, f"c{index}") for index in range(rng.randint(0, 3))
    ]
    return OptimizationProblem(
        version="1.0" if all_binary else "1.1",
        name="random",
        variables=variables,
        objective=random_objective(rng, variables),
        constraints=constraints,
    )


def assignments(bounds: dict[str, tuple[int, int]]) -> list[dict[str, int]]:
    names = list(bounds)
    ranges = [range(bounds[name][0], bounds[name][1] + 1) for name in names]
    return [dict(zip(names, values)) for values in itertools.product(*ranges)]


def evaluate_affine(coefficients: dict[str, float], assignment: dict[str, int]) -> float:
    return sum(value * assignment[name] for name, value in coefficients.items())


def evaluate_objective(objective: Objective, assignment: dict[str, int]) -> float:
    total = objective.constant
    total += sum(term.coefficient * assignment[term.variable] for term in objective.linear_terms)
    total += sum(
        term.coefficient * assignment[term.variable1] * assignment[term.variable2]
        for term in objective.quadratic_terms
    )
    return total


def slack_values(slack_range: int) -> list[int]:
    """Every value the compiler's slack bits can represent."""
    coefficients = compute_slack_coefficients(max(slack_range, 0))
    values = set()
    for bits in itertools.product((0, 1), repeat=len(coefficients)):
        values.add(sum(c * b for c, b in zip(coefficients, bits)))
    return sorted(values)


def compiled_soft_energies(
    constraint: Constraint, bounds: dict[str, tuple[int, int]]
) -> list[float]:
    """``weight * (lhs [+ slack] - rhs)^2`` over every assignment and slack value.

    This is exactly the penalty the BQM compiler emits for a soft constraint
    (equality: no slack; inequality: normalized to ``<=`` with the slack
    bits of ``encode_slack``, clamped to none when trivially infeasible).
    """
    weight = constraint.weight
    assert weight is not None
    if constraint.operator == "==":
        coefficients = accumulate_terms(constraint.terms)
        return [
            weight * (evaluate_affine(coefficients, point) - constraint.rhs) ** 2
            for point in assignments(bounds)
        ]
    analysis = analyze_inequality(constraint, bounds)
    if analysis.redundant:
        return [0.0]
    assert analysis.slack_range is not None
    return [
        weight * (evaluate_affine(analysis.coefficients, point) + slack - analysis.rhs) ** 2
        for point in assignments(bounds)
        for slack in slack_values(analysis.slack_range)
    ]


SEEDS = list(range(60))


# --------------------------------------------------------------------------
# Brute force on mixed binary / integer problems
# --------------------------------------------------------------------------


class TestBruteForce:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_lhs_bounds_are_attained(self, seed):
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=False)
        bounds = variable_bounds(problem)
        for constraint in problem.constraints:
            coefficients = accumulate_terms(constraint.terms)
            values = [evaluate_affine(coefficients, point) for point in assignments(bounds)]
            lhs_min, lhs_max = lhs_bounds(coefficients, bounds)
            assert lhs_min == min(values)
            assert lhs_max == max(values)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_objective_scale_bounds_the_objective(self, seed):
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=False)
        bounds = variable_bounds(problem)
        scale = compute_objective_scale(problem.objective, bounds)
        assert scale >= 1.0
        # Spec §18 needs ``penalty_scale >= objective_max - objective_min``:
        # the *range* over the declared bounds (the constant cancels). With
        # negative lower bounds this can exceed ``max|objective|``, and with
        # a range that excludes zero it can be smaller; the bound is on the
        # range, never on the absolute value.
        values = [evaluate_objective(problem.objective, point) for point in assignments(bounds)]
        assert scale >= max(values) - min(values)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_soft_energy_bound_is_the_exact_maximum(self, seed):
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=False)
        bounds = variable_bounds(problem)
        for constraint in problem.constraints:
            bound = compute_soft_energy_bound(constraint, bounds)
            if constraint.type == "hard":
                assert bound == 0.0
                continue
            energies = compiled_soft_energies(constraint, bounds)
            assert bound >= max(energies)
            # The docstring promises the exact maximum for a single constraint.
            assert bound == pytest.approx(max(energies))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_soft_penalty_scale_dominates_every_soft_energy(self, seed):
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=False)
        bounds = variable_bounds(problem)
        soft = [c for c in problem.constraints if c.type == "soft"]
        scale = estimates.compute_penalty_scale(problem)
        # The hard penalty must dominate objective + soft energy whatever the
        # business assignment and whatever the slack bits do, so the worst
        # slack per constraint is taken at every point.
        worst_soft_energy = sum(max(compiled_soft_energies(c, bounds)) for c in soft)
        # Spec §18: ``penalty_scale >= (objective_max - objective_min) +
        # soft_bound``. Any assignment violating a hard constraint by one
        # unit then costs at least ``objective_min + penalty_scale``, which
        # exceeds ``objective_max + soft_bound``, the most the best feasible
        # assignment can cost, as soon as ``multiplier > 1``.
        values = [evaluate_objective(problem.objective, point) for point in assignments(bounds)]
        assert scale >= (max(values) - min(values)) + worst_soft_energy

    @pytest.mark.parametrize("seed", SEEDS)
    def test_slack_range_covers_every_feasible_slack(self, seed):
        # For a non-redundant, feasible inequality the slack bits must reach
        # every value ``rhs - lhs`` takes on a satisfying assignment.
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=False)
        bounds = variable_bounds(problem)
        for constraint in problem.constraints:
            if constraint.operator == "==":
                continue
            analysis = analyze_inequality(constraint, bounds)
            if analysis.redundant or analysis.slack_range is None or analysis.slack_range < 0:
                continue
            needed = {
                int(analysis.rhs - evaluate_affine(analysis.coefficients, point))
                for point in assignments(bounds)
                if evaluate_affine(analysis.coefficients, point) <= analysis.rhs
            }
            assert needed <= set(slack_values(analysis.slack_range))
            assert count_slack_bits(constraint, bounds) == len(
                compute_slack_coefficients(analysis.slack_range)
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_objective_scale_is_tight_for_single_term_objectives(self, seed):
        # The per-term widths are exact, not just upper bounds: an objective
        # made of one term has ``scale == max(1, range)`` bit for bit.
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=False)
        bounds = variable_bounds(problem)
        objective = problem.objective
        singles = [
            Objective(direction="minimize", linear_terms=[term], quadratic_terms=[])
            for term in objective.linear_terms
        ] + [
            Objective(direction="minimize", linear_terms=[], quadratic_terms=[term])
            for term in objective.quadratic_terms
        ]
        for single in singles:
            values = [evaluate_objective(single, point) for point in assignments(bounds)]
            assert compute_objective_scale(single, bounds) == max(1.0, max(values) - min(values))


# --------------------------------------------------------------------------
# Objective scale: hand-computed per-term widths (review F-06, 2026-09-09)
# --------------------------------------------------------------------------


def objective_of(linear=(), quadratic=()) -> Objective:
    return Objective(
        direction="minimize",
        linear_terms=[LinearTerm(variable=name, coefficient=c) for name, c in linear],
        quadratic_terms=[
            QuadraticTerm(variable1=a, variable2=b, coefficient=c) for a, b, c in quadratic
        ],
    )


class TestObjectiveScaleWidths:
    """``compute_objective_scale`` sums the exact range of every raw term.

    Linear ``c * x`` spans ``|c| * (upper - lower)``; a product ``c * x * y``
    of two different variables spans ``|c| * (max P - min P)`` over the four
    corner products ``P``; a square ``c * x * x`` spans ``|c| * (M^2 - m^2)``
    with ``M = max(|lower|, |upper|)`` and ``m = 0`` when the range contains
    zero, else ``min(|lower|, |upper|)``. Summing per-term ranges bounds the
    range of the sum for *any* term list (sub-additivity), which is what
    spec §18 needs.
    """

    def test_linear_term_spans_upper_minus_lower(self):
        # Range 8, while the former ``max(|lower|, |upper|)`` reading gave 4:
        # this is the review's counterexample.
        assert compute_objective_scale(objective_of([("x", 1)]), {"x": (-4, 4)}) == 8.0
        # A range that excludes zero is *narrower* than its magnitude: 3
        # instead of the former 9. The penalty only has to dominate how much
        # the objective can change, never its absolute value.
        assert compute_objective_scale(objective_of([("x", 3)]), {"x": (2, 3)}) == 3.0
        assert compute_objective_scale(objective_of([("x", -2)]), {"x": (-3, -1)}) == 4.0

    def test_square_term_uses_the_true_range_of_x_squared(self):
        # ``x*x`` on -4..4 ranges over 0..16 -> 16. The four corner products
        # would give 16 - (-16) = 32, a valid but loose bound; the square is
        # handled exactly.
        assert compute_objective_scale(objective_of(quadratic=[("x", "x", 1)]), {"x": (-4, 4)}) == 16.0
        # Ranges that exclude zero: 25 - 4 on 2..5 and on -5..-2 alike.
        assert compute_objective_scale(objective_of(quadratic=[("x", "x", 1)]), {"x": (2, 5)}) == 21.0
        assert compute_objective_scale(objective_of(quadratic=[("x", "x", -1)]), {"x": (-5, -2)}) == 21.0
        # A range touching zero at one end: 0..3 -> 9 - 0.
        assert compute_objective_scale(objective_of(quadratic=[("x", "x", 2)]), {"x": (0, 3)}) == 18.0

    def test_product_term_uses_the_four_corners(self):
        # Corners of -2..3 x -1..5: {2, -10, -3, 15} -> 15 - (-10) = 25.
        assert compute_objective_scale(
            objective_of(quadratic=[("x", "y", 1)]), {"x": (-2, 3), "y": (-1, 5)}
        ) == 25.0
        # Integer times binary: corners {0, 2, 0, 3} -> 3, times |c| = 2.
        assert compute_objective_scale(
            objective_of(quadratic=[("x", "y", -2)]), {"x": (2, 3)}
        ) == 6.0
        # Two negative ranges: corners {6, 2, 3, 1} -> 6 - 1 = 5.
        assert compute_objective_scale(
            objective_of(quadratic=[("x", "y", 1)]), {"x": (-3, -1), "y": (-2, -1)}
        ) == 5.0

    def test_terms_are_summed_verbatim_including_duplicates(self):
        # Raw term lists, no accumulation: ``2x - 2x`` still contributes
        # 6 + 6 even though the sum is identically zero. Per-term widths are
        # a valid (sub-additive) bound either way, exactly as before.
        objective = objective_of([("x", 2), ("x", -2)], [("x", "y", 1), ("y", "x", -1)])
        assert compute_objective_scale(objective, {"x": (0, 3), "y": (0, 2)}) == 6.0 + 6.0 + 6.0 + 6.0

    def test_floor_of_one(self):
        assert compute_objective_scale(objective_of(), {}) == 1.0
        assert compute_objective_scale(objective_of([("x", 0.25)]), {"x": (0, 2)}) == 1.0
        assert compute_objective_scale(objective_of([("b", 0.5)]), {"b": (0, 1)}) == 1.0

    def test_review_counterexample_range_dominates_the_infeasible_gain(self):
        # x in -4..4, y binary, minimize x, hard x + 9y == 4: the infeasible
        # x = -4, y = 1 gains 8 over the feasible x = 4, y = 0 and violates
        # by one unit, so any penalty above the scale 8 rules it out.
        scale = compute_objective_scale(objective_of([("x", 1)]), {"x": (-4, 4), "y": (0, 1)})
        assert scale == 8.0
        assert scale >= 4 - (-4)


class TestIntegerEncodingBits:
    @pytest.mark.parametrize(
        "lower, upper, bits",
        [(0, 1, 1), (0, 2, 2), (0, 3, 2), (0, 4, 3), (-3, 4, 3), (5, 5, 0), (-8, -1, 3)],
    )
    def test_bit_count(self, lower, upper, bits):
        assert integer_encoding_bits(lower, upper) == bits
        assert bits == len(compute_slack_coefficients(upper - lower))

    def test_reversed_bounds_raise(self):
        with pytest.raises(ValueError):
            integer_encoding_bits(3, 2)

    @pytest.mark.parametrize("span", range(0, 20))
    def test_expansion_reaches_every_value_and_nothing_more(self, span):
        assert set(slack_values(span)) == set(range(span + 1))


class TestVariableBounds:
    def test_binary_and_integer(self):
        problem = OptimizationProblem(
            version="1.1",
            name="p",
            variables=[
                Variable(name="b"),
                Variable(name="i", type="integer", lower_bound=-2, upper_bound=7),
            ],
            objective=Objective(direction="minimize", linear_terms=[]),
            constraints=[],
        )
        assert variable_bounds(problem) == {"b": (0, 1), "i": (-2, 7)}

    def test_incomplete_integer_bounds_raise(self):
        problem = OptimizationProblem(
            version="1.1",
            name="p",
            variables=[Variable(name="i", type="integer", lower_bound=0)],
            objective=Objective(direction="minimize", linear_terms=[]),
            constraints=[],
        )
        with pytest.raises(ValueError):
            variable_bounds(problem)


def _hand_problem(constraints: list[Constraint], quadratic: list[QuadraticTerm]):
    return OptimizationProblem(
        version="1.1",
        name="hand",
        variables=[
            Variable(name="x", type="integer", lower_bound=0, upper_bound=7),  # 3 bits
            Variable(name="y", type="integer", lower_bound=0, upper_bound=3),  # 2 bits
            Variable(name="b"),
        ],
        objective=Objective(
            direction="minimize",
            linear_terms=[LinearTerm(variable="x", coefficient=1)],
            quadratic_terms=quadratic,
        ),
        constraints=constraints,
    )


class TestHandComputedEstimates:
    def test_compiled_variables_count_encoding_and_slack_bits(self):
        # x + y <= 5: lhs range [0, 10], slack range 5 -> 3 bits.
        problem = _hand_problem(
            [
                Constraint(
                    id="cap",
                    type="hard",
                    terms=[LinearTerm(variable="x", coefficient=1), LinearTerm(variable="y", coefficient=1)],
                    operator="<=",
                    rhs=5,
                )
            ],
            [],
        )
        assert estimate_compiled_variables(problem) == 3 + 2 + 1 + 3
        assert constraint_bit_count(problem.constraints[0], variable_bounds(problem)) == 8

    def test_encoded_interactions(self):
        problem = _hand_problem(
            [
                Constraint(
                    id="cap",
                    type="hard",
                    terms=[LinearTerm(variable="x", coefficient=1), LinearTerm(variable="y", coefficient=1)],
                    operator="<=",
                    rhs=5,
                ),
                # Always true: contributes nothing, like the compiler.
                Constraint(
                    id="loose",
                    type="hard",
                    terms=[LinearTerm(variable="x", coefficient=1)],
                    operator="<=",
                    rhs=100,
                ),
                # b + x == 2 couples 1 + 3 bits.
                Constraint(
                    id="eq",
                    type="hard",
                    terms=[LinearTerm(variable="b", coefficient=1), LinearTerm(variable="x", coefficient=1)],
                    operator="==",
                    rhs=2,
                ),
            ],
            [
                QuadraticTerm(variable1="x", variable2="y", coefficient=1.0),  # 3 * 2
                QuadraticTerm(variable1="x", variable2="x", coefficient=1.0),  # 3 * 2 / 2
                QuadraticTerm(variable1="b", variable2="y", coefficient=1.0),  # 1 * 2
            ],
        )
        expected = (3 * 2) + (3 * 2 // 2) + (1 * 2) + (8 * 7 // 2) + 0 + (4 * 3 // 2)
        assert estimate_encoded_interactions(problem) == expected

    def test_all_binary_problem_has_pairwise_interactions_only(self):
        problem = OptimizationProblem(
            name="bin",
            variables=[Variable(name="a"), Variable(name="b"), Variable(name="c")],
            objective=Objective(
                direction="minimize",
                linear_terms=[],
                quadratic_terms=[QuadraticTerm(variable1="a", variable2="b", coefficient=1.0)],
            ),
            constraints=[
                Constraint(
                    id="one",
                    type="hard",
                    terms=[LinearTerm(variable=n, coefficient=1) for n in ("a", "b", "c")],
                    operator="==",
                    rhs=1,
                )
            ],
        )
        assert estimate_encoded_interactions(problem) == 1 + 3
        assert estimate_cqm_variables(problem) == 3
        assert estimate_compiled_variables(problem) == 3


# --------------------------------------------------------------------------
# All-binary problems: bounds-aware == bounds-free, function by function
# --------------------------------------------------------------------------


def outcome(function, *args, **kwargs):
    """Value or exception type, so raising paths compare too."""
    try:
        return ("value", function(*args, **kwargs))
    except ValueError as exc:
        return ("raises", type(exc), str(exc))


BINARY_SEEDS = list(range(250))


class TestBinaryDegeneracy:
    @pytest.mark.parametrize("seed", BINARY_SEEDS)
    def test_every_function_agrees_with_and_without_bounds(self, seed):
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=True)
        bounds = variable_bounds(problem)
        assert all(value == (0, 1) for value in bounds.values())

        for variant in (bounds, {}):
            assert compute_objective_scale(problem.objective) == compute_objective_scale(
                problem.objective, variant
            )
            for constraint in problem.constraints:
                coefficients = accumulate_terms(constraint.terms)
                assert lhs_bounds(coefficients) == lhs_bounds(coefficients, variant)
                assert estimates._max_abs_affine(
                    coefficients, -constraint.rhs
                ) == estimates._max_abs_affine(coefficients, -constraint.rhs, variant)
                assert compute_soft_energy_bound(constraint) == compute_soft_energy_bound(
                    constraint, variant
                )
                if constraint.operator != "==":
                    assert analyze_inequality(constraint) == analyze_inequality(
                        constraint, variant
                    )
                assert outcome(count_slack_bits, constraint) == outcome(
                    count_slack_bits, constraint, variant
                )
                assert outcome(constraint_bit_count, constraint) == outcome(
                    constraint_bit_count, constraint, variant
                )

    @pytest.mark.parametrize("seed", BINARY_SEEDS[:50])
    def test_signs_of_results_are_identical_too(self, seed):
        # ``==`` treats 0.0 and -0.0 alike; the messages that print these
        # numbers must not change either, so compare the repr as well.
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=True)
        bounds = variable_bounds(problem)
        for constraint in problem.constraints:
            coefficients = accumulate_terms(constraint.terms)
            assert repr(lhs_bounds(coefficients)) == repr(lhs_bounds(coefficients, bounds))
        assert repr(compute_objective_scale(problem.objective)) == repr(
            compute_objective_scale(problem.objective, bounds)
        )

    def test_infinite_coefficient_does_not_hide_behind_bounds(self):
        # Sanity: the arithmetic is plain floats either way.
        coefficients = {"a": math.inf}
        assert lhs_bounds(coefficients) == lhs_bounds(coefficients, {"a": (0, 1)})

    @pytest.mark.parametrize("seed", BINARY_SEEDS)
    def test_objective_scale_equals_the_phase1_magnitude_formula(self, seed):
        # Review F-06 (2026-09-09) replaced ``sum(|c| * M)`` with per-term
        # ranges. For a binary variable ``upper - lower == 1 == max(|lower|,
        # |upper|)`` and the product corners are {0, 0, 0, 1}, so the two
        # formulas are the same arithmetic (``abs(c) * 1``) term by term: the
        # 1.0 golden numbers cannot move. ``phase1_magnitude_scale`` is the
        # former implementation, kept here verbatim as the reference.
        rng = random.Random(seed)
        problem = random_problem(rng, all_binary=True)
        bounds = variable_bounds(problem)
        assert repr(compute_objective_scale(problem.objective, bounds)) == repr(
            phase1_magnitude_scale(problem.objective, bounds)
        )
        assert repr(compute_objective_scale(problem.objective)) == repr(
            phase1_magnitude_scale(problem.objective, {})
        )

    def test_reference_formula_really_is_the_old_one(self):
        # The reference must disagree exactly where the review found the
        # defect, otherwise the binary comparison above proves nothing.
        objective = Objective(
            direction="minimize",
            linear_terms=[LinearTerm(variable="x", coefficient=1)],
            quadratic_terms=[],
        )
        assert phase1_magnitude_scale(objective, {"x": (-4, 4)}) == 4.0
        assert compute_objective_scale(objective, {"x": (-4, 4)}) == 8.0


def phase1_magnitude_scale(objective: Objective, bounds: dict[str, tuple[int, int]]) -> float:
    """The pre-F-06 ``compute_objective_scale``: ``max(1, sum(|c| * M))``."""

    def magnitude(name: str) -> int:
        lower, upper = bounds.get(name, (0, 1))
        return max(abs(lower), abs(upper))

    total = sum(abs(term.coefficient) * magnitude(term.variable) for term in objective.linear_terms)
    total += sum(
        abs(term.coefficient) * magnitude(term.variable1) * magnitude(term.variable2)
        for term in objective.quadratic_terms
    )
    return max(1.0, total)
