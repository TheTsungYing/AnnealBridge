"""Hard-penalty dominance over integer variables with a negative lower bound.

2026-09-09 code review, finding F-06. ``compute_objective_scale`` used to
bound ``max|objective|`` (``sum(|c| * max(|lower|, |upper|))``), while the
Phase 1 §18 derivation needs a bound on the objective's *range*
``objective_max - objective_min``. For a binary problem the two coincide;
for an integer variable with ``lower < 0`` the range can be twice the
magnitude bound, so with ``penalty_multiplier`` 1.5 the compiled model's
global minimum was infeasible and with the default 2.0 an infeasible
assignment tied the feasible optimum.

This is the review's counterexample, run end to end through the real
:class:`BQMCompiler` and :class:`ScaledPenaltyStrategy`, with every bit
assignment of the compiled model enumerated: the lowest-energy assignment
must be *unique* and feasible for both multipliers.
"""

import itertools
import math

import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    SolverPreferences,
    Variable,
)
from annealbridge.penalty import ScaledPenaltyStrategy
from annealbridge.validation import validate_problem, validate_solution
from annealbridge.validation.estimates import (
    compute_objective_scale,
    compute_penalty_scale,
    variable_bounds,
)


def counterexample(multiplier: float) -> OptimizationProblem:
    """``x in [-4, 4]``, ``y`` binary, minimize ``x``, hard ``x + 9y == 4``.

    The only feasible assignment is ``x = 4, y = 0`` (objective 4). The
    infeasible ``x = -4, y = 1`` has objective -4 and violates the equality
    by exactly one unit, so it costs ``-4 + lambda``: it beats the feasible
    optimum whenever ``lambda < 8``, i.e. whenever the penalty scale is
    below the objective's true range of 8.
    """
    return OptimizationProblem(
        version="1.1",
        name="review F-06 counterexample",
        variables=[
            Variable(name="x", type="integer", lower_bound=-4, upper_bound=4),
            Variable(name="y"),
        ],
        objective=Objective(
            direction="minimize",
            linear_terms=[LinearTerm(variable="x", coefficient=1)],
        ),
        constraints=[
            Constraint(
                id="pin",
                type="hard",
                terms=[
                    LinearTerm(variable="x", coefficient=1),
                    LinearTerm(variable="y", coefficient=9),
                ],
                operator="==",
                rhs=4,
            )
        ],
        solver=SolverPreferences(penalty_multiplier=multiplier),
    )


def lowest_energy_assignments(compiled) -> list[dict[str, int]]:
    """Decoded business assignments attaining the model's minimum energy."""
    names = [str(variable) for variable in compiled.model.variables]
    best = math.inf
    winners: list[dict[str, int]] = []
    for bits in itertools.product((0, 1), repeat=len(names)):
        sample = dict(zip(names, bits))
        energy = compiled.model.energy(sample)
        if energy < best - 1e-9:
            best, winners = energy, [sample]
        elif abs(energy - best) <= 1e-9:
            winners.append(sample)

    decoded = []
    for sample in winners:
        assignment: dict[str, int] = {}
        for variable in compiled.original_problem.variables:
            encoding = compiled.integer_encodings.get(variable.name)
            if encoding is None:
                assignment[variable.name] = int(sample[variable.name])
            else:
                assignment[variable.name] = encoding.lower + sum(
                    coefficient * sample[bit]
                    for coefficient, bit in zip(encoding.coefficients, encoding.bits)
                )
        decoded.append(assignment)
    return decoded


class TestReviewCounterexample:
    def test_problem_is_valid_and_has_one_feasible_assignment(self):
        problem = counterexample(2.0)
        assert validate_problem(problem) == []
        bounds = variable_bounds(problem)
        feasible = [
            {"x": x, "y": y}
            for x in range(bounds["x"][0], bounds["x"][1] + 1)
            for y in (0, 1)
            if validate_solution(problem, {"x": x, "y": y}).feasible
        ]
        assert feasible == [{"x": 4, "y": 0}]

    def test_objective_scale_is_the_objective_range(self):
        problem = counterexample(2.0)
        bounds = variable_bounds(problem)
        # ``x`` spans -4..4, so the objective ``x`` varies by 8, not by
        # ``max(|-4|, |4|) == 4``.
        assert compute_objective_scale(problem.objective, bounds) == 8.0
        assert compute_penalty_scale(problem) == 8.0

    @pytest.mark.parametrize("multiplier", [1.5, 2.0])
    def test_lowest_energy_assignment_is_unique_and_feasible(self, multiplier):
        problem = counterexample(multiplier)
        hard_penalty = ScaledPenaltyStrategy().initial_penalty(problem)
        compiled = BQMCompiler().compile(problem, hard_penalty)
        winners = lowest_energy_assignments(compiled)
        assert winners == [{"x": 4, "y": 0}]
        assert validate_solution(problem, winners[0]).feasible
        assert hard_penalty == pytest.approx(8.0 * multiplier)

    def test_penalty_below_the_range_is_beaten_by_an_infeasible_assignment(self):
        # Documents *why* the range matters: at the old scale (4.0) times
        # 1.5 the infeasible ``x = -4, y = 1`` is the global minimum, and at
        # exactly the range (8.0) it ties the feasible optimum. This is the
        # behaviour the review reproduced; it is independent of the formula.
        problem = counterexample(1.0)
        assert lowest_energy_assignments(BQMCompiler().compile(problem, 6.0)) == [
            {"x": -4, "y": 1}
        ]
        tied = lowest_energy_assignments(BQMCompiler().compile(problem, 8.0))
        assert sorted(tied, key=lambda a: a["x"]) == [{"x": -4, "y": 1}, {"x": 4, "y": 0}]
