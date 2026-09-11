"""Unit tests for the penalty strategy (spec §18, §33).

``penalty_scale = objective_scale + soft energy bound``: the objective-only
``objective_scale`` keeps its meaning (what an agent compares soft weights
against), while the hard penalty is sized against the whole non-penalty
energy landscape so a large soft weight can never drown a hard constraint.
"""

import itertools
import logging

import dimod
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    QuadraticTerm,
    SolverPreferences,
    Variable,
)
from annealbridge.orchestration import OptimizationService
from annealbridge.penalty import (
    ScaledPenaltyStrategy,
    compute_objective_scale,
    compute_penalty_scale,
)
from annealbridge.solvers import SolverRegistry
from annealbridge.validation.estimates import compute_soft_energy_bound
from tests.fakes.local_cqm_backend import FAKE_LOCAL_CQM_NAME, FakeLocalCQMBackend


def make_problem(
    penalty_multiplier: float = 2.0,
    extra_constraints: list[Constraint] | None = None,
) -> OptimizationProblem:
    """A small problem with |linear| = 10 + 8 and |quadratic| = 3 -> scale 21."""
    return OptimizationProblem(
        name="penalty-test",
        variables=[Variable(name="x"), Variable(name="y")],
        objective=Objective(
            direction="maximize",
            linear_terms=[
                LinearTerm(variable="x", coefficient=10),
                LinearTerm(variable="y", coefficient=-8),
            ],
            quadratic_terms=[
                QuadraticTerm(variable1="x", variable2="y", coefficient=-3)
            ],
        ),
        constraints=[
            Constraint(
                id="cap",
                type="hard",
                terms=[
                    LinearTerm(variable="x", coefficient=1),
                    LinearTerm(variable="y", coefficient=1),
                ],
                operator="<=",
                rhs=1,
            ),
            *(extra_constraints or []),
        ],
        solver=SolverPreferences(penalty_multiplier=penalty_multiplier),
    )


def soft(constraint_id: str, operator: str, rhs: float, weight: float) -> Constraint:
    """Soft ``x + y <op> rhs`` with the given weight."""
    return Constraint(
        id=constraint_id,
        type="soft",
        weight=weight,
        terms=[
            LinearTerm(variable="x", coefficient=1),
            LinearTerm(variable="y", coefficient=1),
        ],
        operator=operator,
        rhs=rhs,
    )


EXPECTED_SCALE = 21.0  # |10| + |-8| + |-3|

# soft x + y == 0 with weight 1000: max |x + y - 0| over binaries is 2,
# so the soft term can add up to 1000 * 2^2 = 4000 energy.
BIG_SOFT = soft("prefer-empty", "==", 0, 1000.0)
BIG_SOFT_BOUND = 4000.0


class TestScaledPenaltyStrategy:
    def test_objective_scale(self):
        strategy = ScaledPenaltyStrategy()
        assert strategy.objective_scale(make_problem()) == EXPECTED_SCALE

    def test_penalty_scale_equals_objective_scale_without_soft_constraints(self):
        # Phase 1 behaviour must be byte-for-byte unchanged for problems
        # that only have hard constraints (spec §18).
        strategy = ScaledPenaltyStrategy()
        problem = make_problem()
        assert strategy.penalty_scale(problem) == strategy.objective_scale(problem)
        assert strategy.penalty_scale(problem) == EXPECTED_SCALE

    def test_initial_penalty_uses_scale_and_multiplier(self):
        strategy = ScaledPenaltyStrategy()
        assert strategy.initial_penalty(make_problem()) == pytest.approx(
            EXPECTED_SCALE * 2.0
        )

    def test_objective_scale_ignores_soft_constraints(self):
        # objective_scale stays objective-only: it is the reference the agent
        # (and the SOFT_WEIGHT_SMALL warning) compares weights against, so
        # folding weights into it would be circular.
        strategy = ScaledPenaltyStrategy()
        problem = make_problem(extra_constraints=[BIG_SOFT])
        assert strategy.objective_scale(problem) == EXPECTED_SCALE
        assert compute_objective_scale(problem.objective) == EXPECTED_SCALE

    def test_penalty_scale_adds_soft_energy_bound(self):
        strategy = ScaledPenaltyStrategy()
        problem = make_problem(extra_constraints=[BIG_SOFT])
        assert strategy.penalty_scale(problem) == pytest.approx(
            EXPECTED_SCALE + BIG_SOFT_BOUND
        )
        assert compute_penalty_scale(problem) == strategy.penalty_scale(problem)

    def test_initial_penalty_dominates_soft_terms(self):
        strategy = ScaledPenaltyStrategy()
        problem = make_problem(extra_constraints=[BIG_SOFT])
        assert strategy.initial_penalty(problem) == pytest.approx(
            (EXPECTED_SCALE + BIG_SOFT_BOUND) * 2.0
        )
        # The hard penalty is derived from the problem structure, never
        # copied from a weight (§10.4): it is neither the weight nor a
        # multiple of the weight alone.
        assert strategy.initial_penalty(problem) > BIG_SOFT.weight

    def test_deterministic_for_same_problem(self):
        strategy = ScaledPenaltyStrategy()
        for factory in (
            make_problem,
            lambda: make_problem(extra_constraints=[BIG_SOFT]),
        ):
            first = strategy.initial_penalty(factory())
            second = strategy.initial_penalty(factory())
            assert first == second
            assert strategy.objective_scale(factory()) == strategy.objective_scale(
                factory()
            )
            assert strategy.penalty_scale(factory()) == strategy.penalty_scale(
                factory()
            )

    def test_next_penalty_doubles_each_retry(self):
        strategy = ScaledPenaltyStrategy()
        penalty = strategy.initial_penalty(make_problem())
        for attempt in range(1, 4):
            following = strategy.next_penalty(penalty, attempt)
            assert following == pytest.approx(penalty * 2.0)
            assert following > penalty
            penalty = following

    def test_penalty_multiplier_override(self):
        strategy = ScaledPenaltyStrategy()
        assert strategy.initial_penalty(
            make_problem(penalty_multiplier=0.5)
        ) == pytest.approx(EXPECTED_SCALE * 0.5)
        assert strategy.initial_penalty(
            make_problem(penalty_multiplier=10.0)
        ) == pytest.approx(EXPECTED_SCALE * 10.0)


def soft_only_problem(*constraints: Constraint, variables=("x", "y", "z")) -> OptimizationProblem:
    """Zero objective + the given soft constraints, for brute-force checks."""
    return OptimizationProblem(
        name="soft-only",
        variables=[Variable(name=name) for name in variables],
        # A single zero-coefficient term keeps the objective energy at 0 and
        # objective_scale at its floor of 1.0.
        objective=Objective(
            direction="minimize",
            linear_terms=[LinearTerm(variable=variables[0], coefficient=0)],
        ),
        constraints=list(constraints),
    )


def max_compiled_energy(problem: OptimizationProblem) -> float:
    """Exhaustively maximize the compiled BQM energy (slack bits included)."""
    compiled = BQMCompiler().compile(problem, hard_penalty=1.0)
    bqm: dimod.BinaryQuadraticModel = compiled.model
    variables = list(bqm.variables)
    return max(
        bqm.energy(dict(zip(variables, bits)))
        for bits in itertools.product((0, 1), repeat=len(variables))
    )


def three_var(constraint_id, operator, rhs, weight, coefficients=(2, -3, 5)) -> Constraint:
    return Constraint(
        id=constraint_id,
        type="soft",
        weight=weight,
        terms=[
            LinearTerm(variable=name, coefficient=value)
            for name, value in zip(("x", "y", "z"), coefficients)
        ],
        operator=operator,
        rhs=rhs,
    )


class TestSoftEnergyBound:
    """``compute_soft_energy_bound`` must never under-estimate the compiler."""

    def test_hard_constraint_contributes_nothing(self):
        hard = Constraint(
            id="h",
            type="hard",
            terms=[LinearTerm(variable="x", coefficient=1)],
            operator="==",
            rhs=1,
        )
        assert compute_soft_energy_bound(hard) == 0.0

    def test_equality_bound_formula(self):
        # lhs in [-3, 7], rhs 1 -> max |lhs - rhs| = 6 -> 10 * 36
        constraint = three_var("eq", "==", 1, 10.0)
        assert compute_soft_energy_bound(constraint) == pytest.approx(360.0)

    def test_equality_matches_exhaustive_maximum(self):
        constraint = three_var("eq", "==", 1, 10.0)
        problem = soft_only_problem(constraint)
        assert max_compiled_energy(problem) == pytest.approx(
            compute_soft_energy_bound(constraint)
        )

    @pytest.mark.parametrize("operator, rhs", [("<=", 4), (">=", 0), ("<=", 6), (">=", -2)])
    def test_inequality_matches_exhaustive_maximum_with_slack(self, operator, rhs):
        constraint = three_var("ineq", operator, rhs, 3.0)
        problem = soft_only_problem(constraint)
        bound = compute_soft_energy_bound(constraint)
        assert bound > 0.0
        assert max_compiled_energy(problem) == pytest.approx(bound)

    def test_redundant_inequality_is_zero(self):
        # lhs max is 7 <= 100: the compiler emits nothing for it.
        constraint = three_var("redundant", "<=", 100, 50.0)
        assert compute_soft_energy_bound(constraint) == 0.0
        assert max_compiled_energy(soft_only_problem(constraint)) == pytest.approx(0.0)

    def test_trivially_infeasible_soft_inequality_uses_clamped_slack(self):
        # lhs min is -3 > rhs -10: the compiler clamps to zero slack bits, so
        # the bound must follow the same (slack-free) squared term.
        constraint = three_var("never", "<=", -10, 2.0)
        bound = compute_soft_energy_bound(constraint)
        assert bound == pytest.approx(2.0 * (7 + 10) ** 2)
        assert max_compiled_energy(soft_only_problem(constraint)) == pytest.approx(bound)

    def test_duplicate_terms_are_accumulated_first(self):
        # x + x - y == 0 accumulates to 2x - y: lhs in [-1, 2] -> D = 2.
        constraint = Constraint(
            id="dup",
            type="soft",
            weight=1.0,
            terms=[
                LinearTerm(variable="x", coefficient=1),
                LinearTerm(variable="x", coefficient=1),
                LinearTerm(variable="y", coefficient=-1),
            ],
            operator="==",
            rhs=0,
        )
        assert compute_soft_energy_bound(constraint) == pytest.approx(4.0)

    def test_sum_over_constraints_is_conservative(self):
        # Per-constraint maxima are summed; the exhaustive maximum of the
        # combined BQM can never exceed that sum.
        constraints = [
            three_var("eq", "==", 1, 10.0),
            three_var("le", "<=", 2, 4.0, coefficients=(1, 1, 1)),
            three_var("ge", ">=", 1, 7.0, coefficients=(1, -1, 2)),
        ]
        problem = soft_only_problem(*constraints)
        summed = sum(compute_soft_energy_bound(c) for c in constraints)
        assert max_compiled_energy(problem) <= summed + 1e-9
        assert compute_penalty_scale(problem) == pytest.approx(1.0 + summed)


class SpyStrategy:
    """A ``PenaltyStrategy`` that counts the *trace-only* scale calls.

    ``initial_penalty`` / ``next_penalty`` delegate to a private
    ``ScaledPenaltyStrategy``, so the penalty ladder behaves exactly as in
    production and the inner strategy's own use of ``penalty_scale`` is not
    counted: the counters only move when the *service* asks for a scale.
    """

    def __init__(self) -> None:
        self._inner = ScaledPenaltyStrategy()
        self.objective_scale_calls = 0
        self.penalty_scale_calls = 0

    def initial_penalty(self, problem):
        return self._inner.initial_penalty(problem)

    def next_penalty(self, previous, attempt):
        return self._inner.next_penalty(previous, attempt)

    def objective_scale(self, problem):
        self.objective_scale_calls += 1
        return self._inner.objective_scale(problem)

    def penalty_scale(self, problem):
        self.penalty_scale_calls += 1
        return self._inner.penalty_scale(problem)


class TestServiceDoesNotComputeScalesForDisabledLogging:
    """2026-09-09 review F-26b: the two scales only feed one INFO trace line.

    Both walk every objective term and every soft constraint, which on a
    large problem costs real time on every solve — paid even when the log
    line is thrown away. The call must sit behind ``isEnabledFor(INFO)``.
    The guarded logger is ``annealbridge.orchestration.optimizer``, so the
    ``annealbridge`` level decides.

    On the hard-penalty path the penalty scale has already been walked once
    inside ``initial_penalty``, so the trace derives it from the penalty and
    the service never calls ``penalty_scale`` itself; only the CQM
    (native-constraint) path, which computes no penalty, calls it once.
    """

    @staticmethod
    def solve_knapsack(load_example) -> tuple[SpyStrategy, object]:
        spy = SpyStrategy()
        problem = OptimizationProblem.model_validate(
            load_example("knapsack.json", backend="exact")
        )
        result = OptimizationService(penalty_strategy=spy).solve(problem)
        assert result.status == "success"
        return spy, result

    @staticmethod
    def solve_knapsack_on_cqm(load_example) -> tuple[SpyStrategy, object]:
        """The same problem routed to the test-only local CQM backend.

        ``supported_model_types=["cqm"]`` is what makes the service pick
        ``CQMCompiler``, whose ``uses_hard_penalty`` is False; the backend
        name is not in ``SolverPreferences.backend``'s Literal, so it is set
        with ``model_construct`` (as ``test_service_cqm_flow.py`` does).
        """
        spy = SpyStrategy()
        problem = OptimizationProblem.model_validate(load_example("knapsack.json"))
        problem = problem.model_copy(
            update={"solver": SolverPreferences.model_construct(backend=FAKE_LOCAL_CQM_NAME)}
        )
        defaults = SolverRegistry.default()
        backends = {name: defaults.get(name) for name in defaults.names()}
        backends[FAKE_LOCAL_CQM_NAME] = FakeLocalCQMBackend()
        service = OptimizationService(
            registry=SolverRegistry(backends), penalty_strategy=spy
        )
        result = service.solve(problem)
        assert result.status == "success"
        assert result.backend == FAKE_LOCAL_CQM_NAME
        return spy, result

    def test_nothing_is_computed_when_info_is_off(self, caplog, load_example):
        caplog.set_level(logging.WARNING, logger="annealbridge")

        spy, _ = self.solve_knapsack(load_example)

        assert spy.objective_scale_calls == 0
        assert spy.penalty_scale_calls == 0

    def test_hard_penalty_path_derives_the_penalty_scale_when_info_is_on(
        self, caplog, load_example
    ):
        """``knapsack.json`` on ``exact`` is the BQM hard-penalty path."""
        caplog.set_level(logging.INFO, logger="annealbridge")

        spy, _ = self.solve_knapsack(load_example)

        assert spy.objective_scale_calls == 1
        assert spy.penalty_scale_calls == 0

    def test_native_constraint_path_computes_the_penalty_scale_when_info_is_on(
        self, caplog, load_example
    ):
        """The CQM path has no penalty to derive the scale from."""
        caplog.set_level(logging.INFO, logger="annealbridge")

        spy, _ = self.solve_knapsack_on_cqm(load_example)

        assert spy.objective_scale_calls == 1
        assert spy.penalty_scale_calls == 1
