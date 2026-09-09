"""Validator and compilers judge soft constraints alike (review F-04 / F-12).

The contract (overview principle 6, 2026-09-09 review): *any problem the
validator accepts compiles on both the BQM and the CQM path, and
``validate_solution`` judges the same assignment the same way whichever
backend solved it*. Two findings broke it:

* **F-04**: a soft constraint whose coefficients cancel to zero (``x:+1,
  x:-1 == 5``) passed the validator, compiled on the BQM path (constant
  ``w * rhs**2`` in the offset) and was refused by the CQM compiler with
  ``COMPILATION_FAILED`` — the same problem was legal on one backend and
  invalid on another.
* **F-12**: a soft constraint no assignment can satisfy was clamped to zero
  slack with a server-side log line only; the agent saw nothing, while the
  harmless always-true constraint next to it got ``REDUNDANT_CONSTRAINT``.

Now the validator emits ``SOFT_ALWAYS_VIOLATED`` for every soft constraint
that no assignment within the bounds satisfies (the zero-coefficient case
included), and the CQM compiler writes the constant penalty into the
objective exactly as the BQM path does. The fuzz below pins the meaning of
the warning to brute-force enumeration with the solution validator itself.
"""

import itertools
import random

import pytest

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    SolverPreferences,
    Variable,
)
from annealbridge.orchestration import OptimizationService
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import (
    validate_problem,
    validate_problem_full,
    validate_solution,
)
from annealbridge.validation.estimates import (
    accumulate_terms,
    compute_penalty_scale,
    lhs_bounds,
    variable_bounds,
)
from tests.fakes.local_cqm_backend import FAKE_LOCAL_CQM_NAME, FakeLocalCQMBackend


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def binary(name: str) -> Variable:
    return Variable(name=name)


def integer(name: str, lower: int, upper: int) -> Variable:
    return Variable(name=name, type="integer", lower_bound=lower, upper_bound=upper)


def lin(variable: str, coefficient: float) -> LinearTerm:
    return LinearTerm(variable=variable, coefficient=coefficient)


def soft(identifier: str, operator: str, rhs: float, terms, weight: float) -> Constraint:
    return Constraint(
        id=identifier, type="soft", terms=terms, operator=operator, rhs=rhs, weight=weight
    )


def hard(identifier: str, operator: str, rhs: float, terms) -> Constraint:
    return Constraint(id=identifier, type="hard", terms=terms, operator=operator, rhs=rhs)


def make_problem(
    variables: list[Variable],
    constraints: list[Constraint],
    *,
    linear: list[LinearTerm] | None = None,
    direction: str = "minimize",
    name: str = "soft consistency",
) -> OptimizationProblem:
    has_integer = any(variable.type == "integer" for variable in variables)
    return OptimizationProblem(
        version="1.1" if has_integer else "1.0",
        name=name,
        variables=variables,
        objective=Objective(direction=direction, linear_terms=linear or []),
        constraints=constraints,
        solver=SolverPreferences(backend="exact"),
    )


def make_registry(fake: FakeLocalCQMBackend) -> SolverRegistry:
    defaults = SolverRegistry.default()
    backends = {name: defaults.get(name) for name in defaults.names()}
    backends[FAKE_LOCAL_CQM_NAME] = fake
    return SolverRegistry(backends)


def on_fake(problem: OptimizationProblem) -> OptimizationProblem:
    solver = SolverPreferences.model_construct(backend=FAKE_LOCAL_CQM_NAME)
    return problem.model_copy(update={"solver": solver})


def compile_both(problem: OptimizationProblem):
    """Compile on both paths; a ``CompilationError`` fails the test."""
    bqm = BQMCompiler().compile(problem, hard_penalty=2.0 * compute_penalty_scale(problem))
    cqm = CQMCompiler().compile(problem, None)
    return bqm, cqm


def assignments(problem: OptimizationProblem):
    bounds = variable_bounds(problem)
    names = [variable.name for variable in problem.variables]
    ranges = [range(bounds[name][0], bounds[name][1] + 1) for name in names]
    for values in itertools.product(*ranges):
        yield dict(zip(names, values))


def satisfiable_by_enumeration(problem: OptimizationProblem) -> dict[str, bool]:
    """``constraint id -> some assignment satisfies it`` per ``validate_solution``."""
    satisfiable = {constraint.id: False for constraint in problem.constraints}
    for sample in assignments(problem):
        for evaluation in validate_solution(problem, sample).evaluations:
            if evaluation.satisfied:
                satisfiable[evaluation.constraint_id] = True
        if all(satisfiable.values()):
            break
    return satisfiable


def warning_paths(result, code: str) -> set[str]:
    return {warning.path for warning in result.warnings if warning.code == code}


def range_misses_rhs(problem: OptimizationProblem, constraint: Constraint) -> bool:
    """The structural verdict both TRIVIALLY_INFEASIBLE and SOFT_ALWAYS_VIOLATED
    are defined on: the accumulated lhs *range* cannot meet the rhs.

    For an inequality with integer coefficients this is exact (the range end
    is attained). For an equality it is only necessary: ``2x == 1`` has a
    range ``[0, 2]`` containing 1 yet no assignment hits it, and neither the
    error nor the warning claims to detect such gaps.
    """
    lhs_min, lhs_max = lhs_bounds(accumulate_terms(constraint.terms), variable_bounds(problem))
    if constraint.operator == "<=":
        return lhs_min > constraint.rhs
    if constraint.operator == ">=":
        return lhs_max < constraint.rhs
    return constraint.rhs < lhs_min or constraint.rhs > lhs_max


# --------------------------------------------------------------------------
# F-04: zero coefficients after accumulation, rhs != 0, soft
# --------------------------------------------------------------------------


def f04_problem() -> OptimizationProblem:
    """The review's reproduction: ``soft c1: x:+1, x:-1 == 5, weight 2``."""
    return make_problem(
        [binary("x"), binary("y")],
        [soft("c1", "==", 5, [lin("x", 1), lin("x", -1)], 2.0)],
        linear=[lin("x", 1), lin("y", 2)],
        name="F-04",
    )


class TestF04ZeroCoefficientSoftConstraint:
    def test_validator_warns_instead_of_staying_silent(self):
        result = validate_problem_full(f04_problem())
        assert result.valid is True
        assert result.errors == []
        assert warning_paths(result, "SOFT_ALWAYS_VIOLATED") == {"constraints[0]"}
        (warning,) = [w for w in result.warnings if w.code == "SOFT_ALWAYS_VIOLATED"]
        assert "c1" in warning.message
        assert warning.recommended_action

    def test_both_compilers_accept_it(self):
        bqm, cqm = compile_both(f04_problem())
        # Both paths carry the constant penalty ``w * rhs**2 == 50``.
        assert bqm.model.offset == pytest.approx(50.0)
        assert cqm.model.objective.offset == pytest.approx(50.0)
        assert list(cqm.model.constraint_labels) == []

    def test_exact_and_cqm_backends_agree_through_the_service(self):
        fake = FakeLocalCQMBackend()
        service = OptimizationService(registry=make_registry(fake))

        exact = service.solve(f04_problem())
        via_cqm = service.solve(on_fake(f04_problem()))

        assert exact.status == "success"
        assert via_cqm.status == exact.status
        assert exact.backend == "exact"
        assert via_cqm.backend == FAKE_LOCAL_CQM_NAME
        assert exact.solutions[0].soft_violation_score == 50.0
        assert via_cqm.solutions[0].soft_violation_score == 50.0
        assert via_cqm.solutions[0].objective_value == exact.solutions[0].objective_value
        assert via_cqm.solutions[0].variables == exact.solutions[0].variables


# --------------------------------------------------------------------------
# F-12: structurally unsatisfiable soft inequality next to a redundant one
# --------------------------------------------------------------------------


def f12_problem() -> OptimizationProblem:
    """``soft impossible: -x - y <= -5 (w 10)`` and ``soft always: x <= 3``."""
    return make_problem(
        [binary("x"), binary("y")],
        [
            soft("impossible", "<=", -5, [lin("x", -1), lin("y", -1)], 10.0),
            soft("always", "<=", 3, [lin("x", 1)], 1.0),
        ],
        linear=[lin("x", 1), lin("y", 2)],
        name="F-12",
    )


class TestF12AlwaysViolatedSoftInequality:
    def test_validator_reports_both_constraints_symmetrically(self):
        fake = FakeLocalCQMBackend()
        service = OptimizationService(registry=make_registry(fake))
        for problem in (f12_problem(), on_fake(f12_problem())):
            result = service.validate(problem)
            assert result.valid is True
            assert warning_paths(result, "SOFT_ALWAYS_VIOLATED") == {"constraints[0]"}
            assert warning_paths(result, "REDUNDANT_CONSTRAINT") == {"constraints[1]"}
            (warning,) = [w for w in result.warnings if w.code == "SOFT_ALWAYS_VIOLATED"]
            assert "impossible" in warning.message
            assert "10.0" in warning.message

    def test_both_backends_solve_it_with_the_same_soft_score(self):
        fake = FakeLocalCQMBackend()
        service = OptimizationService(registry=make_registry(fake))

        exact = service.solve(f12_problem())
        via_cqm = service.solve(on_fake(f12_problem()))

        assert exact.status == via_cqm.status == "success"
        assert exact.solutions[0].soft_violation_score == 90.0
        assert via_cqm.solutions[0].soft_violation_score == 90.0
        assert via_cqm.solutions[0].variables == exact.solutions[0].variables == {
            "x": 1,
            "y": 1,
        }


# --------------------------------------------------------------------------
# Fuzz: the warning means exactly "no assignment satisfies it", and every
# validated problem compiles on both paths
# --------------------------------------------------------------------------


def random_problem(rng: random.Random, index: int) -> OptimizationProblem:
    variables: list[Variable] = []
    for position in range(rng.randint(1, 4)):
        name = f"v{position}"
        if rng.random() < 0.35:
            lower = rng.randint(-2, 1)
            variables.append(integer(name, lower, lower + rng.randint(1, 3)))
        else:
            variables.append(binary(name))
    names = [variable.name for variable in variables]

    constraints: list[Constraint] = []
    for k in range(rng.randint(1, 4)):
        terms: list[LinearTerm] = []
        for _ in range(rng.randint(1, 4)):
            terms.append(lin(rng.choice(names), rng.choice([-2, -1, 0, 0, 1, 2])))
        if rng.random() < 0.25:
            # A cancelling pair: the accumulated coefficient of ``name`` is 0.
            name = rng.choice(names)
            value = rng.choice([1, 2, 3])
            terms += [lin(name, value), lin(name, -value)]
        if rng.random() < 0.6:
            constraints.append(
                soft(
                    f"c{k}",
                    rng.choice(["==", "<=", ">="]),
                    rng.randint(-6, 6),
                    terms,
                    float(rng.randint(1, 5)),
                )
            )
        else:
            constraints.append(
                hard(f"c{k}", rng.choice(["==", "<=", ">="]), rng.randint(-6, 6), terms)
            )

    linear = [lin(name, rng.choice([-3, -1, 1, 2])) for name in names if rng.random() < 0.8]
    return make_problem(
        variables,
        constraints,
        linear=linear,
        direction=rng.choice(["minimize", "maximize"]),
        name=f"fuzz-{index}",
    )


FUZZ_COUNT = 500
FUZZ_PROBLEMS = [random_problem(random.Random(20260909 + i), i) for i in range(FUZZ_COUNT)]


def constraint_at(problem: OptimizationProblem, path: str) -> Constraint:
    index = int(path[len("constraints[") : -1])
    return problem.constraints[index]


class TestFuzzValidatorAndCompilersAgree:
    def test_corpus_size(self):
        assert len(FUZZ_PROBLEMS) == FUZZ_COUNT

    @pytest.mark.parametrize("index", range(FUZZ_COUNT))
    def test_one_problem(self, index):
        problem = FUZZ_PROBLEMS[index]
        errors = validate_problem(problem)
        satisfiable = satisfiable_by_enumeration(problem)

        # Every error is a TRIVIALLY_INFEASIBLE on a hard constraint whose
        # range misses the rhs, which enumeration confirms unsatisfiable;
        # every hard constraint whose range misses the rhs is rejected.
        rejected = set()
        for error in errors:
            assert error.code == "TRIVIALLY_INFEASIBLE", error
            constraint = constraint_at(problem, error.path)
            assert constraint.type == "hard"
            assert range_misses_rhs(problem, constraint)
            assert satisfiable[constraint.id] is False
            rejected.add(constraint.id)
        for constraint in problem.constraints:
            if constraint.type == "hard" and constraint.id not in rejected:
                assert not range_misses_rhs(problem, constraint), constraint.id
                if constraint.operator != "==":
                    assert satisfiable[constraint.id] is True, constraint.id
        if errors:
            return

        # A validated problem compiles on both paths, no exceptions.
        compile_both(problem)

        # SOFT_ALWAYS_VIOLATED: sound (every warned constraint really is
        # unsatisfiable) and complete on the structural verdict (every soft
        # constraint whose range misses the rhs is warned; for inequalities
        # that is every unsatisfiable one).
        result = validate_problem_full(problem)
        assert result.valid is True
        warned = {
            constraint_at(problem, path).id
            for path in warning_paths(result, "SOFT_ALWAYS_VIOLATED")
        }
        for constraint in problem.constraints:
            if constraint.type != "soft":
                continue
            if constraint.id in warned:
                assert satisfiable[constraint.id] is False, constraint.id
                assert range_misses_rhs(problem, constraint), constraint.id
            else:
                assert not range_misses_rhs(problem, constraint), constraint.id
                if constraint.operator != "==":
                    assert satisfiable[constraint.id] is True, constraint.id

    def test_corpus_covers_the_interesting_cases(self):
        """The fuzz proves nothing unless the cases it is about occur."""
        always_violated = {"==": 0, "<=": 0, ">=": 0}
        zero_coefficient_soft = 0
        zero_coefficient_hard_rejected = 0
        validated = 0
        for problem in FUZZ_PROBLEMS:
            errors = validate_problem(problem)
            for error in errors:
                constraint = constraint_at(problem, error.path)
                totals: dict[str, float] = {}
                for term in constraint.terms:
                    totals[term.variable] = totals.get(term.variable, 0.0) + term.coefficient
                if not any(totals.values()):
                    zero_coefficient_hard_rejected += 1
            if errors:
                continue
            validated += 1
            result = validate_problem_full(problem)
            for path in warning_paths(result, "SOFT_ALWAYS_VIOLATED"):
                constraint = constraint_at(problem, path)
                always_violated[constraint.operator] += 1
                totals = {}
                for term in constraint.terms:
                    totals[term.variable] = totals.get(term.variable, 0.0) + term.coefficient
                if not any(totals.values()):
                    zero_coefficient_soft += 1
        assert validated >= 200, validated
        assert all(count >= 1 for count in always_violated.values()), always_violated
        assert zero_coefficient_soft >= 1
        assert zero_coefficient_hard_rejected >= 1
