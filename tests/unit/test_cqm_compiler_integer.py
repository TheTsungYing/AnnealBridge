"""CQM 路徑整數變數的行為測試（3b spec §15、§25 步驟 4、§26.1）。

證據分四層：

* **§15.3 一致性證明**：對含整數 soft constraint 的問題（手寫五類 +
  固定種子隨機 corpus），``ExactCQMSolver`` 全枚舉後，對每個業務
  assignment 取所有 slack 值上的**最小** energy，必須等於
  ``sign * objective + Σ w * max(0, violation)²``（獨立手算），也必須等於
  ``sign * objective + validate_solution(...).soft_violation_score``。
* **模型形狀**：全 binary soft 仍走 3a 原生寫法；hard constraint 對整數
  變數直通（QM lhs、無 slack）；INTEGER 的 ``x·x`` 保留二次項。
* **端到端**：``examples/integer_knapsack.json`` 走 service +
  ``FakeLocalCQMBackend``，最佳解與 BQM / exact 路徑相同；負下界問題 decode
  值在 bounds 內、``is_feasible`` 只看 hard。
* **估算層**：``estimate_cqm_variables == compiled.num_variables``。

所有隨機部分都用固定種子；不 skip、不 xfail。
"""

import itertools
import json
import random

import dimod
import numpy as np
import pytest

from annealbridge.compiler import CQMCompiler, expand_square_qm

# The rhs half of the non-finite guard has no route through a validated
# problem, so it is pinned on the helper directly (as
# ``test_candidate_arrays.py`` does with the optimizer's private helpers).
from annealbridge.compiler.cqm import _check_finite as check_cqm_finite
from annealbridge.exceptions import CompilationError, NonFiniteModelError
from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    QuadraticTerm,
    SolverPreferences,
    Variable,
)
from annealbridge.orchestration import OptimizationService, evaluate_objective
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import (
    validate_problem,
    validate_problem_full,
    validate_solution,
)
from annealbridge.validation.estimates import (
    accumulate_terms,
    analyze_inequality,
    estimate_cqm_variables,
    variable_bounds,
)
from tests.conftest import EXAMPLES_DIR
from tests.fakes.local_cqm_backend import FAKE_LOCAL_CQM_NAME, FakeLocalCQMBackend
from tests.model_builders import lin, quad

# --------------------------------------------------------------------------
# Helpers（與 test_bqm_compiler_integer.py 同形式；lin / quad 共用 tests/model_builders.py）
# --------------------------------------------------------------------------


def integer(name: str, lower: int, upper: int) -> Variable:
    return Variable(name=name, type="integer", lower_bound=lower, upper_bound=upper)


def binary(name: str) -> Variable:
    return Variable(name=name)


def constraint(
    identifier: str,
    operator: str,
    rhs: float,
    terms: list[LinearTerm],
    *,
    type: str = "hard",
    weight: float | None = None,
) -> Constraint:
    return Constraint(
        id=identifier,
        type=type,
        terms=terms,
        operator=operator,
        rhs=rhs,
        weight=weight,
    )


def make_problem(
    variables: list[Variable],
    *,
    direction: str = "minimize",
    linear: list[LinearTerm] | None = None,
    quadratic: list[QuadraticTerm] | None = None,
    constant: float = 0.0,
    constraints: list[Constraint] | None = None,
    version: str = "1.1",
    name: str = "cqm integer compiler test",
) -> OptimizationProblem:
    return OptimizationProblem(
        version=version,
        name=name,
        variables=variables,
        objective=Objective(
            direction=direction,
            linear_terms=linear or [],
            quadratic_terms=quadratic or [],
            constant=constant,
        ),
        constraints=constraints or [],
    )


def compile_problem(problem: OptimizationProblem):
    return CQMCompiler().compile(problem, None)


def business_assignments(problem: OptimizationProblem) -> list[dict[str, int]]:
    """Every assignment of the declared variables over their bounds."""
    bounds = variable_bounds(problem)
    names = list(bounds)
    ranges = [range(bounds[name][0], bounds[name][1] + 1) for name in names]
    return [dict(zip(names, values)) for values in itertools.product(*ranges)]


def soft_penalty_by_hand(problem: OptimizationProblem, assignment: dict[str, int]) -> float:
    """``Σ w * max(0, violation)²`` straight from the constraint definition.

    Independent of both the compiler and ``validate_solution``: it only
    accumulates the constraint's own terms and applies the operator.
    """
    total = 0.0
    for c in problem.constraints:
        if c.type != "soft":
            continue
        lhs = sum(
            value * assignment[name] for name, value in accumulate_terms(c.terms).items()
        )
        if c.operator == "==":
            violation = abs(lhs - c.rhs)
        elif c.operator == "<=":
            violation = max(0.0, lhs - c.rhs)
        else:
            violation = max(0.0, c.rhs - lhs)
        assert c.weight is not None
        total += c.weight * violation * violation
    return total


def min_energy_per_assignment(problem: OptimizationProblem, compiled) -> dict[tuple, float]:
    """Exhaustive ``ExactCQMSolver`` energies, minimised over the slack columns.

    Keyed by the business values in ``problem.variables`` order. Uses the
    sampleset *record* (sampler order), never ``samples()`` which sorts by
    energy and would break the row/energy pairing.
    """
    sampleset = dimod.ExactCQMSolver().sample_cqm(compiled.model)
    names = [str(variable) for variable in sampleset.variables]
    business = [variable.name for variable in problem.variables]
    columns = [names.index(name) for name in business]
    best: dict[tuple, float] = {}
    for row, energy in zip(sampleset.record.sample, sampleset.record.energy):
        key = tuple(int(row[column]) for column in columns)
        best[key] = min(best.get(key, float("inf")), float(energy))
    return best


def model_size(problem: OptimizationProblem, compiled) -> int:
    """Number of rows ``ExactCQMSolver`` will enumerate for ``compiled``."""
    size = 1
    for variable in compiled.model.variables:
        lower = compiled.model.lower_bound(variable)
        upper = compiled.model.upper_bound(variable)
        size *= int(upper - lower) + 1
    return size


# --------------------------------------------------------------------------
# Problem corpus for the §15.3 proof
# --------------------------------------------------------------------------


def hand_written_soft_problems() -> list[tuple[str, OptimizationProblem]]:
    """One problem per §15.3 case, each with an integer soft constraint."""
    return [
        (
            "equality in objective form",
            make_problem(
                [integer("x", -3, 3), binary("b")],
                direction="maximize",
                linear=[lin("x", 4), lin("b", 1)],
                quadratic=[quad("x", "x", 1)],
                constraints=[
                    constraint(
                        "eq", "==", 1, [lin("x", 1), lin("b", 2)], type="soft", weight=0.5
                    )
                ],
            ),
        ),
        (
            "less-equal with integer slack",
            make_problem(
                [integer("x", 0, 5), integer("y", -2, 2)],
                linear=[lin("x", -1), lin("y", -3)],
                constraints=[
                    constraint(
                        "cap", "<=", 3, [lin("x", 1), lin("y", 2)], type="soft", weight=2.5
                    ),
                    constraint("hard", ">=", 0, [lin("x", 1), lin("y", 1)]),
                ],
            ),
        ),
        (
            "greater-equal with integer slack",
            make_problem(
                [integer("x", -3, 3), binary("b")],
                direction="maximize",
                linear=[lin("x", -2), lin("b", 3)],
                constraints=[
                    constraint(
                        "cover", ">=", 1, [lin("x", 1), lin("b", 1)], type="soft", weight=1.5
                    )
                ],
            ),
        ),
        (
            "trivially infeasible soft (S < 0, clamped)",
            make_problem(
                [integer("x", -3, 3), binary("b")],
                linear=[lin("x", 1), lin("b", -1)],
                constraints=[
                    constraint("clamp", ">=", 5, [lin("x", 1)], type="soft", weight=1.0),
                    constraint(
                        "clamp_le", "<=", -4, [lin("x", 1), lin("b", 1)], type="soft", weight=2.0
                    ),
                ],
            ),
        ),
        (
            "redundant soft inequality adds nothing",
            make_problem(
                [integer("x", 0, 4), binary("b")],
                linear=[lin("x", 1), lin("b", 2)],
                quadratic=[quad("x", "b", -1)],
                constraints=[
                    constraint("loose", "<=", 10, [lin("x", 1), lin("b", 1)], type="soft", weight=1.0),
                    constraint("also_loose", ">=", -1, [lin("x", 1)], type="soft", weight=3.0),
                    constraint("real", "<=", 2, [lin("x", 1)], type="soft", weight=1.0),
                ],
            ),
        ),
        (
            "everything at once, with a binary-only soft beside",
            make_problem(
                [integer("x", -2, 2), integer("y", 0, 3), binary("b")],
                direction="maximize",
                linear=[lin("x", 1), lin("y", 1), lin("b", 1)],
                quadratic=[quad("x", "x", 1), quad("x", "y", -1), quad("y", "b", 2)],
                constraints=[
                    constraint("eq", "==", 3, [lin("y", 1), lin("b", 1)], type="soft", weight=1.0),
                    constraint("cap", "<=", 1, [lin("x", 2), lin("y", 1)], type="soft", weight=2.0),
                    constraint("cover", ">=", 0, [lin("x", 1), lin("y", 1)], type="soft", weight=0.5),
                    constraint("bin", ">=", 1, [lin("b", 1)], type="soft", weight=3.0),
                    constraint("hard", "<=", 4, [lin("x", 1), lin("y", 1), lin("b", 1)]),
                ],
            ),
        ),
    ]


def random_variables(rng: random.Random) -> list[Variable]:
    variables: list[Variable] = []
    for index in range(rng.randint(1, 3)):
        name = f"v{index}"
        if rng.random() < 0.3:
            variables.append(binary(name))
        else:
            lower = rng.randint(-4, 2)
            variables.append(integer(name, lower, lower + rng.randint(1, 5)))
    if not any(variable.type == "integer" for variable in variables):
        variables[0] = integer(variables[0].name, -1, 2)
    return variables


def random_terms(rng: random.Random, names: list[str]) -> list[LinearTerm]:
    # Repeated variables on purpose: accumulation is part of the contract.
    return [
        lin(rng.choice(names), rng.choice([-3, -2, -1, 1, 2, 3]))
        for _ in range(rng.randint(1, len(names) + 1))
    ]


def random_problem(rng: random.Random) -> OptimizationProblem:
    variables = random_variables(rng)
    names = [variable.name for variable in variables]
    integer_names = [v.name for v in variables if v.type == "integer"]

    quadratic: list[QuadraticTerm] = []
    for _ in range(rng.randint(0, 2)):
        first = rng.choice(names)
        second = (
            first
            if first in integer_names and rng.random() < 0.4
            else rng.choice(names)
        )
        if first == second and first not in integer_names:
            continue  # a binary self-product is rejected by the validator
        quadratic.append(quad(first, second, rng.choice([-2, -1, 1, 2])))

    constraints: list[Constraint] = []
    # At least one soft constraint over an integer variable (the point of
    # this corpus); the remaining ones are free.
    first_terms = random_terms(rng, names) + [lin(rng.choice(integer_names), 1)]
    constraints.append(
        constraint(
            "c0",
            rng.choice(["==", "<=", ">="]),
            rng.randint(-6, 6),
            first_terms,
            type="soft",
            weight=float(rng.randint(1, 5)),
        )
    )
    for index in range(1, rng.randint(1, 4)):
        soft = rng.random() < 0.6
        constraints.append(
            constraint(
                f"c{index}",
                rng.choice(["==", "<=", ">="]),
                rng.randint(-6, 6),
                random_terms(rng, names),
                type="soft" if soft else "hard",
                weight=float(rng.randint(1, 5)) if soft else None,
            )
        )

    return make_problem(
        variables,
        direction=rng.choice(["minimize", "maximize"]),
        linear=random_terms(rng, names) if rng.random() < 0.9 else [],
        quadratic=quadratic,
        constant=rng.randint(-3, 3),
        constraints=constraints,
        name="random",
    )


def integer_soft_traces(compiled) -> list:
    return [trace for trace in compiled.constraint_trace if not trace.native]


def valid_random_problems(count: int, *, seed: int = 20260908) -> list[OptimizationProblem]:
    """``count`` random problems the validator accepts, each with an
    objective-form soft constraint and small enough to enumerate."""
    rng = random.Random(seed)
    problems: list[OptimizationProblem] = []
    attempts = 0
    while len(problems) < count and attempts < count * 60:
        attempts += 1
        problem = random_problem(rng)
        if validate_problem(problem):
            continue
        # Review F-04: a validated problem always compiles; a soft constraint
        # whose terms cancel to an impossible constant is now a constant
        # penalty, so a CompilationError here is a test failure.
        compiled = compile_problem(problem)
        if not integer_soft_traces(compiled):
            continue
        if model_size(problem, compiled) > 3000:
            continue
        problems.append(problem)
    assert len(problems) == count, f"only {len(problems)} valid problems in {attempts} tries"
    return problems


RANDOM_PROBLEMS = valid_random_problems(40)
HAND_WRITTEN = hand_written_soft_problems()
ALL_SOFT_PROBLEMS = [problem for _, problem in HAND_WRITTEN] + RANDOM_PROBLEMS


def assert_energy_identity(problem: OptimizationProblem) -> None:
    """§15.3: ``min_s energy == sign * objective + Σ w * max(0, v)²`` for every
    business assignment, and that equals the validator's soft score."""
    compiled = compile_problem(problem)
    best = min_energy_per_assignment(problem, compiled)
    sign = -1.0 if problem.objective.direction == "maximize" else 1.0
    assignments = business_assignments(problem)
    assert len(best) == len(assignments)
    for assignment in assignments:
        key = tuple(assignment[variable.name] for variable in problem.variables)
        objective = sign * evaluate_objective(problem.objective, assignment)
        expected = objective + soft_penalty_by_hand(problem, assignment)
        verdict = validate_solution(problem, assignment)
        assert best[key] == pytest.approx(expected), (assignment, best[key], expected)
        assert best[key] == pytest.approx(objective + verdict.soft_violation_score)


# --------------------------------------------------------------------------
# (a) §15.3 consistency proof
# --------------------------------------------------------------------------


class TestSoftEnergyIdentity:
    @pytest.mark.parametrize(
        "problem", [problem for _, problem in HAND_WRITTEN], ids=[name for name, _ in HAND_WRITTEN]
    )
    def test_hand_written(self, problem):
        assert_energy_identity(problem)

    @pytest.mark.parametrize("index", range(len(RANDOM_PROBLEMS)))
    def test_random(self, index):
        assert_energy_identity(RANDOM_PROBLEMS[index])

    def test_corpus_covers_every_objective_form_case(self):
        """The proof is only as strong as its corpus: every §15.3 branch
        must be exercised, on the traces the compiler actually produced."""
        seen = {"==": 0, "<=": 0, ">=": 0, "clamp": 0, "redundant": 0}
        for problem in ALL_SOFT_PROBLEMS:
            compiled = compile_problem(problem)
            bounds = variable_bounds(problem)
            by_id = {c.id: c for c in problem.constraints}
            for trace in integer_soft_traces(compiled):
                if trace.redundant:
                    seen["redundant"] += 1
                    continue
                if trace.operator == "==":
                    seen["=="] += 1
                    continue
                analysis = analyze_inequality(by_id[trace.constraint_id], bounds)
                assert analysis.slack_range is not None
                if analysis.slack_range < 0:
                    assert trace.slack_range == 0 and trace.generated_variables == []
                    seen["clamp"] += 1
                else:
                    assert trace.slack_range == analysis.slack_range
                    seen[trace.operator] += 1
        assert all(count >= 1 for count in seen.values()), seen
        assert len(RANDOM_PROBLEMS) >= 30

    def test_objective_form_traces_are_marked(self):
        """``native=False``, ``penalty=w``, and a slack only when ``S > 0``."""
        for problem in ALL_SOFT_PROBLEMS:
            compiled = compile_problem(problem)
            by_id = {c.id: c for c in problem.constraints}
            for trace in integer_soft_traces(compiled):
                source = by_id[trace.constraint_id]
                assert source.type == "soft"
                assert trace.penalty == source.weight
                assert trace.compiler == "CQMCompiler"
                if trace.slack_range and trace.slack_range > 0:
                    assert trace.generated_variables == [f"__slack_{source.id}"]
                    slack = trace.generated_variables[0]
                    assert slack in compiled.internal_variables
                    assert compiled.model.vartype(slack) is dimod.INTEGER
                    assert compiled.model.lower_bound(slack) == 0
                    assert compiled.model.upper_bound(slack) == trace.slack_range
                else:
                    assert trace.generated_variables == []
                # Never a native CQM constraint for the objective form.
                assert trace.constraint_id not in compiled.model.constraint_labels


# --------------------------------------------------------------------------
# (h) expand_square_qm against direct arithmetic
# --------------------------------------------------------------------------


def random_square_case(rng: random.Random):
    """A QM with mixed INTEGER / BINARY variables plus random square data."""
    qm = dimod.QuadraticModel()
    bounds: dict[str, tuple[int, int]] = {}
    for index in range(rng.randint(1, 4)):
        name = f"v{index}"
        if rng.random() < 0.4:
            qm.add_variable("BINARY", name)
            bounds[name] = (0, 1)
        else:
            lower = rng.randint(-5, 3)
            upper = lower + rng.randint(1, 6)
            qm.add_variable("INTEGER", name, lower_bound=lower, upper_bound=upper)
            bounds[name] = (lower, upper)
    # Some variables are left out of the square on purpose.
    names = [name for name in bounds if rng.random() < 0.8] or [next(iter(bounds))]
    coefficients = {name: float(rng.choice([-3, -2, -1, 1, 2, 3])) for name in names}
    constant = float(rng.randint(-5, 5))
    weight = float(rng.choice([0.5, 1.0, 2.0, 3.5]))
    return qm, bounds, coefficients, constant, weight


class TestExpandSquareQM:
    @pytest.mark.parametrize("seed", range(60))
    def test_energy_equals_direct_square(self, seed):
        rng = random.Random(1000 + seed)
        qm, bounds, coefficients, constant, weight = random_square_case(rng)
        # A pre-existing term: the expansion must accumulate, not overwrite.
        first = next(iter(bounds))
        qm.add_linear(first, 0.75)
        qm.offset += 1.25
        expand_square_qm(qm, coefficients, constant, weight)
        for _ in range(10):
            assignment = {
                name: rng.randint(lower, upper) for name, (lower, upper) in bounds.items()
            }
            affine = sum(a * assignment[name] for name, a in coefficients.items()) + constant
            expected = weight * affine * affine + 0.75 * assignment[first] + 1.25
            assert qm.energy(assignment) == pytest.approx(expected)

    @pytest.mark.parametrize("seed", range(20))
    def test_squares_stay_quadratic_for_integer_only(self, seed):
        rng = random.Random(2000 + seed)
        qm, bounds, coefficients, constant, weight = random_square_case(rng)
        expand_square_qm(qm, coefficients, constant, weight)
        quadratic = dict(qm.quadratic)
        for name, a in coefficients.items():
            if qm.vartype(name) is dimod.BINARY:
                assert (name, name) not in quadratic
                # Folded square: weight * a² + 2 * weight * constant * a.
                assert qm.get_linear(name) == pytest.approx(weight * (a * a + 2 * constant * a))
            else:
                assert qm.get_quadratic(name, name) == pytest.approx(weight * a * a)
                assert qm.get_linear(name) == pytest.approx(2 * weight * constant * a)
        assert qm.offset == pytest.approx(weight * constant * constant)

    def test_accumulates_twice(self):
        qm = dimod.QuadraticModel()
        qm.add_variable("INTEGER", "x", lower_bound=0, upper_bound=4)
        qm.add_variable("BINARY", "b")
        expand_square_qm(qm, {"x": 1.0, "b": 2.0}, -1.0, 1.0)
        expand_square_qm(qm, {"x": 1.0, "b": 2.0}, -1.0, 1.0)
        for x in range(5):
            for b in (0, 1):
                assert qm.energy({"x": x, "b": b}) == pytest.approx(2 * (x + 2 * b - 1) ** 2)

    def test_unknown_variable_is_rejected(self):
        qm = dimod.QuadraticModel()
        qm.add_variable("INTEGER", "x", lower_bound=0, upper_bound=4)
        with pytest.raises(ValueError):
            expand_square_qm(qm, {"x": 1.0, "ghost": 1.0}, 0.0, 1.0)


# --------------------------------------------------------------------------
# 非有限模型攔截（2026-09-11 review F06）
# --------------------------------------------------------------------------

# 問題本身只帶有限數字，validator 因此接受；但編譯時的
# ``weight * coefficient²`` 展開或係數累加會溢位成 ``inf``，此時 compiler
# 是唯一防線——BQM 路徑早就有 ``_check_finite``，CQM 路徑必須一致。


def overflowing_soft_problem() -> OptimizationProblem:
    """Soft equality whose objective-form penalty expands to ``inf``.

    ``weight * coefficient**2 == 1.0 * 1e200**2`` leaves the float range
    inside ``expand_square_qm``; the resulting ``(x, x)`` bias is ``inf``.
    """
    return make_problem(
        [integer("x", 0, 1)],
        constraints=[
            constraint("huge", "==", 0.0, [lin("x", 1e200)], type="soft", weight=1.0)
        ],
        name="cqm soft weight overflow",
    )


def overflowing_hard_problem() -> OptimizationProblem:
    """Hard native constraint whose *accumulated* lhs coefficient is ``inf``.

    Two finite terms on the same variable: ``nonzero_coefficients`` sums
    them to ``inf``, which lands in the constraint's QM lhs rather than in
    the objective, so the guard must look at the constraints too.
    """
    return make_problem(
        [integer("x", 0, 1)],
        constraints=[constraint("huge", "==", 0.0, [lin("x", 1e308), lin("x", 1e308)])],
        name="cqm hard lhs overflow",
    )


class TestNonFiniteModelIsRefused:
    def test_non_finite_error_is_a_compilation_error(self):
        # The service reports it as invalid_problem / COMPILATION_FAILED on
        # this path (no hard penalty), which needs the subclass relation.
        assert issubclass(NonFiniteModelError, CompilationError)

    @pytest.mark.parametrize(
        "problem",
        [overflowing_soft_problem(), overflowing_hard_problem()],
        ids=["soft weight expansion", "hard constraint lhs"],
    )
    def test_valid_problem_that_compiles_to_inf_is_refused(self, problem):
        # The problem passes full validation: every declared number is
        # finite, so nothing before the compiler can catch this.
        assert validate_problem_full(problem, model_type="cqm").valid is True

        with pytest.raises(NonFiniteModelError) as info:
            compile_problem(problem)

        message = str(info.value)
        assert problem.name in message
        assert "non-finite" in message
        # §10.4: there is no hard penalty on the CQM path, so the message
        # must not blame one.
        assert "penalty" not in message

    def test_large_but_finite_coefficients_still_compile(self):
        # The guard refuses ``inf``, not merely big numbers: 1e100² is
        # 1e200, still finite, and must compile as before.
        problem = make_problem(
            [integer("x", 0, 1)],
            constraints=[
                constraint(
                    "large", "==", 0.0, [lin("x", 1e100)], type="soft", weight=1.0
                )
            ],
            name="cqm large but finite",
        )
        compiled = compile_problem(problem)

        assert compiled.model.objective.get_quadratic("x", "x") == pytest.approx(1e200)

    def test_the_rhs_of_a_native_constraint_is_covered(self):
        # A non-finite rhs cannot come from a validated problem
        # (NON_FINITE_COEFFICIENT rejects it), so the guard's rhs check is
        # defence in depth; pinned directly on the helper instead.
        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_variable("BINARY", "b")
        lhs = dimod.QuadraticModel()
        lhs.add_variable("BINARY", "b")
        lhs.add_linear("b", 1.0)
        cqm.add_constraint_from_model(
            lhs, sense="<=", rhs=float("inf"), label="c", weight=None
        )

        with pytest.raises(NonFiniteModelError):
            check_cqm_finite(cqm, make_problem([binary("b")], version="1.0"))


# --------------------------------------------------------------------------
# (b)–(g): model shape, end-to-end and estimate
# --------------------------------------------------------------------------

# Service-level helpers, copied from ``test_service_cqm_flow.py`` on purpose:
# test modules never import each other.


def make_registry(fake: FakeLocalCQMBackend) -> SolverRegistry:
    """Every default backend plus the fake, so the exact path is still there."""
    defaults = SolverRegistry.default()
    backends = {name: defaults.get(name) for name in defaults.names()}
    assert FAKE_LOCAL_CQM_NAME not in backends
    backends[FAKE_LOCAL_CQM_NAME] = fake
    return SolverRegistry(backends)


def route_to_fake(problem: OptimizationProblem, **preferences) -> OptimizationProblem:
    """Point ``problem`` at ``FakeLocalCQMBackend``.

    ``SolverPreferences.backend`` is a Literal of the shipped names, so a
    test-only backend is set with ``model_construct``.
    """
    solver = SolverPreferences.model_construct(backend=FAKE_LOCAL_CQM_NAME, **preferences)
    return problem.model_copy(update={"solver": solver})


def solve_on_fake(problem: OptimizationProblem, **preferences):
    """Solve ``problem`` through the service on the CQM fake backend."""
    service = OptimizationService(registry=make_registry(FakeLocalCQMBackend()))
    return service.solve(route_to_fake(problem, **preferences))


def load_example(filename: str) -> OptimizationProblem:
    payload = json.loads((EXAMPLES_DIR / filename).read_text(encoding="utf-8"))
    return OptimizationProblem.model_validate(payload)


def brute_force_ranked(problem: OptimizationProblem) -> list[tuple[float, dict[str, int]]]:
    """``(ranking, assignment)`` of every hard-feasible assignment, best first.

    Ranking follows the optimizer (§25.1): ``objective - soft_violation_score``
    for maximize, ``objective + soft_violation_score`` for minimize.
    """
    maximize = problem.objective.direction == "maximize"
    rows: list[tuple[float, dict[str, int]]] = []
    for assignment in business_assignments(problem):
        verdict = validate_solution(problem, assignment)
        if not verdict.feasible:
            continue
        objective = evaluate_objective(problem.objective, assignment)
        if maximize:
            ranking = objective - verdict.soft_violation_score
        else:
            ranking = objective + verdict.soft_violation_score
        rows.append((ranking, assignment))
    rows.sort(key=lambda row: row[0], reverse=maximize)
    return rows


def trace_by_id(compiled) -> dict:
    return {trace.constraint_id: trace for trace in compiled.constraint_trace}


# --------------------------------------------------------------------------
# (b) an all-binary soft constraint keeps the 3a native writing
# --------------------------------------------------------------------------


def binary_only_problem(version: str) -> OptimizationProblem:
    """Three binary variables, one hard and one all-binary soft constraint."""
    return make_problem(
        [binary("x1"), binary("x2"), binary("x3")],
        linear=[lin("x1", -3), lin("x2", -2), lin("x3", -1)],
        constraints=[
            constraint("cap", "<=", 2, [lin("x1", 1), lin("x2", 1), lin("x3", 1)]),
            constraint(
                "pref", ">=", 2, [lin("x1", 1), lin("x2", 1)], type="soft", weight=4.0
            ),
        ],
        version=version,
        name=f"binary only soft (IR {version})",
    )


def mixed_soft_problem() -> OptimizationProblem:
    """One all-binary soft constraint beside one that mentions an integer."""
    return make_problem(
        [integer("x", 0, 3), binary("b1"), binary("b2")],
        direction="maximize",
        linear=[lin("x", 2), lin("b1", 3), lin("b2", 1)],
        constraints=[
            constraint("h", "<=", 4, [lin("x", 1), lin("b2", 1)]),
            constraint(
                "bin_only", "<=", 1, [lin("b1", 1), lin("b2", 1)], type="soft", weight=2.0
            ),
            constraint(
                "int_soft", "<=", 2, [lin("x", 1), lin("b1", 1)], type="soft", weight=1.5
            ),
        ],
        name="mixed native and objective-form softs",
    )


BINARY_ONLY_PROBLEMS = [binary_only_problem("1.0"), binary_only_problem("1.1")]


class TestBinaryOnlySoftStaysNative:
    """An all-binary soft constraint must still compile to the native dimod
    form ``weight=w, penalty="quadratic"`` (3a §15), whatever the IR version
    and even when an integer soft constraint sits beside it."""

    @pytest.mark.parametrize("problem", BINARY_ONLY_PROBLEMS, ids=["ir-1.0", "ir-1.1"])
    def test_soft_constraint_is_a_native_weighted_constraint(self, problem):
        compiled = compile_problem(problem)
        cqm = compiled.model

        trace = trace_by_id(compiled)["pref"]
        assert trace.native is True
        assert trace.penalty == 4.0
        assert trace.generated_variables == []
        assert trace.slack_range is None
        assert trace.redundant is False

        assert "pref" in cqm.constraints
        assert cqm._soft["pref"].weight == 4.0
        assert str(cqm._soft["pref"].penalty) == "quadratic"

    @pytest.mark.parametrize("problem", BINARY_ONLY_PROBLEMS, ids=["ir-1.0", "ir-1.1"])
    def test_no_internal_variables_are_generated(self, problem):
        compiled = compile_problem(problem)

        assert compiled.internal_variables == set()
        assert compiled.num_variables == len(problem.variables)
        assert compiled.num_variables == 3
        assert not any(str(name).startswith("__") for name in compiled.model.variables)

    @pytest.mark.parametrize("problem", BINARY_ONLY_PROBLEMS, ids=["ir-1.0", "ir-1.1"])
    def test_hard_constraint_beside_it_carries_no_penalty(self, problem):
        trace = trace_by_id(compile_problem(problem))["cap"]

        assert trace.native is True
        assert trace.penalty is None
        assert trace.generated_variables == []
        assert trace.slack_range is None

    def test_mixed_problem_splits_the_two_writings(self):
        problem = mixed_soft_problem()
        compiled = compile_problem(problem)
        cqm = compiled.model
        traces = trace_by_id(compiled)

        # The all-binary soft stays native, exactly as in the pure-binary case.
        assert traces["bin_only"].native is True
        assert traces["bin_only"].penalty == 2.0
        assert traces["bin_only"].generated_variables == []
        assert traces["bin_only"].slack_range is None
        assert cqm._soft["bin_only"].weight == 2.0
        assert str(cqm._soft["bin_only"].penalty) == "quadratic"

        # The one mentioning an integer goes into the objective instead.
        assert traces["int_soft"].native is False
        assert traces["int_soft"].penalty == 1.5
        assert traces["int_soft"].generated_variables == ["__slack_int_soft"]
        assert traces["int_soft"].slack_range == 2

        # Only the hard constraint and the native soft are CQM constraints.
        assert set(cqm.constraint_labels) == {"h", "bin_only"}
        assert "int_soft" not in cqm._soft
        assert compiled.internal_variables == {"__slack_int_soft"}
        assert compiled.num_variables == len(problem.variables) + 1


# --------------------------------------------------------------------------
# (c) hard constraints pass straight through, integer variables included
# --------------------------------------------------------------------------


def hard_passthrough_problem() -> OptimizationProblem:
    """One hard constraint per operator over integer variables.

    Every constraint repeats a variable (accumulation) and carries one whose
    coefficients cancel to zero (elimination).
    """
    return make_problem(
        [integer("x", -3, 4), integer("y", -2, 2), integer("z", 0, 3), binary("b")],
        linear=[lin("x", 1), lin("y", -2), lin("z", 1), lin("b", -1)],
        quadratic=[quad("x", "x", 1)],
        constraints=[
            # 1x + 2x == 3x, -1y, and z cancels out.
            constraint(
                "eq",
                "==",
                3,
                [lin("x", 1), lin("x", 2), lin("y", -1), lin("z", 1), lin("z", -1)],
            ),
            # b cancels out.
            constraint(
                "le", "<=", 5, [lin("x", 1), lin("y", 2), lin("b", 1), lin("b", -1)]
            ),
            # 2z - 1z == 1z.
            constraint("ge", ">=", -1, [lin("z", 2), lin("y", 1), lin("z", -1)]),
        ],
        name="hard constraints over integers",
    )


HARD_PASSTHROUGH_EXPECTED = {
    "eq": ({"x": 3.0, "y": -1.0}, dimod.sym.Sense.Eq, 3.0),
    "le": ({"x": 1.0, "y": 2.0}, dimod.sym.Sense.Le, 5.0),
    "ge": ({"z": 1.0, "y": 1.0}, dimod.sym.Sense.Ge, -1.0),
}

DECLARED_BOUNDS = {"x": (-3, 4), "y": (-2, 2), "z": (0, 3), "b": (0, 1)}


class TestHardConstraintsPassThrough:
    """Hard constraints become native CQM constraints with a QuadraticModel
    lhs (3b §15.2): accumulated coefficients, cancelled variables dropped,
    integer vartypes and declared bounds preserved, no slack anywhere."""

    def test_problem_is_valid(self):
        assert validate_problem(hard_passthrough_problem()) == []

    @pytest.mark.parametrize("label", sorted(HARD_PASSTHROUGH_EXPECTED))
    def test_lhs_sense_and_rhs(self, label):
        compiled = compile_problem(hard_passthrough_problem())
        expected_linear, expected_sense, expected_rhs = HARD_PASSTHROUGH_EXPECTED[label]

        comparison = compiled.model.constraints[label]
        assert dict(comparison.lhs.linear) == expected_linear
        assert dict(comparison.lhs.quadratic) == {}
        assert comparison.sense is expected_sense
        assert comparison.rhs == expected_rhs

    @pytest.mark.parametrize("label", sorted(HARD_PASSTHROUGH_EXPECTED))
    def test_lhs_keeps_integer_vartypes_and_declared_bounds(self, label):
        compiled = compile_problem(hard_passthrough_problem())
        lhs = compiled.model.constraints[label].lhs

        for name in HARD_PASSTHROUGH_EXPECTED[label][0]:
            lower, upper = DECLARED_BOUNDS[name]
            assert lhs.vartype(name) is dimod.INTEGER
            assert lhs.lower_bound(name) == lower
            assert lhs.upper_bound(name) == upper

    @pytest.mark.parametrize("label", sorted(HARD_PASSTHROUGH_EXPECTED))
    def test_hard_constraints_are_never_soft(self, label):
        compiled = compile_problem(hard_passthrough_problem())

        assert label not in compiled.model._soft
        trace = trace_by_id(compiled)[label]
        assert trace.native is True
        assert trace.penalty is None
        assert trace.generated_variables == []
        assert trace.slack_range is None
        assert trace.redundant is False

    def test_no_slack_and_no_internal_variables(self):
        problem = hard_passthrough_problem()
        compiled = compile_problem(problem)

        assert compiled.internal_variables == set()
        assert compiled.num_variables == len(problem.variables)
        assert not any(str(name).startswith("__") for name in compiled.model.variables)

    def test_declared_variables_keep_their_vartype_and_bounds(self):
        compiled = compile_problem(hard_passthrough_problem())
        cqm = compiled.model

        assert set(map(str, cqm.variables)) == set(DECLARED_BOUNDS)
        for name, (lower, upper) in DECLARED_BOUNDS.items():
            expected = dimod.BINARY if name == "b" else dimod.INTEGER
            assert cqm.vartype(name) is expected
            assert cqm.lower_bound(name) == lower
            assert cqm.upper_bound(name) == upper


# --------------------------------------------------------------------------
# (d) an integer self-product stays a quadratic objective term
# --------------------------------------------------------------------------


def self_product_problem(direction: str) -> OptimizationProblem:
    """``2.5 * x * x - 1.5 * x * b + 4x - 2b + 1.5``, no constraints."""
    return make_problem(
        [integer("x", -2, 3), binary("b")],
        direction=direction,
        linear=[lin("x", 4), lin("b", -2)],
        quadratic=[quad("x", "x", 2.5), quad("x", "b", -1.5)],
        constant=1.5,
        name=f"integer self product ({direction})",
    )


class TestSelfProductObjective:
    """``x * x`` of an INTEGER variable stays a ``(x, x)`` quadratic entry
    (3b §15.1), negated as a whole when the direction is maximize."""

    def test_minimize_keeps_the_coefficient(self):
        cqm = compile_problem(self_product_problem("minimize")).model

        assert cqm.objective.get_quadratic("x", "x") == pytest.approx(2.5)

    def test_maximize_negates_the_coefficient(self):
        cqm = compile_problem(self_product_problem("maximize")).model

        assert cqm.objective.get_quadratic("x", "x") == pytest.approx(-2.5)

    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_binary_never_gets_a_self_product(self, direction):
        cqm = compile_problem(self_product_problem(direction)).model

        assert ("b", "b") not in dict(cqm.objective.quadratic)
        assert cqm.objective.vartype("b") is dimod.BINARY

    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_objective_energy_equals_the_signed_objective(self, direction):
        problem = self_product_problem(direction)
        cqm = compile_problem(problem).model
        sign = -1.0 if direction == "maximize" else 1.0

        assignments = business_assignments(problem)
        assert len(assignments) == 12
        for assignment in assignments:
            expected = sign * evaluate_objective(problem.objective, assignment)
            assert cqm.objective.energy(assignment) == pytest.approx(expected), assignment


# --------------------------------------------------------------------------
# (e) examples/integer_knapsack.json through the service on a CQM backend
# --------------------------------------------------------------------------


INTEGER_KNAPSACK_EXPECTED = {"item_a": 0, "item_b": 1, "item_c": 1, "item_d": 3}
INTEGER_KNAPSACK_OBJECTIVE = 34.0


class TestIntegerKnapsackOnFakeCQM:
    """The shipped integer example must reach the same optimum on the CQM
    path as on the shipped exact (BQM) path, with no internal column leaking
    into the reported solutions."""

    def test_optimum_matches_the_hand_checked_expectation(self):
        result = solve_on_fake(load_example("integer_knapsack.json"))

        assert result.status == "success"
        assert result.backend == FAKE_LOCAL_CQM_NAME
        assert result.metadata is not None
        assert result.metadata.model_type == "cqm"
        best = result.solutions[0]
        assert best.rank == 1
        assert best.objective_value == pytest.approx(INTEGER_KNAPSACK_OBJECTIVE)
        assert best.variables == INTEGER_KNAPSACK_EXPECTED
        assert best.hard_constraints_satisfied is True

    def test_optimum_equals_the_exact_bqm_path(self):
        problem = load_example("integer_knapsack.json")
        assert problem.solver.backend == "exact"
        exact = OptimizationService().solve(problem)
        assert exact.status == "success"
        assert exact.backend == "exact"

        result = solve_on_fake(problem)

        assert result.solutions[0].variables == exact.solutions[0].variables
        assert result.solutions[0].objective_value == pytest.approx(
            exact.solutions[0].objective_value
        )

    def test_solutions_hold_business_variables_in_problem_order(self):
        problem = load_example("integer_knapsack.json")
        expected_order = [variable.name for variable in problem.variables]

        result = solve_on_fake(problem, top_k=10)

        assert result.solutions
        for solution in result.solutions:
            assert list(solution.variables) == expected_order
            assert not any(name.startswith("__") for name in solution.variables)

    def test_single_attempt_without_penalty(self):
        result = solve_on_fake(load_example("integer_knapsack.json"))

        assert len(result.attempts) == 1
        assert result.attempts[0].penalty is None


# --------------------------------------------------------------------------
# (f) negative lower bounds end to end, and decode on a raw fake result
# --------------------------------------------------------------------------


def negative_bounds_problem() -> OptimizationProblem:
    """Two integers with negative lower bounds, a hard cap and an integer
    soft floor (which generates ``__slack_floor``)."""
    return make_problem(
        [integer("x", -4, 2), integer("y", -1, 5), binary("b")],
        linear=[lin("x", -1), lin("y", 2), lin("b", -3)],
        quadratic=[quad("x", "x", 1)],
        constraints=[
            constraint("cap", "<=", 3, [lin("x", 1), lin("y", 1)]),
            constraint(
                "floor", ">=", 0, [lin("x", 1), lin("y", 1)], type="soft", weight=2.0
            ),
        ],
        name="negative lower bounds",
    )


def soft_violating_optimum_problem() -> OptimizationProblem:
    """The best hard-feasible assignment breaks the soft constraint: the
    objective gain (1 per unit) beats the weight-0.1 quadratic penalty."""
    return make_problem(
        [integer("x", -4, 2), binary("b")],
        linear=[lin("x", 1), lin("b", -1)],
        constraints=[
            constraint("cap", "<=", 2, [lin("x", 1), lin("b", 1)]),
            constraint("nonneg", ">=", 0, [lin("x", 1)], type="soft", weight=0.1),
        ],
        name="optimum violates the soft constraint",
    )


NEGATIVE_BOUNDS = {"x": (-4, 2), "y": (-1, 5), "b": (0, 1)}


class TestNegativeLowerBoundsOnFakeCQM:
    """Negative integer bounds survive the round trip: every decoded value is
    an in-range int, feasibility is decided by the hard constraints alone and
    the slack never reaches the reported solutions."""

    def test_problems_are_valid(self):
        assert validate_problem(negative_bounds_problem()) == []
        assert validate_problem(soft_violating_optimum_problem()) == []

    def test_slack_is_generated_for_the_integer_soft_constraint(self):
        compiled = compile_problem(negative_bounds_problem())

        assert compiled.internal_variables == {"__slack_floor"}
        assert trace_by_id(compiled)["floor"].native is False
        assert trace_by_id(compiled)["cap"].native is True

    def test_every_decoded_value_is_an_int_inside_its_bounds(self):
        result = solve_on_fake(negative_bounds_problem(), top_k=25)

        assert result.status == "success"
        assert result.solutions
        for solution in result.solutions:
            assert list(solution.variables) == list(NEGATIVE_BOUNDS)
            assert solution.hard_constraints_satisfied is True
            for name, value in solution.variables.items():
                lower, upper = NEGATIVE_BOUNDS[name]
                assert not name.startswith("__")
                assert isinstance(value, int)
                assert not isinstance(value, bool)
                assert lower <= value <= upper

    def test_first_place_equals_the_brute_force_optimum(self):
        problem = negative_bounds_problem()
        ranked = brute_force_ranked(problem)
        best_ranking, best_assignment = ranked[0]
        # A unique optimum, so the comparison cannot be blurred by a tie.
        assert ranked[1][0] > best_ranking

        result = solve_on_fake(problem)

        best = result.solutions[0]
        assert best.variables == best_assignment
        assert best.ranking_score == pytest.approx(best_ranking)

    def test_feasibility_ignores_the_soft_constraint(self):
        problem = soft_violating_optimum_problem()
        ranked = brute_force_ranked(problem)
        assert ranked[1][0] > ranked[0][0]

        result = solve_on_fake(problem)

        best = result.solutions[0]
        assert best.variables == ranked[0][1]
        assert best.variables == {"x": -4, "b": 1}
        assert best.hard_constraints_satisfied is True
        assert best.soft_violation_score > 0
        assert best.soft_violation_score == pytest.approx(1.6)

    def test_decode_drops_the_slack_column_and_restores_order(self):
        problem = route_to_fake(negative_bounds_problem())
        compiled = compile_problem(problem)
        raw = FakeLocalCQMBackend().solve(compiled, problem.solver)
        # The raw result really does carry the slack, in the sampler's own
        # (binary-first) order rather than the problem's.
        assert "__slack_floor" in raw.variables
        assert raw.variables != [variable.name for variable in problem.variables]

        decoded = CQMCompiler().decode(compiled, raw)

        assert decoded.variables == [variable.name for variable in problem.variables]
        assert not any(name.startswith("__") for name in decoded.variables)
        assert decoded.samples.dtype == np.int64
        assert decoded.samples.shape == (raw.samples.shape[0], len(problem.variables))
        assert np.array_equal(decoded.energies, raw.energies)
        for index, name in enumerate(decoded.variables):
            source = list(raw.variables).index(name)
            assert np.array_equal(decoded.samples[:, index], raw.samples[:, source])


# --------------------------------------------------------------------------
# (g) the §9.4 estimate is exactly the compiled variable count
# --------------------------------------------------------------------------


def estimate_corpus() -> list[OptimizationProblem]:
    """Every problem this module builds, plus the two shipped knapsacks."""
    return [
        *ALL_SOFT_PROBLEMS,
        *(problem for _, problem in hand_written_soft_problems()),
        *BINARY_ONLY_PROBLEMS,
        mixed_soft_problem(),
        hard_passthrough_problem(),
        self_product_problem("minimize"),
        self_product_problem("maximize"),
        negative_bounds_problem(),
        soft_violating_optimum_problem(),
        load_example("integer_knapsack.json"),
        load_example("knapsack.json"),
    ]


ESTIMATE_CORPUS = estimate_corpus()


class TestEstimateMatchesCompiledCount:
    """``estimate_cqm_variables`` (3b §9.4) must predict the compiled model
    exactly: declared variables plus the objective-form integer slacks."""

    @pytest.mark.parametrize("index", range(len(ESTIMATE_CORPUS)))
    def test_estimate_equals_the_compiled_model(self, index):
        problem = ESTIMATE_CORPUS[index]
        compiled = compile_problem(problem)

        assert estimate_cqm_variables(problem) == compiled.num_variables
        assert compiled.num_variables == len(list(compiled.model.variables))

    def test_corpus_actually_exercises_slack_generation(self):
        with_slack = [
            problem
            for problem in ESTIMATE_CORPUS
            if compile_problem(problem).internal_variables
        ]
        assert with_slack, "the estimate would be trivially right without a slack"

    @pytest.mark.parametrize(
        "problem",
        [
            load_example("knapsack.json"),
            load_example("integer_knapsack.json"),
            negative_bounds_problem(),
        ],
        ids=["knapsack", "integer knapsack", "negative bounds with slack"],
    )
    def test_validate_problem_full_reports_the_same_estimate(self, problem):
        compiled = compile_problem(problem)

        result = validate_problem_full(problem, model_type="cqm")

        assert result.valid is True
        assert result.model_type == "cqm"
        assert result.estimated_compiled_variables == compiled.num_variables

    def test_shipped_examples_have_the_expected_counts(self):
        # Guards the examples themselves: 4 binaries, and 4 integers plus the
        # one slack the soft "<=" over integers needs.
        assert estimate_cqm_variables(load_example("knapsack.json")) == 4
        assert estimate_cqm_variables(load_example("integer_knapsack.json")) == 5
