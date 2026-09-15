"""BQM 路徑整數變數 binary expansion 的行為測試（3b spec §8、§12-§14）。

證據分三層：

* **端到端**：``examples/integer_knapsack.json`` 走完整 service（exact
  backend），第一名必須等於獨立暴力枚舉出來的唯一最佳解。
* **位元層**：對整個 BQM 的所有位元組合枚舉，逐一比對
  ``bqm.energy(bits) == sign * objective(decode 值) + Σ penalty(decode 值)``，
  並確認 hard penalty 對整數變數的可行/不可行判別與
  ``validate_solution`` 一致。
* **估算層**：``estimate_compiled_variables`` 與實際
  ``compile(...).num_variables`` 在手寫問題與大量隨機小問題上完全相等。

所有隨機部分都用固定種子；不 skip、不 xfail。
"""

import itertools
import json
import math
import random

import numpy as np
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import (
    Constraint,
    LinearTerm,
    Objective,
    OptimizationProblem,
    QuadraticTerm,
    Variable,
)
from annealbridge.orchestration import OptimizationService, evaluate_objective
from annealbridge.solvers.base import RawSolverResult
from annealbridge.validation import validate_problem, validate_solution
from annealbridge.validation.estimates import (
    accumulate_terms,
    analyze_inequality,
    compute_penalty_scale,
    compute_slack_coefficients,
    estimate_compiled_variables,
    integer_encoding_bits,
    variable_bounds,
)
from tests.conftest import EXAMPLES_DIR
from tests.model_builders import lin, quad

HARD_PENALTY = 10.0


# --------------------------------------------------------------------------
# Helpers
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
    name: str = "integer compiler test",
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


def compile_problem(problem: OptimizationProblem, hard_penalty: float = HARD_PENALTY):
    return BQMCompiler().compile(problem, hard_penalty)


def model_variables(compiled) -> list[str]:
    """The compiled model's variable names, in the model's own order."""
    return [str(variable) for variable in compiled.model.variables]


def all_bit_samples(names: list[str]):
    """Every 0/1 assignment of ``names`` as a dict."""
    for bits in itertools.product((0, 1), repeat=len(names)):
        yield dict(zip(names, bits))


def decode_by_hand(compiled, sample: dict[str, int]) -> dict[str, int]:
    """``lower + sum(coefficient * bit)`` per business variable, from the encodings."""
    assignment: dict[str, int] = {}
    for variable in compiled.original_problem.variables:
        encoding = compiled.integer_encodings.get(variable.name)
        if encoding is None:
            assignment[variable.name] = int(sample[variable.name])
            continue
        assignment[variable.name] = encoding.lower + sum(
            coefficient * sample[bit]
            for coefficient, bit in zip(encoding.coefficients, encoding.bits)
        )
    return assignment


def slack_values_by_hand(compiled, sample: dict[str, int]) -> dict[str, int]:
    """``sum(coefficient * bit)`` per constraint id, 0 when no slack was generated."""
    values: dict[str, int] = {}
    for trace in compiled.constraint_trace:
        if not trace.generated_variables:
            values[trace.constraint_id] = 0
            continue
        assert trace.slack_range is not None
        coefficients = compute_slack_coefficients(trace.slack_range)
        values[trace.constraint_id] = sum(
            coefficient * sample[name]
            for coefficient, name in zip(coefficients, trace.generated_variables)
        )
    return values


def penalty_energy(compiled, assignment: dict[str, int], slack: dict[str, int]) -> float:
    """``sum(lambda * (lhs [+ slack] - rhs)^2)`` over the business assignment.

    Computed from the *original* constraints and the pure ``estimates``
    arithmetic only — never from the compiled bit-level expansion — so it is
    an independent check of what the compiler emitted.
    """
    problem = compiled.original_problem
    bounds = variable_bounds(problem)
    total = 0.0
    for constraint_model, trace in zip(problem.constraints, compiled.constraint_trace):
        assert trace.constraint_id == constraint_model.id
        if trace.redundant:
            continue
        lam = trace.penalty
        if constraint_model.operator == "==":
            coefficients = accumulate_terms(constraint_model.terms)
            lhs = sum(
                value * assignment[name] for name, value in coefficients.items()
            )
            total += lam * (lhs - constraint_model.rhs) ** 2
            continue
        analysis = analyze_inequality(constraint_model, bounds)
        lhs = sum(
            value * assignment[name] for name, value in analysis.coefficients.items()
        )
        deviation = lhs + slack[constraint_model.id] - analysis.rhs
        total += lam * deviation * deviation
    return total


def expected_energy(compiled, sample: dict[str, int]) -> float:
    """``sign * objective(decoded) + penalties(decoded)`` for one bit sample."""
    problem = compiled.original_problem
    sign = -1.0 if problem.objective.direction == "maximize" else 1.0
    assignment = decode_by_hand(compiled, sample)
    slack = slack_values_by_hand(compiled, sample)
    return sign * evaluate_objective(problem.objective, assignment) + penalty_energy(
        compiled, assignment, slack
    )


def business_assignments(problem: OptimizationProblem) -> list[dict[str, int]]:
    """Every assignment of the declared variables over their bounds."""
    bounds = variable_bounds(problem)
    names = list(bounds)
    ranges = [range(bounds[name][0], bounds[name][1] + 1) for name in names]
    return [dict(zip(names, values)) for values in itertools.product(*ranges)]


def raw_of(compiled, samples: np.ndarray, dtype=np.int8) -> RawSolverResult:
    names = model_variables(compiled)
    matrix = np.asarray(samples, dtype=dtype).reshape(-1, len(names))
    return RawSolverResult(
        variables=names,
        samples=matrix,
        energies=np.zeros(matrix.shape[0], dtype=np.float64),
        backend="fake",
    )


def exhaustive_raw(compiled, dtype=np.int8) -> RawSolverResult:
    """A raw result holding *every* bit combination of the compiled model."""
    names = model_variables(compiled)
    rows = list(itertools.product((0, 1), repeat=len(names)))
    return raw_of(compiled, np.asarray(rows, dtype=dtype), dtype=dtype)


# --------------------------------------------------------------------------
# 1. examples/integer_knapsack.json: service optimum == brute force
# --------------------------------------------------------------------------

KNAPSACK_EXPECTED = {"item_a": 0, "item_b": 1, "item_c": 1, "item_d": 3}
KNAPSACK_OBJECTIVE = 34.0


def load_knapsack() -> OptimizationProblem:
    payload = json.loads((EXAMPLES_DIR / "integer_knapsack.json").read_text(encoding="utf-8"))
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


class TestIntegerKnapsackExample:
    def test_service_optimum_equals_brute_force(self):
        problem = load_knapsack()
        assert validate_problem(problem) == []
        assert problem.solver.backend == "exact"

        ranked = brute_force_ranked(problem)
        assert len(ranked) > 1
        best_ranking, best_assignment = ranked[0]
        # (a) the brute-force optimum is unique.
        assert ranked[1][0] < best_ranking

        result = OptimizationService().solve(problem)
        assert result.status == "success"
        assert result.backend == "exact"
        best = result.solutions[0]

        # (b) / (c): service first place == brute force optimum.
        assert best.variables == best_assignment
        assert best.objective_value == pytest.approx(KNAPSACK_OBJECTIVE)
        assert best.ranking_score == pytest.approx(best_ranking)
        # The hard-coded expectation guards the example file itself.
        assert best_assignment == KNAPSACK_EXPECTED
        assert best.variables == KNAPSACK_EXPECTED

    def test_brute_force_runner_up_is_strictly_worse(self):
        # The example's docstring claims a unique optimum with a clear gap;
        # pin the runner-up so a coefficient edit cannot silently blur it.
        ranked = brute_force_ranked(load_knapsack())
        assert ranked[0][0] == pytest.approx(34.0)
        assert ranked[1][0] == pytest.approx(32.0)

    def test_solutions_hold_business_integers_only(self):
        problem = load_knapsack()
        result = OptimizationService().solve(problem)
        assert result.solutions
        for solution in result.solutions:
            assert set(solution.variables) == set(KNAPSACK_EXPECTED)
            for name, value in solution.variables.items():
                # (d): no internal name leaks, every value is an int in range.
                assert not name.startswith("__")
                assert isinstance(value, int)
                assert not isinstance(value, bool)
                assert 0 <= value <= 3

    def test_estimate_matches_compiled_count(self):
        problem = load_knapsack()
        compiled = compile_problem(problem)
        # 4 * 2 encoding bits + 5 capacity slack bits + 2 soft slack bits.
        assert compiled.num_variables == 15
        assert estimate_compiled_variables(problem) == compiled.num_variables


# --------------------------------------------------------------------------
# 2. decode with negative lower bounds
# --------------------------------------------------------------------------


def negative_bounds_problem() -> OptimizationProblem:
    return make_problem(
        [integer("x", -3, 2), integer("y", -1, 1), binary("b")],
        linear=[lin("x", 1), lin("y", 2), lin("b", 3)],
        constraints=[
            constraint("cap", "<=", 1, [lin("x", 1), lin("y", 1)]),
        ],
    )


class TestDecodeNegativeLowerBounds:
    def test_decode_folds_bits_back_into_integers(self):
        problem = negative_bounds_problem()
        compiled = compile_problem(problem)
        raw = exhaustive_raw(compiled)
        decoded = BQMCompiler().decode(compiled, raw)

        assert decoded.variables == [v.name for v in problem.variables]
        assert decoded.samples.dtype == np.int64
        assert decoded.num_samples == raw.num_samples
        assert decoded.backend == raw.backend
        assert np.array_equal(decoded.energies, raw.energies)

        bounds = variable_bounds(problem)
        seen: dict[str, set[int]] = {name: set() for name in decoded.variables}
        for row_bits, row_values in zip(raw.as_dicts(), decoded.samples.tolist()):
            by_hand = decode_by_hand(compiled, row_bits)
            for position, name in enumerate(decoded.variables):
                value = row_values[position]
                assert value == by_hand[name]
                lower, upper = bounds[name]
                assert lower <= value <= upper
                seen[name].add(value)

        for name, values in seen.items():
            lower, upper = bounds[name]
            assert values == set(range(lower, upper + 1))

    def test_encoding_shape(self):
        problem = negative_bounds_problem()
        compiled = compile_problem(problem)
        encodings = compiled.integer_encodings
        assert set(encodings) == {"x", "y"}
        assert encodings["x"].lower == -3
        assert encodings["x"].bits == ["__int_x_0", "__int_x_1", "__int_x_2"]
        assert encodings["x"].coefficients == compute_slack_coefficients(5)
        assert encodings["y"].lower == -1
        assert encodings["y"].bits == ["__int_y_0", "__int_y_1"]
        assert len(encodings["y"].coefficients) == integer_encoding_bits(-1, 1)
        assert "b" in model_variables(compiled)

    def test_decode_reports_a_missing_encoding_bit(self):
        problem = negative_bounds_problem()
        compiled = compile_problem(problem)
        names = [name for name in model_variables(compiled) if name != "__int_x_0"]
        raw = RawSolverResult(
            variables=names,
            samples=np.zeros((1, len(names)), dtype=np.int8),
            energies=np.zeros(1),
            backend="fake",
        )
        with pytest.raises(ValueError, match="__int_x_0"):
            BQMCompiler().decode(compiled, raw)


# --------------------------------------------------------------------------
# 3. energy identity with x*x in the objective
# --------------------------------------------------------------------------


def squared_objective_problem(direction: str) -> OptimizationProblem:
    # 2*x*x - 3*x*y + 4*x*b + x over x in [-2, 3], y in [0, 2], b binary,
    # with one equality (no slack) and one inequality (slack bits).
    return make_problem(
        [integer("x", -2, 3), integer("y", 0, 2), binary("b")],
        direction=direction,
        linear=[lin("x", 1)],
        quadratic=[quad("x", "x", 2), quad("x", "y", -3), quad("x", "b", 4)],
        constraints=[
            constraint("eq", "==", 2, [lin("x", 1), lin("y", 1)]),
            constraint("cap", "<=", 2, [lin("x", 1), lin("y", -1)]),
        ],
    )


class TestEnergyIdentityWithSquares:
    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_energy_matches_objective_plus_penalties(self, direction):
        problem = squared_objective_problem(direction)
        compiled = compile_problem(problem, hard_penalty=7.0)
        bqm = compiled.model
        names = model_variables(compiled)
        # 3 bits (x) + 2 bits (y) + b + 3 slack bits for "cap".
        assert len(names) == 9
        assert compiled.num_variables == estimate_compiled_variables(problem)
        sign = -1.0 if direction == "maximize" else 1.0

        checked = 0
        for sample in all_bit_samples(names):
            assignment = decode_by_hand(compiled, sample)
            x, y, b = assignment["x"], assignment["y"], assignment["b"]
            objective = 2 * x * x - 3 * x * y + 4 * x * b + x
            assert objective == evaluate_objective(problem.objective, assignment)
            slack = slack_values_by_hand(compiled, sample)
            expected = (
                sign * objective
                + 7.0 * (x + y - 2) ** 2
                + 7.0 * (x - y + slack["cap"] - 2) ** 2
            )
            assert math.isclose(bqm.energy(sample), expected, rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(
                expected_energy(compiled, sample), expected, rel_tol=1e-9, abs_tol=1e-9
            )
            checked += 1
        assert checked == 2 ** len(names)

    def test_pure_square_without_constraints(self):
        # x*x alone: every bit pattern must reproduce the integer square.
        problem = make_problem(
            [integer("x", -2, 3)],
            quadratic=[quad("x", "x", 1)],
            constant=1.5,
        )
        compiled = compile_problem(problem)
        bqm = compiled.model
        for sample in all_bit_samples(model_variables(compiled)):
            x = decode_by_hand(compiled, sample)["x"]
            assert math.isclose(bqm.energy(sample), x * x + 1.5, abs_tol=1e-9)


# --------------------------------------------------------------------------
# 4. estimate_compiled_variables == compiled.num_variables
# --------------------------------------------------------------------------


def hand_written_problems() -> list[tuple[str, OptimizationProblem]]:
    return [
        (
            "negative lower bound + equality",
            make_problem(
                [integer("x", -4, 2), integer("y", -1, 5), binary("b")],
                linear=[lin("x", 1), lin("y", -2)],
                constraints=[
                    constraint("eq", "==", 0, [lin("x", 1), lin("y", 1), lin("b", 3)])
                ],
            ),
        ),
        (
            "less-equal with slack",
            make_problem(
                [integer("x", 0, 7), integer("y", 0, 3)],
                linear=[lin("x", 2)],
                constraints=[constraint("cap", "<=", 5, [lin("x", 1), lin("y", 1)])],
            ),
        ),
        (
            "greater-equal with slack",
            make_problem(
                [integer("x", -2, 4), binary("b")],
                linear=[lin("x", 1)],
                constraints=[constraint("cover", ">=", -1, [lin("x", 1), lin("b", 2)])],
            ),
        ),
        (
            "redundant inequality adds nothing",
            make_problem(
                [integer("x", 0, 3), integer("y", 0, 3)],
                linear=[lin("x", 1)],
                constraints=[
                    constraint("loose", "<=", 100, [lin("x", 1), lin("y", 1)]),
                    constraint("also_loose", ">=", -100, [lin("x", 1)]),
                ],
            ),
        ),
        (
            "soft inequality",
            make_problem(
                [integer("x", -3, 3), binary("b")],
                direction="maximize",
                linear=[lin("x", 4), lin("b", 1)],
                constraints=[
                    constraint(
                        "prefer",
                        "<=",
                        1,
                        [lin("x", 1), lin("b", 1)],
                        type="soft",
                        weight=2.5,
                    )
                ],
            ),
        ),
        (
            "pure binary (IR 1.0)",
            make_problem(
                [binary("a"), binary("b"), binary("c")],
                version="1.0",
                linear=[lin("a", 1), lin("b", 2)],
                quadratic=[quad("a", "b", 3)],
                constraints=[
                    constraint("pick", "==", 1, [lin("a", 1), lin("b", 1), lin("c", 1)]),
                    constraint("cap", "<=", 2, [lin("a", 1), lin("c", 1)]),
                ],
            ),
        ),
        (
            "mixed with quadratic and every operator",
            make_problem(
                [integer("x", -2, 2), integer("y", 0, 6), binary("b")],
                direction="maximize",
                linear=[lin("x", 1), lin("y", 1), lin("b", 1)],
                quadratic=[quad("x", "x", 1), quad("x", "y", -1), quad("y", "b", 2)],
                constraints=[
                    constraint("eq", "==", 3, [lin("y", 1), lin("b", 1)]),
                    constraint("cap", "<=", 4, [lin("x", 2), lin("y", 1)]),
                    constraint("cover", ">=", 0, [lin("x", 1), lin("y", 1)]),
                    constraint(
                        "soft_cap",
                        "<=",
                        3,
                        [lin("y", 1)],
                        type="soft",
                        weight=1.0,
                    ),
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
            variables.append(integer(name, lower, lower + rng.randint(1, 6)))
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
    for index in range(rng.randint(0, 3)):
        soft = rng.random() < 0.5
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


def valid_random_problems(count: int, *, seed: int = 20260908) -> list[OptimizationProblem]:
    """``count`` random problems that ``validate_problem`` accepts."""
    rng = random.Random(seed)
    problems: list[OptimizationProblem] = []
    attempts = 0
    while len(problems) < count and attempts < count * 40:
        attempts += 1
        problem = random_problem(rng)
        if validate_problem(problem):
            continue
        problems.append(problem)
    assert len(problems) == count, f"only {len(problems)} valid problems in {attempts} tries"
    return problems


RANDOM_PROBLEMS = valid_random_problems(100)


def assert_estimate_matches(problem: OptimizationProblem) -> None:
    compiled = compile_problem(problem)
    assert estimate_compiled_variables(problem) == compiled.num_variables
    assert compiled.num_variables == len(model_variables(compiled))

    encoding_bits = {
        bit for encoding in compiled.integer_encodings.values() for bit in encoding.bits
    }
    slack_bits = {
        name for trace in compiled.constraint_trace for name in trace.generated_variables
    }
    assert set(compiled.internal_variables) == encoding_bits | slack_bits
    assert set(compiled.integer_encodings) == {
        variable.name for variable in problem.variables if variable.type == "integer"
    }
    for name, encoding in compiled.integer_encodings.items():
        lower, upper = variable_bounds(problem)[name]
        assert encoding.lower == lower
        assert len(encoding.bits) == integer_encoding_bits(lower, upper)
        assert encoding.coefficients == compute_slack_coefficients(upper - lower)
        assert encoding.variable == name
    # Business names and internal names never overlap.
    business = {variable.name for variable in problem.variables}
    assert business.isdisjoint(compiled.internal_variables)


class TestCompiledVariableEstimate:
    @pytest.mark.parametrize(
        "problem",
        [problem for _, problem in hand_written_problems()],
        ids=[label for label, _ in hand_written_problems()],
    )
    def test_hand_written(self, problem):
        assert validate_problem(problem) == []
        assert_estimate_matches(problem)

    @pytest.mark.parametrize("index", range(len(RANDOM_PROBLEMS)))
    def test_random(self, index):
        assert_estimate_matches(RANDOM_PROBLEMS[index])

    def test_random_corpus_actually_contains_integers(self):
        with_integers = [
            problem
            for problem in RANDOM_PROBLEMS
            if any(v.type == "integer" for v in problem.variables)
        ]
        assert len(with_integers) >= 80
        assert any(
            v.lower_bound is not None and v.lower_bound < 0
            for problem in RANDOM_PROBLEMS
            for v in problem.variables
        )


# --------------------------------------------------------------------------
# 5. compute_penalty_scale as an upper bound
# --------------------------------------------------------------------------


def small_random_integer_problems(count: int = 40, *, seed: int = 5150):
    """Random problems small enough to enumerate every integer assignment."""
    chosen = []
    for problem in valid_random_problems(count * 4, seed=seed):
        if not any(v.type == "integer" for v in problem.variables):
            continue
        bounds = variable_bounds(problem)
        span = 1
        for lower, upper in bounds.values():
            span *= upper - lower + 1
        if span > 400:
            continue
        chosen.append(problem)
        if len(chosen) == count:
            break
    assert len(chosen) == count
    return chosen


PENALTY_PROBLEMS = small_random_integer_problems()


class TestPenaltyScaleBound:
    @pytest.mark.parametrize("index", range(len(PENALTY_PROBLEMS)))
    def test_scale_dominates_objective_and_soft_energy(self, index):
        problem = PENALTY_PROBLEMS[index]
        scale = compute_penalty_scale(problem)

        objectives = []
        softs = []
        for assignment in business_assignments(problem):
            objective = evaluate_objective(problem.objective, assignment)
            objectives.append(objective)
            softs.append(validate_solution(problem, assignment).soft_violation_score)
        # ``compute_objective_scale`` bounds the objective's *range* (spec §18:
        # ``penalty_scale >= objective_max - objective_min + soft_bound``), so the
        # constant cancels and is not subtracted.
        objective_range = max(objectives) - min(objectives)
        worst_soft = max(softs)
        assert scale >= objective_range + worst_soft - 1e-9

    def test_hand_written_problems_are_dominated(self):
        for _, problem in hand_written_problems():
            bounds = variable_bounds(problem)
            span = 1
            for lower, upper in bounds.values():
                span *= upper - lower + 1
            if span > 4000:
                continue
            scale = compute_penalty_scale(problem)
            objectives = []
            softs = []
            for assignment in business_assignments(problem):
                objectives.append(evaluate_objective(problem.objective, assignment))
                softs.append(validate_solution(problem, assignment).soft_violation_score)
            objective_range = max(objectives) - min(objectives)
            assert scale >= objective_range + max(softs) - 1e-9


# --------------------------------------------------------------------------
# 6. hard == and <= penalties over integer variables
# --------------------------------------------------------------------------


EQUALITY_PROBLEM = make_problem(
    [integer("x", 0, 3), integer("y", -1, 2)],
    linear=[lin("x", 1), lin("y", -1)],
    constraints=[constraint("eq", "==", 3, [lin("x", 1), lin("y", 2)])],
    name="hard equality",
)

INEQUALITY_PROBLEM = make_problem(
    [integer("x", 0, 3), integer("y", -1, 2)],
    direction="maximize",
    linear=[lin("x", 2), lin("y", 1)],
    constraints=[constraint("cap", "<=", 3, [lin("x", 2), lin("y", -1)])],
    name="hard inequality",
)


class TestHardPenaltyOverIntegers:
    @pytest.mark.parametrize(
        "problem", [EQUALITY_PROBLEM, INEQUALITY_PROBLEM], ids=["==", "<="]
    )
    def test_feasible_reaches_zero_penalty_and_infeasible_never_does(self, problem):
        compiled = compile_problem(problem, hard_penalty=9.0)
        bqm = compiled.model
        names = model_variables(compiled)
        slack_names = sorted(compiled.internal_variables - set(
            bit
            for encoding in compiled.integer_encodings.values()
            for bit in encoding.bits
        ))
        business_bits = [name for name in names if name not in slack_names]
        sign = -1.0 if problem.objective.direction == "maximize" else 1.0

        feasible_seen = infeasible_seen = 0
        for business in all_bit_samples(business_bits):
            assignment = decode_by_hand(compiled, business)
            objective = evaluate_objective(problem.objective, assignment)
            residuals = []
            for slack in all_bit_samples(slack_names):
                sample = {**business, **slack}
                residuals.append(bqm.energy(sample) - sign * objective)
            best = min(residuals)
            assert all(residual >= -1e-9 for residual in residuals)
            if validate_solution(problem, assignment).feasible:
                feasible_seen += 1
                assert math.isclose(best, 0.0, abs_tol=1e-9)
            else:
                infeasible_seen += 1
                # Every slack choice is strictly penalised: one unit of
                # violation costs at least the hard penalty.
                assert best > 1e-9
                assert best >= 9.0 - 1e-9
        assert feasible_seen > 0
        assert infeasible_seen > 0

    def test_equality_has_no_slack_and_inequality_does(self):
        eq = compile_problem(EQUALITY_PROBLEM)
        (eq_trace,) = eq.constraint_trace
        assert eq_trace.generated_variables == []
        assert eq_trace.slack_range is None

        ineq = compile_problem(INEQUALITY_PROBLEM)
        (ineq_trace,) = ineq.constraint_trace
        # lhs of 2x - y over x in [0,3], y in [-1,2] has minimum -2, so the
        # slack must reach 3 - (-2) = 5.
        assert ineq_trace.slack_range == 5
        assert ineq_trace.generated_variables == [
            "__slack_cap_0",
            "__slack_cap_1",
            "__slack_cap_2",
        ]


# --------------------------------------------------------------------------
# 7. all-binary problems are byte-for-byte the old behaviour
# --------------------------------------------------------------------------


class TestAllBinaryUnchanged:
    def test_no_encodings_and_int8_is_preserved(self):
        problem = make_problem(
            [binary("a"), binary("b"), binary("c")],
            version="1.0",
            linear=[lin("a", 1), lin("b", -2)],
            quadratic=[quad("a", "b", 3)],
            constraints=[
                constraint("pick", "==", 1, [lin("a", 1), lin("b", 1)]),
                constraint("cap", "<=", 1, [lin("b", 1), lin("c", 1)]),
            ],
        )
        compiled = compile_problem(problem)
        assert compiled.integer_encodings == {}
        assert compiled.internal_variables == {"__slack_cap_0"}

        raw = exhaustive_raw(compiled)
        assert raw.samples.dtype == np.int8
        decoded = BQMCompiler().decode(compiled, raw)
        assert decoded.variables == ["a", "b", "c"]
        assert decoded.samples.dtype == np.int8
        assert decoded.num_samples == raw.num_samples
        for row_bits, row_values in zip(raw.as_dicts(), decoded.samples.tolist()):
            assert row_values == [row_bits["a"], row_bits["b"], row_bits["c"]]

    def test_binary_typed_as_integer_zero_one_still_encodes(self):
        # An integer variable declared as 0..1 is one bit, but it is still an
        # integer: it gets an encoding entry and decode returns int64.
        problem = make_problem([integer("x", 0, 1)], linear=[lin("x", 1)])
        compiled = compile_problem(problem)
        assert set(compiled.integer_encodings) == {"x"}
        assert compiled.num_variables == 1
        decoded = BQMCompiler().decode(compiled, exhaustive_raw(compiled))
        assert decoded.samples.dtype == np.int64
        assert sorted(decoded.samples.ravel().tolist()) == [0, 1]


# --------------------------------------------------------------------------
# 8. internal names never leak
# --------------------------------------------------------------------------


class TestInternalNamesNeverLeak:
    def test_service_solutions_have_no_internal_keys(self):
        result = OptimizationService().solve(load_knapsack())
        assert result.solutions
        for solution in result.solutions:
            assert all(not name.startswith("__") for name in solution.variables)

    @pytest.mark.parametrize(
        "problem",
        [problem for _, problem in hand_written_problems()],
        ids=[label for label, _ in hand_written_problems()],
    )
    def test_internal_variables_are_prefixed(self, problem):
        compiled = compile_problem(problem)
        for name in compiled.internal_variables:
            assert name.startswith("__int_") or name.startswith("__slack_")
        for name in model_variables(compiled):
            if name in compiled.internal_variables:
                continue
            assert not name.startswith("__")

    @pytest.mark.parametrize("index", range(0, len(RANDOM_PROBLEMS), 5))
    def test_internal_variables_are_prefixed_on_random_problems(self, index):
        compiled = compile_problem(RANDOM_PROBLEMS[index])
        for name in compiled.internal_variables:
            assert name.startswith("__int_") or name.startswith("__slack_")
