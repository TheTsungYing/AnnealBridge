"""Unit tests for the BQM compiler (spec §15, §33)."""

import itertools

import dimod
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.penalty.strategy import compute_objective_scale
from tests.builders import hard, lin, quad, soft


def make_problem(
    *,
    direction: str = "minimize",
    linear: list[dict] | None = None,
    quadratic: list[dict] | None = None,
    constant: float = 0,
    constraints: list[dict] | None = None,
    variables: tuple[str, ...] = ("x1", "x2", "x3"),
) -> OptimizationProblem:
    return OptimizationProblem.model_validate(
        {
            "name": "compiler test problem",
            "variables": [{"name": name} for name in variables],
            "objective": {
                "direction": direction,
                "linear_terms": linear or [],
                "quadratic_terms": quadratic or [],
                "constant": constant,
            },
            "constraints": constraints or [],
        }
    )


def compile_problem(problem: OptimizationProblem, hard_penalty: float = 10.0):
    return BQMCompiler().compile(problem, hard_penalty)


class TestObjective:
    def test_minimize_linear_and_constant(self):
        problem = make_problem(linear=[lin("x1", 2), lin("x2", -3)], constant=5)
        bqm = compile_problem(problem).model
        assert bqm.get_linear("x1") == 2.0
        assert bqm.get_linear("x2") == -3.0
        assert bqm.get_linear("x3") == 0.0
        assert bqm.offset == 5.0

    def test_maximize_negates_all_coefficients_and_constant(self):
        problem = make_problem(
            direction="maximize",
            linear=[lin("x1", 2)],
            quadratic=[quad("x1", "x2", 4)],
            constant=5,
        )
        bqm = compile_problem(problem).model
        assert bqm.get_linear("x1") == -2.0
        assert bqm.get_quadratic("x1", "x2") == -4.0
        assert bqm.offset == -5.0

    def test_duplicate_linear_terms_accumulate(self):
        problem = make_problem(linear=[lin("x1", 2), lin("x1", 3)])
        bqm = compile_problem(problem).model
        assert bqm.get_linear("x1") == 5.0

    def test_duplicate_quadratic_terms_accumulate_across_orderings(self):
        problem = make_problem(
            quadratic=[quad("x1", "x2", 3), quad("x2", "x1", 4)]
        )
        bqm = compile_problem(problem).model
        assert bqm.get_quadratic("x1", "x2") == 7.0

    def test_unreferenced_declared_variable_is_in_bqm(self):
        problem = make_problem(linear=[lin("x1", 1)])
        bqm = compile_problem(problem).model
        assert "x3" in bqm.variables
        assert bqm.get_linear("x3") == 0.0


class TestEqualityConstraints:
    def test_hand_computed_expansion(self):
        # hard_penalty * (x1 + x2 - 1)^2 with binary x:
        # linear += lam*(1 - 2), quadratic += 2*lam, offset += lam
        problem = make_problem(
            constraints=[hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)])]
        )
        compiled = compile_problem(problem, hard_penalty=3.0)
        bqm = compiled.model
        assert bqm.get_linear("x1") == -3.0
        assert bqm.get_linear("x2") == -3.0
        assert bqm.get_quadratic("x1", "x2") == 6.0
        assert bqm.offset == 3.0

    def test_trace_has_no_slack(self):
        problem = make_problem(
            constraints=[hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)])]
        )
        compiled = compile_problem(problem)
        (trace,) = compiled.constraint_trace
        assert trace.slack_range is None
        assert trace.generated_variables == []
        assert trace.redundant is False
        assert trace.compiler == "BQMCompiler"
        assert compiled.internal_variables == set()
        assert compiled.num_variables == 3
        # x1 + x2 == 1 squared couples the pair once.
        assert compiled.num_interactions == compiled.model.num_interactions == 1


class TestInequalityConstraints:
    def test_less_equal_hard_generates_slack(self):
        # 2*x1 + 3*x2 <= 4 -> S = 4 -> bits [1, 2, 1]
        problem = make_problem(
            constraints=[hard("cap", "<=", 4, [lin("x1", 2), lin("x2", 3)])]
        )
        compiled = compile_problem(problem, hard_penalty=7.0)
        expected_slacks = {"__slack_cap_0", "__slack_cap_1", "__slack_cap_2"}
        assert compiled.internal_variables == expected_slacks
        assert compiled.num_variables == 6
        assert expected_slacks <= set(compiled.model.variables)
        (trace,) = compiled.constraint_trace
        assert trace.penalty == 7.0
        assert trace.slack_range == 4
        assert sorted(trace.generated_variables) == sorted(expected_slacks)

    def test_greater_equal_soft_trace_keeps_original_operator(self):
        problem = make_problem(
            constraints=[soft("cover", ">=", 1, [lin("x1", 1), lin("x2", 1)], 2.0)]
        )
        compiled = compile_problem(problem)
        (trace,) = compiled.constraint_trace
        assert trace.operator == ">="
        assert trace.constraint_type == "soft"
        assert trace.penalty == 2.0
        assert trace.slack_range == 1
        assert trace.generated_variables == ["__slack_cover_0"]

    def test_redundant_constraint_adds_nothing(self):
        base = make_problem(linear=[lin("x1", 2), lin("x2", 1)])
        with_redundant = make_problem(
            linear=[lin("x1", 2), lin("x2", 1)],
            constraints=[hard("loose", "<=", 5, [lin("x1", 1), lin("x2", 1)])],
        )
        bare = compile_problem(base).model
        compiled = compile_problem(with_redundant)
        assert dict(compiled.model.linear) == dict(bare.linear)
        assert dict(compiled.model.quadratic) == dict(bare.quadratic)
        assert compiled.model.offset == bare.offset
        (trace,) = compiled.constraint_trace
        assert trace.redundant is True
        assert trace.slack_range is None
        assert trace.generated_variables == []
        assert compiled.internal_variables == set()


class TestPenaltySources:
    def test_hard_and_soft_lambdas_never_mix(self):
        # hard (x1 == 1) uses hard_penalty, soft (x2 == 1) uses its weight;
        # expansion of lam*(x - 1)^2 puts -lam on the linear bias.
        problem = make_problem(
            constraints=[
                hard("h", "==", 1, [lin("x1", 1)]),
                soft("s", "==", 1, [lin("x2", 1)], 5.0),
            ]
        )
        compiled = compile_problem(problem, hard_penalty=3.0)
        bqm = compiled.model
        assert bqm.get_linear("x1") == -3.0
        assert bqm.get_linear("x2") == -5.0
        assert bqm.offset == 8.0
        hard_trace, soft_trace = compiled.constraint_trace
        assert hard_trace.penalty == 3.0
        assert soft_trace.penalty == 5.0


class TestEnergyEquivalence:
    def test_energy_matches_sign_objective_plus_penalties(self):
        # maximize 3*x1 + 4*x2 + 5*x3 - 2*x1*x2 + 1
        # s.t. hard: x1 + x2 == 1; hard: 2*x1 + 3*x2 + 4*x3 <= 5;
        #      soft: x2 + x3 >= 1 (weight 2.5)
        hard_penalty = 7.0
        weight = 2.5
        problem = make_problem(
            direction="maximize",
            linear=[lin("x1", 3), lin("x2", 4), lin("x3", 5)],
            quadratic=[quad("x1", "x2", -2)],
            constant=1,
            constraints=[
                hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)]),
                hard("cap", "<=", 5, [lin("x1", 2), lin("x2", 3), lin("x3", 4)]),
                soft("cover", ">=", 1, [lin("x2", 1), lin("x3", 1)], weight),
            ],
        )
        compiled = compile_problem(problem, hard_penalty=hard_penalty)
        bqm = compiled.model
        cap_slacks = ["__slack_cap_0", "__slack_cap_1", "__slack_cap_2"]
        cap_coeffs = [1, 2, 2]  # S = 5
        assert compiled.internal_variables == set(cap_slacks) | {"__slack_cover_0"}

        names = ["x1", "x2", "x3", *cap_slacks, "__slack_cover_0"]
        for bits in itertools.product((0, 1), repeat=len(names)):
            sample = dict(zip(names, bits))
            x1, x2, x3 = sample["x1"], sample["x2"], sample["x3"]
            objective = 3 * x1 + 4 * x2 + 5 * x3 - 2 * x1 * x2 + 1
            s_cap = sum(
                coeff * sample[name] for coeff, name in zip(cap_coeffs, cap_slacks)
            )
            s_cover = sample["__slack_cover_0"]
            expected = (
                -objective
                + hard_penalty * (x1 + x2 - 1) ** 2
                + hard_penalty * (2 * x1 + 3 * x2 + 4 * x3 + s_cap - 5) ** 2
                + weight * (x2 + x3 - s_cover - 1) ** 2
            )
            assert bqm.energy(sample) == pytest.approx(expected)


class TestPurity:
    def test_problem_not_mutated(self):
        problem = make_problem(
            direction="maximize",
            linear=[lin("x1", 3), lin("x2", 4)],
            constraints=[
                hard("cap", "<=", 4, [lin("x1", 2), lin("x2", 3)]),
                soft("s", "==", 1, [lin("x3", 1)], 2.0),
            ],
        )
        snapshot = problem.model_dump()
        compile_problem(problem)
        assert problem.model_dump() == snapshot


class TestObjectiveScale:
    def test_formula(self):
        problem = make_problem(
            linear=[lin("x1", 2), lin("x2", -3)],
            quadratic=[quad("x1", "x2", 4)],
        )
        assert compute_objective_scale(problem.objective) == 9.0
        assert compile_problem(problem).objective_scale == 9.0

    def test_floor_of_one_for_empty_objective(self):
        problem = make_problem()
        assert compute_objective_scale(problem.objective) == 1.0
        assert compile_problem(problem).objective_scale == 1.0


class TestEndToEndWithExactSolver:
    def test_lowest_energy_sample_is_feasible_optimum(self):
        # Knapsack: maximize 3*x1 + 4*x2 + 5*x3 s.t. 2*x1 + 3*x2 + 4*x3 <= 5.
        # Feasible optimum is x1 = x2 = 1 (value 7, weight 5); with the
        # constant-negation convention the energy of a feasible sample is
        # exactly -objective, so the ground state has energy -7.0.
        problem = make_problem(
            direction="maximize",
            linear=[lin("x1", 3), lin("x2", 4), lin("x3", 5)],
            constraints=[
                hard("cap", "<=", 5, [lin("x1", 2), lin("x2", 3), lin("x3", 4)])
            ],
        )
        compiled = compile_problem(problem, hard_penalty=10.0)
        assert compiled.num_variables == 6  # 3 business + 3 slack bits

        best = dimod.ExactSolver().sample(compiled.model).first
        assert best.energy == pytest.approx(-7.0)
        assert best.sample["x1"] == 1
        assert best.sample["x2"] == 1
        assert best.sample["x3"] == 0
        assert all(best.sample[name] == 0 for name in compiled.internal_variables)
