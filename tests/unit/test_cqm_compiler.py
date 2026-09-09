"""Unit tests for the CQM compiler (3a spec §15, §21.2, §26.1).

The compiler is checked structurally (variables, objective, one native
constraint per business constraint with the right sense / rhs / weight)
and semantically against ``dimod.ExactCQMSolver``:

* §21.2: a soft constraint's energy contribution is ``weight * v**2`` and
  equals the validator's ``weighted_penalty`` — the solver's notion of a
  preference and the ranking's are the same number.
* cross-validation: over *every* assignment, the CQM's ``is_feasible``
  agrees with ``validate_solution(...).feasible``. This is the only place
  the project reads ``is_feasible``; it exists to prove the compiler, the
  service never trusts it.
"""

import json
from pathlib import Path

import dimod
import pytest

from annealbridge.compiler import CQMCompiler
from annealbridge.exceptions import CompilationError
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import evaluate_objective
from annealbridge.penalty.strategy import compute_objective_scale
from annealbridge.validation import (
    validate_problem,
    validate_problem_full,
    validate_solution,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


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
            "name": "cqm compiler test problem",
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


def lin(variable: str, coefficient: float) -> dict:
    return {"variable": variable, "coefficient": coefficient}


def quad(variable1: str, variable2: str, coefficient: float) -> dict:
    return {"variable1": variable1, "variable2": variable2, "coefficient": coefficient}


def hard(constraint_id: str, operator: str, rhs: float, terms: list[dict]) -> dict:
    return {
        "id": constraint_id,
        "type": "hard",
        "terms": terms,
        "operator": operator,
        "rhs": rhs,
    }


def soft(
    constraint_id: str, operator: str, rhs: float, terms: list[dict], weight: float
) -> dict:
    return {
        "id": constraint_id,
        "type": "soft",
        "terms": terms,
        "operator": operator,
        "rhs": rhs,
        "weight": weight,
    }


def compile_problem(problem: OptimizationProblem):
    return CQMCompiler().compile(problem, None)


def enumerate_cqm(cqm: dimod.ConstrainedQuadraticModel):
    """Yield ``(sample_dict, energy, is_feasible)`` for every assignment."""
    sampleset = dimod.ExactCQMSolver().sample_cqm(cqm)
    names = list(sampleset.variables)
    for record in sampleset.record:
        sample = {name: int(value) for name, value in zip(names, record.sample)}
        yield sample, float(record.energy), bool(record.is_feasible)


class TestDeclaration:
    def test_model_type_and_penalty_use(self):
        compiler = CQMCompiler()
        assert compiler.model_type == "cqm"
        assert compiler.uses_hard_penalty is False

    def test_hard_penalty_is_rejected(self):
        with pytest.raises(CompilationError, match="hard_penalty"):
            CQMCompiler().compile(make_problem(), 5.0)


class TestVariables:
    def test_every_declared_variable_is_binary_in_problem_order(self):
        problem = make_problem(linear=[lin("a", 1)], variables=("b", "a", "c"))
        compiled = compile_problem(problem)
        cqm = compiled.model
        assert isinstance(cqm, dimod.ConstrainedQuadraticModel)
        assert list(cqm.variables) == ["b", "a", "c"]
        assert all(cqm.vartype(name) is dimod.BINARY for name in cqm.variables)
        assert compiled.model_type == "cqm"
        assert compiled.num_variables == len(problem.variables) == 3
        assert compiled.internal_variables == set()
        assert compiled.hard_penalty is None


class TestObjective:
    def test_minimize_linear_quadratic_and_constant(self):
        problem = make_problem(
            linear=[lin("x1", 2), lin("x2", -3)],
            quadratic=[quad("x1", "x2", 4)],
            constant=5,
        )
        objective = compile_problem(problem).model.objective
        assert objective.get_linear("x1") == 2.0
        assert objective.get_linear("x2") == -3.0
        assert objective.get_linear("x3") == 0.0
        assert objective.get_quadratic("x1", "x2") == 4.0
        assert objective.offset == 5.0

    def test_maximize_negates_all_coefficients_and_constant(self):
        problem = make_problem(
            direction="maximize",
            linear=[lin("x1", 2)],
            quadratic=[quad("x1", "x2", 4)],
            constant=5,
        )
        objective = compile_problem(problem).model.objective
        assert objective.get_linear("x1") == -2.0
        assert objective.get_quadratic("x1", "x2") == -4.0
        assert objective.offset == -5.0

    def test_objective_scale_uses_shared_formula(self):
        problem = make_problem(
            linear=[lin("x1", 2), lin("x2", -3)],
            quadratic=[quad("x1", "x2", 4)],
        )
        assert compile_problem(problem).objective_scale == 9.0
        assert compile_problem(make_problem()).objective_scale == 1.0
        assert compute_objective_scale(problem.objective) == 9.0


class TestConstraints:
    def test_hard_constraint_is_native_with_no_weight(self):
        problem = make_problem(
            constraints=[hard("cap", "<=", 4, [lin("x1", 2), lin("x2", 3)])]
        )
        compiled = compile_problem(problem)
        cqm = compiled.model
        assert list(cqm.constraint_labels) == ["cap"]
        constraint = cqm.constraints["cap"]
        assert dict(constraint.lhs.linear) == {"x1": 2.0, "x2": 3.0}
        assert constraint.lhs.offset == 0.0
        assert constraint.sense is dimod.sym.Sense.Le
        assert constraint.rhs == 4.0
        # dimod keeps ``weight=None`` (must be satisfied) as "not soft":
        # ``_soft`` is its registry of soft constraints (weight + penalty).
        assert "cap" not in cqm._soft
        assert cqm.num_soft_constraints() == 0

        (trace,) = compiled.constraint_trace
        assert trace.constraint_id == "cap"
        assert trace.constraint_type == "hard"
        assert trace.operator == "<="
        assert trace.penalty is None
        assert trace.native is True
        assert trace.slack_range is None
        assert trace.generated_variables == []
        assert trace.redundant is False
        assert trace.compiler == "CQMCompiler"
        assert compiled.internal_variables == set()
        assert compiled.num_variables == 3

    def test_soft_constraint_carries_weight_and_quadratic_penalty(self):
        problem = make_problem(
            constraints=[soft("cover", ">=", 1, [lin("x1", 1), lin("x2", 1)], 2.5)]
        )
        compiled = compile_problem(problem)
        cqm = compiled.model
        assert list(cqm.constraint_labels) == ["cover"]
        constraint = cqm.constraints["cover"]
        assert dict(constraint.lhs.linear) == {"x1": 1.0, "x2": 1.0}
        assert constraint.sense is dimod.sym.Sense.Ge
        assert constraint.rhs == 1.0
        assert cqm.num_soft_constraints() == 1
        soft_view = cqm._soft["cover"]
        assert soft_view.weight == 2.5 == problem.constraints[0].weight
        assert soft_view.penalty == "quadratic"

        (trace,) = compiled.constraint_trace
        assert trace.constraint_type == "soft"
        assert trace.operator == ">="
        assert trace.penalty == 2.5
        assert trace.native is True
        assert trace.slack_range is None
        assert trace.generated_variables == []

    def test_labels_are_constraint_ids_in_problem_order(self):
        problem = make_problem(
            constraints=[
                hard("zeta", "==", 1, [lin("x1", 1), lin("x2", 1)]),
                soft("alpha", "<=", 1, [lin("x3", 1)], 1.0),
                hard("mid", ">=", 0, [lin("x1", 1)]),
            ]
        )
        cqm = compile_problem(problem).model
        assert list(cqm.constraint_labels) == ["zeta", "alpha", "mid"]
        assert cqm.constraints["zeta"].sense is dimod.sym.Sense.Eq

    def test_duplicate_terms_accumulate_and_zero_coefficients_are_dropped(self):
        problem = make_problem(
            constraints=[
                hard(
                    "acc",
                    "<=",
                    3,
                    [lin("x1", 2), lin("x1", 3), lin("x2", 1), lin("x2", -1), lin("x3", 4)],
                )
            ]
        )
        cqm = compile_problem(problem).model
        assert dict(cqm.constraints["acc"].lhs.linear) == {"x1": 5.0, "x3": 4.0}

    def test_constant_constraint_that_holds_is_redundant_and_not_added(self):
        problem = make_problem(
            linear=[lin("x1", 1)],
            constraints=[
                hard("noop", "<=", 0, [lin("x1", 1), lin("x1", -1)]),
                soft("noop_soft", ">=", -1, [lin("x2", 2), lin("x2", -2)], 3.0),
                hard("noop_eq", "==", 0, [lin("x3", 1), lin("x3", -1)]),
            ],
        )
        compiled = compile_problem(problem)
        cqm = compiled.model
        assert list(cqm.constraint_labels) == []
        assert [trace.redundant for trace in compiled.constraint_trace] == [True] * 3
        assert [trace.native for trace in compiled.constraint_trace] == [True] * 3
        assert [trace.penalty for trace in compiled.constraint_trace] == [None, 3.0, None]
        assert compiled.num_variables == 3

    @pytest.mark.parametrize(
        ("operator", "rhs"),
        [("<=", -1), (">=", 1), ("==", 1)],
    )
    def test_constant_constraint_that_fails_raises(self, operator, rhs):
        problem = make_problem(
            constraints=[hard("bad", operator, rhs, [lin("x1", 1), lin("x1", -1)])]
        )
        with pytest.raises(CompilationError, match="bad"):
            compile_problem(problem)

    # Review F-04 (2026-09-09): a *soft* constant constraint that fails is a
    # legal problem (its weight is always paid). The BQM path puts
    # ``w * rhs**2`` in the offset; the CQM path must do the same in the
    # objective instead of refusing with COMPILATION_FAILED.
    @pytest.mark.parametrize(
        ("operator", "rhs", "expected_slack_range"),
        [("<=", -3, 0), (">=", 4, 0), ("==", 5, None)],
    )
    def test_soft_constant_constraint_that_fails_is_a_constant_penalty(
        self, operator, rhs, expected_slack_range
    ):
        problem = make_problem(
            linear=[lin("x1", 1)],
            constraints=[soft("paid", operator, rhs, [lin("x2", 1), lin("x2", -1)], 2.0)],
        )
        compiled = compile_problem(problem)
        cqm = compiled.model
        assert list(cqm.constraint_labels) == []
        assert cqm.objective.offset == pytest.approx(2.0 * rhs * rhs)
        assert compiled.num_variables == 3
        assert compiled.internal_variables == set()
        (trace,) = compiled.constraint_trace
        assert trace.native is False
        assert trace.redundant is False
        assert trace.penalty == 2.0
        assert trace.slack_range == expected_slack_range
        assert trace.generated_variables == []

    def test_soft_constant_penalty_matches_the_validator_score_on_every_assignment(self):
        problem = make_problem(
            linear=[lin("x1", 1)],
            constraints=[soft("paid", "==", 5, [lin("x2", 1), lin("x2", -1)], 2.0)],
        )
        cqm = compile_problem(problem).model
        for sample, energy, _ in enumerate_cqm(cqm):
            expected = evaluate_objective(problem.objective, sample) + validate_solution(
                problem, sample
            ).soft_violation_score
            assert energy == pytest.approx(expected)
            assert validate_solution(problem, sample).soft_violation_score == 50.0

    def test_hard_constant_equality_within_tolerance_is_redundant(self):
        # The validator accepts ``0 == 1e-9`` under the §23.1 tolerance, so
        # the compiler must not refuse it with an exact comparison.
        problem = make_problem(
            constraints=[hard("tiny", "==", 1e-9, [lin("x1", 1), lin("x1", -1)])]
        )
        assert validate_problem(problem) == []
        compiled = compile_problem(problem)
        assert list(compiled.model.constraint_labels) == []
        assert compiled.constraint_trace[0].redundant is True

    @pytest.mark.parametrize("rhs", [0.0, 1e-9, 9e-9, 1.5e-8, 1e-7, 1.0])
    def test_hard_constant_verdict_agrees_with_the_validator(self, rhs):
        problem = make_problem(
            constraints=[hard("c", "==", rhs, [lin("x1", 1), lin("x1", -1)])]
        )
        rejected = any(error.code == "TRIVIALLY_INFEASIBLE" for error in validate_problem(problem))
        if rejected:
            with pytest.raises(CompilationError, match="c"):
                compile_problem(problem)
        else:
            assert compile_problem(problem).constraint_trace[0].redundant is True

    @pytest.mark.parametrize("rhs", [0.0, 1e-9, 9e-9, 1.5e-8, 1e-7, 1.0])
    def test_soft_constant_verdict_agrees_with_the_validator(self, rhs):
        problem = make_problem(
            constraints=[soft("c", "==", rhs, [lin("x1", 1), lin("x1", -1)], 3.0)]
        )
        warned = any(
            warning.code == "SOFT_ALWAYS_VIOLATED"
            for warning in validate_problem_full(problem).warnings
        )
        compiled = compile_problem(problem)
        (trace,) = compiled.constraint_trace
        assert trace.redundant is (not warned)
        assert compiled.model.objective.offset == pytest.approx(
            3.0 * rhs * rhs if warned else 0.0
        )


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


class TestDeterminism:
    def test_two_compilations_are_identical(self):
        problem = make_problem(
            direction="maximize",
            linear=[lin("x3", 3), lin("x1", 4), lin("x2", 5)],
            quadratic=[quad("x2", "x1", -2)],
            constant=1,
            constraints=[
                hard("pick", "==", 1, [lin("x2", 1), lin("x1", 1)]),
                hard("cap", "<=", 5, [lin("x3", 4), lin("x1", 2), lin("x2", 3)]),
                soft("cover", ">=", 1, [lin("x2", 1), lin("x3", 1)], 2.5),
            ],
        )
        first = compile_problem(problem)
        second = compile_problem(problem)
        cqm_a, cqm_b = first.model, second.model

        assert list(cqm_a.variables) == list(cqm_b.variables)
        assert list(cqm_a.constraint_labels) == list(cqm_b.constraint_labels)
        for label in cqm_a.constraint_labels:
            lhs_a = cqm_a.constraints[label].lhs
            lhs_b = cqm_b.constraints[label].lhs
            assert list(lhs_a.linear.items()) == list(lhs_b.linear.items())
            assert cqm_a.constraints[label].sense is cqm_b.constraints[label].sense
            assert cqm_a.constraints[label].rhs == cqm_b.constraints[label].rhs
        assert list(cqm_a.objective.linear.items()) == list(cqm_b.objective.linear.items())
        assert dict(cqm_a.objective.quadratic) == dict(cqm_b.objective.quadratic)
        assert cqm_a.objective.offset == cqm_b.objective.offset
        assert first.objective_scale == second.objective_scale
        assert first.constraint_trace == second.constraint_trace


class TestSoftWeightSemantics:
    """3a spec §21.2: CQM soft energy == validator ``weighted_penalty``."""

    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_soft_energy_is_weight_times_violation_squared(self, direction):
        # objective: 2 x1 - 5 x2 + x3; soft: x1 + x2 + x3 == 0 (weight 10).
        # Sample x1 = x2 = 1, x3 = 0 violates by v = 2 (v = 1 would not
        # separate a linear from a quadratic penalty).
        weight = 10.0
        problem = make_problem(
            direction=direction,
            linear=[lin("x1", 2), lin("x2", -5), lin("x3", 1)],
            constraints=[
                soft("s", "==", 0, [lin("x1", 1), lin("x2", 1), lin("x3", 1)], weight)
            ],
        )
        cqm = compile_problem(problem).model
        sample = {"x1": 1, "x2": 1, "x3": 0}
        violation = 2.0
        sign = -1.0 if direction == "maximize" else 1.0

        assert cqm.violations(sample)["s"] == violation

        objective_value = evaluate_objective(problem.objective, sample)
        assert objective_value == -3.0
        validation = validate_solution(problem, sample)
        (evaluation,) = validation.evaluations
        assert evaluation.violation_amount == violation
        assert evaluation.weighted_penalty == weight * violation * violation
        assert validation.feasible is True  # soft violations never break feasibility

        energies = {
            tuple(found[name] for name in ("x1", "x2", "x3")): energy
            for found, energy, _feasible in enumerate_cqm(cqm)
        }
        energy = energies[(1, 1, 0)]
        assert energy == pytest.approx(sign * objective_value + weight * violation**2)
        assert energy == pytest.approx(sign * objective_value + evaluation.weighted_penalty)
        # And the whole landscape follows the same law, not just one point.
        for bits, energy in energies.items():
            found = dict(zip(("x1", "x2", "x3"), bits))
            expected = sign * evaluate_objective(problem.objective, found)
            expected += validate_solution(problem, found).soft_violation_score
            assert energy == pytest.approx(expected)


def _assert_feasibility_agrees(problem: OptimizationProblem) -> None:
    """Every assignment: CQM ``is_feasible`` == independent validator verdict."""
    cqm = compile_problem(problem).model
    names = [variable.name for variable in problem.variables]
    seen_feasible = seen_infeasible = 0
    total = 0
    for sample, _energy, is_feasible in enumerate_cqm(cqm):
        assert set(sample) == set(names)
        verdict = validate_solution(problem, sample).feasible
        assert is_feasible == verdict, sample
        total += 1
        seen_feasible += verdict
        seen_infeasible += not verdict
    assert total == 2 ** len(names)
    assert seen_feasible > 0 and seen_infeasible > 0


class TestCrossValidationWithExactCQMSolver:
    def test_knapsack_example(self):
        raw = json.loads((EXAMPLES / "knapsack.json").read_text(encoding="utf-8"))
        problem = OptimizationProblem.model_validate(raw)
        _assert_feasibility_agrees(problem)

    def test_mixed_ge_eq_hard_constraints_with_a_soft_one(self):
        # hard: x1 + x2 == 1; hard: x2 + x3 + x4 >= 2; soft: x1 + x4 <= 1
        # (weight 2). The soft constraint must not influence feasibility.
        problem = make_problem(
            direction="maximize",
            linear=[lin("x1", 3), lin("x2", 1), lin("x3", 2), lin("x4", 4)],
            quadratic=[quad("x3", "x4", -1)],
            variables=("x1", "x2", "x3", "x4"),
            constraints=[
                hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)]),
                hard("cover", ">=", 2, [lin("x2", 1), lin("x3", 1), lin("x4", 1)]),
                soft("apart", "<=", 1, [lin("x1", 1), lin("x4", 1)], 2.0),
            ],
        )
        _assert_feasibility_agrees(problem)
        # Soft violations are visible to the sampler but do not make a
        # sample infeasible: x1 = x4 = 1, x3 = 1, x2 = 0 is feasible.
        cqm = compile_problem(problem).model
        assert cqm.violations({"x1": 1, "x2": 0, "x3": 1, "x4": 1})["apart"] == 1.0
        assert validate_solution(problem, {"x1": 1, "x2": 0, "x3": 1, "x4": 1}).feasible

    def test_lowest_energy_feasible_sample_is_the_business_optimum(self):
        # Knapsack: maximize 3 x1 + 4 x2 + 5 x3 s.t. 2 x1 + 3 x2 + 4 x3 <= 5.
        # Optimum x1 = x2 = 1 (value 7); with the negation convention a
        # feasible sample's energy is exactly -objective.
        problem = make_problem(
            direction="maximize",
            linear=[lin("x1", 3), lin("x2", 4), lin("x3", 5)],
            constraints=[
                hard("cap", "<=", 5, [lin("x1", 2), lin("x2", 3), lin("x3", 4)])
            ],
        )
        compiled = compile_problem(problem)
        assert compiled.num_variables == 3  # no slack bits on this path
        feasible = [
            (energy, sample)
            for sample, energy, _flag in enumerate_cqm(compiled.model)
            if validate_solution(problem, sample).feasible
        ]
        energy, best = min(feasible, key=lambda item: item[0])
        assert energy == pytest.approx(-7.0)
        assert best == {"x1": 1, "x2": 1, "x3": 0}
