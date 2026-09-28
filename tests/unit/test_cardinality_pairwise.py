"""The pairwise encoding of a declared hard at-most-one (schema 1.2 spec §7.1).

Spec 2026-09-25 §13 items 4 and 5. A hard ``"<=", 1`` cardinality constraint
over two or more variables compiles on the BQM path to
``λ · Σ_{i<j} x_i x_j``: no slack variable, no linear part, no constant.
Evidence, each piece independent of ``test_bqm_prepared._reference_compile``
(which stays the frozen baseline for the other encodings, spec §8.4):

* the shape: every pair's coupling is exactly λ, the trace has no generated
  variable, and the coupling follows λ from one compile to the next;
* ``PreparedBQM.compile`` equals a fresh compile bit for bit;
* the estimates agree with the compiled model;
* the identity ``E_S − E_P = λ Σ [max(0, k−1)² − C(k, 2)]`` between the slack
  encoding (minimised over its slack bit) and the pairwise one, enumerated;
* dominance: the compiled model's lowest-energy states decode to feasible
  assignments with the true best ranking, found by enumeration;
* a model whose pairwise couplings overflow is refused as PENALTY_OVERFLOW;
* the compiler's own guard against a problem that skipped the validator.
"""

import itertools
import math
from math import comb

import dimod
import pytest

from annealbridge.compiler import BQMCompiler
from annealbridge.exceptions import CompilationError, NonFiniteModelError
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.candidates import evaluate_objective
from annealbridge.penalty import ScaledPenaltyStrategy
from annealbridge.solvers import RawSolverResult
from annealbridge.solvers.base import sampleset_to_arrays
from annealbridge.validation import (
    validate_problem,
    validate_problem_full,
    validate_solution,
)
from annealbridge.validation.estimates import (
    estimate_compiled_variables,
    estimate_encoded_interactions,
    uses_pairwise_penalty,
)

# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def lin(variable: str, coefficient: float) -> dict:
    return {"variable": variable, "coefficient": coefficient}


def quad(variable1: str, variable2: str, coefficient: float) -> dict:
    return {"variable1": variable1, "variable2": variable2, "coefficient": coefficient}


def at_most_one(constraint_id: str, variables: list[str]) -> dict:
    return {
        "id": constraint_id,
        "type": "hard",
        "variables": variables,
        "operator": "<=",
        "rhs": 1,
    }


def binaries(count: int, prefix: str = "x") -> list[str]:
    return [f"{prefix}{index}" for index in range(count)]


def problem(
    *,
    variables: list,
    linear: list[dict] | None = None,
    quadratic: list[dict] | None = None,
    direction: str = "minimize",
    constraints: list[dict] | None = None,
    cardinality: list[dict] | None = None,
    solver: dict | None = None,
) -> OptimizationProblem:
    payload = {
        "version": "1.2",
        "name": "pairwise encoding",
        "variables": [
            entry if isinstance(entry, dict) else {"name": entry} for entry in variables
        ],
        "objective": {
            "direction": direction,
            "linear_terms": linear or [],
            "quadratic_terms": quadratic or [],
        },
        "constraints": constraints or [],
        "cardinality_constraints": cardinality or [],
    }
    if solver is not None:
        payload["solver"] = solver
    return OptimizationProblem.model_validate(payload)


def linear_rewrite(source: OptimizationProblem) -> OptimizationProblem:
    """``source`` with every cardinality constraint moved to ``constraints``.

    Coefficient 1 per counted variable, appended in declaration order: the
    slack encoding of the same constraints (spec §7.1, "與 slack 編碼比較").
    """
    data = source.model_dump()
    for entry in data.pop("cardinality_constraints", []):
        entry["terms"] = [lin(name, 1) for name in entry.pop("variables")]
        data["constraints"].append(entry)
    return OptimizationProblem.model_validate(data)


def bqm_shape(bqm: dimod.BinaryQuadraticModel) -> dict:
    """Variable order, biases in iteration order and offset: bit-for-bit."""
    return {
        "variables": [str(v) for v in bqm.variables],
        "linear": [(str(v), float(bqm.get_linear(v))) for v in bqm.variables],
        "quadratic": [(str(u), str(v), float(bias)) for u, v, bias in bqm.iter_quadratic()],
        "offset": float(bqm.offset),
    }


def compiled_shape(compiled) -> dict:
    return {
        **bqm_shape(compiled.model),
        "internal_variables": sorted(compiled.internal_variables),
        "trace": [trace.model_dump() for trace in compiled.constraint_trace],
        "num_variables": compiled.num_variables,
        "num_interactions": compiled.num_interactions,
        "hard_penalty": compiled.hard_penalty,
        "integer_encodings": {
            name: encoding.model_dump() for name, encoding in compiled.integer_encodings.items()
        },
    }


def ranking(problem_: OptimizationProblem, sample: dict[str, int]) -> float:
    objective = evaluate_objective(problem_.objective, sample)
    soft = validate_solution(problem_, sample).soft_violation_score
    if problem_.objective.direction == "minimize":
        return objective + soft
    return objective - soft


def business_assignments(problem_: OptimizationProblem):
    names = [variable.name for variable in problem_.variables]
    domains = [range(v.bounds()[0], v.bounds()[1] + 1) for v in problem_.variables]
    for values in itertools.product(*domains):
        yield dict(zip(names, values))


# --------------------------------------------------------------------------
# Shape of the encoding
# --------------------------------------------------------------------------


class TestPairwiseShape:
    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_no_slack_and_coupling_exactly_lambda(self, count):
        names = binaries(count)
        p = problem(variables=names, cardinality=[at_most_one("amo", names)])
        assert validate_problem(p) == []
        (lowered,) = p.all_constraints()
        assert uses_pairwise_penalty(lowered)

        lam = 7.25
        compiled = BQMCompiler().compile(p, lam)
        bqm = compiled.model

        assert [str(v) for v in bqm.variables] == names
        assert compiled.internal_variables == set()
        assert all(bqm.get_linear(name) == 0.0 for name in names)
        assert bqm.offset == 0.0
        expected_pairs = {frozenset(pair) for pair in itertools.combinations(names, 2)}
        assert {frozenset((u, v)) for u, v, _ in bqm.iter_quadratic()} == expected_pairs
        assert all(bias == lam for _, _, bias in bqm.iter_quadratic())
        assert bqm.num_interactions == comb(count, 2)

        (trace,) = compiled.constraint_trace
        assert trace.constraint_id == "amo"
        assert trace.constraint_type == "hard"
        assert trace.operator == "<="
        assert trace.generated_variables == []
        assert trace.slack_range is None
        assert trace.redundant is False
        assert trace.penalty == lam
        assert trace.native is False

    def test_couplings_accumulate_with_the_objective_and_other_constraints(self):
        names = binaries(3)
        p = problem(
            variables=names,
            quadratic=[quad("x0", "x1", 2)],
            cardinality=[at_most_one("first", names), at_most_one("second", ["x0", "x1"])],
        )
        bqm = BQMCompiler().compile(p, 3.0).model
        assert bqm.get_quadratic("x0", "x1") == 2.0 + 3.0 + 3.0
        assert bqm.get_quadratic("x0", "x2") == 3.0
        assert bqm.get_quadratic("x1", "x2") == 3.0

    def test_a_single_variable_at_most_one_adds_nothing(self):
        p = problem(variables=["x0"], linear=[lin("x0", 2)], cardinality=[at_most_one("one", ["x0"])])
        (lowered,) = p.all_constraints()
        assert not uses_pairwise_penalty(lowered)
        compiled = BQMCompiler().compile(p, 5.0)
        assert [str(v) for v in compiled.model.variables] == ["x0"]
        assert compiled.model.get_linear("x0") == 2.0
        assert compiled.model.num_interactions == 0
        assert compiled.model.offset == 0.0
        (trace,) = compiled.constraint_trace
        assert trace.redundant is True
        assert trace.generated_variables == []

    @pytest.mark.parametrize(
        "entry",
        [
            {"type": "soft", "weight": 2, "operator": "<=", "rhs": 1},
            {"type": "hard", "operator": "<=", "rhs": 2},
            {"type": "hard", "operator": "==", "rhs": 1},
            {"type": "hard", "operator": ">=", "rhs": 1},
        ],
        ids=["soft", "rhs_2", "equality", "at_least"],
    )
    def test_other_shapes_are_not_pairwise(self, entry):
        p = problem(
            variables=binaries(3),
            cardinality=[{"id": "c", "variables": binaries(3), **entry}],
        )
        (lowered,) = p.all_constraints()
        assert not uses_pairwise_penalty(lowered)

    def test_a_linear_at_most_one_is_never_pairwise(self):
        """Only the declaration takes the new encoding (1.0 compiles as before)."""
        names = binaries(3)
        p = problem(
            variables=names,
            constraints=[
                {
                    "id": "amo",
                    "type": "hard",
                    "terms": [lin(name, 1) for name in names],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
        )
        (constraint,) = p.all_constraints()
        assert not uses_pairwise_penalty(constraint)
        compiled = BQMCompiler().compile(p, 2.0)
        assert compiled.constraint_trace[0].generated_variables == ["__slack_amo_0"]


# --------------------------------------------------------------------------
# Retry cache and fresh compile
# --------------------------------------------------------------------------


def mixed_problem(direction: str = "minimize") -> OptimizationProblem:
    """Pairwise constraints next to slack, integer and soft constraints."""
    return problem(
        direction=direction,
        variables=[
            *binaries(5),
            {"name": "k", "type": "integer", "lower_bound": 0, "upper_bound": 3},
        ],
        linear=[lin("x0", -3), lin("x1", -2), lin("x2", -4), lin("x3", 1), lin("x4", -1), lin("k", -2)],
        quadratic=[quad("x0", "x2", 1.5), quad("k", "x4", -1)],
        constraints=[
            {
                "id": "budget",
                "type": "hard",
                "terms": [lin("x0", 2), lin("x3", 1), lin("k", 1)],
                "operator": "<=",
                "rhs": 3,
            },
            {
                "id": "soft_lin",
                "type": "soft",
                "weight": 1.5,
                "terms": [lin("x1", 1), lin("k", 1)],
                "operator": ">=",
                "rhs": 2,
            },
        ],
        cardinality=[
            at_most_one("amo_a", ["x0", "x1", "x2"]),
            at_most_one("amo_b", ["x2", "x3", "x4"]),
            {"id": "one_hot", "type": "hard", "variables": ["x3", "x4"], "operator": "==", "rhs": 1},
            {
                "id": "soft_card",
                "type": "soft",
                "weight": 2,
                "variables": ["x0", "x1", "x4"],
                "operator": ">=",
                "rhs": 2,
            },
        ],
    )


class TestPreparedAndFreshCompile:
    def test_coupling_doubles_with_the_penalty(self):
        names = binaries(4)
        p = problem(variables=names, cardinality=[at_most_one("amo", names)])
        prepared = BQMCompiler().prepare(p)
        lam = 3.5
        first = prepared.compile(lam).model
        second = prepared.compile(2 * lam).model
        assert {bias for _, _, bias in first.iter_quadratic()} == {lam}
        assert {bias for _, _, bias in second.iter_quadratic()} == {2 * lam}
        # Replaying at the first penalty again is unaffected by the second.
        assert bqm_shape(prepared.compile(lam).model) == bqm_shape(first)

    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_prepared_compile_equals_fresh_compile(self, direction):
        p = mixed_problem(direction)
        lam = ScaledPenaltyStrategy().initial_penalty(p)
        prepared = BQMCompiler().prepare(p)
        for penalty in (lam, 2 * lam, 4 * lam):
            assert compiled_shape(prepared.compile(penalty)) == compiled_shape(
                BQMCompiler().compile(p, penalty)
            )

    def test_mixed_problem_couplings_are_lambda_on_pure_pairs(self):
        """A pair only the pairwise constraint couples carries exactly λ."""
        p = mixed_problem()
        compiled = BQMCompiler().compile(p, 10.0)
        # x1-x2 appear together only in amo_a.
        assert compiled.model.get_quadratic("x1", "x2") == 10.0


ESTIMATE_PROBLEMS = {
    "pairwise_only": lambda: problem(
        variables=binaries(6),
        cardinality=[at_most_one("a", binaries(6)[:4]), at_most_one("b", binaries(6)[2:])],
    ),
    "mixed": mixed_problem,
    "with_linear_inequalities": lambda: problem(
        variables=[*binaries(4), {"name": "m", "type": "integer", "lower_bound": -2, "upper_bound": 5}],
        linear=[lin("m", 1)],
        constraints=[
            {"id": "c1", "type": "hard", "terms": [lin("x0", 3), lin("m", 1)], "operator": ">=", "rhs": 1},
            {"id": "c2", "type": "soft", "weight": 1, "terms": [lin("x1", 1), lin("m", 2)], "operator": "<=", "rhs": 4},
        ],
        cardinality=[
            at_most_one("amo", binaries(4)),
            {"id": "le2", "type": "hard", "variables": binaries(4), "operator": "<=", "rhs": 2},
            {"id": "ge1", "type": "soft", "weight": 3, "variables": ["x0", "x3"], "operator": ">=", "rhs": 1},
            {"id": "single", "type": "hard", "variables": ["x2"], "operator": "<=", "rhs": 1},
        ],
    ),
}


class TestEstimates:
    @pytest.mark.parametrize("name", sorted(ESTIMATE_PROBLEMS))
    def test_estimates_match_the_compiled_model(self, name):
        p = ESTIMATE_PROBLEMS[name]()
        assert validate_problem(p) == []
        compiled = BQMCompiler().compile(p, ScaledPenaltyStrategy().initial_penalty(p))
        assert estimate_compiled_variables(p) == compiled.num_variables
        assert estimate_encoded_interactions(p) >= compiled.num_interactions

    def test_pairwise_only_interactions_are_exact(self):
        """C(n, 2) per pairwise constraint, the shared pair counted once by the model."""
        p = ESTIMATE_PROBLEMS["pairwise_only"]()
        compiled = BQMCompiler().compile(p, 1.0)
        assert estimate_encoded_interactions(p) == comb(4, 2) + comb(4, 2)
        # x2-x3 is in both constraints: one model interaction.
        assert compiled.num_interactions == comb(4, 2) + comb(4, 2) - 1


# --------------------------------------------------------------------------
# The identity between the slack and the pairwise encodings
# --------------------------------------------------------------------------


IDENTITY_PROBLEMS = {
    "one_constraint": lambda: problem(
        variables=binaries(5),
        linear=[lin("x0", 2), lin("x1", -1), lin("x3", 0.5)],
        cardinality=[at_most_one("amo", binaries(5))],
    ),
    "overlapping": lambda: problem(
        variables=binaries(8),
        linear=[lin(name, (-1) ** index * (index + 1)) for index, name in enumerate(binaries(8))],
        quadratic=[quad("x0", "x5", 3), quad("x2", "x7", -2)],
        cardinality=[
            at_most_one("a", ["x0", "x1", "x2", "x3"]),
            at_most_one("b", ["x2", "x3", "x4", "x5", "x6"]),
            at_most_one("c", ["x6", "x7"]),
            {"id": "eq", "type": "hard", "variables": ["x0", "x7"], "operator": "==", "rhs": 1},
        ],
    ),
    "maximize_ten_variables": lambda: problem(
        direction="maximize",
        variables=binaries(10),
        linear=[lin(name, index % 3) for index, name in enumerate(binaries(10))],
        cardinality=[
            at_most_one("left", binaries(10)[:6]),
            at_most_one("right", binaries(10)[4:]),
        ],
    ),
}


def pairwise_counts(p: OptimizationProblem, sample: dict[str, int]) -> list[int]:
    return [
        sum(sample[term.variable] for term in constraint.terms)
        for constraint in p.all_constraints()
        if uses_pairwise_penalty(constraint)
    ]


class TestSlackVersusPairwiseIdentity:
    @pytest.mark.parametrize("name", sorted(IDENTITY_PROBLEMS))
    @pytest.mark.parametrize("lam", [1.0, 6.5])
    def test_energy_difference_is_lambda_times_the_count_gap(self, name, lam):
        pairwise_problem = IDENTITY_PROBLEMS[name]()
        slack_problem = linear_rewrite(pairwise_problem)
        assert validate_problem(pairwise_problem) == []
        assert validate_problem(slack_problem) == []

        pairwise = BQMCompiler().compile(pairwise_problem, lam)
        slack = BQMCompiler().compile(slack_problem, lam)
        names = [variable.name for variable in pairwise_problem.variables]
        # The pairwise model's bits are the business variables themselves.
        assert pairwise.internal_variables == set()
        assert sorted(str(v) for v in pairwise.model.variables) == sorted(names)
        slack_bits = sorted(slack.internal_variables)
        assert slack_bits  # one bit per at-most-one

        checked = 0
        for values in itertools.product((0, 1), repeat=len(names)):
            sample = dict(zip(names, values))
            energy_pairwise = pairwise.model.energy(sample)
            energy_slack = min(
                slack.model.energy({**sample, **dict(zip(slack_bits, bits))})
                for bits in itertools.product((0, 1), repeat=len(slack_bits))
            )
            gap = lam * sum(
                max(0, k - 1) ** 2 - comb(k, 2) for k in pairwise_counts(pairwise_problem, sample)
            )
            assert energy_slack - energy_pairwise == pytest.approx(gap, abs=1e-9), sample
            checked += 1
        assert checked == 2 ** len(names)


# --------------------------------------------------------------------------
# Dominance: the compiled model's minimum is the true optimum
# --------------------------------------------------------------------------


DOMINANCE_PROBLEMS = {
    "mixed_minimize": lambda: mixed_problem("minimize"),
    "mixed_maximize": lambda: mixed_problem("maximize"),
    # Tempting objective: every x wants to be chosen, the pairwise
    # constraints and a slack-encoded budget hold it back.
    "greedy_objective": lambda: problem(
        variables=[*binaries(6), {"name": "k", "type": "integer", "lower_bound": 0, "upper_bound": 2}],
        linear=[*(lin(name, -5) for name in binaries(6)), lin("k", -3)],
        quadratic=[quad("x0", "x3", -4)],
        constraints=[
            {
                "id": "budget",
                "type": "hard",
                "terms": [lin("x0", 1), lin("x3", 2), lin("k", 2)],
                "operator": "<=",
                "rhs": 4,
            },
            {
                "id": "soft_pref",
                "type": "soft",
                "weight": 2.5,
                "terms": [lin("x5", 1), lin("k", 1)],
                "operator": "<=",
                "rhs": 1,
            },
        ],
        cardinality=[
            at_most_one("row", ["x0", "x1", "x2"]),
            at_most_one("column", ["x0", "x3", "x5"]),
            at_most_one("tail", ["x3", "x4", "x5"]),
            {"id": "soft_few", "type": "soft", "weight": 1, "variables": binaries(6), "operator": "<=", "rhs": 2},
        ],
    ),
}


def lowest_energy_business_samples(p: OptimizationProblem, lam: float) -> list[dict[str, int]]:
    compiler = BQMCompiler()
    compiled = compiler.compile(p, lam)
    assert compiled.num_variables <= 16
    sampleset = dimod.ExactSolver().sample(compiled.model).lowest(rtol=0.0, atol=1e-9)
    variables, samples, energies = sampleset_to_arrays(sampleset)
    raw = RawSolverResult(variables=variables, samples=samples, energies=energies, backend="exact")
    decoded = compiler.decode(compiled, raw)
    return [
        {name: int(value) for name, value in zip(decoded.variables, row)} for row in decoded.samples
    ]


class TestDominance:
    @pytest.mark.parametrize("name", sorted(DOMINANCE_PROBLEMS))
    @pytest.mark.parametrize("factor", [1.0, 2.0], ids=["initial", "doubled"])
    def test_lowest_energy_states_are_feasible_and_optimal(self, name, factor):
        p = DOMINANCE_PROBLEMS[name]()
        assert validate_problem(p) == []
        assert any(uses_pairwise_penalty(c) for c in p.all_constraints())
        lam = ScaledPenaltyStrategy().initial_penalty(p) * factor

        feasible_rankings = [
            ranking(p, sample)
            for sample in business_assignments(p)
            if validate_solution(p, sample).feasible
        ]
        assert feasible_rankings
        minimize = p.objective.direction == "minimize"
        best = min(feasible_rankings) if minimize else max(feasible_rankings)

        winners = lowest_energy_business_samples(p, lam)
        assert winners
        for sample in winners:
            assert validate_solution(p, sample).feasible, sample
            assert ranking(p, sample) == pytest.approx(best, abs=1e-9), sample

    def test_the_check_can_fail(self):
        """Harness guard: below the penalty scale the minimum is infeasible."""
        p = DOMINANCE_PROBLEMS["greedy_objective"]()
        winners = lowest_energy_business_samples(p, 0.5)
        assert not all(validate_solution(p, sample).feasible for sample in winners)
        assert any(
            sum(sample[name] for name in ("x0", "x1", "x2")) >= 2 for sample in winners
        )


@pytest.mark.parametrize(
    "name",
    [f"identity:{name}" for name in sorted(IDENTITY_PROBLEMS)]
    + [f"dominance:{name}" for name in sorted(DOMINANCE_PROBLEMS)],
)
def test_estimates_match_on_every_problem_here(name):
    kind, key = name.split(":")
    p = (IDENTITY_PROBLEMS if kind == "identity" else DOMINANCE_PROBLEMS)[key]()
    compiled = BQMCompiler().compile(p, ScaledPenaltyStrategy().initial_penalty(p))
    assert estimate_compiled_variables(p) == compiled.num_variables
    assert estimate_encoded_interactions(p) >= compiled.num_interactions
    # The validator reports the same estimate for the BQM path.
    full = validate_problem_full(p, model_type="bqm")
    assert full.estimated_compiled_variables == compiled.num_variables


# --------------------------------------------------------------------------
# Overflow (spec §8.5): refused as PENALTY_OVERFLOW, never raised
# --------------------------------------------------------------------------


def overflow_problem(*, twice: bool) -> OptimizationProblem:
    """λ = 2 × 5e307 = 1e308 is finite; two at-most-ones on one pair are not.

    Each ``<=`` 1 over ``{x, y}`` adds λ to the pair's coupling, so the
    second one takes it to 2e308, past the float range.
    """
    cardinality = [at_most_one("first", ["x", "y"])]
    if twice:
        cardinality.append(at_most_one("second", ["y", "x"]))
    return problem(
        variables=["x", "y"],
        linear=[lin("x", 5e307)],
        cardinality=cardinality,
        solver={"backend": "exact"},
    )


class TestPairwiseOverflow:
    def test_the_penalty_itself_is_finite(self):
        p = overflow_problem(twice=True)
        lam = ScaledPenaltyStrategy().initial_penalty(p)
        assert math.isfinite(lam)
        assert lam == 1e308

    def test_one_constraint_compiles_two_overflow(self):
        once = overflow_problem(twice=False)
        compiled = BQMCompiler().compile(once, ScaledPenaltyStrategy().initial_penalty(once))
        assert math.isfinite(compiled.model.get_quadratic("x", "y"))

        twice = overflow_problem(twice=True)
        with pytest.raises(NonFiniteModelError):
            BQMCompiler().compile(twice, ScaledPenaltyStrategy().initial_penalty(twice))

    def test_the_service_reports_penalty_overflow(self):
        p = overflow_problem(twice=True)
        assert validate_problem(p) == []
        result = OptimizationService().solve(p)
        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["PENALTY_OVERFLOW"]
        assert result.attempts == []
        assert result.solutions == []
        assert result.errors[0].recommended_action


# --------------------------------------------------------------------------
# The compiler's own guard (a problem that skipped the validator)
# --------------------------------------------------------------------------


class TestCompilerGuard:
    def test_repeated_variable_is_a_compilation_error(self):
        p = problem(variables=["x", "y"], cardinality=[at_most_one("dup", ["x", "y", "x"])])
        assert "DUPLICATE_CARDINALITY_VARIABLE" in {e.code for e in validate_problem(p)}
        with pytest.raises(CompilationError, match="dup"):
            BQMCompiler().compile(p, 1.0)
        with pytest.raises(CompilationError):
            BQMCompiler().prepare(p)

    def test_integer_variable_is_a_compilation_error(self):
        p = problem(
            variables=["x", {"name": "k", "type": "integer", "lower_bound": 0, "upper_bound": 1}],
            cardinality=[at_most_one("int", ["x", "k"])],
        )
        assert "CARDINALITY_VARIABLE_NOT_BINARY" in {e.code for e in validate_problem(p)}
        with pytest.raises(CompilationError, match="'k'"):
            BQMCompiler().compile(p, 1.0)

    def test_undeclared_variable_is_a_compilation_error(self):
        p = problem(variables=["x", "y"], cardinality=[at_most_one("ghost", ["x", "z"])])
        assert "UNKNOWN_VARIABLE" in {e.code for e in validate_problem(p)}
        with pytest.raises(CompilationError, match="'z'"):
            BQMCompiler().compile(p, 1.0)
