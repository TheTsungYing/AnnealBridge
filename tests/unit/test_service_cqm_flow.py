"""The service's CQM path, end to end through a real constraint model (3a §26.1).

``FakeLocalCQMBackend`` (``tests/fakes/local_cqm_backend.py``) is the first
backend declaring ``supported_model_types=["cqm"]``, so these tests are the
first time the service, ``CQMCompiler`` and a real
``dimod.ExactCQMSolver`` run together. What must hold (3a §16, §9.2, §9.3,
§22, §31):

* the compiler is chosen from the declaration (``model_type == "cqm"`` in
  ``validate()`` and in the result metadata), never from the name;
* hard constraints are native: ``hard_penalty is None``, no slack, every
  trace ``native=True`` with ``penalty=None``;
* exactly one attempt with ``penalty=None``, whatever ``max_retries`` says,
  and no ``REMOTE_RETRIES_DISABLED`` warning;
* the knapsack optimum equals the exact (BQM) path's, bit for bit;
* the sampler's ``is_feasible`` verdict is only *reported*: a tampered
  sampleset flagging an infeasible row as feasible still loses that row
  under independent validation (overview principle 2);
* an exhaustive CQM backend proves infeasibility (§16.2 step 15);
* a service without a CQM compiler refuses with
  ``NO_COMPILER_FOR_MODEL_TYPE`` instead of routing elsewhere;
* ``validate()`` estimates ``len(variables)`` on this path (§9.2) and warns
  ``PARAMETER_IGNORED`` for ``penalty_multiplier`` / ``max_retries`` (§9.3).
"""

import json
from pathlib import Path

import dimod
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
from annealbridge.orchestration import OptimizationService
from annealbridge.solvers import SolverRegistry
from tests.fakes.local_cqm_backend import (
    FAKE_LOCAL_CQM_NAME,
    FakeLocalCQMBackend,
    all_feasible_sampleset,
)

KNAPSACK = Path(__file__).resolve().parents[2] / "examples" / "knapsack.json"
KNAPSACK_BUSINESS_NAMES = {"item_a", "item_b", "item_c", "item_d"}
KNAPSACK_OPTIMUM_VALUE = 17.0
KNAPSACK_OPTIMUM_SELECTION = {"item_a": 1, "item_b": 0, "item_c": 1, "item_d": 0}


def make_registry(fake: FakeLocalCQMBackend) -> SolverRegistry:
    """Every default backend plus the fake, so the exact path is still there."""
    defaults = SolverRegistry.default()
    backends = {name: defaults.get(name) for name in defaults.names()}
    assert FAKE_LOCAL_CQM_NAME not in backends
    backends[FAKE_LOCAL_CQM_NAME] = fake
    return SolverRegistry(backends)


def route_to_fake(problem: OptimizationProblem, **preferences) -> OptimizationProblem:
    """Point ``problem`` at the fake backend.

    ``SolverPreferences.backend`` is a Literal of the shipped names, so a
    test-only backend is set with ``model_construct`` (defaults are still
    filled in), the same way ``test_fifth_backend.py`` does it.
    """
    solver = SolverPreferences.model_construct(backend=FAKE_LOCAL_CQM_NAME, **preferences)
    return problem.model_copy(update={"solver": solver})


def load_knapsack() -> OptimizationProblem:
    payload = json.loads(KNAPSACK.read_text(encoding="utf-8"))
    return OptimizationProblem.model_validate(payload)


def knapsack_on_fake(**preferences) -> OptimizationProblem:
    return route_to_fake(load_knapsack(), **preferences)


def contradictory_problem() -> OptimizationProblem:
    """x1 + x2 >= 2 and x1 + x2 <= 1: no assignment satisfies both."""
    terms = [LinearTerm(variable="x1", coefficient=1), LinearTerm(variable="x2", coefficient=1)]
    return route_to_fake(
        OptimizationProblem(
            name="contradiction",
            variables=[Variable(name="x1"), Variable(name="x2")],
            objective=Objective(direction="minimize", linear_terms=[terms[0]]),
            constraints=[
                Constraint(id="at-least-two", type="hard", terms=terms, operator=">=", rhs=2),
                Constraint(id="at-most-one", type="hard", terms=terms, operator="<=", rhs=1),
            ],
            solver=SolverPreferences(backend="exact"),
        )
    )


def overflowing_soft_problem() -> OptimizationProblem:
    """Valid problem whose CQM compilation leaves the floating-point range.

    One integer variable and one soft equality with a ``1e200``
    coefficient: ``weight * coefficient**2`` is ``inf`` once
    ``expand_square_qm`` squares it into the objective.
    """
    return route_to_fake(
        OptimizationProblem(
            version="1.1",
            name="cqm soft weight overflow",
            variables=[Variable(name="x", type="integer", lower_bound=0, upper_bound=1)],
            objective=Objective(direction="minimize", linear_terms=[]),
            constraints=[
                Constraint(
                    id="huge",
                    type="soft",
                    weight=1.0,
                    terms=[LinearTerm(variable="x", coefficient=1e200)],
                    operator="==",
                    rhs=0.0,
                )
            ],
        )
    )


@pytest.fixture
def fake() -> FakeLocalCQMBackend:
    return FakeLocalCQMBackend()


@pytest.fixture
def service(fake) -> OptimizationService:
    return OptimizationService(registry=make_registry(fake))


@pytest.fixture
def exact_result():
    """The knapsack solved on the shipped exact (BQM) path, for comparison."""
    result = OptimizationService().solve(load_knapsack())
    assert result.status == "success"
    assert result.backend == "exact"
    return result


class TestCompilerSelection:
    def test_declaration_selects_the_cqm_compiler(self, service, fake):
        assert service.validate(knapsack_on_fake()).model_type == "cqm"

        result = service.solve(knapsack_on_fake())

        assert result.status == "success"
        assert result.backend == FAKE_LOCAL_CQM_NAME
        assert result.metadata is not None
        assert result.metadata.model_type == "cqm"
        assert result.metadata.remote is False
        assert fake.solve_calls == 1
        compiled = fake.last_compiled
        assert compiled is not None
        assert compiled.model_type == "cqm"
        assert isinstance(compiled.model, dimod.ConstrainedQuadraticModel)

    def test_hard_constraints_are_native_without_penalty_or_slack(self, service, fake):
        service.solve(knapsack_on_fake())

        compiled = fake.last_compiled
        assert compiled.hard_penalty is None
        assert compiled.internal_variables == set()
        assert compiled.num_variables == 4
        assert [trace.constraint_id for trace in compiled.constraint_trace] == ["capacity"]
        (trace,) = compiled.constraint_trace
        assert trace.constraint_type == "hard"
        assert trace.native is True
        assert trace.penalty is None
        assert trace.slack_range is None
        assert trace.generated_variables == []


class TestSingleAttemptWithoutPenalty:
    def test_exactly_one_attempt_with_penalty_none(self, service):
        result = service.solve(knapsack_on_fake())

        assert len(result.attempts) == 1
        attempt = result.attempts[0]
        assert attempt.attempt == 1
        assert attempt.penalty is None
        # ExactCQMSolver enumerates all 2**4 assignments; none is dropped.
        assert attempt.samples_received == 16
        assert attempt.unique_samples == 16
        assert result.warnings == []

    def test_max_retries_does_not_add_attempts(self, service, fake):
        result = service.solve(knapsack_on_fake(max_retries=5))

        assert result.status == "success"
        assert len(result.attempts) == 1
        assert result.attempts[0].penalty is None
        assert fake.solve_calls == 1
        assert "REMOTE_RETRIES_DISABLED" not in [w.code for w in result.warnings]


class TestKnapsackMatchesTheExactPath:
    def test_optimum_and_ranking_agree_with_exact(self, service, exact_result):
        result = service.solve(knapsack_on_fake())

        assert result.status == "success"
        best = result.solutions[0]
        assert best.rank == 1
        assert best.objective_value == pytest.approx(KNAPSACK_OPTIMUM_VALUE)
        assert best.objective_value == exact_result.solutions[0].objective_value
        assert best.variables == exact_result.solutions[0].variables
        assert best.variables == KNAPSACK_OPTIMUM_SELECTION
        assert best.hard_constraints_satisfied is True
        # Same top-k, same order: ranking is computed from the original
        # problem, so the model type cannot change it.
        assert [s.variables for s in result.solutions] == [
            s.variables for s in exact_result.solutions
        ]
        assert [s.ranking_score for s in result.solutions] == [
            s.ranking_score for s in exact_result.solutions
        ]

    def test_no_internal_variable_leaks(self, service):
        result = service.solve(knapsack_on_fake(top_k=16))

        assert result.solutions
        for solution in result.solutions:
            assert set(solution.variables) == KNAPSACK_BUSINESS_NAMES
            assert not any(name.startswith("__") for name in solution.variables)


class TestSamplerVerdictIsNotTrusted:
    # Rows for the knapsack, all flagged feasible by the "sampler": the
    # first packs every item (weight 18 > 10) and must be thrown out.
    ALL_ITEMS = {"item_a": 1, "item_b": 1, "item_c": 1, "item_d": 1}
    ROWS = [
        ALL_ITEMS,
        {"item_a": 1, "item_b": 0, "item_c": 1, "item_d": 0},
        {"item_a": 0, "item_b": 1, "item_c": 0, "item_d": 1},
    ]

    def test_tampered_feasible_flag_does_not_reach_solutions(self):
        tampered = all_feasible_sampleset(self.ROWS, [-31.0, -17.0, -14.0])
        assert bool(tampered.record.is_feasible.all())
        fake = FakeLocalCQMBackend(override_sampleset=tampered)
        service = OptimizationService(registry=make_registry(fake))

        result = service.solve(knapsack_on_fake(top_k=10))

        assert result.status == "success"
        assert result.attempts[0].samples_received == 3
        assert result.attempts[0].feasible_samples == 2
        selections = [s.variables for s in result.solutions]
        assert self.ALL_ITEMS not in selections
        assert selections[0] == KNAPSACK_OPTIMUM_SELECTION
        assert all(s.hard_constraints_satisfied for s in result.solutions)
        # The sampler's count is reported as given (§22): informational,
        # and visibly at odds with the 2 the validator accepted.
        assert result.metadata is not None
        assert result.metadata.sampler_reported_feasible == 3
        assert result.metadata.model_type == "cqm"

    def test_untampered_count_matches_the_validator_on_knapsack(self, service):
        result = service.solve(knapsack_on_fake())

        # 10 feasible subsets (weights 6/5/4/3, capacity 10). ExactCQMSolver's
        # own verdict agrees with the validator here (the
        # compiler cross-check in test_cqm_compiler proves that in
        # general); the point is that the number is reported, not used.
        assert result.metadata.sampler_reported_feasible == result.attempts[0].feasible_samples
        assert result.metadata.sampler_reported_feasible == 10


class TestInfeasibilityIsProven:
    def test_contradictory_constraints_are_proven_infeasible(self, service, fake):
        result = service.solve(contradictory_problem())

        assert result.status == "infeasible"
        assert result.infeasibility_proven is True
        assert result.solutions == []
        assert len(result.attempts) == 1
        assert result.attempts[0].penalty is None
        assert result.attempts[0].samples_received == 4
        assert result.attempts[0].feasible_samples == 0
        assert result.warnings == []
        assert "enumerated every assignment" in result.message
        assert result.metadata is not None
        assert result.metadata.sampler_reported_feasible == 0
        assert fake.solve_calls == 1


class TestNonFiniteModelNeverReachesTheBackend:
    """2026-09-11 review F06, through the service.

    The soft penalty ``weight * coefficient**2`` (here ``1.0 * 1e200**2``)
    overflows to ``inf`` while the problem itself carries only finite
    numbers, so validation passes and the compiler's guard is the only
    thing between the backend and an infinite bias. With no hard penalty on
    this path the service reports it as ``invalid_problem`` /
    COMPILATION_FAILED, never PENALTY_OVERFLOW.
    """

    def test_overflowing_soft_penalty_is_an_invalid_problem(self, service, fake):
        assert service.validate(overflowing_soft_problem()).valid is True

        result = service.solve(overflowing_soft_problem())

        assert result.status == "invalid_problem"
        assert [e.code for e in result.errors] == ["COMPILATION_FAILED"]
        assert result.backend == FAKE_LOCAL_CQM_NAME
        assert result.solutions == []
        assert result.attempts == []
        assert result.metadata is None
        # The whole point: the backend was never called with the inf model.
        assert fake.solve_calls == 0


class TestNoCqmCompiler:
    def test_bqm_only_service_is_a_configuration_error(self, fake):
        service = OptimizationService(
            compilers=[BQMCompiler()], registry=make_registry(fake)
        )

        result = service.solve(knapsack_on_fake())

        assert result.status == "configuration_error"
        assert [e.code for e in result.errors] == ["NO_COMPILER_FOR_MODEL_TYPE"]
        assert result.backend == FAKE_LOCAL_CQM_NAME
        assert result.solutions == []
        assert result.attempts == []
        # Never silently compiled to a BQM instead (overview principle 5).
        assert fake.solve_calls == 0
        assert service.validate(knapsack_on_fake()).model_type is None


class TestValidateOnTheCqmPath:
    def test_estimate_is_the_variable_count_without_slack(self, service):
        result = service.validate(knapsack_on_fake())

        assert result.valid is True
        assert result.model_type == "cqm"
        assert result.estimated_compiled_variables == 4
        # §9.2: the BQM path pays slack bits for the "<=" constraint.
        bqm_estimate = OptimizationService().validate(load_knapsack())
        assert bqm_estimate.model_type == "bqm"
        assert bqm_estimate.estimated_compiled_variables > 4

    def test_defaults_raise_no_backend_warnings(self, service):
        result = service.validate(knapsack_on_fake())

        codes = {w.code for w in result.warnings}
        assert "PARAMETER_IGNORED" not in codes
        assert "SEED_IGNORED" not in codes
        assert "UNKNOWN_BACKEND" not in codes
        assert "NO_COMPILER_FOR_MODEL_TYPE" not in codes

    def test_penalty_multiplier_is_reported_ignored(self, service):
        result = service.validate(knapsack_on_fake(penalty_multiplier=3.0))

        assert result.valid is True
        (warning,) = [w for w in result.warnings if w.code == "PARAMETER_IGNORED"]
        assert warning.path == "solver.penalty_multiplier"
        assert FAKE_LOCAL_CQM_NAME in warning.message

    def test_max_retries_is_reported_ignored(self, service):
        result = service.validate(knapsack_on_fake(max_retries=2))

        (warning,) = [w for w in result.warnings if w.code == "PARAMETER_IGNORED"]
        assert warning.path == "solver.max_retries"
        assert "CQM" in warning.message

    def test_seed_is_reported_ignored(self, service):
        result = service.validate(knapsack_on_fake(seed=7))
        (warning,) = [w for w in result.warnings if w.code == "SEED_IGNORED"]
        assert warning.path == "solver.seed"
