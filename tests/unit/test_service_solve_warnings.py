"""``OptimizationService.solve()`` carries the validator's warnings.

``validate`` and ``solve`` share one backend-aware validation pass
(``_validate_against_backend``), so a solve result reports the same
advisory warnings a validate call would for that backend — an agent that
skips ``validate_optimization_problem`` (the MCP tool description only
suggests it for remote backends) still sees why an answer may be worse
than expected. Before this, ``SolveResult.warnings`` was empty on every
path but one (REMOTE_RETRIES_DISABLED), while the CLI and MCP docs already
promised otherwise.
"""

import pytest

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import SolverRegistry

VALIDATE_ONLY_CODES = {"UNKNOWN_BACKEND", "NO_COMPILER_FOR_MODEL_TYPE"}


def make_problem(**solver) -> OptimizationProblem:
    """max 3 x1 + 2 x2 s.t. x1 + x2 <= 1 (optimum x1=1, objective 3)."""
    return OptimizationProblem.model_validate(
        {
            "name": "solve warnings",
            "variables": [{"name": "x1"}, {"name": "x2"}],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "x1", "coefficient": 3},
                    {"variable": "x2", "coefficient": 2},
                ],
            },
            "constraints": [
                {
                    "id": "capacity",
                    "type": "hard",
                    "terms": [
                        {"variable": "x1", "coefficient": 1},
                        {"variable": "x2", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
            "solver": solver,
        }
    )


def make_wide_integer_problem(**solver) -> OptimizationProblem:
    """maximize a with a <= 999 over a in 0..1000000 (needs 20 encoding bits)."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.1",
            "name": "wide integer",
            "variables": [
                {
                    "name": "a",
                    "type": "integer",
                    "lower_bound": 0,
                    "upper_bound": 1_000_000,
                }
            ],
            "objective": {
                "direction": "maximize",
                "linear_terms": [{"variable": "a", "coefficient": 1}],
            },
            "constraints": [
                {
                    "id": "cap",
                    "type": "hard",
                    "terms": [{"variable": "a", "coefficient": 1}],
                    "operator": "<=",
                    "rhs": 999,
                }
            ],
            "solver": solver,
        }
    )


def make_tiny_soft_weight_problem(**solver) -> OptimizationProblem:
    """A soft constraint whose weight is negligible against the objective."""
    problem = make_problem(**solver).model_dump()
    problem["constraints"].append(
        {
            "id": "prefer_x2",
            "type": "soft",
            "terms": [{"variable": "x2", "coefficient": 1}],
            "operator": "==",
            "rhs": 1,
            "weight": 0.0001,
        }
    )
    return OptimizationProblem.model_validate(problem)


@pytest.fixture
def service() -> OptimizationService:
    return OptimizationService(
        registry=SolverRegistry.default(), policy=ExecutionPolicy()
    )


def codes(result) -> list[str]:
    return [warning.code for warning in result.warnings]


class TestSolveReportsTheValidatorsWarnings:
    def test_seed_on_an_exhaustive_backend(self, service):
        result = service.solve(make_problem(backend="exact", seed=7))

        assert result.status == "success"
        assert codes(result) == ["SEED_IGNORED"]
        warning = result.warnings[0]
        assert warning.path == "solver.seed"
        assert warning.retryable is False
        assert warning.recommended_action

    def test_retries_on_an_exhaustive_backend(self, service):
        result = service.solve(make_problem(backend="exact", max_retries=5))

        assert result.status == "success"
        assert codes(result) == ["PARAMETER_IGNORED"]
        assert result.warnings[0].path == "solver.max_retries"

    def test_wide_integer_range_on_a_bqm_backend(self, service):
        result = service.solve(
            make_wide_integer_problem(backend="simulated_annealing", seed=1)
        )

        # The heuristic may land below 999; the warning is what explains it.
        assert result.status == "success"
        assert "LARGE_INTEGER_RANGE" in codes(result)
        assert result.solutions[0].objective_value <= 999

    def test_tiny_soft_weight(self, service):
        result = service.solve(make_tiny_soft_weight_problem(backend="exact"))

        assert result.status == "success"
        assert codes(result) == ["SOFT_WEIGHT_SMALL"]

    def test_no_warnings_when_there_is_nothing_to_say(self, service):
        result = service.solve(make_problem(backend="exact"))

        assert result.status == "success"
        assert result.warnings == []


class TestSolveMatchesValidate:
    @pytest.mark.parametrize(
        "factory, solver",
        [
            (make_problem, {"backend": "exact", "seed": 7, "max_retries": 5}),
            (make_problem, {"backend": "simulated_annealing", "seed": 7}),
            (make_wide_integer_problem, {"backend": "simulated_annealing", "seed": 1}),
            (make_tiny_soft_weight_problem, {"backend": "exact"}),
        ],
        ids=["exact-seed-retries", "sa-seed", "sa-wide-integer", "exact-tiny-weight"],
    )
    def test_same_codes_in_the_same_order(self, service, factory, solver):
        problem = factory(**solver)

        validated = service.validate(problem)
        solved = service.solve(problem)

        assert solved.status == "success"
        expected = [
            warning.code
            for warning in validated.warnings
            if warning.code not in VALIDATE_ONLY_CODES
        ]
        assert codes(solved) == expected
        # Same objects' worth of content, not just the same codes.
        assert [w.path for w in solved.warnings] == [
            w.path for w in validated.warnings if w.code not in VALIDATE_ONLY_CODES
        ]


class TestWarningsOnOtherStatuses:
    def test_invalid_problem_carries_no_warnings(self, service):
        # Warnings are only produced for an error-free problem: the
        # validator's gate, kept by solve.
        problem = make_problem(backend="exact", seed=7).model_dump()
        problem["constraints"][0]["terms"].append(
            {"variable": "ghost", "coefficient": 1}
        )

        result = service.solve(OptimizationProblem.model_validate(problem))

        assert result.status == "invalid_problem"
        assert [error.code for error in result.errors] == ["UNKNOWN_VARIABLE"]
        assert result.warnings == []

    def test_structured_failure_still_carries_them(self, service):
        # The problem is fine and the validator's advice about it still
        # holds when the run is refused for another reason.
        result = service.solve(make_problem(backend="exact", seed=7, top_k=10_000))

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["TOP_K_LIMIT"]
        assert codes(result) == ["SEED_IGNORED"]

    def test_exact_over_limit_warning_precedes_the_matching_error(self):
        # EXACT_OVER_LIMIT is validate's prediction of EXACT_VARIABLE_LIMIT;
        # on solve both appear, the warning as advice and the error as the
        # verdict — consistent, not contradictory.
        service = OptimizationService(
            registry=SolverRegistry.default(),
            policy=ExecutionPolicy(exact_max_variables=2),
        )

        result = service.solve(make_problem(backend="exact"))

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["EXACT_VARIABLE_LIMIT"]
        assert codes(result) == ["EXACT_OVER_LIMIT"]

    def test_backend_unavailable_still_carries_them(self, service):
        # dwave_qpu is refused by the default policy (remote disabled); the
        # seed advice about it is still reported.
        result = service.solve(make_problem(backend="dwave_qpu", seed=7))

        assert result.status == "backend_unavailable"
        assert "REMOTE_DISABLED" in [error.code for error in result.errors]
        assert codes(result) == ["SEED_IGNORED"]
