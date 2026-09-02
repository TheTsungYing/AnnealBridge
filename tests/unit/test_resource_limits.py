"""Service-level resource limit enforcement (Phase 2 spec §8, §14).

The exact backend no longer polices its own size: the service compares the
*compiled* variable count (slack variables included) against
``ExecutionPolicy.exact_max_variables`` and returns a structured
``resource_limit_exceeded`` result instead of raising.
"""

from annealbridge.compiler import BQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.penalty import ScaledPenaltyStrategy


def make_problem() -> OptimizationProblem:
    """A small feasible 0/1 problem: maximize 3a + 2b + c s.t. a + b + c <= 2.

    The ``<=`` constraint adds slack variables, so the compiled model has
    strictly more variables than the three business ones.
    """
    return OptimizationProblem.model_validate(
        {
            "name": "resource limit test problem",
            "variables": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"},
            ],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "a", "coefficient": 3},
                    {"variable": "b", "coefficient": 2},
                    {"variable": "c", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "at_most_two",
                    "type": "hard",
                    "terms": [
                        {"variable": "a", "coefficient": 1},
                        {"variable": "b", "coefficient": 1},
                        {"variable": "c", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 2,
                }
            ],
            "solver": {"backend": "exact"},
        }
    )


def compiled_variable_count(problem: OptimizationProblem) -> int:
    """Compile with the same initial penalty the service uses."""
    penalty = ScaledPenaltyStrategy().initial_penalty(problem)
    return BQMCompiler().compile(problem, penalty).num_variables


class TestExactVariableLimit:
    def test_status_and_backend_report_the_limit(self):
        service = OptimizationService(policy=ExecutionPolicy(exact_max_variables=1))
        result = service.solve(make_problem())

        assert result.status == "resource_limit_exceeded"
        assert result.backend == "exact"
        assert result.solutions == []

    def test_single_structured_error_with_the_limit_code(self):
        service = OptimizationService(policy=ExecutionPolicy(exact_max_variables=1))
        result = service.solve(make_problem())

        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.code == "EXACT_VARIABLE_LIMIT"
        assert error.path is None

    def test_message_reports_actual_and_allowed_variable_counts(self):
        problem = make_problem()
        num_variables = compiled_variable_count(problem)
        assert num_variables > 1  # slack variables push it past the limit

        service = OptimizationService(policy=ExecutionPolicy(exact_max_variables=1))
        result = service.solve(problem)

        message = result.errors[0].message
        assert str(num_variables) in message
        assert "1" in message

    def test_limit_counts_compiled_variables_including_slack(self):
        problem = make_problem()
        business_variables = len(problem.variables)
        assert compiled_variable_count(problem) > business_variables

        # A limit equal to the business variable count is still exceeded,
        # because slack variables count toward the compiled total.
        service = OptimizationService(
            policy=ExecutionPolicy(exact_max_variables=business_variables)
        )
        assert service.solve(problem).status == "resource_limit_exceeded"


class TestDefaultPolicySolvesNormally:
    def test_default_policy_still_succeeds(self):
        result = OptimizationService().solve(make_problem())

        assert result.status == "success"
        assert result.backend == "exact"
        assert result.solutions

    def test_generous_limit_is_not_clamped_or_downgraded(self):
        problem = make_problem()
        policy = ExecutionPolicy(
            exact_max_variables=compiled_variable_count(problem)
        )
        result = OptimizationService(policy=policy).solve(problem)

        # Exactly at the limit is allowed: only *over* the limit is rejected.
        assert result.status == "success"
        assert result.backend == "exact"
        assert result.errors == []
