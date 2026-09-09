"""Service-level resource limit enforcement (Phase 2 spec §8, §14).

The exact backend no longer polices its own size: the service compares the
*compiled* variable count (slack variables included) against
``ExecutionPolicy.exact_max_variables`` and returns a structured
``resource_limit_exceeded`` result instead of raising.
"""

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import OptimizationProblem, SolverPreferences
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.penalty import ScaledPenaltyStrategy
from annealbridge.solvers import SolverRegistry
from annealbridge.validation.estimates import (
    estimate_compiled_variables,
    estimate_cqm_variables,
)
from tests.fakes.local_cqm_backend import FAKE_LOCAL_CQM_NAME, FakeLocalCQMBackend


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


def make_large_problem(
    num_variables: int = 1000, rhs: int = 500
) -> OptimizationProblem:
    """``num_variables`` binaries under one ``<=`` constraint spanning them all.

    Far above the default exhaustive ceiling (24), and expensive enough to
    compile that the difference between "refused before compile" and
    "refused after compile" is visible as wall-clock time as well as a call
    count (2026-09-09 review F-14).
    """
    names = [f"x{index}" for index in range(num_variables)]
    return OptimizationProblem.model_validate(
        {
            "name": "thousand binary variables",
            "variables": [{"name": name} for name in names],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": name, "coefficient": 1} for name in names
                ],
            },
            "constraints": [
                {
                    "id": "budget",
                    "type": "hard",
                    "terms": [{"variable": name, "coefficient": 1} for name in names],
                    "operator": "<=",
                    "rhs": rhs,
                }
            ],
            "solver": {"backend": "exact"},
        }
    )


def make_cqm_problem() -> OptimizationProblem:
    """Three binaries routed at the exhaustive *CQM* fake (3a §26.1).

    ``SolverPreferences.backend`` is a Literal of the shipped names, so a
    test-only backend is selected with ``model_construct``.
    """
    problem = OptimizationProblem.model_validate(
        {
            "name": "three binary variables on the cqm path",
            "variables": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
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
        }
    )
    solver = SolverPreferences.model_construct(backend=FAKE_LOCAL_CQM_NAME)
    return problem.model_copy(update={"solver": solver})


class TestLimitIsCheckedBeforeCompile:
    """2026-09-09 review F-14: the exhaustive ceiling is enforced *before* compile.

    ``estimate_model_variables`` equals what the chosen compiler produces
    (pinned by the compiler tests), so an over-limit problem can be refused
    without paying for a compilation that is thrown away. The post-compile
    check stays as the final guarantee, and both report the same code and
    the same wording.
    """

    @staticmethod
    def count_compile_calls(monkeypatch, compiler_class) -> list[int]:
        """Wrap ``compiler_class.compile`` with a call counter."""
        calls = [0]
        original = compiler_class.compile

        def counting(self, problem, hard_penalty):
            calls[0] += 1
            return original(self, problem, hard_penalty)

        monkeypatch.setattr(compiler_class, "compile", counting)
        return calls

    def test_bqm_path_refuses_without_compiling(self, monkeypatch):
        problem = make_large_problem()
        estimated = estimate_compiled_variables(problem)
        calls = self.count_compile_calls(monkeypatch, BQMCompiler)

        result = OptimizationService().solve(problem)

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["EXACT_VARIABLE_LIMIT"]
        assert calls[0] == 0
        message = result.errors[0].message
        # The estimate is the number the compiled model would have had, and
        # the ceiling is the default ``exact_max_variables``.
        assert str(estimated) in message
        assert "24" in message

    def test_bqm_estimate_is_the_business_variables_plus_slack_bits(self):
        # Pins the number the refusal message quotes: 1000 binaries plus the
        # nine slack bits an ``x0 + ... + x999 <= 500`` needs.
        assert estimate_compiled_variables(make_large_problem()) == 1009

    def test_cqm_path_refuses_without_compiling(self, monkeypatch):
        problem = make_cqm_problem()
        assert estimate_cqm_variables(problem) == 3
        backend = FakeLocalCQMBackend()
        service = OptimizationService(
            registry=SolverRegistry({FAKE_LOCAL_CQM_NAME: backend}),
            policy=ExecutionPolicy(exact_max_variables=2),
        )
        calls = self.count_compile_calls(monkeypatch, CQMCompiler)

        result = service.solve(problem)

        assert result.status == "resource_limit_exceeded"
        assert [error.code for error in result.errors] == ["EXACT_VARIABLE_LIMIT"]
        assert calls[0] == 0
        assert backend.last_compiled is None
        assert backend.solve_calls == 0
        assert "3" in result.errors[0].message

    def test_within_the_limit_still_compiles_and_succeeds(self, monkeypatch):
        problem = make_problem()
        calls = self.count_compile_calls(monkeypatch, BQMCompiler)

        result = OptimizationService().solve(problem)

        assert result.status == "success"
        assert result.solutions
        assert calls[0] >= 1


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
