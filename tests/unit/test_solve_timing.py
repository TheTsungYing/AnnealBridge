"""Wall-clock timing, ``optimality_proven`` and the version stamp on results.

The service measures its own stages with ``time.perf_counter`` — nothing
here comes from a vendor — so every status carries ``elapsed_ms`` and every
recorded attempt carries the three stage timings plus the compiled size.
"""

from importlib.metadata import version

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.validation.estimates import estimate_compiled_variables
from tests.conftest import EXAMPLES_DIR


def _knapsack(backend: str, **solver) -> OptimizationProblem:
    problem = OptimizationProblem.model_validate_json(
        (EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8")
    )
    return problem.model_copy(
        update={"solver": problem.solver.model_copy(update={"backend": backend, **solver})}
    )


def _infeasible_pair(backend: str) -> OptimizationProblem:
    """x1 + x2 >= 2 together with x1 + x2 <= 1: each rhs reachable, jointly not."""
    return OptimizationProblem.model_validate(
        {
            "name": "jointly-infeasible",
            "variables": [
                {"name": "x1", "type": "binary"},
                {"name": "x2", "type": "binary"},
            ],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "x1", "coefficient": 1},
                    {"variable": "x2", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "at-least-two",
                    "type": "hard",
                    "terms": [
                        {"variable": "x1", "coefficient": 1},
                        {"variable": "x2", "coefficient": 1},
                    ],
                    "operator": ">=",
                    "rhs": 2,
                },
                {
                    "id": "at-most-one",
                    "type": "hard",
                    "terms": [
                        {"variable": "x1", "coefficient": 1},
                        {"variable": "x2", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 1,
                },
            ],
            "solver": {"backend": backend},
        }
    )


class TestTiming:
    def test_success_carries_stage_and_total_timings(self):
        result = OptimizationService().solve(_knapsack("exact"))

        assert result.status == "success"
        assert result.elapsed_ms is not None and result.elapsed_ms >= 0
        (attempt,) = result.attempts
        assert attempt.compile_ms >= 0
        assert attempt.solve_ms >= 0
        assert attempt.validate_ms >= 0
        # The total spans the stages plus validation and bookkeeping.
        assert result.elapsed_ms >= attempt.compile_ms + attempt.solve_ms + attempt.validate_ms

    def test_attempt_records_the_actual_compiled_size(self):
        problem = _knapsack("exact")
        result = OptimizationService().solve(problem)

        (attempt,) = result.attempts
        assert attempt.compiled_variables == estimate_compiled_variables(problem) == 8
        # Squared capacity constraint couples every pair of its 8 columns.
        assert attempt.compiled_interactions == 28

    def test_infeasible_attempts_are_timed_too(self):
        result = OptimizationService().solve(_infeasible_pair("exact"))

        assert result.status == "infeasible"
        assert result.elapsed_ms >= 0
        (attempt,) = result.attempts
        assert attempt.compile_ms >= 0
        assert attempt.solve_ms >= 0
        assert attempt.validate_ms >= 0
        assert attempt.compiled_variables is not None

    def test_invalid_problem_is_stamped_but_has_no_attempts(self):
        problem = _knapsack("exact").model_copy(update={"constraints": []})
        broken = problem.model_copy(
            update={
                "variables": problem.variables[:1],
            }
        )
        # The objective still references the dropped items: a semantic error.
        result = OptimizationService().solve(broken)

        assert result.status == "invalid_problem"
        assert result.attempts == []
        assert result.elapsed_ms >= 0
        assert result.annealbridge_version == version("annealbridge")

    def test_backend_unavailable_is_stamped(self):
        result = OptimizationService().solve(_knapsack("no-such-backend"))

        assert result.status == "backend_unavailable"
        assert result.elapsed_ms >= 0
        assert result.annealbridge_version == version("annealbridge")


class TestVersionStamp:
    def test_success_carries_the_installed_version(self):
        result = OptimizationService().solve(_knapsack("exact"))

        assert result.annealbridge_version == version("annealbridge")


class TestOptimalityProven:
    def test_exhaustive_backend_proves_optimality(self):
        result = OptimizationService().solve(_knapsack("exact"))

        assert result.status == "success"
        assert result.optimality_proven is True
        assert result.infeasibility_proven is False

    def test_heuristic_backend_never_claims_it(self):
        result = OptimizationService().solve(
            _knapsack("simulated_annealing", num_reads=50, seed=7)
        )

        assert result.status == "success"
        assert result.optimality_proven is False

    def test_infeasible_result_never_claims_it(self):
        result = OptimizationService().solve(_infeasible_pair("exact"))

        assert result.status == "infeasible"
        assert result.infeasibility_proven is True
        assert result.optimality_proven is False
