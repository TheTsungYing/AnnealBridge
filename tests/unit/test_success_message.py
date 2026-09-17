"""``SolveResult.message`` on a successful solve, through the real service.

``tests/unit/test_run_attempts_helpers.py`` pins the wording on the pure
helper; here the point is that a real solve fills the field from its own
result — the same backend, proof flag, rank-1 objective and attempt counts
the caller can read — so the sentence can never contradict the payload, and
that it stays deterministic (no timings) run to run.
"""

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.orchestration.optimizer import _success_message
from tests.conftest import EXAMPLES_DIR


def _knapsack(backend: str, **solver) -> OptimizationProblem:
    problem = OptimizationProblem.model_validate_json(
        (EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8")
    )
    return problem.model_copy(
        update={
            "solver": problem.solver.model_copy(update={"backend": backend, **solver})
        }
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


class TestSuccessMessage:
    def test_exhaustive_backend_reports_the_proof(self):
        result = OptimizationService().solve(_knapsack("exact"))

        assert result.status == "success"
        assert result.optimality_proven is True
        # The sentence docs/output-format.md shows for this very example.
        assert result.message == (
            "exact proved optimality: rank 1 has objective 17 (maximize); "
            "10 of 16 distinct candidates were feasible, 5 returned."
        )
        # Rebuilt from the result's own fields: the sentence states nothing
        # the payload does not already carry.
        assert result.message == _success_message(
            result.backend,
            result.objective_direction,
            result.optimality_proven,
            result.attempts[-1],
            result.solutions,
        )
        assert "proved optimality" in result.message
        assert "not proven" not in result.message

    def test_heuristic_backend_says_optimality_is_not_proven(self):
        result = OptimizationService().solve(
            _knapsack("simulated_annealing", num_reads=50, seed=7)
        )

        assert result.status == "success"
        assert result.optimality_proven is False
        assert result.message.endswith("optimality is not proven.")
        assert "distinct candidates" in result.message
        assert f", {len(result.solutions)} returned" in result.message
        assert "proved optimality" not in result.message

    def test_same_seed_gives_the_same_message_without_timings(self):
        first = OptimizationService().solve(
            _knapsack("simulated_annealing", num_reads=50, seed=7)
        )
        second = OptimizationService().solve(
            _knapsack("simulated_annealing", num_reads=50, seed=7)
        )

        assert first.status == second.status == "success"
        assert first.message == second.message
        # Timings differ on every run, so they must stay out of the wording.
        assert "ms" not in first.message

    def test_infeasible_result_still_explains_itself(self):
        result = OptimizationService().solve(_infeasible_pair("exact"))

        assert result.status == "infeasible"
        assert result.message.startswith("No feasible solution")
