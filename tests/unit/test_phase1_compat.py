"""Phase 1 backward-compatibility tests (Phase 2 spec §12).

The Phase 2 schema extension (new backend literals, ``dwave_qpu`` and
``leap_hybrid_bqm`` option blocks) must be purely additive: the Phase 1
example JSON files parse unchanged, without a single character edited,
and the new option blocks default to ``None``.
"""

from pathlib import Path

import pytest

from annealbridge.models import OptimizationProblem

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
KNAPSACK_PATH = EXAMPLES_DIR / "knapsack.json"
ASSIGNMENT_PATH = EXAMPLES_DIR / "assignment.json"


class TestPhase1ExamplesStillParse:
    @pytest.mark.parametrize(
        ("path", "expected_name"),
        [(KNAPSACK_PATH, "knapsack"), (ASSIGNMENT_PATH, "assignment")],
    )
    def test_example_parses_unchanged(self, path, expected_name):
        problem = OptimizationProblem.model_validate_json(path.read_text())
        assert problem.name == expected_name
        assert problem.version == "1.0"
        assert problem.variables
        assert problem.constraints

    @pytest.mark.parametrize("path", [KNAPSACK_PATH, ASSIGNMENT_PATH])
    def test_new_backend_options_default_to_none(self, path):
        problem = OptimizationProblem.model_validate_json(path.read_text())
        assert problem.solver.dwave_qpu is None
        assert problem.solver.leap_hybrid_bqm is None
