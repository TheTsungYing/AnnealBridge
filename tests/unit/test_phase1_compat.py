"""Phase 1 backward-compatibility tests (Phase 2 spec §12).

The Phase 2 schema extension (new backend literals, ``dwave_qpu`` and
``leap_hybrid_bqm`` option blocks) must be purely additive: the Phase 1
example JSON files parse unchanged, without a single character edited,
and the new option blocks default to ``None``.
"""

import pytest

from annealbridge.models import OptimizationProblem


class TestPhase1ExamplesStillParse:
    @pytest.mark.parametrize(
        ("filename", "expected_name"),
        [("knapsack.json", "knapsack"), ("assignment.json", "assignment")],
    )
    def test_example_parses_unchanged(self, examples_dir, filename, expected_name):
        path = examples_dir / filename
        problem = OptimizationProblem.model_validate_json(path.read_text())
        assert problem.name == expected_name
        assert problem.version == "1.0"
        assert problem.variables
        assert problem.constraints

    @pytest.mark.parametrize("filename", ["knapsack.json", "assignment.json"])
    def test_new_backend_options_default_to_none(self, examples_dir, filename):
        path = examples_dir / filename
        problem = OptimizationProblem.model_validate_json(path.read_text())
        assert problem.solver.dwave_qpu is None
        assert problem.solver.leap_hybrid_bqm is None
