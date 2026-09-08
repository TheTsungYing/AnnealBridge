"""Phase 1 backward-compatibility tests (Phase 2 spec §12, 3a spec §18).

The Phase 2 schema extension (new backend literals, ``dwave_qpu`` and
``leap_hybrid_bqm`` option blocks) and the 3a extension (the
``leap_hybrid_cqm`` literal and option block) must be purely additive: the
Phase 1 example JSON files parse unchanged, without a single character
edited, ``version`` stays ``"1.0"`` and the new option blocks default to
``None``.

3b (schema ``version 1.1`` with ``type: "integer"`` variables and their
bounds) is additive in exactly the same way: the new example parses as
1.1, while the Phase 1 examples still parse as 1.0 *and* still compile to
byte-for-byte the same models they did before the integer work landed.
"""

import json

import pytest

from annealbridge.models import OptimizationProblem
from tests.golden.record_phase3a_golden import GOLDEN_PATH, snapshot_problem


@pytest.fixture(scope="module")
def golden() -> dict:
    """The 3b step 0 recording, read once per module."""
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


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

    @pytest.mark.parametrize(
        "filename", ["knapsack.json", "assignment.json", "tsp.json"]
    )
    def test_cqm_backend_options_default_to_none(self, examples_dir, filename):
        """3a §18: the ``leap_hybrid_cqm`` block is additive and defaults to None."""
        path = examples_dir / filename
        problem = OptimizationProblem.model_validate_json(path.read_text())
        assert problem.version == "1.0"
        assert problem.solver.leap_hybrid_cqm is None

    @pytest.mark.parametrize("name", ["knapsack", "assignment", "tsp"])
    def test_phase1_examples_compile_bit_identical_to_golden(
        self, examples_dir, golden, name
    ):
        """The 3b integer work changed nothing for a ``version 1.0`` example.

        The comparison uses the golden recorded in 3b step 0 by
        ``tests/golden/record_phase3a_golden.py`` — the very same data
        ``tests/unit/test_golden_phase3a.py`` asserts against; nothing is
        recorded here, this only pins the *shipped examples* half of it to
        the backward-compatibility file.
        """
        path = examples_dir / f"{name}.json"
        problem = OptimizationProblem.model_validate_json(path.read_text())
        assert problem.version == "1.0"

        current = json.loads(json.dumps(snapshot_problem(problem)))
        assert current == golden["problems"][f"example_{name}"]


class TestPhase3bExampleParses:
    def test_integer_knapsack_parses(self, examples_dir):
        """3b §7: ``version 1.1`` with four bounded integer variables."""
        path = examples_dir / "integer_knapsack.json"
        problem = OptimizationProblem.model_validate_json(path.read_text())

        assert problem.version == "1.1"
        assert problem.name == "integer_knapsack"
        assert len(problem.variables) == 4
        for variable in problem.variables:
            assert variable.type == "integer"
            assert variable.bounds() == (0, 3)
        assert problem.solver.backend == "exact"
