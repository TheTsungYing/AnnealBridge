"""Every shipped example through the full service pipeline.

The examples ``annealbridge example`` prints and the MCP server serves are
what a new user runs first, so each must solve on the backend its own
``solver`` block names, reach the optimum its description states, and
satisfy every hard constraint. Read from the packaged copies (the ones an
installed wheel has); ``tests/mcp/test_resources.py`` holds them equal to
the repository's ``examples/``.
"""

import pytest

from annealbridge.interfaces.bundled_examples import example_names, example_text
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

# The optimum each example's description states.
OPTIMA = {
    "knapsack": 17.0,
    "integer_knapsack": 34.0,
    "assignment": 8.0,
    "tsp": 8.0,
    "shift_scheduling": 7.0,
}

# shift_scheduling, worked by hand over all 81 ways to give each of the four
# shifts to one of three people. Dislike (mon_day, mon_night, tue_day,
# tue_night): ann 2 5 2 5; ben 3 1 3 1; cal 5 2 1 3.
#   ann, cal, ann, ben: 2 + 2 + 2 + 1 = 7, no rule broken      <-- optimum
#   ann, cal, cal, ben: 6, but cal works mon_night then tue_day (hard rule)
#   ben, cal, cal, ben: 7, the same hard rule broken
#   ann, ben, cal, ben: 5, but ben works mon_night: soft weight 3 -> score 8
#   ann, cal, ben, ben / ben, cal, ann, ben: 8, no rule broken
# Every other feasible roster scores at least 8, so the optimum is unique.
SHIFT_SCHEDULING_OPTIMUM = {
    "ann_mon_day": 1,
    "ann_mon_night": 0,
    "ann_tue_day": 1,
    "ann_tue_night": 0,
    "ben_mon_day": 0,
    "ben_mon_night": 0,
    "ben_tue_day": 0,
    "ben_tue_night": 1,
    "cal_mon_day": 0,
    "cal_mon_night": 1,
    "cal_tue_day": 0,
    "cal_tue_night": 0,
}
# The exhaustive backend's variable limit, which the estimate (slack bits
# included) must stay within.
EXACT_MAX_VARIABLES = 24


def _problem(name: str) -> OptimizationProblem:
    return OptimizationProblem.model_validate_json(example_text(name))


def test_every_shipped_example_has_a_recorded_optimum():
    assert set(OPTIMA) == set(example_names())


@pytest.mark.parametrize("name", example_names())
def test_example_solves_to_its_stated_optimum(name):
    problem = _problem(name)

    result = OptimizationService().solve(problem)

    assert result.status == "success"
    assert result.backend == problem.solver.backend
    best = result.solutions[0]
    assert best.rank == 1
    assert best.hard_constraints_satisfied is True
    assert all(
        evaluation.satisfied
        for evaluation in best.constraint_evaluations
        if evaluation.constraint_type == "hard"
    )
    assert best.objective_value == pytest.approx(OPTIMA[name])


class TestShiftScheduling:
    def test_is_a_binary_version_1_0_document_for_the_exact_backend(self):
        problem = _problem("shift_scheduling")

        assert problem.version == "1.0"
        assert problem.solver.backend == "exact"
        assert len(problem.variables) == 12
        assert {variable.type for variable in problem.variables} == {"binary"}
        hard = [c for c in problem.constraints if c.type == "hard"]
        soft = [c for c in problem.constraints if c.type == "soft"]
        assert len(hard) == 10
        assert len(soft) == 1

    def test_compiled_estimate_fits_the_exact_backend(self):
        result = OptimizationService().validate(_problem("shift_scheduling"))

        assert result.valid is True
        # 12 binaries + 2 slack bits for each "<= 2 shifts" + 1 for each rest
        # rule; the "== 1" coverage constraints need none.
        assert result.estimated_compiled_variables == 21
        assert result.estimated_compiled_variables <= EXACT_MAX_VARIABLES

    def test_optimum_is_proven_and_honours_the_preference(self):
        result = OptimizationService().solve(_problem("shift_scheduling"))

        assert result.status == "success"
        assert result.optimality_proven is True
        best = result.solutions[0]
        assert best.variables == SHIFT_SCHEDULING_OPTIMUM
        assert best.objective_value == pytest.approx(7.0)
        assert best.soft_violation_score == pytest.approx(0.0)
        assert best.ranking_score == pytest.approx(7.0)
        # The unique optimum: the runner-up scores strictly worse.
        assert result.solutions[1].ranking_score > best.ranking_score
