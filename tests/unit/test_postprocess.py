"""Opt-in post-processing over the original variables (batch 4 G).

Postprocess spec 2026-09-23 (``.ai-docs/annealbridge_postprocess_spec_2026-09-23.md``,
§10 revision v3 first), acceptance conditions 2-10. The raw samples come
from :class:`FixedSamplesBackend`, a non-exhaustive local backend that
returns the same business rows on every call, so every repair and local
search below is deterministic and can be pinned by hand:

* every returned solution is re-validated like a solver sample, and a
  produced one has no energy, a zero sample count and a truthful source;
* repair turns infeasible samples feasible and stops the retry ladder,
  while a repair that cannot succeed leaves the ladder untouched;
* local search reaches the known optimum, never leaves the bounds, never
  surfaces internal variables, and adds nothing at a local optimum;
* both ceilings are refused before solving only while post-processing is
  on, and a ceiling reached mid-run is reported, never hidden;
* an exhaustive backend ignores it with PARAMETER_IGNORED;
* the output is reproducible and the BQM and CQM paths agree.
"""

import itertools
import json

import numpy as np
import pytest

from annealbridge.interfaces.capabilities import build_capabilities
from annealbridge.models import (
    CompiledProblem,
    OptimizationProblem,
    SolverPreferences,
)
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.orchestration.candidates import evaluate_objective
from annealbridge.orchestration.postprocess import postprocess_costs, scan_cost
from annealbridge.solvers import (
    AvailabilityStatus,
    RawSolverResult,
    SolverCapabilities,
    SolverRegistry,
)
from annealbridge.validation import validate_problem, validate_solution
from tests.conftest import EXAMPLES_DIR
from tests.fakes.local_cqm_backend import FAKE_LOCAL_CQM_NAME, FakeLocalCQMBackend

ON = "repair_local_search"
FIXED_BQM = "fixed_samples_bqm"
FIXED_CQM = "fixed_samples_cqm"
PRODUCED_SOURCES = {"repaired", "local_search", "repaired_local_search"}


# --------------------------------------------------------------------------
# A deterministic, non-exhaustive local backend
# --------------------------------------------------------------------------


class FixedSamplesBackend:
    """Returns the same business ``rows`` on every call (test-only).

    Rows name business variables only; the compiled model's other columns
    are filled in: the bits of an integer variable's binary encoding (BQM
    path) are chosen so the row decodes back to the given value, and every
    other internal column (slack) is 0. The compiler's ``decode`` strips
    them again, exactly as it does for a real sampler.
    """

    def __init__(self, rows: list[dict[str, int]], model_type: str = "bqm") -> None:
        self.rows = [dict(row) for row in rows]
        self.model_type = model_type
        self.capabilities = SolverCapabilities(
            name=FIXED_BQM if model_type == "bqm" else FIXED_CQM,
            remote=False,
            heuristic=True,
            exhaustive=False,
            supports_seed=False,
            supports_num_reads=False,
            supports_time_limit=False,
            supported_model_types=[model_type],
            returns_multiple_samples=True,
            description="test-only: fixed business samples",
        )
        self.solve_calls = 0

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return False

    def is_available(self) -> AvailabilityStatus:
        return AvailabilityStatus(category="available")

    def resolve_time_limit(self, compiled, preferences):
        return None

    @staticmethod
    def _compiled_row(compiled: CompiledProblem, row: dict[str, int]) -> dict[str, int]:
        values = dict(row)
        for name, encoding in compiled.integer_encodings.items():
            target = values.pop(name) - encoding.lower
            for bits in itertools.product((0, 1), repeat=len(encoding.bits)):
                if sum(c * b for c, b in zip(encoding.coefficients, bits)) == target:
                    values.update(zip(encoding.bits, bits))
                    break
            else:  # pragma: no cover - a test wrote an out-of-range value
                raise AssertionError(f"{name}: no encoding for {target}")
        return values

    def solve(self, compiled: CompiledProblem, preferences: SolverPreferences):
        self.solve_calls += 1
        model = compiled.model
        variables = [str(v) for v in model.variables]
        rows = []
        for row in self.rows:
            values = self._compiled_row(compiled, row)
            rows.append({v: int(values.get(v, 0)) for v in variables})
        if self.model_type == "bqm":
            energies = [float(model.energy(row)) for row in rows]
        else:
            energies = [float(model.objective.energy(row)) for row in rows]
        return RawSolverResult.from_dicts(
            rows, energies, backend=self.name, variables=variables, dtype=np.int64
        )


def route(problem: OptimizationProblem, backend: str, **preferences) -> OptimizationProblem:
    """Point ``problem`` at a (possibly test-only) backend.

    ``SolverPreferences.backend`` is a Literal of the shipped names, so the
    preferences are built with ``model_construct`` (defaults still filled
    in), as ``tests/architecture/test_fifth_backend.py`` does.
    """
    solver = SolverPreferences.model_construct(backend=backend, **preferences)
    return problem.model_copy(update={"solver": solver})


def fixed_service(
    rows: list[dict[str, int]], model_type: str = "bqm", **policy
) -> tuple[FixedSamplesBackend, OptimizationService]:
    backend = FixedSamplesBackend(rows, model_type)
    service = OptimizationService(
        registry=SolverRegistry({backend.name: backend}),
        policy=ExecutionPolicy(**policy),
    )
    return backend, service


def solve_fixed(problem, rows, *, model_type="bqm", policy=None, **preferences):
    backend, service = fixed_service(rows, model_type, **(policy or {}))
    result = service.solve(route(problem, backend.name, **preferences))
    return backend, result


def codes(errors) -> list[str]:
    return [error.code for error in errors]


def assert_revalidated(problem: OptimizationProblem, result) -> None:
    """Spec §9 item 2: every returned field is the re-validation's own."""
    minimize = problem.objective.direction == "minimize"
    names = {variable.name for variable in problem.variables}
    for solution in result.solutions:
        assert set(solution.variables) == names
        check = validate_solution(problem, solution.variables)
        objective = evaluate_objective(problem.objective, solution.variables)
        soft = check.soft_violation_score
        assert solution.hard_constraints_satisfied is check.feasible is True
        assert solution.objective_value == objective
        assert solution.soft_violation_score == soft
        assert solution.ranking_score == (objective + soft if minimize else objective - soft)
        if solution.source == "solver":
            assert solution.energy is not None
            assert solution.sample_count >= 1
        else:
            assert solution.source in PRODUCED_SOURCES
            assert solution.energy is None
            assert solution.sample_count == 0
    scores = [s.ranking_score for s in result.solutions]
    assert scores == sorted(scores, reverse=not minimize)


def domain(variable) -> range:
    lower, upper = variable.bounds()
    return range(lower, upper + 1)


def brute_force_best(problem: OptimizationProblem) -> float:
    """Best ranking score over every feasible assignment (small problems only)."""
    minimize = problem.objective.direction == "minimize"
    names = [variable.name for variable in problem.variables]
    best = None
    for values in itertools.product(*(domain(v) for v in problem.variables)):
        sample = dict(zip(names, values))
        check = validate_solution(problem, sample)
        if not check.feasible:
            continue
        objective = evaluate_objective(problem.objective, sample)
        score = (
            objective + check.soft_violation_score
            if minimize
            else objective - check.soft_violation_score
        )
        if best is None or (score < best if minimize else score > best):
            best = score
    assert best is not None
    return best


def strip_timings(value):
    """Drop every ``*_ms`` key (and ``elapsed_ms``) at any depth."""
    if isinstance(value, dict):
        return {k: strip_timings(v) for k, v in value.items() if not k.endswith("_ms")}
    if isinstance(value, list):
        return [strip_timings(v) for v in value]
    return value


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------

ASSIGNMENT_VARIABLES = ["t1_w0", "t1_w1", "t1_w2", "t2_w0", "t2_w1", "t2_w2"]


def assignment_problem() -> OptimizationProblem:
    """Two tasks, three workers, one worker per task (two one-hot groups).

    Costs t1: 1, 2, 5 and t2: 1, 3, 5, plus 10 when both tasks share a
    worker. Optimum: t1 -> w1, t2 -> w0, cost 3.
    """
    costs = {"t1_w0": 1, "t1_w1": 2, "t1_w2": 5, "t2_w0": 1, "t2_w1": 3, "t2_w2": 5}
    return OptimizationProblem.model_validate(
        {
            "name": "two-task assignment",
            "variables": [{"name": name} for name in ASSIGNMENT_VARIABLES],
            "objective": {
                "direction": "minimize",
                "linear_terms": [
                    {"variable": name, "coefficient": cost} for name, cost in costs.items()
                ],
                "quadratic_terms": [
                    {"variable1": f"t1_w{k}", "variable2": f"t2_w{k}", "coefficient": 10}
                    for k in range(3)
                ],
            },
            "constraints": [
                {
                    "id": f"{task}_one_worker",
                    "type": "hard",
                    "terms": [
                        {"variable": f"{task}_w{k}", "coefficient": 1} for k in range(3)
                    ],
                    "operator": "==",
                    "rhs": 1,
                }
                for task in ("t1", "t2")
            ],
            "solver": {"backend": "exact"},
        }
    )


def assignment(t1: int | None, t2: int | None) -> dict[str, int]:
    """The assignment row picking worker ``t1`` / ``t2`` (None: nobody)."""
    row = dict.fromkeys(ASSIGNMENT_VARIABLES, 0)
    if t1 is not None:
        row[f"t1_w{t1}"] = 1
    if t2 is not None:
        row[f"t2_w{t2}"] = 1
    return row


def knapsack_problem() -> OptimizationProblem:
    """``examples/knapsack.json``: optimum {item_a, item_c}, value 17."""
    payload = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
    return OptimizationProblem.model_validate(payload)


def knapsack(*items: str) -> dict[str, int]:
    row = dict.fromkeys(["item_a", "item_b", "item_c", "item_d"], 0)
    for item in items:
        row[f"item_{item}"] = 1
    return row


def contradictory_problem() -> OptimizationProblem:
    """x1 + x2 >= 2 and x1 + x2 <= 1: the validator lets it through, but no
    assignment is feasible, so every repair must fail."""
    terms = [{"variable": "x1", "coefficient": 1}, {"variable": "x2", "coefficient": 1}]
    return OptimizationProblem.model_validate(
        {
            "name": "contradiction",
            "variables": [{"name": "x1"}, {"name": "x2"}],
            "objective": {
                "direction": "minimize",
                "linear_terms": [{"variable": "x1", "coefficient": 1}],
            },
            "constraints": [
                {"id": "at_least_two", "type": "hard", "terms": terms, "operator": ">=", "rhs": 2},
                {"id": "at_most_one", "type": "hard", "terms": terms, "operator": "<=", "rhs": 1},
            ],
            "solver": {"backend": "exact"},
        }
    )


def integer_problem() -> OptimizationProblem:
    """maximize 2x + y, x in 0..3, y in -2..2, x + y <= 3 (optimum x=3, y=0)."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.1",
            "name": "bounded integers",
            "variables": [
                {"name": "x", "type": "integer", "lower_bound": 0, "upper_bound": 3},
                {"name": "y", "type": "integer", "lower_bound": -2, "upper_bound": 2},
            ],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "x", "coefficient": 2},
                    {"variable": "y", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "budget",
                    "type": "hard",
                    "terms": [
                        {"variable": "x", "coefficient": 1},
                        {"variable": "y", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 3,
                }
            ],
            "solver": {"backend": "exact"},
        }
    )


def wide_integer_problem(direction: str, constraint: dict | None) -> OptimizationProblem:
    """One integer ``x`` in 0..100: long ±1 walks hit the 4·n step cap."""
    return OptimizationProblem.model_validate(
        {
            "version": "1.1",
            "name": "wide integer",
            "variables": [
                {"name": "x", "type": "integer", "lower_bound": 0, "upper_bound": 100}
            ],
            "objective": {
                "direction": direction,
                "linear_terms": [{"variable": "x", "coefficient": 1}],
            },
            "constraints": [] if constraint is None else [constraint],
            "solver": {"backend": "exact"},
        }
    )


def soft_only_problem() -> OptimizationProblem:
    """No hard constraint: minimize a + 2b - 3c, soft a + b + c == 2 (weight 4).

    Optimum a=1, b=0, c=1: objective -2, residual 0.
    """
    return OptimizationProblem.model_validate(
        {
            "name": "soft only",
            "variables": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
            "objective": {
                "direction": "minimize",
                "linear_terms": [
                    {"variable": "a", "coefficient": 1},
                    {"variable": "b", "coefficient": 2},
                    {"variable": "c", "coefficient": -3},
                ],
            },
            "constraints": [
                {
                    "id": "about_two",
                    "type": "soft",
                    "weight": 4,
                    "terms": [
                        {"variable": "a", "coefficient": 1},
                        {"variable": "b", "coefficient": 1},
                        {"variable": "c", "coefficient": 1},
                    ],
                    "operator": "==",
                    "rhs": 2,
                }
            ],
            "solver": {"backend": "exact"},
        }
    )


def unconstrained_problem() -> OptimizationProblem:
    """No constraint at all: maximize a - b + 2c + ab/2 (optimum a=1, b=0, c=1)."""
    return OptimizationProblem.model_validate(
        {
            "name": "unconstrained",
            "variables": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "a", "coefficient": 1},
                    {"variable": "b", "coefficient": -1},
                    {"variable": "c", "coefficient": 2},
                ],
                "quadratic_terms": [
                    {"variable1": "a", "variable2": "b", "coefficient": 0.5}
                ],
            },
            "constraints": [],
            "solver": {"backend": "exact"},
        }
    )


# --------------------------------------------------------------------------
# Re-validation, sources and repair (spec §9 items 2, 3, 4)
# --------------------------------------------------------------------------


class TestRepair:
    def test_infeasible_samples_are_repaired_and_revalidated(self):
        problem = assignment_problem()
        # Both rows infeasible: nobody assigned, and task 2 left unassigned.
        _, result = solve_fixed(
            problem, [assignment(None, None), assignment(2, None)], postprocess=ON
        )

        assert result.status == "success", result.errors
        assert len(result.attempts) == 1
        assert_revalidated(problem, result)
        assert result.solutions[0].ranking_score == brute_force_best(problem) == 3.0
        assert [(s.variables, s.source) for s in result.solutions] == [
            # (w2, -) is selected first (V = 1), repaired to (w2, w0) and
            # then moved by a swap inside task 1 to the optimum.
            (assignment(1, 0), "repaired_local_search"),
            # (-, -) needs two repair steps; (w0, w1) is a local optimum.
            (assignment(0, 1), "repaired"),
            (assignment(2, 0), "repaired"),
        ]
        attempt = result.attempts[0]
        assert attempt.unique_samples == 2
        assert attempt.feasible_samples == 0  # solver samples only (spec §5)
        assert attempt.postprocess is not None
        assert attempt.postprocess.model_dump() == {
            "candidates_selected": 2,
            "repair_attempted": 2,
            "repair_succeeded": 2,
            "local_search_started": 2,
            "local_search_improved": 1,
            "new_candidates": 3,
            "feasible_added": 3,
            "limit_reached": [],
            "wall_clock_limit_reached": False,
        }
        assert attempt.postprocess_ms is not None and attempt.postprocess_ms >= 0
        assert attempt.validate_ms >= 0
        assert "post-processing" in result.message
        assert "repaired_local_search" in result.message
        assert "POSTPROCESS_LIMIT_REACHED" not in codes(result.warnings)

    def test_successful_repair_stops_the_retry_ladder(self):
        rows = [assignment(None, None)]
        backend, result = solve_fixed(
            assignment_problem(), rows, postprocess=ON, max_retries=2
        )

        assert result.status == "success"
        assert len(result.attempts) == 1
        assert backend.solve_calls == 1
        assert result.solutions[0].source in {"repaired", "repaired_local_search"}

    def test_without_postprocessing_the_ladder_retries_as_before(self):
        rows = [assignment(None, None)]
        backend, result = solve_fixed(assignment_problem(), rows, max_retries=2)

        assert result.status == "infeasible"
        assert len(result.attempts) == 3
        assert backend.solve_calls == 3
        assert all(attempt.postprocess is None for attempt in result.attempts)
        assert all(attempt.postprocess_ms is None for attempt in result.attempts)

    def test_failed_repairs_leave_the_ladder_alone(self):
        problem = contradictory_problem()
        assert validate_problem(problem) == []  # nothing stops it before solving
        backend, result = solve_fixed(
            problem, [{"x1": 0, "x2": 0}], postprocess=ON, max_retries=2
        )

        assert result.status == "infeasible"
        assert result.solutions == []
        assert len(result.attempts) == 3
        assert backend.solve_calls == 3
        for attempt in result.attempts:
            assert attempt.postprocess is not None
            assert attempt.postprocess.repair_attempted == 1
            assert attempt.postprocess.repair_succeeded == 0
            assert attempt.postprocess.local_search_started == 0
            assert attempt.postprocess.new_candidates == 0
            assert attempt.postprocess.limit_reached == []
        # Diagnostics stay about the solver's own samples (spec §8).
        assert result.infeasibility is not None
        assert "POSTPROCESS_LIMIT_REACHED" not in codes(result.warnings)

    def test_equal_products_keep_the_best_selected_label(self):
        # The feasible (w2, w0) is selected first and local search moves it
        # to (w1, w0); repairing (w1, -) reaches the same assignment later.
        # The first producer's label wins, and the row is counted once.
        problem = assignment_problem()
        _, result = solve_fixed(
            problem, [assignment(2, 0), assignment(1, None)], postprocess=ON
        )

        assert_revalidated(problem, result)
        assert result.solutions[0].variables == assignment(1, 0)
        assert result.solutions[0].source == "local_search"
        stats = result.attempts[0].postprocess
        assert stats.repair_succeeded == 1
        assert stats.new_candidates == 1
        assert stats.feasible_added == 1

    def test_zero_samples_still_report_statistics(self):
        _, result = solve_fixed(assignment_problem(), [], postprocess=ON, max_retries=0)

        assert result.status == "infeasible"
        (attempt,) = result.attempts
        assert attempt.postprocess is not None
        assert attempt.postprocess.model_dump() == {
            "candidates_selected": 0,
            "repair_attempted": 0,
            "repair_succeeded": 0,
            "local_search_started": 0,
            "local_search_improved": 0,
            "new_candidates": 0,
            "feasible_added": 0,
            "limit_reached": [],
            "wall_clock_limit_reached": False,
        }


# --------------------------------------------------------------------------
# Local search (spec §9 item 3)
# --------------------------------------------------------------------------


class TestLocalSearch:
    def test_knapsack_reaches_the_known_optimum(self):
        problem = knapsack_problem()
        _, result = solve_fixed(problem, [knapsack("d")], postprocess=ON)

        assert result.status == "success"
        assert_revalidated(problem, result)
        best = result.solutions[0]
        assert best.ranking_score == brute_force_best(problem) == 17.0
        assert best.variables == knapsack("a", "c")
        assert best.source == "local_search"
        assert [s.source for s in result.solutions] == ["local_search", "solver"]
        stats = result.attempts[0].postprocess
        assert stats.candidates_selected == 1
        assert stats.repair_attempted == 0
        assert stats.local_search_started == 1
        assert stats.local_search_improved == 1
        # Only the end point joins the pool, not the steps on the way.
        assert stats.new_candidates == 1
        assert result.attempts[0].feasible_samples == 1
        assert len(result.solutions) == 2  # may exceed feasible_samples

    def test_a_local_optimum_adds_nothing(self):
        problem = knapsack_problem()
        _, result = solve_fixed(problem, [knapsack("a", "c")], postprocess=ON)

        assert_revalidated(problem, result)
        stats = result.attempts[0].postprocess
        assert stats.local_search_started == 1
        assert stats.local_search_improved == 0
        assert stats.new_candidates == 0
        assert stats.feasible_added == 0
        assert [s.source for s in result.solutions] == ["solver"]
        assert "post-processing" not in result.message

    def test_end_point_the_solver_also_returned_stays_the_solvers(self):
        problem = knapsack_problem()
        _, result = solve_fixed(
            problem, [knapsack("d"), knapsack("a", "c")], postprocess=ON
        )

        assert_revalidated(problem, result)
        stats = result.attempts[0].postprocess
        assert stats.local_search_improved == 1  # d walked to {a, c}
        assert stats.new_candidates == 0  # ... which the solver returned too
        best = result.solutions[0]
        assert best.variables == knapsack("a", "c")
        assert best.source == "solver"
        assert best.energy is not None
        assert best.sample_count == 1

    def test_candidates_limit_how_many_samples_are_selected(self):
        problem = knapsack_problem()
        rows = [knapsack("d"), knapsack("c"), knapsack("b")]
        _, result = solve_fixed(problem, rows, postprocess=ON, postprocess_candidates=1)

        stats = result.attempts[0].postprocess
        # The best by ranking cost is b (value 8); only it is searched.
        assert stats.candidates_selected == 1
        assert stats.local_search_started == 1

    @pytest.mark.parametrize("model_type", ["bqm", "cqm"])
    def test_integer_moves_stay_inside_the_bounds(self, model_type):
        problem = integer_problem()
        rows = [{"x": 3, "y": -2}, {"x": 3, "y": 2}]  # at the bounds; 2nd infeasible
        _, result = solve_fixed(problem, rows, model_type=model_type, postprocess=ON)

        assert result.status == "success"
        assert_revalidated(problem, result)
        for solution in result.solutions:
            assert set(solution.variables) == {"x", "y"}  # no bit, no slack
            assert 0 <= solution.variables["x"] <= 3
            assert -2 <= solution.variables["y"] <= 2
        best = result.solutions[0]
        assert best.variables == {"x": 3, "y": 0}
        assert best.ranking_score == brute_force_best(problem) == 6.0
        assert best.source in PRODUCED_SOURCES

    def test_bqm_and_cqm_paths_agree(self):
        problem = integer_problem()
        rows = [{"x": 3, "y": -2}, {"x": 3, "y": 2}, {"x": 0, "y": 0}]
        outcomes = []
        for model_type in ("bqm", "cqm"):
            _, result = solve_fixed(problem, rows, model_type=model_type, postprocess=ON)
            outcomes.append(
                (
                    [
                        (s.variables, s.source, s.objective_value, s.ranking_score)
                        for s in result.solutions
                    ],
                    result.attempts[-1].postprocess,
                )
            )
        assert outcomes[0] == outcomes[1]

    def test_soft_only_problem_uses_single_variable_moves(self):
        problem = soft_only_problem()
        # No hard constraint: no pair move at all (spec §1). The one soft
        # row over a, b, c is still charged: 2·(3 variables + 3 entries)
        # + 4·(0 pairs + 3 in-row pairs).
        assert scan_cost(problem, 10**6) == 2 * (3 + 3) + 4 * (0 + 3)
        # From all-zero (cost 16): c (cost 1), then a (cost -2), the optimum.
        _, result = solve_fixed(problem, [{"a": 0, "b": 0, "c": 0}], postprocess=ON)

        assert result.status == "success"
        assert_revalidated(problem, result)
        best = result.solutions[0]
        assert best.variables == {"a": 1, "b": 0, "c": 1}
        assert best.ranking_score == brute_force_best(problem) == -2.0
        assert best.source == "local_search"
        stats = result.attempts[0].postprocess
        assert stats.repair_attempted == 0  # V is always 0 without hard rows

    def test_unconstrained_problem(self):
        problem = unconstrained_problem()
        # From (0, 1, 0), value -1: c (1), then a (2.5), then b off (3).
        _, result = solve_fixed(problem, [{"a": 0, "b": 1, "c": 0}], postprocess=ON)

        assert_revalidated(problem, result)
        best = result.solutions[0]
        assert best.variables == {"a": 1, "b": 0, "c": 1}
        assert best.ranking_score == brute_force_best(problem) == 3.0
        assert best.source == "local_search"


# --------------------------------------------------------------------------
# Ceilings (spec §6, §9 items 5 and 7)
# --------------------------------------------------------------------------


def assignment_scan() -> int:
    """Scan cost of the assignment problem: 6 variables, two one-hot rows
    of 3 (6 entries, 6 hard pairs, 6 in-row pairs)."""
    value = scan_cost(assignment_problem(), 10**9)
    assert value == 2 * (6 + 6) + 4 * (6 + 6)
    return value


def assignment_setup() -> int:
    """Per-attempt setup of the assignment problem: 6 pairs, 1 entry per variable."""
    setup, _ = postprocess_costs(assignment_problem(), 10**9)
    assert setup == 6 * (1 + 1)
    return setup


class TestCeilings:
    def test_too_many_candidates_is_refused_before_solving(self):
        backend, result = solve_fixed(
            assignment_problem(),
            [assignment(0, 1)],
            postprocess=ON,
            postprocess_candidates=101,
        )

        assert result.status == "resource_limit_exceeded"
        assert codes(result.errors) == ["POSTPROCESS_LIMIT"]
        assert result.errors[0].path is None  # like TOP_K_LIMIT (spec §10 v3)
        assert "101" in result.errors[0].message
        assert backend.solve_calls == 0
        assert result.attempts == []

    def test_a_scan_over_the_budget_is_refused_before_solving(self):
        backend, result = solve_fixed(
            assignment_problem(),
            [assignment(0, 1)],
            policy={
                "max_postprocess_evaluations": assignment_setup() + assignment_scan() - 1
            },
            postprocess=ON,
        )

        assert result.status == "resource_limit_exceeded"
        assert codes(result.errors) == ["POSTPROCESS_LIMIT"]
        assert backend.solve_calls == 0

    def test_both_ceilings_are_reported_at_once(self):
        _, result = solve_fixed(
            assignment_problem(),
            [assignment(0, 1)],
            policy={"max_postprocess_evaluations": 1, "max_postprocess_candidates": 1},
            postprocess=ON,
            postprocess_candidates=2,
        )
        assert codes(result.errors) == ["POSTPROCESS_LIMIT", "POSTPROCESS_LIMIT"]

    def test_validate_does_not_report_the_ceilings(self):
        # Spec §10 v3: like the other preference ceilings, only solve and
        # recommend refuse; validate stays silent about them.
        _, service = fixed_service([], max_postprocess_candidates=1)
        problem = route(
            assignment_problem(), FIXED_BQM, postprocess=ON, postprocess_candidates=2
        )
        result = service.validate(problem)
        assert "POSTPROCESS_LIMIT" not in codes(result.errors) + codes(result.warnings)

    def test_off_never_checks_the_ceilings(self):
        backend, result = solve_fixed(
            assignment_problem(),
            [assignment(0, 1)],
            policy={"max_postprocess_evaluations": 1, "max_postprocess_candidates": 1},
            postprocess_candidates=101,
        )

        assert result.status == "success"
        assert backend.solve_calls == 1
        assert codes(result.warnings) == ["PARAMETER_IGNORED"]
        assert result.warnings[0].path == "solver.postprocess_candidates"
        assert result.attempts[0].postprocess is None
        assert [s.source for s in result.solutions] == ["solver"]

    def test_budget_used_up_after_a_repair_keeps_its_output(self):
        # The setup and two scans: one repair step, one local-search step,
        # then the budget is gone. The search had already moved, so its end point
        # (still feasible) is kept.
        problem = assignment_problem()
        _, result = solve_fixed(
            problem,
            [assignment(2, None)],
            policy={
                "max_postprocess_evaluations": assignment_setup() + 2 * assignment_scan()
            },
            postprocess=ON,
        )

        assert result.status == "success"
        assert len(result.attempts) == 1
        assert_revalidated(problem, result)
        stats = result.attempts[0].postprocess
        assert stats.limit_reached == ["evaluations"]
        assert stats.repair_succeeded == 1
        assert [s.source for s in result.solutions] == [
            "repaired_local_search",
            "repaired",
        ]
        assert codes(result.warnings).count("POSTPROCESS_LIMIT_REACHED") == 1

    def test_budget_used_up_mid_repair_leaves_the_pool_infeasible(self):
        # One scan per attempt, but the empty assignment needs two repair
        # steps: every attempt stops, nothing is added, the ladder climbs,
        # and the three stops fold into one warning.
        backend, result = solve_fixed(
            assignment_problem(),
            [assignment(None, None)],
            policy={"max_postprocess_evaluations": assignment_setup() + assignment_scan()},
            postprocess=ON,
            max_retries=2,
        )

        assert result.status == "infeasible"
        assert len(result.attempts) == 3 == backend.solve_calls
        for attempt in result.attempts:
            assert attempt.postprocess.limit_reached == ["evaluations"]
            assert attempt.postprocess.repair_succeeded == 0
            assert attempt.postprocess.new_candidates == 0
        reached = [w for w in result.warnings if w.code == "POSTPROCESS_LIMIT_REACHED"]
        assert len(reached) == 1
        for number in (1, 2, 3):
            assert f"attempt {number}" in reached[0].message

    def test_repair_step_cap(self):
        # x >= 50 from x = 0 takes 50 steps of +1; the cap is 4·n = 4.
        problem = wide_integer_problem(
            "minimize",
            {
                "id": "at_least_fifty",
                "type": "hard",
                "terms": [{"variable": "x", "coefficient": 1}],
                "operator": ">=",
                "rhs": 50,
            },
        )
        _, result = solve_fixed(problem, [{"x": 0}], model_type="cqm", postprocess=ON)

        assert result.status == "infeasible"
        stats = result.attempts[0].postprocess
        assert stats.limit_reached == ["steps"]
        assert stats.repair_attempted == 1
        assert stats.repair_succeeded == 0
        assert codes(result.warnings).count("POSTPROCESS_LIMIT_REACHED") == 1

    def test_local_search_step_cap_keeps_the_last_feasible_point(self):
        problem = wide_integer_problem("maximize", None)
        _, result = solve_fixed(problem, [{"x": 0}], model_type="cqm", postprocess=ON)

        assert result.status == "success"
        assert_revalidated(problem, result)
        stats = result.attempts[0].postprocess
        assert stats.limit_reached == ["steps"]
        assert result.solutions[0].variables == {"x": 4}  # 4·n steps of +1
        assert result.solutions[0].source == "local_search"
        assert codes(result.warnings).count("POSTPROCESS_LIMIT_REACHED") == 1

    def test_candidates_must_be_positive(self):
        payload = assignment_problem().model_dump(mode="json")
        payload["solver"] = {"backend": "simulated_annealing", "postprocess_candidates": 0}
        errors = validate_problem(OptimizationProblem.model_validate(payload))
        assert [(e.code, e.path) for e in errors] == [
            ("INVALID_SOLVER_PREFERENCE", "solver.postprocess_candidates")
        ]


class TestScanCost:
    @staticmethod
    def problem(constraints: list[dict]) -> OptimizationProblem:
        return OptimizationProblem.model_validate(
            {
                "name": "bound",
                "variables": [{"name": v} for v in "abcd"],
                "objective": {
                    "direction": "minimize",
                    "linear_terms": [{"variable": "a", "coefficient": 1}],
                },
                "constraints": constraints,
                "solver": {"backend": "exact"},
            }
        )

    @staticmethod
    def hard(cid: str, variables: str, kind: str = "hard", **extra) -> dict:
        return {
            "id": cid,
            "type": kind,
            "terms": [{"variable": v, "coefficient": 1} for v in variables],
            "operator": "<=",
            "rhs": 1,
            **extra,
        }

    # scan cost = 2·(n + entries) + 4·(hard pairs + Σ in-row pairs)

    def test_singles_only_without_constraints(self):
        assert scan_cost(self.problem([]), 100) == 2 * 4

    def test_one_hard_row(self):
        # {a, b, c}: 3 entries, pairs ab ac bc, 3 in-row pairs.
        assert scan_cost(self.problem([self.hard("h", "abc")]), 100) == (
            2 * (4 + 3) + 4 * (3 + 3)
        )

    def test_overlapping_constraints_count_a_shared_pair_once(self):
        # {a, b, c} and {b, c, d} share b and c: ab ac bc + bd cd = 5 distinct
        # pairs, while the in-row count (the shared-row work) is 3 + 3.
        problem = self.problem([self.hard("h1", "abc"), self.hard("h2", "bcd")])
        assert scan_cost(problem, 10**6) == 2 * (4 + 6) + 4 * (5 + 6)

    def test_cap_boundary(self):
        problem = self.problem([self.hard("h1", "abc"), self.hard("h2", "bcd")])
        assert scan_cost(problem, 64) == 64
        assert scan_cost(problem, 63) is None
        assert scan_cost(problem, 7) is None  # below 2n already

    def test_setup_is_charged_once_on_top_of_a_scan(self):
        # Pairs ab ac bc bd cd; entries per variable a 1, b 2, c 2, d 1:
        # setup = (1+2) + (1+2) + (2+2) + (2+1) + (2+1) = 16, scan 64.
        problem = self.problem([self.hard("h1", "abc"), self.hard("h2", "bcd")])
        assert postprocess_costs(problem, 10**6) == (16, 64)
        assert postprocess_costs(problem, 80) == (16, 64)
        assert postprocess_costs(problem, 79) is None
        assert scan_cost(problem, 79) == 64  # the scan alone still fits

    def test_dense_rows_are_charged_for_their_shared_work(self):
        # Review follow-up: many dense rows over the same variables add few
        # distinct pairs but a lot of shared-row work, and are charged for it.
        rows = [self.hard(f"h{r}", "abcd") for r in range(10)]
        assert scan_cost(self.problem(rows), 10**6) == 2 * (4 + 40) + 4 * (6 + 60)

    def test_soft_constraints_and_cancelled_terms_add_no_pair(self):
        cancelled = {
            "id": "cancelled",
            "type": "hard",
            "terms": [
                {"variable": "a", "coefficient": 1},
                {"variable": "a", "coefficient": -1},
                {"variable": "b", "coefficient": 1},
            ],
            "operator": "<=",
            "rhs": 1,
        }
        problem = self.problem([self.hard("s", "abcd", kind="soft", weight=1), cancelled])
        # No hard pair; the soft row (4 entries, 6 in-row pairs) and the
        # surviving entry b are still charged.
        assert scan_cost(problem, 100) == 2 * (4 + 5) + 4 * (0 + 6)


# --------------------------------------------------------------------------
# Exhaustive backends (spec §3, §9 item 6)
# --------------------------------------------------------------------------


def exhaustive_services():
    defaults = SolverRegistry.default()
    backends = {name: defaults.get(name) for name in defaults.names()}
    backends[FAKE_LOCAL_CQM_NAME] = FakeLocalCQMBackend()
    return OptimizationService(registry=SolverRegistry(backends))


class TestExhaustive:
    @pytest.mark.parametrize("backend", ["exact", FAKE_LOCAL_CQM_NAME])
    def test_ignored_with_one_warning_and_identical_results(self, backend):
        service = exhaustive_services()
        problem = knapsack_problem()
        off = service.solve(route(problem, backend))
        # Over both ceilings, yet not refused: nothing runs, nothing to check.
        on = service.solve(
            route(problem, backend, postprocess=ON, postprocess_candidates=101)
        )

        assert on.status == off.status == "success"
        ignored = [w for w in on.warnings if w.code == "PARAMETER_IGNORED"]
        assert [w.path for w in ignored] == ["solver.postprocess"]
        assert "POSTPROCESS_LIMIT" not in codes(on.errors) + codes(on.warnings)
        assert all(attempt.postprocess is None for attempt in on.attempts)
        assert all(attempt.postprocess_ms is None for attempt in on.attempts)
        assert [s.model_dump() for s in on.solutions] == [
            s.model_dump() for s in off.solutions
        ]
        assert on.optimality_proven is off.optimality_proven is True


# --------------------------------------------------------------------------
# Reproducibility (spec §7, §9 item 8)
# --------------------------------------------------------------------------


class TestReproducibility:
    def test_same_input_same_output_with_fixed_samples(self):
        problem = assignment_problem()
        rows = [assignment(None, None), assignment(2, None), assignment(0, 2)]
        dumps = [
            strip_timings(
                solve_fixed(problem, rows, postprocess=ON)[1].model_dump(mode="json")
            )
            for _ in range(2)
        ]
        assert dumps[0] == dumps[1]
        assert dumps[0]["attempts"][0]["postprocess"]["new_candidates"] > 0

    def test_same_seed_same_output_on_simulated_annealing(self):
        payload = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
        payload["solver"] = {
            "backend": "simulated_annealing",
            "seed": 7,
            "num_reads": 20,
            "num_sweeps": 50,
            "postprocess": ON,
        }
        problem = OptimizationProblem.model_validate(payload)
        service = OptimizationService()
        first = service.solve(problem)
        second = service.solve(problem)

        assert first.status == "success"
        assert first.attempts[0].postprocess is not None
        assert_revalidated(problem, first)
        assert strip_timings(first.model_dump(mode="json")) == strip_timings(
            second.model_dump(mode="json")
        )


# --------------------------------------------------------------------------
# Capabilities and recommend (spec §6, §10 v3)
# --------------------------------------------------------------------------

POSTPROCESS_KEYS = ("max_postprocess_candidates", "max_postprocess_evaluations")


class TestCapabilitiesAndRecommend:
    def test_ceilings_are_listed_for_non_exhaustive_backends_only(self):
        defaults = SolverRegistry.default()
        backends = {name: defaults.get(name) for name in defaults.names()}
        backends[FAKE_LOCAL_CQM_NAME] = FakeLocalCQMBackend()
        backends[FIXED_BQM] = FixedSamplesBackend([])
        registry = SolverRegistry(backends)
        policy = ExecutionPolicy(max_postprocess_candidates=7, max_postprocess_evaluations=99)

        view = build_capabilities(registry, policy)

        exhaustive_seen = non_exhaustive_seen = False
        for entry in view.backends:
            exhaustive = registry.get(entry.name).capabilities.exhaustive
            if exhaustive:
                exhaustive_seen = True
                assert not set(POSTPROCESS_KEYS) & set(entry.limits), entry.name
            else:
                non_exhaustive_seen = True
                assert entry.limits["max_postprocess_candidates"] == 7, entry.name
                assert entry.limits["max_postprocess_evaluations"] == 99, entry.name
                assert list(entry.limits)[-2:] == list(POSTPROCESS_KEYS)
        assert exhaustive_seen and non_exhaustive_seen

    def test_recommend_blocks_on_the_candidate_ceiling(self):
        payload = knapsack_problem().model_dump(mode="json")
        payload["solver"] = {
            "backend": "simulated_annealing",
            "postprocess": ON,
            "postprocess_candidates": 101,
        }
        result = OptimizationService().recommend(OptimizationProblem.model_validate(payload))
        by_name = {entry.backend: entry for entry in result.recommendations}

        assert "POSTPROCESS_LIMIT" in codes(by_name["simulated_annealing"].blocking)
        # Exhaustive: post-processing would not run, so it blocks nothing.
        assert "POSTPROCESS_LIMIT" not in codes(by_name["exact"].blocking)

    def test_recommend_blocks_on_the_evaluation_budget(self):
        payload = knapsack_problem().model_dump(mode="json")
        payload["solver"] = {"backend": "simulated_annealing", "postprocess": ON}
        problem = OptimizationProblem.model_validate(payload)
        service = OptimizationService(policy=ExecutionPolicy(max_postprocess_evaluations=8))

        result = service.recommend(problem)
        by_name = {entry.backend: entry for entry in result.recommendations}

        assert "POSTPROCESS_LIMIT" in codes(by_name["simulated_annealing"].blocking)
        assert "POSTPROCESS_LIMIT" not in codes(by_name["exact"].blocking)

    def test_recommend_ignores_the_ceilings_while_off(self):
        payload = knapsack_problem().model_dump(mode="json")
        payload["solver"] = {"backend": "simulated_annealing", "postprocess_candidates": 101}
        result = OptimizationService().recommend(OptimizationProblem.model_validate(payload))
        for entry in result.recommendations:
            assert "POSTPROCESS_LIMIT" not in codes(entry.blocking), entry.backend
