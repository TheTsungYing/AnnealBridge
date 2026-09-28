"""Record the ``version 1.0`` / ``1.1`` compatibility golden (commit ``a377f35``).

Before the next batches touch the validator, the solution validator,
post-processing, routing and the compilers, the exact output the code at
``a377f35`` produces for a *fixed* list of ``1.0`` and ``1.1`` problems is
recorded to ``compat_a377f35.json``; ``tests/unit/test_golden_compat.py``
asserts that every later commit still produces it bit for bit. The list is
spelled out here on purpose (no discovery):

* every ``1.0`` problem of ``record_phase3a_golden.PROBLEMS`` (its builders
  are imported and reused, not copied), plus ``examples/shift_scheduling.json``;
* ``examples/integer_knapsack.json``, fifteen inline integer problems rebuilt
  from ``tests/unit/test_bqm_compiler_integer.py``,
  ``test_cqm_compiler_integer.py`` and ``test_integer_encoding.py`` (rebuilt
  here, never imported, so renaming a test cannot break the golden), and the
  assignment / knapsack / tsp examples relabelled ``"version": "1.1"``;
* three problems written for this golden (two ``1.0``, one ``1.1``) whose
  hard constraints are each satisfiable alone but not together, so they pass
  the validator's ``TRIVIALLY_INFEASIBLE`` check and the exact solve ends
  ``infeasible`` with its diagnostics (see :data:`INFEASIBLE_PROBLEMS`).

Each problem records these sections (see :data:`SECTIONS`):

``snapshot``
    ``record_phase3a_golden.snapshot_problem`` unchanged: both compiled
    models, the estimates, the penalty scale and both full validations.
``cqm_domains``
    What ``snapshot`` leaves out of the compiled CQM: every variable, in
    ``cqm.variables`` order, with its vartype name and its lower / upper
    bound as floats, and for every constraint (in label order) whether its
    ``lhs.is_discrete()`` (one-hot marking); the compiler marks none today.
``samples``
    ``validate_solution`` (the full ``ValidationResult``) on the first three
    fixed rows and ``validate_batch`` on all of them. The rows are the
    all-lower-bound row, the all-upper-bound row and 16 rows drawn uniformly
    inside the bounds by ``numpy.random.default_rng(20260925)`` (a fresh
    generator per problem, so adding a problem never shifts another's rows).
    The rows are stored in the golden as ``sample_rows`` and the test replays
    *those*, so a change in numpy's random stream between versions (the CI
    lowest-direct job) can never change the inputs.
``recommend``
    ``routing.recommend`` over a registry of the four local backends only
    (``exact``, ``simulated_annealing``, ``tabu``, ``simulated_bifurcation``
    on the CPU), a default ``ExecutionPolicy`` and both compilers. No remote
    backend is registered, so neither cloud credentials nor an installed
    ``dwave-system`` can change the ranking, ``usable`` or ``blocking``;
    every other field is computed from the problem, the backends' fixed
    declarations and the policy.
``postprocess_costs``
    ``postprocess.postprocess_costs(problem, 10**12)`` as a list (or null).
``exact_solve``
    ``OptimizationService.solve`` on the ``exact`` backend (same local
    registry, ``top_k`` the problem's own) for problems whose compiled BQM
    has at most 20 variables; null for the others. Only a whitelist of
    fields is kept, because a golden must hold nothing that changes between
    runs, machines or dependency versions:

    - kept: ``status``, ``backend``, ``objective_direction``, the two
      ``*_proven`` flags, the errors' and warnings' ``(code, path)``, the
      metadata's ``backend`` / ``remote`` / ``model_type``, every attempt's
      ``attempt``, ``penalty``, ``samples_received``, ``unique_samples``
      (candidates), ``feasible_samples``, ``compiled_variables`` and
      ``compiled_interactions``, and every solution's ``rank``,
      ``variables``, ``objective_value``, ``soft_violation_score``,
      ``ranking_score``, ``sample_count``, ``source``,
      ``hard_constraints_satisfied`` and ``constraint_evaluations`` -- all
      recomputed by this package from the original problem, the compiled
      model and a fully deterministic ranking key;
    - dropped: ``elapsed_ms`` and every attempt's ``*_ms`` (wall clock),
      ``annealbridge_version`` (the installed package), ``metadata`` apart
      from the three fields above (``timing_us`` and the vendor-side
      fields), ``wall_clock_limit_reached`` (time-driven by definition), each
      solution's ``energy`` (dimod computes it, so the last bits may follow
      the dimod version; it never feeds feasibility or ranking, and the
      compiled model it comes from is pinned by ``snapshot``), the
      warnings' and errors' ``message`` / ``recommended_action`` (text; the
      same warnings are recorded in full under ``recommend``) and the result
      ``message`` (a summary of fields already kept).

    ``infeasibility`` is kept only where it cannot depend on the order in
    which dimod's ``ExactSolver`` lists its rows. On the exact backend the
    deduplicated candidates are every business assignment inside the bounds,
    whatever the row order, so ``hard_violation_rates`` (per hard
    constraint, in problem order: violated count, candidate count and their
    ratio) and the closest candidate's ``hard_violation_total`` (the minimum
    over that set) are order-free. *Which* candidate is the closest is not:
    on a tie the service takes the first one in row order. So its assignment
    and evaluations are recorded only when an independent enumeration of the
    full bound product (``validate_batch``) finds exactly one assignment at
    that minimum (``closest_is_unique``); with a tie they are null, never the
    tie-break's pick.

A separate part records the schema errors ``interfaces.problem_input.
parse_problem`` reports for a few malformed documents, as ``(code, path)``
pairs only: the message is pydantic's wording and changes with its version.
Two of them are expected to change in the next batch ("1.2" becomes a valid
version and ``cardinality_constraints`` a valid field), so they live under
``format_errors_expected_to_change`` and are recorded but not asserted.

Every value is a plain ``list`` / ``dict`` / ``str`` / ``float`` / ``int`` /
``bool`` / ``None`` so a ``json`` round-trip is exact. Re-record (only ever
intentionally, from the ``a377f35`` code) with::

    .venv/Scripts/python.exe tests/golden/record_compat_golden.py
"""

from __future__ import annotations

import itertools
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.interfaces.problem_input import parse_problem
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy, OptimizationService, recommend
from annealbridge.orchestration.postprocess import postprocess_costs
from annealbridge.penalty.strategy import ScaledPenaltyStrategy
from annealbridge.solvers import (
    ExactSolverBackend,
    SimulatedAnnealingBackend,
    SimulatedBifurcationBackend,
    SolverRegistry,
    TabuBackend,
)
from annealbridge.validation import validate_batch, validate_solution

# Run as a script only this file's directory is on ``sys.path``; the Phase 3a
# builders are imported as ``tests.golden...`` exactly as the test suite
# imports them, so the repository root has to be importable too.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.golden.record_phase3a_golden import (
    EXAMPLES_DIR,
    hard,
    lin,
    make_problem,
    quad,
    snapshot_problem,
    soft,
)
from tests.golden.record_phase3a_golden import PROBLEMS as PHASE3A_PROBLEMS

GOLDEN_PATH = Path(__file__).resolve().parent / "compat_a377f35.json"
RECORDED_FROM = "a377f35"

SAMPLE_SEED = 20260925
RANDOM_ROWS = 16
FULL_VALIDATE_ROWS = 3
POSTPROCESS_CAP = 10**12
EXACT_MAX_BQM_VARIABLES = 20


# --------------------------------------------------------------------------
# Problem builders. Each entry of PROBLEMS is a zero-argument callable that
# builds one problem; the key is its stable name.
# --------------------------------------------------------------------------


def _load_example(name: str, *, version: str | None = None) -> OptimizationProblem:
    data = json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))
    if version is not None:
        data["version"] = version
    return OptimizationProblem.model_validate(data)


def example_shift_scheduling() -> OptimizationProblem:
    return _load_example("shift_scheduling.json")


def example_integer_knapsack() -> OptimizationProblem:
    return _load_example("integer_knapsack.json")


def example_assignment_v1_1() -> OptimizationProblem:
    return _load_example("assignment.json", version="1.1")


def example_knapsack_v1_1() -> OptimizationProblem:
    return _load_example("knapsack.json", version="1.1")


def example_tsp_v1_1() -> OptimizationProblem:
    return _load_example("tsp.json", version="1.1")


# --- 1.1 builders: the shape of the integer test modules' make_problem ---


def integer(name: str, lower: int, upper: int) -> dict:
    return {"name": name, "type": "integer", "lower_bound": lower, "upper_bound": upper}


def binary(name: str) -> dict:
    return {"name": name}


def make_integer_problem(
    variables: list[dict],
    *,
    name: str,
    direction: str = "minimize",
    linear: list[dict] | None = None,
    quadratic: list[dict] | None = None,
    constant: float = 0.0,
    constraints: list[dict] | None = None,
) -> OptimizationProblem:
    return OptimizationProblem.model_validate(
        {
            "version": "1.1",
            "name": name,
            "variables": variables,
            "objective": {
                "direction": direction,
                "linear_terms": linear or [],
                "quadratic_terms": quadratic or [],
                "constant": constant,
            },
            "constraints": constraints or [],
        }
    )


# BEGIN INTEGER PROBLEMS
# Rebuilt by hand from the integer test modules (never imported). Each comment
# names the builder or test the problem was copied from, and ``name`` is the
# problem name that module gives it.
_BQM_NAME = "integer compiler test"  # test_bqm_compiler_integer.make_problem
_CQM_NAME = "cqm integer compiler test"  # test_cqm_compiler_integer.make_problem


def _bqm_squared_objective(direction: str) -> OptimizationProblem:
    # test_bqm_compiler_integer.py::squared_objective_problem(direction)
    return make_integer_problem(
        [integer("x", -2, 3), integer("y", 0, 2), binary("b")],
        name=_BQM_NAME,
        direction=direction,
        linear=[lin("x", 1)],
        quadratic=[quad("x", "x", 2), quad("x", "y", -3), quad("x", "b", 4)],
        constraints=[
            hard("eq", "==", 2, [lin("x", 1), lin("y", 1)]),
            hard("cap", "<=", 2, [lin("x", 1), lin("y", -1)]),
        ],
    )


INTEGER_PROBLEMS: dict[str, Callable[[], OptimizationProblem]] = {
    # --- tests/unit/test_bqm_compiler_integer.py ---
    # negative_bounds_problem(): negative lower bounds, hard <=, mixed.
    "int_bqm_negative_bounds": lambda: make_integer_problem(
        [integer("x", -3, 2), integer("y", -1, 1), binary("b")],
        name=_BQM_NAME,
        linear=[lin("x", 1), lin("y", 2), lin("b", 3)],
        constraints=[hard("cap", "<=", 1, [lin("x", 1), lin("y", 1)])],
    ),
    # squared_objective_problem("minimize" / "maximize"): x*x, x*y, x*b,
    # hard == (no slack) and hard <= (slack) over integers.
    "int_bqm_squared_objective_minimize": lambda: _bqm_squared_objective("minimize"),
    "int_bqm_squared_objective_maximize": lambda: _bqm_squared_objective("maximize"),
    # hand_written_problems() "negative lower bound + equality".
    "int_bqm_negative_lower_bound_equality": lambda: make_integer_problem(
        [integer("x", -4, 2), integer("y", -1, 5), binary("b")],
        name=_BQM_NAME,
        linear=[lin("x", 1), lin("y", -2)],
        constraints=[
            hard("eq", "==", 0, [lin("x", 1), lin("y", 1), lin("b", 3)])
        ],
    ),
    # hand_written_problems() "greater-equal with slack".
    "int_bqm_greater_equal_with_slack": lambda: make_integer_problem(
        [integer("x", -2, 4), binary("b")],
        name=_BQM_NAME,
        linear=[lin("x", 1)],
        constraints=[hard("cover", ">=", -1, [lin("x", 1), lin("b", 2)])],
    ),
    # hand_written_problems() "soft inequality".
    "int_bqm_soft_inequality": lambda: make_integer_problem(
        [integer("x", -3, 3), binary("b")],
        name=_BQM_NAME,
        direction="maximize",
        linear=[lin("x", 4), lin("b", 1)],
        constraints=[soft("prefer", "<=", 1, [lin("x", 1), lin("b", 1)], 2.5)],
    ),
    # hand_written_problems() "mixed with quadratic and every operator".
    "int_bqm_mixed_every_operator": lambda: make_integer_problem(
        [integer("x", -2, 2), integer("y", 0, 6), binary("b")],
        name=_BQM_NAME,
        direction="maximize",
        linear=[lin("x", 1), lin("y", 1), lin("b", 1)],
        quadratic=[quad("x", "x", 1), quad("x", "y", -1), quad("y", "b", 2)],
        constraints=[
            hard("eq", "==", 3, [lin("y", 1), lin("b", 1)]),
            hard("cap", "<=", 4, [lin("x", 2), lin("y", 1)]),
            hard("cover", ">=", 0, [lin("x", 1), lin("y", 1)]),
            soft("soft_cap", "<=", 3, [lin("y", 1)], 1.0),
        ],
    ),
    # --- tests/unit/test_cqm_compiler_integer.py ---
    # hand_written_soft_problems() "equality in objective form".
    "int_cqm_soft_equality_objective_form": lambda: make_integer_problem(
        [integer("x", -3, 3), binary("b")],
        name=_CQM_NAME,
        direction="maximize",
        linear=[lin("x", 4), lin("b", 1)],
        quadratic=[quad("x", "x", 1)],
        constraints=[soft("eq", "==", 1, [lin("x", 1), lin("b", 2)], 0.5)],
    ),
    # hand_written_soft_problems() "less-equal with integer slack".
    "int_cqm_soft_less_equal_with_hard": lambda: make_integer_problem(
        [integer("x", 0, 5), integer("y", -2, 2)],
        name=_CQM_NAME,
        linear=[lin("x", -1), lin("y", -3)],
        constraints=[
            soft("cap", "<=", 3, [lin("x", 1), lin("y", 2)], 2.5),
            hard("hard", ">=", 0, [lin("x", 1), lin("y", 1)]),
        ],
    ),
    # hand_written_soft_problems() "trivially infeasible soft (S < 0, clamped)".
    "int_cqm_soft_clamped": lambda: make_integer_problem(
        [integer("x", -3, 3), binary("b")],
        name=_CQM_NAME,
        linear=[lin("x", 1), lin("b", -1)],
        constraints=[
            soft("clamp", ">=", 5, [lin("x", 1)], 1.0),
            soft("clamp_le", "<=", -4, [lin("x", 1), lin("b", 1)], 2.0),
        ],
    ),
    # hand_written_soft_problems() "everything at once, with a binary-only
    # soft beside".
    "int_cqm_everything_at_once": lambda: make_integer_problem(
        [integer("x", -2, 2), integer("y", 0, 3), binary("b")],
        name=_CQM_NAME,
        direction="maximize",
        linear=[lin("x", 1), lin("y", 1), lin("b", 1)],
        quadratic=[quad("x", "x", 1), quad("x", "y", -1), quad("y", "b", 2)],
        constraints=[
            soft("eq", "==", 3, [lin("y", 1), lin("b", 1)], 1.0),
            soft("cap", "<=", 1, [lin("x", 2), lin("y", 1)], 2.0),
            soft("cover", ">=", 0, [lin("x", 1), lin("y", 1)], 0.5),
            soft("bin", ">=", 1, [lin("b", 1)], 3.0),
            hard("hard", "<=", 4, [lin("x", 1), lin("y", 1), lin("b", 1)]),
        ],
    ),
    # hard_passthrough_problem(): one hard constraint per operator over
    # integers, with accumulated and cancelling terms.
    "int_cqm_hard_passthrough": lambda: make_integer_problem(
        [integer("x", -3, 4), integer("y", -2, 2), integer("z", 0, 3), binary("b")],
        name="hard constraints over integers",
        linear=[lin("x", 1), lin("y", -2), lin("z", 1), lin("b", -1)],
        quadratic=[quad("x", "x", 1)],
        constraints=[
            hard(
                "eq",
                "==",
                3,
                [lin("x", 1), lin("x", 2), lin("y", -1), lin("z", 1), lin("z", -1)],
            ),
            hard("le", "<=", 5, [lin("x", 1), lin("y", 2), lin("b", 1), lin("b", -1)]),
            hard("ge", ">=", -1, [lin("z", 2), lin("y", 1), lin("z", -1)]),
        ],
    ),
    # mixed_soft_problem(): an all-binary soft beside an integer soft.
    "int_cqm_mixed_soft": lambda: make_integer_problem(
        [integer("x", 0, 3), binary("b1"), binary("b2")],
        name="mixed native and objective-form softs",
        direction="maximize",
        linear=[lin("x", 2), lin("b1", 3), lin("b2", 1)],
        constraints=[
            hard("h", "<=", 4, [lin("x", 1), lin("b2", 1)]),
            soft("bin_only", "<=", 1, [lin("b1", 1), lin("b2", 1)], 2.0),
            soft("int_soft", "<=", 2, [lin("x", 1), lin("b1", 1)], 1.5),
        ],
    ),
    # negative_bounds_problem(): negative lower bounds, x*x, a hard cap and an
    # integer soft floor.
    "int_cqm_negative_bounds": lambda: make_integer_problem(
        [integer("x", -4, 2), integer("y", -1, 5), binary("b")],
        name="negative lower bounds",
        linear=[lin("x", -1), lin("y", 2), lin("b", -3)],
        quadratic=[quad("x", "x", 1)],
        constraints=[
            hard("cap", "<=", 3, [lin("x", 1), lin("y", 1)]),
            soft("floor", ">=", 0, [lin("x", 1), lin("y", 1)], 2.0),
        ],
    ),
    # --- tests/unit/test_integer_encoding.py ---
    # TestEncodeIntegerVariables.test_form_keys_follow_the_declaration_order:
    # integers and binaries interleaved, built by that module's make_problem.
    "int_enc_declaration_order": lambda: make_integer_problem(
        [integer("z", -3, 2), binary("a"), integer("m", 0, 5), binary("b")],
        name="integer-encoding",
        linear=[lin("z", 1.0)],
    ),
}
# END INTEGER PROBLEMS


# BEGIN INFEASIBLE PROBLEMS
# Written for this golden. Every hard constraint can be satisfied on its own,
# so none is TRIVIALLY_INFEASIBLE, but no assignment satisfies all of them:
# the exact solve proves the problem infeasible and returns its diagnostics.
INFEASIBLE_PROBLEMS: dict[str, Callable[[], OptimizationProblem]] = {
    # 1.0: x + y == 1 against x + y == 0, plus x - y == 0 so the smallest
    # total hard violation (1, at x = y = 0) is reached by one assignment
    # only, and a soft constraint that assignment violates, so the closest
    # candidate's evaluations carry a soft row with a non-zero penalty.
    "infeasible_equalities_with_soft": lambda: make_problem(
        variables=("x", "y"),
        linear=[lin("x", 1), lin("y", 2)],
        constraints=[
            hard("one", "==", 1, [lin("x", 1), lin("y", 1)]),
            hard("none", "==", 0, [lin("x", 1), lin("y", 1)]),
            hard("same", "==", 0, [lin("x", 1), lin("y", -1)]),
            soft("prefer_some", ">=", 1, [lin("x", 1), lin("y", 1)], 2.5),
        ],
    ),
    # 1.0: a + b + c >= 2 against a + b + c <= 1. The smallest total (1) is
    # shared by the six assignments with one or two ones, so the closest
    # candidate is a tie and is not recorded.
    "infeasible_cardinality_window": lambda: make_problem(
        direction="maximize",
        variables=("a", "b", "c"),
        linear=[lin("a", 1), lin("b", 2), lin("c", 3)],
        constraints=[
            hard("at_least_two", ">=", 2, [lin("a", 1), lin("b", 1), lin("c", 1)]),
            hard("at_most_one", "<=", 1, [lin("a", 1), lin("b", 1), lin("c", 1)]),
        ],
    ),
    # 1.1: x + b >= 3 needs x >= 2, 2x - b <= -3 needs x <= -1; y - x == 2
    # ties y to x. The smallest total (3) is reached only at x = -1, y = 1,
    # b = 1; the soft floor on y is violated there.
    "infeasible_integer_exclusive": lambda: make_integer_problem(
        [integer("x", -2, 3), integer("y", 0, 2), binary("b")],
        name="integer mutually exclusive constraints",
        linear=[lin("x", 1), lin("y", 1), lin("b", -1)],
        constraints=[
            hard("low", ">=", 3, [lin("x", 1), lin("b", 1)]),
            hard("high", "<=", -3, [lin("x", 2), lin("b", -1)]),
            hard("link", "==", 2, [lin("y", 1), lin("x", -1)]),
            soft("floor_y", ">=", 2, [lin("y", 1)], 1.5),
        ],
    ),
}
# END INFEASIBLE PROBLEMS


def _merge(
    *parts: dict[str, Callable[[], OptimizationProblem]],
) -> dict[str, Callable[[], OptimizationProblem]]:
    merged: dict[str, Callable[[], OptimizationProblem]] = {}
    for part in parts:
        for name, build in part.items():
            if name in merged:
                raise ValueError(f"duplicate golden problem name {name!r}")
            merged[name] = build
    return merged


PROBLEMS: dict[str, Callable[[], OptimizationProblem]] = _merge(
    # version 1.0
    PHASE3A_PROBLEMS,
    {"example_shift_scheduling": example_shift_scheduling},
    # version 1.1
    {
        "example_integer_knapsack": example_integer_knapsack,
        "example_assignment_v1_1": example_assignment_v1_1,
        "example_knapsack_v1_1": example_knapsack_v1_1,
        "example_tsp_v1_1": example_tsp_v1_1,
    },
    INTEGER_PROBLEMS,
    # 1.0 and 1.1, infeasible as a whole
    INFEASIBLE_PROBLEMS,
)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def sample_rows(problem: OptimizationProblem) -> list[list[int]]:
    """All lower bounds, all upper bounds, then 16 uniform rows in the bounds.

    Only the recorder calls this; the test replays the rows stored in the
    golden (see the module docstring).
    """
    bounds = [variable.bounds() for variable in problem.variables]
    lower = [low for low, _ in bounds]
    upper = [high for _, high in bounds]
    rng = np.random.default_rng(SAMPLE_SEED)
    drawn = rng.integers(
        np.asarray(lower, dtype=np.int64),
        np.asarray(upper, dtype=np.int64),
        size=(RANDOM_ROWS, len(bounds)),
        endpoint=True,
        dtype=np.int64,
    )
    return [lower, upper, *([int(v) for v in row] for row in drawn.tolist())]


def _local_registry() -> SolverRegistry:
    backends = (
        ExactSolverBackend(),
        SimulatedAnnealingBackend(workers=1),
        TabuBackend(workers=1),
        SimulatedBifurcationBackend(device="cpu"),
    )
    return SolverRegistry({backend.name: backend for backend in backends})


def section_snapshot(problem: OptimizationProblem, rows: list[list[int]]) -> dict:
    return snapshot_problem(problem)


def section_cqm_domains(problem: OptimizationProblem, rows: list[list[int]]) -> dict:
    cqm = CQMCompiler().compile(problem, None).model
    return {
        "variables": [
            {
                "name": str(variable),
                "vartype": cqm.vartype(variable).name,
                "lower_bound": float(cqm.lower_bound(variable)),
                "upper_bound": float(cqm.upper_bound(variable)),
            }
            for variable in cqm.variables
        ],
        "constraints": [
            {
                "label": str(label),
                "is_discrete": bool(cqm.constraints[label].lhs.is_discrete()),
            }
            for label in cqm.constraint_labels
        ],
    }


def section_samples(problem: OptimizationProblem, rows: list[list[int]]) -> dict:
    names = [variable.name for variable in problem.variables]
    batch = validate_batch(problem, names, np.asarray(rows, dtype=np.int64))
    return {
        "variables": names,
        "validate": [
            validate_solution(problem, dict(zip(names, row))).model_dump(mode="json")
            for row in rows[:FULL_VALIDATE_ROWS]
        ],
        "validate_batch": {
            "feasible": batch.feasible.tolist(),
            "soft_violation_score": batch.soft_violation_score.tolist(),
            "hard_violation_total": batch.hard_violation_total.tolist(),
            "hard_constraint_ids": list(batch.hard_constraint_ids),
            "hard_violated_counts": batch.hard_violated_counts.tolist(),
        },
    }


def section_recommend(problem: OptimizationProblem, rows: list[list[int]]) -> dict:
    result = recommend(
        problem,
        _local_registry(),
        ExecutionPolicy(),
        {"bqm": BQMCompiler(), "cqm": CQMCompiler()},
    )
    return result.model_dump(mode="json")


def section_postprocess_costs(
    problem: OptimizationProblem, rows: list[list[int]]
) -> list[int] | None:
    costs = postprocess_costs(problem, POSTPROCESS_CAP)
    return None if costs is None else [int(cost) for cost in costs]


_SOLUTION_FIELDS = (
    "rank",
    "variables",
    "objective_value",
    "soft_violation_score",
    "ranking_score",
    "sample_count",
    "source",
    "hard_constraints_satisfied",
)
_EVALUATION_FIELDS = (
    "constraint_id",
    "constraint_type",
    "satisfied",
    "actual_value",
    "operator",
    "expected_value",
    "violation_amount",
    "weighted_penalty",
)
_ATTEMPT_FIELDS = (
    "attempt",
    "penalty",
    "samples_received",
    "unique_samples",
    "feasible_samples",
    "compiled_variables",
    "compiled_interactions",
)
_METADATA_FIELDS = ("backend", "remote", "model_type")
_RATE_FIELDS = ("constraint_id", "violated_candidates", "candidates", "violated_fraction")


def _pick(model: Any, fields: tuple[str, ...]) -> dict:
    return {field: getattr(model, field) for field in fields}


def _code_path(entries: list) -> list[list[str | None]]:
    return [[entry.code, entry.path] for entry in entries]


def _closest_is_unique(problem: OptimizationProblem) -> bool:
    """Whether one business assignment alone has the smallest hard violation.

    Enumerates the full bound product -- the exact backend's deduplicated
    candidate set, in an order of our own -- so the answer never depends on
    the order dimod lists its rows in.
    """
    names = [variable.name for variable in problem.variables]
    ranges = [
        range(low, high + 1)
        for low, high in (variable.bounds() for variable in problem.variables)
    ]
    grid = np.asarray(list(itertools.product(*ranges)), dtype=np.int64)
    totals = validate_batch(problem, names, grid).hard_violation_total
    return int(np.count_nonzero(totals == totals.min())) == 1


def _infeasibility(problem: OptimizationProblem, infeasibility: Any) -> dict:
    """The order-free part of the diagnostics (see the module docstring)."""
    closest = infeasibility.closest_candidate
    unique = _closest_is_unique(problem)
    return {
        "closest_hard_violation_total": closest.hard_violation_total,
        "closest_is_unique": unique,
        "closest_candidate": (
            {
                "variables": closest.variables,
                "constraint_evaluations": [
                    _pick(evaluation, _EVALUATION_FIELDS)
                    for evaluation in closest.constraint_evaluations
                ],
            }
            if unique
            else None
        ),
        "hard_violation_rates": [
            _pick(rate, _RATE_FIELDS) for rate in infeasibility.hard_violation_rates
        ],
    }


def section_exact_solve(
    problem: OptimizationProblem, rows: list[list[int]]
) -> dict | None:
    """The whitelisted fields of an exact solve (see the module docstring)."""
    penalty = ScaledPenaltyStrategy().initial_penalty(problem)
    if BQMCompiler().compile(problem, penalty).num_variables > EXACT_MAX_BQM_VARIABLES:
        return None
    on_exact = problem.model_copy(
        update={"solver": problem.solver.model_copy(update={"backend": "exact"})}
    )
    service = OptimizationService(registry=_local_registry(), policy=ExecutionPolicy())
    result = service.solve(on_exact)
    return {
        "status": result.status,
        "backend": result.backend,
        "objective_direction": result.objective_direction,
        "infeasibility_proven": result.infeasibility_proven,
        "optimality_proven": result.optimality_proven,
        "errors": _code_path(result.errors),
        "warnings": _code_path(result.warnings),
        "metadata": (
            None if result.metadata is None else _pick(result.metadata, _METADATA_FIELDS)
        ),
        "attempts": [_pick(attempt, _ATTEMPT_FIELDS) for attempt in result.attempts],
        "solutions": [
            {
                **_pick(solution, _SOLUTION_FIELDS),
                "constraint_evaluations": [
                    _pick(evaluation, _EVALUATION_FIELDS)
                    for evaluation in solution.constraint_evaluations
                ],
            }
            for solution in result.solutions
        ],
        "infeasibility": (
            None
            if result.infeasibility is None
            else _infeasibility(problem, result.infeasibility)
        ),
    }


# Section name → ``(problem, sample rows) -> JSON value``. Only ``samples``
# reads the rows; the others take them so every section is called alike.
SECTIONS: dict[str, Callable[[OptimizationProblem, list[list[int]]], Any]] = {
    "snapshot": section_snapshot,
    "cqm_domains": section_cqm_domains,
    "samples": section_samples,
    "recommend": section_recommend,
    "postprocess_costs": section_postprocess_costs,
    "exact_solve": section_exact_solve,
}


# --------------------------------------------------------------------------
# Malformed documents
# --------------------------------------------------------------------------


def _format_base() -> dict:
    """A small valid ``1.0`` document every malformed one starts from."""
    return {
        "version": "1.0",
        "name": "format error base",
        "variables": [{"name": "x"}, {"name": "y"}],
        "objective": {
            "direction": "minimize",
            "linear_terms": [{"variable": "x", "coefficient": 1}],
        },
        "constraints": [
            {
                "id": "pick",
                "type": "hard",
                "terms": [
                    {"variable": "x", "coefficient": 1},
                    {"variable": "y", "coefficient": 1},
                ],
                "operator": "==",
                "rhs": 1,
            }
        ],
    }


def _missing_constraint_rhs() -> dict:
    document = _format_base()
    del document["constraints"][0]["rhs"]
    return document


def _operator_less_than() -> dict:
    document = _format_base()
    document["constraints"][0]["operator"] = "<"
    return document


def _unknown_constraint_key() -> dict:
    document = _format_base()
    document["constraints"][0]["unexpected_key"] = 1
    return document


def _version_9_9() -> dict:
    document = _format_base()
    document["version"] = "9.9"
    return document


def _version_1_2() -> dict:
    document = _format_base()
    document["version"] = "1.2"
    return document


def _top_level_cardinality_constraints() -> dict:
    document = _format_base()
    document["cardinality_constraints"] = []
    return document


FORMAT_ERROR_DOCUMENTS: dict[str, Callable[[], dict]] = {
    "missing_constraint_rhs": _missing_constraint_rhs,
    "operator_less_than": _operator_less_than,
    "unknown_constraint_key": _unknown_constraint_key,
    "version_9_9": _version_9_9,
}

# The next batch makes "1.2" a valid version and ``cardinality_constraints`` a
# valid top-level field, so these two are recorded for reference only.
FORMAT_ERROR_DOCUMENTS_EXPECTED_TO_CHANGE: dict[str, Callable[[], dict]] = {
    "version_1_2": _version_1_2,
    "top_level_cardinality_constraints": _top_level_cardinality_constraints,
}


def format_errors(document: dict) -> list[list[str | None]] | None:
    """``[code, path]`` per schema error; ``None`` if the document parses."""
    parsed = parse_problem(document)
    if isinstance(parsed, OptimizationProblem):
        return None
    return _code_path(parsed)


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def record_problem(problem: OptimizationProblem) -> dict:
    rows = sample_rows(problem)
    entry: dict[str, Any] = {"version": problem.version, "sample_rows": rows}
    for section, snapshot in SECTIONS.items():
        entry[section] = snapshot(problem, rows)
    return entry


def record() -> dict:
    documents = {
        "format_errors": FORMAT_ERROR_DOCUMENTS,
        "format_errors_expected_to_change": FORMAT_ERROR_DOCUMENTS_EXPECTED_TO_CHANGE,
    }
    recorded: dict[str, Any] = {
        "recorded_from": RECORDED_FROM,
        "problems": {name: record_problem(build()) for name, build in PROBLEMS.items()},
    }
    for key, group in documents.items():
        recorded[key] = {}
        for name, build in group.items():
            errors = format_errors(build())
            if errors is None:
                raise ValueError(f"malformed document {name!r} parsed without error")
            recorded[key][name] = errors
    return recorded


def main() -> None:
    GOLDEN_PATH.write_text(
        json.dumps(record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"recorded {len(PROBLEMS)} problems to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
