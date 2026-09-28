"""Record the schema ``1.2`` golden (``cardinality_constraints``, spec §12.2).

Recorded *after* ``cardinality_constraints`` was implemented, from the
working tree ``a377f35`` plus the schema 1.2 changes. It freezes what the
code does today for a fixed list of ``"version": "1.2"`` problems so a later
change cannot move it unnoticed; it does not prove that output correct --
spec §13 and the ``test_cardinality_*`` modules do that. The list is spelled
out here on purpose (no discovery), every problem is ``1.2``:

* ``examples/exam_timetabling.json`` (the schema 1.2 example);
* hard at-most-one with the pairwise encoding: two and four variables, and
  two of them sharing a variable;
* hard ``== 1`` and ``== 2``; hard ``>= 2`` and ``<= 2`` over four variables
  (both with slack);
* soft ``==``, ``<=`` and ``>=``, the last with an rhs no count can reach
  (always violated);
* redundant: ``"<=", n`` and a one-variable ``"<=", 1``;
* a mixed problem: cardinality constraints over binaries beside linear
  constraints over an integer variable, a linear hard ``<= 1`` that raises
  ``CARDINALITY_FORM_AVAILABLE`` and a soft linear constraint;
* infeasible as a whole: ``== 1`` against ``== 2`` over the same variables,
  once alone (the closest candidate is a tie) and once with a linear
  constraint that makes it unique;
* a maximize set packing built from at-most-ones;
* ``examples/knapsack.json`` and ``examples/shift_scheduling.json``
  relabelled ``1.2`` (no cardinality constraint; the latter raises
  ``CARDINALITY_FORM_AVAILABLE``).

Each problem records the sections of ``record_compat_golden`` -- its section
functions are imported and reused, so ``snapshot``, ``cqm_domains``,
``samples``, ``recommend``, ``postprocess_costs`` and ``exact_solve`` follow
the same determinism rules (whitelisted fields, no timing or version field,
the sample rows stored and replayed, no tie-break pick) -- plus one more:

``trace_kinds``
    For every entry of the compiled BQM's ``constraint_trace`` (one per
    ``problem.all_constraints()`` entry, in that order): its
    ``constraint_id``, ``generated_variables`` and ``slack_range``, and
    whether ``validation.estimates.uses_pairwise_penalty`` holds for the
    constraint it traces.

Recording refuses (raises) instead of writing a golden when a pairwise
constraint generates slack, when ``estimate_compiled_variables`` differs
from the compiled BQM, or when an infeasible exact solve leaves a hard
cardinality constraint out of its violation rates.

Every value is a plain ``list`` / ``dict`` / ``str`` / ``float`` / ``int`` /
``bool`` / ``None`` so a ``json`` round-trip is exact. Re-record (only ever
intentionally) with::

    .venv/Scripts/python.exe tests/golden/record_schema_1_2_golden.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from annealbridge.compiler import BQMCompiler
from annealbridge.models import OptimizationProblem
from annealbridge.penalty.strategy import ScaledPenaltyStrategy
from annealbridge.validation.estimates import (
    estimate_compiled_variables,
    uses_pairwise_penalty,
)

# Run as a script only this file's directory is on ``sys.path``; the compat
# golden's helpers are imported as ``tests.golden...`` exactly as the test
# suite imports them, so the repository root has to be importable too.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.golden.record_compat_golden import SECTIONS as COMPAT_SECTIONS
from tests.golden.record_compat_golden import (
    _load_example,
    _merge,
    binary,
    integer,
    record_problem,
)
from tests.golden.record_phase3a_golden import hard, lin, quad, soft

GOLDEN_PATH = Path(__file__).resolve().parent / "schema_1_2_compile.json"
RECORDED_FROM = "a377f35+schema-1.2"
VERSION = "1.2"


# --------------------------------------------------------------------------
# Problem builders. Each entry of PROBLEMS is a zero-argument callable that
# builds one ``1.2`` problem; the key is its stable name.
# --------------------------------------------------------------------------


def hard_card(
    constraint_id: str, operator: str, rhs: int, variables: list[str]
) -> dict:
    return {
        "id": constraint_id,
        "type": "hard",
        "variables": variables,
        "operator": operator,
        "rhs": rhs,
    }


def soft_card(
    constraint_id: str, operator: str, rhs: int, variables: list[str], weight: float
) -> dict:
    return {
        "id": constraint_id,
        "type": "soft",
        "variables": variables,
        "operator": operator,
        "rhs": rhs,
        "weight": weight,
    }


def make_problem_1_2(
    variables: list[dict],
    *,
    name: str,
    direction: str = "minimize",
    linear: list[dict] | None = None,
    quadratic: list[dict] | None = None,
    constraints: list[dict] | None = None,
    cardinality: list[dict] | None = None,
) -> OptimizationProblem:
    """``record_compat_golden.make_integer_problem`` for ``1.2``.

    That builder fixes ``"version": "1.1"`` and has no
    ``cardinality_constraints``, so this one takes both.
    """
    return OptimizationProblem.model_validate(
        {
            "version": VERSION,
            "name": name,
            "variables": variables,
            "objective": {
                "direction": direction,
                "linear_terms": linear or [],
                "quadratic_terms": quadratic or [],
            },
            "constraints": constraints or [],
            "cardinality_constraints": cardinality or [],
        }
    )


def _binaries(*names: str) -> list[dict]:
    return [binary(name) for name in names]


def example_exam_timetabling() -> OptimizationProblem:
    return _load_example("exam_timetabling.json")


def example_knapsack_v1_2() -> OptimizationProblem:
    return _load_example("knapsack.json", version=VERSION)


def example_shift_scheduling_v1_2() -> OptimizationProblem:
    return _load_example("shift_scheduling.json", version=VERSION)


# BEGIN CARDINALITY PROBLEMS
# Written for this golden. Objective coefficients are distinct wherever a tie
# would only blur what the constraint does.
CARDINALITY_PROBLEMS: dict[str, Callable[[], OptimizationProblem]] = {
    # --- hard at-most-one: the pairwise encoding, no slack ---
    "pairwise_at_most_one_n2": lambda: make_problem_1_2(
        _binaries("a", "b"),
        name="pairwise at most one of two",
        linear=[lin("a", -1), lin("b", -2)],
        cardinality=[hard_card("one_of_ab", "<=", 1, ["a", "b"])],
    ),
    "pairwise_at_most_one_n4": lambda: make_problem_1_2(
        _binaries("a", "b", "c", "d"),
        name="pairwise at most one of four",
        linear=[lin("a", -1), lin("b", -4), lin("c", -2), lin("d", -3)],
        cardinality=[hard_card("one_of_abcd", "<=", 1, ["a", "b", "c", "d"])],
    ),
    # Two at-most-ones sharing ``c``: the couplings of both land on c.
    "pairwise_shared_variable": lambda: make_problem_1_2(
        _binaries("a", "b", "c", "d", "e"),
        name="two at-most-ones sharing a variable",
        direction="maximize",
        linear=[lin("a", 1), lin("b", 2), lin("c", 3), lin("d", 2.5), lin("e", 1.5)],
        cardinality=[
            hard_card("left", "<=", 1, ["a", "b", "c"]),
            hard_card("right", "<=", 1, ["c", "d", "e"]),
        ],
    ),
    # --- hard equalities: squared penalty, no slack ---
    "hard_exactly_one": lambda: make_problem_1_2(
        _binaries("a", "b", "c"),
        name="hard exactly one",
        linear=[lin("a", 3), lin("b", 1), lin("c", 2)],
        cardinality=[hard_card("pick_one", "==", 1, ["a", "b", "c"])],
    ),
    "hard_exactly_two": lambda: make_problem_1_2(
        _binaries("a", "b", "c", "d"),
        name="hard exactly two",
        linear=[lin("a", 4), lin("b", 1), lin("c", 3), lin("d", 2)],
        quadratic=[quad("b", "d", 2)],
        cardinality=[hard_card("pick_two", "==", 2, ["a", "b", "c", "d"])],
    ),
    # --- hard inequalities other than at-most-one: slack ---
    "hard_at_least_two": lambda: make_problem_1_2(
        _binaries("a", "b", "c", "d"),
        name="hard at least two of four",
        linear=[lin("a", 4), lin("b", 1), lin("c", 3), lin("d", 2)],
        cardinality=[hard_card("at_least_two", ">=", 2, ["a", "b", "c", "d"])],
    ),
    "hard_at_most_two": lambda: make_problem_1_2(
        _binaries("a", "b", "c", "d"),
        name="hard at most two of four",
        direction="maximize",
        linear=[lin("a", 4), lin("b", 1), lin("c", 3), lin("d", 2)],
        cardinality=[hard_card("at_most_two", "<=", 2, ["a", "b", "c", "d"])],
    ),
    # --- soft cardinality: slack or squared penalty, weighted ---
    "soft_exactly_one": lambda: make_problem_1_2(
        _binaries("a", "b", "c"),
        name="soft exactly one",
        direction="maximize",
        linear=[lin("a", 1), lin("b", 2), lin("c", 1.5)],
        cardinality=[soft_card("prefer_one", "==", 1, ["a", "b", "c"], 2.5)],
    ),
    "soft_at_most_one": lambda: make_problem_1_2(
        _binaries("a", "b", "c"),
        name="soft at most one (slack, never pairwise)",
        direction="maximize",
        linear=[lin("a", 2), lin("b", 3), lin("c", 1)],
        cardinality=[soft_card("prefer_few", "<=", 1, ["a", "b", "c"], 1.5)],
    ),
    # At least 4 of 3: no count reaches it, SOFT_ALWAYS_VIOLATED, and the
    # compiler clamps the slack to zero bits.
    "soft_at_least_out_of_range": lambda: make_problem_1_2(
        _binaries("a", "b", "c"),
        name="soft at least four of three",
        linear=[lin("a", 1), lin("b", 3), lin("c", 2)],
        cardinality=[soft_card("unreachable", ">=", 4, ["a", "b", "c"], 2.0)],
    ),
    # --- redundant: always holds, adds nothing ---
    "redundant_at_most_n": lambda: make_problem_1_2(
        _binaries("a", "b", "c"),
        name="hard at most three of three",
        linear=[lin("a", -1), lin("b", 2), lin("c", -3)],
        cardinality=[hard_card("all_allowed", "<=", 3, ["a", "b", "c"])],
    ),
    # One variable: too short for the pairwise encoding, and redundant.
    "redundant_single_at_most_one": lambda: make_problem_1_2(
        _binaries("a", "b"),
        name="hard at most one of one",
        linear=[lin("a", -2), lin("b", 1)],
        cardinality=[hard_card("lonely", "<=", 1, ["a"])],
    ),
    # --- mixed: cardinality over binaries, linear over an integer too ---
    # ``ab_at_most_one`` is a linear hard ``<= 1`` over two binaries with
    # every coefficient 1: slack encoded, CARDINALITY_FORM_AVAILABLE.
    "mixed_cardinality_linear_integer": lambda: make_problem_1_2(
        [*_binaries("a", "b", "c", "d"), integer("k", 0, 3)],
        name="cardinality beside linear and integer",
        direction="maximize",
        linear=[lin("a", 3), lin("b", 2), lin("c", 1), lin("d", 2.5), lin("k", 1)],
        quadratic=[quad("a", "k", -0.5)],
        constraints=[
            hard("ab_at_most_one", "<=", 1, [lin("a", 1), lin("b", 1)]),
            hard("budget", "<=", 5, [lin("a", 2), lin("b", 3), lin("k", 1)]),
            soft("k_with_c", ">=", 2, [lin("k", 1), lin("c", 1)], 1.5),
        ],
        cardinality=[
            hard_card("cd_one", "==", 1, ["c", "d"]),
            hard_card("bcd_at_most_one", "<=", 1, ["b", "c", "d"]),
        ],
    ),
    # --- infeasible as a whole: each hard constraint holds alone ---
    # Every count violates one of the two by at least 1; the minimum (1) is
    # shared by the six assignments choosing one or two, so the closest
    # candidate is a tie and is not recorded.
    "infeasible_exactly_one_and_two": lambda: make_problem_1_2(
        _binaries("a", "b", "c"),
        name="exactly one and exactly two",
        linear=[lin("a", 1), lin("b", 2), lin("c", 3)],
        cardinality=[
            hard_card("exactly_one", "==", 1, ["a", "b", "c"]),
            hard_card("exactly_two", "==", 2, ["a", "b", "c"]),
        ],
    ),
    # The same pair plus a linear a + b == 0: the smallest total (1) is then
    # reached at a = b = 0, c = 1 only, and the soft cardinality on a is
    # violated there, so the closest candidate's evaluations are recorded
    # with a cardinality row of each type.
    "infeasible_exactly_one_and_two_unique_closest": lambda: make_problem_1_2(
        _binaries("a", "b", "c"),
        name="exactly one and exactly two, unique closest",
        linear=[lin("a", 1), lin("b", 2), lin("c", 3)],
        constraints=[hard("ab_off", "==", 0, [lin("a", 1), lin("b", 1)])],
        cardinality=[
            hard_card("exactly_one", "==", 1, ["a", "b", "c"]),
            hard_card("exactly_two", "==", 2, ["a", "b", "c"]),
            soft_card("prefer_a", ">=", 1, ["a"], 2.0),
        ],
    ),
    # --- maximize: weighted set packing over four elements ---
    # Sets s1 {e1,e2} 3, s2 {e2,e3} 4, s3 {e3,e4} 3, s4 {e1,e4} 1,
    # s5 {e1,e3} 5; one at-most-one per element. Optimum s1 + s3 = 6, unique.
    "maximize_set_packing": lambda: make_problem_1_2(
        _binaries("s1", "s2", "s3", "s4", "s5"),
        name="weighted set packing",
        direction="maximize",
        linear=[lin("s1", 3), lin("s2", 4), lin("s3", 3), lin("s4", 1), lin("s5", 5)],
        cardinality=[
            hard_card("e1_once", "<=", 1, ["s1", "s4", "s5"]),
            hard_card("e2_once", "<=", 1, ["s1", "s2"]),
            hard_card("e3_once", "<=", 1, ["s2", "s3", "s5"]),
            hard_card("e4_once", "<=", 1, ["s3", "s4"]),
        ],
    ),
}
# END CARDINALITY PROBLEMS


PROBLEMS: dict[str, Callable[[], OptimizationProblem]] = _merge(
    {"example_exam_timetabling": example_exam_timetabling},
    CARDINALITY_PROBLEMS,
    # 1.2 without any cardinality constraint
    {
        "example_knapsack_v1_2": example_knapsack_v1_2,
        "example_shift_scheduling_v1_2": example_shift_scheduling_v1_2,
    },
)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _compiled_bqm(problem: OptimizationProblem):
    penalty = ScaledPenaltyStrategy().initial_penalty(problem)
    return BQMCompiler().compile(problem, penalty)


def section_trace_kinds(problem: OptimizationProblem, rows: list[list[int]]) -> list:
    """One entry per BQM trace, paired with ``all_constraints()`` by position."""
    constraints = problem.all_constraints()
    trace = _compiled_bqm(problem).constraint_trace
    if len(trace) != len(constraints):
        raise ValueError(
            f"{len(trace)} BQM trace entries for {len(constraints)} constraints"
        )
    kinds = []
    for constraint, entry in zip(constraints, trace):
        if entry.constraint_id != constraint.id:
            raise ValueError(
                f"BQM trace {entry.constraint_id!r} out of step with "
                f"constraint {constraint.id!r}"
            )
        kinds.append(
            {
                "constraint_id": entry.constraint_id,
                "generated_variables": list(entry.generated_variables),
                "slack_range": entry.slack_range,
                "pairwise": uses_pairwise_penalty(constraint),
            }
        )
    return kinds


# The compat sections unchanged, then the one this golden adds.
SECTIONS: dict[str, Callable[[OptimizationProblem, list[list[int]]], Any]] = {
    **COMPAT_SECTIONS,
    "trace_kinds": section_trace_kinds,
}


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def _refuse_suspicious(name: str, problem: OptimizationProblem, entry: dict) -> None:
    """Raise rather than freeze an output that looks wrong (see the docstring)."""
    for kind in entry["trace_kinds"]:
        if kind["pairwise"] and (
            kind["generated_variables"] or kind["slack_range"] is not None
        ):
            raise ValueError(f"{name}: pairwise {kind['constraint_id']} has slack")
    compiled = _compiled_bqm(problem).num_variables
    estimated = estimate_compiled_variables(problem)
    if estimated != compiled:
        raise ValueError(f"{name}: estimated {estimated} != compiled {compiled}")
    solve = entry["exact_solve"]
    if solve is not None and solve["infeasibility"] is not None:
        rated = {
            rate["constraint_id"] for rate in solve["infeasibility"]["hard_violation_rates"]
        }
        declared = {c.id for c in problem.cardinality_constraints if c.type == "hard"}
        if not declared <= rated:
            raise ValueError(f"{name}: diagnostics miss {sorted(declared - rated)}")


def record_problem_1_2(name: str, problem: OptimizationProblem) -> dict:
    """``record_compat_golden.record_problem`` plus this golden's sections."""
    if problem.version != VERSION:
        raise ValueError(f"{name}: version {problem.version!r}, expected {VERSION!r}")
    entry = record_problem(problem)
    for section, snapshot in SECTIONS.items():
        if section not in COMPAT_SECTIONS:
            entry[section] = snapshot(problem, entry["sample_rows"])
    _refuse_suspicious(name, problem, entry)
    return entry


def record() -> dict:
    return {
        "recorded_from": RECORDED_FROM,
        "problems": {
            name: record_problem_1_2(name, build()) for name, build in PROBLEMS.items()
        },
    }


def main() -> None:
    GOLDEN_PATH.write_text(
        json.dumps(record(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"recorded {len(PROBLEMS)} problems to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
