"""Record the Phase 3a (commit ``20ba640``) compile / estimate / validate output.

Phase 3b spec §25 step 0 and §26.1: before any 3b change touches ``src/``,
the exact numbers the 3a code produces for a *fixed* list of ``version 1.0``
problems are recorded here, and ``tests/unit/test_golden_phase3a.py`` asserts
from step 1 onwards that every later commit still produces the same numbers
bit for bit. The list is spelled out in this module on purpose (no
discovery): the three shipped examples, every problem the scenario tests use,
and every problem ``tests/unit/test_bqm_compiler.py`` /
``test_cqm_compiler.py`` build inline.

The snapshot is a custom JSON shape (never ``bqm.to_serializable()``, which is
not a stable contract). Every value is a plain ``list`` / ``dict`` / ``str`` /
``float`` / ``int`` / ``bool`` / ``None`` so a ``json`` round-trip is exact.

Re-record (only ever intentionally, from the 3a code) with::

    .venv/Scripts/python.exe tests/golden/record_phase3a_golden.py
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import dimod

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.models import OptimizationProblem, SolverCapabilities
from annealbridge.penalty.strategy import ScaledPenaltyStrategy
from annealbridge.solvers import SolverRegistry
from annealbridge.validation import validate_problem_full
from annealbridge.validation.estimates import (
    compute_penalty_scale,
    estimate_compiled_variables,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "phase3a_compile.json"
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
RECORDED_FROM = "20ba640"


# --------------------------------------------------------------------------
# Problem builders. Each entry of PROBLEMS is a zero-argument callable that
# builds one ``version 1.0`` problem; the key is its stable name.
# --------------------------------------------------------------------------


def _load_example(name: str, **solver_overrides) -> OptimizationProblem:
    data = json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))
    if solver_overrides:
        data["solver"] = {**data.get("solver", {}), **solver_overrides}
    return OptimizationProblem.model_validate(data)


def example_assignment() -> OptimizationProblem:
    return _load_example("assignment.json")


def example_knapsack() -> OptimizationProblem:
    return _load_example("knapsack.json")


def example_tsp() -> OptimizationProblem:
    return _load_example("tsp.json")


# tests/scenarios/test_soft_weight_dominance.py::soft_dominated_problem
def _soft_dominated_problem(backend: str) -> OptimizationProblem:
    return OptimizationProblem.model_validate(
        {
            "name": "soft-dominated",
            "variables": [{"name": "x"}, {"name": "y"}],
            "objective": {
                "direction": "minimize",
                "linear_terms": [
                    {"variable": "x", "coefficient": 1},
                    {"variable": "y", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "at_least_one",
                    "type": "hard",
                    "terms": [
                        {"variable": "x", "coefficient": 1},
                        {"variable": "y", "coefficient": 1},
                    ],
                    "operator": ">=",
                    "rhs": 1,
                },
                {
                    "id": "prefer_none",
                    "type": "soft",
                    "weight": 1000,
                    "terms": [
                        {"variable": "x", "coefficient": 1},
                        {"variable": "y", "coefficient": 1},
                    ],
                    "operator": "==",
                    "rhs": 0,
                },
            ],
            "solver": {"backend": backend, "seed": 1},
        }
    )


def scenario_soft_dominated_exact() -> OptimizationProblem:
    return _soft_dominated_problem("exact")


def scenario_soft_dominated_sa() -> OptimizationProblem:
    return _soft_dominated_problem("simulated_annealing")


# tests/scenarios/test_soft_weight_dominance.py::knapsack_with_soft_take_everything
def scenario_knapsack_with_soft_take_everything() -> OptimizationProblem:
    data = json.loads((EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8"))
    data["solver"] = {
        **data.get("solver", {}),
        "backend": "simulated_annealing",
        "seed": 1,
        "num_reads": 100,
    }
    data["constraints"].append(
        {
            "id": "take_everything",
            "type": "soft",
            "weight": 10_000.0,
            "terms": [
                {"variable": name, "coefficient": 1}
                for name in ("item_a", "item_b", "item_c", "item_d")
            ],
            "operator": "==",
            "rhs": 4,
        }
    )
    return OptimizationProblem.model_validate(data)


# tests/scenarios/test_retry.py (TINY_MULTIPLIER = 0.01, SEED = 0)
def scenario_knapsack_retry_tiny_multiplier() -> OptimizationProblem:
    return _load_example(
        "knapsack.json",
        backend="simulated_annealing",
        penalty_multiplier=0.01,
        seed=0,
    )


# tests/scenarios/test_knapsack_cqm.py (the real CQM backend name)
def scenario_knapsack_cqm() -> OptimizationProblem:
    return _load_example("knapsack.json", backend="leap_hybrid_cqm")


# --- helpers copied from tests/unit/test_bqm_compiler.py / test_cqm_compiler.py


def make_problem(
    *,
    direction: str = "minimize",
    linear: list[dict] | None = None,
    quadratic: list[dict] | None = None,
    constant: float = 0,
    constraints: list[dict] | None = None,
    variables: tuple[str, ...] = ("x1", "x2", "x3"),
) -> OptimizationProblem:
    return OptimizationProblem.model_validate(
        {
            "name": "compiler test problem",
            "variables": [{"name": name} for name in variables],
            "objective": {
                "direction": direction,
                "linear_terms": linear or [],
                "quadratic_terms": quadratic or [],
                "constant": constant,
            },
            "constraints": constraints or [],
        }
    )


def lin(variable: str, coefficient: float) -> dict:
    return {"variable": variable, "coefficient": coefficient}


def quad(variable1: str, variable2: str, coefficient: float) -> dict:
    return {"variable1": variable1, "variable2": variable2, "coefficient": coefficient}


def hard(constraint_id: str, operator: str, rhs: float, terms: list[dict]) -> dict:
    return {
        "id": constraint_id,
        "type": "hard",
        "terms": terms,
        "operator": operator,
        "rhs": rhs,
    }


def soft(
    constraint_id: str, operator: str, rhs: float, terms: list[dict], weight: float
) -> dict:
    return {
        "id": constraint_id,
        "type": "soft",
        "terms": terms,
        "operator": operator,
        "rhs": rhs,
        "weight": weight,
    }


# BEGIN COMPILER TEST PROBLEMS
# Problems built inline by tests/unit/test_bqm_compiler.py (``bqm_<test>``)
# and tests/unit/test_cqm_compiler.py (``cqm_<test>``), copied verbatim.
#
# Deliberately not recorded (``snapshot_problem`` compiles every problem with
# *both* compilers, so a problem either compiler rejects cannot be snapshotted):
#
# * ``test_cqm_compiler.py::test_constant_constraint_that_fails_raises`` — all
#   three parameters (``("<=", -1)``, ``(">=", 1)``, ``("==", 1)``) build a
#   hard constraint whose terms cancel to a constant that cannot hold, which
#   is exactly what the test asserts raises ``CompilationError``.
# * ``test_cqm_compiler.py::test_knapsack_example`` — reads
#   ``examples/knapsack.json``; already recorded as ``example_knapsack``.
#
# ``make_problem()`` with no arguments is built by
# ``test_bqm_compiler.py::test_floor_of_one_for_empty_objective`` and by
# ``test_cqm_compiler.py::test_hard_penalty_is_rejected`` /
# ``test_objective_scale_uses_shared_formula``; recorded once as
# ``cqm_test_default``.
COMPILER_TEST_PROBLEMS: dict[str, Callable[[], OptimizationProblem]] = {
    # --- tests/unit/test_bqm_compiler.py ---
    "bqm_test_minimize_linear_and_constant": lambda: make_problem(
        linear=[lin("x1", 2), lin("x2", -3)], constant=5
    ),
    "bqm_test_maximize_negates_all_coefficients_and_constant": lambda: make_problem(
        direction="maximize",
        linear=[lin("x1", 2)],
        quadratic=[quad("x1", "x2", 4)],
        constant=5,
    ),
    "bqm_test_duplicate_linear_terms_accumulate": lambda: make_problem(
        linear=[lin("x1", 2), lin("x1", 3)]
    ),
    "bqm_test_duplicate_quadratic_terms_accumulate_across_orderings": (
        lambda: make_problem(quadratic=[quad("x1", "x2", 3), quad("x2", "x1", 4)])
    ),
    "bqm_test_unreferenced_declared_variable_is_in_bqm": lambda: make_problem(
        linear=[lin("x1", 1)]
    ),
    "bqm_test_hand_computed_expansion": lambda: make_problem(
        constraints=[hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)])]
    ),
    "bqm_test_trace_has_no_slack": lambda: make_problem(
        constraints=[hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)])]
    ),
    "bqm_test_less_equal_hard_generates_slack": lambda: make_problem(
        constraints=[hard("cap", "<=", 4, [lin("x1", 2), lin("x2", 3)])]
    ),
    "bqm_test_greater_equal_soft_trace_keeps_original_operator": lambda: make_problem(
        constraints=[soft("cover", ">=", 1, [lin("x1", 1), lin("x2", 1)], 2.0)]
    ),
    "bqm_test_redundant_constraint_adds_nothing_base": lambda: make_problem(
        linear=[lin("x1", 2), lin("x2", 1)]
    ),
    "bqm_test_redundant_constraint_adds_nothing_with_redundant": lambda: make_problem(
        linear=[lin("x1", 2), lin("x2", 1)],
        constraints=[hard("loose", "<=", 5, [lin("x1", 1), lin("x2", 1)])],
    ),
    "bqm_test_hard_and_soft_lambdas_never_mix": lambda: make_problem(
        constraints=[
            hard("h", "==", 1, [lin("x1", 1)]),
            soft("s", "==", 1, [lin("x2", 1)], 5.0),
        ]
    ),
    "bqm_test_energy_matches_sign_objective_plus_penalties": lambda: make_problem(
        direction="maximize",
        linear=[lin("x1", 3), lin("x2", 4), lin("x3", 5)],
        quadratic=[quad("x1", "x2", -2)],
        constant=1,
        constraints=[
            hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)]),
            hard("cap", "<=", 5, [lin("x1", 2), lin("x2", 3), lin("x3", 4)]),
            soft("cover", ">=", 1, [lin("x2", 1), lin("x3", 1)], 2.5),
        ],
    ),
    "bqm_test_problem_not_mutated": lambda: make_problem(
        direction="maximize",
        linear=[lin("x1", 3), lin("x2", 4)],
        constraints=[
            hard("cap", "<=", 4, [lin("x1", 2), lin("x2", 3)]),
            soft("s", "==", 1, [lin("x3", 1)], 2.0),
        ],
    ),
    "bqm_test_formula": lambda: make_problem(
        linear=[lin("x1", 2), lin("x2", -3)],
        quadratic=[quad("x1", "x2", 4)],
    ),
    "bqm_test_lowest_energy_sample_is_feasible_optimum": lambda: make_problem(
        direction="maximize",
        linear=[lin("x1", 3), lin("x2", 4), lin("x3", 5)],
        constraints=[hard("cap", "<=", 5, [lin("x1", 2), lin("x2", 3), lin("x3", 4)])],
    ),
    # --- tests/unit/test_cqm_compiler.py ---
    "cqm_test_default": lambda: make_problem(),
    "cqm_test_every_declared_variable_is_binary_in_problem_order": lambda: make_problem(
        linear=[lin("a", 1)], variables=("b", "a", "c")
    ),
    "cqm_test_minimize_linear_quadratic_and_constant": lambda: make_problem(
        linear=[lin("x1", 2), lin("x2", -3)],
        quadratic=[quad("x1", "x2", 4)],
        constant=5,
    ),
    "cqm_test_maximize_negates_all_coefficients_and_constant": lambda: make_problem(
        direction="maximize",
        linear=[lin("x1", 2)],
        quadratic=[quad("x1", "x2", 4)],
        constant=5,
    ),
    "cqm_test_objective_scale_uses_shared_formula": lambda: make_problem(
        linear=[lin("x1", 2), lin("x2", -3)],
        quadratic=[quad("x1", "x2", 4)],
    ),
    "cqm_test_hard_constraint_is_native_with_no_weight": lambda: make_problem(
        constraints=[hard("cap", "<=", 4, [lin("x1", 2), lin("x2", 3)])]
    ),
    "cqm_test_soft_constraint_carries_weight_and_quadratic_penalty": (
        lambda: make_problem(
            constraints=[soft("cover", ">=", 1, [lin("x1", 1), lin("x2", 1)], 2.5)]
        )
    ),
    "cqm_test_labels_are_constraint_ids_in_problem_order": lambda: make_problem(
        constraints=[
            hard("zeta", "==", 1, [lin("x1", 1), lin("x2", 1)]),
            soft("alpha", "<=", 1, [lin("x3", 1)], 1.0),
            hard("mid", ">=", 0, [lin("x1", 1)]),
        ]
    ),
    "cqm_test_duplicate_terms_accumulate_and_zero_coefficients_are_dropped": (
        lambda: make_problem(
            constraints=[
                hard(
                    "acc",
                    "<=",
                    3,
                    [
                        lin("x1", 2),
                        lin("x1", 3),
                        lin("x2", 1),
                        lin("x2", -1),
                        lin("x3", 4),
                    ],
                )
            ]
        )
    ),
    "cqm_test_constant_constraint_that_holds_is_redundant_and_not_added": (
        lambda: make_problem(
            linear=[lin("x1", 1)],
            constraints=[
                hard("noop", "<=", 0, [lin("x1", 1), lin("x1", -1)]),
                soft("noop_soft", ">=", -1, [lin("x2", 2), lin("x2", -2)], 3.0),
                hard("noop_eq", "==", 0, [lin("x3", 1), lin("x3", -1)]),
            ],
        )
    ),
    "cqm_test_problem_not_mutated": lambda: make_problem(
        direction="maximize",
        linear=[lin("x1", 3), lin("x2", 4)],
        constraints=[
            hard("cap", "<=", 4, [lin("x1", 2), lin("x2", 3)]),
            soft("s", "==", 1, [lin("x3", 1)], 2.0),
        ],
    ),
    "cqm_test_two_compilations_are_identical": lambda: make_problem(
        direction="maximize",
        linear=[lin("x3", 3), lin("x1", 4), lin("x2", 5)],
        quadratic=[quad("x2", "x1", -2)],
        constant=1,
        constraints=[
            hard("pick", "==", 1, [lin("x2", 1), lin("x1", 1)]),
            hard("cap", "<=", 5, [lin("x3", 4), lin("x1", 2), lin("x2", 3)]),
            soft("cover", ">=", 1, [lin("x2", 1), lin("x3", 1)], 2.5),
        ],
    ),
    "cqm_test_soft_energy_is_weight_times_violation_squared_minimize": (
        lambda: make_problem(
            direction="minimize",
            linear=[lin("x1", 2), lin("x2", -5), lin("x3", 1)],
            constraints=[
                soft("s", "==", 0, [lin("x1", 1), lin("x2", 1), lin("x3", 1)], 10.0)
            ],
        )
    ),
    "cqm_test_soft_energy_is_weight_times_violation_squared_maximize": (
        lambda: make_problem(
            direction="maximize",
            linear=[lin("x1", 2), lin("x2", -5), lin("x3", 1)],
            constraints=[
                soft("s", "==", 0, [lin("x1", 1), lin("x2", 1), lin("x3", 1)], 10.0)
            ],
        )
    ),
    "cqm_test_mixed_ge_eq_hard_constraints_with_a_soft_one": lambda: make_problem(
        direction="maximize",
        linear=[lin("x1", 3), lin("x2", 1), lin("x3", 2), lin("x4", 4)],
        quadratic=[quad("x3", "x4", -1)],
        variables=("x1", "x2", "x3", "x4"),
        constraints=[
            hard("pick", "==", 1, [lin("x1", 1), lin("x2", 1)]),
            hard("cover", ">=", 2, [lin("x2", 1), lin("x3", 1), lin("x4", 1)]),
            soft("apart", "<=", 1, [lin("x1", 1), lin("x4", 1)], 2.0),
        ],
    ),
    "cqm_test_lowest_energy_feasible_sample_is_the_business_optimum": (
        lambda: make_problem(
            direction="maximize",
            linear=[lin("x1", 3), lin("x2", 4), lin("x3", 5)],
            constraints=[
                hard("cap", "<=", 5, [lin("x1", 2), lin("x2", 3), lin("x3", 4)])
            ],
        )
    ),
}
# END COMPILER TEST PROBLEMS


PROBLEMS: dict[str, Callable[[], OptimizationProblem]] = {
    "example_assignment": example_assignment,
    "example_knapsack": example_knapsack,
    "example_tsp": example_tsp,
    "scenario_soft_dominated_exact": scenario_soft_dominated_exact,
    "scenario_soft_dominated_sa": scenario_soft_dominated_sa,
    "scenario_knapsack_with_soft_take_everything": (
        scenario_knapsack_with_soft_take_everything
    ),
    "scenario_knapsack_retry_tiny_multiplier": scenario_knapsack_retry_tiny_multiplier,
    "scenario_knapsack_cqm": scenario_knapsack_cqm,
    **COMPILER_TEST_PROBLEMS,
}


# --------------------------------------------------------------------------
# Snapshot
# --------------------------------------------------------------------------


def _bqm_shape(bqm: dimod.BinaryQuadraticModel) -> dict:
    """``linear`` dict, sorted ``[u, v, bias]`` quadratic list and offset."""
    quadratic = sorted(
        sorted((str(u), str(v))) + [float(bias)] for u, v, bias in bqm.iter_quadratic()
    )
    return {
        "linear": {str(v): float(bqm.get_linear(v)) for v in bqm.variables},
        "quadratic": quadratic,
        "offset": float(bqm.offset),
    }


def _exact_capabilities() -> SolverCapabilities:
    return SolverRegistry.default().get("exact").capabilities


def snapshot_problem(problem: OptimizationProblem) -> dict:
    """Everything 3b must keep bit-identical for a ``version 1.0`` problem."""
    hard_penalty = ScaledPenaltyStrategy().initial_penalty(problem)

    compiled_bqm = BQMCompiler().compile(problem, hard_penalty)
    bqm = compiled_bqm.model
    bqm_snapshot = {
        "hard_penalty": float(hard_penalty),
        "variables": [str(v) for v in bqm.variables],
        **_bqm_shape(bqm),
        "num_variables": int(compiled_bqm.num_variables),
        "internal_variables": sorted(compiled_bqm.internal_variables),
        "objective_scale": float(compiled_bqm.objective_scale),
        "constraint_trace": [
            trace.model_dump(mode="json") for trace in compiled_bqm.constraint_trace
        ],
    }

    compiled_cqm = CQMCompiler().compile(problem, None)
    cqm = compiled_cqm.model
    constraints = []
    for label in cqm.constraint_labels:
        constraint = cqm.constraints[label]
        soft_view = cqm._soft.get(label)
        constraints.append(
            {
                "label": str(label),
                "lhs": _bqm_shape(constraint.lhs),
                "sense": constraint.sense.value,
                "rhs": float(constraint.rhs),
                "weight": None if soft_view is None else float(soft_view.weight),
                "penalty": None if soft_view is None else str(soft_view.penalty),
            }
        )
    cqm_snapshot = {
        "variables": [str(v) for v in cqm.variables],
        "objective": _bqm_shape(cqm.objective),
        "constraints": constraints,
        "num_variables": int(compiled_cqm.num_variables),
        "internal_variables": sorted(compiled_cqm.internal_variables),
        "objective_scale": float(compiled_cqm.objective_scale),
        "constraint_trace": [
            trace.model_dump(mode="json") for trace in compiled_cqm.constraint_trace
        ],
    }

    return {
        "bqm": bqm_snapshot,
        "cqm": cqm_snapshot,
        "estimate_compiled_variables": int(estimate_compiled_variables(problem)),
        "compute_penalty_scale": float(compute_penalty_scale(problem)),
        "validate_full_exact": validate_problem_full(
            problem, capabilities=_exact_capabilities()
        ).model_dump(mode="json"),
        "validate_full_cqm_path": validate_problem_full(
            problem, model_type="cqm"
        ).model_dump(mode="json"),
    }


def record() -> dict:
    return {
        "recorded_from": RECORDED_FROM,
        "problems": {name: snapshot_problem(build()) for name, build in PROBLEMS.items()},
    }


def main() -> None:
    GOLDEN_PATH.write_text(
        json.dumps(record(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"recorded {len(PROBLEMS)} problems to {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
