"""A cardinality constraint compiles exactly like its linear rewrite.

Schema 1.2 spec 2026-09-25 §13 item 3. Every cardinality form except the
declared hard at-most-one (which takes the pairwise encoding, see
``test_cardinality_pairwise.py``) is lowered to a linear constraint whose
coefficients are all 1 and goes down the existing path. So:

(a) the same problem written with ``cardinality_constraints`` and written
    with the equivalent linear constraints appended to the *end* of
    ``constraints`` (declaration order kept, spec §5.3) produces the same
    BQM, CQM, compiled-variable estimate and penalty scale, entry by entry.
    The comparison uses ``record_phase3a_golden.snapshot_problem``, whose
    CQM part carries each soft constraint's weight and penalty; ``cqm.is_equal``
    sees neither (spec §7.4), so it is never relied on here. No CQM
    constraint is marked discrete on either side (spec §7.4).
(b) a ``"1.2"`` document without cardinality constraints compiles and
    estimates exactly like the same document marked ``"1.1"``, and its
    validation output differs only by the ``CARDINALITY_FORM_AVAILABLE``
    advice (spec §12.1 item 3), shown on a problem that triggers the advice
    and one that does not.
"""

import copy
import json

import pytest

from annealbridge.models import OptimizationProblem
from tests.conftest import EXAMPLES_DIR
from tests.golden.record_phase3a_golden import snapshot_problem

COMPILED_PARTS = ("bqm", "cqm", "estimate_compiled_variables", "compute_penalty_scale")
FORM_ADVICE = "CARDINALITY_FORM_AVAILABLE"


# --------------------------------------------------------------------------
# (a) cardinality declaration == linear rewrite
# --------------------------------------------------------------------------


def lin(variable: str, coefficient: float) -> dict:
    return {"variable": variable, "coefficient": coefficient}


def cardinality(
    constraint_id: str,
    kind: str,
    variables: list[str],
    operator: str,
    rhs: int,
    weight: float | None = None,
    description: str | None = None,
) -> dict:
    entry: dict = {
        "id": constraint_id,
        "type": kind,
        "variables": variables,
        "operator": operator,
        "rhs": rhs,
    }
    if weight is not None:
        entry["weight"] = weight
    if description is not None:
        entry["description"] = description
    return entry


def base_payload() -> dict:
    """A 1.2 problem with binaries, one integer and three linear constraints.

    The cardinality forms below only ever count the binaries ``b0..b5``;
    the integer ``n`` and the linear constraints are there so the lowered
    constraints share the model with integer encoding bits and slack of
    other constraints, and so the slack registration order is exercised.
    """
    return {
        "version": "1.2",
        "name": "cardinality equivalence",
        "variables": [
            *({"name": f"b{index}"} for index in range(6)),
            {"name": "n", "type": "integer", "lower_bound": 0, "upper_bound": 5},
        ],
        "objective": {
            "direction": "minimize",
            "linear_terms": [
                lin("b0", 3),
                lin("b1", -2),
                lin("b2", 1.5),
                lin("b3", 4),
                lin("b4", -1),
                lin("b5", 2),
                lin("n", -1),
            ],
            "quadratic_terms": [
                {"variable1": "b0", "variable2": "b1", "coefficient": 2},
                {"variable1": "n", "variable2": "b2", "coefficient": -0.5},
            ],
            "constant": 1,
        },
        "constraints": [
            {
                "id": "budget",
                "type": "hard",
                "terms": [lin("b0", 2), lin("b1", 3), lin("n", 1)],
                "operator": "<=",
                "rhs": 6,
            },
            {
                "id": "pair",
                "type": "hard",
                "terms": [lin("b3", 1), lin("b4", 1)],
                "operator": "==",
                "rhs": 1,
            },
            {
                "id": "soft_mix",
                "type": "soft",
                "weight": 1.5,
                "terms": [lin("b2", 1), lin("n", 1)],
                "operator": ">=",
                "rhs": 2,
            },
        ],
    }


# Every non-pairwise form of spec §13 item 3 (a). The hard at-most-one over
# one variable is included: it is redundant, so it is not pairwise either.
FORMS = {
    "hard_eq_1": cardinality("eq1", "hard", ["b0", "b1", "b2"], "==", 1),
    "hard_eq_2": cardinality("eq2", "hard", ["b1", "b3", "b4", "b5"], "==", 2),
    "hard_le_2": cardinality("le2", "hard", ["b0", "b2", "b4"], "<=", 2),
    "hard_le_3_of_5": cardinality("le3", "hard", ["b0", "b1", "b2", "b3", "b4"], "<=", 3),
    "hard_ge_1": cardinality("ge1", "hard", ["b2", "b5"], ">=", 1),
    "hard_ge_2": cardinality("ge2", "hard", ["b0", "b3", "b5"], ">=", 2),
    "soft_eq": cardinality("s_eq", "soft", ["b0", "b1", "b2"], "==", 2, weight=2.5),
    # A *soft* at-most-one keeps the slack encoding (spec §7.2).
    "soft_le_1": cardinality("s_le1", "soft", ["b3", "b4", "b5"], "<=", 1, weight=3),
    "soft_le_2": cardinality("s_le2", "soft", ["b0", "b2", "b4", "b5"], "<=", 2, weight=0.75),
    "soft_ge": cardinality("s_ge", "soft", ["b1", "b2", "b4"], ">=", 2, weight=1.25),
    "soft_ge_above_n": cardinality("s_ge_over", "soft", ["b0", "b1", "b2"], ">=", 4, weight=2),
    "soft_le_negative": cardinality("s_le_neg", "soft", ["b0", "b5"], "<=", -1, weight=0.5),
    "soft_eq_above_n": cardinality("s_eq_over", "soft", ["b3", "b4"], "==", 3, weight=1),
    "redundant_le_n": cardinality("red", "hard", ["b1", "b3", "b5"], "<=", 3),
    "redundant_ge_0": cardinality("red_ge", "hard", ["b0", "b4"], ">=", 0),
    "single_le_1": cardinality("single", "hard", ["b5"], "<=", 1),
    "described": cardinality(
        "desc", "hard", ["b1", "b2", "b3"], ">=", 1, description="at least one of three"
    ),
}


def linear_rewrite(entry: dict) -> dict:
    """``entry`` as a linear constraint: coefficient 1 per counted variable."""
    rewritten = {key: value for key, value in entry.items() if key != "variables"}
    rewritten["terms"] = [lin(name, 1) for name in entry["variables"]]
    return rewritten


def pair_of_problems(entries: list[dict]) -> tuple[OptimizationProblem, OptimizationProblem]:
    """(cardinality version, linear rewrite appended to ``constraints``)."""
    declared = base_payload()
    declared["cardinality_constraints"] = copy.deepcopy(entries)
    rewritten = base_payload()
    rewritten["constraints"].extend(linear_rewrite(entry) for entry in entries)
    return (
        OptimizationProblem.model_validate(declared),
        OptimizationProblem.model_validate(rewritten),
    )


CASES = [pytest.param([entry], id=name) for name, entry in FORMS.items()] + [
    pytest.param(list(FORMS.values()), id="all_forms_together")
]


class TestDeclarationEqualsLinearRewrite:
    @pytest.mark.parametrize("entries", CASES)
    def test_compiled_models_estimates_and_penalty_scale_match(self, entries):
        declared, rewritten = pair_of_problems(entries)
        assert len(declared.cardinality_constraints) == len(entries)
        assert len(rewritten.cardinality_constraints) == 0

        declared_snapshot = snapshot_problem(declared)
        rewritten_snapshot = snapshot_problem(rewritten)

        for part in COMPILED_PARTS:
            assert declared_snapshot[part] == rewritten_snapshot[part], part

    @pytest.mark.parametrize("entries", CASES)
    def test_no_cqm_constraint_is_marked_discrete(self, entries):
        from annealbridge.compiler import CQMCompiler

        for problem in pair_of_problems(entries):
            cqm = CQMCompiler().compile(problem, None).model
            assert len(cqm.constraint_labels) > 0
            for label in cqm.constraint_labels:
                assert cqm.constraints[label].lhs.is_discrete() is False, label

    def test_the_soft_part_of_the_cqm_snapshot_is_compared(self):
        """The comparison sees soft weights and penalties (spec §7.4)."""
        declared, _ = pair_of_problems([FORMS["soft_eq"]])
        constraints = snapshot_problem(declared)["cqm"]["constraints"]
        (soft,) = [entry for entry in constraints if entry["label"] == "s_eq"]
        assert soft["weight"] == 2.5
        assert soft["penalty"] == "quadratic"

    def test_a_different_weight_is_told_apart(self):
        """Guard for the harness: a changed soft weight breaks the equality."""
        declared, _ = pair_of_problems([FORMS["soft_eq"]])
        changed_entry = {**FORMS["soft_eq"], "weight": 5}
        _, rewritten = pair_of_problems([changed_entry])
        declared_cqm = snapshot_problem(declared)["cqm"]
        rewritten_cqm = snapshot_problem(rewritten)["cqm"]
        assert declared_cqm != rewritten_cqm


# The pairwise forms (hard "<=" 1 over two or more variables): only the BQM
# path encodes them differently, so their CQM must still equal the rewrite's
# (spec §7.4: the CQM path keeps every hard constraint native).
PAIRWISE_FORMS = {
    "hard_le_1_of_2": cardinality("amo2", "hard", ["b3", "b5"], "<=", 1),
    "hard_le_1_of_4": cardinality("amo4", "hard", ["b0", "b1", "b2", "b4"], "<=", 1),
}
PAIRWISE_CASES = [pytest.param([entry], id=name) for name, entry in PAIRWISE_FORMS.items()] + [
    pytest.param(
        [*PAIRWISE_FORMS.values(), FORMS["soft_le_1"], FORMS["hard_eq_2"]],
        id="pairwise_with_other_forms",
    )
]


class TestPairwiseFormOnTheCqmPath:
    @pytest.mark.parametrize("entries", PAIRWISE_CASES)
    def test_cqm_equals_the_linear_rewrite(self, entries):
        declared, rewritten = pair_of_problems(entries)
        assert snapshot_problem(declared)["cqm"] == snapshot_problem(rewritten)["cqm"]

    @pytest.mark.parametrize("entries", PAIRWISE_CASES)
    def test_bqm_differs_from_the_linear_rewrite(self, entries):
        # The guard that the pairwise form really is encoded differently on
        # the BQM path, so the CQM equality above is not vacuous.
        declared, rewritten = pair_of_problems(entries)
        assert snapshot_problem(declared)["bqm"] != snapshot_problem(rewritten)["bqm"]


# --------------------------------------------------------------------------
# (b) "1.2" without cardinality == "1.1"
# --------------------------------------------------------------------------


def load_example(name: str, version: str) -> OptimizationProblem:
    data = json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))
    data["version"] = version
    return OptimizationProblem.model_validate(data)


def without_form_advice(validation: dict) -> dict:
    return {
        **validation,
        "warnings": [w for w in validation["warnings"] if w["code"] != FORM_ADVICE],
    }


def advice_paths(validation: dict) -> list[str]:
    return [w["path"] for w in validation["warnings"] if w["code"] == FORM_ADVICE]


EXAMPLES = ["shift_scheduling.json", "knapsack.json", "assignment.json", "integer_knapsack.json"]


class TestVersion12WithoutCardinalityMatchesVersion11:
    @pytest.mark.parametrize("name", EXAMPLES)
    def test_compilation_and_estimates_are_identical(self, name):
        as_11 = snapshot_problem(load_example(name, "1.1"))
        as_12 = snapshot_problem(load_example(name, "1.2"))
        for part in COMPILED_PARTS:
            assert as_12[part] == as_11[part], part

    @pytest.mark.parametrize("name", EXAMPLES)
    def test_validation_differs_only_by_the_form_advice(self, name):
        as_11 = snapshot_problem(load_example(name, "1.1"))
        as_12 = snapshot_problem(load_example(name, "1.2"))
        for part in ("validate_full_exact", "validate_full_cqm_path"):
            assert advice_paths(as_11[part]) == []
            assert without_form_advice(as_12[part]) == as_11[part], part
        # The CQM path compiles both forms alike, so it never advises.
        assert as_12["validate_full_cqm_path"] == as_11["validate_full_cqm_path"]

    def test_shift_scheduling_triggers_the_advice(self):
        """Its three rest rules are linear hard at-most-ones over two binaries."""
        as_12 = snapshot_problem(load_example("shift_scheduling.json", "1.2"))
        assert advice_paths(as_12["validate_full_exact"]) == [
            "constraints[7]",
            "constraints[8]",
            "constraints[9]",
        ]

    def test_knapsack_does_not_trigger_the_advice(self):
        as_12 = snapshot_problem(load_example("knapsack.json", "1.2"))
        assert advice_paths(as_12["validate_full_exact"]) == []
        assert as_12["validate_full_exact"]["valid"] is True
