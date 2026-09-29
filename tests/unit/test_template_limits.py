"""Expansion ceilings and bounded error lists (validation/expansion, validate_expanded).

Schema 1.3 spec 2026-09-25 §14.9, §14.11 and §14.16 item 4 (v4), with the
user's choice of §15 (6 = B: ``max_template_bindings`` defaults to 250,000):

* the bound U on the bindings the templates iterate is computed without
  iterating: a document with U = 10⁹ is refused with one
  TEMPLATE_EXPANSION_LIMIT (``limit_exceeded``), path at the entry where
  the running sum first passes the ceiling, in far under a mebibyte and a
  fraction of a second;
* the work W (every generated entry plus C(k,2) per generated constraint of
  k terms or members, counted while the members are collected, so a
  constraint later left out at a boundary is not refunded) stops a single
  huge constraint part-way, using a fraction of the full expansion's memory;
* both counts have exact boundaries: U or W equal to the ceiling expands,
  one less is refused;
* static errors win over U;
* the expander keeps at most 20 errors per source and 100 in total, the
  rest counted per code on the last listed error;
* ``validate_expanded`` keeps at most 20 errors per template however many
  generated entries fail, and its peak memory does not grow with them.
"""

import gc
import time
import tracemalloc

from annealbridge.models import OptimizationProblem
from annealbridge.validation import (
    DEFAULT_MAX_TEMPLATE_BINDINGS,
    ExpandedProblem,
    expand_problem,
    validate_expanded,
)

MIB = 1024 * 1024


def document(**fields) -> dict:
    data = {
        "version": "1.3",
        "name": "limits",
        "variables": [],
        "objective": {"direction": "minimize", "linear_terms": []},
        "constraints": [],
    }
    data.update(fields)
    return data


def problem(data: dict) -> OptimizationProblem:
    return OptimizationProblem.model_validate(data)


def measured(call):
    """``(result, seconds, peak bytes)`` of ``call()`` under tracemalloc."""
    gc.collect()
    tracemalloc.start()
    try:
        started = time.perf_counter()
        result = call()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak


def u_error(total: int, limit: int, source: str) -> tuple[str, str, str]:
    """The U refusal: ``total`` is the running sum when it first passed
    ``limit`` (each product stops early once past the ceiling, so it may be
    less than the true U)."""
    return (
        "TEMPLATE_EXPANSION_LIMIT",
        source,
        f"The templates would iterate over {total} or more bindings by the time "
        f"{source} is expanded, above this server's max_template_bindings of "
        f"{limit}; nothing was generated",
    )


def w_error(work: int, limit: int, source: str) -> tuple[str, str, str]:
    """The W refusal: ``work`` is W at the moment it first passed ``limit``."""
    return (
        "TEMPLATE_EXPANSION_LIMIT",
        source,
        f"Generating the templates reached {work} units of work (one per "
        "generated entry plus, for each generated constraint, one per pair of "
        "its terms or members), above this server's max_template_bindings of "
        f"{limit}; it stopped while expanding {source}",
    )


def refused(result: ExpandedProblem) -> tuple[str, str | None, str]:
    """The one limit error of ``result``; nothing else may be reported."""
    assert result.limit_exceeded is True
    assert result.problem is None
    assert result.warnings == ()
    assert len(result.errors) == 1, [(e.code, e.path) for e in result.errors]
    error = result.errors[0]
    assert "ANNEALBRIDGE_" not in error.message
    return error.code, error.path, error.message


def fits(result: ExpandedProblem) -> OptimizationProblem:
    assert result.errors == (), [(e.code, e.path, e.message) for e in result.errors]
    assert result.limit_exceeded is False
    assert result.problem is not None
    return result.problem


def _three_big_sets() -> list[dict]:
    return [{"name": f"s{k}", "elements": list(range(1000))} for k in range(3)]


class TestDefault:
    def test_the_default_ceiling(self):
        """§15 choice 6 = B."""
        assert DEFAULT_MAX_TEMPLATE_BINDINGS == 250_000


class TestBoundU:
    """U > L is refused before anything is iterated (§14.9, §14.16 item 4)."""

    def test_a_family_of_a_billion_variables(self):
        source = problem(
            document(
                index_sets=_three_big_sets(),
                variable_families=[{"name": "x", "indices": ["s0", "s1", "s2"]}],
            )
        )
        result, elapsed, peak = measured(lambda: expand_problem(source))
        # 1000 x 1000 = 10^6 already passes the ceiling; the product stops there.
        assert refused(result) == u_error(1_000_000, 250_000, "variable_families[0]")
        assert peak < 1 * MIB, peak
        assert elapsed < 0.2, elapsed

    def test_the_path_is_the_entry_that_first_passes_the_ceiling(self):
        """A small family first, then a template of 10⁹ bindings."""
        source = problem(
            document(
                index_sets=_three_big_sets(),
                parameters=[{"name": "c", "indices": ["s0", "s1", "s2"], "default": 1}],
                variable_families=[{"name": "x", "indices": ["s0"]}],
                objective={
                    "direction": "minimize",
                    "linear_terms": [],
                    "linear_term_templates": [
                        {
                            "for_each": ["i in s0", "j in s1", "k in s2"],
                            "coefficient": "c[i,j,k]",
                            "variable": "x[i]",
                        }
                    ],
                },
            )
        )
        result, elapsed, peak = measured(lambda: expand_problem(source))
        # 1000 (the family) + 10^6 (1000 x 1000, stopped before the third set).
        assert refused(result) == u_error(
            1_001_000, 250_000, "objective.linear_term_templates[0]"
        )
        assert peak < 1 * MIB, peak
        assert elapsed < 0.2, elapsed

    def test_a_constraint_template_counts_outer_times_members(self):
        """Π(outer) × (1 + Σ Π(member)): 1000 × (1 + 1000) > 250,000."""
        source = problem(
            document(
                index_sets=_three_big_sets()[:2],
                variable_families=[{"name": "x", "indices": ["s1"]}],
                parameters=[{"name": "a", "indices": ["s0", "s1"], "default": 1}],
                constraint_templates=[
                    {
                        "id": "c",
                        "type": "hard",
                        "for_each": ["i in s0"],
                        "terms": [{"for_each": ["j in s1"], "coefficient": "a[i,j]", "variable": "x[j]"}],
                        "operator": "<=",
                        "rhs": 1,
                    }
                ],
            )
        )
        # 1000 (the family) + 1000 x 1001.
        assert refused(expand_problem(source)) == u_error(
            1_002_000, 250_000, "constraint_templates[0]"
        )

    def test_static_errors_win_over_the_bound(self):
        source = problem(
            document(
                index_sets=_three_big_sets(),
                variable_families=[{"name": "x", "indices": ["s0", "s1", "s2"]}],
                objective={
                    "direction": "minimize",
                    "linear_terms": [],
                    "linear_term_templates": [{"for_each": ["i in s0"], "coefficient": 1, "variable": "q[i]"}],
                },
            )
        )
        result = expand_problem(source)
        assert result.limit_exceeded is False
        assert [(e.code, e.path, e.message) for e in result.errors] == [
            (
                "TEMPLATE_REFERENCE_INVALID",
                "objective.linear_term_templates[0].variable",
                "q is not a declared variable family",
            )
        ]


def _u_document() -> dict:
    """U = 3 + 9 + 3 × (1 + 1) = 18, while W = 3 + 3 + 3 × 2 = 12."""
    return document(
        index_sets=[{"name": "s", "elements": ["a", "b", "c"]}],
        variable_families=[{"name": "x", "indices": ["s"]}],
        objective={
            "direction": "minimize",
            "linear_terms": [],
            "quadratic_term_templates": [
                {
                    "for_each": ["i in s", "j in s"],
                    "where": ["i < j"],
                    "coefficient": 1,
                    "variable1": "x[i]",
                    "variable2": "x[j]",
                }
            ],
        },
        constraint_templates=[
            {
                "id": "c",
                "type": "hard",
                "for_each": ["i in s"],
                "terms": [{"coefficient": 1, "variable": "x[i]"}],
                "operator": "<=",
                "rhs": 1,
            }
        ],
    )


class TestBoundaryU:
    """U equal to the ceiling expands; one less is refused (§14.16 item 4)."""

    def test_at_the_ceiling(self):
        expanded = fits(expand_problem(problem(_u_document()), max_template_bindings=18))
        assert len(expanded.objective.quadratic_terms) == 3
        assert len(expanded.constraints) == 3

    def test_one_below(self):
        result = expand_problem(problem(_u_document()), max_template_bindings=17)
        assert refused(result) == u_error(18, 17, "constraint_templates[0]")

    def test_the_running_sum_names_the_entry(self):
        """Running sums 3, 12, 18: the first one above the ceiling is named."""
        source = problem(_u_document())
        assert refused(expand_problem(source, max_template_bindings=11)) == u_error(
            12, 11, "objective.quadratic_term_templates[0]"
        )
        assert refused(expand_problem(source, max_template_bindings=12)) == u_error(
            18, 12, "constraint_templates[0]"
        )
        assert refused(expand_problem(source, max_template_bindings=2)) == u_error(
            3, 2, "variable_families[0]"
        )


def _w_document(members: int = 4) -> dict:
    """One cardinality constraint of ``members`` members.

    U = members + 1 × (1 + members); W = members (variables) + 1 (the
    constraint) + members + C(members, 2).
    """
    return document(
        index_sets=[{"name": "s", "elements": list(range(members))}],
        variable_families=[{"name": "x", "indices": ["s"]}],
        cardinality_constraint_templates=[
            {
                "id": "k",
                "type": "hard",
                "variables": [{"for_each": ["i in s"], "variable": "x[i]"}],
                "operator": "<=",
                "rhs": 1,
            }
        ],
    )


class TestBoundaryW:
    """W counts entries plus C(k,2) per constraint (§14.9)."""

    def test_at_the_ceiling(self):
        """W = 4 + 1 + 4 + C(4,2) = 15."""
        expanded = fits(expand_problem(problem(_w_document()), max_template_bindings=15))
        assert expanded.cardinality_constraints[0].variables == ["x[0]", "x[1]", "x[2]", "x[3]"]

    def test_one_below(self):
        result = expand_problem(problem(_w_document()), max_template_bindings=14)
        assert refused(result) == w_error(15, 14, "cardinality_constraint_templates[0]")

    def test_w_is_checked_whenever_u_fits(self):
        """U = 9: from 9 to 14 it is W that refuses, below 9 U does.

        W runs 1, 2, 3, 4 (variables), 5 (the constraint), then 6, 8, 11, 15
        (member m adds m); the refusal names the first value above the ceiling.
        """
        source = problem(_w_document())
        reached = {9: 11, 10: 11, 11: 15, 12: 15, 13: 15, 14: 15}
        for limit, work in reached.items():
            assert refused(expand_problem(source, max_template_bindings=limit)) == w_error(
                work, limit, "cardinality_constraint_templates[0]"
            )
        # Running sums 4, 9.
        assert refused(expand_problem(source, max_template_bindings=8)) == u_error(
            9, 8, "cardinality_constraint_templates[0]"
        )

    def test_a_linear_constraint_counts_its_term_pairs(self):
        """Same count for terms: 3 + 1 + 3 + C(3,2) = 10."""
        data = document(
            index_sets=[{"name": "s", "elements": ["a", "b", "c"]}],
            variable_families=[{"name": "x", "indices": ["s"]}],
            constraint_templates=[
                {
                    "id": "c",
                    "type": "hard",
                    "terms": [{"for_each": ["i in s"], "coefficient": 1, "variable": "x[i]"}],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
        )
        fits(expand_problem(problem(data), max_template_bindings=10))
        assert refused(expand_problem(problem(data), max_template_bindings=9)) == w_error(
            10, 9, "constraint_templates[0]"
        )

    def test_a_constraint_left_out_at_a_boundary_is_still_counted(self):
        """Counted while its members are collected, never refunded (§14.9).

        y over a linear t = [0, 1, 2]; for each p the constraint y[p] + y[p+1].
        U = 3 + 3 × (1 + 1 + 1) = 12. W = 3 (variables) + 4 (p=0: 1 + 1 + 2)
        + 4 (p=1) + 2 (p=2: 1 + 1, then y[3] leaves it out) = 13, although
        only 11 units belong to what is generated.
        """
        data = document(
            index_sets=[{"name": "t", "elements": [0, 1, 2], "order": "linear"}],
            variable_families=[{"name": "y", "indices": ["t"]}],
            constraint_templates=[
                {
                    "id": "c",
                    "type": "hard",
                    "for_each": ["p in t"],
                    "terms": [
                        {"coefficient": 1, "variable": "y[p]"},
                        {"coefficient": 1, "variable": "y[p+1]"},
                    ],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
        )
        result = expand_problem(problem(data), max_template_bindings=13)
        assert [c.id for c in fits(result).constraints] == ["c[0]", "c[1]"]
        assert [w.code for w in result.warnings] == ["TEMPLATE_BOUNDARY_SKIPPED"]
        assert refused(expand_problem(problem(data), max_template_bindings=12)) == w_error(
            13, 12, "constraint_templates[0]"
        )

    def _missing_values_document(self) -> dict:
        """W = 4 (variables) + 1 (the one term w has a value for)
        + 1 + 4 + C(4,2) (the cardinality constraint) = 16; U = 13."""
        return document(
            index_sets=[{"name": "s", "elements": list(range(4))}],
            parameters=[{"name": "w", "indices": ["s"], "values": [{"key": [0], "value": 1}]}],
            variable_families=[{"name": "x", "indices": ["s"]}],
            objective={
                "direction": "minimize",
                "linear_terms": [],
                "linear_term_templates": [{"for_each": ["i in s"], "coefficient": "w[i]", "variable": "x[i]"}],
            },
            cardinality_constraint_templates=[
                {
                    "id": "k",
                    "type": "hard",
                    "variables": [{"for_each": ["i in s"], "variable": "x[i]"}],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
        )

    def test_entries_that_are_not_generated_are_not_counted(self):
        """The three terms without a value cost nothing: W = 16 fits."""
        result = expand_problem(problem(self._missing_values_document()), max_template_bindings=16)
        assert result.limit_exceeded is False
        assert [(e.code, e.path) for e in result.errors] == [
            ("PARAMETER_VALUE_MISSING", "objective.linear_term_templates[0].coefficient")
        ] * 3

    def test_the_limit_drops_the_errors_collected_before_it(self):
        """Only the limit error is reported (§14.9)."""
        result = expand_problem(problem(self._missing_values_document()), max_template_bindings=15)
        assert refused(result) == w_error(16, 15, "cardinality_constraint_templates[0]")

    def test_a_single_huge_cardinality_stops_part_way(self):
        """2,000 members: C(2000, 2) ≈ 2·10⁶ > 250,000, while U = 4,001."""
        # W = 2000 (variables) + 1 (the constraint) + m(m+1)/2 after m members:
        # 249,457 at m = 703, 250,161 at m = 704, where it stops.
        result = expand_problem(problem(_w_document(2000)))
        assert refused(result) == w_error(
            250_161, 250_000, "cardinality_constraint_templates[0]"
        )

    def test_stopping_part_way_needs_a_fraction_of_the_memory(self):
        """One cardinality constraint of 10,000 members over 100 variables.

        The members come from 100 member templates, so they -- not the
        variables -- are what a full expansion holds; the limited one stops
        after about 700 of them.
        """
        data = document(
            index_sets=[{"name": "s", "elements": list(range(100))}],
            variable_families=[{"name": "x", "indices": ["s"]}],
            cardinality_constraint_templates=[
                {
                    "id": "k",
                    "type": "hard",
                    "variables": [{"for_each": ["i in s"], "variable": "x[i]"}] * 100,
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
        )
        source = problem(data)
        limited, _, limited_peak = measured(lambda: expand_problem(source))
        # 100 + 1 + m(m+1)/2: 249,672 at m = 706, 250,379 at m = 707.
        assert refused(limited) == w_error(
            250_379, 250_000, "cardinality_constraint_templates[0]"
        )
        full, _, full_peak = measured(lambda: expand_problem(source, max_template_bindings=10**9))
        assert len(fits(full).cardinality_constraints[0].variables) == 10_000
        assert limited_peak * 3 < full_peak, (limited_peak, full_peak)

    def test_a_huge_linear_constraint_stops_part_way(self):
        """10,000 terms over 100 variables (x[i] once per j)."""
        data = document(
            index_sets=[{"name": "s", "elements": list(range(100))}],
            parameters=[{"name": "a", "indices": ["s", "s"], "default": 1}],
            variable_families=[{"name": "x", "indices": ["s"]}],
            constraint_templates=[
                {
                    "id": "c",
                    "type": "hard",
                    "terms": [{"for_each": ["i in s", "j in s"], "coefficient": "a[i,j]", "variable": "x[i]"}],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
        )
        source = problem(data)
        limited, _, limited_peak = measured(lambda: expand_problem(source))
        # The same count as for members: 250,379 at the 707th term.
        assert refused(limited) == w_error(250_379, 250_000, "constraint_templates[0]")
        full, _, full_peak = measured(lambda: expand_problem(source, max_template_bindings=10**9))
        assert len(fits(full).constraints[0].terms) == 10_000
        assert limited_peak * 10 < full_peak, (limited_peak, full_peak)


def _bad_elements(count: int) -> list[str]:
    return [f"bad {k}" for k in range(count)]


def _element_error(set_index: int, j: int) -> tuple[str, str, str]:
    return (
        "INDEX_SET_INVALID",
        f"index_sets[{set_index}].elements[{j}]",
        f'Element "bad {j}" of index set "s{set_index}" must be 1 to 64 letters, '
        "digits, underscores, dots or hyphens",
    )


class TestExpansionErrorCeilings:
    """At most 20 errors per source and 100 in all (§14.11)."""

    def test_twenty_per_source(self):
        data = document(index_sets=[{"name": "s0", "elements": _bad_elements(25)}])
        errors = expand_problem(problem(data)).errors
        found = [(e.code, e.path, e.message) for e in errors]
        assert found[:19] == [_element_error(0, j) for j in range(19)]
        code, path, message = _element_error(0, 19)
        assert found[19] == (
            code,
            path,
            message + "; 5 more errors from index_sets[0] are not listed (INDEX_SET_INVALID ×5)",
        )
        assert len(found) == 20

    def test_one_more_error(self):
        data = document(index_sets=[{"name": "s0", "elements": _bad_elements(21)}])
        errors = expand_problem(problem(data)).errors
        assert len(errors) == 20
        assert errors[-1].message.endswith(
            "; 1 more error from index_sets[0] is not listed (INDEX_SET_INVALID ×1)"
        )

    def test_exactly_twenty_need_no_suffix(self):
        data = document(index_sets=[{"name": "s0", "elements": _bad_elements(20)}])
        errors = expand_problem(problem(data)).errors
        assert [(e.code, e.path, e.message) for e in errors] == [
            _element_error(0, j) for j in range(20)
        ]

    def test_codes_are_counted_in_order_of_appearance(self):
        """Each member: an unknown family, then a fractional coefficient."""
        data = document(
            index_sets=[{"name": "s", "elements": ["a"]}],
            constraint_templates=[
                {
                    "id": "c",
                    "type": "hard",
                    "for_each": ["i in s"],
                    "terms": [{"coefficient": 0.5, "variable": "q[i]"}] * 15,
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
        )
        errors = expand_problem(problem(data)).errors
        assert len(errors) == 20
        assert [e.code for e in errors] == ["TEMPLATE_REFERENCE_INVALID", "NON_INTEGER_INEQUALITY"] * 10
        assert errors[-1].path == "constraint_templates[0].terms[9].coefficient"
        assert errors[-1].message == (
            "Inequality constraint c requires integer coefficients and rhs for "
            "slack encoding, got 0.5; 10 more errors from constraint_templates[0] "
            "are not listed (TEMPLATE_REFERENCE_INVALID ×5, NON_INTEGER_INEQUALITY ×5)"
        )

    def test_one_hundred_in_all(self):
        """Six sources of 25: five fill the list, the sixth is only counted."""
        data = document(
            index_sets=[{"name": f"s{k}", "elements": _bad_elements(25)} for k in range(6)]
        )
        errors = expand_problem(problem(data)).errors
        assert len(errors) == 100
        found = [(e.code, e.path, e.message) for e in errors]
        for k in range(5):
            block = found[20 * k : 20 * (k + 1)]
            assert [path for _, path, _ in block] == [
                f"index_sets[{k}].elements[{j}]" for j in range(20)
            ]
        for k in range(4):
            assert found[20 * k + 19][2] == _element_error(k, 19)[2] + (
                f"; 5 more errors from index_sets[{k}] are not listed (INDEX_SET_INVALID ×5)"
            )
        assert found[99][2] == _element_error(4, 19)[2] + (
            "; 5 more errors from index_sets[4] are not listed (INDEX_SET_INVALID ×5)"
            "; 25 more errors from index_sets[5] are not listed (INDEX_SET_INVALID ×25)"
        )


def _generated_errors_document(count: int, rhs: float) -> dict:
    """``count`` generated inequalities whose rhs comes from a parameter.

    With ``rhs`` 0.5 every one of them is NON_INTEGER_INEQUALITY for the
    ordinary validator (a parameter value is not checked statically,
    §14.10); with 1 the same document is valid.
    """
    return document(
        index_sets=[
            {"name": "g", "elements": list(range(count))},
            {"name": "s", "elements": ["a", "b"]},
        ],
        parameters=[{"name": "r", "indices": ["g"], "default": rhs}],
        variable_families=[{"name": "x", "indices": ["s"]}],
        constraint_templates=[
            {
                "id": "c",
                "type": "hard",
                "for_each": ["k in g"],
                "terms": [{"for_each": ["i in s"], "coefficient": 1, "variable": "x[i]"}],
                "operator": "<=",
                "rhs": "r[k]",
            }
        ],
    )


class TestBoundedValidatorCollector:
    """``validate_expanded`` keeps 20 errors per template (§14.11)."""

    COUNT = 10_000

    def _expansion(self, rhs: float) -> ExpandedProblem:
        result = expand_problem(problem(_generated_errors_document(self.COUNT, rhs)))
        assert result.errors == ()
        return result

    def test_twenty_errors_and_a_count(self):
        expansion = self._expansion(0.5)
        for errors_only in (True, False):
            result = validate_expanded(expansion, errors_only=errors_only)
            assert result.valid is False
            assert result.warnings == []
            assert len(result.errors) == 20
            assert {(e.code, e.path) for e in result.errors} == {
                ("NON_INTEGER_INEQUALITY", "constraint_templates[0].rhs")
            }
            assert result.errors[0].message == (
                "Inequality constraint c[0] requires integer coefficients and rhs "
                "for slack encoding, got 0.5 (generated by constraint_templates[0] "
                "with k=0)"
            )
            assert result.errors[-1].message == (
                "Inequality constraint c[19] requires integer coefficients and rhs "
                "for slack encoding, got 0.5 (generated by constraint_templates[0] "
                f"with k=19); {self.COUNT - 20} more errors from "
                "constraint_templates[0] are not listed "
                f"(NON_INTEGER_INEQUALITY ×{self.COUNT - 20})"
            )

    def test_peak_memory_does_not_grow_with_the_errors(self):
        """The same document with and without 10,000 errors: same peak.

        Kept, 10,000 errors of about 1.2 KB each would add some 12 MB; the
        validator's own per-constraint structures are the same either way.
        """
        failing = self._expansion(0.5)
        passing = self._expansion(1)
        result, _, failing_peak = measured(lambda: validate_expanded(failing, errors_only=True))
        assert len(result.errors) == 20
        result, _, passing_peak = measured(lambda: validate_expanded(passing, errors_only=True))
        assert result.errors == []
        assert failing_peak - passing_peak < 256 * 1024, (failing_peak, passing_peak)
