"""Bit-for-bit compatibility of ``version 1.0`` / ``1.1`` behaviour with ``a377f35``.

``tests/golden/compat_a377f35.json`` was recorded from commit ``a377f35`` by
``tests/golden/record_compat_golden.py`` for a fixed list of ``1.0`` and
``1.1`` problems, before the batches that rework the validator, the solution
validator, post-processing, routing and the compilers. Every later change
must leave each recorded section -- compiled models and estimates, the CQM
variable domains, fixed-sample validation, the recommendation, the
post-processing costs and the whitelisted exact solve (infeasibility
diagnostics included) -- *exactly* unchanged for those problems; this test
is the proof. Each problem and each section is its own case, so a failure
names the section that moved.

The comparison goes through a ``json`` round-trip on both sides so tuples and
lists compare alike; floats survive the round-trip exactly.
"""

import json

import pytest

from tests.golden.record_compat_golden import (
    FORMAT_ERROR_DOCUMENTS,
    FORMAT_ERROR_DOCUMENTS_EXPECTED_TO_CHANGE,
    GOLDEN_PATH,
    PROBLEMS,
    RECORDED_FROM,
    SECTIONS,
    format_errors,
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _round_trip(value):
    return json.loads(json.dumps(value))


def test_recording_provenance(golden):
    assert RECORDED_FROM == "a377f35"
    assert golden["recorded_from"] == RECORDED_FROM


def test_problem_list_matches_recording(golden):
    # The fixed list and the recording must describe the same problems: a
    # problem added to the script without re-recording (or vice versa) is a
    # mistake, not a silent gap in coverage.
    assert set(golden["problems"]) == set(PROBLEMS)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_recorded_sections_match_the_script(golden, name):
    assert set(golden["problems"][name]) == {"version", "sample_rows", *SECTIONS}


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_problem_version(golden, name):
    problem = PROBLEMS[name]()
    assert problem.version in ("1.0", "1.1")
    assert problem.version == golden["problems"][name]["version"]


@pytest.mark.parametrize(
    ("name", "section"),
    [(name, section) for name in sorted(PROBLEMS) for section in SECTIONS],
)
def test_section_is_bit_identical_to_a377f35(golden, name, section):
    recorded = golden["problems"][name]
    problem = PROBLEMS[name]()
    # The fixed sample rows are replayed from the recording rather than drawn
    # again, so numpy's random stream is never part of the comparison.
    current = SECTIONS[section](problem, recorded["sample_rows"])
    assert _round_trip(current) == _round_trip(recorded[section])


def test_format_error_list_matches_recording(golden):
    assert set(golden["format_errors"]) == set(FORMAT_ERROR_DOCUMENTS)


@pytest.mark.parametrize("name", sorted(FORMAT_ERROR_DOCUMENTS))
def test_format_errors_are_unchanged(golden, name):
    current = format_errors(FORMAT_ERROR_DOCUMENTS[name]())
    assert _round_trip(current) == _round_trip(golden["format_errors"][name])


def test_format_errors_expected_to_change_are_recorded(golden):
    # Recorded for reference only and deliberately not compared with the
    # current output: the next batch makes "1.2" a valid version and
    # ``cardinality_constraints`` a valid top-level field, so both documents
    # are meant to stop failing. Only the section's presence is pinned, so
    # the recording of the a377f35 behaviour stays in the golden.
    assert set(golden["format_errors_expected_to_change"]) == set(
        FORMAT_ERROR_DOCUMENTS_EXPECTED_TO_CHANGE
    )
