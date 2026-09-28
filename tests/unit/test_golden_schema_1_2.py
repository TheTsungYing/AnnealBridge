"""Frozen schema ``1.2`` behaviour (``cardinality_constraints``, spec §12.2).

``tests/golden/schema_1_2_compile.json`` was recorded by
``tests/golden/record_schema_1_2_golden.py`` *after* ``cardinality_constraints``
was implemented, from the working tree ``a377f35`` plus the schema 1.2
changes, for a fixed list of ``"version": "1.2"`` problems. It freezes that
output so a later change cannot move it unnoticed; it is not evidence the
output is correct (spec §13 and the ``test_cardinality_*`` modules are). Every
recorded section -- compiled models and estimates, the CQM variable domains,
fixed-sample validation, the recommendation, the post-processing costs, the
whitelisted exact solve (infeasibility diagnostics included) and the BQM
trace kinds (slack or pairwise) -- must stay exactly unchanged. Each problem
and each section is its own case, so a failure names the section that moved.

The comparison goes through a ``json`` round-trip on both sides so tuples and
lists compare alike; floats survive the round-trip exactly.
"""

import json

import pytest

from tests.golden.record_schema_1_2_golden import (
    GOLDEN_PATH,
    PROBLEMS,
    RECORDED_FROM,
    SECTIONS,
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _round_trip(value):
    return json.loads(json.dumps(value))


def test_recording_provenance(golden):
    assert RECORDED_FROM == "a377f35+schema-1.2"
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
    assert problem.version == "1.2"
    assert golden["problems"][name]["version"] == "1.2"


@pytest.mark.parametrize(
    ("name", "section"),
    [(name, section) for name in sorted(PROBLEMS) for section in SECTIONS],
)
def test_section_is_bit_identical_to_the_recording(golden, name, section):
    recorded = golden["problems"][name]
    problem = PROBLEMS[name]()
    # The fixed sample rows are replayed from the recording rather than drawn
    # again, so numpy's random stream is never part of the comparison.
    current = SECTIONS[section](problem, recorded["sample_rows"])
    assert _round_trip(current) == _round_trip(recorded[section])
