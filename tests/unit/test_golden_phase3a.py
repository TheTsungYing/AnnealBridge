"""Bit-for-bit consistency with the Phase 3a code (3b spec §25 step 0, §26.1).

``tests/golden/phase3a_compile.json`` was recorded from commit ``20ba640`` by
``tests/golden/record_phase3a_golden.py`` for a fixed list of ``version 1.0``
problems. Every later change (integer variables, bounds-aware estimates, the
CQM integer path, ...) must leave the BQM / CQM output, the estimates, the
penalty scale and the full validation result of those problems *exactly*
unchanged; this test is the proof.

The comparison goes through a ``json`` round-trip on both sides so tuples and
lists compare alike; floats survive the round-trip exactly (spec §25).
"""

import json

import pytest

from tests.golden.record_phase3a_golden import (
    GOLDEN_PATH,
    PROBLEMS,
    RECORDED_FROM,
    snapshot_problem,
)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_recording_provenance(golden):
    assert golden["recorded_from"] == RECORDED_FROM


def test_problem_list_matches_recording(golden):
    # The fixed list and the recording must describe the same problems: a
    # problem added to the script without re-recording (or vice versa) is a
    # mistake, not a silent gap in coverage.
    assert set(golden["problems"]) == set(PROBLEMS)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_output_is_bit_identical_to_phase3a(golden, name):
    problem = PROBLEMS[name]()
    assert problem.version == "1.0"
    current = json.loads(json.dumps(snapshot_problem(problem)))
    assert current == golden["problems"][name]
