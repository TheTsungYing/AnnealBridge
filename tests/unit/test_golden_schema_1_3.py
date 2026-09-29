"""Frozen schema ``1.3`` behaviour (templates, spec §14.15).

``tests/golden/schema_1_3_expand.json`` was recorded by
``tests/golden/record_schema_1_3_golden.py`` *after* the templates were
implemented, from the working tree ``c69f704`` plus the schema 1.3 changes,
for a fixed list of ``"version": "1.3"`` template problems and a fixed list
of documents the expansion refuses. It freezes that output so a later change
cannot move it unnoticed; it is not evidence the output is correct (spec
§14.16 and the ``test_template_*`` modules are). Every recorded section --
the expanded problem, the expansion warnings, the compiled models, estimates
and mapped validations of the snapshot, and the whitelisted exact solve of
the template problem -- and every refused document's ``[code, path,
message]`` list must stay exactly unchanged. Each problem, section and
document is its own case, so a failure names what moved.

The comparison is on canonical JSON text (``sort_keys``) of both sides, so
tuples and lists compare alike while ``1`` and ``1.0`` do not: the expanded
problem's generated entries must keep the types ``model_validate`` gives.
Floats survive the round-trip exactly. Nothing is re-recorded here.
"""

import json

import pytest

from tests.golden.record_schema_1_3_golden import (
    ERROR_DOCUMENTS,
    GOLDEN_PATH,
    PROBLEMS,
    RECORDED_FROM,
    SECTIONS,
    VERSION,
    expansion_errors,
)

# Spec §14.10: every new error code, and the version gate, has a document.
NEW_ERROR_CODES = {
    "TEMPLATE_REFERENCE_INVALID",
    "INDEX_SET_INVALID",
    "PARAMETER_TABLE_INVALID",
    "PARAMETER_VALUE_MISSING",
    "DUPLICATE_TEMPLATE_NAME",
    "TEMPLATE_EXPANSION_LIMIT",
}
NEW_WARNING_CODES = {
    "EMPTY_TEMPLATE_EXPANSION",
    "TEMPLATE_BOUNDARY_SKIPPED",
    "TEMPLATE_TERMS_MERGED",
    "UNUSED_TEMPLATE_VARIABLES",
}


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def test_recording_provenance(golden):
    assert RECORDED_FROM == "c69f704+schema-1.3"
    assert VERSION == "1.3"
    assert golden["recorded_from"] == RECORDED_FROM


def test_problem_list_matches_recording(golden):
    # The fixed list and the recording must describe the same problems: a
    # problem added to the script without re-recording (or vice versa) is a
    # mistake, not a silent gap in coverage.
    assert set(golden["problems"]) == set(PROBLEMS)


def test_error_document_list_matches_recording(golden):
    assert set(golden["expansion_errors"]) == set(ERROR_DOCUMENTS)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_recorded_sections_match_the_script(golden, name):
    assert set(golden["problems"][name]) == {"version", *SECTIONS}


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_problem_version(golden, name):
    problem = PROBLEMS[name]()
    assert problem.version == "1.3"
    assert problem.has_templates()
    assert golden["problems"][name]["version"] == "1.3"
    # Expansion keeps the version.
    assert golden["problems"][name]["expanded"]["version"] == "1.3"


@pytest.mark.parametrize(
    ("name", "section"),
    [(name, section) for name in sorted(PROBLEMS) for section in SECTIONS],
)
def test_section_is_bit_identical_to_the_recording(golden, name, section):
    recorded = golden["problems"][name][section]
    current = SECTIONS[section](PROBLEMS[name]())
    assert _canonical(current) == _canonical(recorded)


@pytest.mark.parametrize("name", sorted(ERROR_DOCUMENTS))
def test_expansion_errors_are_bit_identical_to_the_recording(golden, name):
    build, ceiling = ERROR_DOCUMENTS[name]
    current = expansion_errors(build(), ceiling)
    assert _canonical(current) == _canonical(golden["expansion_errors"][name])


def test_the_recording_covers_every_new_code(golden):
    # A guard on the golden's own coverage (spec §14.15): the refused
    # documents hold every new error code, the version gate and a truncated
    # error list; the problems raise every new warning code.
    documents = golden["expansion_errors"].values()
    error_codes = {code for entry in documents for code, _, _ in entry["errors"]}
    assert NEW_ERROR_CODES | {"FEATURE_REQUIRES_NEWER_VERSION"} <= error_codes
    assert any(
        "more errors from" in message and "are not listed" in message
        for entry in documents
        for _, _, message in entry["errors"]
    )
    warning_codes = {
        code
        for entry in golden["problems"].values()
        for code, _, _ in entry["expansion_warnings"]
    }
    assert warning_codes == NEW_WARNING_CODES
