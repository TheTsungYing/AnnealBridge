"""``annealbridge expand``: a schema 1.3 document's templates, expanded.

The command prints the explicit problem the templates stand for, so a user
can see what the server will solve and hand it to a server that only reads
schema 1.2 (schema 1.3 spec §14.13). It expands under the server's own
``max_template_bindings`` and nothing more: no other check runs. stdout
carries only the JSON (UTF-8, LF), so every report — warnings, expansion
errors — goes to stderr. The problem file is read by the loader every
command shares, which also refuses JSON the decoder cannot read without a
traceback.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from annealbridge.interfaces.cli.main import app
from annealbridge.models import OptimizationProblem
from annealbridge.validation import expand_problem
from tests.conftest import EXAMPLES_DIR
from tests.unit.test_cli_parity import _json_stdout, _platform_env, _stderr

runner = CliRunner()

TSP_TEMPLATE = EXAMPLES_DIR / "tsp_template.json"
KNAPSACK = EXAMPLES_DIR / "knapsack.json"
EXAM_TIMETABLING = EXAMPLES_DIR / "exam_timetabling.json"

# The template fields, none of which an expanded document carries.
TEMPLATE_KEYS = {
    "index_sets",
    "parameters",
    "variable_families",
    "constraint_templates",
    "cardinality_constraint_templates",
}
OBJECTIVE_TEMPLATE_KEYS = {"linear_term_templates", "quadratic_term_templates"}

# Neither ASCII nor encodable in cp1252: proves the output is UTF-8 bytes.
UNICODE_DESCRIPTION = "旅行推銷員 — 四座城市"

# Documents the JSON decoder refuses with something other than a
# JSONDecodeError: nesting past the recursion limit, and an integer literal
# past the integer-string conversion limit (5001 digits).
MALICIOUS_JSON = {
    "deep_nesting": "[" * 100_000,
    "long_integer": '{"a": 1' + "0" * 5000 + "}",
}


def _cli(
    *args: str, stdin: bytes | None = None, **env: str
) -> subprocess.CompletedProcess[bytes]:
    """The CLI in a real process, with the platform's own stdio encoding
    (plus ``env``), so stdout is exactly the bytes the command wrote."""
    return subprocess.run(
        [sys.executable, "-m", "annealbridge.interfaces.cli.main", *args],
        input=stdin,
        capture_output=True,
        timeout=120,
        env=_platform_env(**env),
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict, name: str = "problem.json") -> Path:
    path = tmp_path / name
    path.write_bytes(json.dumps(document, ensure_ascii=False).encode("utf-8"))
    return path


def _expected_text(path: Path, version: str) -> str:
    """What ``expand`` must print for ``path``: the library expansion,
    labelled ``version``, as indented JSON and one newline."""
    source = OptimizationProblem.model_validate_json(path.read_bytes())
    expansion = expand_problem(source)
    assert expansion.errors == ()
    expanded = expansion.problem.model_copy(update={"version": version})
    return expanded.model_dump_json(indent=2) + "\n"


@pytest.fixture
def unicode_tsp_template(tmp_path) -> Path:
    document = _load(TSP_TEMPLATE)
    document["description"] = UNICODE_DESCRIPTION
    return _write(tmp_path, document, "unicode_tsp_template.json")


@pytest.fixture
def unused_family(tmp_path) -> Path:
    """tsp_template plus a family no term or constraint uses: its variables
    are left out with an UNUSED_TEMPLATE_VARIABLES warning."""
    document = _load(TSP_TEMPLATE)
    document["variable_families"].append(
        {"name": "y", "indices": ["city"], "type": "binary"}
    )
    return _write(tmp_path, document)


@pytest.fixture
def unknown_parameter(tmp_path) -> Path:
    """tsp_template whose coefficient names an undeclared parameter."""
    document = _load(TSP_TEMPLATE)
    document["objective"]["quadratic_term_templates"][0]["coefficient"] = (
        "distance[i,j]"
    )
    return _write(tmp_path, document)


# ---------------------------------------------------------------------------
# Success: the expanded problem on stdout
# ---------------------------------------------------------------------------


class TestExpandedOutput:
    """Real processes: stdout exactly as written, no newline translation."""

    def test_a_file_is_printed_as_the_library_expansion_labelled_1_2(self):
        result = _cli("expand", str(TSP_TEMPLATE))

        payload = _json_stdout(result)
        assert result.stderr == b""
        assert result.stdout.decode("utf-8") == _expected_text(TSP_TEMPLATE, "1.2")
        # Parses back as an ordinary version 1.2 problem.
        problem = OptimizationProblem.model_validate(payload)
        assert problem.version == "1.2"
        assert not problem.has_templates()
        assert len(problem.variables) == 16
        assert problem.variables[0].name == "x[a,0]"
        assert len(problem.objective.quadratic_terms) == 48
        assert [c.id for c in problem.cardinality_constraints][:2] == [
            "city_once[a]",
            "city_once[b]",
        ]
        # The empty template fields are not written at all.
        assert not TEMPLATE_KEYS & set(payload)
        assert not OBJECTIVE_TEMPLATE_KEYS & set(payload["objective"])

    def test_stdin_gives_the_same_bytes_as_the_file(self):
        from_file = _cli("expand", str(TSP_TEMPLATE))
        from_stdin = _cli("expand", "-", stdin=TSP_TEMPLATE.read_bytes())

        _json_stdout(from_stdin)
        assert from_stdin.stdout == from_file.stdout

    def test_non_ascii_text_is_written_as_utf8_under_a_legacy_code_page(
        self, unicode_tsp_template
    ):
        result = _cli("expand", str(unicode_tsp_template), PYTHONIOENCODING="cp1252")

        payload = _json_stdout(result)
        assert payload["description"] == UNICODE_DESCRIPTION
        assert result.stdout.decode("utf-8") == _expected_text(
            unicode_tsp_template, "1.2"
        )


class TestVersionLabel:
    @pytest.mark.parametrize(
        ("path", "version"), [(KNAPSACK, "1.0"), (EXAM_TIMETABLING, "1.2")]
    )
    def test_a_document_before_1_3_keeps_its_version_and_content(self, path, version):
        result = runner.invoke(app, ["expand", str(path)])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["version"] == version
        assert result.stdout == _expected_text(path, version)
        assert OptimizationProblem.model_validate(payload) == (
            OptimizationProblem.model_validate(_load(path))
        )

    def test_a_1_3_document_without_templates_is_labelled_1_2(self, tmp_path):
        document = _load(EXAM_TIMETABLING)
        document["version"] = "1.3"
        path = _write(tmp_path, document)

        result = runner.invoke(app, ["expand", str(path)])

        assert result.exit_code == 0
        assert json.loads(result.stdout)["version"] == "1.2"
        assert result.stdout == _expected_text(path, "1.2")
        # Exactly the 1.2 example it was made from.
        assert result.stdout == _expected_text(EXAM_TIMETABLING, "1.2")


class TestWarnings:
    def test_expansion_warnings_go_to_stderr_with_their_actions(self, unused_family):
        result = runner.invoke(app, ["expand", str(unused_family)])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        # The unused family's variables were left out of the problem.
        names = [variable["name"] for variable in payload["variables"]]
        assert len(names) == 16
        assert not any(name.startswith("y[") for name in names)
        stderr = _stderr(result)
        assert stderr.startswith("Warnings (1):\n")
        assert "[UNUSED_TEMPLATE_VARIABLES] variable_families[1]: " in stderr
        assert "recommended action: " in stderr
        # Nothing but the JSON on stdout.
        assert "UNUSED_TEMPLATE_VARIABLES" not in result.stdout

    def test_the_warning_reaches_stderr_of_a_real_process(self, unused_family):
        result = _cli("expand", str(unused_family))

        _json_stdout(result)
        assert b"[UNUSED_TEMPLATE_VARIABLES]" in result.stderr


class TestOnlyExpands:
    def test_a_document_validate_refuses_still_expands(self, tmp_path):
        # An explicit term on an undeclared variable: no template names it,
        # so expansion has nothing to say; the general checks are validate's.
        document = _load(TSP_TEMPLATE)
        document["objective"]["linear_terms"].append(
            {"variable": "ghost", "coefficient": 1}
        )
        path = _write(tmp_path, document)

        expanded = runner.invoke(app, ["expand", str(path)])
        validated = runner.invoke(app, ["validate", str(path)])

        assert expanded.exit_code == 0
        terms = json.loads(expanded.stdout)["objective"]["linear_terms"]
        assert terms == [{"variable": "ghost", "coefficient": 1.0}]
        assert validated.exit_code == 1
        assert "UNKNOWN_VARIABLE" in validated.stdout


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


class TestExpansionErrors:
    def test_exit_1_with_the_code_and_template_path_on_stderr(self, unknown_parameter):
        result = runner.invoke(app, ["expand", str(unknown_parameter)])

        assert result.exit_code == 1
        assert result.stdout == ""
        stderr = _stderr(result)
        assert stderr.startswith(
            f"Error: '{unknown_parameter}' could not be expanded.\n\n"
            "Expansion errors (1):\n"
        )
        assert (
            "[TEMPLATE_REFERENCE_INVALID] "
            "objective.quadratic_term_templates[0].coefficient: "
        ) in stderr
        assert "recommended action: " in stderr

    def test_stdin_is_named_in_the_report(self, unknown_parameter):
        result = runner.invoke(
            app, ["expand", "-"], input=unknown_parameter.read_bytes()
        )

        assert result.exit_code == 1
        assert _stderr(result).startswith("Error: '<stdin>' could not be expanded.")

    def test_the_servers_template_bindings_ceiling_applies(self, monkeypatch):
        monkeypatch.setenv("ANNEALBRIDGE_MAX_TEMPLATE_BINDINGS", "10")

        result = runner.invoke(app, ["expand", str(TSP_TEMPLATE)])

        assert result.exit_code == 1
        assert result.stdout == ""
        stderr = _stderr(result)
        assert "Expansion errors (1):" in stderr
        assert "[TEMPLATE_EXPANSION_LIMIT] " in stderr


class TestInputErrors:
    def test_invalid_settings_exit_2(self, monkeypatch):
        monkeypatch.setenv("ANNEALBRIDGE_MAX_TEMPLATE_BINDINGS", "0")

        result = runner.invoke(app, ["expand", str(TSP_TEMPLATE)])

        assert result.exit_code == 2
        assert "Error: Invalid server settings" in _stderr(result)
        assert result.stdout == ""

    def test_a_schema_error_exits_2(self, tmp_path):
        document = _load(TSP_TEMPLATE)
        document["index_sets"][0]["unexpected_field"] = 1

        result = runner.invoke(app, ["expand", str(_write(tmp_path, document))])

        assert result.exit_code == 2
        stderr = _stderr(result)
        assert "is not a valid optimization problem" in stderr
        assert "[UNKNOWN_FIELD] index_sets[0].unexpected_field" in stderr
        assert result.stdout == ""

    def test_invalid_json_exits_2(self):
        result = runner.invoke(app, ["expand", "-"], input=b"{not json")

        assert result.exit_code == 2
        assert "'<stdin>' is not valid JSON" in _stderr(result)

    def test_a_missing_file_exits_2(self, tmp_path):
        result = runner.invoke(app, ["expand", str(tmp_path / "missing.json")])

        assert result.exit_code == 2
        assert "cannot read" in _stderr(result)


class TestJsonTheDecoderRefuses:
    """The loader every command shares turns these into one line and exit 2
    instead of a traceback, and echoes nothing of the document."""

    @pytest.mark.parametrize("command", ["expand", "validate", "recommend", "solve"])
    @pytest.mark.parametrize("kind", sorted(MALICIOUS_JSON))
    def test_every_command_exits_2_from_stdin(self, command, kind):
        result = runner.invoke(app, [command, "-"], input=MALICIOUS_JSON[kind])

        assert result.exit_code == 2
        assert result.exception is None or isinstance(result.exception, SystemExit)
        stderr = _stderr(result)
        assert stderr.startswith("Error: '<stdin>' is not valid JSON: ")
        assert "Traceback" not in stderr
        assert "[[[[" not in stderr
        assert "0000000000" not in stderr
        assert result.stdout == ""

    @pytest.mark.parametrize("kind", sorted(MALICIOUS_JSON))
    def test_a_real_process_prints_no_traceback(self, kind, tmp_path):
        path = tmp_path / f"{kind}.json"
        path.write_text(MALICIOUS_JSON[kind], encoding="utf-8")

        result = _cli("expand", str(path))

        assert result.returncode == 2
        stderr = result.stderr.decode("utf-8")
        assert stderr.startswith(f"Error: '{path}' is not valid JSON: ")
        assert "Traceback" not in stderr
        assert result.stdout == b""


def test_the_help_lists_the_command():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "expand" in result.stdout
