"""CLI commands that give a core-only user what the MCP server gives an agent.

``annealbridge example`` prints the same shipped example files the MCP
server serves as resources, ``capabilities --json`` prints the structure of
``get_optimization_capabilities`` (the comparison over a real MCP client is
in ``tests/mcp/test_cli_capabilities_parity.py``), and ``-`` reads the
problem from stdin so a document can be piped straight into ``solve``,
``validate`` or ``recommend``. None of it may need the ``[mcp]`` extra: the
last tests run the CLI in a subprocess where ``import mcp`` fails.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from annealbridge.interfaces import bundled_examples
from annealbridge.interfaces.bundled_examples import (
    UnknownExampleError,
    example_names,
    example_text,
)
from annealbridge.interfaces.capabilities import OptimizationCapabilities
from annealbridge.interfaces.cli.main import app
from tests.conftest import EXAMPLES_DIR

runner = CliRunner()

PACKAGED_EXAMPLES = Path(bundled_examples.__file__).parent / "examples"
KNAPSACK = EXAMPLES_DIR / "knapsack.json"
UTF8_BOM = b"\xef\xbb\xbf"


def _stderr(result) -> str:
    """stderr, or the mixed output where an older runner does not split it."""
    try:
        return result.stderr
    except ValueError:
        return result.output


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_every_registered_example_is_a_shipped_file_and_vice_versa(self):
        assert sorted(f"{name}.json" for name in example_names()) == sorted(
            path.name for path in PACKAGED_EXAMPLES.glob("*.json")
        )

    @pytest.mark.parametrize("name", example_names())
    def test_example_text_is_the_file_byte_for_byte(self, name):
        path = PACKAGED_EXAMPLES / f"{name}.json"
        assert example_text(name).encode("utf-8") == path.read_bytes()

    def test_an_unknown_name_raises(self):
        with pytest.raises(UnknownExampleError):
            example_text("nope")
        # A KeyError, so a caller treating the registry as a mapping works.
        with pytest.raises(KeyError):
            example_text("../pyproject")


# ---------------------------------------------------------------------------
# annealbridge example
# ---------------------------------------------------------------------------


class TestExampleCommand:
    def test_without_a_name_lists_every_example_with_a_summary(self):
        result = runner.invoke(app, ["example"])

        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert [line.split()[0] for line in lines] == list(example_names())
        for line, entry in zip(lines, bundled_examples.EXAMPLES, strict=True):
            summary = line[len(entry.name) :].strip()
            assert summary
            assert summary in entry.title

    @pytest.mark.parametrize("name", example_names())
    def test_a_name_prints_the_file_byte_for_byte(self, name):
        result = runner.invoke(app, ["example", name])

        assert result.exit_code == 0
        # No trailing newline added, no newline translation: the bytes are
        # the packaged file's, which are the repository file's.
        assert result.stdout_bytes == (PACKAGED_EXAMPLES / f"{name}.json").read_bytes()
        assert result.stdout_bytes == (EXAMPLES_DIR / f"{name}.json").read_bytes()

    def test_an_unknown_name_lists_the_available_ones_and_exits_2(self):
        result = runner.invoke(app, ["example", "nope"])

        assert result.exit_code == 2
        assert result.stdout == ""
        stderr = _stderr(result)
        assert "nope" in stderr
        for name in example_names():
            assert name in stderr


# ---------------------------------------------------------------------------
# annealbridge capabilities --json
# ---------------------------------------------------------------------------


class TestCapabilitiesJson:
    def test_prints_the_capabilities_model_without_the_schema(self):
        result = runner.invoke(app, ["capabilities", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert set(payload) == set(OptimizationCapabilities.model_fields)
        assert payload["problem_json_schema"] is None
        # Round-trips through the model the MCP tool returns.
        OptimizationCapabilities.model_validate(payload)
        assert [backend["name"] for backend in payload["backends"]] == [
            "exact",
            "simulated_annealing",
            "tabu",
            "simulated_bifurcation",
            "dwave_qpu",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
            "fujitsu_da",
        ]

    def test_without_json_the_output_is_still_the_table(self):
        result = runner.invoke(app, ["capabilities"])

        assert result.exit_code == 0
        assert result.stdout.splitlines()[0].split() == [
            "Backend",
            "Available",
            "Enabled",
            "Remote",
            "Limits",
        ]


# ---------------------------------------------------------------------------
# '-' reads the problem from stdin
# ---------------------------------------------------------------------------


class TestStdin:
    @pytest.mark.parametrize("command", ["validate", "recommend"])
    def test_stdin_gives_the_same_result_as_the_file(self, command):
        from_file = runner.invoke(app, [command, str(KNAPSACK), "--json"])
        from_stdin = runner.invoke(
            app, [command, "-", "--json"], input=KNAPSACK.read_bytes()
        )

        assert from_file.exit_code == 0
        assert from_stdin.exit_code == 0
        assert json.loads(from_stdin.stdout) == json.loads(from_file.stdout)

    def test_solve_reads_stdin(self):
        result = runner.invoke(
            app, ["solve", "-", "--json"], input=KNAPSACK.read_bytes()
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "success"
        assert payload["solutions"][0]["objective_value"] == 17

    def test_solve_reads_stdin_in_the_human_report_too(self):
        result = runner.invoke(app, ["solve", "-"], input=KNAPSACK.read_bytes())

        assert result.exit_code == 0
        assert "Problem:   knapsack" in result.stdout

    def test_a_utf8_bom_on_stdin_is_accepted(self):
        result = runner.invoke(
            app, ["validate", "-", "--json"], input=UTF8_BOM + KNAPSACK.read_bytes()
        )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["valid"] is True

    def test_non_ascii_text_on_stdin_is_decoded_as_utf8(self):
        problem = json.loads(KNAPSACK.read_text(encoding="utf-8"))
        problem["description"] = "背包問題 — café"
        encoded = json.dumps(problem, ensure_ascii=False).encode("utf-8")

        result = runner.invoke(app, ["validate", "-", "--json"], input=encoded)

        assert result.exit_code == 0
        assert json.loads(result.stdout)["valid"] is True

    def test_invalid_json_on_stdin_names_stdin(self):
        result = runner.invoke(app, ["solve", "-"], input=b"{not json")

        assert result.exit_code == 2
        assert "'<stdin>' is not valid JSON" in _stderr(result)

    def test_bytes_that_are_not_utf8_name_stdin(self):
        result = runner.invoke(app, ["validate", "-"], input=b"\xff\xfe{}")

        assert result.exit_code == 2
        assert "'<stdin>' is not valid UTF-8" in _stderr(result)

    def test_a_schema_error_on_stdin_names_stdin(self):
        problem = json.loads(KNAPSACK.read_text(encoding="utf-8"))
        problem["unexpected_field"] = 1

        result = runner.invoke(
            app, ["validate", "-"], input=json.dumps(problem).encode("utf-8")
        )

        assert result.exit_code == 2
        assert "'<stdin>' is not a valid optimization problem" in _stderr(result)

    def test_the_help_mentions_stdin(self):
        result = runner.invoke(app, ["solve", "--help"])

        assert result.exit_code == 0
        assert "stdin" in result.stdout


# ---------------------------------------------------------------------------
# Core-only install: the same commands with ``import mcp`` failing
# ---------------------------------------------------------------------------

# Runs the CLI in a fresh interpreter in which the ``mcp`` package cannot be
# imported, as on an install without the extra. Blocking the import (rather
# than relying on the environment lacking it) keeps the test meaningful in
# the development environment, which does have it.
_CORE_ONLY_CLI = """\
import sys
sys.modules["mcp"] = None
from annealbridge.interfaces.cli.main import main
sys.argv = ["annealbridge", *sys.argv[1:]]
main()
"""


def _platform_env(**overrides: str) -> dict[str, str]:
    """The inherited environment without any stdio encoding override, so a
    child process encodes its text streams the way the platform does (the
    locale's code page on Windows when piped), plus ``overrides``."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in ("PYTHONIOENCODING", "PYTHONUTF8")
    }
    env.update(overrides)
    return env


def _core_only(*args: str) -> subprocess.CompletedProcess[bytes]:
    # No encoding is pinned: every machine-readable output is written as
    # UTF-8 bytes, and the example list is ASCII.
    return subprocess.run(
        [sys.executable, "-c", _CORE_ONLY_CLI, *args],
        capture_output=True,
        timeout=120,
        env=_platform_env(),
    )


class TestWithoutTheMcpExtra:
    def test_the_blocked_import_really_fails(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                'import sys; sys.modules["mcp"] = None; '
                "import annealbridge.interfaces.mcp",
            ],
            capture_output=True,
            timeout=120,
        )
        assert result.returncode != 0
        assert b"ModuleNotFoundError" in result.stderr

    def test_example_prints_the_file_byte_for_byte(self):
        result = _core_only("example", "knapsack")

        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        # Real process stdout: on Windows this also proves no \r\n appears.
        assert result.stdout == KNAPSACK.read_bytes()

    def test_example_lists_every_example(self):
        result = _core_only("example")

        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        names = [line.split()[0] for line in result.stdout.decode().splitlines()]
        assert names == list(example_names())

    def test_capabilities_json(self):
        result = _core_only("capabilities", "--json")

        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        payload = json.loads(result.stdout)
        assert set(payload) == set(OptimizationCapabilities.model_fields)
        assert payload["problem_json_schema"] is None


# ---------------------------------------------------------------------------
# Machine-readable output is UTF-8 whatever the platform's stdio encoding
# ---------------------------------------------------------------------------

# CJK and an em dash: neither ASCII nor encodable in cp1252 (the code page of
# a GitHub Windows runner). The capabilities descriptions carry em dashes too.
UNICODE_NAME = "物品—甲"


def _cli(*args: str, **env: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "annealbridge.interfaces.cli.main", *args],
        capture_output=True,
        timeout=120,
        env=_platform_env(**env),
    )


def _json_stdout(result: subprocess.CompletedProcess[bytes], code: int = 0):
    """stdout decoded strictly as UTF-8, with LF newlines, then parsed."""
    assert result.returncode == code, result.stderr.decode("utf-8", "replace")
    text = result.stdout.decode("utf-8")
    assert "\r" not in text
    assert text.endswith("}\n")
    return json.loads(text)


@pytest.fixture
def unicode_knapsack(tmp_path) -> Path:
    """The knapsack example with a non-ASCII variable (in the optimum) and
    constraint id."""
    text = KNAPSACK.read_text(encoding="utf-8")
    text = text.replace('"item_a"', f'"{UNICODE_NAME}"')
    text = text.replace('"capacity"', f'"容量{UNICODE_NAME}"')
    path = tmp_path / "unicode_knapsack.json"
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.fixture
def unknown_unicode_variable(tmp_path) -> Path:
    """The knapsack example with a term on an undeclared non-ASCII variable,
    which the error message echoes."""
    problem = json.loads(KNAPSACK.read_text(encoding="utf-8"))
    problem["objective"]["linear_terms"].append(
        {"variable": UNICODE_NAME, "coefficient": 1}
    )
    path = tmp_path / "unknown_variable.json"
    path.write_bytes(json.dumps(problem, ensure_ascii=False).encode("utf-8"))
    return path


def _assert_solved_with_unicode_names(payload) -> None:
    assert payload["status"] == "success"
    best = payload["solutions"][0]
    assert best["objective_value"] == 17
    assert best["variables"][UNICODE_NAME] == 1
    assert f"容量{UNICODE_NAME}" in {
        evaluation["constraint_id"] for evaluation in best["constraint_evaluations"]
    }


class TestMachineReadableOutputIsUtf8:
    """Real processes, no ``PYTHONIOENCODING``: on Windows a piped stdout
    otherwise follows the console code page and translates newlines."""

    def test_capabilities_json(self):
        payload = _json_stdout(_cli("capabilities", "--json"))

        assert set(payload) == set(OptimizationCapabilities.model_fields)
        assert any(
            not backend["description"].isascii() for backend in payload["backends"]
        )

    def test_export_schema(self):
        assert "properties" in _json_stdout(_cli("export-schema"))

    def test_solve_json(self, unicode_knapsack):
        payload = _json_stdout(_cli("solve", str(unicode_knapsack), "--json"))

        _assert_solved_with_unicode_names(payload)

    @pytest.mark.parametrize("command", ["validate", "recommend"])
    def test_error_messages_keep_the_name(self, command, unknown_unicode_variable):
        payload = _json_stdout(
            _cli(command, str(unknown_unicode_variable), "--json"), code=1
        )

        assert payload["valid"] is False
        assert any(UNICODE_NAME in error["message"] for error in payload["errors"])

    # A legacy code page forced on the text layer, so the bytes are proven to
    # bypass it on every platform, not only where the default is not UTF-8.

    def test_capabilities_json_under_a_legacy_code_page(self):
        result = _cli("capabilities", "--json", PYTHONIOENCODING="cp1252")

        assert set(_json_stdout(result)) == set(OptimizationCapabilities.model_fields)

    def test_solve_json_under_a_legacy_code_page(self, unicode_knapsack):
        result = _cli(
            "solve", str(unicode_knapsack), "--json", PYTHONIOENCODING="cp1252"
        )

        _assert_solved_with_unicode_names(_json_stdout(result))
