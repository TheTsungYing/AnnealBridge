"""CLI tests for the spec §30 commands (capabilities, export-schema, solve errors)."""

import json

import pytest
from typer.testing import CliRunner

from annealbridge.interfaces.cli.main import app
from tests.conftest import EXAMPLES_DIR

runner = CliRunner()


def _output(result) -> str:
    """Combined stdout+stderr regardless of the installed click version."""
    text = result.output
    try:
        text += result.stderr
    except ValueError:
        pass  # older click mixes stderr into output
    return text


class TestCapabilities:
    def test_table_with_default_policy(self):
        result = runner.invoke(app, ["capabilities"])
        assert result.exit_code == 0
        lines = result.output.splitlines()
        assert lines[0].split() == ["Backend", "Available", "Enabled", "Remote", "Limits"]

        rows = {line.split()[0]: line for line in lines[1:]}
        assert set(rows) == {
            "exact",
            "simulated_annealing",
            "dwave_qpu",
            "leap_hybrid_bqm",
        }
        # Local backends: available, enabled, not remote.
        assert rows["exact"].split()[1:4] == ["yes", "yes", "no"]
        assert "max_variables=24" in rows["exact"]
        assert rows["simulated_annealing"].split()[1:4] == ["yes", "yes", "no"]
        # Remote backends are not enabled while allow_remote is off (default).
        assert rows["dwave_qpu"].split()[2] == "no"
        assert rows["leap_hybrid_bqm"].split()[2] == "no"
        # Limits come from the policy, same source as MCP capabilities.
        assert "max_reads=1000" in rows["dwave_qpu"]
        assert "max_annealing_time_us=2000" in rows["dwave_qpu"]
        assert "max_time=300s" in rows["leap_hybrid_bqm"]

    def test_remote_backends_enabled_when_policy_allows(self, monkeypatch):
        monkeypatch.setenv("ANNEALBRIDGE_ALLOW_REMOTE", "true")
        result = runner.invoke(app, ["capabilities"])
        assert result.exit_code == 0
        rows = {
            line.split()[0]: line for line in result.output.splitlines()[1:]
        }
        assert rows["dwave_qpu"].split()[2] == "yes"
        assert rows["leap_hybrid_bqm"].split()[2] == "yes"

    def test_unavailable_reason_shown_in_parentheses(self):
        # In an environment without configured D-Wave access the remote rows
        # must carry a parenthesised reason; either classified reason is valid
        # depending on whether dwave-system is installed.
        result = runner.invoke(app, ["capabilities"])
        rows = {
            line.split()[0]: line for line in result.output.splitlines()[1:]
        }
        for name in ("dwave_qpu", "leap_hybrid_bqm"):
            if rows[name].split()[1] == "no":
                assert "(" in rows[name] and rows[name].endswith(")")

    def test_no_config_values_leak(self, monkeypatch):
        monkeypatch.setenv("DWAVE_API_TOKEN", "DEV-FAKE-TOKEN-1234567890abcdefghij")
        result = runner.invoke(app, ["capabilities"])
        assert "DEV-FAKE-TOKEN" not in result.output
        assert "127.0.0.1" not in result.output
        assert "8000" not in result.output


class TestSolveErrors:
    def test_remote_backend_disabled_message(self):
        result = runner.invoke(
            app,
            ["solve", str(EXAMPLES_DIR / "knapsack.json"), "--backend", "dwave_qpu"],
        )
        assert result.exit_code == 1
        assert "Status:    backend_unavailable" in result.output
        assert "[REMOTE_DISABLED]" in result.output
        assert "remote solving is disabled" in result.output
        assert "recommended action:" in result.output

    def test_unknown_backend_lists_all_known_backends(self):
        result = runner.invoke(
            app,
            ["solve", str(EXAMPLES_DIR / "knapsack.json"), "--backend", "nosuch"],
        )
        assert result.exit_code == 2
        text = _output(result)
        for name in ("simulated_annealing", "exact", "dwave_qpu", "leap_hybrid_bqm"):
            assert name in text


class TestExportSchema:
    def test_outputs_problem_json_schema(self):
        result = runner.invoke(app, ["export-schema"])
        assert result.exit_code == 0
        schema = json.loads(result.output)
        assert "properties" in schema
        assert "variables" in schema["properties"]


class TestInvalidSettings:
    """A bad ANNEALBRIDGE_* value is a configuration error: a clear message
    on stderr and exit code 2, never a traceback."""

    @pytest.mark.parametrize(
        "args",
        [
            ["capabilities"],
            ["solve", str(EXAMPLES_DIR / "knapsack.json")],
        ],
        ids=["capabilities", "solve"],
    )
    def test_reports_the_variable_and_exits_2(self, monkeypatch, args):
        monkeypatch.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "-1")

        result = runner.invoke(app, args)

        assert result.exit_code == 2
        text = _output(result)
        assert "Error: Invalid server settings" in text
        assert "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES" in text
        assert "Traceback" not in text
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_export_schema_does_not_need_settings(self, monkeypatch):
        monkeypatch.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "-1")

        result = runner.invoke(app, ["export-schema"])

        assert result.exit_code == 0
