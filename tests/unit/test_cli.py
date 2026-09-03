"""CLI tests for the spec §30 commands (capabilities, export-schema, solve errors)."""

import json

import pytest
from typer.testing import CliRunner

from annealbridge.interfaces.cli.main import _render_human, app
from annealbridge.models import OptimizationProblem, SolveAttempt, SolveResult
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
            "leap_hybrid_cqm",
        }
        # 3a §17.7: rows follow the fixed registry order.
        assert [line.split()[0] for line in lines[1:]] == [
            "exact",
            "simulated_annealing",
            "dwave_qpu",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
        ]
        # Local backends: available, enabled, not remote.
        assert rows["exact"].split()[1:4] == ["yes", "yes", "no"]
        assert "max_variables=24" in rows["exact"]
        assert rows["simulated_annealing"].split()[1:4] == ["yes", "yes", "no"]
        # Remote backends are not enabled while allow_remote is off (default).
        assert rows["dwave_qpu"].split()[2] == "no"
        assert rows["leap_hybrid_bqm"].split()[2] == "no"
        assert rows["leap_hybrid_cqm"].split()[2] == "no"
        # Limits come from the policy, same source as MCP capabilities.
        assert "max_reads=1000" in rows["dwave_qpu"]
        assert "max_annealing_time_us=2000" in rows["dwave_qpu"]
        assert "max_time=300s" in rows["leap_hybrid_bqm"]
        assert "max_time=300s" in rows["leap_hybrid_cqm"]

    def test_remote_backends_enabled_when_policy_allows(self, monkeypatch):
        monkeypatch.setenv("ANNEALBRIDGE_ALLOW_REMOTE", "true")
        result = runner.invoke(app, ["capabilities"])
        assert result.exit_code == 0
        rows = {
            line.split()[0]: line for line in result.output.splitlines()[1:]
        }
        assert rows["dwave_qpu"].split()[2] == "yes"
        assert rows["leap_hybrid_bqm"].split()[2] == "yes"
        assert rows["leap_hybrid_cqm"].split()[2] == "yes"

    def test_unavailable_reason_shown_in_parentheses(self):
        # In an environment without configured D-Wave access the remote rows
        # must carry a parenthesised reason; either classified reason is valid
        # depending on whether dwave-system is installed.
        result = runner.invoke(app, ["capabilities"])
        rows = {
            line.split()[0]: line for line in result.output.splitlines()[1:]
        }
        for name in ("dwave_qpu", "leap_hybrid_bqm", "leap_hybrid_cqm"):
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
        for name in (
            "simulated_annealing",
            "exact",
            "dwave_qpu",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
        ):
            assert name in text


class TestValidate:
    """``annealbridge validate`` (3a §10): a one-line delegation to
    ``service.validate()`` with exit codes valid 0 / invalid 1 / load 2."""

    def test_valid_example_exits_0_with_report(self):
        result = runner.invoke(app, ["validate", str(EXAMPLES_DIR / "knapsack.json")])
        assert result.exit_code == 0
        lines = result.output.splitlines()
        assert lines[0] == "Problem:   knapsack"
        assert lines[1].startswith("Backend:   ")
        assert "(model type: bqm)" in lines[1]
        assert lines[2] == "Valid:     yes"
        assert lines[3].startswith("Estimated compiled variables: ")
        assert lines[4].startswith("Objective scale: ")

    def test_backend_override_and_warning_block(self):
        # exact + seed -> SEED_IGNORED (3a §9.3 drift 2), still valid.
        result = runner.invoke(
            app,
            [
                "validate",
                str(EXAMPLES_DIR / "knapsack.json"),
                "--backend",
                "exact",
            ],
        )
        assert result.exit_code == 0
        assert "Backend:   exact  (model type: bqm)" in result.output

    def test_warning_is_rendered_with_recommended_action(self, tmp_path):
        problem = json.loads((EXAMPLES_DIR / "knapsack.json").read_text())
        problem["solver"] = {"backend": "exact", "seed": 42}
        path = tmp_path / "seeded.json"
        path.write_text(json.dumps(problem))

        result = runner.invoke(app, ["validate", str(path)])

        assert result.exit_code == 0
        assert "Warnings (1):" in result.output
        assert "[SEED_IGNORED] solver.seed:" in result.output
        assert "recommended action:" in result.output

    def test_invalid_problem_exits_1_with_errors(self, tmp_path):
        problem = json.loads((EXAMPLES_DIR / "knapsack.json").read_text())
        problem["constraints"][0]["terms"][0]["variable"] = "ghost"
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(problem))

        result = runner.invoke(app, ["validate", str(path)])

        assert result.exit_code == 1
        assert "Valid:     no" in result.output
        assert "Errors (1):" in result.output
        assert "[UNKNOWN_VARIABLE]" in result.output
        assert "Estimated compiled variables" not in result.output

    def test_missing_file_exits_2(self, tmp_path):
        result = runner.invoke(app, ["validate", str(tmp_path / "nope.json")])
        assert result.exit_code == 2
        assert "cannot read" in _output(result)

    def test_schema_error_exits_2(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"name": "no variables here"}))
        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 2
        assert "not a valid optimization problem" in _output(result)

    def test_unknown_backend_override_exits_2(self):
        result = runner.invoke(
            app,
            ["validate", str(EXAMPLES_DIR / "knapsack.json"), "--backend", "nosuch"],
        )
        assert result.exit_code == 2

    def test_json_output_is_a_problem_validation_result(self):
        from annealbridge.validation import ProblemValidationResult

        result = runner.invoke(
            app,
            [
                "validate",
                str(EXAMPLES_DIR / "knapsack.json"),
                "--backend",
                "exact",
                "--json",
            ],
        )
        assert result.exit_code == 0
        parsed = ProblemValidationResult.model_validate_json(result.output)
        assert parsed.valid is True
        assert parsed.model_type == "bqm"
        assert isinstance(parsed.estimated_compiled_variables, int)

    def test_json_output_for_invalid_problem_exits_1(self, tmp_path):
        problem = json.loads((EXAMPLES_DIR / "knapsack.json").read_text())
        problem["variables"] = []
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(problem))

        result = runner.invoke(app, ["validate", str(path), "--json"])

        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["valid"] is False
        assert "NO_VARIABLES" in [e["code"] for e in parsed["errors"]]

    def test_invalid_settings_exit_2(self, monkeypatch):
        monkeypatch.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "-1")
        result = runner.invoke(app, ["validate", str(EXAMPLES_DIR / "knapsack.json")])
        assert result.exit_code == 2
        assert "Error: Invalid server settings" in _output(result)


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


class TestRenderInfeasibleAttempts:
    """3a §16.4: an attempt without a hard penalty prints ``penalty=-``."""

    def _render(self, penalty):
        problem = OptimizationProblem.model_validate_json(
            (EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8")
        )
        result = SolveResult(
            status="infeasible",
            backend="simulated_annealing",
            objective_direction="maximize",
            solutions=[],
            attempts=[
                SolveAttempt(
                    attempt=1,
                    penalty=penalty,
                    samples_received=10,
                    unique_samples=4,
                    feasible_samples=0,
                )
            ],
            message="No feasible solution found",
        )
        return _render_human(problem, result)

    def test_none_penalty_prints_a_dash(self):
        text = self._render(None)

        assert "attempt 1: penalty=-, samples=10, unique=4, feasible=0" in text

    def test_float_penalty_still_prints_the_number(self):
        text = self._render(62.0)

        assert "attempt 1: penalty=62, samples=10, unique=4, feasible=0" in text
