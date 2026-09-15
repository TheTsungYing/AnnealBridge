"""CLI tests for the spec §30 commands (capabilities, export-schema, solve errors)."""

import json
from importlib import metadata

import pytest
from typer.testing import CliRunner

from annealbridge.interfaces.cli.main import _render_human, app
from annealbridge.models import (
    ClosestCandidate,
    HardViolationRate,
    InfeasibilityDiagnostics,
    OptimizationProblem,
    Solution,
    SolveAttempt,
    SolveError,
    SolveResult,
)
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
            "fujitsu_da",
        }
        # 3a §17.7: rows follow the fixed registry order.
        assert [line.split()[0] for line in lines[1:]] == [
            "exact",
            "simulated_annealing",
            "dwave_qpu",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
            "fujitsu_da",
        ]
        # Local backends: available, enabled, not remote.
        assert rows["exact"].split()[1:4] == ["yes", "yes", "no"]
        assert "max_variables=24" in rows["exact"]
        assert rows["simulated_annealing"].split()[1:4] == ["yes", "yes", "no"]
        # Remote backends are not enabled while allow_remote is off (default).
        assert rows["dwave_qpu"].split()[2] == "no"
        assert rows["leap_hybrid_bqm"].split()[2] == "no"
        assert rows["leap_hybrid_cqm"].split()[2] == "no"
        assert rows["fujitsu_da"].split()[2] == "no"
        # Limits come from the policy, same source as MCP capabilities.
        assert "max_reads=1000" in rows["dwave_qpu"]
        assert "max_annealing_time_us=2000" in rows["dwave_qpu"]
        assert "max_time=300s" in rows["leap_hybrid_bqm"]
        assert "max_time=300s" in rows["leap_hybrid_cqm"]
        assert "max_time=300s" in rows["fujitsu_da"]

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
        assert rows["fujitsu_da"].split()[2] == "yes"

    def test_unavailable_reason_shown_in_parentheses(self):
        # In an environment without configured D-Wave or Fujitsu access the
        # remote rows must carry a parenthesised reason; either classified
        # reason is valid depending on whether dwave-system is installed.
        result = runner.invoke(app, ["capabilities"])
        rows = {
            line.split()[0]: line for line in result.output.splitlines()[1:]
        }
        for name in (
            "dwave_qpu",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
            "fujitsu_da",
        ):
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
            "fujitsu_da",
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

    @pytest.mark.parametrize("command", ["validate", "solve", "recommend"])
    def test_unknown_field_exits_2_naming_the_path(self, tmp_path, command):
        # models/strict.py: an invented field is refused on the type layer,
        # never dropped; the message names the path in the problem's own
        # words and points at the schema.
        problem = json.loads(
            (EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8-sig")
        )
        problem["objective"]["cubic_terms"] = [{"coefficient": 100}]
        path = tmp_path / "unknown_field.json"
        path.write_text(json.dumps(problem))

        result = runner.invoke(app, [command, str(path)])

        assert result.exit_code == 2
        output = _output(result)
        assert "not a valid optimization problem" in output
        assert "objective.cubic_terms: unknown field" in output
        assert "export-schema" in output

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


class TestRecommend:
    """``annealbridge recommend`` (3a §25): a one-line delegation to
    ``service.recommend()``; exit 0 unless the problem itself is invalid."""

    def test_knapsack_table_matches_the_spec_example(self):
        result = runner.invoke(app, ["recommend", str(EXAMPLES_DIR / "knapsack.json")])

        assert result.exit_code == 0
        lines = result.output.splitlines()
        assert lines[0] == "Problem:   knapsack"
        assert lines[1] == (
            "Advisory:  recommendations only; `solve` uses solver.backend as given"
        )
        assert lines[2] == ""
        assert lines[3].split() == ["Rank", "Backend", "Usable", "Model", "Reasons"]
        rows = [line.split() for line in lines[4:10]]
        assert [row[1] for row in rows] == [
            "exact",
            "simulated_annealing",
            "leap_hybrid_cqm",
            "dwave_qpu",
            "leap_hybrid_bqm",
            "fujitsu_da",
        ]
        assert [row[0] for row in rows] == ["1", "2", "3", "4", "5", "6"]
        assert [row[2] for row in rows] == ["yes", "yes", "no", "no", "no", "no"]
        assert [row[3] for row in rows] == ["bqm", "bqm", "cqm", "bqm", "bqm", "bqm"]
        assert lines[4].endswith("R_EXACT_FITS")
        assert lines[5].endswith("R_LOCAL_HEURISTIC")
        assert "R_UNUSABLE, R_NATIVE_CONSTRAINTS   [REMOTE_DISABLED]" in lines[6]
        assert "R_UNUSABLE, R_REMOTE   [REMOTE_DISABLED]" in lines[7]
        assert "R_UNUSABLE, R_REMOTE, R_SINGLE_SAMPLE   [REMOTE_DISABLED]" in lines[8]
        assert "R_UNUSABLE, R_REMOTE   [REMOTE_DISABLED]" in lines[9]
        assert len(lines) == 10

    def test_json_output_is_a_recommendation_result(self):
        from annealbridge.validation import BackendRecommendationResult

        result = runner.invoke(
            app, ["recommend", str(EXAMPLES_DIR / "knapsack.json"), "--json"]
        )

        assert result.exit_code == 0
        parsed = BackendRecommendationResult.model_validate_json(result.output)
        assert parsed.valid is True
        assert parsed.recommendations[0].backend == "exact"
        assert [e.rank for e in parsed.recommendations] == [1, 2, 3, 4, 5, 6]

    def test_invalid_problem_exits_1_with_errors(self, tmp_path):
        problem = json.loads((EXAMPLES_DIR / "knapsack.json").read_text())
        problem["constraints"][0]["terms"][0]["variable"] = "ghost"
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(problem))

        result = runner.invoke(app, ["recommend", str(path)])

        assert result.exit_code == 1
        assert "Valid:     no" in result.output
        assert "[UNKNOWN_VARIABLE]" in result.output
        assert "Rank" not in result.output

    def test_invalid_problem_json_exits_1(self, tmp_path):
        problem = json.loads((EXAMPLES_DIR / "knapsack.json").read_text())
        problem["variables"] = []
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(problem))

        result = runner.invoke(app, ["recommend", str(path), "--json"])

        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["valid"] is False
        assert parsed["recommendations"] == []

    def test_missing_file_exits_2(self, tmp_path):
        result = runner.invoke(app, ["recommend", str(tmp_path / "nope.json")])
        assert result.exit_code == 2
        assert "cannot read" in _output(result)


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


class TestVersion:
    """2026-09-15 consolidation: ``annealbridge --version`` prints the
    installed version and exits 0 before any command body, so no setting is
    ever read."""

    EXPECTED = f"annealbridge {metadata.version('annealbridge')}\n"

    def test_prints_the_installed_version(self):
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert result.output == self.EXPECTED

    def test_works_even_with_invalid_settings(self, monkeypatch):
        monkeypatch.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "0")

        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert result.output == self.EXPECTED

    def test_top_level_help_lists_it_and_keeps_the_description(self):
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "--version" in result.output
        assert "Show the version and exit." in result.output
        assert "Optimization Tool Middleware CLI" in result.output

    def test_no_command_is_still_an_error(self):
        result = runner.invoke(app, [])

        assert result.exit_code == 2
        assert "Missing command" in _output(result)


class TestRenderInfeasibleAttempts:
    """3a §16.4: an attempt without a hard penalty prints ``penalty=-``."""

    def _render(self, penalty, infeasibility=None):
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
            infeasibility=infeasibility,
            message="No feasible solution found",
        )
        return _render_human(problem, result)

    @staticmethod
    def _diagnostics() -> InfeasibilityDiagnostics:
        return InfeasibilityDiagnostics(
            closest_candidate=ClosestCandidate(
                variables={"x2": 0, "x1": 0, "x3": 1},  # deliberately unsorted
                hard_violation_total=2.0,
                constraint_evaluations=[],
            ),
            hard_violation_rates=[
                HardViolationRate(
                    constraint_id="all_three",
                    violated_candidates=7,
                    candidates=8,
                    violated_fraction=0.875,
                ),
                HardViolationRate(
                    constraint_id="budget",
                    violated_candidates=6,
                    candidates=8,
                    violated_fraction=0.75,
                ),
            ],
        )

    def test_none_penalty_prints_a_dash(self):
        text = self._render(None)

        assert "attempt 1: penalty=-, samples=10, unique=4, feasible=0" in text

    def test_float_penalty_still_prints_the_number(self):
        text = self._render(62.0)

        assert "attempt 1: penalty=62, samples=10, unique=4, feasible=0" in text

    def test_diagnostics_print_the_closest_candidate_and_the_rates(self):
        text = self._render(None, self._diagnostics())
        lines = text.splitlines()
        header = "Closest candidate (hard violation total 2):"

        assert header in lines
        # Name-sorted, like the success block's variable listing.
        start = lines.index(header) + 1
        assert lines[start : start + 3] == ["  x1 = 0", "  x2 = 0", "  x3 = 1"]
        assert lines[start + 3] == "Hard constraint violation rates:"
        # 0.875 and 0.75 through _format_number: no trailing ".0" on 75.
        assert lines[start + 4 : start + 6] == [
            "  all_three: 7 / 8 (87.5%)",
            "  budget: 6 / 8 (75%)",
        ]
        # Between the proven line and the message, as spec'd.
        assert lines[lines.index(header) - 1] == "Infeasibility proven: no"
        assert lines[start + 6] == "No feasible solution found"

    def test_without_diagnostics_nothing_extra_is_printed(self):
        text = self._render(None)

        assert "Closest candidate" not in text
        assert "Hard constraint violation rates" not in text


class TestRenderHumanErrorsAndWarnings:
    """Review F-13d / F-22: every human-mode status renders its structured
    errors through the single ``_render_errors`` helper, and warnings are
    never silently dropped."""

    @staticmethod
    def _problem() -> OptimizationProblem:
        return OptimizationProblem.model_validate_json(
            (EXAMPLES_DIR / "knapsack.json").read_text(encoding="utf-8")
        )

    def _render(self, **overrides) -> str:
        fields = {
            "backend": "simulated_annealing",
            "objective_direction": "maximize",
            "solutions": [],
            "attempts": [],
        }
        fields.update(overrides)
        return _render_human(self._problem(), SolveResult(**fields))

    def test_solver_error_renders_code_and_recommended_action(self):
        text = self._render(
            status="solver_error",
            message="sampler exploded",
            errors=[
                SolveError(
                    code="SOLVER_ERROR",
                    message="sampler exploded",
                    recommended_action="retry with fewer reads",
                )
            ],
        )

        assert "Solver error (1):" in text
        assert "[SOLVER_ERROR]: sampler exploded" in text
        assert "recommended action: retry with fewer reads" in text

    def test_solver_error_without_errors_keeps_the_message_fallback(self):
        text = self._render(status="solver_error", message="sampler exploded")

        assert "Solver error: sampler exploded" in text
        assert "Solver error (" not in text

    def test_solver_error_without_errors_or_message_says_unknown(self):
        text = self._render(status="solver_error")

        assert "Solver error: unknown error" in text

    def test_invalid_problem_renders_the_recommended_action(self):
        text = self._render(
            status="invalid_problem",
            backend=None,
            errors=[
                SolveError(
                    code="UNKNOWN_VARIABLE",
                    path="constraints[0].terms[0].variable",
                    message="unknown variable 'ghost'",
                    recommended_action="declare the variable or fix the name",
                )
            ],
        )

        assert "Validation errors (1):" in text
        assert (
            "[UNKNOWN_VARIABLE] constraints[0].terms[0].variable: "
            "unknown variable 'ghost'" in text
        )
        assert "recommended action: declare the variable or fix the name" in text

    def test_resource_limit_exceeded_keeps_its_own_title(self):
        text = self._render(
            status="resource_limit_exceeded",
            errors=[
                SolveError(code="TOO_MANY_VARIABLES", message="24 variables max")
            ],
        )

        assert "Resource limit errors (1):" in text
        assert "[TOO_MANY_VARIABLES]: 24 variables max" in text

    def test_warnings_are_rendered_after_a_successful_solve(self):
        text = self._render(
            status="success",
            backend="exact",
            solutions=[
                Solution(
                    rank=1,
                    variables={"x": 1},
                    objective_value=3.0,
                    soft_violation_score=0.0,
                    ranking_score=3.0,
                    energy=None,
                    sample_count=1,
                    hard_constraints_satisfied=True,
                    constraint_evaluations=[],
                )
            ],
            warnings=[
                SolveError(
                    code="SEED_IGNORED",
                    path="solver.seed",
                    message="seed is ignored",
                    recommended_action="drop solver.seed",
                )
            ],
        )

        assert "Warnings (1):" in text
        assert "[SEED_IGNORED] solver.seed: seed is ignored" in text
        assert "recommended action: drop solver.seed" in text

    def test_warnings_are_rendered_for_an_infeasible_result(self):
        text = self._render(
            status="infeasible",
            message="No feasible solution found",
            warnings=[
                SolveError(code="PENALTY_CAPPED", message="penalty hit the ceiling")
            ],
        )

        assert "Warnings (1):" in text
        assert "[PENALTY_CAPPED]: penalty hit the ceiling" in text
