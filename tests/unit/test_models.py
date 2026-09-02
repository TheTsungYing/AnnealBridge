"""Schema-level tests for the domain models.

Cross-field business validation (duplicate variables, unknown variables,
hard-with-weight, ...) belongs to the Problem Validator and is tested in
test_problem_validator.py, not here.
"""

import pytest
from pydantic import ValidationError

from annealbridge.models import (
    Constraint,
    DWaveQPUOptions,
    LeapHybridBQMOptions,
    Objective,
    OptimizationProblem,
    ProblemError,
    SolverPreferences,
    SolveError,
    SolveResult,
    Variable,
)


def make_problem_dict() -> dict:
    """A complete, legal OptimizationProblem payload."""
    return {
        "version": "1.0",
        "name": "test problem",
        "description": "a small test problem",
        "variables": [
            {"name": "x1", "description": "first"},
            {"name": "x2", "type": "binary"},
            {"name": "x3"},
        ],
        "objective": {
            "direction": "maximize",
            "linear_terms": [
                {"variable": "x1", "coefficient": 10},
                {"variable": "x2", "coefficient": 8},
            ],
            "quadratic_terms": [
                {"variable1": "x1", "variable2": "x2", "coefficient": -3},
            ],
            "constant": 1.5,
        },
        "constraints": [
            {
                "id": "cap",
                "type": "hard",
                "terms": [
                    {"variable": "x1", "coefficient": 6},
                    {"variable": "x2", "coefficient": 5},
                ],
                "operator": "<=",
                "rhs": 10,
            },
            {
                "id": "prefer_x3",
                "description": "soft preference",
                "type": "soft",
                "terms": [{"variable": "x3", "coefficient": 1}],
                "operator": "==",
                "rhs": 1,
                "weight": 4.0,
            },
        ],
        "solver": {
            "backend": "exact",
            "num_reads": 50,
            "seed": 42,
        },
    }


class TestValidParse:
    def test_full_problem_parses(self):
        problem = OptimizationProblem.model_validate(make_problem_dict())
        assert problem.version == "1.0"
        assert problem.name == "test problem"
        assert [v.name for v in problem.variables] == ["x1", "x2", "x3"]
        assert all(v.type == "binary" for v in problem.variables)
        assert problem.objective.direction == "maximize"
        assert problem.objective.linear_terms[0].variable == "x1"
        assert problem.objective.linear_terms[0].coefficient == 10
        assert problem.objective.quadratic_terms[0].variable2 == "x2"
        assert problem.objective.constant == 1.5
        assert problem.constraints[0].operator == "<="
        assert problem.constraints[0].rhs == 10
        assert problem.constraints[0].terms[1].coefficient == 5
        assert problem.constraints[1].type == "soft"
        assert problem.constraints[1].weight == 4.0
        assert problem.solver.backend == "exact"
        assert problem.solver.num_reads == 50
        assert problem.solver.seed == 42

    def test_json_roundtrip(self):
        problem = OptimizationProblem.model_validate(make_problem_dict())
        again = OptimizationProblem.model_validate_json(problem.model_dump_json())
        assert again == problem

    def test_solver_defaults_when_omitted(self):
        data = make_problem_dict()
        del data["solver"]
        problem = OptimizationProblem.model_validate(data)
        assert problem.solver == SolverPreferences()


class TestVersion:
    def test_version_defaults_to_1_0(self):
        data = make_problem_dict()
        del data["version"]
        assert OptimizationProblem.model_validate(data).version == "1.0"

    @pytest.mark.parametrize("bad", ["2.0", "1.1", "1", ""])
    def test_other_versions_rejected(self, bad):
        data = make_problem_dict()
        data["version"] = bad
        with pytest.raises(ValidationError):
            OptimizationProblem.model_validate(data)


class TestIllegalLiterals:
    @pytest.mark.parametrize("bad_op", ["<", ">", "!=", "="])
    def test_invalid_operator_rejected(self, bad_op):
        data = make_problem_dict()
        data["constraints"][0]["operator"] = bad_op
        with pytest.raises(ValidationError):
            OptimizationProblem.model_validate(data)

    @pytest.mark.parametrize("bad_type", ["medium", "HARD", ""])
    def test_invalid_constraint_type_rejected(self, bad_type):
        data = make_problem_dict()
        data["constraints"][0]["type"] = bad_type
        with pytest.raises(ValidationError):
            OptimizationProblem.model_validate(data)

    @pytest.mark.parametrize("bad_dir", ["max", "min", "MINIMIZE"])
    def test_invalid_direction_rejected(self, bad_dir):
        data = make_problem_dict()
        data["objective"]["direction"] = bad_dir
        with pytest.raises(ValidationError):
            OptimizationProblem.model_validate(data)

    @pytest.mark.parametrize("bad_vartype", ["integer", "real", "spin"])
    def test_invalid_variable_type_rejected(self, bad_vartype):
        with pytest.raises(ValidationError):
            Variable.model_validate({"name": "x", "type": bad_vartype})

    @pytest.mark.parametrize("bad_backend", ["qpu", "cqm", "sa"])
    def test_invalid_backend_rejected(self, bad_backend):
        with pytest.raises(ValidationError):
            SolverPreferences.model_validate({"backend": bad_backend})

    def test_invalid_solve_status_rejected(self):
        with pytest.raises(ValidationError):
            SolveResult.model_validate(
                {
                    "status": "done",
                    "backend": None,
                    "objective_direction": None,
                    "solutions": [],
                    "attempts": [],
                }
            )


class TestDefaults:
    def test_solver_preferences_defaults(self):
        prefs = SolverPreferences()
        assert prefs.backend == "simulated_annealing"
        assert prefs.num_reads == 100
        assert prefs.num_sweeps == 1000
        assert prefs.seed is None
        assert prefs.top_k == 5
        assert prefs.max_retries == 3
        assert prefs.penalty_multiplier == 2.0
        assert prefs.dwave_qpu is None
        assert prefs.leap_hybrid_bqm is None

    def test_dwave_qpu_options_defaults(self):
        options = DWaveQPUOptions()
        assert options.annealing_time_us is None
        assert options.chain_strength is None
        assert options.auto_scale is True

    def test_leap_hybrid_bqm_options_defaults(self):
        options = LeapHybridBQMOptions()
        assert options.time_limit_seconds is None

    def test_variable_defaults(self):
        var = Variable(name="x")
        assert var.type == "binary"
        assert var.description is None

    def test_constraint_weight_defaults_to_none(self):
        constraint = Constraint.model_validate(
            {
                "id": "c1",
                "type": "hard",
                "terms": [{"variable": "x", "coefficient": 1}],
                "operator": "==",
                "rhs": 1,
            }
        )
        assert constraint.weight is None
        assert constraint.description is None

    def test_objective_defaults(self):
        obj = Objective.model_validate(
            {
                "direction": "minimize",
                "linear_terms": [{"variable": "x", "coefficient": 1}],
            }
        )
        assert obj.quadratic_terms == []
        assert obj.constant == 0

    def test_solve_result_defaults(self):
        result = SolveResult.model_validate(
            {
                "status": "infeasible",
                "backend": "simulated_annealing",
                "objective_direction": "minimize",
                "solutions": [],
                "attempts": [],
            }
        )
        assert result.infeasibility_proven is False
        assert result.errors == []
        assert result.warnings == []
        assert result.message is None


class TestRemoteBackendOptions:
    """Phase 2 §12: remote backends and their per-backend option blocks."""

    @pytest.mark.parametrize(
        "backend",
        ["simulated_annealing", "exact", "dwave_qpu", "leap_hybrid_bqm"],
    )
    def test_backend_accepted(self, backend):
        prefs = SolverPreferences.model_validate({"backend": backend})
        assert prefs.backend == backend

    def test_dwave_qpu_options_parsed(self):
        prefs = SolverPreferences.model_validate(
            {
                "backend": "dwave_qpu",
                "dwave_qpu": {
                    "annealing_time_us": 20.0,
                    "chain_strength": 3.5,
                    "auto_scale": False,
                },
            }
        )
        assert prefs.dwave_qpu is not None
        assert prefs.dwave_qpu.annealing_time_us == 20.0
        assert prefs.dwave_qpu.chain_strength == 3.5
        assert prefs.dwave_qpu.auto_scale is False
        assert prefs.leap_hybrid_bqm is None

    def test_leap_hybrid_bqm_options_parsed(self):
        prefs = SolverPreferences.model_validate(
            {
                "backend": "leap_hybrid_bqm",
                "leap_hybrid_bqm": {"time_limit_seconds": 5.0},
            }
        )
        assert prefs.leap_hybrid_bqm is not None
        assert prefs.leap_hybrid_bqm.time_limit_seconds == 5.0
        assert prefs.dwave_qpu is None


class TestSolveError:
    """Phase 2 §13: structured error entries."""

    def test_defaults(self):
        error = SolveError(code="UNKNOWN_BACKEND", message="unknown backend")
        assert error.code == "UNKNOWN_BACKEND"
        assert error.message == "unknown backend"
        assert error.path is None
        assert error.retryable is False
        assert error.recommended_action is None

    def test_all_fields(self):
        error = SolveError(
            code="REMOTE_TIMEOUT",
            path="solver.backend",
            message="remote solve timed out",
            retryable=True,
            recommended_action="retry later",
        )
        assert error.path == "solver.backend"
        assert error.retryable is True
        assert error.recommended_action == "retry later"

    def test_problem_error_is_alias(self):
        assert ProblemError is SolveError


class TestSolveResultStatuses:
    """Phase 2 §13: the seven terminal statuses."""

    @pytest.mark.parametrize(
        "status",
        [
            "success",
            "infeasible",
            "invalid_problem",
            "solver_error",
            "backend_unavailable",
            "resource_limit_exceeded",
            "configuration_error",
        ],
    )
    def test_status_accepted(self, status):
        result = SolveResult.model_validate(
            {
                "status": status,
                "backend": None,
                "objective_direction": None,
                "solutions": [],
                "attempts": [],
            }
        )
        assert result.status == status

    def test_warnings_default_empty(self):
        result = SolveResult.model_validate(
            {
                "status": "success",
                "backend": "exact",
                "objective_direction": "maximize",
                "solutions": [],
                "attempts": [],
            }
        )
        assert result.warnings == []
