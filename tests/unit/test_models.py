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
    FujitsuDAOptions,
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

    def test_version_1_1_accepted(self):
        """3b §7: 1.1 is the integer-variable schema version."""
        data = make_problem_dict()
        data["version"] = "1.1"
        assert OptimizationProblem.model_validate(data).version == "1.1"

    @pytest.mark.parametrize("bad", ["2.0", "1.2", "1", ""])
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

    @pytest.mark.parametrize("bad_vartype", ["int", "real", "spin"])
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


class TestVariableBounds:
    """3b §7: bounded integer variables at the *schema* level.

    Only what the model itself enforces is tested here. The semantic rules
    (bounds required on integer, absent on binary, upper > lower, within
    ±(2^31-1), version 1.1) belong to the Problem Validator so they surface
    as ``invalid_problem`` errors, and are tested with it.
    """

    def test_integer_variable_with_bounds_parses(self):
        var = Variable.model_validate(
            {"name": "x", "type": "integer", "lower_bound": -2, "upper_bound": 5}
        )
        assert var.type == "integer"
        assert var.lower_bound == -2
        assert var.upper_bound == 5
        assert var.bounds() == (-2, 5)

    def test_binary_bounds_are_zero_one(self):
        assert Variable(name="x").bounds() == (0, 1)

    def test_integer_without_bounds_bounds_raises(self):
        with pytest.raises(ValueError):
            Variable(name="x", type="integer").bounds()
        with pytest.raises(ValueError):
            Variable(name="x", type="integer", lower_bound=0).bounds()

    @pytest.mark.parametrize("bad", [1.5, "abc", [1], {"a": 1}])
    def test_non_integer_bound_rejected(self, bad):
        with pytest.raises(ValidationError):
            Variable.model_validate(
                {"name": "x", "type": "integer", "lower_bound": bad, "upper_bound": 5}
            )

    @pytest.mark.parametrize("field", ["lower_bound", "upper_bound"])
    @pytest.mark.parametrize("bad", [True, False])
    def test_boolean_bound_rejected(self, field, bad):
        """A bound is a quantity, not a flag: lax mode must not coerce it."""
        with pytest.raises(ValidationError) as excinfo:
            Variable.model_validate({"name": "x", "type": "integer", field: bad})
        assert "boolean" in str(excinfo.value)

    def test_integer_string_bound_is_rejected_but_integral_float_is_accepted(self):
        """A bound is a quantity, not text (2026-09-09 review F-11).

        ``"3"`` is refused along with ``True`` — every IR numeric field now
        uses the shared ``Count`` / ``Quantity`` types — while the rest of
        pydantic's lax mode survives, so an integral float still parses.
        """
        with pytest.raises(ValidationError) as excinfo:
            Variable.model_validate(
                {"name": "x", "type": "integer", "upper_bound": "3"}
            )
        assert "string" in str(excinfo.value)
        assert (
            Variable.model_validate(
                {"name": "x", "type": "integer", "upper_bound": 2.0}
            ).upper_bound
            == 2
        )

    def test_bounds_are_schema_legal_on_binary(self):
        """Bounds on a binary variable are a *validator* error, not a type error."""
        var = Variable.model_validate({"name": "x", "lower_bound": 0, "upper_bound": 1})
        assert var.type == "binary"
        assert var.lower_bound == 0

    def test_problem_json_schema_lists_bounds(self):
        """Agents discover the new fields through the published schema."""
        variable_schema = OptimizationProblem.model_json_schema()["$defs"]["Variable"]
        assert "lower_bound" in variable_schema["properties"]
        assert "upper_bound" in variable_schema["properties"]
        assert "integer" in variable_schema["properties"]["type"]["enum"]


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
        assert var.lower_bound is None
        assert var.upper_bound is None
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
        assert result.infeasibility is None
        assert result.errors == []
        assert result.warnings == []
        assert result.message is None

    def test_solve_result_round_trips_infeasibility_diagnostics(self):
        payload = {
            "status": "infeasible",
            "backend": "exact",
            "objective_direction": "maximize",
            "solutions": [],
            "attempts": [],
            "infeasibility_proven": True,
            "infeasibility": {
                "closest_candidate": {
                    "variables": {"x1": 0, "x2": 0, "x3": 1},
                    "hard_violation_total": 2.0,
                    "constraint_evaluations": [
                        {
                            "constraint_id": "all_three",
                            "constraint_type": "hard",
                            "satisfied": False,
                            "actual_value": 1.0,
                            "operator": ">=",
                            "expected_value": 3.0,
                            "violation_amount": 2.0,
                            "weighted_penalty": None,
                        },
                        {
                            "constraint_id": "budget",
                            "constraint_type": "hard",
                            "satisfied": True,
                            "actual_value": 1.0,
                            "operator": "<=",
                            "expected_value": 1.0,
                            "violation_amount": 0.0,
                            "weighted_penalty": None,
                        },
                    ],
                },
                "hard_violation_rates": [
                    {
                        "constraint_id": "all_three",
                        "violated_candidates": 7,
                        "candidates": 8,
                        "violated_fraction": 0.875,
                    },
                    {
                        "constraint_id": "budget",
                        "violated_candidates": 6,
                        "candidates": 8,
                        "violated_fraction": 0.75,
                    },
                ],
            },
        }

        result = SolveResult.model_validate(payload)

        assert result.infeasibility is not None
        closest = result.infeasibility.closest_candidate
        assert closest.variables == {"x1": 0, "x2": 0, "x3": 1}
        assert closest.hard_violation_total == 2.0
        assert [e.constraint_id for e in closest.constraint_evaluations] == [
            "all_three",
            "budget",
        ]
        rates = result.infeasibility.hard_violation_rates
        assert [r.constraint_id for r in rates] == ["all_three", "budget"]
        assert [r.violated_fraction for r in rates] == [0.875, 0.75]
        # The dump is the same payload again, so an MCP round trip is lossless.
        dumped = result.model_dump(mode="json")
        assert dumped["infeasibility"] == payload["infeasibility"]


class TestRemoteBackendOptions:
    """Phase 2 §12: remote backends and their per-backend option blocks."""

    @pytest.mark.parametrize(
        "backend",
        [
            "simulated_annealing",
            "exact",
            "dwave_qpu",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
            "fujitsu_da",
        ],
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


class TestFujitsuDAOptions:
    """3b §20.2: the Digital Annealer option block and its vendor ranges."""

    def test_all_fields_parsed(self):
        prefs = SolverPreferences.model_validate(
            {
                "backend": "fujitsu_da",
                "fujitsu_da": {
                    "time_limit_seconds": 30,
                    "num_run": 32,
                    "num_group": 2,
                    "num_output_solution": 10,
                },
            }
        )
        assert prefs.fujitsu_da is not None
        assert prefs.fujitsu_da.time_limit_seconds == 30
        assert prefs.fujitsu_da.num_run == 32
        assert prefs.fujitsu_da.num_group == 2
        assert prefs.fujitsu_da.num_output_solution == 10
        assert prefs.dwave_qpu is None
        assert prefs.leap_hybrid_cqm is None

    def test_defaults_are_all_none(self):
        """``None`` means "do not send the field"; the vendor default applies."""
        options = FujitsuDAOptions()
        assert options.time_limit_seconds is None
        assert options.num_run is None
        assert options.num_group is None
        assert options.num_output_solution is None

    def test_num_run_above_vendor_maximum_rejected(self):
        with pytest.raises(ValidationError):
            FujitsuDAOptions(num_run=2000)

    def test_time_limit_seconds_zero_rejected(self):
        with pytest.raises(ValidationError):
            FujitsuDAOptions(time_limit_seconds=0)

    def test_num_group_above_vendor_maximum_rejected(self):
        with pytest.raises(ValidationError):
            FujitsuDAOptions(num_group=17)


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


# --- Unknown fields (models/strict.py) --------------------------------------
#
# The caller is usually an LLM, and inventing a plausible field name is its
# characteristic mistake. pydantic's default would drop the key silently and
# solve a *different* problem that passes every guarantee; every input model
# therefore refuses keys it does not declare, on the type layer, like a
# boolean in a numeric field (2026-09-09 review F-11).


def _with_key(payload: dict, path: str, key: str) -> dict:
    """Copy ``payload`` with ``key: 1`` inserted at the node named by ``path``."""
    node = payload
    for part in path.split(".") if path else []:
        if part.endswith("]"):
            name, _, index = part[:-1].partition("[")
            node = node[name][int(index)]
        else:
            node = node[part]
    node[key] = 1
    return payload


class TestUnknownFields:
    @pytest.mark.parametrize(
        "path, key, expected_loc",
        [
            ("", "minimize_secondary", ("minimize_secondary",)),
            ("variables[0]", "weight", ("variables", 0, "weight")),
            ("objective", "cubic_terms", ("objective", "cubic_terms")),
            (
                "objective.quadratic_terms[0]",
                "variable3",
                ("objective", "quadratic_terms", 0, "variable3"),
            ),
            (
                "objective.linear_terms[0]",
                "coeff",
                ("objective", "linear_terms", 0, "coeff"),
            ),
            ("constraints[0]", "penalty", ("constraints", 0, "penalty")),
            ("solver", "num_restarts", ("solver", "num_restarts")),
            (
                "solver.dwave_qpu",
                "num_spin_reversal_transforms",
                ("solver", "dwave_qpu", "num_spin_reversal_transforms"),
            ),
        ],
        ids=[
            "top-level",
            "variable",
            "objective",
            "quadratic-term",
            "linear-term",
            "constraint",
            "solver",
            "option-block",
        ],
    )
    def test_unknown_key_is_rejected_with_its_path(self, path, key, expected_loc):
        payload = make_problem_dict()
        payload["solver"] = {"backend": "dwave_qpu", "dwave_qpu": {}}
        _with_key(payload, path, key)

        with pytest.raises(ValidationError) as excinfo:
            OptimizationProblem.model_validate(payload)

        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"
        assert errors[0]["loc"] == expected_loc

    @pytest.mark.parametrize(
        "model",
        [
            Variable,
            Objective,
            Constraint,
            SolverPreferences,
            DWaveQPUOptions,
            LeapHybridBQMOptions,
            FujitsuDAOptions,
            OptimizationProblem,
        ],
    )
    def test_every_input_model_forbids_extra(self, model):
        assert model.model_config.get("extra") == "forbid"

    def test_published_schema_declares_no_additional_properties(self):
        """A schema-aware host can refuse the document before sending it."""
        schema = OptimizationProblem.model_json_schema()
        assert schema["additionalProperties"] is False
        for name, definition in schema["$defs"].items():
            assert definition.get("additionalProperties") is False, name

    def test_output_models_keep_the_default(self):
        """Results are built by our own code; only caller input is strict."""
        assert SolveResult.model_config.get("extra") is None


class TestFieldDescriptions:
    """Every public input field must document itself in the JSON Schema.

    The schema is what an LLM agent sees as the MCP tool's inputSchema, so a
    property without a description is a field the agent has to guess at.
    """

    def test_every_schema_property_has_a_description(self):
        schema = OptimizationProblem.model_json_schema()
        missing: list[str] = []

        def check(model_name: str, definition: dict) -> None:
            for field, prop in definition.get("properties", {}).items():
                description = prop.get("description")
                if not isinstance(description, str) or not description.strip():
                    missing.append(f"{model_name}.{field}")

        check("OptimizationProblem", schema)
        for name, definition in schema.get("$defs", {}).items():
            check(name, definition)

        assert not missing, "properties without a description: " + ", ".join(
            sorted(missing)
        )
