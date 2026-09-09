"""Every IR numeric field rejects booleans and strings (2026-09-09 review F-11).

pydantic's lax mode would happily read ``true`` as ``1`` and ``"10"`` as
``10``. For a *quantity* — a coefficient, a right-hand side, a weight, a read
count, a bound — that is a silently different problem, so the schema layer
refuses both (``models/quantities.py``). Everything else keeps lax semantics:
a JSON integer still fills a float field, and an integral float (``10.0``)
still fills an integer field, while ``10.5`` stays a type error.

The table below is the whole IR surface: if a numeric field is added to a
model it belongs here too.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from annealbridge.models import (
    Constraint,
    DWaveQPUOptions,
    FujitsuDAOptions,
    LeapHybridBQMOptions,
    LeapHybridCQMOptions,
    LinearTerm,
    Objective,
    OptimizationProblem,
    QuadraticTerm,
    SolverPreferences,
    Variable,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

# (model, legal base payload, field under test, "int" | "float")
NUMERIC_FIELDS = [
    (LinearTerm, {"variable": "x"}, "coefficient", "float"),
    (QuadraticTerm, {"variable1": "x", "variable2": "y"}, "coefficient", "float"),
    (Objective, {"direction": "minimize", "linear_terms": []}, "constant", "float"),
    (
        Constraint,
        {"id": "c1", "type": "soft", "terms": [], "operator": "<=", "rhs": 1},
        "rhs",
        "float",
    ),
    (
        Constraint,
        {"id": "c1", "type": "soft", "terms": [], "operator": "<=", "rhs": 1},
        "weight",
        "float",
    ),
    (Variable, {"name": "x", "type": "integer"}, "lower_bound", "int"),
    (Variable, {"name": "x", "type": "integer"}, "upper_bound", "int"),
    (SolverPreferences, {}, "num_reads", "int"),
    (SolverPreferences, {}, "num_sweeps", "int"),
    (SolverPreferences, {}, "seed", "int"),
    (SolverPreferences, {}, "top_k", "int"),
    (SolverPreferences, {}, "max_retries", "int"),
    (SolverPreferences, {}, "penalty_multiplier", "float"),
    (DWaveQPUOptions, {}, "annealing_time_us", "float"),
    (DWaveQPUOptions, {}, "chain_strength", "float"),
    (LeapHybridBQMOptions, {}, "time_limit_seconds", "float"),
    (LeapHybridCQMOptions, {}, "time_limit_seconds", "float"),
    # The Fujitsu block carries vendor ranges (ge/le); 10 is legal in all four.
    (FujitsuDAOptions, {}, "time_limit_seconds", "int"),
    (FujitsuDAOptions, {}, "num_run", "int"),
    (FujitsuDAOptions, {}, "num_group", "int"),
    (FujitsuDAOptions, {}, "num_output_solution", "int"),
]

CASES = [
    pytest.param(
        model, base, field, kind, id=f"{model.__name__}.{field}"
    )
    for model, base, field, kind in NUMERIC_FIELDS
]


def _validate(model, base: dict, field: str, value):
    return model.model_validate({**base, field: value})


@pytest.mark.parametrize("model, base, field, kind", CASES)
@pytest.mark.parametrize("flag", [True, False], ids=["true", "false"])
def test_boolean_is_rejected(model, base, field, kind, flag):
    with pytest.raises(ValidationError) as excinfo:
        _validate(model, base, field, flag)
    message = str(excinfo.value)
    assert field in message
    assert "boolean" in message


@pytest.mark.parametrize("model, base, field, kind", CASES)
def test_numeric_string_is_rejected(model, base, field, kind):
    with pytest.raises(ValidationError) as excinfo:
        _validate(model, base, field, "10")
    message = str(excinfo.value)
    assert field in message
    assert "string" in message


@pytest.mark.parametrize("model, base, field, kind", CASES)
def test_lax_numeric_coercion_survives(model, base, field, kind):
    """Only bool and str are refused; the rest of lax mode is untouched."""
    if kind == "int":
        for accepted in (10, 10.0):
            value = getattr(_validate(model, base, field, accepted), field)
            assert value == 10
            assert type(value) is int
        with pytest.raises(ValidationError):
            _validate(model, base, field, 10.5)
    else:
        value = getattr(_validate(model, base, field, 2), field)
        assert value == 2.0
        assert type(value) is float


def test_every_shipped_example_still_parses():
    """The examples are the contract agents copy from: none may regress."""
    examples = sorted(EXAMPLES_DIR.glob("*.json"))
    assert len(examples) >= 4
    for path in examples:
        problem = OptimizationProblem.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        assert problem.name


def test_before_validators_do_not_change_the_published_schema():
    """BeforeValidator has no schema effect: agents still see number/integer."""
    defs = OptimizationProblem.model_json_schema()["$defs"]

    coefficient = defs["LinearTerm"]["properties"]["coefficient"]
    assert coefficient["type"] == "number"
    assert set(coefficient) <= {"type", "title", "description"}

    num_reads = defs["SolverPreferences"]["properties"]["num_reads"]
    assert num_reads["type"] == "integer"
    assert num_reads["default"] == 100


def _problem_payload() -> dict:
    return {
        "version": "1.0",
        "name": "numeric-fields",
        "variables": [{"name": "x", "type": "binary"}],
        "objective": {
            "direction": "maximize",
            "linear_terms": [{"variable": "x", "coefficient": 3}],
        },
        "constraints": [
            {
                "id": "c1",
                "type": "hard",
                "terms": [{"variable": "x", "coefficient": 1}],
                "operator": "<=",
                "rhs": 1,
            }
        ],
    }


def test_whole_problem_rejects_boolean_coefficient():
    payload = _problem_payload()
    payload["objective"]["linear_terms"][0]["coefficient"] = True

    with pytest.raises(ValidationError) as excinfo:
        OptimizationProblem.model_validate(payload)
    assert "coefficient" in str(excinfo.value)
    assert "boolean" in str(excinfo.value)

    # Same payload over the JSON path an MCP client would take.
    with pytest.raises(ValidationError) as excinfo:
        OptimizationProblem.model_validate_json(json.dumps(payload))
    assert "boolean" in str(excinfo.value)


def test_whole_problem_rejects_string_rhs():
    payload = _problem_payload()
    payload["constraints"][0]["rhs"] = "10"

    with pytest.raises(ValidationError) as excinfo:
        OptimizationProblem.model_validate(payload)
    assert "rhs" in str(excinfo.value)
    assert "string" in str(excinfo.value)

    with pytest.raises(ValidationError) as excinfo:
        OptimizationProblem.model_validate_json(json.dumps(payload))
    assert "string" in str(excinfo.value)
