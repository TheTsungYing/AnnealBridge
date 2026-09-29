"""Schema 1.3 templates through a real MCP client (spec §14.16 item 10).

An agent writes index sets, parameters, variable families and templates
because the server instructions and tool descriptions tell it to; this file
holds that the tools accept such a document end to end: the template TSP
solves to its optimum with the generated variable names as solution keys, an
expansion error comes back as a structured ``PARAMETER_VALUE_MISSING`` whose
path points into the template, a malformed template document comes back as
the schema layer's ``INVALID_FIELD_VALUE`` with the right path, and the
capabilities announce version ``1.3`` and ``max_template_bindings``.
"""

import json

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp
from annealbridge.models.error_catalog import RECOMMENDED_ACTIONS
from tests.conftest import EXAMPLES_DIR

pytestmark = pytest.mark.anyio


def _tsp_template() -> dict:
    return json.loads((EXAMPLES_DIR / "tsp_template.json").read_text(encoding="utf-8"))


async def _call(tool: str, arguments: dict) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments)
    # A structured result, never an SDK tool error.
    assert result.is_error is False, result.content
    return result.structured_content


async def test_the_template_tsp_solves_to_its_optimum_with_generated_names():
    content = await _call("solve_optimization", {"problem": _tsp_template()})

    assert content["status"] == "success"
    assert content["backend"] == "exact"
    assert content["optimality_proven"] is True
    assert content["errors"] == []
    assert content["warnings"] == []
    best = content["solutions"][0]
    assert best["objective_value"] == 8.0
    assert best["hard_constraints_satisfied"] is True
    # The solution is keyed by the generated names, city-major.
    assert list(best["variables"]) == [
        f"x[{city},{position}]" for city in "abcd" for position in range(4)
    ]
    assert sum(best["variables"].values()) == 4
    # The generated constraint ids, in template then binding order.
    assert [e["constraint_id"] for e in best["constraint_evaluations"]] == [
        *(f"city_once[{city}]" for city in "abcd"),
        *(f"position_once[{position}]" for position in range(4)),
    ]
    assert all(e["satisfied"] for e in best["constraint_evaluations"])


async def test_a_missing_parameter_value_is_a_structured_expansion_error():
    document = _tsp_template()
    rows = document["parameters"][0]["values"]
    document["parameters"][0]["values"] = [row for row in rows if row["key"] != ["a", "b"]]

    content = await _call("validate_optimization_problem", {"problem": document})

    assert content["valid"] is False
    assert content["warnings"] == []
    assert [(e["code"], e["path"]) for e in content["errors"]] == [
        ("PARAMETER_VALUE_MISSING", "objective.quadratic_term_templates[0].coefficient")
    ]
    error = content["errors"][0]
    assert error["message"] == "Parameter dist has no value for (a,b) and no default"
    assert error["retryable"] is False
    assert error["recommended_action"] == RECOMMENDED_ACTIONS["PARAMETER_VALUE_MISSING"]


async def test_a_boolean_index_set_element_is_a_schema_error():
    document = _tsp_template()
    document["index_sets"][0]["elements"][1] = True

    content = await _call("validate_optimization_problem", {"problem": document})

    assert content["valid"] is False
    assert [(e["code"], e["path"]) for e in content["errors"]] == [
        ("INVALID_FIELD_VALUE", "index_sets[0].elements[1]")
    ]
    error = content["errors"][0]
    assert error["message"] == "index set elements must be strings or integers"
    assert error["recommended_action"] == RECOMMENDED_ACTIONS["INVALID_FIELD_VALUE"]


async def test_capabilities_announce_version_1_3_and_the_template_ceiling():
    content = await _call("get_optimization_capabilities", {})

    assert "1.3" in content["schema_versions"]
    assert content["backends"]
    for backend in content["backends"]:
        # A service-level ceiling every backend reports, always last.
        assert list(backend["limits"])[-1] == "max_template_bindings", backend["name"]
