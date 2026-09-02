"""get_optimization_capabilities over the in-memory MCP client (spec §22, §26)."""

import pytest
from mcp import Client

from annealbridge.interfaces.mcp import mcp
from annealbridge.solvers import (
    REASON_CONFIG_INVALID,
    REASON_CREDENTIALS_MISSING,
    REASON_NOT_INSTALLED,
    DWaveQPUBackend,
    ExactSolverBackend,
    LeapHybridBQMBackend,
    SimulatedAnnealingBackend,
)

pytestmark = pytest.mark.anyio

BACKEND_CLASSES = (
    ExactSolverBackend,
    SimulatedAnnealingBackend,
    DWaveQPUBackend,
    LeapHybridBQMBackend,
)

# The categorical strings is_available() may return for the D-Wave backends;
# they must never contain configuration values (spec §10).
DWAVE_UNAVAILABLE_REASONS = {
    REASON_NOT_INSTALLED,
    REASON_CREDENTIALS_MISSING,
    REASON_CONFIG_INVALID,
}


async def _get_capabilities():
    async with Client(mcp) as client:
        result = await client.call_tool("get_optimization_capabilities", {})
    assert result.is_error is False
    return result


async def test_structured_content_is_dict():
    result = await _get_capabilities()
    assert isinstance(result.structured_content, dict)


async def test_all_four_backends_listed():
    content = (await _get_capabilities()).structured_content
    names = [backend["name"] for backend in content["backends"]]
    assert sorted(names) == [
        "dwave_qpu",
        "exact",
        "leap_hybrid_bqm",
        "simulated_annealing",
    ]


async def test_local_backends_available_and_enabled():
    content = (await _get_capabilities()).structured_content
    by_name = {backend["name"]: backend for backend in content["backends"]}
    for name in ("exact", "simulated_annealing"):
        assert by_name[name]["available"] is True
        assert by_name[name]["enabled"] is True
        assert by_name[name]["unavailable_reason"] is None


async def test_dwave_backends_unavailable_with_categorical_reason():
    content = (await _get_capabilities()).structured_content
    by_name = {backend["name"]: backend for backend in content["backends"]}
    for name in ("dwave_qpu", "leap_hybrid_bqm"):
        assert by_name[name]["available"] is False
        assert by_name[name]["unavailable_reason"] in DWAVE_UNAVAILABLE_REASONS
        # Default policy has allow_remote=False, so remote backends are disabled.
        assert by_name[name]["enabled"] is False


async def test_schema_metadata():
    content = (await _get_capabilities()).structured_content
    assert content["schema_version"] == "1.0"
    assert content["supported_variable_types"] == ["binary"]
    assert content["supported_constraint_operators"] == ["==", "<=", ">="]
    assert content["inequality_requires_integer_coefficients"] is True
    assert content["problem_json_schema"]["title"] == "OptimizationProblem"


async def test_limits_come_from_policy():
    content = (await _get_capabilities()).structured_content
    by_name = {backend["name"]: backend for backend in content["backends"]}
    assert by_name["exact"]["limits"] == {"max_variables": 24}
    assert by_name["dwave_qpu"]["limits"] == {
        "max_reads": 1000,
        "max_annealing_time_us": 2000.0,
    }
    assert by_name["leap_hybrid_bqm"]["limits"] == {"max_time_seconds": 300}
    assert by_name["simulated_annealing"]["limits"] == {}


async def test_capabilities_never_calls_solve(monkeypatch):
    calls: list[str] = []

    def make_spy(cls):
        def spy(self, *args, **kwargs):
            calls.append(cls.__name__)
            raise AssertionError(f"{cls.__name__}.solve called during capabilities")

        return spy

    for cls in BACKEND_CLASSES:
        monkeypatch.setattr(cls, "solve", make_spy(cls))

    result = await _get_capabilities()
    assert result.is_error is False
    assert calls == []
