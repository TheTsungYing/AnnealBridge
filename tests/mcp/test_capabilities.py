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
    FujitsuDABackend,
    LeapHybridBQMBackend,
    LeapHybridCQMBackend,
    SimulatedAnnealingBackend,
    SimulatedBifurcationBackend,
    TabuBackend,
)
from annealbridge.solvers.fujitsu_da import (
    REASON_API_KEY_MISSING,
    REASON_URL_NOT_HTTPS,
)

pytestmark = pytest.mark.anyio

BACKEND_CLASSES = (
    ExactSolverBackend,
    SimulatedAnnealingBackend,
    TabuBackend,
    SimulatedBifurcationBackend,
    DWaveQPUBackend,
    LeapHybridBQMBackend,
    LeapHybridCQMBackend,
    FujitsuDABackend,
)

REMOTE_BACKEND_NAMES = (
    "dwave_qpu",
    "leap_hybrid_bqm",
    "leap_hybrid_cqm",
    "fujitsu_da",
)

# The categorical strings is_available() may return for the remote backends;
# they must never contain configuration values (spec §10, 3b §20.5).
UNAVAILABLE_REASONS = {
    REASON_NOT_INSTALLED,
    REASON_CREDENTIALS_MISSING,
    REASON_CONFIG_INVALID,
    REASON_API_KEY_MISSING,
    REASON_URL_NOT_HTTPS,
}


async def _get_capabilities():
    async with Client(mcp) as client:
        result = await client.call_tool("get_optimization_capabilities", {})
    assert result.is_error is False
    return result


async def test_structured_content_is_dict():
    result = await _get_capabilities()
    assert isinstance(result.structured_content, dict)


async def test_all_backends_listed():
    content = (await _get_capabilities()).structured_content
    names = [backend["name"] for backend in content["backends"]]
    assert sorted(names) == [
        "dwave_qpu",
        "exact",
        "fujitsu_da",
        "leap_hybrid_bqm",
        "leap_hybrid_cqm",
        "simulated_annealing",
        "simulated_bifurcation",
        "tabu",
    ]


async def test_local_backends_available_and_enabled():
    content = (await _get_capabilities()).structured_content
    by_name = {backend["name"]: backend for backend in content["backends"]}
    for name in ("exact", "simulated_annealing", "tabu", "simulated_bifurcation"):
        assert by_name[name]["available"] is True
        assert by_name[name]["enabled"] is True
        assert by_name[name]["unavailable_reason"] is None


async def test_remote_backends_unavailable_with_categorical_reason():
    content = (await _get_capabilities()).structured_content
    by_name = {backend["name"]: backend for backend in content["backends"]}
    for name in REMOTE_BACKEND_NAMES:
        assert by_name[name]["available"] is False
        assert by_name[name]["unavailable_reason"] in UNAVAILABLE_REASONS
        # Default policy has allow_remote=False, so remote backends are disabled.
        assert by_name[name]["enabled"] is False


async def test_schema_metadata():
    content = (await _get_capabilities()).structured_content
    # 3b §10: schema_version is the newest version accepted, and every version
    # in schema_versions is still accepted (1.1 is a superset of 1.0).
    assert content["schema_version"] == "1.1"
    assert content["schema_versions"] == ["1.0", "1.1"]
    assert content["supported_variable_types"] == ["binary", "integer"]
    assert content["supported_constraint_operators"] == ["==", "<=", ">="]
    assert content["inequality_requires_integer_coefficients"] is True
    assert content["problem_json_schema"]["title"] == "OptimizationProblem"


async def test_limits_come_from_policy():
    content = (await _get_capabilities()).structured_content
    by_name = {backend["name"]: backend for backend in content["backends"]}
    # Every backend also carries the two service-level ceilings of spec
    # §11.4: the retry ceiling chosen by the ``remote`` flag, then top_k.
    assert by_name["exact"]["limits"] == {
        "max_variables": 24,
        "max_local_retries": 10,
        "max_top_k": 1000,
    }
    assert by_name["dwave_qpu"]["limits"] == {
        "max_reads": 1000,
        "max_annealing_time_us": 2000.0,
        "max_remote_retries": 3,
        "max_top_k": 1000,
    }
    assert by_name["leap_hybrid_bqm"]["limits"] == {
        "max_time_seconds": 300,
        "max_remote_retries": 3,
        "max_top_k": 1000,
    }
    assert by_name["leap_hybrid_cqm"]["limits"] == {
        "max_time_seconds": 300,
        "max_remote_retries": 3,
        "max_top_k": 1000,
    }
    assert by_name["fujitsu_da"]["limits"] == {
        "max_time_seconds": 300,
        "max_remote_retries": 3,
        "max_top_k": 1000,
    }
    assert by_name["simulated_annealing"]["limits"] == {
        "max_local_reads": 100000,
        "max_sweeps": 100000,
        "max_local_retries": 10,
        "max_top_k": 1000,
    }
    # ``tabu`` takes no sweeps, so it declares no sweep ceiling either.
    assert by_name["tabu"]["limits"] == {
        "max_local_reads": 100000,
        "max_local_retries": 10,
        "max_top_k": 1000,
    }
    # Simulated bifurcation's sweeps are its integration steps, so it is
    # capped under the same two local keys as the annealer; its own
    # dense-matrix cap (declared, not policy) shows as max_variables.
    assert by_name["simulated_bifurcation"]["limits"] == {
        "max_variables": 10000,
        "max_local_reads": 100000,
        "max_sweeps": 100000,
        "max_local_retries": 10,
        "max_top_k": 1000,
    }


async def test_seed_range_is_declared_per_backend():
    content = (await _get_capabilities()).structured_content
    by_name = {backend["name"]: backend for backend in content["backends"]}
    for backend in by_name.values():
        assert "seed_min" in backend and "seed_max" in backend, backend["name"]
    # Each seeded backend publishes its own sampler's rule; they do not all
    # agree (2**31 - 1 against 2**32 - 1) and none is a shared constant.
    declared = {
        "simulated_annealing": (0, 2147483647),
        "tabu": (0, 4294967295),
        "simulated_bifurcation": (0, 4294967295),
    }
    for name, backend in by_name.items():
        expected = declared.get(name, (None, None))
        assert (backend["seed_min"], backend["seed_max"]) == expected, name


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
