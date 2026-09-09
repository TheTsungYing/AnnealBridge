"""The shared capabilities view (interfaces/capabilities.py).

The problem JSON schema is generated once per process; backend availability
is never cached because credentials can change between two calls.

2026-09-09 review F-22: the cache holds the *serialized* schema and every
response gets its own ``json.loads`` copy, so one caller mutating the dict
it received can never corrupt the next response; and ``BackendCapability.name``
is the registry key — the value that goes into ``solver.backend`` and the one
``enabled_backends`` is matched against.
"""

from annealbridge.interfaces.capabilities import (
    _problem_json_schema_json,
    build_capabilities,
)
from annealbridge.models import AvailabilityStatus, OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy
from annealbridge.solvers import ExactSolverBackend, SolverRegistry
import annealbridge.solvers.dwave_qpu as qpu_module


class TestSchemaCache:
    def test_schema_matches_the_model_and_is_served_from_the_cache(self):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy()

        before = _problem_json_schema_json.cache_info().hits
        first = build_capabilities(registry, policy).problem_json_schema
        second = build_capabilities(registry, policy).problem_json_schema

        assert first == OptimizationProblem.model_json_schema()
        assert second == first
        assert _problem_json_schema_json.cache_info().hits >= before + 1

    def test_mutating_one_response_does_not_leak_into_the_next(self):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy()

        first = build_capabilities(registry, policy).problem_json_schema
        first["properties"]["name"]["POLLUTED"] = 1
        del first["properties"]["version"]

        second = build_capabilities(registry, policy).problem_json_schema

        assert second == OptimizationProblem.model_json_schema()
        assert "POLLUTED" not in second["properties"]["name"]
        assert "version" in second["properties"]


class TestNameIsTheRegistryKey:
    """F-22b: ``name`` is what the caller must send back, not the backend's
    own ``capabilities.name`` (a custom registry may differ)."""

    def test_a_custom_key_is_reported_instead_of_the_capabilities_name(self):
        registry = SolverRegistry({"my_exact": ExactSolverBackend()})

        (entry,) = build_capabilities(registry, ExecutionPolicy()).backends

        assert entry.name == "my_exact"

    def test_enabled_is_decided_against_the_reported_name(self):
        registry = SolverRegistry({"my_exact": ExactSolverBackend()})

        allowed = build_capabilities(
            registry, ExecutionPolicy(enabled_backends={"my_exact"})
        ).backends
        refused = build_capabilities(
            registry, ExecutionPolicy(enabled_backends={"exact"})
        ).backends

        assert [entry.enabled for entry in allowed] == [True]
        assert [entry.enabled for entry in refused] == [False]

    def test_the_six_built_in_backends_report_their_registration_order(self):
        registry = SolverRegistry.default()

        entries = build_capabilities(registry, ExecutionPolicy()).backends

        assert [entry.name for entry in entries] == registry.names()
        for entry in entries:
            assert entry.name == registry.get(entry.name).capabilities.name


class TestAvailabilityIsLive:
    def test_availability_reflects_each_call(self, monkeypatch):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy(allow_remote=True)

        monkeypatch.setattr(
            qpu_module,
            "dwave_availability",
            lambda: AvailabilityStatus(category="unavailable", detail="down"),
        )
        first = {b.name: b for b in build_capabilities(registry, policy).backends}
        monkeypatch.setattr(
            qpu_module,
            "dwave_availability",
            lambda: AvailabilityStatus(category="available"),
        )
        second = {b.name: b for b in build_capabilities(registry, policy).backends}

        assert first["dwave_qpu"].available is False
        assert first["dwave_qpu"].unavailable_reason == "down"
        assert second["dwave_qpu"].available is True
        assert second["dwave_qpu"].unavailable_reason is None
