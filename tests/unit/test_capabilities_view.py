"""The shared capabilities view (interfaces/capabilities.py).

The problem JSON schema is generated once per process; backend availability
is never cached because credentials can change between two calls.
"""

from annealbridge.interfaces.capabilities import (
    _problem_json_schema,
    build_capabilities,
)
from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy
from annealbridge.solvers import SolverRegistry
import annealbridge.solvers.dwave_qpu as qpu_module


class TestSchemaCache:
    def test_schema_matches_the_model_and_is_served_from_the_cache(self):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy()

        before = _problem_json_schema.cache_info().hits
        first = build_capabilities(registry, policy).problem_json_schema
        second = build_capabilities(registry, policy).problem_json_schema

        assert first == OptimizationProblem.model_json_schema()
        assert second == first
        assert _problem_json_schema.cache_info().hits >= before + 1


class TestAvailabilityIsLive:
    def test_availability_reflects_each_call(self, monkeypatch):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy(allow_remote=True)

        monkeypatch.setattr(qpu_module, "dwave_availability", lambda: (False, "down"))
        first = {b.name: b for b in build_capabilities(registry, policy).backends}
        monkeypatch.setattr(qpu_module, "dwave_availability", lambda: (True, None))
        second = {b.name: b for b in build_capabilities(registry, policy).backends}

        assert first["dwave_qpu"].available is False
        assert first["dwave_qpu"].unavailable_reason == "down"
        assert second["dwave_qpu"].available is True
        assert second["dwave_qpu"].unavailable_reason is None
