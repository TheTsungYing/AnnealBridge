"""Unit tests for the solver backend registry (Phase 2 spec §11).

The registry is the single lookup point from backend name to backend
instance. Unknown names raise ``KeyError`` at the registry level and are
mapped by the service to a structured ``UNKNOWN_BACKEND`` error — never to
a silent fallback onto another backend.
"""

import sys

import pytest

from annealbridge.models import RECOMMENDED_ACTIONS, OptimizationProblem
from annealbridge.orchestration import OptimizationService
from annealbridge.solvers import (
    ExactSolverBackend,
    SimulatedAnnealingBackend,
    SolverRegistry,
)


def make_problem(backend: str) -> OptimizationProblem:
    """A minimal feasible 0/1 problem: maximize 2a + b s.t. a + b <= 1."""
    return OptimizationProblem.model_validate(
        {
            "name": "registry test problem",
            "variables": [{"name": "a"}, {"name": "b"}],
            "objective": {
                "direction": "maximize",
                "linear_terms": [
                    {"variable": "a", "coefficient": 2},
                    {"variable": "b", "coefficient": 1},
                ],
            },
            "constraints": [
                {
                    "id": "at_most_one",
                    "type": "hard",
                    "terms": [
                        {"variable": "a", "coefficient": 1},
                        {"variable": "b", "coefficient": 1},
                    ],
                    "operator": "<=",
                    "rhs": 1,
                }
            ],
            "solver": {"backend": backend},
        }
    )


class TestSolverRegistryLookup:
    def test_get_returns_the_registered_backend(self):
        exact = ExactSolverBackend()
        registry = SolverRegistry({exact.name: exact})

        assert registry.get("exact") is exact

    def test_get_raises_key_error_for_unknown_name(self):
        registry = SolverRegistry({"exact": ExactSolverBackend()})

        with pytest.raises(KeyError):
            registry.get("nope")

    def test_names_lists_the_registered_names(self):
        registry = SolverRegistry({"exact": ExactSolverBackend()})

        assert registry.names() == ["exact"]


class TestDefaultRegistry:
    def test_default_registers_exactly_the_expected_backends_in_order(self):
        # 3a §17.7: the registration order is fixed; capabilities, the CLI
        # table and routing's final tie-break all follow it.
        registry = SolverRegistry.default()

        assert registry.names() == [
            "exact",
            "simulated_annealing",
            "dwave_qpu",
            "leap_hybrid_bqm",
            "leap_hybrid_cqm",
            "fujitsu_da",
        ]

    def test_default_backends_report_their_own_names(self):
        registry = SolverRegistry.default()

        for name in registry.names():
            assert registry.get(name).name == name

    def test_default_registry_needs_no_dwave_system(self):
        # 3a §17.7 / §31: the remote backends lazy-import ``dwave.system``
        # inside their sampler factories, so the default registry (and the
        # CQM backend in it) is importable without the ``dwave`` extra.
        SolverRegistry.default()

        assert "dwave.system" not in sys.modules

    def test_default_registry_needs_no_third_party_http_package(self):
        # 3b §4 / §26.2: the Fujitsu DA backend talks HTTPS through the
        # standard library only, so building the default registry must not
        # drag in a new HTTP dependency.
        SolverRegistry.default()

        for module in ("requests", "httpx", "aiohttp"):
            assert module not in sys.modules


def make_service_without_dwave_qpu() -> OptimizationService:
    """Service whose registry deliberately lacks ``dwave_qpu``.

    ``SolverPreferences.backend`` is a Literal of registered names, so the
    UNKNOWN_BACKEND path is exercised with a schema-valid name that is
    missing from the injected registry.
    """
    exact = ExactSolverBackend()
    annealer = SimulatedAnnealingBackend()
    return OptimizationService(
        registry=SolverRegistry({exact.name: exact, annealer.name: annealer})
    )


class TestServiceUnknownBackend:
    def test_status_and_backend_report_the_requested_name(self):
        result = make_service_without_dwave_qpu().solve(make_problem("dwave_qpu"))

        assert result.status == "backend_unavailable"
        assert result.backend == "dwave_qpu"

    def test_no_fallback_solutions_are_produced(self):
        result = make_service_without_dwave_qpu().solve(make_problem("dwave_qpu"))

        assert result.solutions == []

    def test_single_structured_unknown_backend_error(self):
        result = make_service_without_dwave_qpu().solve(make_problem("dwave_qpu"))

        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.code == "UNKNOWN_BACKEND"
        assert error.retryable is False
        assert error.recommended_action == RECOMMENDED_ACTIONS["UNKNOWN_BACKEND"]

    def test_message_names_the_request_and_the_available_backends(self):
        result = make_service_without_dwave_qpu().solve(make_problem("dwave_qpu"))

        message = result.errors[0].message
        assert "dwave_qpu" in message
        assert "exact" in message
        assert "simulated_annealing" in message


class TestCustomRegistryInjection:
    def test_injected_registry_solves_a_registered_backend(self):
        exact = ExactSolverBackend()
        service = OptimizationService(registry=SolverRegistry({exact.name: exact}))

        result = service.solve(make_problem("exact"))

        assert result.status == "success"
        assert result.backend == "exact"
        assert result.solutions
