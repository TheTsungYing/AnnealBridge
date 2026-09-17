"""The shared capabilities view (interfaces/capabilities.py).

The problem JSON schema is generated once per process; backend availability
is never cached because credentials can change between two calls.

2026-09-09 review F-22: the cache holds the *serialized* schema and every
response gets its own ``json.loads`` copy, so one caller mutating the dict
it received can never corrupt the next response; and ``BackendCapability.name``
is the registry key — the value that goes into ``solver.backend`` and the one
``enabled_backends`` is matched against.

2026-09-15: availability goes through the same guard as ``gate_errors``, so
one backend whose check raises or reports an unknown category is listed as
unavailable with a (redacted) reason instead of failing the whole view.
"""

import logging
from importlib.metadata import version

import pytest

import annealbridge.solvers.dwave_qpu as qpu_module
from annealbridge.interfaces.capabilities import (
    BackendCapability,
    _problem_json_schema_json,
    build_capabilities,
)
from annealbridge.models import AvailabilityStatus, OptimizationProblem
from annealbridge.orchestration import ExecutionPolicy
from annealbridge.solvers import ExactSolverBackend, SolverRegistry
from tests.unit.test_limits import FAKE_TOKEN, SpyBackend, make_capabilities

BROKEN_KEY = "broken_backend"
HEALTHY_KEY = "my_exact"


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

    def test_include_schema_false_skips_the_schema_entirely(self):
        registry = SolverRegistry.default()
        policy = ExecutionPolicy()

        before = _problem_json_schema_json.cache_info()
        view = build_capabilities(registry, policy, include_schema=False)

        assert view.problem_json_schema is None
        # Not merely dropped from the response: the cached schema is never
        # even asked for, so the caller pays nothing for it. The whole
        # cache_info is compared so a cold cache (a miss, not a hit) would
        # be caught too when this test runs on its own.
        assert _problem_json_schema_json.cache_info() == before


class TestPackageVersion:
    def test_the_view_carries_the_installed_distribution_version(self):
        view = build_capabilities(SolverRegistry.default(), ExecutionPolicy())

        assert view.annealbridge_version == version("annealbridge")


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


def _entries(registry: SolverRegistry) -> dict:
    """The view's backend entries keyed by registry key, in listed order."""
    return {
        entry.name: entry
        for entry in build_capabilities(registry, ExecutionPolicy()).backends
    }


class TestOneBrokenAvailabilityCheck:
    """2026-09-15: a backend's availability check is third-party code, so
    the view guards it exactly like ``gate_errors`` does; one broken check
    only marks its own backend unavailable."""

    def test_a_raising_check_marks_only_its_own_backend_unavailable(self):
        broken = SpyBackend(
            make_capabilities(name="fake_broken", remote=False),
            raise_on_available=RuntimeError(f"vendor blew up token={FAKE_TOKEN}"),
        )
        registry = SolverRegistry({BROKEN_KEY: broken, HEALTHY_KEY: ExactSolverBackend()})

        capabilities = build_capabilities(registry, ExecutionPolicy())
        entries = {entry.name: entry for entry in capabilities.backends}

        assert list(entries) == [BROKEN_KEY, HEALTHY_KEY]
        assert entries[BROKEN_KEY].available is False
        # ``enabled`` is policy only; the broken check does not change it.
        assert entries[BROKEN_KEY].enabled is True
        reason = entries[BROKEN_KEY].unavailable_reason
        assert "availability check failed" in reason
        assert "RuntimeError" in reason
        assert FAKE_TOKEN not in capabilities.model_dump_json()
        assert broken.availability_calls == 1
        # The healthy backend is reported exactly as it is on its own.
        alone = _entries(SolverRegistry({HEALTHY_KEY: ExactSolverBackend()}))
        assert entries[HEALTHY_KEY] == alone[HEALTHY_KEY]
        assert entries[HEALTHY_KEY].available is True

    @pytest.mark.parametrize(
        "detail", [None, "scheduled maintenance"], ids=["no-detail", "detail"]
    )
    def test_an_unknown_category_is_listed_with_the_reason(self, detail):
        weird = SpyBackend(
            make_capabilities(name="fake_weird", remote=False),
            AvailabilityStatus.model_construct(
                category="weird", detail=detail, error_code=None
            ),
        )
        registry = SolverRegistry({BROKEN_KEY: weird, HEALTHY_KEY: ExactSolverBackend()})

        entries = _entries(registry)

        assert list(entries) == [BROKEN_KEY, HEALTHY_KEY]
        assert entries[BROKEN_KEY].available is False
        reason = entries[BROKEN_KEY].unavailable_reason
        assert "unknown availability category 'weird'" in reason
        assert reason.startswith(detail if detail is not None else "no reason reported")
        assert entries[HEALTHY_KEY].available is True

    def test_an_unknown_category_detail_is_redacted_in_the_view(self):
        weird = SpyBackend(
            make_capabilities(name="fake_weird", remote=False),
            AvailabilityStatus.model_construct(
                category="weird", detail=f"vendor said token={FAKE_TOKEN}", error_code=None
            ),
        )
        registry = SolverRegistry({BROKEN_KEY: weird, HEALTHY_KEY: ExactSolverBackend()})

        view = build_capabilities(registry, ExecutionPolicy())

        assert FAKE_TOKEN not in view.model_dump_json()
        broken = next(entry for entry in view.backends if entry.name == BROKEN_KEY)
        assert broken.unavailable_reason.startswith("vendor said token=***")

    def test_the_logged_warning_is_redacted_too(self, caplog):
        broken = SpyBackend(
            make_capabilities(name="fake_broken", remote=False),
            raise_on_available=RuntimeError(f"vendor blew up token={FAKE_TOKEN}"),
        )

        with caplog.at_level(logging.WARNING):
            build_capabilities(SolverRegistry({BROKEN_KEY: broken}), ExecutionPolicy())

        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert any("availability check failed" in record.getMessage() for record in warnings)
        assert FAKE_TOKEN not in caplog.text


class TestSeedRange:
    """The declared seed range is part of the view, so an agent can choose a
    seed the backend accepts before submitting anything."""

    def test_each_seeded_backend_declares_its_own_seed_range(self):
        entries = _entries(SolverRegistry.default())

        assert len(entries) == 8
        # The seeded backends wrap different samplers and their accepted
        # ranges differ, so the view carries one range per backend.
        declared = {
            "simulated_annealing": (0, 2**31 - 1),
            "tabu": (0, 2**32 - 1),
            "simulated_bifurcation": (0, 2**32 - 1),
        }
        for name, entry in entries.items():
            assert (entry.seed_min, entry.seed_max) == declared.get(name, (None, None)), name

    def test_seed_range_fields_are_described(self):
        for field in ("seed_min", "seed_max"):
            assert field in BackendCapability.model_fields
            assert BackendCapability.model_fields[field].description
