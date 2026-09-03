"""Architecture test: a fifth backend needs zero core changes (3a spec §13.1).

``FakeDeclaredBackend`` (``tests/fakes/declared_backend.py``) is registered
next to every default backend. The service, the validator and the
capabilities view must handle it purely from its declaration:

* the capabilities view and the service read the same policy limit for
  its custom ``iterations`` key (drift 1 of the 3a plan);
* its declared parameter limit is enforced by the generic check, never
  clamped, under its own (uncatalogued) error code;
* the validator's advisory warnings follow the capability flags;
* a policy that lacks the declared key is refused at construction.

The proof that none of this needed a code change is
``test_core_sources_never_mention_the_fake``: the files the fake flows
through do not contain its name.

``service.recommend()`` does not exist yet (3a step 9); the routing
assertion of §13.1 is added there — see the marker at the end of the file.
"""

import json
from pathlib import Path

import pytest

from annealbridge.config import SettingsError
from annealbridge.interfaces.capabilities import build_capabilities
from annealbridge.interfaces.composition import build_state_from_policy
from annealbridge.models import OptimizationProblem, SolverPreferences, catalog_error
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import SolverRegistry
from tests.fakes.declared_backend import (
    FAKE_DECLARED_NAME,
    FAKE_LIMIT_ERROR_CODE,
    FAKE_LIMIT_KEY,
    FakeDeclaredBackend,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "annealbridge"
KNAPSACK = REPO_ROOT / "examples" / "knapsack.json"

ITERATIONS_LIMIT = 1000.0

# The files §13.1 requires to stay untouched when a backend is added.
CORE_FILES_THE_FAKE_FLOWS_THROUGH = [
    "orchestration/optimizer.py",
    "orchestration/limits.py",
    "orchestration/policy.py",
    "interfaces/capabilities.py",
    "validation/problem_validator.py",
]


def make_registry(fake: FakeDeclaredBackend) -> SolverRegistry:
    """Every default backend (built dynamically, not a fixed count) plus the fake."""
    defaults = SolverRegistry.default()
    backends = {name: defaults.get(name) for name in defaults.names()}
    assert FAKE_DECLARED_NAME not in backends
    backends[FAKE_DECLARED_NAME] = fake
    return SolverRegistry(backends)


def make_policy(**overrides) -> ExecutionPolicy:
    return ExecutionPolicy(
        allow_remote=True, limits={FAKE_LIMIT_KEY: ITERATIONS_LIMIT}, **overrides
    )


def make_knapsack(**preferences) -> OptimizationProblem:
    """``examples/knapsack.json`` routed to the fake backend.

    ``SolverPreferences.backend`` is a Literal of the shipped backend names
    (spec §13.2 allows that list to live in ``models/problem.py``), so a
    test-only backend cannot be named through ``model_validate``. The
    preferences are built with ``model_construct`` instead: everything
    downstream (validator, limits, service) reads the object and never
    re-validates it, which is exactly the path a real fifth backend takes
    once its name is added to the Literal.
    """
    payload = json.loads(KNAPSACK.read_text(encoding="utf-8"))
    problem = OptimizationProblem.model_validate(payload)
    solver = SolverPreferences.model_construct(backend=FAKE_DECLARED_NAME, **preferences)
    return problem.model_copy(update={"solver": solver})


@pytest.fixture
def fake() -> FakeDeclaredBackend:
    return FakeDeclaredBackend()


@pytest.fixture
def registry(fake) -> SolverRegistry:
    return make_registry(fake)


@pytest.fixture
def policy() -> ExecutionPolicy:
    return make_policy()


@pytest.fixture
def service(registry, policy) -> OptimizationService:
    return OptimizationService(registry=registry, policy=policy)


class TestCapabilitiesView:
    def test_limits_come_from_the_policy_custom_key(self, registry, policy):
        capabilities = build_capabilities(registry, policy)
        (entry,) = [b for b in capabilities.backends if b.name == FAKE_DECLARED_NAME]
        assert entry.limits == {f"max_{FAKE_LIMIT_KEY}": ITERATIONS_LIMIT}
        assert entry.available is True
        assert entry.enabled is True
        assert entry.remote is True
        assert entry.unavailable_reason is None

    def test_default_backends_are_still_listed(self, registry, policy):
        names = [b.name for b in build_capabilities(registry, policy).backends]
        assert names == registry.names()
        assert names[-1] == FAKE_DECLARED_NAME

    def test_view_and_service_share_one_limit_source(self, registry, policy, fake):
        # The view's number is exactly what the service compares against.
        (entry,) = [
            b
            for b in build_capabilities(registry, policy).backends
            if b.name == FAKE_DECLARED_NAME
        ]
        assert entry.limits[f"max_{FAKE_LIMIT_KEY}"] == policy.limit(FAKE_LIMIT_KEY)
        assert policy.limits_for(fake.capabilities) == entry.limits


class TestSolveThroughTheDeclaredLimit:
    def test_over_limit_is_refused_under_the_declared_code(self, service, fake):
        result = service.solve(make_knapsack(num_reads=5000))

        assert result.status == "resource_limit_exceeded"
        assert result.backend == FAKE_DECLARED_NAME
        assert result.solutions == []
        (error,) = result.errors
        assert error.code == FAKE_LIMIT_ERROR_CODE
        assert "5000" in error.message
        assert "1000" in error.message
        assert error.retryable is False
        # Never clamped: the backend was not called at all.
        assert fake.solve_calls == 0

    def test_uncatalogued_code_has_no_recommended_action(self, service):
        # FAKE_ITERATIONS_LIMIT is not in the catalog; catalog_error's
        # existing behaviour for unknown codes is recommended_action=None.
        assert catalog_error(FAKE_LIMIT_ERROR_CODE, "x").recommended_action is None
        result = service.solve(make_knapsack(num_reads=5000))
        assert result.errors[0].recommended_action is None

    def test_within_limit_solves_successfully(self, service, fake):
        result = service.solve(make_knapsack(num_reads=10))

        assert result.status == "success"
        assert result.backend == FAKE_DECLARED_NAME
        assert result.errors == []
        assert fake.solve_calls == 1
        assert fake.last_preferences is not None
        assert fake.last_preferences.num_reads == 10
        best = result.solutions[0]
        assert best.variables == {"item_a": 1, "item_b": 0, "item_c": 1, "item_d": 0}
        assert best.objective_value == 17
        assert best.hard_constraints_satisfied is True

    def test_exactly_at_limit_is_allowed(self, service, fake):
        result = service.solve(make_knapsack(num_reads=int(ITERATIONS_LIMIT)))
        assert result.status == "success"
        assert fake.solve_calls == 1


class TestValidateFollowsTheDeclaration:
    def test_seed_is_reported_ignored_without_any_name_list(self, service):
        result = service.validate(make_knapsack(seed=7))

        assert result.valid is True
        codes = [warning.code for warning in result.warnings]
        assert "SEED_IGNORED" in codes
        assert "UNKNOWN_BACKEND" not in codes
        (warning,) = [w for w in result.warnings if w.code == "SEED_IGNORED"]
        assert warning.path == "solver.seed"
        assert FAKE_DECLARED_NAME in warning.message

    def test_without_seed_there_is_no_seed_warning(self, service):
        result = service.validate(make_knapsack())
        assert result.valid is True
        assert "SEED_IGNORED" not in [w.code for w in result.warnings]


class TestPolicyMustSupplyTheDeclaredKey:
    def test_service_refuses_to_build_without_the_key(self, registry):
        policy = ExecutionPolicy(allow_remote=True)  # no "iterations"
        with pytest.raises(ValueError) as excinfo:
            OptimizationService(registry=registry, policy=policy)
        message = str(excinfo.value)
        assert FAKE_DECLARED_NAME in message
        assert FAKE_LIMIT_KEY in message

    def test_composition_root_reports_it_as_a_settings_error(self, registry):
        policy = ExecutionPolicy(allow_remote=True)
        with pytest.raises(SettingsError) as excinfo:
            build_state_from_policy(policy, registry)
        message = str(excinfo.value)
        assert FAKE_DECLARED_NAME in message
        assert FAKE_LIMIT_KEY in message

    def test_composition_root_builds_with_the_key(self, registry, policy):
        state = build_state_from_policy(policy, registry)
        assert FAKE_DECLARED_NAME in state.registry.names()
        assert state.service.solve(make_knapsack(num_reads=10)).status == "success"


def test_core_sources_never_mention_the_fake() -> None:
    """§13.1: registering the fake required no edit to the core files."""
    for relative in CORE_FILES_THE_FAKE_FLOWS_THROUGH:
        source = (SRC_ROOT / relative).read_text(encoding="utf-8")
        assert FAKE_DECLARED_NAME not in source, relative
        assert FAKE_LIMIT_ERROR_CODE not in source, relative


# TODO(3a step 9): once ``OptimizationService.recommend()`` exists, add the
# §13.1 assertion that its result lists ``fake_declared`` with
# ``usable=True`` — again with no change to ``orchestration/routing.py``.
