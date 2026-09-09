"""Architecture test: a fifth backend needs zero core changes (3a spec §13.1).

``FakeDeclaredBackend`` (``tests/fakes/declared_backend.py``) is registered
next to every default backend. The service, the validator and the
capabilities view must handle it purely from its declaration:

* the capabilities view and the service read the same policy limit for
  its custom ``iterations`` key (drift 1 of the 3a plan);
* its declared parameter limit is enforced by the generic check, never
  clamped, under its own (uncatalogued) error code;
* the validator's advisory warnings follow the capability flags;
* a policy that lacks the declared key is refused at construction;
* ``service.recommend()`` lists it as usable, ranked purely from its
  declaration (3a step 9);
* its credential (env var / header declared in ``capabilities.credentials``)
  is masked by the shared redaction the moment it is registered — no edit
  to ``solvers/metadata.py`` (2026-09-09 review F-10).

The proof that none of this needed a code change is
``test_core_sources_never_mention_the_fake``: the files the fake flows
through do not contain its name, its error code or its credential names.
"""

import json
import logging
from pathlib import Path

import pytest

from annealbridge.config import SettingsError
from annealbridge.exceptions import SolverExecutionError
from annealbridge.interfaces.capabilities import build_capabilities
from annealbridge.interfaces.composition import build_state_from_policy
from annealbridge.models import OptimizationProblem, SolverPreferences, catalog_error
from annealbridge.orchestration import ExecutionPolicy, OptimizationService
from annealbridge.solvers import SolverRegistry
import annealbridge.solvers.metadata as metadata_module
from annealbridge.solvers.metadata import guarded_call, redact
from tests.fakes.declared_backend import (
    FAKE_CREDENTIAL_ENV,
    FAKE_CREDENTIAL_HEADER,
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
    "orchestration/routing.py",
    "interfaces/capabilities.py",
    "validation/problem_validator.py",
    # Review F-10: the redaction data is declared by the backend, so the
    # shared redaction module is on the zero-change list too.
    "solvers/metadata.py",
]

# Never a real key; hyphenated so no vendor token-shape pattern could match
# it — only the declared env var can mask it.
FAKE_KEY = "sk-7thvendor-SUPERSECRET-0123456789"


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
        # The declared key, then the two service-level ceilings every
        # backend carries (spec §11.4): remote retries by the ``remote``
        # flag, then top_k.
        assert entry.limits == {
            f"max_{FAKE_LIMIT_KEY}": ITERATIONS_LIMIT,
            "max_remote_retries": 3,
            "max_top_k": 1000,
        }
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


class TestRecommendFollowsTheDeclaration:
    def test_fake_is_listed_and_usable(self, service, fake):
        result = service.recommend(make_knapsack())

        assert result.valid is True
        (entry,) = [e for e in result.recommendations if e.backend == FAKE_DECLARED_NAME]
        assert entry.usable is True
        assert entry.blocking == []
        assert entry.model_type == "bqm"
        # Remote, BQM, multiple samples: tier 4 with no single-sample note.
        assert entry.reasons == ["R_REMOTE"]
        assert fake.solve_calls == 0

    def test_over_limit_preference_makes_it_unusable(self, service, fake):
        result = service.recommend(make_knapsack(num_reads=5000))

        (entry,) = [e for e in result.recommendations if e.backend == FAKE_DECLARED_NAME]
        assert entry.usable is False
        assert [b.code for b in entry.blocking] == [FAKE_LIMIT_ERROR_CODE]
        assert entry.reasons == ["R_UNUSABLE", "R_REMOTE"]
        assert fake.solve_calls == 0

    def test_every_default_backend_is_still_ranked(self, service, registry):
        result = service.recommend(make_knapsack())
        assert sorted(e.backend for e in result.recommendations) == sorted(
            registry.names()
        )
        assert [e.rank for e in result.recommendations] == list(
            range(1, len(registry.names()) + 1)
        )


class TestCredentialsFollowTheDeclaration:
    """Review F-10: the fake's key is masked because it *declared* the env
    var and header, not because ``solvers/metadata.py`` knows the vendor."""

    @pytest.fixture(autouse=True)
    def _key_in_env(self, monkeypatch):
        monkeypatch.setenv(FAKE_CREDENTIAL_ENV, FAKE_KEY)

    def test_declared_env_value_is_masked_once_registered(self, registry):
        assert FAKE_KEY not in redact(f"auth failed for key {FAKE_KEY}")
        assert redact(f"auth failed for key {FAKE_KEY}") == "auth failed for key ***"

    def test_declared_header_line_is_masked_even_for_an_unknown_value(self, registry):
        text = f"400 Bad Request\n{FAKE_CREDENTIAL_HEADER}: some-other-key-value\n"
        assert redact(text) == f"400 Bad Request\n{FAKE_CREDENTIAL_HEADER}: ***\n"
        json_form = f'{{"{FAKE_CREDENTIAL_HEADER}": "some-other-key-value"}}'
        assert redact(json_form) == f'{{"{FAKE_CREDENTIAL_HEADER}": "***"}}'

    def test_guarded_call_never_lets_the_key_out(self, registry):
        def fail():
            raise RuntimeError(f"401 Unauthorized ({FAKE_CREDENTIAL_HEADER} {FAKE_KEY})")

        with pytest.raises(SolverExecutionError) as excinfo:
            guarded_call("Acme solve failed", lambda exc: "REMOTE_SOLVER_ERROR", fail)

        message = str(excinfo.value)
        assert FAKE_KEY not in message
        assert "***" in message
        assert excinfo.value.__cause__ is None

    def test_full_solve_failure_leaks_nothing(self, policy, caplog):
        fake = FakeDeclaredBackend(
            raise_on_solve=RuntimeError(f"vendor rejected key {FAKE_KEY}")
        )
        service = OptimizationService(registry=make_registry(fake), policy=policy)
        caplog.set_level(logging.DEBUG, logger="annealbridge")

        result = service.solve(make_knapsack(num_reads=10))

        assert result.status == "solver_error"
        assert result.backend == FAKE_DECLARED_NAME
        assert fake.solve_calls == 1
        assert FAKE_KEY not in result.model_dump_json()
        assert "***" in result.errors[0].message
        assert caplog.records
        for record in caplog.records:
            assert FAKE_KEY not in record.getMessage()

    def test_masking_really_comes_from_the_declaration(self, monkeypatch):
        # With no declaration at all the value is not a candidate: the
        # protection is the declaration, not something hidden in the core.
        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        assert redact(f"key {FAKE_KEY}") == f"key {FAKE_KEY}"
        SolverRegistry({FAKE_DECLARED_NAME: FakeDeclaredBackend()})
        assert redact(f"key {FAKE_KEY}") == "key ***"

    def test_a_backend_that_declares_nothing_masks_nothing_and_breaks_nothing(
        self, monkeypatch
    ):
        defaults = SolverRegistry.default()
        local = {
            name: defaults.get(name)
            for name in defaults.names()
            if not defaults.get(name).capabilities.remote
        }
        assert local
        # Only *after* the default registry has been built: building one
        # (re)declares every shipped backend.
        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        SolverRegistry(local)
        assert redact(f"key {FAKE_KEY}") == f"key {FAKE_KEY}"
        assert metadata_module.credential_env_vars() == []


def test_core_sources_never_mention_the_fake() -> None:
    """§13.1: registering the fake required no edit to the core files."""
    for relative in CORE_FILES_THE_FAKE_FLOWS_THROUGH:
        source = (SRC_ROOT / relative).read_text(encoding="utf-8")
        assert FAKE_DECLARED_NAME not in source, relative
        assert FAKE_LIMIT_ERROR_CODE not in source, relative
        assert FAKE_CREDENTIAL_ENV not in source, relative
        assert FAKE_CREDENTIAL_HEADER not in source, relative

