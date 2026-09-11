"""Shared pytest fixtures for the AnnealBridge test suite."""

import json
from pathlib import Path

import pytest

import annealbridge.solvers.ocean as ocean_module
from annealbridge.solvers import SolverRegistry

# The repository's ``examples/`` directory, shared by every test that loads a
# shipped example problem.
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def examples_dir() -> Path:
    """The repository's ``examples/`` directory."""
    return EXAMPLES_DIR


@pytest.fixture
def load_example():
    """Return ``load_example(name, **solver_overrides) -> dict``.

    Loads a shipped example problem as a plain payload dict with its
    ``solver`` block merged with the given overrides.
    """

    def _load(name: str, **solver_overrides) -> dict:
        data = json.loads((EXAMPLES_DIR / name).read_text())
        data["solver"] = {**data.get("solver", {}), **solver_overrides}
        return data

    return _load


def _declared_credential_env_vars() -> tuple[str, ...]:
    """Every credential variable the default registry's backends declare.

    Derived from ``SolverCapabilities.credentials`` rather than hardcoded
    (2026-09-09 review F-10), so a new backend is covered by the fixture
    below the moment it declares its env vars — no test-suite change.
    """
    registry = SolverRegistry.default()
    names: list[str] = []
    for name in registry.names():
        for env_var in registry.get(name).capabilities.credentials.env_vars:
            if env_var not in names:
                names.append(env_var)
    return tuple(names)


# ``FUJITSU_DA_URL`` is not a secret (so no backend declares it), but it is
# part of the same "is this backend configured?" answer, so it is cleared
# with the declared keys.
_CREDENTIAL_ENV_VARS = _declared_credential_env_vars() + ("FUJITSU_DA_URL",)


@pytest.fixture(autouse=True)
def _clear_credential_env(monkeypatch):
    """Remove every declared vendor credential variable from the environment.

    Ocean honours ``DWAVE_API_TOKEN`` and the Fujitsu Digital Annealer
    backend honours ``FUJITSU_DA_API_KEY`` / ``FUJITSU_DA_URL``, so a
    developer machine that has any of them set would otherwise change what
    ``is_available()`` reports — and with it the capabilities view, the
    recommend ranking, the CLI's backend table and every credential and
    redaction test. Tests that need a variable set it themselves after this
    fixture has run, and are unaffected.
    """
    for name in _CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _reset_ocean_config_cache(monkeypatch):
    """Give every test an empty ``_CONFIG_RESOLUTION_CACHE``.

    ``_resolve_ocean_config()`` memoises its parse in a single module-level
    slot keyed by ``(DWAVE_API_TOKEN, _config_fingerprint())``. On a machine
    whose ``.venv`` has dwave-cloud-client installed but no Ocean config
    file, that fingerprint is ``((None, None), ())`` — exactly the key a
    test produces when it fakes ``get_configfile_paths`` to return an empty
    list. A resolution cached by one test would then be served to the next
    one under a different fake ``load_config``. Swapping the dict (restored
    by monkeypatch) starts every test from an empty cache instead.
    """
    monkeypatch.setattr(ocean_module, "_CONFIG_RESOLUTION_CACHE", {})
