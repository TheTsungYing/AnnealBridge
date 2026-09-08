"""Shared pytest fixtures for the AnnealBridge test suite."""

import json
from pathlib import Path

import pytest

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


# Every vendor credential variable the runtime honours (3b spec §20.5).
# ``FUJITSU_DA_URL`` is not a secret, but it is part of the same "is this
# backend configured?" answer, so it is cleared with the keys.
_CREDENTIAL_ENV_VARS = ("DWAVE_API_TOKEN", "FUJITSU_DA_API_KEY", "FUJITSU_DA_URL")


@pytest.fixture(autouse=True)
def _clear_credential_env(monkeypatch):
    """Remove every vendor credential variable from the environment.

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
