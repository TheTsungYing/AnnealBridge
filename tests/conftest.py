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


@pytest.fixture(autouse=True)
def _clear_dwave_api_token(monkeypatch):
    """Remove ``DWAVE_API_TOKEN`` from the environment for every test.

    Ocean honours this variable, so a developer machine that has it set
    would otherwise change the outcome of credential and redaction tests.
    Tests that need the variable set it themselves after this fixture has
    run, and are unaffected.
    """
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
