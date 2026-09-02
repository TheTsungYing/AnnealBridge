"""Shared pytest fixtures for the AnnealBridge test suite."""

import pytest


@pytest.fixture(autouse=True)
def _clear_dwave_api_token(monkeypatch):
    """Remove ``DWAVE_API_TOKEN`` from the environment for every test.

    Ocean honours this variable, so a developer machine that has it set
    would otherwise change the outcome of credential and redaction tests.
    Tests that need the variable set it themselves after this fixture has
    run, and are unaffected.
    """
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
