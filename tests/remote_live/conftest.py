"""Fixtures for the opt-in live D-Wave tests (spec §28).

These tests talk to real D-Wave hardware and consume Leap quota, so they are
strictly opt-in: run them with ``pytest -m remote`` and a configured
``DWAVE_API_TOKEN``.
"""

import os

import pytest

from annealbridge.orchestration import ExecutionPolicy


@pytest.fixture(autouse=True)
def _clear_dwave_api_token():
    """Override the root conftest fixture of the same name with a no-op.

    The root fixture deletes ``DWAVE_API_TOKEN`` for every test so credential
    and redaction tests are deterministic; the live tests are the one place
    that needs the real token to reach D-Wave.
    """
    yield


@pytest.fixture(autouse=True)
def _require_live_opt_in(request):
    # This skip is the ONLY allowed skip in the whole suite (spec §28, §36):
    # it does not hide a bug, it prevents accidental Leap-quota consumption
    # when the tests are invoked without explicit opt-in (`-m remote`) or
    # without credentials. Everything else in the suite must run for real.
    markexpr = request.config.getoption("markexpr", default="")
    if "remote" not in markexpr or "not remote" in markexpr:
        pytest.skip("live D-Wave tests are opt-in: run with `pytest -m remote`")
    if not os.environ.get("DWAVE_API_TOKEN"):
        pytest.skip("DWAVE_API_TOKEN is not set; cannot reach D-Wave Leap")


@pytest.fixture
def live_policy():
    """Policy that permits remote solving; retries stay disabled (default)."""
    return ExecutionPolicy(allow_remote=True)
