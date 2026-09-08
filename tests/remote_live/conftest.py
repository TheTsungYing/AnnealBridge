"""Fixtures for the opt-in live remote tests (spec §28, 3b §20.5).

These tests talk to real vendor hardware and consume paid quota, so they
are strictly opt-in: run them with ``pytest -m remote`` and the credential
the module needs.

Which credential that is comes from the test module itself: a module sets
``REQUIRED_ENV = "FUJITSU_DA_API_KEY"`` (say) at import time and
:func:`_require_live_opt_in` skips on *that* variable. Modules that do not
declare one fall back to ``DWAVE_API_TOKEN``, so the three shipped D-Wave
live files need no change.
"""

import os

import pytest

from annealbridge.orchestration import ExecutionPolicy


@pytest.fixture(autouse=True)
def _clear_credential_env():
    """Override the root conftest fixture of the same name with a no-op.

    The root fixture deletes every vendor credential variable for every
    test so credential and redaction tests are deterministic; the live
    tests are the one place that needs the real credentials to reach the
    vendor.
    """
    yield


@pytest.fixture(autouse=True)
def _require_live_opt_in(request):
    # This skip is the ONLY allowed skip in the whole suite (spec §28, §36):
    # it does not hide a bug, it prevents accidental paid-quota consumption
    # when the tests are invoked without explicit opt-in (`-m remote`) or
    # without credentials. Everything else in the suite must run for real.
    markexpr = request.config.getoption("markexpr", default="")
    if "remote" not in markexpr or "not remote" in markexpr:
        pytest.skip("live remote tests are opt-in: run with `pytest -m remote`")
    required = getattr(request.module, "REQUIRED_ENV", "DWAVE_API_TOKEN")
    if not os.environ.get(required):
        pytest.skip(f"{required} is not set; cannot reach the remote solver")


@pytest.fixture
def live_policy():
    """Policy that permits remote solving; retries stay disabled (default)."""
    return ExecutionPolicy(allow_remote=True)
