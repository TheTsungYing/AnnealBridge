"""Shared pytest fixtures for the AnnealBridge test suite."""

import json
import os
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


# Prefix of every variable ``annealbridge.config`` reads.
_SETTINGS_ENV_PREFIX = "ANNEALBRIDGE_"


@pytest.fixture(scope="session")
def _empty_ocean_config_file(tmp_path_factory) -> Path:
    """One empty Ocean config file for the whole session.

    Session-scoped so the suite does not create a temporary directory per
    test; nothing ever writes to it.
    """
    path = tmp_path_factory.mktemp("ocean-config") / "dwave.conf"
    path.write_text("", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolate_local_configuration(monkeypatch, _empty_ocean_config_file):
    """Hide the developer machine's own configuration (2026-11-18 review R-07).

    Clearing the credential variables above is not enough on its own:

    * every ``ANNEALBRIDGE_*`` variable changes what the settings layer
      builds -- ``ANNEALBRIDGE_ENABLED_BACKENDS=exact`` alone removes seven
      backends from the CLI and MCP views -- so all of them are removed;
    * Ocean reads its token from a ``dwave.conf`` found through
      ``DWAVE_CONFIG_FILE`` or the per-user and system search paths, and
      ``DWAVE_PROFILE`` selects a section in it. Pointing
      ``DWAVE_CONFIG_FILE`` at an empty file makes Ocean read that file and
      nothing else, so a real token on the machine never reaches a test.

    Unlike ``_clear_credential_env`` this fixture is not overridden for the
    live tests: they reach the vendor through ``DWAVE_API_TOKEN`` /
    ``FUJITSU_DA_API_KEY``, which it leaves alone. Tests that need any of
    these variables set them themselves after this fixture has run.
    """
    for name in list(os.environ):
        if name.startswith(_SETTINGS_ENV_PREFIX):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DWAVE_CONFIG_FILE", str(_empty_ocean_config_file))
    monkeypatch.delenv("DWAVE_PROFILE", raising=False)


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


# ---------------------------------------------------------------------------
# 2026-11-18 review R-24: "no skip, no xfail" is enforced, not just written
# down. A test that is skipped or xfailed fails the session unless it carries
# the ``remote`` marker -- the live tests' opt-in / missing-credential guard
# is the one skip the suite allows (docs/testing.md).
#
# The decision is taken per report rather than from the ``-m`` expression, so
# it holds for a single file, a ``-k`` selection and ``-m remote`` alike: a
# deselected test produces no report at all, and only a test that actually
# ran and was skipped or xfailed without the ``remote`` marker is recorded.
# ---------------------------------------------------------------------------

REMOTE_MARKER = "remote"
_REMOTE_LIVE_DIRECTORY = "remote_live"

# ``(node id, reason)`` of every forbidden skip / xfail seen in this session.
_FORBIDDEN_OUTCOMES: list[tuple[str, str]] = []


def _skip_reason(report) -> str:
    """Short human-readable reason of a skipped or xfailed report."""
    if hasattr(report, "wasxfail"):
        return f"xfail: {report.wasxfail}" if report.wasxfail else "xfail"
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return "skipped"


def pytest_runtest_logreport(report) -> None:
    # An xfailed test is reported as skipped too (with ``wasxfail`` set).
    if report.skipped and REMOTE_MARKER not in report.keywords:
        _FORBIDDEN_OUTCOMES.append((report.nodeid, _skip_reason(report)))


def pytest_collectreport(report) -> None:
    # A module-level ``pytest.skip`` / ``importorskip`` skips the collector,
    # so no test report ever exists for it.
    if report.skipped and _REMOTE_LIVE_DIRECTORY not in report.nodeid.split("/"):
        _FORBIDDEN_OUTCOMES.append((report.nodeid, _skip_reason(report)))


def pytest_sessionfinish(session, exitstatus) -> None:
    # Only a run that would otherwise succeed is turned into a failure --
    # "no tests collected" included, which is what a run whose only module
    # skipped itself reports. Any other exit status already says the run is
    # not clean and is left as it is.
    clean = (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED)
    if _FORBIDDEN_OUTCOMES and exitstatus in clean:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter) -> None:
    if not _FORBIDDEN_OUTCOMES:
        return
    terminalreporter.section("forbidden skip / xfail", sep="=", red=True, bold=True)
    terminalreporter.line(
        f"Only tests marked '{REMOTE_MARKER}' may skip; nothing may xfail "
        "(docs/testing.md). The run is reported as failed because of:"
    )
    for nodeid, reason in _FORBIDDEN_OUTCOMES:
        terminalreporter.line(f"  {nodeid}: {reason}")
