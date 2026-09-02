"""``dwave_availability()`` returns a structured ``AvailabilityStatus`` (3a spec §8).

Installability is decided by ``importlib.util.find_spec("dwave.system")``
and credentials by ``dwave.cloud.config.load_config``; both are patched
here so every category is reachable without any D-Wave package installed.
No network I/O is possible in these tests: nothing is imported from
``dwave.system``.
"""

import importlib.util
import sys
import types

import pytest

from annealbridge.models import AvailabilityStatus
import annealbridge.solvers.metadata as metadata_module
from annealbridge.solvers.metadata import (
    REASON_CONFIG_INVALID,
    REASON_CREDENTIALS_MISSING,
    REASON_NOT_INSTALLED,
    dwave_availability,
)

FAKE_TOKEN = "DEV-FAKE-TOKEN-1234567890abcdefghij"
FAKE_ENDPOINT = "https://fake.dwavesys.example/sapi/"


def _install_fake_ocean_config(monkeypatch, load_config) -> None:
    """Register fake ``dwave.cloud.config`` modules exposing ``load_config``."""
    dwave_mod = types.ModuleType("dwave")
    cloud_mod = types.ModuleType("dwave.cloud")
    config_mod = types.ModuleType("dwave.cloud.config")
    config_mod.load_config = load_config
    dwave_mod.cloud = cloud_mod
    cloud_mod.config = config_mod
    monkeypatch.setitem(sys.modules, "dwave", dwave_mod)
    monkeypatch.setitem(sys.modules, "dwave.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "dwave.cloud.config", config_mod)


def _set_dwave_system_installed(monkeypatch, installed: bool) -> None:
    """Patch ``find_spec`` for ``dwave.system`` only; everything else is real."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "dwave.system":
            return types.SimpleNamespace(name=name) if installed else None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


@pytest.fixture(autouse=True)
def _no_env_token(monkeypatch):
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)


class TestCategories:
    def test_not_installed(self, monkeypatch):
        _set_dwave_system_installed(monkeypatch, False)
        _install_fake_ocean_config(monkeypatch, lambda: {"token": FAKE_TOKEN})

        status = dwave_availability()

        assert isinstance(status, AvailabilityStatus)
        assert status == AvailabilityStatus(
            category="not_installed", detail=REASON_NOT_INSTALLED
        )
        assert status.available is False

    def test_credentials_missing(self, monkeypatch):
        _set_dwave_system_installed(monkeypatch, True)
        _install_fake_ocean_config(monkeypatch, lambda: {"endpoint": FAKE_ENDPOINT})

        assert dwave_availability() == AvailabilityStatus(
            category="credentials_missing", detail=REASON_CREDENTIALS_MISSING
        )

    def test_config_invalid_names_the_dwave_error_code(self, monkeypatch):
        _set_dwave_system_installed(monkeypatch, True)

        def broken_load_config():
            raise ValueError(f"cannot parse config with token={FAKE_TOKEN}")

        _install_fake_ocean_config(monkeypatch, broken_load_config)

        assert dwave_availability() == AvailabilityStatus(
            category="config_invalid",
            detail=REASON_CONFIG_INVALID,
            error_code="DWAVE_CONFIG_INVALID",
        )

    def test_available(self, monkeypatch):
        _set_dwave_system_installed(monkeypatch, True)
        _install_fake_ocean_config(
            monkeypatch, lambda: {"token": FAKE_TOKEN, "endpoint": FAKE_ENDPOINT}
        )

        status = dwave_availability()

        assert status == AvailabilityStatus(category="available")
        assert status.available is True
        assert status.detail is None
        assert status.error_code is None

    def test_env_token_counts_as_configured(self, monkeypatch):
        _set_dwave_system_installed(monkeypatch, True)
        _install_fake_ocean_config(monkeypatch, lambda: {})
        monkeypatch.setenv("DWAVE_API_TOKEN", FAKE_TOKEN)

        assert dwave_availability().available is True


class TestDetailNeverCarriesConfigValues:
    @pytest.mark.parametrize("installed", [False, True])
    def test_detail_is_categorical(self, monkeypatch, installed):
        _set_dwave_system_installed(monkeypatch, installed)

        def broken_load_config():
            raise RuntimeError(f"token={FAKE_TOKEN} endpoint={FAKE_ENDPOINT}")

        _install_fake_ocean_config(monkeypatch, broken_load_config)

        status = dwave_availability()

        assert status.available is False
        for value in (status.detail or "", status.error_code or ""):
            assert FAKE_TOKEN not in value
            assert FAKE_ENDPOINT not in value
        assert status.detail in {
            REASON_NOT_INSTALLED,
            REASON_CREDENTIALS_MISSING,
            REASON_CONFIG_INVALID,
        }

    def test_status_is_evaluated_live_not_cached(self, monkeypatch):
        _set_dwave_system_installed(monkeypatch, True)
        _install_fake_ocean_config(monkeypatch, lambda: {})

        assert dwave_availability().category == "credentials_missing"

        monkeypatch.setenv("DWAVE_API_TOKEN", FAKE_TOKEN)

        assert dwave_availability().category == "available"


class TestBackendsShareTheCheck:
    def test_both_dwave_backends_delegate_to_dwave_availability(self, monkeypatch):
        import annealbridge.solvers.dwave_qpu as qpu_module
        import annealbridge.solvers.leap_hybrid_bqm as leap_module

        sentinel = AvailabilityStatus(category="unavailable", detail="patched")
        monkeypatch.setattr(qpu_module, "dwave_availability", lambda: sentinel)
        monkeypatch.setattr(leap_module, "dwave_availability", lambda: sentinel)

        assert qpu_module.DWaveQPUBackend().is_available() is sentinel
        assert leap_module.LeapHybridBQMBackend().is_available() is sentinel

    def test_module_level_helper_is_what_the_backends_import(self):
        assert metadata_module.dwave_availability is dwave_availability
