"""The Ocean configuration source behind availability and redaction.

Everything D-Wave-specific about credentials lives in ``solvers.ocean``
(2026-09-09 review F-10): the ``dwave.cloud.config`` lazy import, the
``ok`` / ``missing`` / ``invalid`` classification, and the config-*file*
token that no environment variable can mask. A D-Wave backend contributes
that token to the shared redaction from its constructor
(``register_ocean_config_token`` → ``metadata.register_secret_source``),
which is what the redaction tests at the bottom pin.

Nothing here imports ``dwave.system``: no network I/O is possible.
All tokens are synthetic test values, never real credentials.
"""

import sys
import types

import pytest

import annealbridge.solvers.metadata as metadata_module
from annealbridge.solvers.metadata import redact
import annealbridge.solvers.ocean as ocean_module
from annealbridge.solvers.ocean import (
    TOKEN_ENV,
    _resolve_ocean_config,
    ocean_config_status,
    ocean_config_token,
)

FAKE_TOKEN = "DEV-" + "a" * 24
# Deliberately does not match the ``DEV-[A-Za-z0-9]{20,}`` shape D-Wave
# declares, so only a live lookup (env var or config source) can mask it.
FAKE_ENV_TOKEN = "DEV-FAKE-TOKEN-1234567890abcdefghij"
FAKE_CONFIG_TOKEN = "s3cr3t-ocean-token-value"


def _install_fake_ocean_config(monkeypatch, load_config) -> None:
    """Register fake dwave.cloud.config modules exposing ``load_config``."""
    dwave_mod = types.ModuleType("dwave")
    cloud_mod = types.ModuleType("dwave.cloud")
    config_mod = types.ModuleType("dwave.cloud.config")
    config_mod.load_config = load_config
    dwave_mod.cloud = cloud_mod
    cloud_mod.config = config_mod
    monkeypatch.setitem(sys.modules, "dwave", dwave_mod)
    monkeypatch.setitem(sys.modules, "dwave.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "dwave.cloud.config", config_mod)


def _block_ocean_config_import(monkeypatch) -> None:
    """Make ``from dwave.cloud.config import load_config`` fail.

    Mirrors an environment without dwave-cloud-client installed, so the
    env var is the only remaining configuration source.
    """
    monkeypatch.setitem(sys.modules, "dwave.cloud.config", None)


class TestOceanConfigStatus:
    def test_missing_without_env_token(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)

        assert ocean_config_status() == "missing"

    def test_ok_with_env_token(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(TOKEN_ENV, FAKE_ENV_TOKEN)

        assert ocean_config_status() == "ok"

    def test_empty_env_token_is_missing(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(TOKEN_ENV, "")

        assert ocean_config_status() == "missing"

    def test_no_error_when_dwave_is_not_installed(self, monkeypatch) -> None:
        for name in list(sys.modules):
            if name == "dwave" or name.startswith("dwave."):
                monkeypatch.delitem(sys.modules, name)

        assert ocean_config_status() in {"ok", "missing", "invalid"}


class TestResolveOceanConfig:
    """The single lazy touch of ``dwave.cloud.config`` behind both
    ``ocean_config_status()`` and the config-token secret source."""

    def test_missing_without_any_source(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)

        assert _resolve_ocean_config() == ("missing", None)

    def test_env_token_counts_as_configured_but_is_not_the_config_token(
        self, monkeypatch
    ) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(TOKEN_ENV, FAKE_ENV_TOKEN)

        # The env var is masked through the credential declaration; the
        # config-token slot only ever carries what load_config() returned.
        assert _resolve_ocean_config() == ("ok", None)

    def test_config_token_is_returned_with_ok(self, monkeypatch) -> None:
        _install_fake_ocean_config(monkeypatch, lambda: {"token": FAKE_TOKEN})

        assert _resolve_ocean_config() == ("ok", FAKE_TOKEN)
        assert ocean_config_token() == FAKE_TOKEN

    def test_unparseable_config_is_invalid_even_with_env_token(
        self, monkeypatch
    ) -> None:
        def broken_load_config():
            raise RuntimeError("bad dwave.conf")

        _install_fake_ocean_config(monkeypatch, broken_load_config)
        monkeypatch.setenv(TOKEN_ENV, FAKE_ENV_TOKEN)

        assert _resolve_ocean_config() == ("invalid", None)
        assert ocean_config_status() == "invalid"

    def test_empty_config_token_is_missing(self, monkeypatch) -> None:
        _install_fake_ocean_config(monkeypatch, lambda: {"token": ""})

        assert _resolve_ocean_config() == ("missing", None)

    @pytest.mark.parametrize("config", [None, {"endpoint": "x"}, {"token": ""}])
    def test_missing_or_empty_token_yields_no_config_token(
        self, monkeypatch, config
    ) -> None:
        _install_fake_ocean_config(monkeypatch, lambda: config)

        assert ocean_config_token() is None


class TestEnvToken:
    def test_env_token_is_read_live(self, monkeypatch) -> None:
        assert ocean_module._env_token() is None

        monkeypatch.setenv(TOKEN_ENV, FAKE_ENV_TOKEN)

        assert ocean_module._env_token() == FAKE_ENV_TOKEN

    def test_empty_env_token_is_none(self, monkeypatch) -> None:
        monkeypatch.setenv(TOKEN_ENV, "")

        assert ocean_module._env_token() is None


class TestConfigTokenReachesTheSharedRedaction:
    """A config-file token is in no environment variable, so the backend
    registers it as a live secret source when it is constructed."""

    def test_token_is_masked_once_a_dwave_backend_exists(self, monkeypatch) -> None:
        from annealbridge.solvers.dwave_qpu import DWaveQPUBackend

        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})
        DWaveQPUBackend(sampler_factory=lambda: None)
        monkeypatch.setattr(
            ocean_module, "_resolve_ocean_config", lambda: ("ok", FAKE_CONFIG_TOKEN)
        )

        redacted = redact(f"request failed with credential {FAKE_CONFIG_TOKEN}")

        assert FAKE_CONFIG_TOKEN not in redacted
        assert "***" in redacted

    def test_the_backend_constructor_is_what_registers_the_source(
        self, monkeypatch
    ) -> None:
        from annealbridge.solvers.dwave_qpu import DWaveQPUBackend

        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})
        monkeypatch.setattr(
            ocean_module, "_resolve_ocean_config", lambda: ("ok", FAKE_CONFIG_TOKEN)
        )
        text = f"request failed with credential {FAKE_CONFIG_TOKEN}"

        # No D-Wave backend has been built since the tables were emptied:
        # nothing knows about the Ocean config file.
        assert redact(text) == text

        DWaveQPUBackend(sampler_factory=lambda: None)

        assert "ocean_config" in metadata_module._SECRET_SOURCES
        assert redact(text) == "request failed with credential ***"

    def test_a_failing_config_read_does_not_break_redaction(self, monkeypatch) -> None:
        from annealbridge.solvers.dwave_qpu import DWaveQPUBackend

        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})
        DWaveQPUBackend(sampler_factory=lambda: None)

        def broken_resolve():
            raise RuntimeError("unreadable config")

        monkeypatch.setattr(ocean_module, "_resolve_ocean_config", broken_resolve)

        assert redact("plain message") == "plain message"
