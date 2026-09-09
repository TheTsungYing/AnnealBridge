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

import configparser
import os
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


def _install_fake_ocean_config(
    monkeypatch, load_config, get_configfile_paths=None
) -> None:
    """Register fake dwave.cloud.config modules exposing ``load_config``.

    ``get_configfile_paths`` is optional: when it is omitted the fake
    module lacks the attribute entirely, which is how an Ocean version
    without that helper (or no Ocean at all) looks to
    ``_config_fingerprint()`` — and therefore the "no fingerprint, no
    caching" path (2026-09-09 review F-19).
    """
    dwave_mod = types.ModuleType("dwave")
    cloud_mod = types.ModuleType("dwave.cloud")
    config_mod = types.ModuleType("dwave.cloud.config")
    config_mod.load_config = load_config
    if get_configfile_paths is not None:
        config_mod.get_configfile_paths = get_configfile_paths
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


class TestConfigSecretCache:
    """The config-file secret source is cached on a credential fingerprint.

    2026-09-09 review F-19: ``redact()`` runs on every log line and every
    error message, and the source used to call ``load_config()`` — which
    reads and parses files off disk — each time. The value is now cached
    against ``(DWAVE_CONFIG_FILE, DWAVE_PROFILE, [(path, mtime, size)…])``,
    so a rotated token is still picked up but an unchanged config is read
    once. When the fingerprint cannot be computed (no
    ``get_configfile_paths``), nothing is cached and the old live
    behaviour stands.

    All tokens here are synthetic test values, never real credentials.
    """

    TOKEN_A = "s3cr3t-config-token-alpha"
    TOKEN_B = "s3cr3t-config-token-bravo-and-longer"
    RAW_TOKEN = "s3cr3t-raw-config-token-value"

    @pytest.fixture(autouse=True)
    def _isolated_registration(self, monkeypatch):
        """Empty cache, empty redaction tables, one registered source.

        ``_CONFIG_SECRET_CACHE`` is process-level, so it is swapped for a
        fresh dict (and restored) exactly like the declaration tables.
        The two Ocean config env vars are cleared so a developer machine
        that sets them cannot change a fingerprint mid-test.
        """
        monkeypatch.delenv("DWAVE_CONFIG_FILE", raising=False)
        monkeypatch.delenv("DWAVE_PROFILE", raising=False)
        monkeypatch.setattr(ocean_module, "_CONFIG_SECRET_CACHE", {})
        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})
        ocean_module.register_ocean_config_token()

    @staticmethod
    def _counting_load_config(calls: list[int], token: str):
        def load_config():
            calls.append(1)
            return {"token": token}

        return load_config

    def test_an_unchanged_config_is_read_once_for_many_redactions(
        self, monkeypatch
    ) -> None:
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, FAKE_CONFIG_TOKEN),
            get_configfile_paths=lambda: [],
        )

        for _ in range(5):
            assert redact("x " + FAKE_CONFIG_TOKEN) == "x ***"

        assert len(calls) == 1

    def test_switching_profile_invalidates_the_cache(self, monkeypatch) -> None:
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, FAKE_CONFIG_TOKEN),
            get_configfile_paths=lambda: [],
        )

        assert redact("x " + FAKE_CONFIG_TOKEN) == "x ***"
        assert len(calls) == 1

        monkeypatch.setenv("DWAVE_PROFILE", "other")

        assert redact("x " + FAKE_CONFIG_TOKEN) == "x ***"
        assert len(calls) == 2

    def test_an_edited_config_file_invalidates_the_cache(
        self, monkeypatch, tmp_path
    ) -> None:
        path = tmp_path / "dwave.conf"
        path.write_text(f"[defaults]\ntoken = {self.TOKEN_A}\n")
        calls: list[int] = []

        def load_config():
            calls.append(1)
            parser = configparser.ConfigParser()
            parser.read(path)
            return {"token": parser["defaults"]["token"]}

        _install_fake_ocean_config(
            monkeypatch, load_config, get_configfile_paths=lambda: [str(path)]
        )

        assert redact("x " + self.TOKEN_A) == "x ***"
        assert len(calls) == 1

        # Different length (so size changes) *and* a later mtime: either
        # one alone is enough for the fingerprint, both together make the
        # test independent of the filesystem's timestamp resolution.
        path.write_text(f"[defaults]\ntoken = {self.TOKEN_B}\n")
        later = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(later, later))

        assert redact("x " + self.TOKEN_B) == "x ***"
        assert len(calls) == 2

    def test_an_unparseable_config_still_yields_its_raw_token(
        self, monkeypatch, tmp_path
    ) -> None:
        """``invalid`` means ``load_config()`` raised — but the token is
        still on disk, and still must not reach an error message."""
        path = tmp_path / "dwave.conf"
        path.write_text(f"[defaults]\ntoken = {self.RAW_TOKEN}\n")

        def load_config():
            raise RuntimeError("bad config")

        _install_fake_ocean_config(
            monkeypatch, load_config, get_configfile_paths=lambda: [str(path)]
        )

        redacted = redact(f"request failed with credential {self.RAW_TOKEN}")

        assert self.RAW_TOKEN not in redacted
        assert "***" in redacted

    def test_nothing_is_cached_without_a_fingerprint(self, monkeypatch) -> None:
        """No ``get_configfile_paths`` → no fingerprint → the old live read."""
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch, self._counting_load_config(calls, FAKE_CONFIG_TOKEN)
        )

        for _ in range(3):
            assert redact("x " + FAKE_CONFIG_TOKEN) == "x ***"

        assert len(calls) == 3
