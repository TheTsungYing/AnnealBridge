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
import annealbridge.solvers.ocean as ocean_module
from annealbridge.solvers.metadata import redact
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
    registers it as a live secret source when it is constructed.

    These tests patch ``_resolve_ocean_config`` directly, which bypasses
    the config fingerprint that ``_ocean_config_secrets()`` caches on, so
    each one starts from an empty cache (see the fixture below).
    """

    @pytest.fixture(autouse=True)
    def _fresh_config_secret_cache(self, monkeypatch):
        """Swap ``_CONFIG_SECRET_CACHE`` for an empty dict per test.

        With dwave-cloud-client installed ``_config_fingerprint()`` is a
        real value, so a secret tuple cached by an earlier test (in this
        file or another) under the same fingerprint would be returned
        instead of the patched ``_resolve_ocean_config`` result — the
        tests then pass alone but fail inside the whole ``tests/unit``
        run. Without dwave-cloud-client nothing is cached and the swap is
        a no-op. monkeypatch restores the module dict afterwards.
        """
        monkeypatch.setattr(ocean_module, "_CONFIG_SECRET_CACHE", {})

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
    against ``(DWAVE_API_TOKEN, (DWAVE_CONFIG_FILE, DWAVE_PROFILE,
    [(path, mtime, size)…]))``, so a rotated token is still picked up but an
    unchanged config is read once. When the fingerprint cannot be computed
    (no ``get_configfile_paths``), nothing is cached and the old live
    behaviour stands.

    The env token is in that key since 2026-09-11 review F-02; the
    ``load_config`` fakes below merge it the way Ocean's own does.

    All tokens here are synthetic test values, never real credentials.
    """

    TOKEN_A = "s3cr3t-config-token-alpha"
    TOKEN_B = "s3cr3t-config-token-bravo-and-longer"
    RAW_TOKEN = "s3cr3t-raw-config-token-value"
    ENV_TOKEN = "s3cr3t-env-token-charlie-value"
    ENV_TOKEN_ROTATED = "s3cr3t-env-token-delta-rotated"

    @pytest.fixture(autouse=True)
    def _isolated_registration(self, monkeypatch):
        """Empty cache, empty redaction tables, one registered source.

        ``_CONFIG_SECRET_CACHE`` is process-level, so it is swapped for a
        fresh dict (and restored) exactly like the declaration tables; so
        is ``_CONFIG_RESOLUTION_CACHE``, which shares that key shape, so
        that the two caches miss and hit together and the ``load_config``
        counts below mean what they say.
        The Ocean config env vars — the two selectors and the API token —
        are cleared so a developer machine that sets them cannot change a
        cache key mid-test. With ``_DECLARATIONS`` empty, nothing masks the
        env token *except* the config secret source, which is what makes
        the F-02 assertions below meaningful.
        """
        monkeypatch.delenv("DWAVE_CONFIG_FILE", raising=False)
        monkeypatch.delenv("DWAVE_PROFILE", raising=False)
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        monkeypatch.setattr(ocean_module, "_CONFIG_SECRET_CACHE", {})
        monkeypatch.setattr(ocean_module, "_CONFIG_RESOLUTION_CACHE", {})
        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})
        ocean_module.register_ocean_config_token()

    @staticmethod
    def _counting_load_config(calls: list[int], token: str):
        """A counting ``load_config()`` that merges the env var as Ocean does.

        Ocean's real ``load_config()`` lets ``DWAVE_API_TOKEN`` override the
        config file's ``token``, so whenever the env var is set the value the
        secret source caches is the env token and *not* the file's — the
        reason the env token has to be part of the cache key (F-02).
        """

        def load_config():
            calls.append(1)
            env_token = os.environ.get(TOKEN_ENV)
            return {"token": env_token or token}

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

    def test_removing_the_env_token_uncovers_the_config_file_token(
        self, monkeypatch
    ) -> None:
        """2026-09-11 review F-02: the fix, stated as the leak it closes.

        The config file holds ``TOKEN_B`` while the env var holds
        ``ENV_TOKEN``; Ocean's merge means the secret source sees only the
        env token. Unsetting the env var edits no file, so the config
        fingerprint does not move — with the fingerprint as the whole cache
        key, the source kept masking the env token that is gone and the file
        token, now the effective credential, went out unmasked.
        """
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_B),
            get_configfile_paths=lambda: [],
        )
        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN)

        assert redact("x " + self.ENV_TOKEN) == "x ***"
        assert len(calls) == 1

        monkeypatch.delenv(TOKEN_ENV)

        assert redact("x " + self.TOKEN_B) == "x ***"
        # The config was genuinely re-read: the cached tuple could not have
        # held TOKEN_B.
        assert len(calls) == 2

    def test_rotating_the_env_token_invalidates_the_cache(self, monkeypatch) -> None:
        """Same key, other direction: a replaced env token is re-read."""
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )
        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN)

        assert redact("x " + self.ENV_TOKEN) == "x ***"
        assert len(calls) == 1

        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN_ROTATED)

        assert redact("x " + self.ENV_TOKEN_ROTATED) == "x ***"
        assert len(calls) == 2

    def test_an_unchanged_env_token_still_hits_the_cache(self, monkeypatch) -> None:
        """Adding the env token to the key must not defeat the cache."""
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )
        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN)

        for _ in range(5):
            assert redact("x " + self.ENV_TOKEN) == "x ***"

        assert len(calls) == 1

    def test_the_cache_stays_a_single_slot(self, monkeypatch) -> None:
        """Env tokens come and go; the cache never grows past one entry."""
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )

        for token in (self.ENV_TOKEN, self.ENV_TOKEN_ROTATED):
            monkeypatch.setenv(TOKEN_ENV, token)
            assert redact("x " + token) == "x ***"
            assert len(ocean_module._CONFIG_SECRET_CACHE) == 1

        monkeypatch.delenv(TOKEN_ENV)

        assert redact("x " + self.TOKEN_A) == "x ***"
        assert len(ocean_module._CONFIG_SECRET_CACHE) == 1
        assert len(calls) == 3


class TestResolveOceanConfigCache:
    """The Ocean config *parse* is memoised on the same credential fingerprint.

    ``build_capabilities()`` asks all eight backends for ``is_available()``,
    and the three D-Wave ones each reach ``_resolve_ocean_config()``, so one
    capabilities query used to parse the INI three times. The resolution is
    now cached in a single slot keyed by ``(DWAVE_API_TOKEN,
    _config_fingerprint())`` — the very key ``_ocean_config_secrets()``
    already used — so identical inputs parse once while *any* change to the
    env token, a selector env var or a config file's path / mtime / size
    recomputes on the next call. Without a fingerprint (no
    ``get_configfile_paths``) every call parses live, as before.

    Each test drives invalidation through the fingerprint — env vars or file
    mtime — never by emptying the cache by hand, so what is pinned is the
    key, not the bookkeeping.

    All tokens here are synthetic test values, never real credentials.
    """

    TOKEN_A = "s3cr3t-resolved-token-alpha"
    TOKEN_B = "s3cr3t-resolved-token-bravo-and-longer"
    ENV_TOKEN = "s3cr3t-resolved-env-token-echo"
    ENV_TOKEN_ROTATED = "s3cr3t-resolved-env-token-foxtrot"

    @pytest.fixture(autouse=True)
    def _isolated_caches(self, monkeypatch):
        """Empty both fingerprint caches and clear the Ocean config env vars.

        The root ``conftest`` already swaps ``_CONFIG_RESOLUTION_CACHE`` per
        test; doing it here too keeps this class readable on its own, and
        ``_CONFIG_SECRET_CACHE`` is swapped alongside it because both hold
        the same key and the redaction path would otherwise hide a parse.
        The two selector env vars are cleared so a developer machine that
        sets them cannot move a cache key mid-test.
        """
        monkeypatch.delenv("DWAVE_CONFIG_FILE", raising=False)
        monkeypatch.delenv("DWAVE_PROFILE", raising=False)
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        monkeypatch.setattr(ocean_module, "_CONFIG_SECRET_CACHE", {})
        monkeypatch.setattr(ocean_module, "_CONFIG_RESOLUTION_CACHE", {})

    @staticmethod
    def _counting_load_config(calls: list[int], token: str):
        """A counting ``load_config()`` that merges the env var as Ocean does."""

        def load_config():
            calls.append(1)
            env_token = os.environ.get(TOKEN_ENV)
            return {"token": env_token or token}

        return load_config

    def test_an_unchanged_config_is_parsed_once(self, monkeypatch) -> None:
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )

        results = [_resolve_ocean_config() for _ in range(5)]

        assert results == [("ok", self.TOKEN_A)] * 5
        assert len(calls) == 1

    def test_an_edited_config_file_is_reflected_immediately(
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

        assert _resolve_ocean_config() == ("ok", self.TOKEN_A)
        assert len(calls) == 1

        # Different length (so size changes) *and* a later mtime: either one
        # alone moves the fingerprint, both together make the test
        # independent of the filesystem's timestamp resolution.
        path.write_text(f"[defaults]\ntoken = {self.TOKEN_B}\n")
        later = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(later, later))

        assert _resolve_ocean_config() == ("ok", self.TOKEN_B)
        assert len(calls) == 2

    def test_setting_the_env_token_is_reflected_immediately(
        self, monkeypatch
    ) -> None:
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )

        assert _resolve_ocean_config() == ("ok", self.TOKEN_A)

        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN)

        assert _resolve_ocean_config() == ("ok", self.ENV_TOKEN)
        assert len(calls) == 2

    def test_removing_the_env_token_uncovers_the_config_file_token(
        self, monkeypatch
    ) -> None:
        """2026-09-11 review F-02, restated for the resolution cache.

        Ocean's merge means that with the env var set the resolved token is
        the env one; unsetting it edits no file, so the fingerprint alone
        would not move and the stale resolution would keep reporting a token
        that is gone.
        """
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_B),
            get_configfile_paths=lambda: [],
        )
        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN)

        assert _resolve_ocean_config() == ("ok", self.ENV_TOKEN)
        assert len(calls) == 1

        monkeypatch.delenv(TOKEN_ENV)

        assert _resolve_ocean_config() == ("ok", self.TOKEN_B)
        assert len(calls) == 2

    def test_rotating_the_env_token_is_reflected_immediately(
        self, monkeypatch
    ) -> None:
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )
        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN)

        assert _resolve_ocean_config() == ("ok", self.ENV_TOKEN)
        assert len(calls) == 1

        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN_ROTATED)

        assert _resolve_ocean_config() == ("ok", self.ENV_TOKEN_ROTATED)
        assert len(calls) == 2

    def test_an_unchanged_env_token_still_hits_the_cache(self, monkeypatch) -> None:
        """Adding the env token to the key must not defeat the cache."""
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )
        monkeypatch.setenv(TOKEN_ENV, self.ENV_TOKEN)

        for _ in range(5):
            assert _resolve_ocean_config() == ("ok", self.ENV_TOKEN)

        assert len(calls) == 1

    def test_switching_profile_invalidates_the_cache(self, monkeypatch) -> None:
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )

        assert _resolve_ocean_config() == ("ok", self.TOKEN_A)
        assert len(calls) == 1

        monkeypatch.setenv("DWAVE_PROFILE", "other")

        assert _resolve_ocean_config() == ("ok", self.TOKEN_A)
        assert len(calls) == 2

    def test_nothing_is_cached_without_a_fingerprint(self, monkeypatch) -> None:
        """No ``get_configfile_paths`` → no fingerprint → the old live parse."""
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch, self._counting_load_config(calls, self.TOKEN_A)
        )

        for _ in range(3):
            assert _resolve_ocean_config() == ("ok", self.TOKEN_A)

        assert len(calls) == 3
        assert ocean_module._CONFIG_RESOLUTION_CACHE == {}

    def test_the_cache_stays_a_single_slot(self, monkeypatch) -> None:
        """Env tokens come and go; the cache never grows past one entry."""
        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )

        for token in (self.ENV_TOKEN, self.ENV_TOKEN_ROTATED):
            monkeypatch.setenv(TOKEN_ENV, token)
            assert _resolve_ocean_config() == ("ok", token)
            assert len(ocean_module._CONFIG_RESOLUTION_CACHE) == 1

        monkeypatch.delenv(TOKEN_ENV)

        assert _resolve_ocean_config() == ("ok", self.TOKEN_A)
        assert len(ocean_module._CONFIG_RESOLUTION_CACHE) == 1
        assert len(calls) == 3

    def test_a_config_that_breaks_becomes_invalid_immediately(
        self, monkeypatch, tmp_path
    ) -> None:
        """End to end through the public status helper."""
        path = tmp_path / "dwave.conf"
        path.write_text(f"[defaults]\ntoken = {self.TOKEN_A}\n")

        def load_config():
            parser = configparser.ConfigParser()
            # Raises MissingSectionHeaderError once the file is corrupt,
            # which is exactly what Ocean's own loader does.
            parser.read_string(path.read_text())
            return {"token": parser["defaults"]["token"]}

        _install_fake_ocean_config(
            monkeypatch, load_config, get_configfile_paths=lambda: [str(path)]
        )

        assert ocean_config_status() == "ok"

        path.write_text("this is not an ini file at all\n")
        later = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(later, later))

        assert ocean_config_status() == "invalid"

    def test_one_capabilities_query_parses_the_config_once(self, monkeypatch) -> None:
        """The reason the cache exists: three D-Wave backends, one parse."""
        from annealbridge.interfaces.capabilities import build_capabilities
        from annealbridge.orchestration import ExecutionPolicy
        from annealbridge.solvers.registry import SolverRegistry

        calls: list[int] = []
        _install_fake_ocean_config(
            monkeypatch,
            self._counting_load_config(calls, self.TOKEN_A),
            get_configfile_paths=lambda: [],
        )
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)

        view = build_capabilities(SolverRegistry.default(), ExecutionPolicy())

        assert len(calls) == 1
        dwave_backends = {
            backend.name: backend.available
            for backend in view.backends
            if backend.name in {"dwave_qpu", "leap_hybrid_bqm", "leap_hybrid_cqm"}
        }
        assert dwave_backends == {
            "dwave_qpu": True,
            "leap_hybrid_bqm": True,
            "leap_hybrid_cqm": True,
        }
