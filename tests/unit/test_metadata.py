"""Unit tests for sampleset-info sanitization and redaction (spec §17, §19).

All tokens in this file are synthetic test values, never real credentials.
"""

import sys
import types

import pytest

from annealbridge.models import SolverExecutionMetadata
from annealbridge.solvers.metadata import (
    ocean_config_status,
    ocean_token_configured,
    redact,
    sanitize_sampleset_info,
)

FAKE_TOKEN = "DEV-" + "a" * 24
# Deliberately does not match the ``DEV-[A-Za-z0-9]{20,}`` pattern, so only
# the live env-var lookup can mask it.
FAKE_ENV_TOKEN = "DEV-FAKE-TOKEN-1234567890abcdefghij"
ENV_VAR = "DWAVE_API_TOKEN"


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


class TestSanitizeSamplesetInfo:
    def test_only_whitelist_keys_survive_and_are_floats(self) -> None:
        info = {
            "timing": {
                "qpu_access_time": 12345,
                "qpu_sampling_time": 100.5,
                "qpu_anneal_time_per_sample": 20,
                "post_processing_overhead_time": 999,  # not whitelisted
                "weird": object(),
            },
            "run_time": 5_000_000,
            "charge_time": 4_000_000,
            "problem_id": "abc-123",  # not a timing key
            "embedding_context": {"embedding": {"a": [0, 1]}},  # nested object
            "blob": b"\x00\x01",  # bytes
            "answer_mode": True,
        }

        metadata = sanitize_sampleset_info(info, backend="dwave_qpu")

        assert metadata.backend == "dwave_qpu"
        assert metadata.remote is True
        assert metadata.timing_us == {
            "qpu_access_time": 12345.0,
            "qpu_sampling_time": 100.5,
            "qpu_anneal_time_per_sample": 20.0,
            "run_time": 5_000_000.0,
            "charge_time": 4_000_000.0,
        }
        assert all(type(v) is float for v in metadata.timing_us.values())

    def test_non_numeric_whitelist_values_are_dropped(self) -> None:
        info = {
            "timing": {"qpu_access_time": "12345", "qpu_programming_time": None},
            "run_time": b"\x01",
            "charge_time": {"nested": 1},
            "qpu_sampling_time": True,  # bool is not a timing number
        }

        metadata = sanitize_sampleset_info(info, backend="dwave_qpu")

        assert metadata.timing_us == {}

    def test_result_is_json_serializable_and_not_raw_info(self) -> None:
        info = {"timing": {"qpu_access_time": 1}, "secret_field": object()}

        metadata = sanitize_sampleset_info(info, backend="leap_hybrid_bqm")

        assert isinstance(metadata, SolverExecutionMetadata)
        dumped = metadata.model_dump_json()
        assert "secret_field" not in dumped
        assert metadata.model_dump() != info

    def test_local_backend_can_mark_remote_false(self) -> None:
        metadata = sanitize_sampleset_info({}, backend="exact", remote=False)

        assert metadata.remote is False
        assert metadata.timing_us == {}


class TestRedact:
    def test_dev_token_form_is_masked(self) -> None:
        text = f"auth failed for {FAKE_TOKEN} on endpoint"

        redacted = redact(text)

        assert FAKE_TOKEN not in redacted
        assert "***" in redacted

    def test_token_query_param_is_masked(self) -> None:
        redacted = redact("GET /solve?token=abc123secret&region=eu")

        assert "abc123secret" not in redacted
        assert "token=***" in redacted
        assert "region=eu" in redacted

    def test_authorization_header_is_masked(self) -> None:
        redacted = redact("Authorization: Bearer abc.def.ghi\nHost: example.com")

        assert "abc.def.ghi" not in redacted
        assert "Authorization: ***" in redacted
        assert "Host: example.com" in redacted

    def test_plain_text_is_untouched(self) -> None:
        text = "embedding failed: problem too dense for topology"

        assert redact(text) == text

    def test_short_dev_prefix_is_not_masked(self) -> None:
        text = "see DEV-notes for details"

        assert redact(text) == text

    def test_no_error_when_dwave_not_installed(self, monkeypatch) -> None:
        for name in list(sys.modules):
            if name == "dwave" or name.startswith("dwave."):
                monkeypatch.delitem(sys.modules, name)

        assert redact("nothing sensitive here") == "nothing sensitive here"

    def test_ocean_config_token_is_masked(self, monkeypatch) -> None:
        token = "s3cr3t-ocean-token-value"
        _install_fake_ocean_config(monkeypatch, lambda: {"token": token})

        redacted = redact(f"request failed with credential {token}")

        assert token not in redacted
        assert "***" in redacted

    def test_load_config_failure_is_ignored_and_regexes_still_apply(
        self, monkeypatch
    ) -> None:
        def broken_load_config():
            raise ValueError("unreadable config")

        _install_fake_ocean_config(monkeypatch, broken_load_config)

        redacted = redact(f"retrying with {FAKE_TOKEN}")

        assert FAKE_TOKEN not in redacted
        assert "***" in redacted

    @pytest.mark.parametrize("config", [None, {"endpoint": "x"}, {"token": ""}])
    def test_missing_or_empty_token_in_config_is_ignored(
        self, monkeypatch, config
    ) -> None:
        _install_fake_ocean_config(monkeypatch, lambda: config)

        assert redact("plain message") == "plain message"

    def test_env_token_is_masked(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(ENV_VAR, FAKE_ENV_TOKEN)

        redacted = redact(f"error: token is {FAKE_ENV_TOKEN}")

        assert FAKE_ENV_TOKEN not in redacted
        assert "***" in redacted

    def test_env_token_is_read_live_on_every_call(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        text = f"error: token is {FAKE_ENV_TOKEN}"

        # No env var yet: the token is not token-shaped, so it survives.
        assert redact(text) == text

        monkeypatch.setenv(ENV_VAR, FAKE_ENV_TOKEN)

        redacted = redact(text)
        assert FAKE_ENV_TOKEN not in redacted
        assert "***" in redacted


class TestOceanConfigStatus:
    def test_missing_without_env_token(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)

        assert ocean_config_status() == "missing"

    def test_ok_with_env_token(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(ENV_VAR, FAKE_ENV_TOKEN)

        assert ocean_config_status() == "ok"

    def test_empty_env_token_is_missing(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(ENV_VAR, "")

        assert ocean_config_status() == "missing"


class TestOceanTokenConfigured:
    def test_false_without_env_token(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)

        assert ocean_token_configured() is False

    def test_true_with_env_token(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(ENV_VAR, FAKE_ENV_TOKEN)

        assert ocean_token_configured() is True
