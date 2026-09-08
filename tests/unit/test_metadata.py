"""Unit tests for sampleset-info sanitization and redaction (spec §17, §19).

All tokens in this file are synthetic test values, never real credentials.
"""

import sys
import types

import pytest

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import SolverExecutionMetadata
from annealbridge.solvers.metadata import (
    _resolve_ocean_config,
    guarded_call,
    ocean_config_status,
    redact,
    sanitize_sampleset_info,
)
from annealbridge.solvers.ocean import call_ocean

FAKE_TOKEN = "DEV-" + "a" * 24
# Deliberately does not match the ``DEV-[A-Za-z0-9]{20,}`` pattern, so only
# the live env-var lookup can mask it.
FAKE_ENV_TOKEN = "DEV-FAKE-TOKEN-1234567890abcdefghij"
ENV_VAR = "DWAVE_API_TOKEN"

# Fujitsu Digital Annealer credential (3b spec §20.5). Deliberately matches
# no redaction pattern, so only the live env-var lookup can mask it.
FUJITSU_ENV_VAR = "FUJITSU_DA_API_KEY"
FAKE_FUJITSU_KEY = "fj-secret-key-987654"


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

    def test_fujitsu_timing_keys_are_whitelisted(self) -> None:
        """3b §21: the DA's two timing facts survive, as floats."""
        info = {
            "timing": {
                "solve_time": 5041.0,
                "total_elapsed_time": 6123.5,
                "cpu_time": 4900.0,  # not whitelisted
            }
        }

        metadata = sanitize_sampleset_info(info, backend="fujitsu_da")

        assert metadata.backend == "fujitsu_da"
        assert metadata.remote is True
        assert metadata.timing_us == {
            "solve_time": 5041.0,
            "total_elapsed_time": 6123.5,
        }
        assert all(type(v) is float for v in metadata.timing_us.values())

    def test_fujitsu_string_timing_values_are_dropped(self) -> None:
        """The vendor reports millisecond *strings*; the backend converts them.

        Sanitization never parses: a value that is still a string is
        dropped, so a backend that forgets the conversion loses the fact
        instead of publishing a wrong unit.
        """
        info = {"timing": {"solve_time": "5041", "total_elapsed_time": "6123"}}

        assert sanitize_sampleset_info(info, backend="fujitsu_da").timing_us == {}


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

    # --- 3b §20.5: the Fujitsu credential and its header forms -------------

    def test_fujitsu_env_key_is_masked(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(FUJITSU_ENV_VAR, FAKE_FUJITSU_KEY)

        redacted = redact(f"POST /da/qubo failed with key {FAKE_FUJITSU_KEY}")

        assert FAKE_FUJITSU_KEY not in redacted
        assert "***" in redacted

    def test_fujitsu_env_key_is_not_masked_without_the_variable(
        self, monkeypatch
    ) -> None:
        """It matches no pattern, so only the live env lookup can mask it."""
        _block_ocean_config_import(monkeypatch)
        text = f"POST /da/qubo failed with key {FAKE_FUJITSU_KEY}"

        assert redact(text) == text

    def test_both_vendor_env_credentials_are_masked_together(
        self, monkeypatch
    ) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(ENV_VAR, FAKE_ENV_TOKEN)
        monkeypatch.setenv(FUJITSU_ENV_VAR, FAKE_FUJITSU_KEY)

        redacted = redact(f"dwave={FAKE_ENV_TOKEN} fujitsu={FAKE_FUJITSU_KEY}")

        assert FAKE_ENV_TOKEN not in redacted
        assert FAKE_FUJITSU_KEY not in redacted
        assert redacted == "dwave=*** fujitsu=***"

    def test_api_key_header_is_masked(self) -> None:
        redacted = redact("X-Api-Key: abc123secret\nHost: example.com")

        assert "abc123secret" not in redacted
        assert "X-Api-Key: ***" in redacted
        assert "Host: example.com" in redacted

    def test_access_token_header_is_masked(self) -> None:
        redacted = redact("X-Access-Token: abc123secret\nHost: example.com")

        assert "abc123secret" not in redacted
        assert "X-Access-Token: ***" in redacted
        assert "Host: example.com" in redacted

    def test_api_key_in_json_headers_is_masked(self) -> None:
        """Requests exceptions print the header dict as JSON, not as a header."""
        redacted = redact('{"X-Api-Key": "abc"}')

        assert redacted == '{"X-Api-Key": "***"}'

    def test_api_key_in_json_headers_with_extra_spacing_is_masked(self) -> None:
        redacted = redact('{"X-Api-Key":   "abc"}')

        assert "abc" not in redacted
        assert redacted == '{"X-Api-Key": "***"}'


class TestGuardedCall:
    """3b §20.8: the vendor-neutral wrapper behind every remote backend."""

    def test_success_returns_the_value_unchanged(self) -> None:
        sentinel = object()

        assert guarded_call("solve", lambda exc: "X", lambda: sentinel) is sentinel

    def test_failure_is_wrapped_with_the_classified_code(self) -> None:
        def boom():
            raise ValueError("upstream is unhappy")

        with pytest.raises(SolverExecutionError) as excinfo:
            guarded_call("submitting job", lambda exc: "REMOTE_BUSY", boom)

        error = excinfo.value
        assert error.code == "REMOTE_BUSY"
        assert "submitting job" in str(error)
        assert "ValueError" in str(error)
        assert "upstream is unhappy" in str(error)

    def test_classify_receives_the_original_exception(self) -> None:
        seen: list[Exception] = []
        original = KeyError("k")

        def boom():
            raise original

        def classify(exc: Exception) -> str:
            seen.append(exc)
            return "REMOTE_SOLVER_ERROR"

        with pytest.raises(SolverExecutionError):
            guarded_call("solve", classify, boom)

        assert seen == [original]

    def test_original_exception_is_not_chained(self) -> None:
        """§19: the raw exception text may embed credentials, so it must not
        be reachable through ``__cause__`` / ``__context__``."""

        def boom():
            raise RuntimeError("secret in here")

        with pytest.raises(SolverExecutionError) as excinfo:
            guarded_call("solve", lambda exc: "REMOTE_SOLVER_ERROR", boom)

        assert excinfo.value.__cause__ is None
        assert excinfo.value.__context__ is None

    def test_message_is_redacted(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(ENV_VAR, FAKE_ENV_TOKEN)

        def boom():
            raise RuntimeError(f"auth failed for {FAKE_ENV_TOKEN}")

        with pytest.raises(SolverExecutionError) as excinfo:
            guarded_call("solve", lambda exc: "REMOTE_AUTH_FAILED", boom)

        message = str(excinfo.value)
        assert FAKE_ENV_TOKEN not in message
        assert "***" in message

    def test_call_ocean_still_classifies_by_class_name(self) -> None:
        """``call_ocean`` stays the Ocean-flavoured wrapper over guarded_call."""

        class SolverAuthenticationError(Exception):
            pass

        def boom():
            raise SolverAuthenticationError("bad token")

        with pytest.raises(SolverExecutionError) as excinfo:
            call_ocean(
                "creating sampler",
                {"SolverAuthenticationError": "REMOTE_AUTH_FAILED"},
                boom,
            )

        assert excinfo.value.code == "REMOTE_AUTH_FAILED"
        assert "creating sampler" in str(excinfo.value)
        assert excinfo.value.__cause__ is None


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


class TestResolveOceanConfig:
    """The single lazy touch of ``dwave.cloud.config`` behind both
    ``ocean_config_status()`` and ``redact()``."""

    def test_missing_without_any_source(self, monkeypatch) -> None:
        _block_ocean_config_import(monkeypatch)

        assert _resolve_ocean_config() == ("missing", None)

    def test_env_token_counts_as_configured_but_is_not_the_config_token(
        self, monkeypatch
    ) -> None:
        _block_ocean_config_import(monkeypatch)
        monkeypatch.setenv(ENV_VAR, FAKE_ENV_TOKEN)

        # The env var is reported through _env_token() for redaction; the
        # config-token slot only ever carries what load_config() returned.
        assert _resolve_ocean_config() == ("ok", None)

    def test_config_token_is_returned_with_ok(self, monkeypatch) -> None:
        _install_fake_ocean_config(monkeypatch, lambda: {"token": FAKE_TOKEN})

        assert _resolve_ocean_config() == ("ok", FAKE_TOKEN)

    def test_unparseable_config_is_invalid_even_with_env_token(
        self, monkeypatch
    ) -> None:
        def broken_load_config():
            raise RuntimeError("bad dwave.conf")

        _install_fake_ocean_config(monkeypatch, broken_load_config)
        monkeypatch.setenv(ENV_VAR, FAKE_ENV_TOKEN)

        assert _resolve_ocean_config() == ("invalid", None)
        assert ocean_config_status() == "invalid"

    def test_empty_config_token_is_missing(self, monkeypatch) -> None:
        _install_fake_ocean_config(monkeypatch, lambda: {"token": ""})

        assert _resolve_ocean_config() == ("missing", None)
