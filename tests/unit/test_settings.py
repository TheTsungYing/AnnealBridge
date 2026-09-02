"""Unit tests for the environment-driven server settings (Phase 2 spec §9)."""

import os

import pytest

from pydantic import ValidationError

from annealbridge.config import ServerSettings, SettingsError, load_settings
from annealbridge.orchestration.policy import ExecutionPolicy

ENV_PREFIX = "ANNEALBRIDGE_"

# Every field ServerSettings reads, with its env-var suffix.
ENV_SUFFIXES = [
    "ALLOW_REMOTE",
    "ALLOW_REMOTE_RETRIES",
    "EXACT_MAX_VARIABLES",
    "MAX_QPU_READS",
    "MAX_QPU_ANNEALING_TIME_US",
    "MAX_REMOTE_TIME_SECONDS",
    "MAX_CONCURRENT_SOLVES",
    "HTTP_HOST",
    "HTTP_PORT",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every ``ANNEALBRIDGE_*`` variable for the duration of a test.

    Settings lookups are case-insensitive, so anything with the prefix is
    cleared, not just the canonical upper-case names. monkeypatch restores
    the real environment afterwards.
    """
    for name in list(os.environ):
        if name.upper().startswith(ENV_PREFIX):
            monkeypatch.delenv(name, raising=False)
    for suffix in ENV_SUFFIXES:
        monkeypatch.delenv(ENV_PREFIX + suffix, raising=False)
    return monkeypatch


class TestServerSettingsDefaults:
    def test_defaults_match_execution_policy(self, clean_env):
        settings = ServerSettings()

        assert settings.allow_remote is False
        assert settings.allow_remote_retries is False
        assert settings.exact_max_variables == 24
        assert settings.max_qpu_reads == 1000
        assert settings.max_qpu_annealing_time_us == 2000.0
        assert settings.max_remote_time_seconds == 300
        assert settings.max_concurrent_solves == 4

    def test_http_defaults_are_loopback_8000(self, clean_env):
        settings = ServerSettings()

        assert settings.http_host == "127.0.0.1"
        assert settings.http_port == 8000


class TestServerSettingsFromEnvironment:
    def test_env_vars_override_every_field(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE", "true")
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE_RETRIES", "true")
        clean_env.setenv("ANNEALBRIDGE_EXACT_MAX_VARIABLES", "8")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_READS", "50")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US", "123.5")
        clean_env.setenv("ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS", "30")
        clean_env.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "2")
        clean_env.setenv("ANNEALBRIDGE_HTTP_HOST", "0.0.0.0")
        clean_env.setenv("ANNEALBRIDGE_HTTP_PORT", "9000")

        settings = ServerSettings()

        assert settings.allow_remote is True
        assert settings.allow_remote_retries is True
        assert settings.exact_max_variables == 8
        assert settings.max_qpu_reads == 50
        assert settings.max_qpu_annealing_time_us == 123.5
        assert settings.max_remote_time_seconds == 30
        assert settings.max_concurrent_solves == 2
        assert settings.http_host == "0.0.0.0"
        assert settings.http_port == 9000

    def test_env_values_are_coerced_to_field_types(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE", "true")
        clean_env.setenv("ANNEALBRIDGE_EXACT_MAX_VARIABLES", "8")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US", "123.5")
        clean_env.setenv("ANNEALBRIDGE_HTTP_PORT", "9000")

        settings = ServerSettings()

        assert isinstance(settings.allow_remote, bool)
        assert isinstance(settings.exact_max_variables, int)
        assert isinstance(settings.max_qpu_annealing_time_us, float)
        assert isinstance(settings.http_port, int)

    def test_false_like_values_disable_remote(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE", "false")
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE_RETRIES", "0")

        settings = ServerSettings()

        assert settings.allow_remote is False
        assert settings.allow_remote_retries is False

    def test_unprefixed_variable_is_ignored(self, clean_env):
        clean_env.setenv("EXACT_MAX_VARIABLES", "3")

        assert ServerSettings().exact_max_variables == 24


class TestToPolicy:
    def test_returns_execution_policy_with_defaults(self, clean_env):
        policy = ServerSettings().to_policy()

        assert isinstance(policy, ExecutionPolicy)
        assert policy.allow_remote is False
        assert policy.allow_remote_retries is False
        assert policy.exact_max_variables == 24
        assert policy.max_qpu_reads == 1000
        assert policy.max_qpu_annealing_time_us == 2000.0
        assert policy.max_remote_time_seconds == 300
        assert policy.max_concurrent_solves == 4

    def test_carries_environment_overrides_into_the_policy(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE", "true")
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE_RETRIES", "true")
        clean_env.setenv("ANNEALBRIDGE_EXACT_MAX_VARIABLES", "8")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_READS", "50")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US", "123.5")
        clean_env.setenv("ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS", "30")
        clean_env.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "2")

        settings = ServerSettings()
        policy = settings.to_policy()

        assert policy.allow_remote is settings.allow_remote
        assert policy.allow_remote_retries is settings.allow_remote_retries
        assert policy.exact_max_variables == settings.exact_max_variables
        assert policy.max_qpu_reads == settings.max_qpu_reads
        assert policy.max_qpu_annealing_time_us == settings.max_qpu_annealing_time_us
        assert policy.max_remote_time_seconds == settings.max_remote_time_seconds
        assert policy.max_concurrent_solves == settings.max_concurrent_solves

    def test_enabled_backends_stays_none(self, clean_env):
        assert ServerSettings().to_policy().enabled_backends is None

    def test_http_fields_do_not_leak_into_the_policy(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_HTTP_HOST", "0.0.0.0")
        clean_env.setenv("ANNEALBRIDGE_HTTP_PORT", "9000")

        policy = ServerSettings().to_policy()

        assert "http_host" not in ExecutionPolicy.model_fields
        assert "http_port" not in ExecutionPolicy.model_fields
        assert not hasattr(policy, "http_host")
        assert not hasattr(policy, "http_port")


class TestLimitBounds:
    """The env-driven limits carry the same lower bounds as ExecutionPolicy,
    so a bad environment fails at startup rather than at the first solve."""

    LIMIT_SUFFIXES = [
        "EXACT_MAX_VARIABLES",
        "MAX_QPU_READS",
        "MAX_QPU_ANNEALING_TIME_US",
        "MAX_REMOTE_TIME_SECONDS",
        "MAX_CONCURRENT_SOLVES",
    ]

    @pytest.mark.parametrize("suffix", LIMIT_SUFFIXES)
    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_limit_is_rejected(self, clean_env, suffix, value):
        clean_env.setenv(ENV_PREFIX + suffix, value)

        with pytest.raises(ValidationError) as exc_info:
            ServerSettings()

        assert [error["loc"] for error in exc_info.value.errors()] == [
            (suffix.lower(),)
        ]

    @pytest.mark.parametrize("suffix", LIMIT_SUFFIXES)
    def test_smallest_positive_limit_is_accepted(self, clean_env, suffix):
        clean_env.setenv(ENV_PREFIX + suffix, "1")

        assert getattr(ServerSettings(), suffix.lower()) == 1


class TestLoadSettings:
    def test_valid_environment_returns_settings(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "2")

        settings = load_settings()

        assert isinstance(settings, ServerSettings)
        assert settings.max_concurrent_solves == 2

    def test_invalid_value_raises_settings_error_naming_the_variable(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "-1")

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        message = str(exc_info.value)
        assert "Invalid server settings" in message
        assert "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES" in message
        assert "greater than or equal to 1" in message
        assert "-1" in message

    def test_every_invalid_field_is_listed(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "0")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_READS", "0")
        clean_env.setenv("ANNEALBRIDGE_HTTP_PORT", "not-a-port")

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        message = str(exc_info.value)
        assert "(3 error(s))" in message
        for variable in (
            "ANNEALBRIDGE_MAX_CONCURRENT_SOLVES",
            "ANNEALBRIDGE_MAX_QPU_READS",
            "ANNEALBRIDGE_HTTP_PORT",
        ):
            assert variable in message

    def test_settings_error_is_a_value_error_without_a_pydantic_chain(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_EXACT_MAX_VARIABLES", "-5")

        with pytest.raises(ValueError) as exc_info:
            load_settings()

        assert isinstance(exc_info.value, SettingsError)
        assert exc_info.value.__cause__ is None
