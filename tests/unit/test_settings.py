"""Unit tests for the environment-driven server settings (Phase 2 spec §9)."""

import logging
import os

import annotated_types
import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import SettingsError as PydanticSettingsError

from annealbridge.config import ServerSettings, SettingsError, load_settings
from annealbridge.config.settings import (
    unknown_settings_variables,
    validate_http_host,
    validate_http_port,
)
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
    # 2026-09-09 review (F-02 / F-07): the five parameter ceilings.
    "MAX_LOCAL_READS",
    "MAX_SWEEPS",
    "MAX_LOCAL_RETRIES",
    "MAX_REMOTE_RETRIES",
    "MAX_TOP_K",
    # The local samplers' worker counts: speed knobs, not policy limits.
    "SA_WORKERS",
    "TABU_WORKERS",
    # The simulated_bifurcation backend's machine knobs: where its dynamics
    # run and how large a dense matrix it will hold.
    "SB_DEVICE",
    "SB_MAX_VARIABLES",
    # 2026-09-09 review (F-18): the enabled-backends gate's env entry.
    "ENABLED_BACKENDS",
    "LIMITS",
    "HTTP_HOST",
    "HTTP_PORT",
]


@pytest.fixture(name="clean_env")
def fixture_clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
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
        clean_env.setenv("ANNEALBRIDGE_MAX_LOCAL_READS", "200")
        clean_env.setenv("ANNEALBRIDGE_MAX_SWEEPS", "300")
        clean_env.setenv("ANNEALBRIDGE_MAX_LOCAL_RETRIES", "4")
        clean_env.setenv("ANNEALBRIDGE_MAX_REMOTE_RETRIES", "1")
        clean_env.setenv("ANNEALBRIDGE_MAX_TOP_K", "50")
        clean_env.setenv("ANNEALBRIDGE_SA_WORKERS", "3")
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
        assert settings.max_local_reads == 200
        assert settings.max_sweeps == 300
        assert settings.max_local_retries == 4
        assert settings.max_remote_retries == 1
        assert settings.max_top_k == 50
        assert settings.sa_workers == 3
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
        assert policy.max_local_reads == 100000
        assert policy.max_sweeps == 100000
        assert policy.max_local_retries == 10
        assert policy.max_remote_retries == 3
        assert policy.max_top_k == 1000

    def test_carries_environment_overrides_into_the_policy(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE", "true")
        clean_env.setenv("ANNEALBRIDGE_ALLOW_REMOTE_RETRIES", "true")
        clean_env.setenv("ANNEALBRIDGE_EXACT_MAX_VARIABLES", "8")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_READS", "50")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US", "123.5")
        clean_env.setenv("ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS", "30")
        clean_env.setenv("ANNEALBRIDGE_MAX_CONCURRENT_SOLVES", "2")
        clean_env.setenv("ANNEALBRIDGE_MAX_LOCAL_READS", "200")
        clean_env.setenv("ANNEALBRIDGE_MAX_SWEEPS", "300")
        clean_env.setenv("ANNEALBRIDGE_MAX_LOCAL_RETRIES", "4")
        clean_env.setenv("ANNEALBRIDGE_MAX_REMOTE_RETRIES", "1")
        clean_env.setenv("ANNEALBRIDGE_MAX_TOP_K", "50")

        settings = ServerSettings()
        policy = settings.to_policy()

        assert policy.allow_remote is settings.allow_remote
        assert policy.allow_remote_retries is settings.allow_remote_retries
        assert policy.exact_max_variables == settings.exact_max_variables
        assert policy.max_qpu_reads == settings.max_qpu_reads
        assert policy.max_qpu_annealing_time_us == settings.max_qpu_annealing_time_us
        assert policy.max_remote_time_seconds == settings.max_remote_time_seconds
        assert policy.max_concurrent_solves == settings.max_concurrent_solves
        assert policy.max_local_reads == settings.max_local_reads
        assert policy.max_sweeps == settings.max_sweeps
        assert policy.max_local_retries == settings.max_local_retries
        assert policy.max_remote_retries == settings.max_remote_retries
        assert policy.max_top_k == settings.max_top_k

    def test_enabled_backends_defaults_to_none(self, clean_env):
        assert ServerSettings().to_policy().enabled_backends is None


def _bounds(field: FieldInfo) -> list[annotated_types.BaseMetadata]:
    """The constraint metadata of ``field`` (``Ge``, ``Gt``, ``Le``, ``Lt``,
    ``MultipleOf``, ...), in declaration order.

    Every ``annotated_types`` constraint derives from ``BaseMetadata``, so a
    bound added later (``le=``, ``multiple_of=``) is compared without editing
    this helper; markers that are not constraints, such as pydantic-settings'
    ``NoDecode`` on ``enabled_backends``, are left out.
    """
    return [item for item in field.metadata if isinstance(item, annotated_types.BaseMetadata)]


class TestPolicyFieldsMirrorExecutionPolicy:
    """2026-09-15 consolidation: the settings cannot drift from the policy
    they build.

    ``ServerSettings`` repeats every ``ExecutionPolicy`` field so that a bad
    environment is refused at startup, and ``to_policy`` copies them across
    by name. Nothing but these tests keeps the two declarations in step: a
    field, type, default or bound changed on one side only would either be
    dropped on the way into the policy or be validated differently at the
    environment boundary than inside the service.
    """

    SETTINGS_ONLY_FIELDS = {
        "sa_workers",
        "tabu_workers",
        "sb_device",
        "sb_max_variables",
        "http_host",
        "http_port",
    }

    @pytest.mark.parametrize("name", sorted(ExecutionPolicy.model_fields))
    def test_policy_field_is_declared_identically_in_settings(self, name):
        assert name in ServerSettings.model_fields, (
            f"ExecutionPolicy.{name} has no ServerSettings field of the same name"
        )
        policy_field = ExecutionPolicy.model_fields[name]
        settings_field = ServerSettings.model_fields[name]

        assert settings_field.annotation == policy_field.annotation
        assert settings_field.default == policy_field.default
        assert (settings_field.default_factory is None) == (
            policy_field.default_factory is None
        )
        if policy_field.default_factory is not None:
            assert settings_field.default_factory() == policy_field.default_factory()
        assert _bounds(settings_field) == _bounds(policy_field)

    def test_settings_only_fields_are_exactly_the_known_ones(self):
        extra = set(ServerSettings.model_fields) - set(ExecutionPolicy.model_fields)

        assert extra == self.SETTINGS_ONLY_FIELDS, (
            "ServerSettings fields missing from ExecutionPolicy changed. A new "
            "settings-only field (one that configures the composition root, "
            "not the policy) must be added to SETTINGS_ONLY_FIELDS; a new "
            "policy field must be added to ExecutionPolicy as well."
        )


class TestEnabledBackends:
    """``ANNEALBRIDGE_ENABLED_BACKENDS`` (2026-09-09 review F-18).

    Comma-separated registry names rather than the JSON list
    pydantic-settings would expect for a set; unset or empty means every
    registered backend, exactly like a policy built without the field.
    """

    def test_unset_means_every_backend(self, clean_env):
        assert ServerSettings().enabled_backends is None

    def test_comma_separated_names_become_a_set(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_ENABLED_BACKENDS", "exact,simulated_annealing")

        assert ServerSettings().enabled_backends == {"exact", "simulated_annealing"}

    def test_whitespace_and_blank_entries_are_dropped(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_ENABLED_BACKENDS", " exact , ,simulated_annealing,")

        assert ServerSettings().enabled_backends == {"exact", "simulated_annealing"}

    @pytest.mark.parametrize("value", ["", "   ", ",", " , "])
    def test_empty_value_means_every_backend(self, clean_env, value):
        clean_env.setenv("ANNEALBRIDGE_ENABLED_BACKENDS", value)

        assert ServerSettings().enabled_backends is None
        assert ServerSettings().to_policy().enabled_backends is None

    def test_reaches_the_policy(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_ENABLED_BACKENDS", "exact")

        assert ServerSettings().to_policy().enabled_backends == {"exact"}

    def test_is_not_parsed_as_json(self, clean_env):
        # A JSON list would be the pydantic-settings default for a set; the
        # documented format is the comma list, so the brackets are literal.
        clean_env.setenv("ANNEALBRIDGE_ENABLED_BACKENDS", '["exact"]')

        assert ServerSettings().enabled_backends == {'["exact"]'}

    def test_http_fields_do_not_leak_into_the_policy(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_HTTP_HOST", "0.0.0.0")
        clean_env.setenv("ANNEALBRIDGE_HTTP_PORT", "9000")

        policy = ServerSettings().to_policy()

        assert "http_host" not in ExecutionPolicy.model_fields
        assert "http_port" not in ExecutionPolicy.model_fields
        assert not hasattr(policy, "http_host")
        assert not hasattr(policy, "http_port")

    @pytest.mark.parametrize("field", ["sa_workers", "tabu_workers"])
    def test_worker_count_defaults_to_auto_and_does_not_leak_into_the_policy(
        self, clean_env, field
    ):
        # A speed knob for one backend, never a limit the service enforces:
        # it reaches the registry through the composition root instead.
        assert getattr(ServerSettings(), field) is None

        clean_env.setenv(ENV_PREFIX + field.upper(), "2")
        policy = ServerSettings().to_policy()

        assert field not in ExecutionPolicy.model_fields
        assert not hasattr(policy, field)

    def test_sb_settings_default_and_do_not_leak_into_the_policy(self, clean_env):
        # Like the worker counts: one backend's machine, not the service's
        # policy, so they reach the registry through the composition root.
        settings = ServerSettings()
        assert settings.sb_device == "cpu"
        assert settings.sb_max_variables == 10_000

        clean_env.setenv("ANNEALBRIDGE_SB_DEVICE", "cuda")
        clean_env.setenv("ANNEALBRIDGE_SB_MAX_VARIABLES", "128")
        settings = ServerSettings()
        assert settings.sb_device == "cuda"
        assert settings.sb_max_variables == 128

        policy = settings.to_policy()
        for field in ("sb_device", "sb_max_variables"):
            assert field not in ExecutionPolicy.model_fields
            assert not hasattr(policy, field)

    @pytest.mark.parametrize("value", ["tpu", "gpu", "CPU", ""])
    def test_sb_device_rejects_anything_but_cpu_or_cuda(self, clean_env, value):
        clean_env.setenv("ANNEALBRIDGE_SB_DEVICE", value)

        with pytest.raises(ValidationError) as exc_info:
            ServerSettings()

        assert [error["loc"] for error in exc_info.value.errors()] == [("sb_device",)]


class TestLimitBounds:
    """The env-driven limits carry the same lower bounds as ExecutionPolicy,
    so a bad environment fails at startup rather than at the first solve."""

    LIMIT_SUFFIXES = [
        "EXACT_MAX_VARIABLES",
        "MAX_QPU_READS",
        "MAX_QPU_ANNEALING_TIME_US",
        "MAX_REMOTE_TIME_SECONDS",
        "MAX_CONCURRENT_SOLVES",
        "SA_WORKERS",
        "TABU_WORKERS",
        "SB_MAX_VARIABLES",
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


class TestHttpBindValidation:
    """2026-09-11 review (F01 / F09): the bind address and the port.

    ``ANNEALBRIDGE_HTTP_HOST=""`` used to be accepted, and an empty host means
    *every interface* to asyncio and uvicorn — on a server with no
    authentication. An out-of-range port used to survive until the socket bind
    raised ``OverflowError`` with a traceback. Both rules now live in one pure
    function each, shared by the settings and by the MCP ``--host`` / ``--port``
    arguments, so the environment and an override cannot disagree.
    """

    UNUSABLE_HOSTS = ["", " ", "   ", "\t", "127.0.0.1 ", " 127.0.0.1", "local host"]

    @pytest.mark.parametrize("value", UNUSABLE_HOSTS)
    def test_function_rejects_empty_blank_or_whitespace_bearing_host(self, value):
        with pytest.raises(ValueError, match="host"):
            validate_http_host(value)

    @pytest.mark.parametrize(
        "value", ["127.0.0.1", "0.0.0.0", "localhost", "::1", "annealbridge.internal"]
    )
    def test_function_returns_any_other_host_unchanged(self, value):
        # Not stripped, not rewritten: an operator who names a non-loopback
        # address keeps that ability, and nothing is silently corrected.
        assert validate_http_host(value) == value

    @pytest.mark.parametrize("value", [0, -1, -65535, 65536, 100000])
    def test_function_rejects_a_port_outside_1_65535(self, value):
        # 0 included: the kernel would pick an arbitrary free port, which no
        # MCP host can then be pointed at.
        with pytest.raises(ValueError, match="1 and 65535"):
            validate_http_port(value)

    @pytest.mark.parametrize("value", [1, 8000, 65535])
    def test_function_accepts_a_port_inside_the_range(self, value):
        assert validate_http_port(value) == value

    @pytest.mark.parametrize("value", UNUSABLE_HOSTS)
    def test_env_host_is_rejected_naming_the_variable(self, clean_env, value):
        clean_env.setenv("ANNEALBRIDGE_HTTP_HOST", value)

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        message = str(exc_info.value)
        assert "Invalid server settings" in message
        assert "ANNEALBRIDGE_HTTP_HOST" in message

    def test_rejected_host_is_not_echoed(self, clean_env):
        # review F-20: the variable is named, the value never repeated.
        clean_env.setenv("ANNEALBRIDGE_HTTP_HOST", "10.11.12.13 ")

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        message = str(exc_info.value)
        assert "ANNEALBRIDGE_HTTP_HOST" in message
        assert "10.11.12.13" not in message

    def test_explicit_non_loopback_host_is_still_accepted(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_HTTP_HOST", "0.0.0.0")

        assert load_settings().http_host == "0.0.0.0"

    @pytest.mark.parametrize("value", ["0", "-1", "65536"])
    def test_env_port_outside_the_range_is_rejected(self, clean_env, value):
        clean_env.setenv("ANNEALBRIDGE_HTTP_PORT", value)

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        message = str(exc_info.value)
        assert "Invalid server settings" in message
        assert "ANNEALBRIDGE_HTTP_PORT" in message
        assert value not in message

    @pytest.mark.parametrize("value", ["1", "65535"])
    def test_env_port_at_the_boundaries_is_accepted(self, clean_env, value):
        clean_env.setenv("ANNEALBRIDGE_HTTP_PORT", value)

        assert load_settings().http_port == int(value)


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
        # 2026-09-09 review (F-20): the variable is named, the value never
        # echoed — an operator who typo'd a credential into a settings
        # variable must not see it come back out on stderr or in a log.
        assert "-1" not in message

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

    def test_invalid_value_is_not_echoed(self, clean_env):
        # 2026-09-09 review (F-20): a secret pasted into the wrong variable
        # is still a rejected value; the report must name the variable only.
        secret = "DEV-" + "a" * 24
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_READS", secret)

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        message = str(exc_info.value)
        assert "ANNEALBRIDGE_MAX_QPU_READS" in message
        assert secret not in message
        assert "DEV-" not in message

    def test_settings_error_is_a_value_error_without_a_pydantic_chain(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_EXACT_MAX_VARIABLES", "-5")

        with pytest.raises(ValueError) as exc_info:
            load_settings()

        assert isinstance(exc_info.value, SettingsError)
        assert exc_info.value.__cause__ is None


class TestUnknownVariables:
    """2026-09-09 review (F-20): an ``ANNEALBRIDGE_*`` variable nobody reads.

    pydantic-settings never sees a typo'd name (its env source only looks up
    the fields it knows), so a misspelt variable is silently ignored and the
    operator keeps the default while believing the setting took effect. The
    loader scans the environment itself and warns. It does *not* refuse: a
    harmless leftover variable must not be able to kill the server. The name
    is reported, never the value — the variable may well hold a secret.
    """

    LOGGER = "annealbridge.config.settings"

    @staticmethod
    def warnings(caplog) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def test_unknown_variable_is_warned_about_without_its_value(
        self, clean_env, caplog
    ):
        clean_env.setenv("ANNEALBRIDGE_FOO", "bar-value-xyz")

        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            settings = load_settings()

        # Warned about, not rejected: the defaults still load.
        assert isinstance(settings, ServerSettings)
        warnings = self.warnings(caplog)
        assert len(warnings) == 1
        assert "ANNEALBRIDGE_FOO" in warnings[0]
        assert "bar-value-xyz" not in caplog.text

    def test_known_variables_alone_warn_about_nothing(self, clean_env, caplog):
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_READS", "5")

        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            settings = load_settings()

        assert settings.max_qpu_reads == 5
        assert self.warnings(caplog) == []

    def test_the_scan_is_a_pure_function_over_a_mapping(self):
        # The scan takes the mapping it is given, so it needs no environment
        # at all — hence no clean_env here.
        #
        # Lookups are case-insensitive, so the lower-case name counts as a
        # settings variable too — and comes back spelled as the environment
        # spells it. Anything without the prefix is none of our business.
        assert unknown_settings_variables(
            {
                "annealbridge_foo": "x",
                "ANNEALBRIDGE_MAX_QPU_READS": "5",
                "OTHER": "y",
            }
        ) == ["annealbridge_foo"]


class TestGenericLimits:
    """Phase 3a §11: ``ANNEALBRIDGE_LIMITS`` is a JSON object of custom keys."""

    def test_defaults_to_empty(self, clean_env):
        settings = ServerSettings()

        assert settings.limits == {}
        assert settings.to_policy().limits == {}

    def test_json_object_is_parsed_into_floats(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_LIMITS", '{"iterations": 100}')

        settings = load_settings()

        assert settings.limits == {"iterations": 100.0}
        assert isinstance(settings.limits["iterations"], float)

    def test_limits_reach_the_policy(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_LIMITS", '{"iterations": 100, "bits": 2048}')

        policy = load_settings().to_policy()

        assert policy.limits == {"iterations": 100.0, "bits": 2048.0}
        assert policy.limit("iterations") == 100.0
        assert policy.limit("bits") == 2048.0

    def test_non_json_value_comes_from_pydantic_settings_not_validation(self, clean_env):
        # Documents the source: pydantic-settings raises its own SettingsError
        # (a ValueError without .errors()), not pydantic's ValidationError.
        clean_env.setenv("ANNEALBRIDGE_LIMITS", "not json")

        with pytest.raises(PydanticSettingsError) as exc_info:
            ServerSettings()

        assert not isinstance(exc_info.value, ValidationError)
        assert not hasattr(exc_info.value, "errors")

    def test_non_json_value_is_a_project_settings_error(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_LIMITS", "not json")

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        message = str(exc_info.value)
        assert message.startswith("Invalid server settings: ")
        assert "limits" in message
        assert exc_info.value.__cause__ is None

    def test_compatibility_key_is_rejected_naming_the_variable(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_LIMITS", '{"reads": 5}')

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        message = str(exc_info.value)
        assert "ANNEALBRIDGE_LIMITS" in message
        assert "max_qpu_reads" in message

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_non_positive_value_is_rejected_naming_the_variable(self, clean_env, value):
        clean_env.setenv("ANNEALBRIDGE_LIMITS", '{"iterations": %s}' % value)

        with pytest.raises(SettingsError) as exc_info:
            load_settings()

        assert "ANNEALBRIDGE_LIMITS" in str(exc_info.value)

    def test_old_env_names_still_drive_the_compatibility_keys(self, clean_env):
        clean_env.setenv("ANNEALBRIDGE_EXACT_MAX_VARIABLES", "8")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_READS", "50")
        clean_env.setenv("ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US", "123.5")
        clean_env.setenv("ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS", "30")
        clean_env.setenv("ANNEALBRIDGE_LIMITS", '{"iterations": 100}')

        policy = load_settings().to_policy()

        assert policy.limit("variables") == 8
        assert policy.limit("reads") == 50
        assert policy.limit("annealing_time_us") == 123.5
        assert policy.limit("time_seconds") == 30
        assert policy.limit("iterations") == 100.0
        # The generic mapping never shadows the four Phase 2 keys.
        assert set(policy.limits) == {"iterations"}
