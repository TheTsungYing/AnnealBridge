"""Unit tests for the execution policy defaults (Phase 2 spec §8)."""

from annealbridge.orchestration import ExecutionPolicy as ExportedExecutionPolicy
from annealbridge.orchestration.policy import ExecutionPolicy


class TestExecutionPolicyDefaults:
    def test_allow_remote_defaults_to_false(self):
        assert ExecutionPolicy().allow_remote is False

    def test_allow_remote_retries_defaults_to_false(self):
        assert ExecutionPolicy().allow_remote_retries is False

    def test_exact_max_variables_defaults_to_24(self):
        assert ExecutionPolicy().exact_max_variables == 24

    def test_max_qpu_reads_defaults_to_1000(self):
        assert ExecutionPolicy().max_qpu_reads == 1000

    def test_max_qpu_annealing_time_us_defaults_to_2000(self):
        assert ExecutionPolicy().max_qpu_annealing_time_us == 2000.0

    def test_max_remote_time_seconds_defaults_to_300(self):
        assert ExecutionPolicy().max_remote_time_seconds == 300

    def test_max_concurrent_solves_defaults_to_4(self):
        assert ExecutionPolicy().max_concurrent_solves == 4

    def test_enabled_backends_defaults_to_none(self):
        # None means "every backend in the registry" (spec §8).
        assert ExecutionPolicy().enabled_backends is None


class TestExecutionPolicyOverrides:
    def test_fields_are_overridable(self):
        policy = ExecutionPolicy(
            allow_remote=True,
            allow_remote_retries=True,
            exact_max_variables=8,
            max_qpu_reads=10,
            max_qpu_annealing_time_us=123.5,
            max_remote_time_seconds=30,
            max_concurrent_solves=1,
            enabled_backends={"exact"},
        )

        assert policy.allow_remote is True
        assert policy.allow_remote_retries is True
        assert policy.exact_max_variables == 8
        assert policy.max_qpu_reads == 10
        assert policy.max_qpu_annealing_time_us == 123.5
        assert policy.max_remote_time_seconds == 30
        assert policy.max_concurrent_solves == 1
        assert policy.enabled_backends == {"exact"}


class TestExecutionPolicyExport:
    def test_package_export_is_the_same_class(self):
        assert ExportedExecutionPolicy is ExecutionPolicy
