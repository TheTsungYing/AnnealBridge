"""Unit tests for sampleset-info sanitization and redaction (spec §17, §19).

Vendor-neutral by construction (2026-09-09 review F-10): ``solvers.metadata``
knows no environment variable, header or token shape of its own, so these
tests declare a *fake* vendor through the public
:func:`~annealbridge.solvers.metadata.declare_credentials` /
:func:`~annealbridge.solvers.metadata.register_secret_source` seams. That a
real backend declares the right thing is pinned in
``tests/unit/test_backend_credentials.py``; the Ocean config-file token
source lives in ``tests/unit/test_ocean_config.py``.

All tokens in this file are synthetic test values, never real credentials.
"""

import pytest

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CredentialDeclaration, SolverExecutionMetadata
import annealbridge.solvers.metadata as metadata_module
from annealbridge.solvers.metadata import (
    credential_env_vars,
    declare_credentials,
    guarded_call,
    redact,
    register_secret_source,
    sanitize_sampleset_info,
)
from annealbridge.solvers.ocean import call_ocean

# A vendor nobody ships: the redaction only knows it because the fixture
# below declares it, which is exactly the property under test.
FAKE_VENDOR = "fake_vendor"
FAKE_VENDOR_ENV = "FAKE_VENDOR_API_KEY"
FAKE_VENDOR_HEADER = "X-Fake-Vendor-Key"
FAKE_VENDOR_PATTERN = r"FK-[A-Za-z0-9]{20,}"

# Matches ``FAKE_VENDOR_PATTERN``: masked on shape alone, even when it never
# was in this process's environment.
FAKE_SHAPED_TOKEN = "FK-" + "a" * 24
# Deliberately matches no pattern, so only the live env-var lookup (or a
# secret source) can mask it.
FAKE_ENV_VALUE = "fake-secret-key-987654"

UNDECLARED_ENV = "SOME_OTHER_API_KEY"


@pytest.fixture
def fake_vendor(monkeypatch):
    """Replace the process-level tables with one fake vendor declaration.

    ``monkeypatch.setattr`` swaps the module's dicts for fresh ones, so the
    declarations every previously built registry contributed (they are a
    process-level union by design) cannot influence these assertions, and
    the real tables are restored afterwards.
    """
    monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
    monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})
    declare_credentials(
        FAKE_VENDOR,
        CredentialDeclaration(
            env_vars=[FAKE_VENDOR_ENV],
            header_names=[FAKE_VENDOR_HEADER],
            value_patterns=[FAKE_VENDOR_PATTERN],
        ),
    )


@pytest.fixture
def no_declarations(monkeypatch):
    """Empty both tables: nothing but the protocol-level fallbacks is left."""
    monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
    monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})


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
    """§19 masking, driven entirely by what a backend declared (F-10)."""

    def test_declared_env_var_value_is_masked(self, fake_vendor, monkeypatch) -> None:
        monkeypatch.setenv(FAKE_VENDOR_ENV, FAKE_ENV_VALUE)

        redacted = redact(f"POST /solve failed with key {FAKE_ENV_VALUE}")

        assert FAKE_ENV_VALUE not in redacted
        assert "***" in redacted

    def test_declared_env_var_is_read_live_on_every_call(
        self, fake_vendor, monkeypatch
    ) -> None:
        text = f"error: key is {FAKE_ENV_VALUE}"

        # Not set yet, and the value matches no pattern, so it survives.
        assert redact(text) == text

        monkeypatch.setenv(FAKE_VENDOR_ENV, FAKE_ENV_VALUE)

        redacted = redact(text)
        assert FAKE_ENV_VALUE not in redacted
        assert "***" in redacted

    def test_undeclared_env_var_is_not_masked(self, fake_vendor, monkeypatch) -> None:
        """Only *declared* variables are candidates; the module reads no others."""
        monkeypatch.setenv(UNDECLARED_ENV, FAKE_ENV_VALUE)
        text = f"error: key is {FAKE_ENV_VALUE}"

        assert redact(text) == text

    def test_declared_value_pattern_is_masked(self, fake_vendor) -> None:
        text = f"auth failed for {FAKE_SHAPED_TOKEN} on endpoint"

        redacted = redact(text)

        assert FAKE_SHAPED_TOKEN not in redacted
        assert "***" in redacted

    def test_short_prefix_is_not_masked(self, fake_vendor) -> None:
        """The declared shape needs 20+ characters; prose keeping the prefix stays."""
        text = "see FK-notes for details"

        assert redact(text) == text

    def test_declared_header_line_is_masked(self, fake_vendor) -> None:
        redacted = redact(f"{FAKE_VENDOR_HEADER}: abc123secret\nHost: example.com")

        assert "abc123secret" not in redacted
        assert f"{FAKE_VENDOR_HEADER}: ***" in redacted
        assert "Host: example.com" in redacted

    def test_declared_header_in_json_is_masked(self, fake_vendor) -> None:
        """Exceptions from an HTTP client print the header dict as JSON."""
        redacted = redact(f'{{"{FAKE_VENDOR_HEADER}": "abc"}}')

        assert redacted == f'{{"{FAKE_VENDOR_HEADER}": "***"}}'

    def test_declared_header_in_json_with_extra_spacing_is_masked(
        self, fake_vendor
    ) -> None:
        redacted = redact(f'{{"{FAKE_VENDOR_HEADER}":   "abc"}}')

        assert "abc" not in redacted
        assert redacted == f'{{"{FAKE_VENDOR_HEADER}": "***"}}'

    def test_undeclared_header_is_not_masked(self, fake_vendor) -> None:
        text = "X-Other-Key: abc123secret"

        assert redact(text) == text

    def test_registered_secret_source_value_is_masked(self, fake_vendor) -> None:
        """A credential that is not an env var at all (e.g. a config file)."""
        secret = "config-file-secret-value"
        register_secret_source("fake_config", lambda: secret)

        redacted = redact(f"request failed with credential {secret}")

        assert secret not in redacted
        assert "***" in redacted

    def test_secret_source_is_read_live_on_every_call(self, fake_vendor) -> None:
        current: list[str | None] = [None]
        register_secret_source("fake_config", lambda: current[0])
        text = "credential is rotating-secret-value"

        assert redact(text) == text

        current[0] = "rotating-secret-value"

        assert redact(text) == "credential is ***"

    def test_failing_secret_source_does_not_break_redaction(self, fake_vendor) -> None:
        def broken() -> str:
            raise RuntimeError("unreadable config")

        register_secret_source("broken", broken)

        redacted = redact(f"retrying with {FAKE_SHAPED_TOKEN}")

        assert FAKE_SHAPED_TOKEN not in redacted
        assert "***" in redacted

    def test_plain_text_is_untouched(self, fake_vendor) -> None:
        text = "embedding failed: problem too dense for topology"

        assert redact(text) == text

    # --- the two protocol-level fallbacks the module owns itself ----------

    def test_token_query_param_is_masked_without_any_declaration(
        self, no_declarations
    ) -> None:
        redacted = redact("GET /solve?token=abc123secret&region=eu")

        assert "abc123secret" not in redacted
        assert "token=***" in redacted
        assert "region=eu" in redacted

    def test_authorization_header_is_masked_without_any_declaration(
        self, no_declarations
    ) -> None:
        redacted = redact("Authorization: Bearer abc.def.ghi\nHost: example.com")

        assert "abc.def.ghi" not in redacted
        assert "Authorization: ***" in redacted
        assert "Host: example.com" in redacted

    def test_nothing_vendor_shaped_is_masked_without_any_declaration(
        self, no_declarations
    ) -> None:
        """Proof the fixture really empties the tables: the fake shape survives."""
        text = f"auth failed for {FAKE_SHAPED_TOKEN}"

        assert redact(text) == text


class TestDeclarationTable:
    def test_redeclaring_a_backend_replaces_instead_of_accumulating(
        self, no_declarations
    ) -> None:
        declare_credentials(FAKE_VENDOR, CredentialDeclaration(env_vars=["FIRST_KEY"]))
        declare_credentials(FAKE_VENDOR, CredentialDeclaration(env_vars=["SECOND_KEY"]))

        assert credential_env_vars() == ["SECOND_KEY"]

    def test_an_empty_declaration_clears_a_stale_one(self, no_declarations) -> None:
        declare_credentials(FAKE_VENDOR, CredentialDeclaration(env_vars=["FIRST_KEY"]))
        declare_credentials(FAKE_VENDOR, CredentialDeclaration())

        assert credential_env_vars() == []

    def test_env_vars_are_deduplicated_and_keep_declaration_order(
        self, no_declarations
    ) -> None:
        declare_credentials("a", CredentialDeclaration(env_vars=["ALPHA", "SHARED"]))
        declare_credentials("b", CredentialDeclaration(env_vars=["SHARED", "BETA"]))

        assert credential_env_vars() == ["ALPHA", "SHARED", "BETA"]

    def test_different_backends_are_masked_together(
        self, no_declarations, monkeypatch
    ) -> None:
        declare_credentials("a", CredentialDeclaration(env_vars=["ALPHA_KEY"]))
        declare_credentials("b", CredentialDeclaration(env_vars=["BETA_KEY"]))
        monkeypatch.setenv("ALPHA_KEY", "alpha-secret")
        monkeypatch.setenv("BETA_KEY", "beta-secret")

        assert redact("a=alpha-secret b=beta-secret") == "a=*** b=***"


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

    def test_message_is_redacted(self, fake_vendor, monkeypatch) -> None:
        """The wrapper's message goes through the same declaration-driven mask."""
        monkeypatch.setenv(FAKE_VENDOR_ENV, FAKE_ENV_VALUE)

        def boom():
            raise RuntimeError(f"auth failed for {FAKE_ENV_VALUE}")

        with pytest.raises(SolverExecutionError) as excinfo:
            guarded_call("solve", lambda exc: "REMOTE_AUTH_FAILED", boom)

        message = str(excinfo.value)
        assert FAKE_ENV_VALUE not in message
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
