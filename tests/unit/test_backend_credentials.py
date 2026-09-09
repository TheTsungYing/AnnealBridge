"""Every backend declares its own credential material (2026-09-09 review F-10).

``solvers/metadata.py`` names no vendor: what gets masked comes from each
backend's ``SolverCapabilities.credentials``, which ``SolverRegistry``
hands to ``metadata.declare_credentials`` at registration time. These tests
own the "the real backends declare the right thing" half of that contract —
``test_metadata.py`` owns the vendor-neutral mechanism, and
``test_ocean_config.py`` the Ocean config-file token.

No network I/O: only ``capabilities`` and constructors are touched, and the
D-Wave backends lazy-import ``dwave.system`` inside their sampler factories
only. All tokens are synthetic test values, never real credentials.
"""

import pytest

from annealbridge.models import CredentialDeclaration
import annealbridge.solvers.fujitsu_da as fujitsu_da
import annealbridge.solvers.metadata as metadata_module
from annealbridge.solvers import SolverRegistry
from annealbridge.solvers.dwave_qpu import DWaveQPUBackend
from annealbridge.solvers.fujitsu_da import FujitsuDABackend
from annealbridge.solvers.leap_hybrid_bqm import LeapHybridBQMBackend
from annealbridge.solvers.leap_hybrid_cqm import LeapHybridCQMBackend
from annealbridge.solvers.metadata import credential_env_vars, redact
from annealbridge.solvers.ocean import OCEAN_CREDENTIALS, TOKEN_ENV

DWAVE_BACKENDS = [DWaveQPUBackend, LeapHybridBQMBackend, LeapHybridCQMBackend]

FAKE_ENV_TOKEN = "DEV-FAKE-TOKEN-1234567890abcdefghij"
# Matches the declared ``DEV-[A-Za-z0-9]{20,}`` shape without ever having
# been in this process's environment.
FAKE_SHAPED_TOKEN = "DEV-" + "a" * 24
FAKE_FUJITSU_KEY = "fj-secret-key-987654"


class TestFujitsuDeclaration:
    def test_env_var_is_the_backend_constant(self) -> None:
        credentials = FujitsuDABackend().capabilities.credentials

        assert credentials.env_vars == [fujitsu_da.API_KEY_ENV]
        assert fujitsu_da.API_KEY_ENV == "FUJITSU_DA_API_KEY"

    def test_both_auth_headers_are_declared(self) -> None:
        headers = FujitsuDABackend().capabilities.credentials.header_names

        assert "X-Api-Key" in headers
        assert "X-Access-Token" in headers


class TestDWaveDeclaration:
    @pytest.mark.parametrize("backend_class", DWAVE_BACKENDS)
    def test_all_three_share_the_ocean_declaration(self, backend_class) -> None:
        credentials = backend_class().capabilities.credentials

        assert credentials == OCEAN_CREDENTIALS
        assert credentials.env_vars == [TOKEN_ENV]
        assert any("DEV-" in pattern for pattern in credentials.value_patterns)

    @pytest.mark.parametrize("backend_class", DWAVE_BACKENDS)
    def test_constructing_one_registers_the_config_token_source(
        self, backend_class, monkeypatch
    ) -> None:
        monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})

        backend_class()

        assert "ocean_config" in metadata_module._SECRET_SOURCES


class TestDefaultRegistryDeclarations:
    def test_every_remote_backend_declares_at_least_one_env_var(self) -> None:
        registry = SolverRegistry.default()

        remote = [
            registry.get(name)
            for name in registry.names()
            if registry.get(name).capabilities.remote
        ]
        assert remote, "the default registry must contain remote backends"
        for backend in remote:
            assert backend.capabilities.credentials.env_vars, (
                f"remote backend {backend.name} declares no credential env var"
            )

    def test_every_local_backend_declares_nothing(self) -> None:
        registry = SolverRegistry.default()

        local = [
            registry.get(name)
            for name in registry.names()
            if not registry.get(name).capabilities.remote
        ]
        assert local, "the default registry must contain local backends"
        for backend in local:
            assert backend.capabilities.credentials.empty

    def test_building_the_registry_declares_both_vendor_env_vars(self) -> None:
        SolverRegistry.default()

        names = credential_env_vars()
        assert TOKEN_ENV in names
        assert fujitsu_da.API_KEY_ENV in names


class TestRedactionAfterTheDefaultRegistryIsBuilt:
    """End to end: registration is the only wiring the redaction needs."""

    @pytest.fixture(autouse=True)
    def _default_registry(self, monkeypatch):
        monkeypatch.setattr(metadata_module, "_DECLARATIONS", {})
        monkeypatch.setattr(metadata_module, "_SECRET_SOURCES", {})
        SolverRegistry.default()

    def test_dwave_env_token_is_masked(self, monkeypatch) -> None:
        monkeypatch.setenv(TOKEN_ENV, FAKE_ENV_TOKEN)

        redacted = redact(f"auth failed for {FAKE_ENV_TOKEN}")

        assert FAKE_ENV_TOKEN not in redacted
        assert "***" in redacted

    def test_dwave_shaped_token_is_masked_without_the_env_var(self) -> None:
        redacted = redact(f"the cloud echoed {FAKE_SHAPED_TOKEN} back")

        assert FAKE_SHAPED_TOKEN not in redacted
        assert "***" in redacted

    def test_fujitsu_api_key_header_is_masked(self) -> None:
        redacted = redact("X-Api-Key: leaked\nHost: example.com")

        assert "leaked" not in redacted
        assert "X-Api-Key: ***" in redacted
        assert "Host: example.com" in redacted

    def test_fujitsu_env_key_is_masked(self, monkeypatch) -> None:
        monkeypatch.setenv(fujitsu_da.API_KEY_ENV, FAKE_FUJITSU_KEY)

        redacted = redact(f"POST /da/qubo failed with key {FAKE_FUJITSU_KEY}")

        assert FAKE_FUJITSU_KEY not in redacted
        assert "***" in redacted

    def test_fujitsu_env_key_is_not_masked_without_the_variable(self) -> None:
        """It matches no pattern, so only the live env lookup can mask it."""
        text = f"POST /da/qubo failed with key {FAKE_FUJITSU_KEY}"

        assert redact(text) == text


class TestCredentialDeclarationModel:
    @pytest.mark.parametrize("field", ["env_vars", "header_names"])
    @pytest.mark.parametrize("name", ["", "   "])
    def test_blank_names_are_rejected(self, field, name) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CredentialDeclaration(**{field: [name]})

    def test_an_uncompilable_pattern_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid credential value pattern"):
            CredentialDeclaration(value_patterns=["("])

    def test_the_default_declaration_is_empty(self) -> None:
        declaration = CredentialDeclaration()

        assert declaration.empty is True
        assert declaration.env_vars == []
        assert declaration.header_names == []
        assert declaration.value_patterns == []

    def test_any_declared_field_makes_it_non_empty(self) -> None:
        assert CredentialDeclaration(env_vars=["X"]).empty is False
        assert CredentialDeclaration(header_names=["X"]).empty is False
        assert CredentialDeclaration(value_patterns=["X"]).empty is False
