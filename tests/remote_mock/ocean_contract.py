"""Contract tests shared by the three Ocean (D-Wave) backend mock suites.

``DWaveQPUBackend``, ``LeapHybridBQMBackend`` and ``LeapHybridCQMBackend``
all go through the same shared code in ``annealbridge.solvers.ocean`` and
``annealbridge.solvers.metadata``: the ``LazySampler`` cache, the
``call_ocean`` / ``guarded_call`` classification and redaction (the sampler
construction table included), the ``dwave_availability`` check and the
spec §17 timing whitelist. The tests of that shared behaviour used to be
repeated, assertion for assertion, in each backend's mock test file; they
live here once.

Every class below is a mixin whose name does not start with ``Test`` (and
this module is not named ``test_*``), so pytest never collects it on its
own. A backend's test module defines one :class:`OceanBackendCase`, keeps
its original ``TestXxx`` class, inherits the matching contract and sets
``case`` to that backend; the node ids stay what they were. Tests that are
specific to one backend stay in that backend's own file.

Never talks to real D-Wave: every test injects a fake sampler from
``tests.remote_mock.conftest`` through the backend's ``sampler_factory``
seam.
"""

import traceback
from dataclasses import dataclass
from typing import Callable, ClassVar

import pytest

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers import AvailabilityStatus
import annealbridge.solvers.ocean as ocean_module
from annealbridge.solvers.ocean import (
    REASON_CONFIG_INVALID,
    REASON_CREDENTIALS_MISSING,
    REASON_NOT_INSTALLED,
)
from tests.remote_mock.conftest import (
    FAKE_TOKEN,
    ConfigFileError,
    CountingFactory,
    RequestTimeout,
    SolverAuthenticationError,
    SolverNotFoundError,
    ValidationError,
)


@dataclass(frozen=True)
class OceanBackendCase:
    """One Ocean backend, as the contracts below drive it.

    ``make_compiled_problem`` and ``make_preferences`` take no arguments and
    build exactly what the backend's own test module passes to ``solve()``;
    ``fake_sampler_cls()`` with no arguments is that module's default fake.
    ``expected_timing_us`` is the whitelisted timing the default fake
    reports, as float microseconds.
    """

    backend_cls: type
    fake_sampler_cls: type
    backend_name: str
    make_compiled_problem: Callable[[], CompiledProblem]
    make_preferences: Callable[[], SolverPreferences]
    expected_timing_us: dict[str, float]


class MetadataSanitizationContract:
    """Spec §17: only the whitelisted timing keys leave the solver layer."""

    case: ClassVar[OceanBackendCase]

    def test_only_whitelisted_timing_keys_survive(self):
        # The default fake also reports ``unlisted_timing``, a plain number
        # the whitelist does not name (see conftest), so this fails when the
        # whitelist is bypassed, not only when the type filter is.
        fake = self.case.fake_sampler_cls()
        backend = self.case.backend_cls(sampler_factory=lambda: fake)

        result = backend.solve(
            self.case.make_compiled_problem(), self.case.make_preferences()
        )

        assert result.metadata.timing_us == self.case.expected_timing_us


class SamplerInitExceptionClassificationContract:
    """Sampler construction has its own classification table.

    The sampler constructor runs Client.from_config() / get_solver(), so a
    ``ValueError`` / ``ValidationError`` / ``SolverNotFoundError`` /
    ``ConfigFileError`` there means invalid configuration, and the message
    is redacted like any other.
    """

    case: ClassVar[OceanBackendCase]

    # Each case is a factory, so every test raises its own exception
    # instance; none is shared between the backends inheriting this.
    @pytest.mark.parametrize(
        ("make_exception", "expected_code"),
        [
            pytest.param(
                lambda: ValueError(f"invalid region, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
                id="value_error",
            ),
            pytest.param(
                lambda: ValidationError(f"1 validation error, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
                id="validation_error",
            ),
            pytest.param(
                lambda: SolverNotFoundError(f"no solver matches, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
                id="solver_not_found",
            ),
            pytest.param(
                lambda: ConfigFileError(f"bad dwave.conf, token={FAKE_TOKEN}"),
                "DWAVE_CONFIG_INVALID",
                id="config_file_error",
            ),
            pytest.param(
                lambda: SolverAuthenticationError(f"invalid token={FAKE_TOKEN}"),
                "REMOTE_AUTH_FAILED",
                id="auth",
            ),
            pytest.param(
                lambda: RequestTimeout(f"timed out, token={FAKE_TOKEN}"),
                "REMOTE_TIMEOUT",
                id="timeout",
            ),
            pytest.param(
                lambda: RuntimeError(f"client exploded, token={FAKE_TOKEN}"),
                "REMOTE_SOLVER_ERROR",
                id="other",
            ),
        ],
    )
    def test_factory_exceptions_map_to_codes_and_are_redacted(
        self, make_exception, expected_code
    ):
        backend = self.case.backend_cls(
            sampler_factory=CountingFactory(
                self.case.fake_sampler_cls(), failures=[make_exception()]
            )
        )

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(
                self.case.make_compiled_problem(), self.case.make_preferences()
            )

        assert exc_info.value.code == expected_code
        message = str(exc_info.value)
        assert FAKE_TOKEN not in message
        assert "***" in message


class OriginalExceptionIsNotReachableContract:
    """Phase 2 spec §19: the wrapped error carries no ``__cause__`` /
    ``__context__`` chain back to the original, so a traceback dump cannot
    print credential-bearing text."""

    case: ClassVar[OceanBackendCase]

    def test_sample_failure_has_no_cause_or_context(self):
        fake = self.case.fake_sampler_cls(
            raise_on_sample=RuntimeError(f"solver exploded, token={FAKE_TOKEN}")
        )
        backend = self.case.backend_cls(sampler_factory=lambda: fake)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(
                self.case.make_compiled_problem(), self.case.make_preferences()
            )

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        formatted = "".join(traceback.format_exception(error))
        assert FAKE_TOKEN not in formatted

    def test_factory_failure_has_no_cause_or_context(self):
        factory = CountingFactory(
            self.case.fake_sampler_cls(),
            failures=[SolverNotFoundError(f"Authorization: Bearer {FAKE_TOKEN}")],
        )
        backend = self.case.backend_cls(sampler_factory=factory)

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(
                self.case.make_compiled_problem(), self.case.make_preferences()
            )

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        formatted = "".join(traceback.format_exception(error))
        assert FAKE_TOKEN not in formatted


class SamplerCachingContract:
    """``LazySampler``: constructing a real sampler is expensive (it fetches
    solver metadata), so it is built once per backend instance — and a
    failed construction is never cached."""

    case: ClassVar[OceanBackendCase]

    def test_sampler_is_built_once_per_backend(self):
        factory = CountingFactory(self.case.fake_sampler_cls())
        backend = self.case.backend_cls(sampler_factory=factory)
        compiled = self.case.make_compiled_problem()

        backend.solve(compiled, self.case.make_preferences())
        backend.solve(compiled, self.case.make_preferences())

        assert factory.calls == 1
        assert factory.sampler.sample_calls == 2

    def test_failed_construction_is_not_cached(self):
        factory = CountingFactory(
            self.case.fake_sampler_cls(),
            failures=[SolverAuthenticationError(f"denied, token={FAKE_TOKEN}")],
        )
        backend = self.case.backend_cls(sampler_factory=factory)
        compiled = self.case.make_compiled_problem()

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compiled, self.case.make_preferences())
        assert exc_info.value.code == "REMOTE_AUTH_FAILED"

        result = backend.solve(compiled, self.case.make_preferences())

        assert factory.calls == 2
        assert result.backend == self.case.backend_name
        assert factory.sampler.sample_calls == 1

    def test_each_backend_instance_builds_its_own_sampler(self):
        factory = CountingFactory(self.case.fake_sampler_cls())
        compiled = self.case.make_compiled_problem()

        self.case.backend_cls(sampler_factory=factory).solve(
            compiled, self.case.make_preferences()
        )
        self.case.backend_cls(sampler_factory=factory).solve(
            compiled, self.case.make_preferences()
        )

        assert factory.calls == 2

    def test_auth_failure_at_sampling_invalidates_the_cache(self):
        """2026-09-09 review F-21: the cloud rejecting the client is the one
        signal a local fingerprint cannot see, and a rejected client is
        known-bad, so the guard that classifies ``REMOTE_AUTH_FAILED`` also
        drops the cached sampler there; the next solve builds a fresh one."""
        sampler = self.case.fake_sampler_cls(
            raise_on_sample=SolverAuthenticationError(f"denied, token={FAKE_TOKEN}")
        )
        factory = CountingFactory(sampler)
        backend = self.case.backend_cls(sampler_factory=factory)
        compiled = self.case.make_compiled_problem()

        with pytest.raises(SolverExecutionError) as exc_info:
            backend.solve(compiled, self.case.make_preferences())
        assert exc_info.value.code == "REMOTE_AUTH_FAILED"
        assert factory.calls == 1

        sampler.raise_on_sample = None
        result = backend.solve(compiled, self.case.make_preferences())

        assert factory.calls == 2
        assert result.backend == self.case.backend_name


class IsAvailableContract:
    """The backend answers through the shared check in solvers.ocean, so
    the installability / credential probes are patched there."""

    case: ClassVar[OceanBackendCase]

    def test_dwave_system_not_installed(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: False)

        assert self.case.backend_cls().is_available() == AvailabilityStatus(
            category="not_installed", detail=REASON_NOT_INSTALLED
        )

    def test_credentials_not_configured(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "missing")

        assert self.case.backend_cls().is_available() == AvailabilityStatus(
            category="credentials_missing", detail=REASON_CREDENTIALS_MISSING
        )

    def test_configuration_invalid(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "invalid")

        assert self.case.backend_cls().is_available() == AvailabilityStatus(
            category="config_invalid",
            detail=REASON_CONFIG_INVALID,
            error_code="DWAVE_CONFIG_INVALID",
        )

    def test_available_when_installed_and_configured(self, monkeypatch):
        monkeypatch.setattr(ocean_module, "dwave_system_installed", lambda: True)
        monkeypatch.setattr(ocean_module, "ocean_config_status", lambda: "ok")

        assert self.case.backend_cls().is_available() == AvailabilityStatus(
            category="available"
        )
