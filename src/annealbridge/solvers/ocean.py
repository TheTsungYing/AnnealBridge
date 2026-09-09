"""Everything the D-Wave (Ocean) backends share (Phase 3a spec §17.5;
Phase 2 spec §10, §19).

``dwave_qpu``, ``leap_hybrid_bqm`` and ``leap_hybrid_cqm`` all need the
same things: a guarded call that turns any Ocean failure into a redacted,
classified :class:`SolverExecutionError`; a way to force a lazy sampleset
to resolve inside that guard; a lazily built, cached sampler; the
classification tables for sampler construction and hybrid sampling; the
availability check that reads the Ocean configuration; and the credential
declaration (env var, token shape, config-file token) that makes the
shared redaction mask D-Wave material. They live here once instead of once
per backend — and only here: since review F-10 the shared solver layer
(``solvers.metadata``) knows no vendor at all.

This module must import cleanly without any D-Wave package installed:
``dwave.cloud.config`` is only touched lazily inside
:func:`_resolve_ocean_config` (the single architecture-boundary exemption),
``dwave.system`` is only probed with ``importlib.util.find_spec``, and the
backends lazy-import ``dwave.system`` inside their own default sampler
factories (spec §4).
"""

import importlib.util
import os
import threading
from typing import Any, Callable, Literal, TypeVar

from annealbridge.models.capabilities import AvailabilityStatus, CredentialDeclaration
from annealbridge.solvers.metadata import (
    classify_exception,
    guarded_call,
    register_secret_source,
)

__all__ = [
    "HYBRID_SAMPLE_EXCEPTION_CODES",
    "OCEAN_CREDENTIALS",
    "REASON_CONFIG_INVALID",
    "REASON_CREDENTIALS_MISSING",
    "REASON_NOT_INSTALLED",
    "SAMPLER_INIT_EXCEPTION_CODES",
    "TOKEN_ENV",
    "LazySampler",
    "call_ocean",
    "dwave_availability",
    "dwave_system_installed",
    "ocean_config_status",
    "ocean_config_token",
    "register_ocean_config_token",
    "resolved",
]

_T = TypeVar("_T")

# The environment variable Ocean itself honours for the Leap API token. It
# counts both as "configured" for the availability check and as material
# for the redaction to mask.
TOKEN_ENV = "DWAVE_API_TOKEN"

# What D-Wave credential material looks like (review F-10): the env var
# above and the ``DEV-…`` shape of a Leap token, so a token that never was
# in this process's environment (echoed by the cloud, from another
# profile) is still masked. The Ocean *config-file* token is not an env
# var; :func:`register_ocean_config_token` contributes it as a live secret
# source instead.
OCEAN_CREDENTIALS = CredentialDeclaration(
    env_vars=[TOKEN_ENV],
    value_patterns=[r"DEV-[A-Za-z0-9]{20,}"],
)

# Spec §10: the categorical ``is_available()`` details for the D-Wave
# backends. They never contain config values. Since Phase 3a the service
# maps availability by ``AvailabilityStatus.category``, not by these strings;
# they remain constants so tests and messages share one wording.
REASON_NOT_INSTALLED = "dwave-system not installed"
REASON_CREDENTIALS_MISSING = "D-Wave credentials not configured"
REASON_CONFIG_INVALID = "D-Wave configuration invalid"

OceanConfigStatus = Literal["ok", "missing", "invalid"]


def _env_token() -> str | None:
    """Return :data:`TOKEN_ENV` from the environment, if set.

    Read live on every call — never cached — so tests (and runtime config
    changes) are honoured.
    """
    token = os.environ.get(TOKEN_ENV)
    if isinstance(token, str) and token:
        return token
    return None


def _resolve_ocean_config() -> tuple[OceanConfigStatus, str | None]:
    """Classify the active Ocean config and return its token, if any.

    The single place that touches ``dwave.cloud.config`` (architecture
    boundary exemption). Returns ``(status, config_token)`` where
    ``config_token`` is the token from the Ocean config file only — the
    :data:`TOKEN_ENV` env var is folded into ``status`` (Ocean honours it)
    but masked separately through the credential declaration.

    - ``"ok"``: a non-empty token resolves (config or env var).
    - ``"invalid"``: ``dwave.cloud`` is importable but ``load_config()``
      raises — a config exists but cannot be parsed. The env var does not
      rescue this case: a broken config file would still break the Ocean
      runtime, so the operator must fix it.
    - ``"missing"``: no token resolves anywhere.

    No network I/O; any failure yields no token rather than an exception.
    """
    try:
        from dwave.cloud.config import load_config
    except Exception:
        # dwave.cloud not installed: the env var is the only config source.
        return ("ok" if _env_token() is not None else "missing"), None
    try:
        config = load_config()
    except Exception:
        return "invalid", None
    token = config.get("token") if isinstance(config, dict) else None
    if not (isinstance(token, str) and token):
        token = None
    # load_config() already merges the env var, but check it explicitly too
    # in case an older Ocean version does not.
    if token is not None or _env_token() is not None:
        return "ok", token
    return "missing", None


def ocean_config_status() -> OceanConfigStatus:
    """Classify the active D-Wave credential configuration. No network I/O.

    Callers only ever see the categorical status, never config values.
    """
    return _resolve_ocean_config()[0]


def ocean_config_token() -> str | None:
    """The token from the Ocean config *file*, read live; None when absent.

    This is the D-Wave backends' secret source for the shared redaction:
    a config-file token is never in the environment, so the env-var
    candidate cannot mask it.
    """
    return _resolve_ocean_config()[1]


def register_ocean_config_token() -> None:
    """Contribute the Ocean config-file token to the shared redaction.

    Every D-Wave backend calls this from its constructor; the registration
    is idempotent, so three backends registering is the same as one.
    """
    register_secret_source("ocean_config", ocean_config_token)


def dwave_system_installed() -> bool:
    """Return whether ``dwave.system`` is importable, without importing it."""
    try:
        return importlib.util.find_spec("dwave.system") is not None
    except (ImportError, ValueError):
        return False


def dwave_availability() -> AvailabilityStatus:
    """The shared ``is_available()`` answer for the D-Wave backends.

    Checks installability, then credentials. No network I/O. Details are
    the categorical constants above and never contain config values
    (spec §10). ``config_invalid`` names the D-Wave-specific catalog code
    so the service can report it without knowing the backend (3a §8.2).
    Evaluated live on every call: credentials can change at any time, so
    this must never be cached.
    """
    if not dwave_system_installed():
        return AvailabilityStatus(category="not_installed", detail=REASON_NOT_INSTALLED)
    status = ocean_config_status()
    if status == "invalid":
        return AvailabilityStatus(
            category="config_invalid",
            detail=REASON_CONFIG_INVALID,
            error_code="DWAVE_CONFIG_INVALID",
        )
    if status == "missing":
        return AvailabilityStatus(
            category="credentials_missing", detail=REASON_CREDENTIALS_MISSING
        )
    return AvailabilityStatus(category="available")

# Ocean exception class names → catalog error codes, matched by name across
# the exception's MRO so classification works (and is testable with fakes)
# without dwave-cloud-client installed. The MRO walk starts at the
# most-derived class, so a named Ocean exception that happens to subclass
# ValueError still wins over the ValueError entry.
#
# Sampler construction (``DWaveSampler()`` / ``LeapHybridSampler()`` /
# ``LeapHybridCQMSampler()`` → Client.from_config() + get_solver()) has its
# own table: a ValueError there (including pydantic's ValidationError, a
# ValueError subclass) means an invalid region / endpoint / profile /
# timeout / solver selection — a configuration problem, never a solve (or
# embedding) failure. Telling the agent to "shrink the problem" would be
# wrong.
SAMPLER_INIT_EXCEPTION_CODES: dict[str, str] = {
    "SolverAuthenticationError": "REMOTE_AUTH_FAILED",
    "RequestTimeout": "REMOTE_TIMEOUT",
    "SolverNotFoundError": "DWAVE_CONFIG_INVALID",
    "ConfigFileError": "DWAVE_CONFIG_INVALID",
    "ValueError": "DWAVE_CONFIG_INVALID",
}

# Sampling stage for the Leap hybrid solvers (BQM and CQM alike). The QPU
# backend keeps its own sampling table because a bare ValueError from
# EmbeddingComposite means "no embedding found" there, which is not a
# hybrid concept.
HYBRID_SAMPLE_EXCEPTION_CODES: dict[str, str] = {
    "SolverAuthenticationError": "REMOTE_AUTH_FAILED",
    "RequestTimeout": "REMOTE_TIMEOUT",
}


def call_ocean(what: str, codes: dict[str, str], fn: Callable[[], _T]) -> _T:
    """Run ``fn`` and convert any Ocean failure into a redacted SolverExecutionError.

    Thin wrapper over the vendor-neutral
    :func:`~annealbridge.solvers.metadata.guarded_call` (3b spec §20.8):
    the only Ocean-specific part is classifying the exception by class
    name through ``codes`` (see :func:`classify_exception`). The wrapped
    error carries neither ``__cause__`` nor ``__context__``, so credential
    material in the original exception text cannot leak (Phase 2 §19).
    """
    return guarded_call(what, lambda exc: classify_exception(exc, codes), fn)


def resolved(sampleset: Any) -> Any:
    """Force a lazy (``SampleSet.from_future``) sampleset to resolve now.

    Ocean samplers return samplesets whose cloud request only completes on
    first attribute access; resolving inside the guarded call is what
    routes RequestTimeout / SolverFailureError through classification and
    redaction instead of letting them escape later, unclassified.
    """
    sampleset.resolve()
    return sampleset


class LazySampler:
    """A sampler built on first use and cached for the owner's lifetime.

    ``factory`` is the backend's single test seam (production passes
    ``None`` and gets ``default_factory``, which lazy-imports
    ``dwave.system``). Real samplers fetch solver metadata / the working
    graph and start worker threads on construction, so rebuilding one per
    solve (or per retry) would be wasteful. Only a *successful*
    construction is cached: a transient failure never poisons the backend.

    Raw factory exceptions propagate from :meth:`get`; callers wrap the
    call in :func:`call_ocean` with :data:`SAMPLER_INIT_EXCEPTION_CODES`.
    """

    def __init__(
        self,
        factory: Callable[[], Any] | None,
        default_factory: Callable[[], Any],
    ) -> None:
        self._factory = factory or default_factory
        self._sampler: Any | None = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        """Return the cached sampler, building it on first use."""
        with self._lock:
            if self._sampler is None:
                self._sampler = self._factory()
            return self._sampler
