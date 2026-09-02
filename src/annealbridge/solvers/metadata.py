"""Sampleset-info sanitization, credential redaction and D-Wave availability
checks (Phase 2 spec §17, §19, §10).

This module must import cleanly without any D-Wave packages installed:
``dwave.cloud`` is only touched lazily inside :func:`_resolve_ocean_config`,
and ``dwave.system`` is only probed with ``importlib.util.find_spec``. It
is the one place in the solver layer that knows how Ocean config works, so
both remote backends share the helpers here instead of carrying copies.
"""

import importlib.util
import os
import re
from typing import Literal

from annealbridge.models.metadata import SolverExecutionMetadata

__all__ = [
    "REASON_CONFIG_INVALID",
    "REASON_CREDENTIALS_MISSING",
    "REASON_NOT_INSTALLED",
    "REMOTE_ERROR_FALLBACK_CODE",
    "classify_exception",
    "dwave_availability",
    "dwave_system_installed",
    "ocean_config_status",
    "redact",
    "sanitize_sampleset_info",
]

# Spec §10: the categorical ``is_available()`` reasons for the D-Wave
# backends. They never contain config values; the service maps them to
# error codes by string, so they are constants rather than free text.
REASON_NOT_INSTALLED = "dwave-system not installed"
REASON_CREDENTIALS_MISSING = "D-Wave credentials not configured"
REASON_CONFIG_INVALID = "D-Wave configuration invalid"

# Catalog code for a remote failure no classification table names.
REMOTE_ERROR_FALLBACK_CODE = "REMOTE_SOLVER_ERROR"

# Spec §17: the only timing keys that may leave the solver layer.
TIMING_WHITELIST = frozenset(
    {
        "qpu_access_time",
        "qpu_sampling_time",
        "qpu_anneal_time_per_sample",
        "qpu_programming_time",
        "total_post_processing_time",
        "run_time",
        "charge_time",
    }
)

# Spec §19: mask token-shaped substrings regardless of Ocean config state.
_REDACTION_PATTERNS = [
    (re.compile(r"DEV-[A-Za-z0-9]{20,}"), "***"),
    (re.compile(r"token=[^\s&]+"), "token=***"),
    (re.compile(r"Authorization: [^\n]+"), "Authorization: ***"),
]

OceanConfigStatus = Literal["ok", "missing", "invalid"]


def _timing_value(value: object) -> float | None:
    """Return ``value`` as a float if it is a plain number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def sanitize_sampleset_info(
    info: dict,
    backend: str,
    *,
    remote: bool = True,
) -> SolverExecutionMetadata:
    """Extract whitelisted timing facts from a raw ``sampleset.info`` dict.

    Only the spec §17 whitelist keys survive, taken from the nested
    ``info["timing"]`` dict (QPU samplesets) and from the top level of
    ``info`` (hybrid samplesets report ``run_time`` / ``charge_time``
    there).  Values must be plain numbers and are coerced to float; any
    other type is dropped.  The raw info dict is never passed through.

    ``remote`` is not part of the spec signature; callers that sanitize a
    local sampleset pass ``remote=False`` so the solver layer never needs a
    hardcoded list of remote backend names.
    """
    timing_us: dict[str, float] = {}

    nested_timing = info.get("timing")
    sources = [info]
    if isinstance(nested_timing, dict):
        sources.append(nested_timing)

    for source in sources:
        for key in TIMING_WHITELIST:
            value = _timing_value(source.get(key))
            if value is not None:
                timing_us[key] = value

    return SolverExecutionMetadata(
        backend=str(backend),
        remote=remote,
        timing_us=timing_us,
    )


def _env_token() -> str | None:
    """Return ``DWAVE_API_TOKEN`` from the environment, if set.

    Ocean itself honours this variable, so it must count both as
    "configured" and as material to redact. Read live on every call —
    never cached — so tests (and runtime config changes) are honoured.
    """
    token = os.environ.get("DWAVE_API_TOKEN")
    if isinstance(token, str) and token:
        return token
    return None


def _resolve_ocean_config() -> tuple[OceanConfigStatus, str | None]:
    """Classify the active Ocean config and return its token, if any.

    The single place that touches ``dwave.cloud.config`` (architecture
    boundary exemption). Returns ``(status, config_token)`` where
    ``config_token`` is the token from the Ocean config file only — the
    ``DWAVE_API_TOKEN`` env var is folded into ``status`` (Ocean honours
    it) but reported separately by :func:`_env_token` so callers that
    redact can mask both.

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


def dwave_system_installed() -> bool:
    """Return whether ``dwave.system`` is importable, without importing it."""
    try:
        return importlib.util.find_spec("dwave.system") is not None
    except (ImportError, ValueError):
        return False


def dwave_availability() -> tuple[bool, str | None]:
    """The shared ``is_available()`` answer for the D-Wave backends.

    Checks installability, then credentials. No network I/O. Reasons are
    the categorical constants above and never contain config values
    (spec §10). Evaluated live on every call: credentials can change at
    any time, so this must never be cached.
    """
    if not dwave_system_installed():
        return (False, REASON_NOT_INSTALLED)
    status = ocean_config_status()
    if status == "invalid":
        return (False, REASON_CONFIG_INVALID)
    if status == "missing":
        return (False, REASON_CREDENTIALS_MISSING)
    return (True, None)


def classify_exception(exc: Exception, codes: dict[str, str]) -> str:
    """Map ``exc`` to a catalog code by class name across its MRO.

    ``codes`` is the per-backend / per-stage table; anything it does not
    name is :data:`REMOTE_ERROR_FALLBACK_CODE`.

    Matching by name (not identity) keeps classification testable with
    fakes and working without dwave-cloud-client installed. The walk starts
    at the most-derived class, so a named Ocean exception that happens to
    subclass ``ValueError`` still wins over a ``ValueError`` entry.
    """
    for klass in type(exc).__mro__:
        code = codes.get(klass.__name__)
        if code is not None:
            return code
    return REMOTE_ERROR_FALLBACK_CODE


def redact(text: str) -> str:
    """Mask credential material in ``text`` (spec §19).

    Every string headed for a SolveError, log line or metadata field must
    pass through here before leaving the solver layer. Candidate tokens
    (Ocean config and the ``DWAVE_API_TOKEN`` env var) are resolved live on
    every call.
    """
    for token in (_resolve_ocean_config()[1], _env_token()):
        if token:
            text = text.replace(token, "***")
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
