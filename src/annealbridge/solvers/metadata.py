"""Sampleset-info sanitization and credential redaction (Phase 2 spec §17, §19).

This module must import cleanly without any D-Wave packages installed:
``dwave.cloud`` is only touched lazily inside :func:`_ocean_config_token`.
"""

import os
import re
from typing import Literal

from annealbridge.models.metadata import SolverExecutionMetadata

__all__ = [
    "SolverExecutionMetadata",
    "ocean_config_status",
    "ocean_token_configured",
    "redact",
    "sanitize_sampleset_info",
]

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


def _ocean_config_token() -> str | None:
    """Return the token from the active Ocean config, if resolvable.

    D-Wave packages are optional; any failure (package missing, unreadable
    config, no token) means there is nothing to redact from config.
    """
    try:
        from dwave.cloud.config import load_config
    except Exception:
        return None
    try:
        config = load_config()
    except Exception:
        return None
    token = config.get("token") if isinstance(config, dict) else None
    if isinstance(token, str) and token:
        return token
    return None


def ocean_config_status() -> Literal["ok", "missing", "invalid"]:
    """Classify the active D-Wave credential configuration. No network I/O.

    - ``"ok"``: a non-empty token resolves (Ocean config or the
      ``DWAVE_API_TOKEN`` env var, which Ocean honours too).
    - ``"invalid"``: ``dwave.cloud`` is importable but ``load_config()``
      raises — a config exists but cannot be parsed. The env var does not
      rescue this case: a broken config file would still break the Ocean
      runtime, so the operator must fix it.
    - ``"missing"``: no token resolves anywhere.

    Lives here because this is the one module allowed to touch
    ``dwave.cloud.config`` (architecture boundary exemption); callers only
    ever see the categorical status, never config values.
    """
    try:
        from dwave.cloud.config import load_config
    except Exception:
        # dwave.cloud not installed: the env var is the only config source.
        return "ok" if _env_token() is not None else "missing"
    try:
        config = load_config()
    except Exception:
        return "invalid"
    # load_config() already merges the env var, but check it explicitly too
    # in case an older Ocean version does not.
    token = config.get("token") if isinstance(config, dict) else None
    if (isinstance(token, str) and token) or _env_token() is not None:
        return "ok"
    return "missing"


def ocean_token_configured() -> bool:
    """Report whether a non-empty D-Wave token is configured.

    Exposes a bare boolean so callers never see config values.
    """
    return ocean_config_status() == "ok"


def redact(text: str) -> str:
    """Mask credential material in ``text`` (spec §19).

    Every string headed for a SolveError, log line or metadata field must
    pass through here before leaving the solver layer. Candidate tokens
    (Ocean config and the ``DWAVE_API_TOKEN`` env var) are resolved live on
    every call.
    """
    for token in (_ocean_config_token(), _env_token()):
        if token:
            text = text.replace(token, "***")
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
