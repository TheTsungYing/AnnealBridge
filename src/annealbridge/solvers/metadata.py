"""Sampleset-info sanitization, credential redaction and the guarded call
(Phase 2 spec §17, §19; 3b spec §20.8).

Vendor-neutral by construction (OVERVIEW §五 principle 4; 2026-09-09 review
F-10): this module names no environment variable, no HTTP header and no
token shape of any vendor. Every backend declares what its credential
material looks like in ``SolverCapabilities.credentials``
(:class:`~annealbridge.models.capabilities.CredentialDeclaration`), and
that declaration reaches :func:`declare_credentials` twice: the backend
declares it in its own ``__init__`` (so a directly constructed instance is
masked too; the Ocean backends do so through
``solvers.ocean.ocean_sampler_holder``, the call that also creates their
sampler holder) and ``SolverRegistry`` declares it again when the backend is
registered. A credential that is not an environment variable at all (e.g.
a token in a vendor config file) is contributed by the backend through
:func:`register_secret_source`. :func:`redact` reads both process-level
tables live on every call.

The D-Wave availability / Ocean-config helpers that used to live here are
in ``solvers.ocean`` — the one module that may lazy-import
``dwave.cloud.config``. This module imports no vendor package, lazily or
otherwise.

Known limits of literal replacement (2026-09-09 review F-19)
------------------------------------------------------------
:func:`redact` masks a credential by replacing its *live value* literally,
plus the declared header / value patterns. That has edges an operator
should know:

* A candidate value is stripped of surrounding whitespace first (a trailing
  newline from a ``.env`` file must not disable the mask), and a value
  shorter than :data:`MIN_LITERAL_SECRET_LENGTH` is not replaced literally
  at all — replacing ``"1"`` everywhere would turn ``"12 samples"`` into
  ``"***2 samples"``. Such a value is only masked by the patterns.
* Besides the verbatim value, its JSON-escaped and URL-encoded forms are
  masked. Any other transformation — base64, a key split across lines by
  the sender, a key a vendor truncates inside its own error body — is out
  of reach of literal replacement. Those cases are covered upstream: the
  key is only ever sent as a header, never in a URL or body; a response
  body is redacted *before* it is summarised or cut to length; and the
  guarded call drops the original exception so no unredacted text survives
  in a traceback.
* Header and protocol patterns are case-insensitive; a backend's
  ``value_patterns`` are applied exactly as declared.
"""

import json
import os
import re
import urllib.parse
from typing import Any, Callable, Iterable, Mapping, TypeVar

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models.capabilities import CredentialDeclaration
from annealbridge.models.metadata import SolverExecutionMetadata

__all__ = [
    "MIN_LITERAL_SECRET_LENGTH",
    "REMOTE_ERROR_FALLBACK_CODE",
    "classify_exception",
    "credential_env_vars",
    "declare_credentials",
    "guarded_call",
    "redact",
    "register_secret_source",
    "remote_metadata",
    "sanitize_sampleset_info",
]

_T = TypeVar("_T")

# Catalog code for a remote failure no classification table names.
REMOTE_ERROR_FALLBACK_CODE = "REMOTE_SOLVER_ERROR"

# A live credential value shorter than this (after stripping) is not
# replaced literally — see the module docstring. Real API tokens are far
# longer; the threshold only guards against a mis-set variable shredding
# every number in a message.
MIN_LITERAL_SECRET_LENGTH = 8

# Spec §17: the only timing keys that may leave the solver layer.
#
# This is an *allow-list on output* and deliberately stays a core decision
# rather than a per-backend declaration: a key a backend reports but this
# set does not name is silently dropped (fail-safe), the opposite failure
# mode from a credential the redaction does not know about (fail-open).
# Widening it is the same kind of decision as adding a ``ModelType``.
TIMING_WHITELIST = frozenset(
    {
        "qpu_access_time",
        "qpu_sampling_time",
        "qpu_anneal_time_per_sample",
        "qpu_programming_time",
        "total_post_processing_time",
        "run_time",
        "charge_time",
        # Digital-annealer style keys (3b spec §21); the backend converts the
        # vendor's millisecond strings to float microseconds itself and
        # passes them to ``remote_metadata``.
        "solve_time",
        "total_elapsed_time",
    }
)

# Spec §19: protocol-level conventions that are nobody's vendor knowledge —
# a ``token=`` URL query parameter and the standard HTTP ``Authorization``
# header. They are the only patterns this module owns; every vendor-shaped
# pattern comes from a backend's declaration.
_FALLBACK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"token=[^\s&]+", re.IGNORECASE), "token=***"),
    (re.compile(r"Authorization:[ \t]*[^\n]+", re.IGNORECASE), "Authorization: ***"),
]


class _CompiledDeclaration:
    """A backend's :class:`CredentialDeclaration` with its patterns compiled once."""

    __slots__ = ("env_vars", "patterns")

    def __init__(self, declaration: CredentialDeclaration) -> None:
        self.env_vars = tuple(declaration.env_vars)
        patterns: list[tuple[re.Pattern[str], str]] = []
        for source in declaration.value_patterns:
            patterns.append((re.compile(source), "***"))
        for header in declaration.header_names:
            escaped = re.escape(header)
            # ``X-Name: value`` line form and JSON ``"X-Name": "value"`` form,
            # both case-insensitive (HTTP header names are; a proxy or the
            # vendor may echo them lower-cased).
            patterns.append(
                (re.compile(rf"{escaped}:[ \t]*[^\n]+", re.IGNORECASE), f"{header}: ***")
            )
            patterns.append(
                (
                    re.compile(rf'"{escaped}":\s*"[^"]*"', re.IGNORECASE),
                    f'"{header}": "***"',
                )
            )
        self.patterns = tuple(patterns)


# Process-level tables, keyed by backend name / source name so that
# re-registering (a second registry in the same process, a test building
# its own) replaces rather than accumulates. They are unions across every
# registry ever built in this process: over-masking is always safe.
_DECLARATIONS: dict[str, _CompiledDeclaration] = {}
_SECRET_SOURCES: dict[str, Callable[[], "str | Iterable[str] | None"]] = {}


def declare_credentials(backend_name: str, declaration: CredentialDeclaration) -> None:
    """Make ``declaration`` part of what :func:`redact` masks.

    Called twice for a shipped backend, on purpose (2026-09-11 review F-03):
    a backend that carries credentials declares its own in ``__init__``, so
    the masking is in place for a backend *constructed directly* — the
    public ``solve()`` of a hand-built instance is a supported path and must
    not leak a key just because no registry was involved — and
    ``SolverRegistry`` declares every registered backend again at
    registration time. Both are plain dict writes keyed by backend name:
    idempotent, and a second call under the same name replaces the first
    rather than accumulating. An empty declaration is recorded too (it
    replaces a stale one under the same name).
    """
    _DECLARATIONS[backend_name] = _CompiledDeclaration(declaration)


def register_secret_source(
    name: str, source: Callable[[], "str | Iterable[str] | None"]
) -> None:
    """Register a live provider of secret values that are not env vars.

    ``source`` is called on every :func:`redact` and must never raise; it
    returns the current value, an iterable of values (a config file with
    several profiles) or None. Whether it re-reads its backing store on
    every call or caches by that store's fingerprint is the source's own
    business. Idempotent per ``name``.
    """
    _SECRET_SOURCES[name] = source


def credential_env_vars() -> list[str]:
    """Every environment variable name currently declared, in declaration order."""
    names: list[str] = []
    for compiled in _DECLARATIONS.values():
        for name in compiled.env_vars:
            if name not in names:
                names.append(name)
    return names


def _timing_value(value: object) -> float | None:
    """Return ``value`` as a float if it is a plain number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def remote_metadata(
    backend: str,
    timing_us: Mapping[str, object],
    *,
    solver_id: str | None = None,
    num_reads_requested: int | None = None,
    effective_time_limit_seconds: float | None = None,
    average_chain_break_fraction: float | None = None,
    embedding_max_chain_length: int | None = None,
    sampler_reported_feasible: int | None = None,
) -> SolverExecutionMetadata:
    """The execution metadata of a remote run, from timing already in microseconds.

    The one place a remote backend's ``SolverExecutionMetadata`` is built
    (2026-09-15 consolidation): :func:`sanitize_sampleset_info` calls it for
    an Ocean ``info`` dict, and a backend whose vendor reports timing some
    other way (the digital annealer's millisecond strings) converts to float
    microseconds itself and calls it directly.

    ``timing_us`` is filtered through the spec §17 whitelist *here*,
    whatever the caller already did, so calling this instead of
    :func:`sanitize_sampleset_info` cannot bypass it: a key
    :data:`TIMING_WHITELIST` does not name, or a value that is not a plain
    number (``bool`` included), is dropped — never raised on, never passed
    through (fail-safe, see the whitelist). Surviving values are coerced to
    float and keep the caller's key order; the mapping itself is never
    kept. The keyword fields are the vendor-side facts a remote backend
    reports, each with a caller; any other keyword is a ``TypeError``. The
    result is always ``remote=True``.
    """
    whitelisted: dict[str, float] = {}
    for key, raw in timing_us.items():
        if key not in TIMING_WHITELIST:
            continue
        value = _timing_value(raw)
        if value is not None:
            whitelisted[key] = value

    return SolverExecutionMetadata(
        backend=str(backend),
        remote=True,
        timing_us=whitelisted,
        solver_id=solver_id,
        num_reads_requested=num_reads_requested,
        effective_time_limit_seconds=effective_time_limit_seconds,
        average_chain_break_fraction=average_chain_break_fraction,
        embedding_max_chain_length=embedding_max_chain_length,
        sampler_reported_feasible=sampler_reported_feasible,
    )


def sanitize_sampleset_info(
    info: dict, backend: str, **fields: Any
) -> SolverExecutionMetadata:
    """Extract whitelisted timing facts from a raw ``sampleset.info`` dict.

    Only the spec §17 whitelist keys survive, taken from the nested
    ``info["timing"]`` dict (QPU samplesets) and from the top level of
    ``info`` (hybrid samplesets report ``run_time`` / ``charge_time``
    there).  Values must be plain numbers and are coerced to float; any
    other type is dropped.  The raw info dict is never passed through.

    This is the remote backends' tool for turning a vendor's ``info`` dict
    into metadata, so the result is always ``remote=True``; the former
    ``remote=`` keyword had no production caller (2026-09-09 review F-18).
    ``fields`` are the vendor-side facts the backend computed itself
    (``num_reads_requested``, ``effective_time_limit_seconds``, …) and go
    to :func:`remote_metadata` unchanged, which accepts only its named
    keywords. The local backends have no vendor info to extract and build
    their own ``SolverExecutionMetadata(remote=False)`` directly, without
    this function. A test double that wants a different flag overrides it
    with ``model_copy``.
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

    return remote_metadata(backend, timing_us, **fields)


def _literal_forms(secret: str) -> list[str]:
    """The strings to replace for one candidate value, or none.

    Stripped of surrounding whitespace, dropped below
    :data:`MIN_LITERAL_SECRET_LENGTH`, and joined by the JSON-escaped and
    URL-encoded spellings when they differ (a key holding ``/`` or ``"``
    looks different inside a JSON body or a query string).
    """
    secret = secret.strip()
    if len(secret) < MIN_LITERAL_SECRET_LENGTH:
        return []
    forms = [secret]
    for variant in (json.dumps(secret)[1:-1], urllib.parse.quote(secret, safe="")):
        if variant not in forms:
            forms.append(variant)
    return forms


def _live_secrets() -> list[str]:
    """Current values of every declared env var and secret source, expanded
    by :func:`_literal_forms`.

    The env vars are read live on every call — never cached — so tests and
    runtime config changes are honoured (Phase 2 §19); a secret source
    decides its own caching.
    """
    candidates: list[str] = []
    for name in credential_env_vars():
        value = os.environ.get(name)
        if isinstance(value, str):
            candidates.append(value)
    for source in _SECRET_SOURCES.values():
        try:
            value = source()
        except Exception:
            value = None
        if isinstance(value, str):
            candidates.append(value)
        elif value is not None:
            try:
                candidates.extend(item for item in value if isinstance(item, str))
            except TypeError:
                pass
    values: list[str] = []
    for candidate in candidates:
        for form in _literal_forms(candidate):
            if form not in values:
                values.append(form)
    return values


def classify_exception(exc: Exception, codes: dict[str, str]) -> str:
    """Map ``exc`` to a catalog code by class name across its MRO.

    ``codes`` is the per-backend / per-stage table; anything it does not
    name is :data:`REMOTE_ERROR_FALLBACK_CODE`.

    Matching by name (not identity) keeps classification testable with
    fakes and working without the vendor SDK installed. The walk starts
    at the most-derived class, so a named vendor exception that happens to
    subclass ``ValueError`` still wins over a ``ValueError`` entry.
    """
    for klass in type(exc).__mro__:
        code = codes.get(klass.__name__)
        if code is not None:
            return code
    return REMOTE_ERROR_FALLBACK_CODE


def redact(text: object) -> str:
    """Mask credential material in ``text`` (spec §19).

    Every string headed for a SolveError, log line or metadata field must
    pass through here before leaving the solver layer. Candidate secrets —
    the live value of every declared credential env var and of every
    registered secret source — are resolved on every call and replaced
    literally (see the module docstring for the edges of that); then every
    declared value / header pattern and the two protocol-level fallbacks
    are applied.

    Defensive on its input: ``None`` becomes ``""`` and any other non-string
    is rendered with ``str()`` first, so a caller that hands over an
    exception object or a vendor payload never gets a ``TypeError`` in
    place of a masked message.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    for secret in _live_secrets():
        text = text.replace(secret, "***")
    for compiled in _DECLARATIONS.values():
        for pattern, replacement in compiled.patterns:
            text = pattern.sub(replacement, text)
    for pattern, replacement in _FALLBACK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def guarded_call(
    what: str,
    classify: Callable[[Exception], str],
    fn: Callable[[], _T],
) -> _T:
    """Run ``fn`` and convert any failure into a redacted SolverExecutionError.

    Vendor-neutral core of the remote backends' error handling (3b spec
    §20.8): ``classify`` maps the caught exception to a catalog code, and
    the message is ``"<what>: <ExceptionClass>: <text>"`` passed through
    :func:`redact`. The wrapped error is raised *after* the ``except`` block
    has finished, so it carries neither ``__cause__`` nor ``__context__``:
    the original exception (whose text may embed credentials) is not
    reachable from the error that leaves the solver layer, and
    ``traceback.format_exception`` / ``logger.exception`` cannot print it
    (Phase 2 spec §19). The original class name is kept in the message
    because it is categorical, not secret.
    """
    try:
        return fn()
    except Exception as exc:
        error = SolverExecutionError(
            redact(f"{what}: {type(exc).__name__}: {exc}"),
            code=classify(exc),
        )
    raise error
