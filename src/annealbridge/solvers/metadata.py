"""Sampleset-info sanitization, credential redaction and the guarded call
(Phase 2 spec §17, §19; 3b spec §20.8).

Vendor-neutral by construction (OVERVIEW §五 principle 4; 2026-09-09 review
F-10): this module names no environment variable, no HTTP header and no
token shape of any vendor. Every backend declares what its credential
material looks like in ``SolverCapabilities.credentials``
(:class:`~annealbridge.models.capabilities.CredentialDeclaration`), and
``SolverRegistry`` hands each declaration to :func:`declare_credentials`
when the backend is registered. A credential that is not an environment
variable at all (e.g. a token in a vendor config file) is contributed by
the backend through :func:`register_secret_source`. :func:`redact` reads
both process-level tables live on every call.

The D-Wave availability / Ocean-config helpers that used to live here are
in ``solvers.ocean`` — the one module that may lazy-import
``dwave.cloud.config``. This module imports no vendor package, lazily or
otherwise.
"""

import os
import re
from typing import Callable, TypeVar

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models.capabilities import CredentialDeclaration
from annealbridge.models.metadata import SolverExecutionMetadata

__all__ = [
    "REMOTE_ERROR_FALLBACK_CODE",
    "classify_exception",
    "credential_env_vars",
    "declare_credentials",
    "guarded_call",
    "redact",
    "register_secret_source",
    "sanitize_sampleset_info",
]

_T = TypeVar("_T")

# Catalog code for a remote failure no classification table names.
REMOTE_ERROR_FALLBACK_CODE = "REMOTE_SOLVER_ERROR"

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
        # vendor's millisecond strings to float microseconds before calling
        # ``sanitize_sampleset_info``.
        "solve_time",
        "total_elapsed_time",
    }
)

# Spec §19: protocol-level conventions that are nobody's vendor knowledge —
# a ``token=`` URL query parameter and the standard HTTP ``Authorization``
# header. They are the only patterns this module owns; every vendor-shaped
# pattern comes from a backend's declaration.
_FALLBACK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"token=[^\s&]+"), "token=***"),
    (re.compile(r"Authorization: [^\n]+"), "Authorization: ***"),
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
            # ``X-Name: value`` line form and JSON ``"X-Name": "value"`` form.
            patterns.append((re.compile(rf"{escaped}: [^\n]+"), f"{header}: ***"))
            patterns.append(
                (re.compile(rf'"{escaped}":\s*"[^"]*"'), f'"{header}": "***"')
            )
        self.patterns = tuple(patterns)


# Process-level tables, keyed by backend name / source name so that
# re-registering (a second registry in the same process, a test building
# its own) replaces rather than accumulates. They are unions across every
# registry ever built in this process: over-masking is always safe.
_DECLARATIONS: dict[str, _CompiledDeclaration] = {}
_SECRET_SOURCES: dict[str, Callable[[], str | None]] = {}


def declare_credentials(backend_name: str, declaration: CredentialDeclaration) -> None:
    """Make ``declaration`` part of what :func:`redact` masks.

    Called by ``SolverRegistry`` for every registered backend; a backend
    author never calls it directly. An empty declaration is recorded too
    (it replaces a stale one under the same name).
    """
    _DECLARATIONS[backend_name] = _CompiledDeclaration(declaration)


def register_secret_source(name: str, source: Callable[[], str | None]) -> None:
    """Register a live provider of one secret value that is not an env var.

    ``source`` is called on every :func:`redact` and must never raise; it
    returns the current value or None. Idempotent per ``name``.
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


def _live_secrets() -> list[str]:
    """Current non-empty values of every declared env var and secret source.

    Read live on every call — never cached — so tests and runtime config
    changes are honoured (Phase 2 §19).
    """
    values: list[str] = []
    for name in credential_env_vars():
        value = os.environ.get(name)
        if isinstance(value, str) and value:
            values.append(value)
    for source in _SECRET_SOURCES.values():
        try:
            value = source()
        except Exception:
            value = None
        if isinstance(value, str) and value:
            values.append(value)
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


def redact(text: str) -> str:
    """Mask credential material in ``text`` (spec §19).

    Every string headed for a SolveError, log line or metadata field must
    pass through here before leaving the solver layer. Candidate secrets —
    the live value of every declared credential env var and of every
    registered secret source — are resolved on every call and replaced
    literally; then every declared value / header pattern and the two
    protocol-level fallbacks are applied.
    """
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
