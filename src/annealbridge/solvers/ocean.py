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

Three things are cached here, all keyed by the same *fingerprint of the
credential sources* (2026-09-09 review F-19 / F-21): the env token value
plus the path, mtime and size of every Ocean config file. The config-file
secrets handed to the redaction are re-read only when that fingerprint
changes (``redact()`` runs on every log line, so re-parsing the config
each time was needless I/O); the resolved Ocean configuration itself is
memoised against the same key (:func:`_resolve_ocean_config`), so the
three D-Wave backends answering one capabilities query parse the INI once
instead of three times; and a :class:`LazySampler` rebuilds its sampler
when the fingerprint changes, so a rotated credential is picked up
without a process restart. The *answer* ``is_available()`` gives is not
cached: every call re-reads the environment and ``stat``s every config
file, and only the parse behind an unchanged fingerprint is memoised, so
any change of credential source shows up on the very next call — as its
docstring promises.

This module must import cleanly without any D-Wave package installed:
``dwave.cloud.config`` is only touched lazily inside
:func:`_load_ocean_config` (the single architecture-boundary exemption),
``dwave.system`` is only probed with ``importlib.util.find_spec``, and the
backends lazy-import ``dwave.system`` inside their own default sampler
factories (spec §4).
"""

import importlib.util
import logging
import os
import re
import threading
import weakref
from typing import Any, Callable, Hashable, Literal, TypeVar

from annealbridge.exceptions import SolverExecutionError
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
    "credential_fingerprint",
    "dwave_availability",
    "dwave_system_installed",
    "ocean_config_status",
    "ocean_config_token",
    "register_ocean_config_token",
    "resolved",
]

logger = logging.getLogger(__name__)

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

# Environment variables Ocean's ``load_config()`` honours for *which* config
# is active; part of the fingerprint because changing either changes the
# token without touching any file.
_CONFIG_SELECTOR_ENVS = ("DWAVE_CONFIG_FILE", "DWAVE_PROFILE")

# ``token = value`` line of an Ocean config file (INI syntax), used only
# when the file cannot be parsed by Ocean itself.
_TOKEN_LINE = re.compile(r"\s*token\s*=\s*(\S+)")

# Single-entry cache: {(env token, config fingerprint): config-file secrets}.
# Module-level so tests can swap it out; a race between two threads merely
# recomputes.
_CONFIG_SECRET_CACHE: dict[tuple, tuple[str, ...]] = {}

# Single-entry cache: {(env token, config fingerprint): resolved config}.
# Module-level so tests can swap it out; a race between two threads merely
# recomputes.
_CONFIG_RESOLUTION_CACHE: dict[tuple, tuple[OceanConfigStatus, str | None]] = {}


def _env_token() -> str | None:
    """Return :data:`TOKEN_ENV` from the environment, if set.

    Read live on every call — never cached — so tests (and runtime config
    changes) are honoured.
    """
    token = os.environ.get(TOKEN_ENV)
    if isinstance(token, str) and token:
        return token
    return None


def _load_ocean_config() -> tuple[OceanConfigStatus, str | None]:
    """The uncached parse; see :func:`_resolve_ocean_config` for the result.

    The single place that touches ``dwave.cloud.config`` (architecture
    boundary exemption). No network I/O; any failure yields no token
    rather than an exception.
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


def _resolve_ocean_config() -> tuple[OceanConfigStatus, str | None]:
    """Classify the active Ocean config and return its token, if any.

    Returns ``(status, config_token)`` where ``config_token`` is the token
    from the Ocean config file only — the :data:`TOKEN_ENV` env var is
    folded into ``status`` (Ocean honours it) but masked separately
    through the credential declaration.

    - ``"ok"``: a non-empty token resolves (config or env var).
    - ``"invalid"``: ``dwave.cloud`` is importable but ``load_config()``
      raises — a config exists but cannot be parsed. The env var does not
      rescue this case: a broken config file would still break the Ocean
      runtime, so the operator must fix it.
    - ``"missing"``: no token resolves anywhere.

    The parse behind that answer (:func:`_load_ocean_config`) is memoised
    in a single slot keyed by the *value* of :data:`TOKEN_ENV` plus
    :func:`_config_fingerprint` — which covers ``DWAVE_CONFIG_FILE``,
    ``DWAVE_PROFILE`` and the path, ``mtime_ns`` and size of every Ocean
    config file that exists. Any change to any of those recomputes.
    Without a fingerprint (``get_configfile_paths`` unavailable) every
    call parses live, exactly as it did before the cache existed.

    Every call still re-reads the environment and ``stat``s every config
    file, so the promise ``dwave_availability()`` / ``is_available()``
    make — read live, a changed credential reflected on the next call —
    is unchanged; what is saved is only the repeated INI parse for
    identical inputs, e.g. the three D-Wave backends that a single
    capabilities query asks in a row.

    The env token is part of the key for the same reason as in
    :func:`_ocean_config_secrets` (2026-09-11 review F-02): ``load_config()``
    merges it over the file's ``token``, and unsetting or rotating an env
    var touches no file, so the fingerprint alone would not move and a
    stale resolution would survive.
    """
    fingerprint = _config_fingerprint()
    if fingerprint is None:
        return _load_ocean_config()
    key = (_env_token(), fingerprint)
    cached = _CONFIG_RESOLUTION_CACHE.get(key)
    if cached is None:
        cached = _load_ocean_config()
        _CONFIG_RESOLUTION_CACHE.clear()
        _CONFIG_RESOLUTION_CACHE[key] = cached
    return cached


def ocean_config_status() -> OceanConfigStatus:
    """Classify the active D-Wave credential configuration. No network I/O.

    Callers only ever see the categorical status, never config values.
    """
    return _resolve_ocean_config()[0]


def ocean_config_token() -> str | None:
    """The token from the Ocean config *file*, read live; None when absent.

    A config-file token is never in the environment, so the env-var
    candidate cannot mask it; :func:`_ocean_config_secrets` (the cached
    secret source behind the shared redaction) builds on the same read.
    """
    return _resolve_ocean_config()[1]


def _ocean_config_paths() -> list[str] | None:
    """The Ocean config files that exist right now, or None when unknown.

    Uses ``dwave.cloud.config.get_configfile_paths`` (lazy import, same
    boundary exemption as :func:`_load_ocean_config`). None means the
    helper is unavailable — dwave-cloud-client not installed, or too old
    to have it — in which case nothing can be fingerprinted or scanned.
    """
    try:
        from dwave.cloud.config import get_configfile_paths
    except Exception:
        return None
    try:
        return [str(path) for path in get_configfile_paths()]
    except Exception:
        return None


def _config_fingerprint() -> tuple | None:
    """What the active Ocean configuration depends on, as a hashable value.

    The selector env vars and, per existing config file, its path, mtime
    and size. Equal fingerprints mean ``load_config()`` would return the
    same thing; None means it cannot be told (see :func:`_ocean_config_paths`).
    """
    paths = _ocean_config_paths()
    if paths is None:
        return None
    stats: list[tuple[str, int | None, int | None]] = []
    for path in paths:
        try:
            stat = os.stat(path)
            stats.append((path, stat.st_mtime_ns, stat.st_size))
        except OSError:
            stats.append((path, None, None))
    selectors = tuple(os.environ.get(name) for name in _CONFIG_SELECTOR_ENVS)
    return (selectors, tuple(stats))


def _raw_config_tokens(paths: list[str]) -> tuple[str, ...]:
    """Every ``token = …`` value found by scanning ``paths`` line by line.

    The fallback for a config file Ocean cannot parse (``"invalid"``): the
    token is still in that file and must still be masked. Unreadable files
    are skipped; nothing here can raise.
    """
    tokens: list[str] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = _TOKEN_LINE.match(line)
                    if match:
                        token = match.group(1).strip("\"'")
                        if token and token not in tokens:
                            tokens.append(token)
        except OSError:
            continue
    return tuple(tokens)


def _compute_config_secrets(paths: list[str] | None) -> tuple[str, ...]:
    status, token = _resolve_ocean_config()
    if status == "invalid":
        return _raw_config_tokens(paths or [])
    return (token,) if token else ()


def _ocean_config_secrets() -> tuple[str, ...]:
    """The D-Wave secret source for the shared redaction (F-19).

    Config-file tokens, cached by ``(env token, config fingerprint)`` — the
    same key shape as :func:`credential_fingerprint` — so the files are
    re-read only when a config file, a selector env var or
    :data:`TOKEN_ENV` changed. Without a fingerprint (helper unavailable)
    every call reads live, exactly as before the cache existed. Never
    raises.

    The env token belongs in the key even though only *config-file* tokens
    come out of here (2026-09-11 review F-02): what
    :func:`_resolve_ocean_config` sees is what Ocean's ``load_config()``
    merged, and there the env var wins over the file, so with it set the
    cached tuple holds the env token rather than the file's. Unsetting the
    env var touches no file, so the fingerprint alone would not move — and
    the stale tuple would keep masking the gone env token while the file
    token, now the effective credential, reached log lines unmasked.
    """
    fingerprint = _config_fingerprint()
    if fingerprint is None:
        return _compute_config_secrets(None)
    key = (_env_token(), fingerprint)
    secrets = _CONFIG_SECRET_CACHE.get(key)
    if secrets is None:
        secrets = _compute_config_secrets([entry[0] for entry in fingerprint[1]])
        _CONFIG_SECRET_CACHE.clear()
        _CONFIG_SECRET_CACHE[key] = secrets
    return secrets


def register_ocean_config_token() -> None:
    """Contribute the Ocean config-file token(s) to the shared redaction.

    Every D-Wave backend calls this from its constructor; the registration
    is idempotent, so three backends registering is the same as one. The
    source is :func:`_ocean_config_secrets`, which caches by env token and
    config fingerprint; :func:`ocean_config_token` itself stays a live read.
    """
    register_secret_source("ocean_config", _ocean_config_secrets)


def credential_fingerprint() -> Hashable:
    """The state of every D-Wave credential source, for cache keys (F-21).

    The env token's value and :func:`_config_fingerprint`. A
    :class:`LazySampler` built under one fingerprint is rebuilt when the
    next ``get()`` sees another, which is how a rotated token or edited
    config file reaches a long-running process. Cheap: two env reads and
    one ``stat`` per config file.
    """
    return (_env_token(), _config_fingerprint())


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
    this answer is never cached. The underlying config parse is memoised
    on the credential fingerprint, which every call recomputes, so any
    change still shows up immediately — see :func:`_resolve_ocean_config`.
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


def call_ocean(
    what: str,
    codes: dict[str, str],
    fn: Callable[[], _T],
    *,
    holder: "LazySampler | None" = None,
) -> _T:
    """Run ``fn`` and convert any Ocean failure into a redacted SolverExecutionError.

    Thin wrapper over the vendor-neutral
    :func:`~annealbridge.solvers.metadata.guarded_call` (3b spec §20.8):
    the only Ocean-specific part is classifying the exception by class
    name through ``codes`` (see :func:`classify_exception`). The wrapped
    error carries neither ``__cause__`` nor ``__context__``, so credential
    material in the original exception text cannot leak (Phase 2 §19).

    ``holder`` is the backend's :class:`LazySampler` when ``fn`` uses the
    cached sampler: a failure classified ``REMOTE_AUTH_FAILED`` invalidates
    it (review F-21), so the next solve builds a fresh sampler from the
    current credentials instead of reusing one whose token was revoked.
    The error is re-raised unchanged.
    """
    try:
        return guarded_call(what, lambda exc: classify_exception(exc, codes), fn)
    except SolverExecutionError as error:
        if holder is not None and error.code == "REMOTE_AUTH_FAILED":
            holder.invalidate()
        raise


def resolved(sampleset: Any) -> Any:
    """Force a lazy (``SampleSet.from_future``) sampleset to resolve now.

    Ocean samplers return samplesets whose cloud request only completes on
    first attribute access; resolving inside the guarded call is what
    routes RequestTimeout / SolverFailureError through classification and
    redaction instead of letting them escape later, unclassified.
    """
    sampleset.resolve()
    return sampleset


def _close_sampler(sampler: Any) -> None:
    """Release a sampler that is being dropped, best effort (review F-07).

    An Ocean sampler owns a ``dwave.cloud.Client``: a worker thread pool and
    an HTTP session. A :class:`LazySampler` that swaps one out (rotated
    credential) or throws one away (:meth:`LazySampler.invalidate` after
    ``REMOTE_AUTH_FAILED``) must hand those back, or a long-lived process —
    an MCP server especially — accumulates both once per rotation.

    A sampler without a callable ``close`` is left alone: that covers every
    test fake as well as any Ocean version or composite that does not expose
    one. A failing ``close()`` is logged at WARNING and swallowed — dropping
    the reference is what matters and the caller is usually already handling
    a failure, so this must never raise. The log line names only the sampler
    and exception classes, never the exception text, which could quote
    credential material.
    """
    close = getattr(sampler, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # best effort: a close failure is never fatal
        logger.warning(
            "Closing the %s sampler failed (%s); dropping it anyway",
            type(sampler).__name__,
            type(exc).__name__,
        )


class LazySampler:
    """A sampler built on first use and cached while its credentials stand.

    ``factory`` is the backend's single test seam (production passes
    ``None`` and gets ``default_factory``, which lazy-imports
    ``dwave.system``). Real samplers fetch solver metadata / the working
    graph and start worker threads on construction, so rebuilding one per
    solve (or per retry) would be wasteful. Only a *successful*
    construction is cached: a transient failure never poisons the backend.

    The cache is keyed by ``fingerprint()`` — by default
    :func:`credential_fingerprint`, the env token plus the Ocean config
    files' path / mtime / size (review F-21). A sampler built under one
    fingerprint is discarded and rebuilt when :meth:`get` sees another,
    so a rotated or revoked-then-replaced credential takes effect without
    a restart, matching ``is_available()`` which re-reads the environment
    every time. :meth:`invalidate` drops the cache explicitly; the
    :func:`call_ocean` ``holder`` hook uses it after ``REMOTE_AUTH_FAILED``.

    Construction happens inside the lock, so concurrent first callers
    share one build. Raw factory exceptions propagate from :meth:`get`;
    callers wrap the call in :func:`call_ocean` with
    :data:`SAMPLER_INIT_EXCEPTION_CODES`.

    A sampler this holder stops caching is closed through
    :func:`_close_sampler` (review F-07), each one exactly once. Known
    limitation: that close is best effort and deliberately unsynchronised
    with in-flight work. The solve paths read the sampler outside this lock,
    and a lazy sampleset only reaches the cloud when :func:`resolved` runs,
    so a request still holding a reference to the closed sampler may fail —
    as an ordinary classified solver error, which the service already
    retries or reports. Reference counting the sampler to avoid that is out
    of scope for this project.
    """

    def __init__(
        self,
        factory: Callable[[], Any] | None,
        default_factory: Callable[[], Any],
        *,
        fingerprint: Callable[[], Hashable] | None = None,
    ) -> None:
        self._factory = factory or default_factory
        self._fingerprint = fingerprint or credential_fingerprint
        self._sampler: Any | None = None
        self._built_for: Hashable = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        """Return the cached sampler, (re)building it when the fingerprint moved.

        The superseded sampler is closed only after the lock is released:
        ``close()`` shuts down a thread pool and an HTTP session and may
        block, and holding the lock across that would stall every other
        caller behind a sampler nobody wants any more. It is closed even
        when the rebuild then fails, because it has already left the cache.
        """
        stale: Any | None = None
        try:
            with self._lock:
                current = self._fingerprint()
                if self._sampler is None or current != self._built_for:
                    stale = self._sampler
                    self._sampler = None
                    self._sampler = self._factory()
                    self._built_for = current
                return self._sampler
        finally:
            if stale is not None:
                _close_sampler(stale)

    def invalidate(self) -> None:
        """Forget the cached sampler; the next :meth:`get` builds a new one.

        The dropped sampler is closed outside the lock, exactly once: a
        second :meth:`invalidate` finds the slot already empty and closes
        nothing.
        """
        with self._lock:
            stale = self._sampler
            self._sampler = None
        if stale is not None:
            _close_sampler(stale)


# ---------------------------------------------------------------------------
# Shared backend steps (2026-09-09 review F-13c)
#
# The three D-Wave backends build their sampler the same way, and the two
# Leap hybrid backends resolve their time limit by the same rule; the code
# lived once per backend and only differed in the message label.
# ---------------------------------------------------------------------------


def create_sampler(holder: LazySampler, label: str) -> Any:
    """Build (or fetch the cached) sampler, classifying construction failures.

    ``label`` names the backend in the error message (``"Leap hybrid"``,
    ``"Leap hybrid CQM"``, ``"D-Wave QPU"``): the wrapped error reads
    ``"<label> sampler could not be created"`` and carries the
    :data:`SAMPLER_INIT_EXCEPTION_CODES` classification.
    """
    return call_ocean(
        f"{label} sampler could not be created",
        SAMPLER_INIT_EXCEPTION_CODES,
        holder.get,
    )


class HybridTimeLimitMemo:
    """The last ``min_time_limit`` a hybrid backend obtained, for reuse within an attempt.

    The service asks a hybrid backend for ``resolve_time_limit()`` before
    every submission (§14 step 9, the policy check) and ``solve()`` then
    resolves the very same value again, so that what is submitted is
    exactly what was checked. Both calls see the same compiled model object
    and the same sampler, and ``min_time_limit(model)`` is a pure local
    interpolation over the model's size and the sampler's properties — for
    a CQM it walks every constraint — so the second computation only
    repeated the first.

    One slot. The sampler and the model are held by weak reference and
    matched by identity: a rebuilt sampler (rotated credential, see
    :class:`LazySampler`) or a freshly compiled model (the next attempt, the
    next solve) is a miss, and a collected model cannot alias a new one at
    the same address. The slot is replaced as one tuple, so a concurrent
    solve on the same backend can only cause a miss, never a wrong hit. A
    model or sampler that cannot be weakly referenced is simply not
    memoised. Nothing here changes the value: with or without a hit the
    submitted ``time_limit`` is the sampler's own minimum for this model,
    floored under the user's value.
    """

    __slots__ = ("_entry",)

    def __init__(self) -> None:
        self._entry: tuple[weakref.ref, weakref.ref, float] | None = None

    def lookup(self, sampler: Any, model: Any) -> float | None:
        """The memoised minimum for exactly this sampler and model, else None."""
        entry = self._entry
        if entry is None:
            return None
        sampler_ref, model_ref, value = entry
        if sampler_ref() is sampler and model_ref() is model:
            return value
        return None

    def store(self, sampler: Any, model: Any, value: float) -> None:
        """Remember ``value`` for this sampler and model, replacing the slot."""
        try:
            self._entry = (weakref.ref(sampler), weakref.ref(model), value)
        except TypeError:
            self._entry = None


def resolve_hybrid_sampler(
    holder: LazySampler,
    model: Any,
    user_time_limit: float | None,
    *,
    label: str,
    memo: HybridTimeLimitMemo | None = None,
) -> tuple[Any, float]:
    """The sampler a Leap hybrid solve would use and the ``time_limit`` it would submit.

    The one rule of both hybrid backends (Phase 2 §16, 3a §17.2): the
    user's value if given, floored at the sampler's ``min_time_limit(model)``;
    the sampler minimum alone when the user gave none. Nothing is
    submitted — ``min_time_limit`` is a local interpolation over solver
    properties fetched at construction. Construction failures are
    classified by :func:`create_sampler`, the ``min_time_limit`` call by
    :data:`HYBRID_SAMPLE_EXCEPTION_CODES`; ``label`` prefixes both messages.

    The sampler comes back alongside the limit so a backend's ``solve()``
    fetches it once — one :meth:`LazySampler.get`, one credential
    fingerprint — for the limit and the submission together instead of
    resolving and then fetching again. That ``get()`` is never skipped: it
    is where a rotated credential is noticed. With a ``memo``, a minimum
    already obtained for this very sampler and model is reused
    (:class:`HybridTimeLimitMemo`), which is what makes the service's
    pre-submission check and the solve of one attempt ask the sampler once.
    """
    sampler = create_sampler(holder, label)
    min_time_limit = memo.lookup(sampler, model) if memo is not None else None
    if min_time_limit is None:
        min_time_limit = call_ocean(
            f"{label} minimum time limit could not be determined",
            HYBRID_SAMPLE_EXCEPTION_CODES,
            lambda: float(sampler.min_time_limit(model)),
            holder=holder,
        )
        if memo is not None:
            memo.store(sampler, model, min_time_limit)
    if user_time_limit is None:
        return sampler, min_time_limit
    return sampler, max(float(user_time_limit), min_time_limit)


def resolve_hybrid_time_limit(
    holder: LazySampler,
    model: Any,
    user_time_limit: float | None,
    *,
    label: str,
    memo: HybridTimeLimitMemo | None = None,
) -> float:
    """Effective ``time_limit`` (seconds) a Leap hybrid solve would submit.

    :func:`resolve_hybrid_sampler` without the sampler: the same rule, the
    same errors, the same value.
    """
    return resolve_hybrid_sampler(
        holder, model, user_time_limit, label=label, memo=memo
    )[1]
