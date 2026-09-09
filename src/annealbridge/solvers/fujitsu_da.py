"""Fujitsu Digital Annealer backend (Phase 3b spec §20).

The first non-D-Wave remote backend: no vendor SDK, only an HTTPS JSON API
(QUBO API V4, solver block ``fujitsuDA3``). HTTP goes through the standard
library (``urllib.request``) behind the :class:`HttpTransport` seam so
nothing here needs a third-party HTTP package and tests inject a scripted
transport. The compiled QUBO is sent as one ``binary_polynomial``; the
solver's native inequality / one-hot features are not used (§30.2).

Credentials are read from the environment inside this module only —
``FUJITSU_DA_API_KEY`` (sent as the ``X-Api-Key`` header) and the optional
``FUJITSU_DA_URL`` base URL — live on every call and never cached, never
stored in ``ServerSettings`` or any model (§20.5). Every string that leaves
this module in an error, a log line or metadata passes through
:func:`~annealbridge.solvers.metadata.redact`.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CompiledProblem, FujitsuDAOptions, SolverPreferences
from annealbridge.solvers.base import (
    AvailabilityStatus,
    CredentialDeclaration,
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
)
from annealbridge.solvers.metadata import (
    REMOTE_ERROR_FALLBACK_CODE,
    guarded_call,
    redact,
    sanitize_sampleset_info,
)

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "REASON_API_KEY_MISSING",
    "REASON_URL_NOT_HTTPS",
    "SOLVER_ID",
    "URL_ENV",
    "FujitsuDABackend",
    "HttpTransport",
    "UrllibTransport",
    "binary_polynomial_terms",
    "build_request_body",
]

logger = logging.getLogger(__name__)

API_KEY_ENV = "FUJITSU_DA_API_KEY"
URL_ENV = "FUJITSU_DA_URL"
DEFAULT_BASE_URL = "https://api.aispf.global.fujitsu.com/da"
SOLVER_ID = "fujitsuDA3/v4"

# Categorical ``is_available()`` details (spec §20.5); never contain values.
REASON_API_KEY_MISSING = "Fujitsu DA API key not configured"
REASON_URL_NOT_HTTPS = "Fujitsu DA URL must start with https://"

# Vendor defaults (spec §20.1) used when the user leaves an option unset:
# the time limit is what ``resolve_time_limit`` reports, the other two feed
# ``num_reads_requested`` so metadata always has a value.
DEFAULT_TIME_LIMIT_SECONDS = 10
DEFAULT_NUM_GROUP = 1
DEFAULT_NUM_OUTPUT_SOLUTION = 5

# Polling waits at most ``effective_time_limit + POLL_GRACE_SECONDS`` (§20.7).
POLL_GRACE_SECONDS = 60.0

# Bytes of a response body quoted (after redaction) in an error message.
_BODY_SUMMARY_CHARS = 200

# Spec §20.8: HTTP status → catalog code for statuses whose meaning does not
# depend on the body. 400 is classified by message (below); 5xx and any
# other status fall back to REMOTE_SOLVER_ERROR.
_HTTP_STATUS_CODES: dict[int, str] = {
    401: "REMOTE_AUTH_FAILED",
    403: "REMOTE_AUTH_FAILED",
    413: "REMOTE_SOLVER_ERROR",
    429: "REMOTE_BUSY",
}

# Spec §20.8: HTTP 400 body substring → catalog code, checked in order. The
# vendor uses 400 both for account state (quota), for a malformed request
# header (our configuration) and for problem-level rejections such as "the
# number of variables exceeds the limit"; only the header errors listed in
# the OpenAPI document are configuration errors, everything else is a solver
# error so the agent is not told to fix a configuration that is fine.
_BAD_REQUEST_MESSAGE_CODES: dict[str, str] = {
    "Monthly usage exceeds": "REMOTE_QUOTA_EXCEEDED",
    "X-Access-Token": "BACKEND_CONFIG_INVALID",
    "X-Api-Key": "BACKEND_CONFIG_INVALID",
    "Invalid request header": "BACKEND_CONFIG_INVALID",
}

_CAPABILITIES = SolverCapabilities(
    name="fujitsu_da",
    remote=True,
    heuristic=True,
    exhaustive=False,
    supports_seed=False,
    supports_num_reads=False,
    supports_time_limit=True,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    parameter_limits=[
        ParameterLimit(
            preference="fujitsu_da.time_limit_seconds",
            limit="time_seconds",
            error_code="REMOTE_TIME_LIMIT",
        ),
    ],
    # Review F-10: the key's env var and the vendor's credential headers
    # (``X-Api-Key`` is what we send; ``X-Access-Token`` is the alternative
    # the vendor's error bodies may echo) — the shared redaction masks both
    # from this declaration alone (spec §20.5, §22).
    credentials=CredentialDeclaration(
        env_vars=[API_KEY_ENV],
        header_names=["X-Api-Key", "X-Access-Token"],
    ),
    description=(
        "Fujitsu Digital Annealer (QUBO API V4, 3rd/4th-generation solver) accessed over HTTPS with an API key. "
        "Receives the compiled QUBO (objective, hard-constraint penalties and slack bits) as one binary polynomial; "
        "its native inequality and one-hot features are not used. Returns up to num_output_solution × num_group "
        "solutions; the solver's energy and penalty_energy are ignored — every solution is re-validated. "
        "Problem size limit 100,000 bits and at most 16 pending jobs per account are enforced by the vendor."
    ),
)


class HttpTransport(Protocol):
    """The backend's only network seam (spec §20.4).

    Returns ``(status, body)`` for every HTTP response, error statuses
    included, so the backend classifies statuses itself. Transport-level
    failures (DNS, refused connection, timeout, dropped connection) are
    raised as the standard ``OSError`` family and classified by the backend.
    """

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]: ...


class UrllibTransport:
    """Standard-library transport over ``urllib.request``.

    ``HTTPError`` is a ``URLError`` subclass that carries a real HTTP
    response, so it is caught first and returned as ``(status, body)``.
    Every other ``OSError`` (``URLError`` for DNS / refused connections,
    a bare ``TimeoutError`` on read timeout, ``http.client.RemoteDisconnected``
    …) propagates unchanged for the backend to classify (spec §20.8).
    """

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url, data=body, headers=dict(headers), method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as error:
            return int(error.code), error.read()


def _classify_transport_exception(exc: Exception) -> str:
    """Transport exception → catalog code (spec §20.8).

    A read timeout after connecting is a bare ``TimeoutError``; a connect
    timeout arrives wrapped as ``URLError`` whose ``reason`` is the
    ``TimeoutError``. Both mean the remote did not answer in time. Every
    other failure (DNS, refused, reset, dropped connection) is a
    ``REMOTE_SOLVER_ERROR``, which the catalog already marks retryable.
    """
    if isinstance(exc, TimeoutError):
        return "REMOTE_TIMEOUT"
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError):
        return "REMOTE_TIMEOUT"
    return REMOTE_ERROR_FALLBACK_CODE


def _body_text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _redacted_summary(text: str) -> str:
    """One-line, length-capped, *redacted* excerpt of a response body.

    :func:`redact` runs first, on the intact text: it masks the configured
    key by literal replacement, so a key cut in half by the length cap or
    pulled onto the cap by whitespace collapsing would no longer match and
    its prefix would leak (code review 2026-09-08, F-01). Callers still wrap
    the final message in ``redact()``; that second pass is harmless.
    """
    collapsed = " ".join(redact(text).split())
    if len(collapsed) > _BODY_SUMMARY_CHARS:
        return collapsed[:_BODY_SUMMARY_CHARS] + "…"
    return collapsed


def _http_failure(what: str, status: int, body: bytes) -> SolverExecutionError:
    """Build the error for a non-2xx response (spec §20.8 table)."""
    text = _body_text(body)
    result_status: str | None = None
    if status == 400:
        code = REMOTE_ERROR_FALLBACK_CODE
        for needle, candidate in _BAD_REQUEST_MESSAGE_CODES.items():
            if needle in text:
                code = candidate
                break
        if code == "BACKEND_CONFIG_INVALID":
            result_status = "configuration_error"
    else:
        code = _HTTP_STATUS_CODES.get(status, REMOTE_ERROR_FALLBACK_CODE)
    detail = "payload too large" if status == 413 else _redacted_summary(text)
    return SolverExecutionError(
        redact(f"{what}: HTTP {status}: {detail}"),
        code=code,
        status=result_status,
    )


def _read_settings() -> tuple[str | None, str]:
    """``(api_key, base_url)`` from the environment, read live (spec §20.5).

    An empty key counts as missing. The base URL defaults to the vendor
    endpoint and loses any trailing slash so paths can be appended.
    """
    key = os.environ.get(API_KEY_ENV)
    if not isinstance(key, str) or not key:
        key = None
    url = os.environ.get(URL_ENV)
    if not isinstance(url, str) or not url:
        url = DEFAULT_BASE_URL
    return key, url.rstrip("/")


def binary_polynomial_terms(bqm: Any, index_of: Mapping[Any, int]) -> list[dict[str, Any]]:
    """The QUBO as DA ``binary_polynomial.terms`` (spec §20.7 step 2).

    One term per variable for its linear bias (zero biases included, so
    every variable exists on the solver side and comes back in
    ``configuration``), one per quadratic interaction, and the offset as a
    constant term (sent even when zero so the reported energy is directly
    comparable with the BQM's). Kept as its own function so a later
    constrained model type only has to add ``penalty_binary_polynomial`` /
    ``inequalities`` sections next to it (§30.2).
    """
    terms: list[dict[str, Any]] = []
    for variable in bqm.variables:
        terms.append(
            {"coefficient": float(bqm.linear[variable]), "polynomials": [index_of[variable]]}
        )
    for (u, v), bias in bqm.quadratic.items():
        terms.append({"coefficient": float(bias), "polynomials": [index_of[u], index_of[v]]})
    terms.append({"coefficient": float(bqm.offset), "polynomials": []})
    return terms


def build_request_body(
    bqm: Any, options: FujitsuDAOptions, effective_time_limit: float
) -> dict[str, Any]:
    """The JSON body of ``POST /v4/async/qubo/solve`` (spec §20.7 step 2).

    Only ``time_limit_sec`` (always) and the options the user actually set
    go into the ``fujitsuDA3`` block; unset options are omitted so the
    vendor default applies. No problem name, description or any other
    business string is ever sent (§24).
    """
    solver: dict[str, Any] = {"time_limit_sec": int(effective_time_limit)}
    for field in ("num_run", "num_group", "num_output_solution"):
        value = getattr(options, field)
        if value is not None:
            solver[field] = int(value)
    index_of = {variable: index for index, variable in enumerate(bqm.variables)}
    return {
        "fujitsuDA3": solver,
        "binary_polynomial": {"terms": binary_polynomial_terms(bqm, index_of)},
    }


def _timing_microseconds(timing: Any) -> dict[str, float]:
    """DA ``timing`` (millisecond strings) → whitelisted float microseconds."""
    result: dict[str, float] = {}
    if not isinstance(timing, dict):
        return result
    for key in ("solve_time", "total_elapsed_time"):
        value = timing.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue
        try:
            milliseconds = float(value)
        except ValueError:
            continue
        result[key] = milliseconds * 1000.0
    return result


def _decode_solutions(
    qubo_solution: Any, num_variables: int, job_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """``qubo_solution.solutions`` → ``(int8 samples, float64 energies)``.

    Every variable index must be present in every ``configuration`` with a
    boolean value; a missing index or a non-boolean is a solver error, not
    a zero (spec §20.7 step 6: inventing a bit on the solver's behalf would
    be a silent decision). ``frequency`` is not expanded into repeated rows
    (§3) and ``penalty_energy`` is ignored (no penalty polynomial was sent).
    """

    def failure(detail: str) -> SolverExecutionError:
        return SolverExecutionError(
            redact(f"Fujitsu DA job {job_id} returned a malformed result: {detail}"),
            code=REMOTE_ERROR_FALLBACK_CODE,
        )

    if not isinstance(qubo_solution, dict):
        raise failure("qubo_solution is not an object")
    solutions = qubo_solution.get("solutions")
    if not isinstance(solutions, list):
        raise failure("solutions is missing or not a list")

    samples = np.empty((len(solutions), num_variables), dtype=np.int8)
    energies = np.empty(len(solutions), dtype=np.float64)
    for row, solution in enumerate(solutions):
        if not isinstance(solution, dict):
            raise failure(f"solution {row} is not an object")
        configuration = solution.get("configuration")
        if not isinstance(configuration, dict):
            raise failure(f"solution {row} has no configuration object")
        for index in range(num_variables):
            value = configuration.get(str(index))
            if not isinstance(value, bool):
                raise failure(
                    f"solution {row} has no boolean value for variable index {index}"
                )
            samples[row, index] = 1 if value else 0
        energy = solution.get("energy")
        if isinstance(energy, bool) or not isinstance(energy, (int, float)):
            raise failure(f"solution {row} has no numeric energy")
        energies[row] = float(energy)
    return samples, energies


class FujitsuDABackend:
    """Solves the compiled QUBO on the Fujitsu Digital Annealer (spec §20).

    ``transport`` is the single network seam (production: a
    :class:`UrllibTransport`; tests inject a scripted fake). ``clock`` and
    ``sleep`` are injectable so the polling loop can be tested without
    waiting. Nothing is cached between calls: credentials are re-read from
    the environment on every ``is_available()`` / ``solve()``.

    Time bounds (spec §20.7; 2026-09-09 review F-23). The *polling budget*
    is ``time_limit_sec + POLL_GRACE_SECONDS``, checked before every status
    request and again after one that reports the job still queued or
    running. On top of it every single HTTP request may take up to
    ``request_timeout_seconds``, so the worst-case wall time of one solve
    is the budget plus that timeout for the submit, the last status
    request, the cancel and the delete — with the defaults, 70 s becomes
    at most about 190 s for a 10 s job. Two limits of that bound are
    inherent to ``urllib``: its timeout applies per socket operation, so a
    remote that keeps trickling bytes can hold a single request longer
    than the timeout (the deadline can only be checked between requests);
    and a submit that times out may already have created the job on the
    vendor side without a ``job_id`` ever reaching this process, so it
    cannot be released here — the error says so.

    ``num_reads`` / ``num_sweeps`` / ``seed`` are meaningless here and are
    never forwarded; only the ``fujitsu_da`` option block is.
    """

    def __init__(
        self,
        transport: HttpTransport | None = None,
        *,
        poll_interval_seconds: float = 2.0,
        request_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport: HttpTransport = (
            transport if transport is not None else UrllibTransport()
        )
        self._poll_interval = float(poll_interval_seconds)
        self._request_timeout = float(request_timeout_seconds)
        self._clock = clock
        self._sleep = sleep

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    @property
    def name(self) -> str:
        """Alias for ``capabilities.name``."""
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        """Alias for ``capabilities.exhaustive``."""
        return self.capabilities.exhaustive

    def is_available(self) -> AvailabilityStatus:
        """Credentials and endpoint sanity from the environment. No network I/O.

        Missing / empty key → ``credentials_missing``; a base URL that is not
        ``https://`` → ``config_invalid`` naming ``BACKEND_CONFIG_INVALID``
        (the detail never contains the value); otherwise available.
        """
        key, base_url = _read_settings()
        if key is None:
            return AvailabilityStatus(
                category="credentials_missing", detail=REASON_API_KEY_MISSING
            )
        if not base_url.startswith("https://"):
            return AvailabilityStatus(
                category="config_invalid",
                detail=REASON_URL_NOT_HTTPS,
                error_code="BACKEND_CONFIG_INVALID",
            )
        return AvailabilityStatus(category="available")

    def resolve_time_limit(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> float:
        """Effective ``time_limit_sec`` a solve would submit (spec §20.6).

        The user's ``fujitsu_da.time_limit_seconds`` if given, else the
        vendor default. Purely local; the service compares it with the
        ``time_seconds`` policy limit before :meth:`solve` runs.
        """
        options = preferences.fujitsu_da
        if options is not None and options.time_limit_seconds is not None:
            return float(options.time_limit_seconds)
        return float(DEFAULT_TIME_LIMIT_SECONDS)

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        """Submit the QUBO, poll the job to completion and return every solution.

        Spec §20.7: variables are indexed in ``bqm.variables`` order; the
        submitted ``time_limit_sec`` is exactly :meth:`resolve_time_limit`;
        after the job is ``Done`` its result is deleted on a best-effort
        basis (the account has a small number of job slots); a job that
        does not finish within the ``time_limit + 60 s`` polling budget is
        cancelled and deleted (best effort) and reported as
        ``REMOTE_TIMEOUT``; any other failure after submission releases the
        job the same best-effort way (F-09, see :meth:`_await_result`).
        Policy ceilings are enforced by the service layer, not here.
        """
        key, base_url = self._require_settings()
        bqm = compiled_problem.model
        options = preferences.fujitsu_da or FujitsuDAOptions()
        effective_time_limit = self.resolve_time_limit(compiled_problem, preferences)
        variables = [str(variable) for variable in bqm.variables]

        body = json.dumps(build_request_body(bqm, options, effective_time_limit)).encode(
            "utf-8"
        )
        headers = {
            "X-Api-Key": key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        job_id = self._submit(base_url, headers, body)
        qubo_solution = self._await_result(base_url, headers, job_id, effective_time_limit)
        self._best_effort(
            f"Fujitsu DA job {job_id} result could not be deleted",
            "DELETE",
            f"{base_url}/v4/async/jobs/result/{job_id}",
            headers,
            None,
        )

        samples, energies = _decode_solutions(qubo_solution, len(variables), job_id)
        timing_us = _timing_microseconds(qubo_solution.get("timing"))
        metadata = sanitize_sampleset_info({"timing": timing_us}, backend=self.name)
        metadata.solver_id = SOLVER_ID
        metadata.num_reads_requested = (
            options.num_output_solution or DEFAULT_NUM_OUTPUT_SOLUTION
        ) * (options.num_group or DEFAULT_NUM_GROUP)
        metadata.effective_time_limit_seconds = effective_time_limit
        logger.info(
            "Backend %s finished job %s: %d variables, %d solutions "
            "(solve_time_us=%s, effective_time_limit_seconds=%s)",
            self.name,
            job_id,
            len(variables),
            len(samples),
            timing_us.get("solve_time"),
            effective_time_limit,
        )
        return RawSolverResult(
            variables=variables,
            samples=samples,
            energies=energies,
            backend=self.name,
            metadata=metadata,
        )

    # -- internals -----------------------------------------------------------

    def _require_settings(self) -> tuple[str, str]:
        """Settings for a solve; the same checks as :meth:`is_available`."""
        key, base_url = _read_settings()
        if key is None:
            raise SolverExecutionError(
                REASON_API_KEY_MISSING, code="REMOTE_CREDENTIALS_MISSING"
            )
        if not base_url.startswith("https://"):
            raise SolverExecutionError(
                REASON_URL_NOT_HTTPS,
                code="BACKEND_CONFIG_INVALID",
                status="configuration_error",
            )
        return key, base_url

    def _request_json(
        self,
        what: str,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> Any:
        """One request → parsed JSON, or a classified, redacted error.

        Transport exceptions go through :func:`guarded_call` (no
        ``__cause__``); non-2xx statuses through the §20.8 table; a 2xx body
        that is not JSON is a solver error. The JSON error is raised outside
        its ``except`` block for the same reason as in ``guarded_call``.
        """
        status, raw = guarded_call(
            what,
            _classify_transport_exception,
            lambda: self._transport.request(
                method, url, headers, body, self._request_timeout
            ),
        )
        if not 200 <= status < 300:
            raise _http_failure(what, status, raw)
        error: SolverExecutionError | None = None
        try:
            return json.loads(_body_text(raw))
        except ValueError:
            error = SolverExecutionError(
                redact(f"{what}: response is not valid JSON: {_redacted_summary(_body_text(raw))}"),
                code=REMOTE_ERROR_FALLBACK_CODE,
            )
        raise error

    def _submit(self, base_url: str, headers: Mapping[str, str], body: bytes) -> str:
        """``POST .../qubo/solve`` → ``job_id`` (spec §20.7 step 3).

        A submit that times out is the one failure this backend cannot
        clean up after (review F-23): the request may have reached the
        vendor and created the job, but without a ``job_id`` there is
        nothing to cancel or delete, so the error tells the operator to
        check the account's job list. The rebuilt error is raised outside
        the ``except`` block for the same no-chain reason as
        :func:`guarded_call`.
        """
        what = "Fujitsu DA job submission failed"
        orphan_hint: SolverExecutionError | None = None
        try:
            payload = self._request_json(
                what, "POST", f"{base_url}/v4/async/qubo/solve", headers, body
            )
        except SolverExecutionError as error:
            if error.code != "REMOTE_TIMEOUT":
                raise
            orphan_hint = SolverExecutionError(
                f"{error}; the vendor may have created the job anyway, but no job_id "
                "was received so it cannot be released from here - check the "
                "account's job list",
                code=error.code,
                status=error.status,
            )
        if orphan_hint is not None:
            raise orphan_hint
        job_id = payload.get("job_id") if isinstance(payload, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise SolverExecutionError(
                redact(f"{what}: response has no job_id"),
                code=REMOTE_ERROR_FALLBACK_CODE,
            )
        logger.info("Backend %s submitted job %s", self.name, job_id)
        return job_id

    def _await_result(
        self,
        base_url: str,
        headers: Mapping[str, str],
        job_id: str,
        effective_time_limit: float,
    ) -> dict[str, Any]:
        """Poll ``GET .../jobs/result/{job_id}`` until ``Done`` (spec §20.7 step 4).

        Every way out of the loop other than a well-formed ``Done`` releases
        the vendor-side job on a best-effort basis before re-raising (code
        review 2026-09-08, F-09): a job left behind occupies one of the
        account's few slots, a paid side effect the user cannot see
        (OVERVIEW principle 5). What is sent depends on what is known:

        * a terminal status was read (``Done`` without a usable result,
          ``Error``, ``Canceled``, anything unexpected) → ``DELETE`` only;
        * the state is unknown (the poll request itself failed, the payload
          was unusable, or an interrupt arrived) → ``POST .../jobs/cancel``
          then ``DELETE``. Both are harmless whatever the job's state: the
          vendor cancels only a *Waiting* job and deletes only a *completed*
          one, answering 200 with the current status otherwise;
        * the polling budget ran out → the same cancel-then-DELETE (review
          F-23): a job already *Running* ignores the cancel and finishes
          on its own, and only the DELETE keeps its result from occupying
          a slot until someone removes it by hand.

        The budget (``effective_time_limit + POLL_GRACE_SECONDS``) is
        checked *before* every status request as well as after one that
        reports ``Waiting`` / ``Running``, so a request is never started
        once the budget is spent; a request already in flight is bounded
        only by the transport timeout (see the class docstring).

        The cleanup never replaces the error being raised: failures are
        logged (redacted) by :meth:`_best_effort`, and ``BaseException``
        (``KeyboardInterrupt``) is released too — a second interrupt during
        the cleanup request propagates, it is not swallowed.
        """
        what = f"Fujitsu DA job {job_id} status request failed"
        url = f"{base_url}/v4/async/jobs/result/{job_id}"
        budget = effective_time_limit + POLL_GRACE_SECONDS
        deadline = self._clock() + budget
        # What to send if the loop is left by an exception: ``"unknown"`` until a
        # terminal status has been read (a timeout therefore cancels *and*
        # deletes, review F-23), ``"terminal"`` once the job is known to have
        # stopped, ``None`` when nothing more should be sent (success).
        release: str | None = "unknown"

        def timed_out() -> SolverExecutionError:
            # Raised with ``release == "unknown"`` so the except block below
            # cancels and deletes; the message states the real bound.
            return SolverExecutionError(
                f"Fujitsu DA job {job_id} did not finish within the {budget:g} s "
                f"polling budget (time_limit_sec {effective_time_limit:g} plus "
                f"{POLL_GRACE_SECONDS:g} s grace; each HTTP request may add up to "
                f"{self._request_timeout:g} s); cancel and delete requests are sent "
                "on a best-effort basis",
                code="REMOTE_TIMEOUT",
            )

        try:
            while True:
                if self._clock() >= deadline:
                    raise timed_out()
                payload = self._request_json(what, "GET", url, headers, None)
                status = payload.get("status") if isinstance(payload, dict) else None
                if status in ("Waiting", "Running"):
                    if self._clock() >= deadline:
                        raise timed_out()
                    self._sleep(self._poll_interval)
                    continue
                if isinstance(status, str):
                    release = "terminal"
                if status == "Done":
                    qubo_solution = payload.get("qubo_solution")
                    if isinstance(qubo_solution, dict):
                        release = None
                        return qubo_solution
                    raise SolverExecutionError(
                        redact(f"Fujitsu DA job {job_id} is Done but has no qubo_solution"),
                        code=REMOTE_ERROR_FALLBACK_CODE,
                    )
                if status == "Error":
                    message = payload.get("message") or "no message reported"
                    raise SolverExecutionError(
                        redact(f"Fujitsu DA job {job_id} failed: {message}"),
                        code=REMOTE_ERROR_FALLBACK_CODE,
                    )
                raise SolverExecutionError(
                    redact(f"Fujitsu DA job {job_id} reported unexpected status {status!r}"),
                    code=REMOTE_ERROR_FALLBACK_CODE,
                )
        except BaseException:
            if release is not None:
                self._release_failed_job(base_url, headers, job_id, release)
            raise

    def _release_failed_job(
        self,
        base_url: str,
        headers: Mapping[str, str],
        job_id: str,
        release: str,
    ) -> None:
        """Best-effort cleanup of a job whose solve failed (F-09; see
        :meth:`_await_result`). ``release`` is ``"unknown"`` (cancel, then
        DELETE) or ``"terminal"`` (DELETE only). Failures are only logged, so
        the caller's exception is raised unchanged; the warnings say the job
        may still occupy a slot so the user can free it by hand."""
        if release == "unknown":
            self._best_effort(
                f"Fujitsu DA job {job_id} could not be cancelled after the solve "
                "failed; it may still occupy a job slot on the vendor side",
                "POST",
                f"{base_url}/v4/async/jobs/cancel",
                headers,
                json.dumps({"job_id": job_id}).encode("utf-8"),
            )
        self._best_effort(
            f"Fujitsu DA job {job_id} result could not be deleted after the solve "
            "failed; the job may still occupy a job slot on the vendor side",
            "DELETE",
            f"{base_url}/v4/async/jobs/result/{job_id}",
            headers,
            None,
        )

    def _best_effort(
        self,
        what: str,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> None:
        """DELETE / cancel: failures are logged (redacted) and never raised."""
        try:
            status, raw = self._transport.request(
                method, url, headers, body, self._request_timeout
            )
        except Exception as exc:
            logger.warning("%s", redact(f"{what}: {type(exc).__name__}: {exc}"))
            return
        if not 200 <= status < 300:
            logger.warning(
                "%s", redact(f"{what}: HTTP {status}: {_redacted_summary(_body_text(raw))}")
            )
