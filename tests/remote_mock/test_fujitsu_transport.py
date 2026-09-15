"""Loopback tests for the Fujitsu DA HTTP transport (Phase 3b spec §20.4, §20.8).

:class:`~tests.fakes.FakeDATransport` replaces the network seam in every
other Fujitsu test, so nothing there proves that the real
:class:`~annealbridge.solvers.fujitsu_da.UrllibTransport` keeps the seam's
contract. These tests drive it against a ``ThreadingHTTPServer`` bound to
``127.0.0.1`` — never an external address, never quota, hence no ``remote``
marker — and exercise the transport and the transport-exception classifier
directly, because the backend itself refuses a non-HTTPS base URL.

Contract checked here:

* every HTTP response, error statuses included, returns ``(status, body)``;
* the wire request carries the method, path and body given, and the headers
  given alongside the ones urllib adds itself;
* a read timeout raises a bare ``TimeoutError``, a connect timeout a
  ``URLError`` wrapping one, and both classify as ``REMOTE_TIMEOUT``;
* a refused connection is a ``URLError`` that classifies as the retryable
  ``REMOTE_SOLVER_ERROR`` fallback;
* through ``guarded_call`` a timeout leaves the solver layer as a
  ``SolverExecutionError`` with no exception chain.

The classifier is always read as a module attribute
(``fujitsu_da._classify_transport_exception``), never bound by a
``from ... import``, so a replacement of the module attribute is seen here.
"""

import json
import socket
import sys
import threading
import time
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple

import pytest

import annealbridge.solvers.fujitsu_da as fujitsu_da
from annealbridge.exceptions import SolverExecutionError
from annealbridge.solvers.metadata import REMOTE_ERROR_FALLBACK_CODE, guarded_call
from tests.remote_mock.test_fujitsu_da_mock import FAKE_KEY

SUBMIT_PATH = "/v4/async/qubo/solve"
RESULT_PATH = "/v4/async/jobs/result/fake-job"
SLOW_PATH = "/v4/async/jobs/result/slow-job"

# Upper bound for a delayed handler's wait. Teardown sets ``release`` first,
# so the cap costs no time; it only keeps ``server_close()``, which joins
# handler threads without a timeout, from hanging should that release be
# skipped. It sits well above SHORT_TIMEOUT, so a slow machine cannot answer
# before the client gives up, and below ELAPSED_CEILING.
HANDLER_DELAY_CAP = 3.0
# Short transport timeout used to provoke read / connect timeouts.
SHORT_TIMEOUT = 0.2
# Loose ceiling for "the timeout actually fired": only a hang fails it.
ELAPSED_CEILING = 5.0

_PROXY_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def backend_headers() -> dict[str, str]:
    """The header set ``FujitsuDABackend`` sends, with a fake key."""
    return {
        "X-Api-Key": FAKE_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


@dataclass(frozen=True)
class Reply:
    """Scripted response for one path."""

    status: int = 200
    body: bytes = b"{}"
    # Wait on ``server.release`` (capped) before answering.
    delay: bool = False


class RecordedRequest(NamedTuple):
    method: str
    path: str
    # ``email.message.Message`` looks header names up case-insensitively.
    headers: Message
    body: bytes


class _Handler(BaseHTTPRequestHandler):
    server: "LoopbackServer"

    # Socket timeout while reading a request: a client that connects but never
    # finishes one cannot keep ``server_close()``'s join waiting forever.
    timeout = 5

    def log_message(self, format: str, *args: object) -> None:
        """Silence the per-request access log on stderr."""

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append(
            RecordedRequest(self.command, self.path, self.headers, body)
        )
        reply = self.server.routes.get(self.path, Reply())
        if reply.delay:
            self.server.release.wait(HANDLER_DELAY_CAP)
        self.send_response(reply.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply.body)))
        self.end_headers()
        self.wfile.write(reply.body)

    do_GET = _serve
    do_POST = _serve
    do_DELETE = _serve


class LoopbackServer(ThreadingHTTPServer):
    """Scripted HTTP server on an ephemeral ``127.0.0.1`` port.

    ``daemon_threads = False`` makes ``server_close()`` join every handler
    thread, so a handler still running when a test ends cannot fail a later
    test. ``handle_error`` swallows only a ``ConnectionError``: once a client
    has timed out, the delayed handler's write may hit the closed connection
    (``ConnectionAbortedError`` on Windows, ``BrokenPipeError`` on Linux),
    which is expected and must not print. Anything else is re-raised, leaves
    the handler thread and fails the test through pytest's unhandled thread
    exception check.
    """

    daemon_threads = False

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.routes: dict[str, Reply] = {}
        self.release = threading.Event()
        super().__init__(("127.0.0.1", 0), _Handler)

    def handle_error(self, request: object, client_address: object) -> None:
        # Called from inside socketserver's ``except Exception`` block, so a
        # bare ``raise`` re-raises the handler's own exception.
        if not isinstance(sys.exc_info()[1], ConnectionError):
            raise

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}{path}"


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """Keep every request on loopback, whatever proxy the machine configures.

    ``urlopen`` caches a module-level opener whose ``ProxyHandler`` proxy
    list is fixed when it is built (on Windows it also reads the registry
    proxy), so deleting the proxy variables alone does not reliably help.
    ``proxy_bypass`` re-reads ``NO_PROXY`` / ``no_proxy`` on every request,
    so setting both spellings bypasses any cached proxy for 127.0.0.1.
    ``urllib.request.install_opener`` is avoided: it is a global change that
    monkeypatch would not undo.
    """
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture
def loopback_server() -> Iterator[LoopbackServer]:
    server = LoopbackServer()
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        name="fujitsu-transport-loopback",
        daemon=True,
    )
    try:
        thread.start()
        yield server
    finally:
        server.release.set()  # wake any delayed handler before joining it
        if thread.is_alive():  # never started: nothing to stop or join
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()
    assert not thread.is_alive()


@pytest.fixture
def saturated_listener() -> Iterator[int]:
    """A port whose accept queue is full, so a new connect attempt times out.

    ``listen(0)`` plus one connected, never-accepted client fills the
    backlog. Linux then drops the next SYN, so the connect times out however
    long the timeout is. Windows answers it with a reset instead and retries
    for about two seconds before reporting a refusal, so there the short
    timeout expires during those retries. Either way the transport sees a
    connect timeout, which is what the test needs.
    """
    listener = socket.socket()
    filler = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(0)
        filler.settimeout(2.0)
        filler.connect(listener.getsockname())
        yield listener.getsockname()[1]
    finally:
        filler.close()
        listener.close()


@pytest.fixture
def closed_port() -> Iterator[int]:
    """A loopback port held bound for the whole test but never listened on.

    Holding it keeps the port from being taken by anything else, or picked as
    the client's own ephemeral port (a TCP self-connect on Linux), so a
    connection to it is always refused.
    """
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        yield holder.getsockname()[1]


def test_post_returns_status_and_body(loopback_server):
    loopback_server.routes[SUBMIT_PATH] = Reply(200, b'{"job_id": "fake-job"}')

    status, body = fujitsu_da.UrllibTransport().request(
        "POST",
        loopback_server.url(SUBMIT_PATH),
        backend_headers(),
        json.dumps({"fujitsuDA3": {"time_limit_sec": 1}}).encode("utf-8"),
        2.0,
    )

    assert type(status) is int
    assert type(body) is bytes
    assert (status, body) == (200, b'{"job_id": "fake-job"}')


@pytest.mark.parametrize("status", [400, 429, 500])
def test_http_error_status_is_returned_not_raised(loopback_server, status):
    error_body = b'{"error":"fake"}'
    loopback_server.routes[SUBMIT_PATH] = Reply(status, error_body)

    result = fujitsu_da.UrllibTransport().request(
        "POST",
        loopback_server.url(SUBMIT_PATH),
        backend_headers(),
        b'{"fake": true}',
        2.0,
    )

    assert result == (status, error_body)
    assert type(result[0]) is int
    assert type(result[1]) is bytes


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", SUBMIT_PATH, json.dumps({"fake": [1, 2, 3]}).encode("utf-8")),
        ("GET", RESULT_PATH, None),
        ("DELETE", RESULT_PATH, None),
    ],
    ids=["post", "get", "delete"],
)
def test_wire_request_matches_arguments(loopback_server, method, path, body):
    status, _ = fujitsu_da.UrllibTransport().request(
        method, loopback_server.url(path), backend_headers(), body, 2.0
    )

    assert status == 200
    assert len(loopback_server.requests) == 1
    seen = loopback_server.requests[0]
    assert seen.method == method
    assert seen.path == path
    assert seen.body == (body if body is not None else b"")
    assert seen.headers["x-api-key"] == FAKE_KEY
    assert seen.headers["content-type"] == "application/json"
    assert seen.headers["accept"] == "application/json"


def test_read_timeout_is_bare_timeout_error(loopback_server):
    loopback_server.routes[SLOW_PATH] = Reply(delay=True)

    started = time.perf_counter()
    with pytest.raises(TimeoutError) as info:
        fujitsu_da.UrllibTransport().request(
            "GET", loopback_server.url(SLOW_PATH), backend_headers(), None, SHORT_TIMEOUT
        )
    elapsed = time.perf_counter() - started

    assert not isinstance(info.value, urllib.error.URLError)
    assert fujitsu_da._classify_transport_exception(info.value) == "REMOTE_TIMEOUT"
    assert elapsed < ELAPSED_CEILING


def test_connect_timeout_is_url_error_wrapping_timeout(saturated_listener):
    started = time.perf_counter()
    with pytest.raises(urllib.error.URLError) as info:
        fujitsu_da.UrllibTransport().request(
            "GET",
            f"http://127.0.0.1:{saturated_listener}{RESULT_PATH}",
            backend_headers(),
            None,
            SHORT_TIMEOUT,
        )
    elapsed = time.perf_counter() - started

    assert isinstance(info.value.reason, TimeoutError)
    assert fujitsu_da._classify_transport_exception(info.value) == "REMOTE_TIMEOUT"
    assert elapsed < ELAPSED_CEILING


def test_refused_connection_is_remote_solver_error(closed_port):
    # Windows retries a refused loopback connect for about 2 s before
    # reporting it, so a short timeout would turn this into a connect
    # timeout; Linux refuses immediately. 5 s leaves room for both.
    with pytest.raises(urllib.error.URLError) as info:
        fujitsu_da.UrllibTransport().request(
            "GET",
            f"http://127.0.0.1:{closed_port}{RESULT_PATH}",
            backend_headers(),
            None,
            5.0,
        )

    assert isinstance(info.value.reason, ConnectionRefusedError)
    code = fujitsu_da._classify_transport_exception(info.value)
    assert code == "REMOTE_SOLVER_ERROR"
    assert code == REMOTE_ERROR_FALLBACK_CODE


def test_guarded_read_timeout_is_unchained_remote_timeout(loopback_server):
    loopback_server.routes[SLOW_PATH] = Reply(delay=True)
    transport = fujitsu_da.UrllibTransport()

    with pytest.raises(SolverExecutionError) as info:
        guarded_call(
            "Fujitsu DA result poll failed",
            fujitsu_da._classify_transport_exception,
            lambda: transport.request(
                "GET",
                loopback_server.url(SLOW_PATH),
                backend_headers(),
                None,
                SHORT_TIMEOUT,
            ),
        )

    error = info.value
    assert error.code == "REMOTE_TIMEOUT"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert str(error).startswith("Fujitsu DA result poll failed: TimeoutError")
