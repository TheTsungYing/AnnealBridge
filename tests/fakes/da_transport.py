"""Scripted Fujitsu Digital Annealer transport for tests (3b spec §26.5).

``FakeDATransport`` stands in for the backend's :class:`HttpTransport` and
behaves like a tiny DA server: ``POST .../qubo/solve`` returns a job id,
each ``GET .../jobs/result/{id}`` walks through ``poll_statuses`` (default
Waiting → Running → Done), ``Done`` carries the configured solutions,
``DELETE`` and ``POST .../jobs/cancel`` answer 200. Any step can be
overridden with a canned ``(status, body)`` response or an exception to
raise, and every request is recorded (method, url, headers, body) so tests
assert on the exact wire shape. A new ``POST`` resets the poll script, so
one instance serves a multi-attempt service solve unchanged.

Test-only; never shipped.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FAKE_JOB_ID",
    "FAKE_TIMING",
    "FakeDATransport",
    "FakeSolution",
    "RecordedRequest",
    "bits_solution",
    "json_response",
]

FAKE_JOB_ID = "contract-a001-1234-5678-fake-000000000001"
FAKE_TIMING = {"solve_time": "5041", "total_elapsed_time": "5050"}

# What a scripted step yields: a canned HTTP response or an exception to raise.
Response = tuple[int, bytes] | Exception


def json_response(status: int, payload: Any) -> tuple[int, bytes]:
    """``(status, body)`` for a JSON payload, as the real transport returns it."""
    return status, json.dumps(payload).encode("utf-8")


@dataclass
class RecordedRequest:
    """One request the backend made, exactly as the transport saw it."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    timeout: float

    @property
    def json(self) -> Any:
        """The body parsed as JSON (None when there was no body)."""
        return None if self.body is None else json.loads(self.body.decode("utf-8"))


@dataclass
class FakeSolution:
    """One entry of ``qubo_solution.solutions`` as the vendor returns it."""

    configuration: dict[str, Any]
    energy: float
    frequency: float = 1.0
    penalty_energy: float = 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "configuration": dict(self.configuration),
            "energy": self.energy,
            "frequency": self.frequency,
            "penalty_energy": self.penalty_energy,
        }


def bits_solution(bits: Sequence[int], energy: float, frequency: float = 1.0) -> FakeSolution:
    """A well-formed solution from a 0/1 row (index ``i`` → ``configuration["i"]``)."""
    return FakeSolution(
        configuration={str(index): bool(bit) for index, bit in enumerate(bits)},
        energy=energy,
        frequency=frequency,
    )


@dataclass
class FakeDATransport:
    """Scripted DA server behind the backend's transport seam.

    ``solutions`` is what a ``Done`` poll returns; ``poll_statuses`` the
    status sequence successive polls report (the last one repeats). Set
    ``submit_response`` / ``delete_response`` / ``cancel_response`` /
    ``result_response`` (the ``Done`` body) or ``poll_responses[k]`` (the
    ``k``-th poll of the current job) to a ``(status, body)`` tuple or an
    exception to script a failure at that step.
    """

    solutions: list[FakeSolution] = field(default_factory=list)
    poll_statuses: Sequence[str] = ("Waiting", "Running", "Done")
    job_id: str = FAKE_JOB_ID
    timing: dict[str, Any] | None = field(default_factory=lambda: dict(FAKE_TIMING))
    error_message: str = "the number of variables exceeds the limit (100000)"
    submit_response: Response | None = None
    result_response: Response | None = None
    poll_responses: dict[int, Response] = field(default_factory=dict)
    delete_response: Response | None = None
    cancel_response: Response | None = None

    requests: list[RecordedRequest] = field(default_factory=list, init=False)
    submit_calls: int = field(default=0, init=False)
    poll_calls: int = field(default=0, init=False)
    delete_calls: int = field(default=0, init=False)
    cancel_calls: int = field(default=0, init=False)
    _poll_index: int = field(default=0, init=False, repr=False)

    # -- HttpTransport -------------------------------------------------------

    def request(
        self,
        method: str,
        url: str,
        headers: Any,
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        self.requests.append(
            RecordedRequest(
                method=method,
                url=url,
                headers=dict(headers),
                body=body,
                timeout=timeout,
            )
        )
        path = url.split("://", 1)[-1].split("/", 1)[-1] if "://" in url else url
        if method == "POST" and path.endswith("v4/async/qubo/solve"):
            return self._submit()
        if method == "GET" and "v4/async/jobs/result/" in path:
            return self._poll()
        if method == "DELETE" and "v4/async/jobs/result/" in path:
            return self._delete()
        if method == "POST" and path.endswith("v4/async/jobs/cancel"):
            return self._cancel()
        return json_response(404, {"message": "Resource not found"})

    # -- steps ---------------------------------------------------------------

    @staticmethod
    def _deliver(response: Response) -> tuple[int, bytes]:
        if isinstance(response, BaseException):
            raise response
        return response

    def _submit(self) -> tuple[int, bytes]:
        self.submit_calls += 1
        self._poll_index = 0
        if self.submit_response is not None:
            return self._deliver(self.submit_response)
        return json_response(200, {"job_id": self.job_id})

    def _poll(self) -> tuple[int, bytes]:
        self.poll_calls += 1
        index = self._poll_index
        self._poll_index += 1
        if index in self.poll_responses:
            return self._deliver(self.poll_responses[index])
        statuses = list(self.poll_statuses)
        status = statuses[min(index, len(statuses) - 1)]
        if status == "Done":
            if self.result_response is not None:
                return self._deliver(self.result_response)
            return json_response(200, self.done_payload())
        if status == "Error":
            return json_response(200, {"message": self.error_message, "status": "Error"})
        return json_response(200, {"status": status})

    def _delete(self) -> tuple[int, bytes]:
        self.delete_calls += 1
        if self.delete_response is not None:
            return self._deliver(self.delete_response)
        return json_response(200, {"job_id": self.job_id, "status": "Deleted"})

    def _cancel(self) -> tuple[int, bytes]:
        self.cancel_calls += 1
        if self.cancel_response is not None:
            return self._deliver(self.cancel_response)
        return json_response(200, {"job_id": self.job_id, "status": "Canceled"})

    # -- helpers for assertions ---------------------------------------------

    def done_payload(self) -> dict[str, Any]:
        """The body a successful ``Done`` poll returns (vendor shape)."""
        qubo_solution: dict[str, Any] = {
            "progress": [],
            "result_status": True,
            "solutions": [solution.as_json() for solution in self.solutions],
        }
        if self.timing is not None:
            qubo_solution["timing"] = dict(self.timing)
        return {"qubo_solution": qubo_solution, "status": "Done"}

    def requests_for(self, method: str | None = None, path_part: str | None = None) -> list[RecordedRequest]:
        """Recorded requests filtered by method and/or a URL substring."""
        return [
            recorded
            for recorded in self.requests
            if (method is None or recorded.method == method)
            and (path_part is None or path_part in recorded.url)
        ]

    @property
    def submit_request(self) -> RecordedRequest:
        """The most recent ``POST .../qubo/solve`` request."""
        return self.requests_for("POST", "v4/async/qubo/solve")[-1]
