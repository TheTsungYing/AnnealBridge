"""Helpers shared by the D-Wave (Ocean) backends (Phase 3a spec §17.5).

``dwave_qpu``, ``leap_hybrid_bqm`` and ``leap_hybrid_cqm`` all need the
same four things: a guarded call that turns any Ocean failure into a
redacted, classified :class:`SolverExecutionError`; a way to force a lazy
sampleset to resolve inside that guard; a lazily built, cached sampler;
and the classification tables for sampler construction and hybrid
sampling. They live here once instead of once per backend.

This module never imports ``dwave.*`` — not even lazily. The backends
lazy-import ``dwave.system`` inside their own default sampler factories,
so the default registry (and this module) import cleanly without the
``dwave`` extra installed (spec §4).
"""

import threading
from typing import Any, Callable, TypeVar

from annealbridge.exceptions import SolverExecutionError
from annealbridge.solvers.metadata import classify_exception, redact

__all__ = [
    "HYBRID_SAMPLE_EXCEPTION_CODES",
    "SAMPLER_INIT_EXCEPTION_CODES",
    "LazySampler",
    "call_ocean",
    "resolved",
]

_T = TypeVar("_T")

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
    """Run ``fn`` and convert any failure into a redacted SolverExecutionError.

    The wrapped error is raised *after* the ``except`` block has finished,
    so it carries neither ``__cause__`` nor ``__context__``: the original
    exception (whose text may embed credentials) is not reachable from the
    error that leaves the solver layer, and ``traceback.format_exception``
    / ``logger.exception`` cannot print it (Phase 2 spec §19). The original
    class name is kept in the message because it is categorical, not
    secret.
    """
    try:
        return fn()
    except Exception as exc:
        error = SolverExecutionError(
            redact(f"{what}: {type(exc).__name__}: {exc}"),
            code=classify_exception(exc, codes),
        )
    raise error


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
