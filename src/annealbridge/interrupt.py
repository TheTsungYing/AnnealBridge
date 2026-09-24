"""Stopping a solve early: the wall-clock limit and caller cancellation.

Two things can ask a running solve to stop before it has done all the work
it was asked for: ``solver.wall_clock_limit_seconds`` running out, and the
caller cancelling (an MCP client that sent ``notifications/cancelled``, or
a library caller holding a :class:`CancelToken`). Both reach the service
and the backends as one :class:`Interrupt`, polled at checkpoints; nothing
is ever stopped from the outside, so every thread a solve with an
``Interrupt`` started has returned by the time the solve returns or raises
-- a failed shard included, see ``solvers.sharding.run_shards_interruptible``.

This module sits beside ``exceptions`` at the top of the package and uses
the standard library only, so every core layer -- solvers, orchestration --
can import it without depending on anything above it.
"""

import threading
import time
from collections.abc import Callable

__all__ = ["CancelToken", "Clock", "Interrupt"]

Clock = Callable[[], float]
"""A monotonic clock in seconds; ``time.perf_counter`` unless a test injects
one. ``OptimizationService`` reads the same clock for the start of a solve
and for its deadline, so the limit is measured from the same instant
``elapsed_ms`` is."""


class CancelToken:
    """A one-way flag a caller sets to cancel a solve from another thread.

    ``OptimizationService.solve(problem, cancel=token)`` polls it at its
    checkpoints; once ``cancel()`` has been called the solve stops at the
    next one and raises :class:`~annealbridge.exceptions.SolveCancelled`.
    Cancelling is permanent and idempotent: a token cannot be reset, and a
    token that is already cancelled makes a solve stop at its first
    checkpoint.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Ask every solve holding this token to stop."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether :meth:`cancel` has been called."""
        return self._event.is_set()


class Interrupt:
    """The stop condition of one solve: a deadline, a cancel token, or both.

    Built by ``OptimizationService.solve`` only when the problem sets a
    wall-clock limit or the caller passed a token, and handed to a backend
    only when its capabilities declare ``supports_interrupt``; without
    either, no ``Interrupt`` exists and the solve runs exactly as before.

    :meth:`should_stop` is the one question a checkpoint asks. It has no
    side effect, never raises and may be called from any number of threads
    at once (a shard pool polls it concurrently), so a backend can hand it
    straight to a vendor callback.
    """

    def __init__(
        self,
        *,
        deadline: float | None,
        token: CancelToken | None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self._deadline = deadline
        self._token = token
        self._clock = clock

    @property
    def cancelled(self) -> bool:
        """The caller cancelled the solve."""
        return self._token is not None and self._token.cancelled

    def deadline_passed(self) -> bool:
        """The wall-clock limit has run out."""
        return self._deadline is not None and self._clock() >= self._deadline

    def should_stop(self) -> bool:
        """Stop now: the solve was cancelled or its deadline has passed."""
        return self.cancelled or self.deadline_passed()
