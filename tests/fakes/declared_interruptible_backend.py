"""``FakeInterruptibleBackend`` -- the fifth backend, declaring ``supports_interrupt``.

Batch 6 (J), time-limit spec 2026-09-24 §5 and §9 item 8: a backend the
core has never heard of joins the wall-clock limit and cancellation purely
by declaring ``supports_interrupt=True`` and taking a keyword-only
``interrupt`` in ``solve``. It is :class:`FakeDeclaredBackend` (same name,
same custom limit key, same credential declaration) with only those two
things added, so the architecture test can show that flipping the one
flag is all a backend has to do.

What it records:

* ``interrupts`` -- one entry per ``solve`` call: the ``Interrupt`` the
  service passed, or :data:`NOT_PASSED` when the keyword was absent;
* when the interrupt already says stop, it follows the backend contract
  (``SolverBackend.solve``): no exception, zero completed reads, and
  ``RawSolverResult.interrupted=True``.

Test-only; never shipped.
"""

from annealbridge.interrupt import Interrupt
from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import RawSolverResult, SolverCapabilities, empty_result
from tests.fakes.declared_backend import _CAPABILITIES, FakeDeclaredBackend

__all__ = ["NOT_PASSED", "FakeInterruptibleBackend"]

NOT_PASSED = object()
"""Recorded in ``interrupts`` for a call made without the keyword."""

_INTERRUPTIBLE_CAPABILITIES: SolverCapabilities = _CAPABILITIES.model_copy(
    update={"supports_interrupt": True}
)


class FakeInterruptibleBackend(FakeDeclaredBackend):
    """The declared fifth backend, able to stop part-way."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.interrupts: list[object] = []

    @property
    def capabilities(self) -> SolverCapabilities:
        return _INTERRUPTIBLE_CAPABILITIES

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
        *,
        interrupt: object = NOT_PASSED,
    ) -> RawSolverResult:
        self.interrupts.append(interrupt)
        if isinstance(interrupt, Interrupt) and interrupt.should_stop():
            self.solve_calls += 1
            self.last_preferences = preferences
            return empty_result(
                compiled_problem.model, backend=self.name, metadata=None
            )
        return super().solve(compiled_problem, preferences)
