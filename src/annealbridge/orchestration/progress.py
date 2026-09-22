"""Progress events a solve emits while it runs.

``OptimizationService.solve`` takes an optional ``on_progress`` callback and
calls it with one :class:`SolveProgress` at the start of each stage of each
attempt: ``compile`` (which is also "attempt *n* of *N* begins"), ``solve``
and ``validate``. Nothing finer — never per read — and nothing here names a
configuration value or a limit, so a host can show the message as is.

The event is a frozen dataclass rather than a pydantic model: it is never
serialised by the core, only handed to a callback, and an interface (the MCP
server) turns it into whatever its protocol carries. This module imports
nothing outside the standard library, so the orchestration package stays free
of any interface dependency.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

__all__ = ["STAGES", "ProgressCallback", "SolveProgress", "Stage"]

Stage = Literal["compile", "solve", "validate"]

# In execution order; ``step`` and ``total_steps`` count in these units.
STAGES: tuple[Stage, ...] = ("compile", "solve", "validate")

_VERBS: dict[str, str] = {
    "compile": "compiling",
    "solve": "solving",
    "validate": "validating",
}


@dataclass(frozen=True)
class SolveProgress:
    """One stage of one attempt is about to start."""

    attempt: int
    """1-based attempt number."""

    max_attempts: int
    """The attempt budget for this solve (``1 + max_retries`` at most)."""

    stage: Stage
    """Which of compile / solve / validate is starting."""

    backend: str
    """The backend's own name (``SolverBackend.name``)."""

    @property
    def step(self) -> int:
        """Stages completed before this one, across all attempts so far."""
        return (self.attempt - 1) * len(STAGES) + STAGES.index(self.stage)

    @property
    def total_steps(self) -> int:
        """Stages the solve would run if it used its whole attempt budget."""
        return self.max_attempts * len(STAGES)

    @property
    def message(self) -> str:
        """A deterministic one-line status, e.g. ``attempt 2 of 3: solving on
        simulated_annealing``."""
        return (
            f"attempt {self.attempt} of {self.max_attempts}: "
            f"{_VERBS[self.stage]} on {self.backend}"
        )


ProgressCallback = Callable[[SolveProgress], None]
"""What ``solve(on_progress=...)`` accepts. It is called on the thread that
runs the solve; an exception it raises is logged and otherwise ignored, so it
can never change the result."""
