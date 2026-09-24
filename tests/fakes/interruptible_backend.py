"""A scripted local backend and a fake clock for the wall-clock limit tests.

Batch 6 (J), time-limit spec 2026-09-24 §9. ``ScriptedInterruptBackend`` is
a non-exhaustive local heuristic that returns fixed business rows and
records the keyword arguments of every ``solve`` call, so a test can prove
exactly when the service hands a backend ``interrupt=`` and when it does
not. Whether it declares ``supports_interrupt`` is a constructor argument;
its ``solve`` takes ``**kwargs`` either way, so the service's construction
check accepts both and the recorded calls show what the service decided.

A ``script`` decides, per call, what the backend does -- return every row,
return them marked interrupted, or return zero rows marked interrupted --
and may move a :class:`FakeClock` or cancel a token first, which is how the
deterministic tests put the service past its deadline at a chosen point.
"""

from collections.abc import Callable
from typing import Any, Literal

from annealbridge.interrupt import Interrupt
from annealbridge.models import (
    CompiledProblem,
    OptimizationProblem,
    SolverPreferences,
)
from annealbridge.solvers.base import (
    AvailabilityStatus,
    RawSolverResult,
    SolverCapabilities,
    empty_result,
)

SCRIPTED_INTERRUPT_NAME = "scripted_interrupt"

Action = Literal["full", "partial", "empty"]
"""``full``: every row, not interrupted. ``partial``: every row, marked
interrupted (a backend that skipped some reads). ``empty``: zero rows,
marked interrupted (stopped before any read completed)."""

Script = Callable[[int, Interrupt | None], Action]
"""Called with the 1-based call number and the ``interrupt`` received."""


class FakeClock:
    """A monotonic clock in seconds that only moves when a test moves it."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedInterruptBackend:
    """Implements ``SolverBackend`` with fixed rows and a per-call script.

    Each row names business variables; every other compiled variable
    (slack) is returned as 0, so a row feasible for the business problem
    stays feasible after the compiler's ``decode``.
    """

    def __init__(
        self,
        rows: list[dict[str, int]],
        *,
        supports_interrupt: bool = True,
        name: str = SCRIPTED_INTERRUPT_NAME,
        script: Script | None = None,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.script = script
        self.calls: list[dict[str, Any]] = []
        self.capabilities = SolverCapabilities(
            name=name,
            remote=False,
            heuristic=True,
            exhaustive=False,
            supports_seed=False,
            supports_num_reads=False,
            supports_time_limit=False,
            supported_model_types=["bqm"],
            returns_multiple_samples=True,
            supports_interrupt=supports_interrupt,
            description="test-only: scripted rows for the wall-clock limit tests",
        )

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return False

    def is_available(self) -> AvailabilityStatus:
        return AvailabilityStatus(category="available")

    def resolve_time_limit(self, compiled, preferences):
        return None

    @property
    def received_interrupt(self) -> list[bool]:
        """Per call: whether the ``interrupt`` keyword was passed at all."""
        return ["interrupt" in call for call in self.calls]

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
        **kwargs: Any,
    ) -> RawSolverResult:
        self.calls.append(dict(kwargs))
        interrupt = kwargs.get("interrupt")
        action: Action = (
            "full" if self.script is None else self.script(len(self.calls), interrupt)
        )
        model = compiled_problem.model
        if action == "empty":
            return empty_result(model, backend=self.name, metadata=None)
        variables = [str(variable) for variable in model.variables]
        rows = [
            {variable: int(row.get(variable, 0)) for variable in variables}
            for row in self.rows
        ]
        result = RawSolverResult.from_dicts(
            rows,
            [float(model.energy(row)) for row in rows],
            backend=self.name,
            variables=variables,
        )
        result.interrupted = action == "partial"
        return result


def route(problem: OptimizationProblem, backend: str, **preferences) -> OptimizationProblem:
    """``problem`` pointed at ``backend`` (possibly a test-only name).

    ``SolverPreferences.backend`` is a Literal of the shipped names, so the
    preferences are built with ``model_construct`` (defaults filled in).
    """
    solver = SolverPreferences.model_construct(backend=backend, **preferences)
    return problem.model_copy(update={"solver": solver})
