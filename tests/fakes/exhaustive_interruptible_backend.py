"""An exhaustive local backend that also declares ``supports_interrupt``.

2026-11-18 review R-18. No shipped backend is both exhaustive and
interruptible, so the service's proof flags were never exercised on an
enumeration the wall-clock limit cut short. ``ScriptedExhaustiveBackend``
reuses :class:`ScriptedInterruptBackend`'s fixed rows and per-call script,
and only changes the declaration: ``exhaustive=True`` and
``supports_interrupt=True``, with ``is_exhaustive`` returning ``True``. A
``partial`` or ``empty`` script action therefore stands for an enumeration
stopped part-way, which must not be reported as a proof.

``SolverCapabilities`` refuses that combination at declaration
(``_exhaustive_is_never_interrupted``), so the capabilities are a valid
declaration updated with ``model_copy``, which skips the model validators:
the fake exercises the service's own guard, the second line of defence
behind the declaration rule.
"""

from annealbridge.solvers.base import SolverCapabilities
from tests.fakes.interruptible_backend import Script, ScriptedInterruptBackend

SCRIPTED_EXHAUSTIVE_NAME = "scripted_exhaustive_interrupt"


class ScriptedExhaustiveBackend(ScriptedInterruptBackend):
    """``ScriptedInterruptBackend`` declared exhaustive and interruptible."""

    def __init__(
        self,
        rows: list[dict[str, int]],
        *,
        name: str = SCRIPTED_EXHAUSTIVE_NAME,
        script: Script | None = None,
    ) -> None:
        super().__init__(rows, supports_interrupt=True, name=name, script=script)
        declared = SolverCapabilities(
            name=name,
            remote=False,
            heuristic=False,
            exhaustive=True,
            supports_seed=False,
            supports_num_reads=False,
            supports_time_limit=False,
            supported_model_types=["bqm"],
            returns_multiple_samples=True,
            supports_interrupt=False,
            description=(
                "test-only: scripted rows declared exhaustive and interruptible "
                "for the proof-flag tests"
            ),
        )
        # ``model_copy`` does not re-run the model validators.
        self.capabilities = declared.model_copy(update={"supports_interrupt": True})
        assert self.capabilities.exhaustive and self.capabilities.supports_interrupt

    @property
    def is_exhaustive(self) -> bool:
        return True
