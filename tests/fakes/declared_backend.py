"""``FakeDeclaredBackend`` — the "fifth backend" of Phase 3a spec §13.1.

A remote-looking backend that the core has never heard of. Everything the
service, the validator and the capabilities view need to know about it
comes from its ``SolverCapabilities`` declaration:

* it declares a policy-limited parameter under a **custom** limit key
  (``num_reads`` → ``iterations``), so a policy must supply
  ``limits={"iterations": ...}`` — this proves the generic ``limits``
  channel of spec §11 works end to end;
* ``FAKE_ITERATIONS_LIMIT`` is deliberately *not* in the error catalog, so
  the test also pins ``catalog_error``'s behaviour for unknown codes.

Registering it must require zero changes to ``orchestration/*``,
``validation/*``, ``interfaces/capabilities.py`` or ``policy.py``; the
architecture test is the proof.
"""

from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import (
    AvailabilityStatus,
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
)

FAKE_DECLARED_NAME = "fake_declared"
FAKE_LIMIT_KEY = "iterations"
FAKE_LIMIT_ERROR_CODE = "FAKE_ITERATIONS_LIMIT"

# Feasible for examples/knapsack.json: weight 6 + 4 = 10 <= 10, value 17
# (the global optimum). Any compiled variable not named here — slack bits
# included — is returned as 0.
DEFAULT_ASSIGNMENT: dict[str, int] = {"item_a": 1, "item_c": 1}

_CAPABILITIES = SolverCapabilities(
    name=FAKE_DECLARED_NAME,
    remote=True,
    heuristic=True,
    exhaustive=False,
    supports_seed=False,
    supports_num_reads=True,
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    description="Test-only declared backend (Phase 3a spec §13.1); never production.",
    parameter_limits=[
        ParameterLimit(
            preference="num_reads",
            limit=FAKE_LIMIT_KEY,
            error_code=FAKE_LIMIT_ERROR_CODE,
        )
    ],
)


class FakeDeclaredBackend:
    """Implements the ``SolverBackend`` protocol with a fixed feasible answer.

    ``assignment`` is a business assignment widened over every compiled
    variable (slack = 0), so the returned sample is feasible for the
    problem the assignment was written for. Records every ``solve`` call
    so a test can prove a refused solve never reached the backend.
    """

    def __init__(self, assignment: dict[str, int] | None = None) -> None:
        self._assignment = dict(DEFAULT_ASSIGNMENT if assignment is None else assignment)
        self.solve_calls = 0
        self.last_preferences: SolverPreferences | None = None

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        """Always available; no network I/O."""
        return AvailabilityStatus(category="available")

    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        return self.capabilities.exhaustive

    def resolve_time_limit(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> float | None:
        """No time limit support: always None."""
        return None

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        self.solve_calls += 1
        self.last_preferences = preferences
        bqm = compiled_problem.model
        variables = [str(variable) for variable in bqm.variables]
        row = {variable: int(self._assignment.get(variable, 0)) for variable in variables}
        return RawSolverResult.from_dicts(
            [row],
            [float(bqm.energy(row))],
            backend=self.name,
            variables=variables,
        )
