"""Exact (exhaustive) solver backend (spec §22)."""

import logging

import dimod

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import (
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_lists,
)

logger = logging.getLogger(__name__)

_CAPABILITIES = SolverCapabilities(
    name="exact",
    remote=False,
    heuristic=False,
    exhaustive=True,
    supports_seed=False,
    supports_num_reads=False,
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    description=(
        "Local exhaustive solver enumerating every assignment; "
        "proves optimality and infeasibility but only suits small problems."
    ),
)


class ExactSolverBackend:
    """Enumerates all assignments via ``dimod.ExactSolver``.

    Intended for small scenarios, compiler correctness checks, and SA
    benchmarking — not production-sized problems. The variable limit is
    enforced by the service layer via ``ExecutionPolicy`` (Phase 2 spec
    §8, §14), not by the backend itself.
    """

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> tuple[bool, str | None]:
        """Local backend, always available. No network I/O."""
        return (True, None)

    @property
    def name(self) -> str:
        """Alias for ``capabilities.name``."""
        return self.capabilities.name

    @property
    def is_exhaustive(self) -> bool:
        """Alias for ``capabilities.exhaustive``."""
        return self.capabilities.exhaustive

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        """Exhaustively solve the compiled problem, returning every sample."""
        try:
            sampleset = dimod.ExactSolver().sample(compiled_problem.model)
        except Exception as exc:
            raise SolverExecutionError(
                f"Exact solver failed: {exc}"
            ) from exc

        samples, energies = sampleset_to_lists(sampleset)
        logger.info(
            "Backend %s solved problem %s: %d variables, %d samples",
            self.name,
            compiled_problem.original_problem.name,
            compiled_problem.num_variables,
            len(samples),
        )
        return RawSolverResult(samples=samples, energies=energies, backend=self.name)
