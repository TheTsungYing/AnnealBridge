"""Simulated annealing solver backend (spec §21)."""

import logging

from dwave.samplers import SimulatedAnnealingSampler

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import (
    RawSolverResult,
    SolverCapabilities,
    sampleset_to_lists,
)

logger = logging.getLogger(__name__)

_CAPABILITIES = SolverCapabilities(
    name="simulated_annealing",
    remote=False,
    heuristic=True,
    exhaustive=False,
    supports_seed=True,
    supports_num_reads=True,
    supports_time_limit=False,
    supported_model_types=["bqm"],
    returns_multiple_samples=True,
    description=(
        "Local heuristic simulated-annealing sampler; scales to larger "
        "problems but does not prove optimality or infeasibility."
    ),
)


class SimulatedAnnealingBackend:
    """Samples the compiled problem with D-Wave's simulated annealer.

    All reads are kept (never just ``.first``) so every candidate reaches the
    solution validator. The seed is passed to the sampler only; global random
    state is never touched (spec §37).
    """

    def __init__(self) -> None:
        self._sampler = SimulatedAnnealingSampler()

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
        """Run simulated annealing, returning all reads as samples."""
        sample_kwargs: dict[str, int] = {
            "num_reads": preferences.num_reads,
            "num_sweeps": preferences.num_sweeps,
        }
        if preferences.seed is not None:
            sample_kwargs["seed"] = preferences.seed

        try:
            sampleset = self._sampler.sample(compiled_problem.model, **sample_kwargs)
        except Exception as exc:
            raise SolverExecutionError(
                f"Simulated annealing solver failed: {exc}"
            ) from exc

        samples, energies = sampleset_to_lists(sampleset)
        logger.info(
            "Backend %s solved problem %s: %d variables, %d samples "
            "(num_reads=%d, num_sweeps=%d, seed=%s)",
            self.name,
            compiled_problem.original_problem.name,
            compiled_problem.num_variables,
            len(samples),
            preferences.num_reads,
            preferences.num_sweeps,
            preferences.seed,
        )
        return RawSolverResult(samples=samples, energies=energies, backend=self.name)
