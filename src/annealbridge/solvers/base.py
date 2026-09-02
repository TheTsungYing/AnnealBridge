"""Solver backend interface and raw result model (spec §20, Phase 2 §10)."""

from typing import Protocol

import dimod
from pydantic import BaseModel, ConfigDict

from annealbridge.models import (
    CompiledProblem,
    SolverExecutionMetadata,
    SolverPreferences,
)


class SolverCapabilities(BaseModel):
    """Static self-description of a solver backend (Phase 2 spec §10)."""

    name: str
    remote: bool
    heuristic: bool
    exhaustive: bool
    supports_seed: bool
    supports_num_reads: bool
    supports_time_limit: bool
    supported_model_types: list[str]
    returns_multiple_samples: bool
    description: str


class RawSolverResult(BaseModel):
    """Raw output of a solver backend, before validation and ranking.

    Samples include internal (slack) variables; downstream code strips them
    using ``CompiledProblem.internal_variables``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    samples: list[dict[str, int]]
    energies: list[float]
    backend: str
    metadata: SolverExecutionMetadata | None = None


class SolverBackend(Protocol):
    """Protocol for solver backends operating on a compiled problem."""

    @property
    def capabilities(self) -> SolverCapabilities: ...

    def is_available(self) -> tuple[bool, str | None]:
        """Return ``(available, reason_if_not)``.

        Must not perform network I/O. The reason must be a categorical
        string (e.g. "dwave-system not installed") and never contain
        configuration values.
        """
        ...

    @property
    def name(self) -> str:
        """Alias for ``capabilities.name``."""
        ...

    @property
    def is_exhaustive(self) -> bool:
        """Alias for ``capabilities.exhaustive``."""
        ...

    def resolve_time_limit(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> float | None:
        """Return the time limit (seconds) :meth:`solve` would submit, or None.

        Backends whose ``capabilities.supports_time_limit`` is False return
        None. A backend that supports a time limit returns the *effective*
        value — the user's preference combined with whatever floor the
        backend applies (e.g. a remote sampler's minimum for this problem
        size) — so the service can compare it against policy *before*
        anything is submitted. Must not submit a problem; must be
        deterministic for the same ``(compiled_problem, preferences)`` so
        :meth:`solve` reuses the same value. May raise
        :class:`~annealbridge.exceptions.SolverExecutionError`.
        """
        ...

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult: ...


def sampleset_to_lists(
    sampleset: dimod.SampleSet,
) -> tuple[list[dict[str, int]], list[float]]:
    """Convert a dimod SampleSet into aligned sample/energy lists.

    Preserves the sampler's row order (no aggregation, no sorting) so every
    read is kept, per spec §21.
    """
    variables = list(sampleset.variables)
    samples = [
        {str(variable): int(value) for variable, value in zip(variables, row)}
        for row in sampleset.record.sample
    ]
    energies = [float(energy) for energy in sampleset.record.energy]
    return samples, energies
