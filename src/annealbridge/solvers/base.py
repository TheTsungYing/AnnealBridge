"""Solver backend interface and raw result model (spec §20, Phase 2 §10)."""

from typing import Any, Protocol

import dimod
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

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

    The reads are stored as arrays, not per-row dicts: ``samples`` is an
    ``int8`` matrix of shape ``(reads, len(variables))`` whose column ``j``
    is the value of ``variables[j]``, and ``energies`` is a ``float64``
    vector aligned with its rows. Row order is the sampler's own order --
    every read is kept, nothing is aggregated or sorted (spec §21). An
    exhaustive backend on 20+ variables returns millions of rows, which a
    list of dicts cannot hold economically; the array form is what lets
    the orchestration layer deduplicate and re-validate every candidate
    without materialising a dict per read.

    Lists are accepted on construction and converted, so small hand-built
    results (tests, fakes) stay easy to write; :meth:`from_dicts` and
    :meth:`as_dicts` convert to and from the per-row dict form.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    variables: list[str]
    samples: np.ndarray
    energies: np.ndarray
    backend: str
    metadata: SolverExecutionMetadata | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_arrays(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        variables = data.get("variables")
        if variables is not None and "samples" in data:
            samples = np.asarray(data["samples"], dtype=np.int8)
            if samples.ndim == 1 and samples.size == 0:
                samples = samples.reshape(0, len(variables))
            data["samples"] = samples
        if "energies" in data:
            data["energies"] = np.asarray(data["energies"], dtype=np.float64)
        return data

    @model_validator(mode="after")
    def _check_shapes(self) -> "RawSolverResult":
        if self.samples.ndim != 2:
            raise ValueError(
                f"samples must be a 2-D array, got {self.samples.ndim} dimension(s)"
            )
        if self.samples.shape[1] != len(self.variables):
            raise ValueError(
                f"samples has {self.samples.shape[1]} column(s) but "
                f"{len(self.variables)} variable name(s) were given"
            )
        if self.energies.ndim != 1 or len(self.energies) != self.samples.shape[0]:
            raise ValueError(
                f"energies must be a 1-D array with one entry per sample row "
                f"({self.samples.shape[0]}), got shape {self.energies.shape}"
            )
        return self

    @property
    def num_samples(self) -> int:
        """Number of reads (rows) returned by the backend."""
        return int(self.samples.shape[0])

    @classmethod
    def from_dicts(
        cls,
        samples: list[dict[str, int]],
        energies: list[float],
        backend: str,
        *,
        variables: list[str] | None = None,
        metadata: SolverExecutionMetadata | None = None,
    ) -> "RawSolverResult":
        """Build a result from per-row dicts (tests and small-scale use).

        ``variables`` defaults to the key order of the first sample; every
        sample must assign exactly that variable set.
        """
        if variables is None:
            variables = list(samples[0]) if samples else []
        for sample in samples:
            if set(sample) != set(variables):
                raise ValueError(
                    "every sample must assign exactly the variables "
                    f"{variables}, got {sorted(sample)}"
                )
        rows = [[int(sample[name]) for name in variables] for sample in samples]
        matrix = np.asarray(rows, dtype=np.int8).reshape(len(rows), len(variables))
        return cls(
            variables=list(variables),
            samples=matrix,
            energies=np.asarray(energies, dtype=np.float64),
            backend=backend,
            metadata=metadata,
        )

    def as_dicts(self) -> list[dict[str, int]]:
        """Per-row ``{variable: value}`` view of ``samples``.

        Materialises one dict per read -- fine for tests and small results,
        not for an exhaustive backend's output.
        """
        return [dict(zip(self.variables, row)) for row in self.samples.tolist()]


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


def sampleset_to_arrays(
    sampleset: dimod.SampleSet,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Convert a dimod SampleSet into ``(variables, samples, energies)``.

    Reads straight from ``sampleset.record`` so no per-row objects are
    built: ``samples`` is a contiguous ``int8`` copy of shape
    ``(reads, variables)`` and ``energies`` a ``float64`` copy, both
    independent of the sampleset's own buffers. The sampler's row order is
    preserved (no aggregation, no sorting) so every read is kept, per spec
    §21.
    """
    variables = [str(variable) for variable in sampleset.variables]
    record = sampleset.record
    samples = np.array(record.sample, dtype=np.int8, order="C")
    if samples.ndim != 2:
        samples = samples.reshape(len(record), len(variables))
    energies = np.array(record.energy, dtype=np.float64)
    return variables, samples, energies
