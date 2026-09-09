"""Solver backend interface and raw result model (spec §20, Phase 2 §10)."""

from collections.abc import Mapping
from typing import Any, Protocol

import dimod
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from annealbridge.exceptions import SolverExecutionError
from annealbridge.models import (
    CompiledProblem,
    SolverExecutionMetadata,
    SolverPreferences,
)

# The capability / availability models live in ``models.capabilities``
# (Phase 3a spec §4) and are re-exported here so existing
# ``from annealbridge.solvers import SolverCapabilities`` imports keep working.
from annealbridge.models.capabilities import (  # noqa: F401
    AvailabilityCategory,
    AvailabilityStatus,
    CredentialDeclaration,
    ModelType,
    ParameterLimit,
    SolverCapabilities,
)


def _coerce_samples(samples: Any, num_variables: int) -> np.ndarray:
    """Turn a ``samples`` input into an integer matrix (see RawSolverResult).

    The empty case comes first on purpose: ``np.asarray([])`` is float64,
    so an empty list would otherwise be rejected as non-integer.
    """
    if not isinstance(samples, np.ndarray) and len(samples) == 0:
        return np.empty((0, num_variables), dtype=np.int8)
    matrix = np.asarray(samples)
    if matrix.shape[0] == 0:
        # A 2-D empty array keeps its own column count so the shape check
        # below still catches a mismatch; a bare ``[]`` gets the variable count.
        columns = matrix.shape[1] if matrix.ndim == 2 else num_variables
        return np.empty((0, columns), dtype=np.int8)
    if not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError(
            f"samples must have an integer dtype, got {matrix.dtype}; "
            "backends assert and cast their values before building the result"
        )
    if not isinstance(samples, np.ndarray):
        info = np.iinfo(np.int8)
        if matrix.min() >= info.min and matrix.max() <= info.max:
            matrix = matrix.astype(np.int8)
    return matrix


class RawSolverResult(BaseModel):
    """Raw output of a solver backend, before validation and ranking.

    Samples are the *model's* variables, internal (slack / encoding bit)
    columns included; the compiler's ``decode`` turns them into business
    variables (3b spec §13).

    The reads are stored as arrays, not per-row dicts: ``samples`` is an
    integer matrix of shape ``(reads, len(variables))`` whose column ``j``
    is the value of ``variables[j]``, and ``energies`` is a ``float64``
    vector aligned with its rows. Any numpy integer dtype is accepted and
    kept as given (3b spec §11): BQM backends return ``int8`` bit columns,
    CQM backends ``int64`` integer values. Non-integer dtypes (float, bool,
    object) are rejected rather than silently cast. Row order is the
    sampler's own order -- every read is kept, nothing is aggregated or
    sorted (spec §21). An exhaustive backend on 20+ variables returns
    millions of rows, which a list of dicts cannot hold economically; the
    array form is what lets the orchestration layer deduplicate and
    re-validate every candidate without materialising a dict per read.

    Lists are accepted on construction and converted, so small hand-built
    results (tests, fakes) stay easy to write: a list whose values all fit
    in ``int8`` becomes ``int8``, anything wider stays ``int64``; an empty
    list is an ``int8`` matrix with zero rows. This widening happens only on
    the constructor's list path (``_coerce_samples``).

    :meth:`from_dicts` and :meth:`as_dicts` convert to and from the per-row
    dict form and are **test-facing** helpers only (their signatures are
    fixed by 3b spec §11); production backends build the matrix with
    ``sampleset_to_arrays`` and pass it to the constructor, so neither
    helper has a production caller. Unlike the constructor,
    :meth:`from_dicts` does *not* widen: it builds the matrix with the
    ``dtype`` argument (``np.int8`` by default), and a value outside that
    dtype's range raises ``OverflowError`` -- pass ``dtype=np.int64`` for
    integer-valued rows.
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
            data["samples"] = _coerce_samples(data["samples"], len(variables))
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
        dtype: Any = np.int8,
    ) -> "RawSolverResult":
        """Build a result from per-row dicts (test-facing; 3b spec §11).

        No production caller: backends go through ``sampleset_to_arrays``
        and the constructor instead.

        ``variables`` defaults to the key order of the first sample; every
        sample must assign exactly that variable set. ``dtype`` is the
        integer dtype of the sample matrix (``int8`` by default, matching
        the bit-valued output of the BQM backends; pass ``np.int64`` for
        integer-valued rows). The matrix is built with that dtype as given
        -- it is never widened, so a value outside its range raises
        ``OverflowError`` rather than being promoted.
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
        matrix = np.asarray(rows, dtype=dtype).reshape(len(rows), len(variables))
        return cls(
            variables=list(variables),
            samples=matrix,
            energies=np.asarray(energies, dtype=np.float64),
            backend=backend,
            metadata=metadata,
        )

    def as_dicts(self) -> list[dict[str, int]]:
        """Per-row ``{variable: value}`` view of ``samples``.

        Test-facing helper (3b spec §11) with no production caller.
        Materialises one dict per read -- fine for tests and small results,
        not for an exhaustive backend's output.
        """
        return [dict(zip(self.variables, row)) for row in self.samples.tolist()]


class SolverBackend(Protocol):
    """Protocol for solver backends operating on a compiled problem."""

    @property
    def capabilities(self) -> SolverCapabilities: ...

    def is_available(self) -> AvailabilityStatus:
        """Return the backend's structured availability (spec §8).

        Must not perform network I/O and must not cache. ``detail`` must be
        a categorical string (e.g. "dwave-system not installed") and never
        contain configuration values.
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
    *,
    dtype: Any = np.int8,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Convert a dimod SampleSet into ``(variables, samples, energies)``.

    Reads straight from ``sampleset.record`` so no per-row objects are
    built: ``samples`` is a contiguous copy of shape ``(reads, variables)``
    in the requested integer ``dtype`` (``int8`` by default, for the
    bit-valued BQM backends; CQM backends pass ``np.int64``) and
    ``energies`` a ``float64`` copy, both independent of the sampleset's
    own buffers. The cast does not check values -- a backend that may
    receive floats or out-of-range integers asserts them first with
    :func:`assert_samples_within_bounds`. The sampler's row order is
    preserved (no aggregation, no sorting) so every read is kept, per spec
    §21.
    """
    variables = [str(variable) for variable in sampleset.variables]
    record = sampleset.record
    samples = np.array(record.sample, dtype=dtype, order="C")
    if samples.ndim != 2:
        samples = samples.reshape(len(record), len(variables))
    energies = np.array(record.energy, dtype=np.float64)
    return variables, samples, energies


def assert_samples_within_bounds(
    sampleset: dimod.SampleSet,
    bounds: Mapping[str, tuple[int, int]],
    *,
    code: str,
) -> None:
    """Raise unless every sample value is an integer inside its bounds.

    ``bounds`` maps each variable name to its inclusive ``(lower, upper)``
    range (for a binary variable ``(0, 1)``). Values are checked as given
    -- a float-typed record is fine as long as each value is integral
    (``np.equal(x, np.floor(x))``; 3a §17.3 records that Leap may return
    floats) -- so this runs *before* :func:`sampleset_to_arrays`, whose
    integer cast would otherwise truncate silently. A violation raises
    :class:`~annealbridge.exceptions.SolverExecutionError` with the given
    ``code``; the message always contains ``"outside its bounds"``.
    """
    variables = [str(variable) for variable in sampleset.variables]
    record = sampleset.record
    values = np.asarray(record.sample)
    if values.ndim != 2:
        values = values.reshape(len(record), len(variables))
    if values.size == 0:
        return
    lower = np.array([bounds[name][0] for name in variables], dtype=np.float64)
    upper = np.array([bounds[name][1] for name in variables], dtype=np.float64)
    bad = (
        ~np.equal(values, np.floor(values))
        | (values < lower[np.newaxis, :])
        | (values > upper[np.newaxis, :])
    )
    if not bad.any():
        return
    row, column = np.argwhere(bad)[0].tolist()
    name = variables[column]
    raise SolverExecutionError(
        f"Sampler returned a value outside its bounds: variable '{name}' "
        f"must be an integer in [{int(lower[column])}, {int(upper[column])}], "
        f"got {values[row, column].item()!r}",
        code=code,
    )
