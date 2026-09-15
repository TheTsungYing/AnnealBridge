"""Leap hybrid CQM solver backend (Phase 3a spec §17).

The first backend that accepts a constrained quadratic model: hard
constraints travel natively (no penalty, no slack), soft constraints as
weighted constraints, and the solver returns several samples. What it
does *not* do is as important as what it does (§17.3, overview
principle 2): the sampler's own ``is_feasible`` verdict is never used to
filter or order samples — every sample goes back in the sampler's order
and the service re-validates each one against the original problem. The
verdict is only *counted* into ``metadata.sampler_reported_feasible`` so
a disagreement between the sampler and the validator stays visible.

``dwave.system`` is imported lazily inside the default sampler factory so
this module — and the default registry that instantiates the backend —
imports cleanly without the ``dwave`` extra installed (spec §4).
"""

import logging
from typing import Any, Callable

import numpy as np

from annealbridge.models import CompiledProblem, SolverPreferences
from annealbridge.solvers.base import (
    AvailabilityStatus,
    BackendAliases,
    ParameterLimit,
    RawSolverResult,
    SolverCapabilities,
    assert_samples_within_bounds,
    log_solved,
    record_column,
    result_from_sampleset,
)
from annealbridge.solvers.metadata import sanitize_sampleset_info
from annealbridge.solvers.ocean import (
    HYBRID_SAMPLE_EXCEPTION_CODES,
    OCEAN_CREDENTIALS,
    HybridTimeLimitMemo,
    call_ocean,
    dwave_availability,
    ocean_sampler_holder,
    resolve_hybrid_sampler,
    resolved,
)

logger = logging.getLogger(__name__)

_CAPABILITIES = SolverCapabilities(
    name="leap_hybrid_cqm",
    remote=True,
    heuristic=True,
    exhaustive=False,
    supports_seed=False,
    supports_num_reads=False,
    supports_time_limit=True,
    supported_model_types=["cqm"],
    returns_multiple_samples=True,
    credentials=OCEAN_CREDENTIALS,
    parameter_limits=[
        ParameterLimit(
            preference="leap_hybrid_cqm.time_limit_seconds",
            limit="time_seconds",
            error_code="REMOTE_TIME_LIMIT",
        ),
    ],
    description=(
        "D-Wave Leap cloud hybrid constrained-quadratic-model solver. Hard "
        "constraints are submitted natively (no penalty, no slack) and soft "
        "constraints as weighted constraints; the solver returns several "
        "samples. The solver's own feasibility flag is ignored: every sample "
        "is re-validated against the original problem. Leap limits "
        "(5,000,000 variables, 100,000 constraints, minimum time_limit 5 s) "
        "are enforced by the solver, not by this server."
    ),
)


def _default_sampler_factory() -> Any:
    """Create a real ``LeapHybridCQMSampler`` (requires the ``dwave`` extra)."""
    from dwave.system import LeapHybridCQMSampler

    return LeapHybridCQMSampler()


def _sampler_reported_feasible(sampleset: Any) -> int | None:
    """Count of rows the sampler flagged ``is_feasible``; None when absent.

    Diagnostic only (§22): never used for filtering or ranking. A
    sampleset built without that vector has no such record field at all
    (attribute access would raise), so presence is checked via
    ``record.dtype.names`` (:func:`record_column`).
    """
    values = record_column(sampleset, "is_feasible")
    if values is None:
        return None
    return int(values.sum())


class LeapHybridCQMBackend(BackendAliases):
    """Solves the compiled CQM on the Leap hybrid cloud solver.

    ``sampler_factory`` is the single test seam: production uses the default
    (a real ``LeapHybridCQMSampler``), tests inject a fake. The sampler is
    built lazily on first use and cached while the credential sources are
    unchanged (construction fetches solver metadata and starts worker
    threads); a rotated token or edited Ocean config, or an authentication
    failure, makes the next solve rebuild it (review F-21); a failed
    construction is never cached.

    ``num_reads`` / ``num_sweeps`` / ``seed`` / ``penalty_multiplier`` are
    meaningless for the hybrid CQM solver and are never forwarded; only
    ``time_limit`` is passed through. No ``label`` is sent either: the
    problem name is business data and must not appear on the Leap
    dashboard (§17.2).
    """

    def __init__(self, sampler_factory: Callable[[], Any] | None = None) -> None:
        # Review F-03: the holder comes with this backend's credential
        # declaration and the Ocean config-file token source, so a directly
        # constructed backend redacts its token too (see ocean_sampler_holder).
        self._sampler = ocean_sampler_holder(
            sampler_factory, _default_sampler_factory, _CAPABILITIES
        )
        # The ``min_time_limit`` of the service's pre-submission check is
        # reused by the solve of the same attempt (see HybridTimeLimitMemo).
        self._time_limit_memo = HybridTimeLimitMemo()

    @property
    def capabilities(self) -> SolverCapabilities:
        return _CAPABILITIES

    def is_available(self) -> AvailabilityStatus:
        """Installability and credentials, via the shared check. No network I/O."""
        return dwave_availability()

    def _resolve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> tuple[Any, float]:
        """The sampler and the effective ``time_limit``, by the shared hybrid rule."""
        user_time_limit = (
            preferences.leap_hybrid_cqm.time_limit_seconds
            if preferences.leap_hybrid_cqm is not None
            else None
        )
        return resolve_hybrid_sampler(
            self._sampler,
            compiled_problem.model,
            user_time_limit,
            label="Leap hybrid CQM",
            memo=self._time_limit_memo,
        )

    def resolve_time_limit(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> float:
        """Effective ``time_limit`` (seconds) a solve would submit (§17.2).

        The shared hybrid rule (:func:`resolve_hybrid_sampler`, the very
        same code as the BQM hybrid backend): the user's value if given,
        floored at the sampler's ``min_time_limit(cqm)``; the sampler
        minimum alone when the user gave none. Nothing is submitted. The
        service compares this value with policy before :meth:`solve` runs;
        :meth:`solve` resolves through the same rule, so both see the same
        number — and, within one attempt, the same sampler and the same
        memoised minimum.
        """
        return self._resolve(compiled_problem, preferences)[1]

    def solve(
        self,
        compiled_problem: CompiledProblem,
        preferences: SolverPreferences,
    ) -> RawSolverResult:
        """Submit the CQM to the hybrid solver and return every sample.

        The submitted ``time_limit`` is exactly :meth:`resolve_time_limit`
        and is recorded in metadata as ``effective_time_limit_seconds``.
        The policy ceiling is enforced by the service layer, not here.
        Samples are returned in the sampler's own order, feasible-flagged
        or not (§17.3); the flag is only counted into
        ``sampler_reported_feasible``.

        Before the conversion every value is asserted to be an integer
        inside its CQM variable's own ``[lower_bound, upper_bound]`` range
        (3b §11) — the sampler's ``record.sample`` dtype is the sampler's
        choice (Leap may return floats), so an out-of-range or fractional
        value would otherwise be silently truncated by the cast. The
        matrix is then built as ``int64``, which holds integer variables
        as well as bits.
        """
        cqm = compiled_problem.model
        sampler, effective_time_limit = self._resolve(compiled_problem, preferences)
        sampleset = call_ocean(
            "Leap hybrid CQM solve failed",
            HYBRID_SAMPLE_EXCEPTION_CODES,
            lambda: resolved(sampler.sample_cqm(cqm, time_limit=effective_time_limit)),
            holder=self._sampler,
        )

        bounds = {
            str(variable): (
                int(cqm.lower_bound(variable)),
                int(cqm.upper_bound(variable)),
            )
            for variable in cqm.variables
        }
        assert_samples_within_bounds(sampleset, bounds, code="REMOTE_SOLVER_ERROR")
        metadata = sanitize_sampleset_info(
            sampleset.info,
            backend=self.name,
            effective_time_limit_seconds=effective_time_limit,
            sampler_reported_feasible=_sampler_reported_feasible(sampleset),
        )
        result = result_from_sampleset(
            sampleset, backend=self.name, metadata=metadata, dtype=np.int64
        )
        log_solved(
            logger,
            self.name,
            compiled_problem.original_problem.name,
            compiled_problem.num_variables,
            len(result.samples),
            effective_time_limit_seconds=effective_time_limit,
            sampler_reported_feasible=metadata.sampler_reported_feasible,
        )
        return result
