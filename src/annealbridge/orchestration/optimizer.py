"""Orchestration service: validate, compile, solve, rank (spec §28).

The service drives the whole pipeline against the *original* problem:
feasibility and objective values are recomputed from the problem itself
(spec §2.2, §24), never taken from BQM energy. It only relies on the
``ModelCompiler`` / ``SolverBackend`` / ``PenaltyStrategy`` protocols,
never on the concrete compiled model type (spec §17).
"""

import logging
import math
import threading
import time
import traceback
from collections.abc import Collection, Iterable
from dataclasses import dataclass

import numpy as np

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.compiler.base import ModelCompiler
from annealbridge.exceptions import (
    CompilationError,
    NonFiniteModelError,
    OptimizerError,
)
from annealbridge.models import (
    ClosestCandidate,
    CompiledProblem,
    HardViolationRate,
    InfeasibilityDiagnostics,
    ModelType,
    Objective,
    OptimizationProblem,
    Solution,
    SolveAttempt,
    SolveError,
    SolveResult,
    SolveStatus,
    SolverCapabilities,
    SolverExecutionMetadata,
    SolverPreferences,
    catalog_error,
)
from annealbridge.orchestration.limits import (
    gate_errors,
    preference_limit_errors,
    read_preference,
    select_model_type,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.orchestration.routing import recommend
from annealbridge.penalty import PenaltyStrategy, ScaledPenaltyStrategy
from annealbridge.solvers import (
    RawSolverResult,
    SolverBackend,
    SolverRegistry,
    redact,
)
from annealbridge.validation import (
    BackendRecommendationResult,
    BatchValidation,
    ProblemValidationResult,
    validate_batch,
    validate_problem_full,
    validate_solution,
)
from annealbridge.validation.estimates import estimate_model_variables
from annealbridge.version import package_version

logger = logging.getLogger(__name__)


def _elapsed_ms(started: float) -> float:
    """Milliseconds of wall clock since ``started`` (a ``perf_counter`` value).

    Rounded to the microsecond: finer digits are clock noise, not information.
    """
    return round((time.perf_counter() - started) * 1000.0, 3)


def evaluate_objective(objective: Objective, sample: dict[str, int]) -> float:
    """Compute the business objective value of ``sample`` (spec §8.1, §24).

    Always uses the original coefficients and constant; the value is the
    same regardless of direction (direction only says how to interpret it).
    """
    value = objective.constant
    for term in objective.linear_terms:
        value += term.coefficient * sample[term.variable]
    for term in objective.quadratic_terms:
        value += term.coefficient * sample[term.variable1] * sample[term.variable2]
    return value


def evaluate_objective_batch(
    objective: Objective, variables: list[str], samples: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`evaluate_objective` over the rows of ``samples``.

    Column ``j`` of ``samples`` is the integer value of ``variables[j]``. Terms
    are accumulated in the same order and association as the scalar
    version (constant first, then linear, then quadratic), so the two agree
    bit for bit; the unit tests assert that equality.
    """
    column = {name: index for index, name in enumerate(variables)}
    value = np.full(samples.shape[0], objective.constant, dtype=np.float64)
    for term in objective.linear_terms:
        value += term.coefficient * samples[:, column[term.variable]]
    for term in objective.quadratic_terms:
        value += (
            term.coefficient
            * samples[:, column[term.variable1]]
            * samples[:, column[term.variable2]]
        )
    return value


def _pack_rows(matrix: np.ndarray) -> np.ndarray:
    """Pack each 0/1 row of ``matrix`` into big-endian 64-bit words.

    Returns an ``uint64`` array of shape ``(rows, ceil(columns / 64))``.
    Column 0 of the input lands in the most significant bit of word 0, so
    comparing rows word by word (word 0 first) orders them exactly like
    comparing the original rows as tuples. Any number of columns is
    supported — wider rows simply produce more words — so there is no
    fallback path to keep correct separately.
    """
    rows, columns = matrix.shape
    words = max(1, -(-columns // 64))
    packed = np.packbits(matrix, axis=1, bitorder="big")  # (rows, ceil(cols/8))
    padded = np.zeros((rows, words * 8), dtype=np.uint8)
    padded[:, : packed.shape[1]] = packed
    return padded.view(">u8").astype(np.uint64)


def _lexsort(keys: list[np.ndarray]) -> np.ndarray:
    """Stable multi-key argsort with ``keys`` given in *priority* order.

    ``np.lexsort`` treats its last key as the most significant, which is
    easy to get backwards; this wrapper takes the natural order instead.
    """
    return np.lexsort(tuple(reversed(keys)))


def _words(packed: np.ndarray) -> list[np.ndarray]:
    """Columns of a :func:`_pack_rows` result, most significant word first."""
    return [packed[:, word] for word in range(packed.shape[1])]


def _row_keys(matrix: np.ndarray) -> list[np.ndarray]:
    """Sort keys (priority order) that compare rows of ``matrix`` as tuples.

    A 0/1 ``int8`` matrix -- the BQM backends' bit path -- packs into the
    :func:`_pack_rows` words, exactly as before 3b. Anything else (integer
    values from a CQM backend or a decoded integer problem) uses one
    ``int64`` key per column. Both orderings equal the lexicographic order
    of the rows, so deduplication and tie-breaking behave identically on
    either path (3b spec §11). The 0/1 check is a min/max scan, not
    ``np.isin`` (two orders of magnitude slower on 16M rows), and it is
    mandatory: ``np.packbits`` would silently treat 2 or -1 as 1.
    """
    if matrix.dtype == np.int8 and (
        matrix.size == 0 or (matrix.min() >= 0 and matrix.max() <= 1)
    ):
        return _words(_pack_rows(matrix))
    return [
        matrix[:, column].astype(np.int64, copy=False)
        for column in range(matrix.shape[1])
    ]


@dataclass(frozen=True)
class CandidateSet:
    """Deduplicated business candidates, aligned row by row.

    ``samples`` is an integer matrix with one row per distinct business
    assignment (column ``j`` is ``variables[j]``, in the input's variable
    order minus the internal ones); ``energies`` is the minimum energy the
    solver reported for that assignment — kept for reporting and debugging
    only, never used for feasibility or ranking (overview principle 2).
    ``counts`` is how many rows of the solver's raw output collapsed into
    each deduplicated assignment, so it sums to ``raw.num_samples``.
    """

    variables: list[str]
    samples: np.ndarray  # integer matrix, shape (candidates, len(variables))
    energies: np.ndarray  # float64, shape (candidates,)
    counts: np.ndarray  # int64, shape (candidates,)

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def sample_dict(self, index: int) -> dict[str, int]:
        return dict(zip(self.variables, self.samples[index].tolist()))

    def as_pairs(self) -> list[tuple[dict[str, int], float]]:
        """``[(business_sample, energy), ...]``.

        Test-facing helper; no production caller. It materialises one dict
        per candidate, so it is only for small-scale use.
        """
        return [
            (self.sample_dict(index), float(self.energies[index]))
            for index in range(len(self))
        ]


def deduplicate_samples(
    raw: RawSolverResult, internal_variables: Collection[str] = frozenset()
) -> CandidateSet:
    """Strip internal variables and deduplicate by business assignment.

    Duplicates keep the minimum energy seen for that assignment (spec §25);
    among equal energies the earliest read wins, and candidates come back
    in order of first appearance, so the result is fully deterministic and
    identical to a row-by-row pass — it is just computed on the arrays.
    Energy is used here only to pick which duplicate's energy to report.
    How many raw rows each survivor stands for is kept in ``counts``.

    ``internal_variables`` is optional since 3b: the service hands over a
    result the compiler has already decoded, so nothing is left to strip.
    """
    business_columns = [
        index
        for index, name in enumerate(raw.variables)
        if name not in internal_variables
    ]
    variables = [raw.variables[index] for index in business_columns]
    business = raw.samples[:, business_columns]
    count = business.shape[0]
    if count == 0:
        return CandidateSet(
            variables=variables,
            samples=business,
            energies=np.asarray(raw.energies, dtype=np.float64),
            counts=np.zeros(0, dtype=np.int64),
        )

    keys = _row_keys(business)
    energies = np.asarray(raw.energies, dtype=np.float64)
    # Sort by assignment, then energy; the sort is stable, so within one
    # assignment equal energies stay in read order.
    order = _lexsort([*keys, energies])
    # A new group starts wherever any key differs from the previous row.
    group_start = np.zeros(count, dtype=bool)
    group_start[0] = True
    for key in keys:
        sorted_key = key[order]
        group_start[1:] |= sorted_key[1:] != sorted_key[:-1]
    starts = np.flatnonzero(group_start)
    representatives = order[starts]  # min-energy read of each assignment
    first_seen = np.minimum.reduceat(order, starts)  # earliest read of each
    # Group ``g`` spans ``starts[g]`` up to the next start (or the end), so
    # its size is how many raw rows carried that assignment.
    group_sizes = np.diff(np.append(starts, count))
    # One permutation for all three arrays, or ``counts`` would describe a
    # different candidate than the row next to it.
    permutation = np.argsort(first_seen, kind="stable")
    keep = representatives[permutation]
    return CandidateSet(
        variables=variables,
        samples=np.ascontiguousarray(business[keep]),
        energies=energies[keep],
        counts=group_sizes[permutation].astype(np.int64, copy=False),
    )


@dataclass(frozen=True)
class ProcessedCandidates:
    """What :func:`process_candidates` found in one attempt's raw output.

    ``unique_samples`` and ``feasible_samples`` are the candidate counts the
    attempt reports; both are counted over deduplicated assignments, not
    over the raw rows. ``infeasibility`` is set only when there were
    candidates and none was feasible.
    """

    solutions: list[Solution]
    unique_samples: int
    feasible_samples: int
    infeasibility: InfeasibilityDiagnostics | None = None


def diagnose_infeasibility(
    problem: OptimizationProblem,
    candidates: CandidateSet,
    verdict: BatchValidation,
) -> InfeasibilityDiagnostics:
    """Explain an attempt whose candidates were all infeasible.

    Picks the candidate with the smallest total hard violation — the first
    such candidate in first-seen order on a tie, so the choice is
    deterministic for a given raw output — and re-runs the full validator
    on it, so the reported evaluations come from the same arithmetic as a
    ranked solution's. Only the original problem is consulted; the
    solver's energy plays no part (overview principle 2).
    """
    closest = int(np.argmin(verdict.hard_violation_total))
    sample = candidates.sample_dict(closest)
    validation = validate_solution(problem, sample)
    # Same order and association as the batch kernel, so the two agree.
    hard_total = 0.0
    for evaluation in validation.evaluations:
        if evaluation.constraint_type == "hard":
            hard_total += evaluation.violation_amount
    total = len(candidates)
    rates = [
        HardViolationRate(
            constraint_id=constraint_id,
            violated_candidates=violated,
            candidates=total,
            violated_fraction=violated / total,
        )
        for constraint_id, violated in zip(
            verdict.hard_constraint_ids, verdict.hard_violated_counts.tolist()
        )
    ]
    return InfeasibilityDiagnostics(
        closest_candidate=ClosestCandidate(
            variables=sample,
            hard_violation_total=hard_total,
            constraint_evaluations=validation.evaluations,
        ),
        hard_violation_rates=rates,
    )


def process_candidates(
    problem: OptimizationProblem,
    raw: RawSolverResult,
    internal_variables: Collection[str] = frozenset(),
    top_k: int = 5,
) -> ProcessedCandidates:
    """Run the §25 candidate pipeline on raw solver output.

    Deduplicates, validates *every* candidate against the original problem,
    keeps feasible candidates only, computes objective/soft-violation/ranking
    scores, and returns the top ``top_k`` solutions ranked from 1, plus the
    unique and feasible candidate counts.

    Validation runs in two layers with one arithmetic: the vectorised
    :func:`validate_batch` judges all candidates (feasibility, soft score)
    and :func:`evaluate_objective_batch` their objective; the ranking is
    computed from those. Only the top-k then go through
    :func:`validate_solution` to build the full per-constraint evaluations
    for the report. Energy is never consulted for either step.
    """
    candidates = deduplicate_samples(raw, internal_variables)
    if len(candidates) == 0:
        return ProcessedCandidates([], 0, 0)
    minimize = problem.objective.direction == "minimize"

    verdict = validate_batch(problem, candidates.variables, candidates.samples)
    feasible = np.flatnonzero(verdict.feasible)
    if len(feasible) == 0:
        return ProcessedCandidates(
            [], len(candidates), 0, diagnose_infeasibility(problem, candidates, verdict)
        )

    feasible_samples = candidates.samples[feasible]
    objective_value = evaluate_objective_batch(
        problem.objective, candidates.variables, feasible_samples
    )
    soft_violation_score = verdict.soft_violation_score[feasible]
    if minimize:
        ranking_score = objective_value + soft_violation_score
    else:
        ranking_score = objective_value - soft_violation_score

    # §25.1: ascending ranking_score for minimize, descending for maximize;
    # ties broken by objective_value in the same direction, then by the
    # name-sorted assignment tuple so the order is fully deterministic.
    sign = 1.0 if minimize else -1.0
    name_order = sorted(
        range(len(candidates.variables)), key=lambda j: candidates.variables[j]
    )
    tie_break = _row_keys(feasible_samples[:, name_order])
    order = _lexsort([sign * ranking_score, sign * objective_value, *tie_break])

    solutions: list[Solution] = []
    for rank, position in enumerate(order[:top_k].tolist(), start=1):
        index = int(feasible[position])
        sample = candidates.sample_dict(index)
        validation = validate_solution(problem, sample)
        solutions.append(
            Solution(
                rank=rank,
                variables=sample,
                objective_value=float(objective_value[position]),
                soft_violation_score=validation.soft_violation_score,
                ranking_score=float(ranking_score[position]),
                energy=float(candidates.energies[index]),
                sample_count=int(candidates.counts[index]),
                hard_constraints_satisfied=validation.feasible,
                constraint_evaluations=validation.evaluations,
            )
        )
    return ProcessedCandidates(solutions, len(candidates), int(len(feasible)))


class OptimizationService:
    """End-to-end solve pipeline over pluggable components (spec §28)."""

    def __init__(
        self,
        compilers: Iterable[ModelCompiler] | None = None,
        penalty_strategy: PenaltyStrategy | None = None,
        registry: SolverRegistry | None = None,
        policy: ExecutionPolicy = ExecutionPolicy(),
    ) -> None:
        self._policy = policy
        # 3a §16.1: one compiler per model type, chosen per solve from the
        # backend's declaration. This is the only place the service names
        # a concrete compiler (§4).
        compiler_list = (
            list(compilers) if compilers is not None else [BQMCompiler(), CQMCompiler()]
        )
        self._compilers: dict[ModelType, ModelCompiler] = {}
        for compiler in compiler_list:
            if compiler.model_type in self._compilers:
                raise ValueError(
                    f"duplicate compiler for model type '{compiler.model_type}'"
                )
            self._compilers[compiler.model_type] = compiler
        self._penalty_strategy: PenaltyStrategy = (
            penalty_strategy if penalty_strategy is not None else ScaledPenaltyStrategy()
        )
        self._registry: SolverRegistry = (
            registry if registry is not None else SolverRegistry.default()
        )
        self._check_declared_limits()
        # Concurrency slots are per service instance (spec §14); a CLI-style
        # single call is unaffected.
        self._solve_slots = threading.BoundedSemaphore(policy.max_concurrent_solves)

    def _check_declared_limits(self) -> None:
        """Spec §11.3: every limit a backend is subject to is checkable before any solve.

        Three things can be wrong: the policy has no value for a limit key
        the backend is subject to (it would run unlimited, or a solve
        would compare against None), a ``ParameterLimit`` preference path
        names nothing on ``SolverPreferences``, or it names a field that is
        not numeric (every solve and recommend would raise from
        ``read_preference``; 2026-09-09 review F-03 and its follow-ups).
        All are composition-root errors, so all fail here, at
        construction, rather than on each request. The keys come from
        ``policy.limits_for(caps)`` — the same single source the
        capabilities view and the service's checks read (spec §12.3) — so
        the declared keys and the flag-driven ones (``variables`` for an
        exhaustive backend, ``time_seconds``, the retry ceiling, ``top_k``)
        are covered alike.
        """
        for name in self._registry.names():
            caps = self._registry.get(name).capabilities
            for key, maximum in self._policy.limits_for(caps).items():
                if maximum is None:
                    raise ValueError(
                        f"backend '{name}' is subject to limit "
                        f"'{key.removeprefix('max_')}' but the policy has no "
                        f"value for it"
                    )
            for declaration in caps.parameter_limits:
                try:
                    read_preference(SolverPreferences(), declaration.preference)
                except ValueError as exc:
                    raise ValueError(
                        f"backend '{name}' declares a limit on preference "
                        f"'{declaration.preference}' (limit "
                        f"'{declaration.limit}', error code "
                        f"'{declaration.error_code}') but that path cannot "
                        f"carry a limit: {exc}"
                    ) from exc

    def _select_model_type(self, caps: SolverCapabilities) -> ModelType | None:
        """3a §16.1: the first declared model type the service can compile.

        Delegates to :func:`select_model_type` so ``validate``, ``solve``
        and ``recommend`` share one rule; it dispatches on the backend's
        *declaration*, never on its name. None means no compiler fits.
        """
        return select_model_type(caps, self._compilers)

    def _select_compiler(self, caps: SolverCapabilities) -> ModelCompiler | None:
        model_type = self._select_model_type(caps)
        return None if model_type is None else self._compilers[model_type]

    def _validate_against_backend(
        self, problem: OptimizationProblem
    ) -> tuple[ProblemValidationResult, SolverCapabilities | None, ModelType | None]:
        """The validator's error pass plus its advisory layer, for the
        backend ``problem`` names — shared by :meth:`validate` and
        :meth:`solve` so both report the same warnings.

        Looks the backend up only to read its *declaration*; nothing is
        compiled or solved, no network is touched and no concurrency slot
        is taken. The policy → validator wiring lives here (not in the
        interfaces) so any Python caller gets the same advice as MCP/CLI.
        Returns the capabilities (None for an unknown backend) and the
        compiler path chosen (None when no compiler fits) so the callers
        can add what only they know: ``validate`` turns those two cases
        into advisory warnings, ``solve`` into errors.
        """
        backend_name = problem.solver.backend
        try:
            caps: SolverCapabilities | None = self._registry.get(
                backend_name
            ).capabilities
        except KeyError:
            caps = None
        # Same selection as solve (§16.1), so the estimate describes the
        # path the problem would actually take; None when no compiler fits
        # (solve would then fail with NO_COMPILER_FOR_MODEL_TYPE).
        model_type = self._select_model_type(caps) if caps is not None else None
        result = validate_problem_full(
            problem,
            capabilities=caps,
            max_compiled_variables=int(self._policy.limit("variables")),
            model_type=model_type,
        )
        return result, caps, model_type

    def validate(self, problem: OptimizationProblem) -> ProblemValidationResult:
        """Dry-run check of ``problem`` against the named backend (3a §10).

        Nothing is compiled or solved, no network is touched and no
        concurrency slot is taken (see :meth:`_validate_against_backend`).

        An unknown backend (possible with a custom registry) still gets
        the backend-independent checks plus an UNKNOWN_BACKEND *warning*;
        the validator does not judge backend existence, ``solve`` does.
        """
        backend_name = problem.solver.backend
        result, caps, model_type = self._validate_against_backend(problem)
        if caps is None:
            result.warnings.append(
                catalog_error(
                    "UNKNOWN_BACKEND",
                    f"Unknown solver backend '{backend_name}'; "
                    f"available backends: {', '.join(self._registry.names())}",
                    path="solver.backend",
                )
            )
        elif model_type is None:
            # The validator assumes the backend's preferred type when told
            # nothing; here we *know* no compiler fits, so report no path
            # and warn the way an unknown backend is warned about (solve
            # would fail with the same code as an error).
            result.model_type = None
            result.warnings.append(
                catalog_error(
                    "NO_COMPILER_FOR_MODEL_TYPE",
                    f"Backend '{backend_name}' accepts model types "
                    f"[{', '.join(caps.supported_model_types)}] but the "
                    f"server has no compiler for any of them",
                    path="solver.backend",
                )
            )
        return result

    def recommend(self, problem: OptimizationProblem) -> BackendRecommendationResult:
        """Rank the registry's backends for ``problem`` (3a §23.5).

        Advisory only: ``solve`` never reads this. Nothing is compiled or
        solved, no network is touched and no concurrency slot is taken.
        """
        return recommend(problem, self._registry, self._policy, self._compilers)

    def solve(self, problem: OptimizationProblem) -> SolveResult:
        """Solve ``problem`` and return a structured :class:`SolveResult`.

        Never raises for domain errors: validation and compilation failures
        become ``invalid_problem`` (no backend was ever called) and solver
        failures become ``solver_error`` (spec §27, §36).

        The result carries the same advisory ``warnings`` :meth:`validate`
        would give for this backend (SEED_IGNORED, LARGE_INTEGER_RANGE,
        SOFT_WEIGHT_SMALL, ...) ahead of any warning raised by the run
        itself (REMOTE_RETRIES_DISABLED), whatever the status — they
        describe the problem as submitted, and an agent that skipped
        ``validate`` must still see them. Only ``invalid_problem`` carries
        none: warnings are produced for an error-free problem only.
        """
        started = time.perf_counter()
        validation, _, _ = self._validate_against_backend(problem)
        if validation.errors:
            logger.info(
                "Problem %s failed validation with %d error(s)",
                problem.name,
                len(validation.errors),
            )
            result = self._failure("invalid_problem", None, None, validation.errors)
        else:
            result = self._dispatch(problem)
        # Every path is stamped the same way: the service's own wall clock
        # (independent of the vendor-reported ``metadata.timing_us``) and
        # the package version that produced the result.
        return result.model_copy(
            update={
                "warnings": [*validation.warnings, *result.warnings],
                "elapsed_ms": _elapsed_ms(started),
                "annealbridge_version": package_version(),
            }
        )

    def _dispatch(self, problem: OptimizationProblem) -> SolveResult:
        """§16.2 steps 2–6 for a validated problem: resolve the backend,
        gate it, take a concurrency slot and run the attempts."""
        direction = problem.objective.direction
        backend_name = problem.solver.backend
        try:
            backend: SolverBackend = self._registry.get(backend_name)
        except KeyError:
            message = (
                f"Unknown solver backend '{backend_name}'; "
                f"available backends: {', '.join(self._registry.names())}"
            )
            return self._failure(
                "backend_unavailable",
                backend_name,
                direction,
                [catalog_error("UNKNOWN_BACKEND", message)],
            )

        gate_result = self._gate_backend(backend_name, backend, direction)
        if gate_result is not None:
            return gate_result

        # §14 step 6: non-blocking — a full server rejects instead of
        # queueing, so the caller decides what to do next.
        if not self._solve_slots.acquire(blocking=False):
            return self._failure(
                "resource_limit_exceeded",
                backend.name,
                direction,
                [
                    catalog_error(
                        "CONCURRENCY_LIMIT",
                        f"Too many concurrent solves: the server allows at "
                        f"most {self._policy.max_concurrent_solves}",
                    )
                ],
            )
        try:
            return self._run_attempts(problem, backend, direction)
        finally:
            self._solve_slots.release()

    def _failure(
        self,
        status: SolveStatus,
        backend: str | None,
        direction: str | None,
        errors: list[SolveError],
        *,
        attempts: list[SolveAttempt] | None = None,
        metadata: SolverExecutionMetadata | None = None,
    ) -> SolveResult:
        """Build a no-solutions SolveResult for a structured failure.

        ``metadata`` carries the last completed attempt's facts (e.g. quota
        spent on a remote solve) when a later attempt failed. ``errors``
        must not be empty: its first message becomes the result message,
        and a failure with nothing to say is a caller bug.
        """
        assert errors, "_failure needs at least one error: it becomes the result message"
        return SolveResult(
            status=status,
            backend=backend,
            objective_direction=direction,
            solutions=[],
            attempts=attempts or [],
            errors=errors,
            metadata=metadata,
            message=errors[0].message,
        )

    def _gate_backend(
        self, backend_name: str, backend: SolverBackend, direction: str
    ) -> SolveResult | None:
        """§16.2 steps 3–5 via :func:`gate_errors`; None when the backend may run."""
        gate = gate_errors(backend_name, backend, self._policy)
        if gate is None:
            return None
        status, reported_name, errors = gate
        return self._failure(status, reported_name, direction, errors)

    def _preference_limit_errors(
        self, backend: SolverBackend, preferences: SolverPreferences
    ) -> list[SolveError]:
        """§16.2 step 8: the backend's declared parameter limits. Never clamps.

        Only the user's own values are known before compile. The
        *effective* hybrid time limit (floored at the sampler minimum, which
        depends on the compiled size) is checked per attempt in
        _effective_time_limit_error, before anything is submitted.
        """
        return preference_limit_errors(backend.capabilities, preferences, self._policy)

    def _effective_time_limit_error(
        self,
        backend: SolverBackend,
        compiled: CompiledProblem,
        preferences: SolverPreferences,
    ) -> SolveError | None:
        """§14 step 9, compiled-dependent part of the hybrid time limit.

        A time-limited remote backend may raise the user's value to its own
        minimum for this problem size (spec §16), so the number actually
        submitted can exceed policy even when the user gave none. Ask the
        backend for the effective value and refuse *before* submission.
        Never clamps. The backend never sees the policy; the service never
        sees the backend's minimum rule.
        """
        caps = backend.capabilities
        if not (caps.remote and caps.supports_time_limit):
            return None
        effective = backend.resolve_time_limit(compiled, preferences)
        maximum = self._policy.limit("time_seconds")
        if effective is None or effective <= maximum:
            return None
        return catalog_error(
            "REMOTE_TIME_LIMIT",
            f"effective time_limit_seconds {effective} (the requested value "
            f"raised to the solver's minimum for this problem size) exceeds "
            f"the server maximum of {maximum}",
        )

    def _penalty_overflow(
        self,
        backend: SolverBackend,
        direction: str,
        attempts: list[SolveAttempt],
        penalty: float,
        *,
        metadata: SolverExecutionMetadata | None,
    ) -> SolveResult:
        """Structured stop of the penalty ladder (2026-09-09 review F-07).

        Reported as ``resource_limit_exceeded`` under PENALTY_OVERFLOW: the
        floating-point range is a hard ceiling the request ran into, the
        attempts made so far are kept, and no backend is called with a
        non-finite model.

        ``metadata`` is the last completed attempt's facts, the same thing
        the infeasible and error paths pass on (2026-09-11 review F08):
        climbing the ladder may have spent real quota, and stopping at the
        ceiling must not throw that away. ``None`` when no attempt finished.
        """
        made = len(attempts)
        last = attempts[-1].penalty if attempts else None
        return self._failure(
            "resource_limit_exceeded",
            backend.name,
            direction,
            [
                catalog_error(
                    "PENALTY_OVERFLOW",
                    f"Hard penalty {penalty!r} exceeds the floating-point range "
                    f"after {made} attempt(s)"
                    + (f" (last finite penalty {last!r})" if last is not None else "")
                    + "; no feasible solution was found before the ceiling",
                )
            ],
            attempts=attempts,
            metadata=metadata,
        )

    @staticmethod
    def _exact_variable_limit_error(num_variables: int, limit: int | float) -> SolveError:
        """§14 step 9 / 3a §12.2: the exhaustive backend's variable ceiling.

        One wording for both places it is checked: from the estimate before
        compile and from the compiled model after (2026-09-09 review
        F-14). The estimate equals the compiled count for every validated
        problem (``estimate_model_variables``), so the sentence is true
        either way.
        """
        return catalog_error(
            "EXACT_VARIABLE_LIMIT",
            f"Compiled problem has {num_variables} variables (including "
            f"internal), exceeding the exhaustive backend limit of {limit}",
        )

    @staticmethod
    def _compile(
        compiler: ModelCompiler, problem: OptimizationProblem, penalty: float | None
    ) -> CompiledProblem:
        """``compiler.compile`` with every failure expressed as a CompilationError.

        No backend has been called yet, so whatever goes wrong here is a
        problem-side failure the caller must see as ``invalid_problem`` /
        COMPILATION_FAILED, never as ``solver_error`` (2026-09-09 review,
        service-layer follow-ups). A ``CompilationError`` (including
        ``NonFiniteModelError``) passes through unchanged so the handlers in
        ``_run_attempts`` keep telling them apart; anything else — a bare
        ``ValueError`` from the bounds / slack arithmetic on a problem that
        bypassed the validator, a dimod or pydantic error — is wrapped with
        its class name kept. The compiler's ``decode`` is deliberately not
        wrapped: by then a backend has answered, and a result missing a
        column is that backend's contract violation (review F-03).
        """
        try:
            return compiler.compile(problem, penalty)
        except CompilationError:
            raise
        except Exception as exc:
            raise CompilationError(
                f"unexpected {type(exc).__name__} while compiling: {exc}"
            ) from exc

    def _max_attempts(
        self,
        backend: SolverBackend,
        compiler: ModelCompiler,
        preferences: SolverPreferences,
    ) -> int:
        # §19: the exact backend enumerates every state, so retrying with a
        # larger penalty can never surface new feasible samples. 3a §16.3:
        # without a hard penalty there is no lever to turn, so a retry
        # would repeat the same submission. §14 step 14: remote retries
        # burn quota, so policy must opt in explicitly.
        if backend.is_exhaustive:
            return 1
        if not compiler.uses_hard_penalty:
            return 1
        if backend.capabilities.remote and not self._policy.allow_remote_retries:
            return 1
        return 1 + preferences.max_retries

    def _run_attempts(
        self,
        problem: OptimizationProblem,
        backend: SolverBackend,
        direction: str,
    ) -> SolveResult:
        """3a §16.2 steps 7–19: the compile/solve/validate loop.

        Everything from compiler selection onward runs under one ``try``
        whose last handler catches any ``Exception`` (2026-09-09 review
        F-03), so the ``solve`` docstring's promise — never raise, report
        ``solver_error`` — is kept by the service itself rather than
        delegated to every backend's own wrapping.
        """
        attempts: list[SolveAttempt] = []
        # Declared outside the try so a failure on a later attempt can still
        # report the last completed attempt's metadata.
        raw: RawSolverResult | None = None
        # The last attempt's diagnosis; each attempt overwrites it, so an
        # infeasible result explains the final (highest-penalty) attempt.
        infeasibility: InfeasibilityDiagnostics | None = None
        # Likewise the current penalty, so the NonFiniteModelError handler
        # can tell the hard-penalty path from a penalty-free one.
        penalty: float | None = None
        try:
            compiler = self._select_compiler(backend.capabilities)
            if compiler is None:
                declared = ", ".join(backend.capabilities.supported_model_types)
                return self._failure(
                    "configuration_error",
                    backend.name,
                    direction,
                    [
                        catalog_error(
                            "NO_COMPILER_FOR_MODEL_TYPE",
                            f"Backend '{backend.name}' accepts model types "
                            f"[{declared}] but the server has no compiler for "
                            f"any of them",
                        )
                    ],
                )
            preference_errors = self._preference_limit_errors(backend, problem.solver)
            if preference_errors:
                return self._failure(
                    "resource_limit_exceeded",
                    backend.name,
                    direction,
                    preference_errors,
                )

            # §14 step 9 / 3a §12.2, checked *before* compile (2026-09-09
            # review F-14): the estimate is pure arithmetic and equals the
            # compiled count, while compiling a large inequality is O(n²).
            # The post-compile check below stays as the final guarantee.
            variable_limit = self._policy.limit("variables")
            if backend.is_exhaustive:
                estimated = estimate_model_variables(problem, compiler.model_type)
                if estimated > variable_limit:
                    return self._failure(
                        "resource_limit_exceeded",
                        backend.name,
                        direction,
                        [self._exact_variable_limit_error(estimated, variable_limit)],
                    )

            max_attempts = self._max_attempts(backend, compiler, problem.solver)
            # §16.2 step 9: a native-constraint model has no hard penalty.
            penalty = (
                self._penalty_strategy.initial_penalty(problem)
                if compiler.uses_hard_penalty
                else None
            )
            # Both scales walk the whole problem, so they are only computed
            # when the line will actually be emitted (2026-09-09 review F-26).
            # On the hard-penalty path ``initial_penalty`` already walked it
            # once for the penalty scale, so the scale is recovered from the
            # penalty instead of walking the problem a second time; only the
            # native-constraint path, which computed no penalty, computes it.
            if logger.isEnabledFor(logging.INFO):
                multiplier = problem.solver.penalty_multiplier
                if penalty is not None and multiplier > 0:
                    penalty_scale = penalty / multiplier
                    scale_source = "derived: hard_penalty / penalty_multiplier"
                else:
                    penalty_scale = self._penalty_strategy.penalty_scale(problem)
                    scale_source = "computed"
                logger.info(
                    "Problem %s: model_type=%s, objective_scale=%s, "
                    "penalty_scale=%s (%s), initial hard_penalty=%s",
                    problem.name,
                    compiler.model_type,
                    self._penalty_strategy.objective_scale(problem),
                    penalty_scale,
                    scale_source,
                    penalty,
                )

            for attempt in range(1, max_attempts + 1):
                # 2026-09-09 review (F-07): the penalty ladder must stop
                # before it leaves the floating-point range. Checked before
                # compile so no backend ever sees an infinite penalty and
                # every recorded attempt keeps a finite value.
                if penalty is not None and not math.isfinite(penalty):
                    return self._penalty_overflow(
                        backend,
                        direction,
                        attempts,
                        penalty,
                        metadata=raw.metadata if raw is not None else None,
                    )
                compile_started = time.perf_counter()
                compiled = self._compile(compiler, problem, penalty)
                compile_ms = _elapsed_ms(compile_started)
                # §14 step 9 on the compiled model (slack included): the
                # final guarantee behind the estimate above. Never clamp,
                # never fall back.
                if (
                    backend.is_exhaustive
                    and compiled.num_variables > variable_limit
                ):
                    return self._failure(
                        "resource_limit_exceeded",
                        backend.name,
                        direction,
                        [
                            self._exact_variable_limit_error(
                                compiled.num_variables, variable_limit
                            )
                        ],
                        attempts=attempts,
                    )
                time_limit_error = self._effective_time_limit_error(
                    backend, compiled, problem.solver
                )
                if time_limit_error is not None:
                    return self._failure(
                        "resource_limit_exceeded",
                        backend.name,
                        direction,
                        [time_limit_error],
                        attempts=attempts,
                    )
                solve_started = time.perf_counter()
                raw = backend.solve(compiled, problem.solver)
                solve_ms = _elapsed_ms(solve_started)
                # 3a §22: the service, not the backend, knows the model
                # type; never create metadata a local backend did not.
                if raw.metadata is not None:
                    raw.metadata = raw.metadata.model_copy(
                        update={"model_type": compiled.model_type}
                    )
                # 3b §16 step 12a: the compiler strips internal columns and
                # decodes integer variables; the candidate pipeline only
                # ever sees business variables in problem order.
                validate_started = time.perf_counter()
                decoded = compiler.decode(compiled, raw)
                processed = process_candidates(
                    problem, decoded, frozenset(), problem.solver.top_k
                )
                solutions = processed.solutions
                unique_samples = processed.unique_samples
                feasible_samples = processed.feasible_samples
                infeasibility = processed.infeasibility
                validate_ms = _elapsed_ms(validate_started)
                attempts.append(
                    SolveAttempt(
                        attempt=attempt,
                        penalty=penalty,
                        samples_received=raw.num_samples,
                        unique_samples=unique_samples,
                        feasible_samples=feasible_samples,
                        compiled_variables=compiled.num_variables,
                        compiled_interactions=compiled.num_interactions,
                        compile_ms=compile_ms,
                        solve_ms=solve_ms,
                        validate_ms=validate_ms,
                    )
                )
                logger.info(
                    "Problem %s backend %s attempt %d: model_type=%s, "
                    "hard_penalty=%s, compiled_variables=%d, samples=%d, "
                    "unique=%d, feasible=%d, best_ranking_score=%s",
                    problem.name,
                    backend.name,
                    attempt,
                    compiled.model_type,
                    penalty,
                    compiled.num_variables,
                    raw.num_samples,
                    unique_samples,
                    feasible_samples,
                    solutions[0].ranking_score if solutions else None,
                )
                if solutions:
                    return SolveResult(
                        status="success",
                        backend=backend.name,
                        objective_direction=direction,
                        solutions=solutions,
                        attempts=attempts,
                        # Mirrors ``infeasibility_proven``: an exhaustive
                        # backend that returned samples enumerated every
                        # assignment, every business assignment was
                        # re-validated and ranked, so rank 1 is the global
                        # optimum of the ranking score.
                        optimality_proven=backend.is_exhaustive and raw.num_samples > 0,
                        metadata=raw.metadata,
                    )
                if attempt < max_attempts and penalty is not None:
                    penalty = self._penalty_strategy.next_penalty(penalty, attempt)

            warnings: list[SolveError] = []
            # Only a real cut deserves a warning: with max_retries == 0 the
            # user asked for a single attempt and policy blocked nothing.
            # 3a §16.2 step 17: only the hard-penalty path ever retries, so
            # only it can have had a retry cut by policy.
            remote_retries_blocked = (
                compiler.uses_hard_penalty
                and backend.capabilities.remote
                and not self._policy.allow_remote_retries
                and problem.solver.max_retries > 0
            )
            # An exhaustive backend proves infeasibility only by actually
            # enumerating assignments: zero samples back is not a proof.
            proven = (
                backend.is_exhaustive and raw is not None and raw.num_samples > 0
            )
            if proven:
                message = (
                    "No feasible solution exists: the exhaustive backend "
                    "enumerated every assignment"
                )
            elif backend.is_exhaustive:
                message = (
                    "No feasible solution found: the exhaustive backend "
                    "returned no samples, so infeasibility is not proven"
                )
            elif not compiler.uses_hard_penalty:
                # §16.2 step 16 / §16.3: one attempt, nothing to retune.
                message = (
                    "No feasible solution found: the constraint-model backend "
                    "returned no sample satisfying the hard constraints under "
                    "independent validation; infeasibility is not proven"
                )
            elif remote_retries_blocked:
                message = (
                    "No feasible solution found in 1 attempt; server policy "
                    "disables automatic remote retries"
                )
                warnings.append(
                    catalog_error(
                        "REMOTE_RETRIES_DISABLED",
                        "1 attempt was made; server policy disables "
                        "automatic remote retries to protect quota",
                    )
                )
            else:
                message = (
                    f"No feasible solution found in {len(attempts)} attempt(s); "
                    "the problem may still be feasible under a different "
                    "solver configuration"
                )
            return SolveResult(
                status="infeasible",
                backend=backend.name,
                objective_direction=direction,
                solutions=[],
                attempts=attempts,
                infeasibility_proven=proven,
                infeasibility=infeasibility,
                warnings=warnings,
                # Timing/usage facts from the last attempt still matter to
                # the caller (e.g. quota spent on a remote solve).
                metadata=raw.metadata if raw is not None else None,
                message=message,
            )
        except NonFiniteModelError as exc:
            # F-07, second guard: the penalty itself was finite but the
            # compiled biases are not (penalty × coefficient² overflowed).
            # On the hard-penalty path that is the penalty ladder hitting
            # the float ceiling; without a penalty it is a coefficient
            # problem and falls through to the compilation error below.
            if penalty is not None:
                logger.warning("Problem %s penalty overflow: %s", problem.name, exc)
                return self._penalty_overflow(
                    backend,
                    direction,
                    attempts,
                    penalty,
                    metadata=raw.metadata if raw is not None else None,
                )
            logger.warning("Problem %s compilation error: %s", problem.name, exc)
            return self._failure(
                "invalid_problem",
                backend.name,
                direction,
                [catalog_error("COMPILATION_FAILED", str(exc))],
                attempts=attempts,
            )
        except CompilationError as exc:
            # The problem passed validation but the compiler still refused
            # it. No backend was called, so this is a problem-side failure
            # (and a validator/compiler mismatch worth reporting), never a
            # solver error.
            logger.warning("Problem %s compilation error: %s", problem.name, exc)
            return self._failure(
                "invalid_problem",
                backend.name,
                direction,
                [catalog_error("COMPILATION_FAILED", str(exc))],
                attempts=attempts,
            )
        except OptimizerError as exc:
            # Remote backends redact their messages before raising; the
            # service must not rewrap them with unredacted content.
            logger.warning("Problem %s solver error: %s", problem.name, exc)
            code = getattr(exc, "code", None) or "SOLVER_ERROR"
            # 3b §20.8: a backend may name the status (e.g. configuration_error
            # for a request the remote side rejected as malformed).
            status = getattr(exc, "status", None) or "solver_error"
            return self._failure(
                status,
                backend.name,
                direction,
                [catalog_error(code, str(exc))],
                attempts=attempts,
                metadata=raw.metadata if raw is not None else None,
            )
        except Exception as exc:
            # 2026-09-09 review F-03: the service-level guarantee (Phase 2
            # §14 step 10, Phase 1 §36). Anything the handlers above did not
            # recognise — a third-party backend raising its vendor's
            # exception, a result missing a business column, a bug of our
            # own — is still reported as a structured solver_error rather
            # than escaping to the MCP/CLI caller. The text never passed
            # through a backend's redaction, so it is redacted here; the
            # class name stays because it is categorical, not secret. Only
            # ``Exception``: KeyboardInterrupt / SystemExit must propagate.
            message = redact(f"unexpected {type(exc).__name__}: {exc}")
            # The innermost frame keeps our own bugs traceable without
            # logging an unredacted traceback (spec §19).
            frames = traceback.extract_tb(exc.__traceback__)
            where = (
                f"{frames[-1].filename}:{frames[-1].lineno} in {frames[-1].name}"
                if frames
                else "unknown location"
            )
            logger.warning(
                "Problem %s backend %s: %s (at %s)",
                problem.name,
                backend.name,
                message,
                where,
            )
            return self._failure(
                "solver_error",
                backend.name,
                direction,
                [catalog_error("SOLVER_ERROR", message)],
                attempts=attempts,
                metadata=raw.metadata if raw is not None else None,
            )
