"""Orchestration service: validate, compile, solve, rank (spec §28).

The service drives the whole pipeline against the *original* problem:
feasibility and objective values are recomputed from the problem itself
(spec §2.2, §24), never taken from BQM energy. It only relies on the
``ModelCompiler`` / ``SolverBackend`` / ``PenaltyStrategy`` protocols,
never on the concrete compiled model type (spec §17).
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

from annealbridge.compiler import BQMCompiler
from annealbridge.compiler.base import ModelCompiler
from annealbridge.exceptions import CompilationError, OptimizerError
from annealbridge.models import (
    CompiledProblem,
    Objective,
    OptimizationProblem,
    Solution,
    SolveAttempt,
    SolveError,
    SolveResult,
    SolverExecutionMetadata,
    SolverPreferences,
    catalog_error,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.penalty import PenaltyStrategy, ScaledPenaltyStrategy
from annealbridge.solvers import (
    REASON_CONFIG_INVALID,
    REASON_CREDENTIALS_MISSING,
    REASON_NOT_INSTALLED,
    RawSolverResult,
    SolverBackend,
    SolverRegistry,
)
from annealbridge.validation import (
    validate_batch,
    validate_problem,
    validate_solution,
)

logger = logging.getLogger(__name__)


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

    Column ``j`` of ``samples`` is the 0/1 value of ``variables[j]``. Terms
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


def _assignment_tuple(sample: dict[str, int]) -> tuple[int, ...]:
    """Deterministic tie-break key: values ordered by variable name."""
    return tuple(sample[name] for name in sorted(sample))


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


@dataclass(frozen=True)
class CandidateSet:
    """Deduplicated business candidates, aligned row by row.

    ``samples`` holds one 0/1 row per distinct business assignment (column
    ``j`` is ``variables[j]``, in the solver's variable order minus the
    internal ones); ``energies`` is the minimum energy the solver reported
    for that assignment — kept for reporting and debugging only, never
    used for feasibility or ranking (overview principle 2).
    """

    variables: list[str]
    samples: np.ndarray  # int8, shape (candidates, len(variables))
    energies: np.ndarray  # float64, shape (candidates,)

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    def sample_dict(self, index: int) -> dict[str, int]:
        return dict(zip(self.variables, self.samples[index].tolist()))

    def as_pairs(self) -> list[tuple[dict[str, int], float]]:
        """``[(business_sample, energy), ...]`` — small-scale/test helper."""
        return [
            (self.sample_dict(index), float(self.energies[index]))
            for index in range(len(self))
        ]


def deduplicate_samples(
    raw: RawSolverResult, internal_variables: set[str]
) -> CandidateSet:
    """Strip internal variables and deduplicate by business assignment.

    Duplicates keep the minimum energy seen for that assignment (spec §25);
    among equal energies the earliest read wins, and candidates come back
    in order of first appearance, so the result is fully deterministic and
    identical to a row-by-row pass — it is just computed on the arrays.
    Energy is used here only to pick which duplicate's energy to report.
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
        )

    keys = _pack_rows(business)
    energies = np.asarray(raw.energies, dtype=np.float64)
    # Sort by assignment, then energy; the sort is stable, so within one
    # assignment equal energies stay in read order.
    order = _lexsort([*_words(keys), energies])
    sorted_keys = keys[order]
    group_start = np.empty(count, dtype=bool)
    group_start[0] = True
    np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1, out=group_start[1:])
    starts = np.flatnonzero(group_start)
    representatives = order[starts]  # min-energy read of each assignment
    first_seen = np.minimum.reduceat(order, starts)  # earliest read of each
    keep = representatives[np.argsort(first_seen, kind="stable")]
    return CandidateSet(
        variables=variables,
        samples=np.ascontiguousarray(business[keep]),
        energies=energies[keep],
    )


def process_candidates(
    problem: OptimizationProblem,
    raw: RawSolverResult,
    internal_variables: set[str],
    top_k: int,
) -> tuple[list[Solution], int, int]:
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
        return [], 0, 0
    minimize = problem.objective.direction == "minimize"

    verdict = validate_batch(problem, candidates.variables, candidates.samples)
    feasible = np.flatnonzero(verdict.feasible)
    if len(feasible) == 0:
        return [], len(candidates), 0

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
    tie_break = _pack_rows(feasible_samples[:, name_order])
    order = _lexsort(
        [sign * ranking_score, sign * objective_value, *_words(tie_break)]
    )

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
                hard_constraints_satisfied=validation.feasible,
                constraint_evaluations=validation.evaluations,
            )
        )
    return solutions, len(candidates), int(len(feasible))


# Categorical is_available() reasons → (result status, error code). Keyed on
# the availability-failure *category*, never on backend identity, so the
# service stays backend-agnostic (overview principle 4). Unknown reasons
# (and a bare ``(False, None)``) fall back to BACKEND_UNAVAILABLE.
_AVAILABILITY_MAP: dict[str, tuple[str, str]] = {
    REASON_NOT_INSTALLED: ("backend_unavailable", "BACKEND_NOT_INSTALLED"),
    REASON_CREDENTIALS_MISSING: ("backend_unavailable", "REMOTE_CREDENTIALS_MISSING"),
    REASON_CONFIG_INVALID: ("configuration_error", "DWAVE_CONFIG_INVALID"),
}


class OptimizationService:
    """End-to-end solve pipeline over pluggable components (spec §28)."""

    def __init__(
        self,
        compiler: ModelCompiler | None = None,
        penalty_strategy: PenaltyStrategy | None = None,
        registry: SolverRegistry | None = None,
        policy: ExecutionPolicy = ExecutionPolicy(),
    ) -> None:
        self._policy = policy
        self._compiler: ModelCompiler = compiler if compiler is not None else BQMCompiler()
        self._penalty_strategy: PenaltyStrategy = (
            penalty_strategy if penalty_strategy is not None else ScaledPenaltyStrategy()
        )
        self._registry: SolverRegistry = (
            registry if registry is not None else SolverRegistry.default()
        )
        # Concurrency slots are per service instance (spec §14); a CLI-style
        # single call is unaffected.
        self._solve_slots = threading.BoundedSemaphore(policy.max_concurrent_solves)

    def solve(self, problem: OptimizationProblem) -> SolveResult:
        """Solve ``problem`` and return a structured :class:`SolveResult`.

        Never raises for domain errors: validation and compilation failures
        become ``invalid_problem`` (no backend was ever called) and solver
        failures become ``solver_error`` (spec §27, §36).
        """
        errors = validate_problem(problem)
        if errors:
            logger.info(
                "Problem %s failed validation with %d error(s)",
                problem.name,
                len(errors),
            )
            return self._failure("invalid_problem", None, None, errors)

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
        status: str,
        backend: str | None,
        direction: str | None,
        errors: list[SolveError],
        *,
        attempts: list[SolveAttempt] | None = None,
        metadata: SolverExecutionMetadata | None = None,
    ) -> SolveResult:
        """Build a no-solutions SolveResult for a structured failure.

        ``metadata`` carries the last completed attempt's facts (e.g. quota
        spent on a remote solve) when a later attempt failed.
        """
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

    @staticmethod
    def _limit_error(
        code: str, label: str, value: object, maximum: object
    ) -> SolveError:
        """A §14 step 9 preference-limit error in the shared wording."""
        return catalog_error(
            code, f"{label} {value} exceeds the server maximum of {maximum}"
        )

    def _gate_backend(
        self, backend_name: str, backend: SolverBackend, direction: str
    ) -> SolveResult | None:
        """§14 steps 3–5: policy and availability gates, in spec order.

        Returns a failure result, or None when the backend may run. Never
        substitutes another backend (no silent fallback).

        ``enabled_backends`` holds registry keys — the names the user
        requests and the capabilities view reports — so the gate compares
        ``backend_name`` (the requested key), not ``capabilities.name``,
        which a custom registry may register under a different key.
        """
        caps = backend.capabilities
        if (
            self._policy.enabled_backends is not None
            and backend_name not in self._policy.enabled_backends
        ):
            return self._failure(
                "backend_unavailable",
                backend_name,
                direction,
                [
                    catalog_error(
                        "BACKEND_DISABLED_BY_POLICY",
                        f"Backend '{backend_name}' is disabled by server "
                        f"policy; enabled backends: "
                        f"{', '.join(sorted(self._policy.enabled_backends))}",
                    )
                ],
            )
        if caps.remote and not self._policy.allow_remote:
            return self._failure(
                "backend_unavailable",
                caps.name,
                direction,
                [
                    catalog_error(
                        "REMOTE_DISABLED",
                        f"Backend '{caps.name}' is remote and remote "
                        f"solving is disabled by server policy",
                    )
                ],
            )
        available, reason = backend.is_available()
        if not available:
            status, code = _AVAILABILITY_MAP.get(
                reason or "", ("backend_unavailable", "BACKEND_UNAVAILABLE")
            )
            return self._failure(
                status,
                caps.name,
                direction,
                [
                    catalog_error(
                        code,
                        f"Backend '{caps.name}' is unavailable: "
                        f"{reason or 'no reason reported'}",
                    )
                ],
            )
        return None

    def _preference_limit_errors(
        self, backend: SolverBackend, preferences: SolverPreferences
    ) -> list[SolveError]:
        """§14 step 9, preference-driven part. Never clamps.

        Dispatched on capabilities, not backend names: QPU-class limits
        apply to remote backends that take num_reads, the hybrid time limit
        to remote backends that take a time limit. All violations are
        collected so the caller can fix everything in one go.
        """
        caps = backend.capabilities
        policy = self._policy
        errors: list[SolveError] = []
        if caps.remote and caps.supports_num_reads:
            if preferences.num_reads > policy.max_qpu_reads:
                errors.append(
                    self._limit_error(
                        "QPU_READS_LIMIT",
                        "num_reads",
                        preferences.num_reads,
                        policy.max_qpu_reads,
                    )
                )
            annealing_time = (
                preferences.dwave_qpu.annealing_time_us
                if preferences.dwave_qpu is not None
                else None
            )
            if (
                annealing_time is not None
                and annealing_time > policy.max_qpu_annealing_time_us
            ):
                errors.append(
                    self._limit_error(
                        "QPU_ANNEALING_TIME_LIMIT",
                        "annealing_time_us",
                        annealing_time,
                        policy.max_qpu_annealing_time_us,
                    )
                )
        if caps.remote and caps.supports_time_limit:
            # Only the user's own value is known before compile. The
            # *effective* value (floored at the sampler minimum, which
            # depends on the compiled size) is checked per attempt in
            # _effective_time_limit_error, before anything is submitted.
            time_limit = (
                preferences.leap_hybrid_bqm.time_limit_seconds
                if preferences.leap_hybrid_bqm is not None
                else None
            )
            if (
                time_limit is not None
                and time_limit > policy.max_remote_time_seconds
            ):
                errors.append(
                    self._limit_error(
                        "REMOTE_TIME_LIMIT",
                        "time_limit_seconds",
                        time_limit,
                        policy.max_remote_time_seconds,
                    )
                )
        return errors

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
        maximum = self._policy.max_remote_time_seconds
        if effective is None or effective <= maximum:
            return None
        return catalog_error(
            "REMOTE_TIME_LIMIT",
            f"effective time_limit_seconds {effective} (the requested value "
            f"raised to the solver's minimum for this problem size) exceeds "
            f"the server maximum of {maximum}",
        )

    def _max_attempts(
        self, backend: SolverBackend, preferences: SolverPreferences
    ) -> int:
        # §19: the exact backend enumerates every state, so retrying with a
        # larger penalty can never surface new feasible samples. §14 step
        # 14: remote retries burn quota, so policy must opt in explicitly.
        if backend.is_exhaustive:
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
        """§14 steps 7–16: the compile/solve/validate loop."""
        preference_errors = self._preference_limit_errors(backend, problem.solver)
        if preference_errors:
            return self._failure(
                "resource_limit_exceeded",
                backend.name,
                direction,
                preference_errors,
            )

        attempts: list[SolveAttempt] = []
        # Declared outside the try so a failure on a later attempt can still
        # report the last completed attempt's metadata.
        raw: RawSolverResult | None = None
        try:
            max_attempts = self._max_attempts(backend, problem.solver)
            penalty = self._penalty_strategy.initial_penalty(problem)
            logger.info(
                "Problem %s: objective_scale=%s, penalty_scale=%s, "
                "initial hard_penalty=%s",
                problem.name,
                self._penalty_strategy.objective_scale(problem),
                self._penalty_strategy.penalty_scale(problem),
                penalty,
            )

            for attempt in range(1, max_attempts + 1):
                compiled = self._compiler.compile(problem, penalty)
                # §14 step 9: this limit needs compiled info (slack included),
                # so it runs after compile. Never clamp, never fall back.
                if (
                    backend.is_exhaustive
                    and compiled.num_variables > self._policy.exact_max_variables
                ):
                    return self._failure(
                        "resource_limit_exceeded",
                        backend.name,
                        direction,
                        [
                            catalog_error(
                                "EXACT_VARIABLE_LIMIT",
                                f"Compiled problem has "
                                f"{compiled.num_variables} variables "
                                f"(including internal), exceeding the "
                                f"exact solver limit of "
                                f"{self._policy.exact_max_variables}",
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
                raw = backend.solve(compiled, problem.solver)
                solutions, unique_samples, feasible_samples = process_candidates(
                    problem, raw, compiled.internal_variables, problem.solver.top_k
                )
                attempts.append(
                    SolveAttempt(
                        attempt=attempt,
                        penalty=penalty,
                        samples_received=raw.num_samples,
                        unique_samples=unique_samples,
                        feasible_samples=feasible_samples,
                    )
                )
                logger.info(
                    "Problem %s backend %s attempt %d: hard_penalty=%s, "
                    "compiled_variables=%d, samples=%d, unique=%d, feasible=%d, "
                    "best_ranking_score=%s",
                    problem.name,
                    backend.name,
                    attempt,
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
                        metadata=raw.metadata,
                    )
                if attempt < max_attempts:
                    penalty = self._penalty_strategy.next_penalty(penalty, attempt)

            warnings: list[SolveError] = []
            # Only a real cut deserves a warning: with max_retries == 0 the
            # user asked for a single attempt and policy blocked nothing.
            remote_retries_blocked = (
                backend.capabilities.remote
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
                warnings=warnings,
                # Timing/usage facts from the last attempt still matter to
                # the caller (e.g. quota spent on a remote solve).
                metadata=raw.metadata if raw is not None else None,
                message=message,
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
            return self._failure(
                "solver_error",
                backend.name,
                direction,
                [catalog_error(code, str(exc))],
                attempts=attempts,
                metadata=raw.metadata if raw is not None else None,
            )
