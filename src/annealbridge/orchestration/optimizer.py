"""Orchestration service: validate, compile, solve, rank (spec §28).

The service drives the whole pipeline against the *original* problem:
feasibility and objective values are recomputed from the problem itself
(spec §2.2, §24), never taken from BQM energy. It only relies on the
``ModelCompiler`` / ``SolverBackend`` / ``PenaltyStrategy`` protocols,
never on the concrete compiled model type (spec §17).
"""

import logging
import threading

from annealbridge.compiler import BQMCompiler
from annealbridge.compiler.base import ModelCompiler
from annealbridge.exceptions import OptimizerError
from annealbridge.models import (
    RECOMMENDED_ACTIONS,
    RETRYABLE_CODES,
    Objective,
    OptimizationProblem,
    Solution,
    SolveAttempt,
    SolveError,
    SolveResult,
    SolverPreferences,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.penalty import PenaltyStrategy, ScaledPenaltyStrategy
from annealbridge.solvers import (
    RawSolverResult,
    SolverBackend,
    SolverRegistry,
)
from annealbridge.validation import validate_problem, validate_solution

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


def _assignment_tuple(sample: dict[str, int]) -> tuple[int, ...]:
    """Deterministic tie-break key: values ordered by variable name."""
    return tuple(sample[name] for name in sorted(sample))


def deduplicate_samples(
    raw: RawSolverResult, internal_variables: set[str]
) -> list[tuple[dict[str, int], float]]:
    """Strip internal variables and deduplicate by business assignment.

    Duplicates keep the minimum energy seen for that assignment (spec §25).
    """
    best: dict[tuple[int, ...], tuple[dict[str, int], float]] = {}
    for sample, energy in zip(raw.samples, raw.energies):
        business = {
            name: value
            for name, value in sample.items()
            if name not in internal_variables
        }
        key = _assignment_tuple(business)
        kept = best.get(key)
        if kept is None or energy < kept[1]:
            best[key] = (business, energy)
    return list(best.values())


def process_candidates(
    problem: OptimizationProblem,
    raw: RawSolverResult,
    internal_variables: set[str],
    top_k: int,
) -> tuple[list[Solution], int, int]:
    """Run the §25 candidate pipeline on raw solver output.

    Deduplicates, validates against the original problem, keeps feasible
    candidates only, computes objective/soft-violation/ranking scores, and
    returns the top ``top_k`` solutions ranked from 1, plus the unique and
    feasible candidate counts.
    """
    deduped = deduplicate_samples(raw, internal_variables)
    minimize = problem.objective.direction == "minimize"

    scored: list[tuple[dict[str, int], float, object, float, float]] = []
    for sample, energy in deduped:
        validation = validate_solution(problem, sample)
        if not validation.feasible:
            continue
        objective_value = evaluate_objective(problem.objective, sample)
        if minimize:
            ranking_score = objective_value + validation.soft_violation_score
        else:
            ranking_score = objective_value - validation.soft_violation_score
        scored.append((sample, energy, validation, objective_value, ranking_score))

    # §25.1: ascending ranking_score for minimize, descending for maximize;
    # ties broken by objective_value in the same direction, then by the
    # name-sorted assignment tuple so the order is fully deterministic.
    sign = 1.0 if minimize else -1.0
    scored.sort(
        key=lambda item: (
            sign * item[4],
            sign * item[3],
            _assignment_tuple(item[0]),
        )
    )

    solutions = [
        Solution(
            rank=rank,
            variables=sample,
            objective_value=objective_value,
            soft_violation_score=validation.soft_violation_score,
            ranking_score=ranking_score,
            energy=energy,
            hard_constraints_satisfied=True,
            constraint_evaluations=validation.evaluations,
        )
        for rank, (sample, energy, validation, objective_value, ranking_score) in
        enumerate(scored[:top_k], start=1)
    ]
    return solutions, len(deduped), len(scored)


# Categorical is_available() reasons → (result status, error code). Keyed on
# the availability-failure *category*, never on backend identity, so the
# service stays backend-agnostic (overview principle 4). Unknown reasons
# (and a bare ``(False, None)``) fall back to BACKEND_UNAVAILABLE.
_AVAILABILITY_MAP: dict[str, tuple[str, str]] = {
    "dwave-system not installed": ("backend_unavailable", "BACKEND_NOT_INSTALLED"),
    "D-Wave credentials not configured": (
        "backend_unavailable",
        "REMOTE_CREDENTIALS_MISSING",
    ),
    "D-Wave configuration invalid": ("configuration_error", "DWAVE_CONFIG_INVALID"),
}


def _catalog_error(code: str, message: str, *, path: str | None = None) -> SolveError:
    """Build a SolveError with the catalog's fixed recommended_action.

    Every service-produced error goes through here so recommended_action
    and retryable stay consistent per code (spec §13.2).
    """
    return SolveError(
        code=code,
        path=path,
        message=message,
        retryable=code in RETRYABLE_CODES,
        recommended_action=RECOMMENDED_ACTIONS.get(code),
    )


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

        Never raises for domain errors: validation failures become
        ``invalid_problem`` and solver/compiler failures become
        ``solver_error`` (spec §27, §36).
        """
        errors = validate_problem(problem)
        if errors:
            logger.info(
                "Problem %s failed validation with %d error(s)",
                problem.name,
                len(errors),
            )
            return SolveResult(
                status="invalid_problem",
                backend=None,
                objective_direction=None,
                solutions=[],
                attempts=[],
                errors=errors,
            )

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
                [_catalog_error("UNKNOWN_BACKEND", message)],
            )

        gate_result = self._gate_backend(backend, direction)
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
                    _catalog_error(
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
    ) -> SolveResult:
        """Build a no-solutions SolveResult for a structured failure."""
        return SolveResult(
            status=status,
            backend=backend,
            objective_direction=direction,
            solutions=[],
            attempts=attempts or [],
            errors=errors,
            message=errors[0].message,
        )

    def _gate_backend(
        self, backend: SolverBackend, direction: str
    ) -> SolveResult | None:
        """§14 steps 3–5: policy and availability gates, in spec order.

        Returns a failure result, or None when the backend may run. Never
        substitutes another backend (no silent fallback).
        """
        caps = backend.capabilities
        if (
            self._policy.enabled_backends is not None
            and caps.name not in self._policy.enabled_backends
        ):
            return self._failure(
                "backend_unavailable",
                caps.name,
                direction,
                [
                    _catalog_error(
                        "BACKEND_DISABLED_BY_POLICY",
                        f"Backend '{caps.name}' is disabled by server "
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
                    _catalog_error(
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
                    _catalog_error(
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
                    _catalog_error(
                        "QPU_READS_LIMIT",
                        f"num_reads {preferences.num_reads} exceeds the "
                        f"server maximum of {policy.max_qpu_reads}",
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
                    _catalog_error(
                        "QPU_ANNEALING_TIME_LIMIT",
                        f"annealing_time_us {annealing_time} exceeds the "
                        f"server maximum of {policy.max_qpu_annealing_time_us}",
                    )
                )
        if caps.remote and caps.supports_time_limit:
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
                    _catalog_error(
                        "REMOTE_TIME_LIMIT",
                        f"time_limit_seconds {time_limit} exceeds the "
                        f"server maximum of {policy.max_remote_time_seconds}",
                    )
                )
        return errors

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
        try:
            max_attempts = self._max_attempts(backend, problem.solver)
            penalty = self._penalty_strategy.initial_penalty(problem)
            raw: RawSolverResult | None = None

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
                            _catalog_error(
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
                raw = backend.solve(compiled, problem.solver)
                solutions, unique_samples, feasible_samples = process_candidates(
                    problem, raw, compiled.internal_variables, problem.solver.top_k
                )
                attempts.append(
                    SolveAttempt(
                        attempt=attempt,
                        penalty=penalty,
                        samples_received=len(raw.samples),
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
                    len(raw.samples),
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
            remote_retries_blocked = (
                backend.capabilities.remote
                and not self._policy.allow_remote_retries
            )
            if backend.is_exhaustive:
                message = (
                    "No feasible solution exists: the exhaustive backend "
                    "enumerated every assignment"
                )
            elif remote_retries_blocked:
                message = (
                    "No feasible solution found in 1 attempt; server policy "
                    "disables automatic remote retries"
                )
                warnings.append(
                    _catalog_error(
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
                infeasibility_proven=backend.is_exhaustive,
                warnings=warnings,
                # Timing/usage facts from the last attempt still matter to
                # the caller (e.g. quota spent on a remote solve).
                metadata=raw.metadata if raw is not None else None,
                message=message,
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
                [_catalog_error(code, str(exc))],
                attempts=attempts,
            )
