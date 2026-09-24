"""Orchestration service: validate, compile, solve, rank (spec §28).

The service drives the whole pipeline against the *original* problem:
feasibility and objective values are recomputed from the problem itself
(spec §2.2, §24), never taken from BQM energy. It only relies on the
``ModelCompiler`` / ``SolverBackend`` / ``PenaltyStrategy`` protocols,
never on the concrete compiled model type (spec §17).
"""

import dataclasses
import inspect
import logging
import math
import threading
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

from annealbridge.compiler import BQMCompiler, CQMCompiler
from annealbridge.compiler.base import ModelCompiler, PreparedModel, SupportsPrepare
from annealbridge.exceptions import (
    CompilationError,
    NonFiniteModelError,
    OptimizerError,
    SolveCancelled,
)
from annealbridge.interrupt import CancelToken, Clock, Interrupt
from annealbridge.models import (
    CompiledProblem,
    InfeasibilityDiagnostics,
    ModelType,
    OptimizationProblem,
    SolveAttempt,
    SolveError,
    SolverCapabilities,
    SolveResult,
    SolverExecutionMetadata,
    SolverPreferences,
    SolveStatus,
    catalog_error,
)
from annealbridge.orchestration.candidates import process_candidates
from annealbridge.orchestration.limits import (
    compiled_variable_limit_error,
    exact_variable_limit_error,
    gate_errors,
    no_compiler_error,
    postprocess_limit_errors,
    preference_limit_errors,
    read_preference,
    select_model_type,
)
from annealbridge.orchestration.messages import (
    _BudgetCut,
    _infeasible_message,
    _postprocess_limit_warnings,
    _success_message,
    _wall_clock_limit_warnings,
)
from annealbridge.orchestration.policy import ExecutionPolicy
from annealbridge.orchestration.postprocess import PostprocessRequest
from annealbridge.orchestration.progress import (
    ProgressCallback,
    SolveProgress,
    Stage,
)
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
    ProblemValidationResult,
    validate_problem_full,
)
from annealbridge.validation.estimates import estimate_model_variables
from annealbridge.version import package_version

logger = logging.getLogger(__name__)


def _elapsed_ms(started: float) -> float:
    """Milliseconds of wall clock since ``started`` (a ``perf_counter`` value).

    Rounded to the microsecond: finer digits are clock noise, not information.
    """
    return round((time.perf_counter() - started) * 1000.0, 3)


class _AttemptBudget(NamedTuple):
    """How many attempts a solve may make, and why fewer than asked (if so).

    ``cut_reason`` is ``None`` when nothing was cut. That includes a remote
    backend whose retries policy disables when the user asked for
    ``max_retries == 0`` anyway: no retry was wanted, so none was cut.
    """

    max_attempts: int
    cut_reason: _BudgetCut | None


@dataclass(frozen=True)
class _AttemptPlan:
    """What ``_prepare_attempts`` settles before the first attempt."""

    compiler: ModelCompiler
    variable_limit: float | int
    budget: _AttemptBudget
    initial_penalty: float | None
    # Batch 4 (G): None unless post-processing was asked for and the backend
    # is not exhaustive (postprocess spec §3).
    postprocess: PostprocessRequest | None = None


@dataclass
class _AttemptState:
    """Mutable facts the attempt loop shares with ``_run_attempts``'s handlers.

    ``last_metadata`` is the last completed attempt's metadata (model type
    already filled in), assigned in exactly one place so no failure path
    can forget it (2026-09-11 review F08). ``raw`` is kept for the
    infeasibility proof (``num_samples``); ``infeasibility`` is the last
    attempt's diagnosis, so an infeasible result explains the final
    (highest-penalty) attempt.

    ``on_progress`` and ``max_attempts`` carry the caller's progress callback
    to ``_solve_attempt`` without widening its signature; ``max_attempts``
    is set once the attempt budget is known.

    ``prepared`` is the hard-penalty-independent part of the compile when
    the compiler offers one (:class:`SupportsPrepare`): made by the first
    attempt and reused by the retries, which only differ in the penalty.
    It lives exactly as long as this state, one solve; nothing is cached
    across solves.
    """

    attempts: list[SolveAttempt] = field(default_factory=list)
    raw: RawSolverResult | None = None
    last_metadata: SolverExecutionMetadata | None = None
    infeasibility: InfeasibilityDiagnostics | None = None
    on_progress: ProgressCallback | None = None
    max_attempts: int = 0
    prepared: PreparedModel | None = None
    # ``(attempt, limit names)`` of every attempt whose post-processing
    # stopped at a ceiling; folded into one warning (postprocess spec §6).
    postprocess_limits: list[tuple[int, list[str]]] = field(default_factory=list)
    # Batch 6 (J): the solve's stop condition, None when it has neither a
    # wall-clock limit nor a cancel token (the solve then runs exactly as
    # before); and whether the limit stopped a retry from starting.
    interrupt: Interrupt | None = None
    retry_skipped: bool = False

    def check_cancelled(self) -> None:
        """A service checkpoint: raise :class:`SolveCancelled` once the
        caller has cancelled. Cancellation wins over the wall-clock limit
        wherever both could apply."""
        if self.interrupt is not None and self.interrupt.cancelled:
            raise SolveCancelled("the solve was cancelled by its caller")

    @property
    def wall_clock_limit_reached(self) -> bool:
        """The limit left work undone: an attempt was cut short or a retry
        was not started. Never true merely because the uninterruptible
        stages (validation, compilation, re-validation) ran past it."""
        return self.retry_skipped or any(
            attempt.wall_clock_limit_reached for attempt in self.attempts
        )

    def emit(self, attempt: int, stage: Stage, backend_name: str) -> None:
        """Call the progress callback, if any, for a stage that is starting.

        The callback runs on the solving thread and outside every
        per-attempt timing window. Whatever it raises is logged and dropped
        here, on purpose:
        ``_run_attempts`` turns any exception into ``solver_error``, and a
        progress listener (a host that went away mid-solve) must never
        change the result.
        """
        if self.on_progress is None:
            return
        event = SolveProgress(
            attempt=attempt,
            max_attempts=self.max_attempts,
            stage=stage,
            backend=backend_name,
        )
        try:
            self.on_progress(event)
        except Exception as exc:
            logger.warning(
                "progress callback raised %s: %s; solving continues",
                type(exc).__name__,
                exc,
            )


class OptimizationService:
    """End-to-end solve pipeline over pluggable components (spec §28)."""

    def __init__(
        self,
        compilers: Iterable[ModelCompiler] | None = None,
        penalty_strategy: PenaltyStrategy | None = None,
        registry: SolverRegistry | None = None,
        policy: ExecutionPolicy = ExecutionPolicy(),
        *,
        clock: Clock = time.perf_counter,
    ) -> None:
        """``clock`` measures ``elapsed_ms`` and the wall-clock limit from
        the same reading; a test may inject a fake one."""
        self._policy = policy
        self._clock = clock
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
        self._check_interrupt_signatures()
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

    def _check_interrupt_signatures(self) -> None:
        """A backend that declares ``supports_interrupt`` must take ``interrupt``.

        The service passes ``interrupt=`` to exactly those backends; one
        whose ``solve`` has no such keyword would fail only on the first
        limited or cancellable solve, as an unclassified ``TypeError``.
        Refused here, at construction, like a limit the policy cannot check.
        """
        for name in self._registry.names():
            backend = self._registry.get(name)
            if not backend.capabilities.supports_interrupt:
                continue
            parameters = inspect.signature(backend.solve).parameters
            accepts = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ) or (
                "interrupt" in parameters
                and parameters["interrupt"].kind
                in (
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            if not accepts:
                raise ValueError(
                    f"backend '{name}' declares supports_interrupt but its "
                    f"solve() takes no 'interrupt' keyword argument"
                )

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
            max_compiled_variables=int(self._policy.required_limit("variables")),
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
                no_compiler_error(backend_name, caps, path="solver.backend")
            )
        return result

    def recommend(self, problem: OptimizationProblem) -> BackendRecommendationResult:
        """Rank the registry's backends for ``problem`` (3a §23.5).

        Advisory only: ``solve`` never reads this. Nothing is compiled or
        solved, no network is touched and no concurrency slot is taken.
        """
        return recommend(problem, self._registry, self._policy, self._compilers)

    def solve(
        self,
        problem: OptimizationProblem,
        *,
        on_progress: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> SolveResult:
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

        ``on_progress``, when given, is called with a
        :class:`~annealbridge.orchestration.progress.SolveProgress` as each
        stage (compile, solve, validate) of each attempt starts, on this
        thread. An exception it raises is logged and ignored; it cannot
        change the result. Without it the behaviour is exactly as before.

        ``solver.wall_clock_limit_seconds`` (batch 6 J) is measured from
        entering this method, the same instant ``elapsed_ms`` starts from.
        When it runs out the solve stops at its next checkpoint and returns
        what it completed, with ``wall_clock_limit_reached`` and a
        WALL_CLOCK_LIMIT_REACHED warning.

        ``cancel``, when given, is polled at the same checkpoints; once it
        is cancelled the solve stops and raises
        :class:`~annealbridge.exceptions.SolveCancelled` instead of
        returning -- the one exception this method raises by design. Every
        thread the solve started has returned and its concurrency slot has
        been released by then. A request that fails before the attempts
        begin (validation, an unavailable backend, a full server) returns
        its result as usual. Without a limit and a token no interrupt
        exists at all and the solve runs exactly as before.
        """
        started = self._clock()
        limit = problem.solver.wall_clock_limit_seconds
        interrupt = (
            Interrupt(
                deadline=None if limit is None else started + limit,
                token=cancel,
                clock=self._clock,
            )
            if limit is not None or cancel is not None
            else None
        )
        validation, _, _ = self._validate_against_backend(problem)
        if validation.errors:
            logger.info(
                "Problem %s failed validation with %d error(s)",
                problem.name,
                len(validation.errors),
            )
            result = self._failure("invalid_problem", None, None, validation.errors)
        else:
            result = self._dispatch(
                problem, on_progress=on_progress, interrupt=interrupt
            )
        # Every path is stamped the same way: the service's own wall clock
        # (independent of the vendor-reported ``metadata.timing_us``) and
        # the package version that produced the result.
        return result.model_copy(
            update={
                "warnings": [*validation.warnings, *result.warnings],
                "elapsed_ms": round((self._clock() - started) * 1000.0, 3),
                "annealbridge_version": package_version(),
            }
        )

    def _dispatch(
        self,
        problem: OptimizationProblem,
        *,
        on_progress: ProgressCallback | None = None,
        interrupt: Interrupt | None = None,
    ) -> SolveResult:
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
            return self._run_attempts(
                problem,
                backend,
                direction,
                on_progress=on_progress,
                interrupt=interrupt,
            )
        finally:
            # Also on SolveCancelled: by then every thread the solve
            # started has returned, so the slot is free for real.
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
        maximum = self._policy.required_limit("time_seconds")
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
    def _compile(
        compiler: ModelCompiler,
        problem: OptimizationProblem,
        penalty: float | None,
        state: _AttemptState,
    ) -> CompiledProblem:
        """``compiler.compile`` with every failure expressed as a CompilationError.

        A compiler that offers :class:`SupportsPrepare` is prepared once per
        solve (on the first attempt, so its time counts in that attempt's
        ``compile_ms``) and each attempt finishes the compile with its own
        penalty; by the protocol's contract the model is bit for bit what
        ``compiler.compile(problem, penalty)`` returns. The check is on the
        capability, never on a compiler's name; any other compiler is
        compiled from scratch each attempt.

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
            if isinstance(compiler, SupportsPrepare):
                if state.prepared is None:
                    state.prepared = compiler.prepare(problem)
                return state.prepared.compile(penalty)
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
    ) -> _AttemptBudget:
        # §19: the exact backend enumerates every state, so retrying with a
        # larger penalty can never surface new feasible samples. 3a §16.3:
        # without a hard penalty there is no lever to turn, so a retry
        # would repeat the same submission. §14 step 14: remote retries
        # burn quota, so policy must opt in explicitly.
        if backend.is_exhaustive:
            return _AttemptBudget(1, "exhaustive")
        if not compiler.uses_hard_penalty:
            return _AttemptBudget(1, "no_hard_penalty")
        # Only a real cut deserves a warning: with max_retries == 0 the
        # user asked for a single attempt and policy blocked nothing.
        # 3a §16.2 step 17: only the hard-penalty path ever retries, so
        # only it can have had a retry cut by policy.
        if backend.capabilities.remote and not self._policy.allow_remote_retries:
            return _AttemptBudget(
                1, "remote_retries_disabled" if preferences.max_retries > 0 else None
            )
        return _AttemptBudget(1 + preferences.max_retries, None)

    def _prepare_attempts(
        self,
        problem: OptimizationProblem,
        backend: SolverBackend,
        direction: str,
    ) -> SolveResult | _AttemptPlan:
        """3a §16.2 steps 7–10: what is settled before the first attempt.

        Returns the plan, or the structured failure that stops the solve.
        """
        compiler = self._select_compiler(backend.capabilities)
        if compiler is None:
            return self._failure(
                "configuration_error",
                backend.name,
                direction,
                [no_compiler_error(backend.name, backend.capabilities)],
            )
        preference_errors = [
            *self._preference_limit_errors(backend, problem.solver),
            *postprocess_limit_errors(problem, backend.capabilities, self._policy),
        ]
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
        variable_limit = self._policy.required_limit("variables")
        if backend.is_exhaustive:
            estimated = estimate_model_variables(problem, compiler.model_type)
            if estimated > variable_limit:
                return self._failure(
                    "resource_limit_exceeded",
                    backend.name,
                    direction,
                    [exact_variable_limit_error(estimated, variable_limit)],
                )
        # The same step for a backend that declares a ceiling of its own
        # (``compiled_variable_limit``): flag-driven like the exhaustive one,
        # never by name, and again refused before the O(n²) compile.
        if backend.capabilities.compiled_variable_limit is not None:
            estimated = estimate_model_variables(problem, compiler.model_type)
            declared_error = compiled_variable_limit_error(
                backend.capabilities, estimated
            )
            if declared_error is not None:
                return self._failure(
                    "resource_limit_exceeded", backend.name, direction, [declared_error]
                )

        budget = self._max_attempts(backend, compiler, problem.solver)
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
        # Postprocess spec §3: an exhaustive backend already returned every
        # assignment, so there is nothing to repair or improve (the
        # validator warns PARAMETER_IGNORED); decided by the flag, never by
        # the backend's name.
        postprocess = None
        if problem.solver.postprocess != "none" and not backend.capabilities.exhaustive:
            postprocess = PostprocessRequest(
                candidates=problem.solver.postprocess_candidates,
                max_evaluations=int(
                    self._policy.required_limit("postprocess_evaluations")
                ),
            )
        return _AttemptPlan(
            compiler=compiler,
            variable_limit=variable_limit,
            budget=budget,
            initial_penalty=penalty,
            postprocess=postprocess,
        )

    def _solve_attempt(
        self,
        problem: OptimizationProblem,
        backend: SolverBackend,
        plan: _AttemptPlan,
        direction: str,
        attempt: int,
        penalty: float | None,
        state: _AttemptState,
    ) -> SolveResult | None:
        """3a §16.2 steps 11–15 for one attempt.

        Returns the success result, a structured failure, or ``None`` when
        no feasible sample came back and the ladder may continue.
        """
        compiler = plan.compiler
        # 2026-09-09 review (F-07): the penalty ladder must stop
        # before it leaves the floating-point range. Checked before
        # compile so no backend ever sees an infinite penalty and
        # every recorded attempt keeps a finite value.
        if penalty is not None and not math.isfinite(penalty):
            return self._penalty_overflow(
                backend,
                direction,
                state.attempts,
                penalty,
                metadata=state.last_metadata,
            )
        state.emit(attempt, "compile", backend.name)
        compile_started = time.perf_counter()
        compiled = self._compile(compiler, problem, penalty, state)
        compile_ms = _elapsed_ms(compile_started)
        # §14 step 9 on the compiled model (slack included): the
        # final guarantee behind the estimate above. Never clamp,
        # never fall back.
        if backend.is_exhaustive and compiled.num_variables > plan.variable_limit:
            return self._failure(
                "resource_limit_exceeded",
                backend.name,
                direction,
                [
                    exact_variable_limit_error(
                        compiled.num_variables, plan.variable_limit
                    )
                ],
                attempts=state.attempts,
            )
        declared_error = compiled_variable_limit_error(
            backend.capabilities, compiled.num_variables
        )
        if declared_error is not None:
            return self._failure(
                "resource_limit_exceeded",
                backend.name,
                direction,
                [declared_error],
                attempts=state.attempts,
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
                attempts=state.attempts,
            )
        state.check_cancelled()
        state.emit(attempt, "solve", backend.name)
        solve_started = time.perf_counter()
        # Batch 6 (J): the interrupt goes only to a backend that declared it
        # can stop part-way (decided by the flag, never by the name); every
        # other call is the plain two-argument one it always was.
        if state.interrupt is not None and backend.capabilities.supports_interrupt:
            raw = backend.solve(compiled, problem.solver, interrupt=state.interrupt)
        else:
            raw = backend.solve(compiled, problem.solver)
        solve_ms = _elapsed_ms(solve_started)
        state.check_cancelled()
        # 3a §22: the service, not the backend, knows the model
        # type; never create metadata a local backend did not.
        if raw.metadata is not None:
            raw.metadata = raw.metadata.model_copy(
                update={"model_type": compiled.model_type}
            )
        state.raw = raw
        state.last_metadata = raw.metadata
        state.emit(attempt, "validate", backend.name)
        # 3b §16 step 12a: the compiler strips internal columns and
        # decodes integer variables; the candidate pipeline only
        # ever sees business variables in problem order.
        validate_started = time.perf_counter()
        decoded = compiler.decode(compiled, raw)
        postprocess = plan.postprocess
        if postprocess is not None and state.interrupt is not None:
            postprocess = dataclasses.replace(
                postprocess, should_stop=state.interrupt.should_stop
            )
        processed = process_candidates(
            problem, decoded, frozenset(), problem.solver.top_k, postprocess
        )
        state.check_cancelled()
        solutions = processed.solutions
        unique_samples = processed.unique_samples
        feasible_samples = processed.feasible_samples
        state.infeasibility = processed.infeasibility
        # Post-processing runs inside the candidate pipeline but is timed on
        # its own, so the two windows never overlap (postprocess spec §5).
        validate_ms = _elapsed_ms(validate_started) - (processed.postprocess_ms or 0.0)
        if processed.postprocess is not None and processed.postprocess.limit_reached:
            state.postprocess_limits.append(
                (attempt, list(processed.postprocess.limit_reached))
            )
        current = SolveAttempt(
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
            postprocess=processed.postprocess,
            postprocess_ms=processed.postprocess_ms,
            # Only a stop that left work undone: reads the backend skipped,
            # or post-processing ended with starts or steps left.
            wall_clock_limit_reached=raw.interrupted
            or (
                processed.postprocess is not None
                and processed.postprocess.wall_clock_limit_reached
            ),
        )
        state.attempts.append(current)
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
            # Mirrors ``infeasibility_proven``: an exhaustive backend
            # that returned samples enumerated every assignment, every
            # business assignment was re-validated and ranked, so rank 1
            # is the global optimum of the ranking score. Computed once
            # so the field and the message can never disagree.
            optimality_proven = backend.is_exhaustive and raw.num_samples > 0
            return SolveResult(
                status="success",
                backend=backend.name,
                objective_direction=direction,
                solutions=solutions,
                attempts=state.attempts,
                optimality_proven=optimality_proven,
                warnings=_postprocess_limit_warnings(state.postprocess_limits),
                metadata=raw.metadata,
                message=_success_message(
                    backend.name, direction, optimality_proven, current, solutions
                ),
            )
        return None

    def _run_attempts(
        self,
        problem: OptimizationProblem,
        backend: SolverBackend,
        direction: str,
        *,
        on_progress: ProgressCallback | None = None,
        interrupt: Interrupt | None = None,
    ) -> SolveResult:
        """3a §16.2 steps 7–19: the compile/solve/validate loop.

        Everything from compiler selection onward runs under one ``try``
        whose last handler catches any ``Exception`` (2026-09-09 review
        F-03), so the ``solve`` docstring's promise — never raise, report
        ``solver_error`` — is kept by the service itself rather than
        delegated to every backend's own wrapping. The one exception let
        through is :class:`SolveCancelled`, which the caller asked for.

        Whatever the loop returns is stamped with the wall-clock facts in
        one place (:meth:`_stamp_wall_clock`), so no status path can forget
        them.
        """
        state = _AttemptState(on_progress=on_progress, interrupt=interrupt)
        result = self._attempt_loop(problem, backend, direction, state)
        return self._stamp_wall_clock(result, state)

    @staticmethod
    def _stamp_wall_clock(result: SolveResult, state: _AttemptState) -> SolveResult:
        """Mark a result the wall-clock limit cut short (batch 6 J).

        ``wall_clock_limit_reached`` and one WALL_CLOCK_LIMIT_REACHED
        warning, whatever the status; on success the message says so as
        well (the infeasible message already does). A result nothing cut is
        returned untouched.
        """
        if not state.wall_clock_limit_reached:
            return result
        cut = [
            attempt.attempt
            for attempt in state.attempts
            if attempt.wall_clock_limit_reached
        ]
        update: dict[str, object] = {
            "wall_clock_limit_reached": True,
            "warnings": [
                *result.warnings,
                *_wall_clock_limit_warnings(cut, state.retry_skipped),
            ],
        }
        if result.status == "success" and result.message is not None:
            update["message"] = (
                f"{result.message} The wall-clock limit cut the search short."
            )
        return result.model_copy(update=update)

    def _attempt_loop(
        self,
        problem: OptimizationProblem,
        backend: SolverBackend,
        direction: str,
        state: _AttemptState,
    ) -> SolveResult:
        """The body of :meth:`_run_attempts`."""
        # The current penalty, so the NonFiniteModelError handler can tell
        # the hard-penalty path from a penalty-free one; None until
        # ``_prepare_attempts`` has chosen it.
        penalty: float | None = None
        try:
            plan = self._prepare_attempts(problem, backend, direction)
            if isinstance(plan, SolveResult):
                return plan
            penalty = plan.initial_penalty
            max_attempts = plan.budget.max_attempts
            state.max_attempts = max_attempts
            for attempt in range(1, max_attempts + 1):
                state.check_cancelled()
                # Attempt 1 always starts (its backend decides how much it
                # can do); a retry is not started once the limit is gone.
                if (
                    attempt > 1
                    and state.interrupt is not None
                    and state.interrupt.deadline_passed()
                ):
                    state.retry_skipped = True
                    break
                result = self._solve_attempt(
                    problem, backend, plan, direction, attempt, penalty, state
                )
                if result is not None:
                    return result
                if attempt < max_attempts and penalty is not None:
                    penalty = self._penalty_strategy.next_penalty(penalty, attempt)

            # An exhaustive backend proves infeasibility only by actually
            # enumerating assignments: zero samples back is not a proof.
            proven = (
                backend.is_exhaustive
                and state.raw is not None
                and state.raw.num_samples > 0
            )
            message, warnings = _infeasible_message(
                proven,
                plan.budget.cut_reason,
                len(state.attempts),
                wall_clock_limited=state.wall_clock_limit_reached,
            )
            return SolveResult(
                status="infeasible",
                backend=backend.name,
                objective_direction=direction,
                solutions=[],
                attempts=state.attempts,
                infeasibility_proven=proven,
                infeasibility=state.infeasibility,
                warnings=[
                    *warnings,
                    *_postprocess_limit_warnings(state.postprocess_limits),
                ],
                # Timing/usage facts from the last attempt still matter to
                # the caller (e.g. quota spent on a remote solve).
                metadata=state.last_metadata,
                message=message,
            )
        except SolveCancelled:
            # Before every other handler: the ``Exception`` one below would
            # otherwise turn a cancellation into a solver_error.
            raise
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
                    state.attempts,
                    penalty,
                    metadata=state.last_metadata,
                )
            logger.warning("Problem %s compilation error: %s", problem.name, exc)
            return self._failure(
                "invalid_problem",
                backend.name,
                direction,
                [catalog_error("COMPILATION_FAILED", str(exc))],
                attempts=state.attempts,
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
                attempts=state.attempts,
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
                attempts=state.attempts,
                metadata=state.last_metadata,
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
                attempts=state.attempts,
                metadata=state.last_metadata,
            )
