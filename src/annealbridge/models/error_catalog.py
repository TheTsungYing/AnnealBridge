"""Central catalog of error codes → fixed recommended_action text (Phase 2 spec §13.2).

Every error code used in ``SolveError.recommended_action`` gets its fixed
wording here, so agents can learn a stable vocabulary. The text is
categorical guidance only and must never contain configuration values.

Errors raised by the validator and by the service layer are built through
:func:`catalog_error`, so their ``recommended_action`` always comes from this
catalog.
"""

from annealbridge.models.solution import SolveError

RECOMMENDED_ACTIONS: dict[str, str] = {
    "UNKNOWN_BACKEND": (
        "The requested backend name is not registered; use one of the "
        "backends listed by the capabilities tool."
    ),
    "BACKEND_NOT_INSTALLED": (
        "The backend's optional dependency is not installed; install the "
        "corresponding extra (e.g. dwave-system) or choose a local backend "
        "such as simulated_annealing, tabu or exact."
    ),
    "REMOTE_DISABLED": (
        "Remote solving is disabled by server policy; ask the operator to "
        "enable remote backends, or use simulated_annealing, tabu or exact."
    ),
    "REMOTE_CREDENTIALS_MISSING": (
        "The remote solver's credentials are not configured on the server; ask "
        "the operator to configure them, or use a local backend."
    ),
    "BACKEND_DISABLED_BY_POLICY": (
        "This backend is not in the server's enabled_backends list; choose "
        "an enabled backend or ask the operator to enable it."
    ),
    "EXACT_VARIABLE_LIMIT": (
        "The compiled problem has more variables than the exact solver "
        "limit; reduce the problem size or use simulated_annealing or tabu."
    ),
    "QPU_READS_LIMIT": (
        "num_reads exceeds the server's QPU limit; lower num_reads. The "
        "server never clamps values silently."
    ),
    "QPU_ANNEALING_TIME_LIMIT": (
        "annealing_time_us exceeds the server's limit; lower it or omit it "
        "to use the QPU default."
    ),
    "REMOTE_TIME_LIMIT": (
        "time_limit_seconds exceeds the server's limit for remote solving; "
        "lower it or omit it to use the backend's default."
    ),
    "LOCAL_READS_LIMIT": (
        "num_reads exceeds the server's limit for local sampling; lower "
        "num_reads. The server never clamps values silently."
    ),
    "SWEEPS_LIMIT": (
        "num_sweeps exceeds the server's limit for local sampling; lower "
        "num_sweeps. The server never clamps values silently."
    ),
    "SB_VARIABLE_LIMIT": (
        "The compiled problem has more variables than the simulated "
        "bifurcation backend's dense-matrix limit; reduce the problem size "
        "or use simulated_annealing or tabu. The server never clamps values "
        "silently."
    ),
    "RETRY_LIMIT": (
        "max_retries exceeds the server's retry limit for this backend "
        "(remote retries are bounded more tightly because each one is a "
        "billed submission); lower max_retries. The server never clamps "
        "values silently."
    ),
    "TOP_K_LIMIT": (
        "top_k exceeds the server's limit on returned solutions; lower "
        "top_k. The server never clamps values silently."
    ),
    "PENALTY_OVERFLOW": (
        "The hard-constraint penalty left the floating-point range before a "
        "feasible solution was found, so the retry ladder stopped; lower "
        "penalty_multiplier or max_retries, or rescale the problem's "
        "coefficients."
    ),
    "CONCURRENCY_LIMIT": (
        "Too many solves are running concurrently on this server; retry "
        "after the current solves finish."
    ),
    "EMBEDDING_FAILED": (
        "Problem too dense or too large for QPU embedding; reduce "
        "variables/constraints, or use leap_hybrid_bqm, simulated_annealing "
        "or tabu."
    ),
    "REMOTE_AUTH_FAILED": (
        "The remote solver rejected the configured credentials; verify the "
        "remote solver's credentials on the server, or use a local backend."
    ),
    "REMOTE_TIMEOUT": (
        "The remote solve timed out; retry later, reduce the problem size, "
        "or use a local backend."
    ),
    "REMOTE_SOLVER_ERROR": (
        "The remote solver reported an error; retry later or use a local "
        "backend such as simulated_annealing or tabu."
    ),
    "REMOTE_RETRIES_DISABLED": (
        "Automatic retries are disabled for remote backends to protect "
        "quota; resubmit the request explicitly if a retry is intended."
    ),
    "REMOTE_QUOTA_EXCEEDED": (
        "The remote solver's usage quota for this billing period is "
        "exhausted; wait for the next period or use a local backend."
    ),
    "REMOTE_BUSY": (
        "The remote solver has too many pending jobs for this account; retry "
        "later, or delete finished job results on the vendor portal."
    ),
    "SOLVER_ERROR": (
        "The solver failed while executing; check the error message, adjust "
        "the problem or solver options, or try another backend."
    ),
    "DWAVE_CONFIG_INVALID": (
        "The D-Wave configuration is invalid: it cannot be parsed, or a "
        "value such as region, endpoint, profile, timeout or solver "
        "selection is rejected; fix or remove the config on the server, or "
        "use a local backend."
    ),
    "BACKEND_UNAVAILABLE": (
        "The backend reported it is unavailable; see the message for the "
        "reason, or choose another backend."
    ),
    "BACKEND_CONFIG_INVALID": (
        "The backend's configuration is present but invalid; fix or remove "
        "it on the server, or use a local backend."
    ),
    "NO_COMPILER_FOR_MODEL_TYPE": (
        "The server has no compiler for the model types this backend "
        "accepts; this is a server configuration error—report it, or "
        "choose another backend."
    ),
    # Problem validator codes (Phase 1 spec §12)
    "UNKNOWN_VARIABLE": (
        "A term references a variable that is not declared; add the variable "
        "to the problem's variables list, or correct the variable name used "
        "by the term."
    ),
    "DUPLICATE_VARIABLE": (
        "The same variable name is declared more than once; remove the "
        "duplicate declarations so each name is declared exactly once."
    ),
    "RESERVED_VARIABLE_NAME": (
        "Names starting with a double underscore are reserved for internal "
        "variables such as slack variables; rename the variable to a name "
        "that does not start with a double underscore."
    ),
    "DUPLICATE_CONSTRAINT_ID": (
        "Two or more constraints share the same id; give every constraint a "
        "unique id so results can be traced back to it."
    ),
    "SELF_QUADRATIC_TERM": (
        "A quadratic term multiplies a variable by itself, which is "
        "meaningless for binary variables because x*x = x; express it as a "
        "linear term on that variable instead."
    ),
    "NON_FINITE_COEFFICIENT": (
        "A coefficient, rhs, constant or weight is NaN or infinite, which "
        "cannot be compiled into a solver model; replace it with a finite "
        "numeric value."
    ),
    "EMPTY_CONSTRAINT": (
        "The constraint has no terms and therefore constrains nothing; add "
        "at least one term to it, or remove the constraint."
    ),
    "NON_INTEGER_INEQUALITY": (
        "An inequality constraint has non-integer coefficients or rhs, but "
        "slack encoding requires integers; scale the coefficients and rhs by "
        "a common multiplier so they become integers, or restate the "
        "constraint as an equality."
    ),
    "HARD_CONSTRAINT_HAS_WEIGHT": (
        "A hard constraint carries a weight, but the penalty for hard "
        "constraints is decided by the server; remove the weight, or make "
        "the constraint soft if it is only a preference."
    ),
    "SOFT_CONSTRAINT_MISSING_WEIGHT": (
        "A soft constraint has no positive weight, so its violation cost is "
        "undefined; supply a positive weight expressed in objective-value "
        "units, or make the constraint hard if it must always hold."
    ),
    "INVALID_SOLVER_PREFERENCE": (
        "A solver preference is outside its allowed range: every numeric "
        "preference must be a finite number greater than zero (a retry "
        "count may also be zero), and a seed must lie within the seed range "
        "the selected backend declares in its capabilities; correct the "
        "value, or omit it to use the default."
    ),
    "TRIVIALLY_INFEASIBLE": (
        "A hard constraint cannot be satisfied by any assignment of the "
        "variables within their bounds, judged from the reachable range of "
        "its combined left-hand side; correct the rhs, the operator or the "
        "coefficients, or make the constraint soft if it is only a "
        "preference."
    ),
    "NO_VARIABLES": (
        "The problem declares no variables, so there is nothing to optimize; "
        "declare at least one variable."
    ),
    # Problem validator codes for integer variables (Phase 3b spec §9.1)
    "INTEGER_BOUNDS_MISSING": (
        "An integer variable needs both lower_bound and upper_bound; add "
        "them, or make the variable binary."
    ),
    "INTEGER_BOUNDS_INVALID": (
        "upper_bound must be greater than lower_bound; a variable with equal "
        "bounds is a constant—fold it into the objective and constraints "
        "instead."
    ),
    "BOUNDS_ON_BINARY": (
        "Binary variables are 0/1 and take no bounds; remove "
        "lower_bound/upper_bound, or set type to integer."
    ),
    "INTEGER_RANGE_TOO_LARGE": (
        "Integer bounds must lie within ±(2^31-1); tighten the bounds or "
        "rescale the variable's unit."
    ),
    "INTEGER_REQUIRES_VERSION_1_1": (
        "Integer variables require schema version 1.1; set version to "
        '"1.1".'
    ),
    # 2026-09-09 review F-24: the slack range must be computed exactly.
    "INEQUALITY_MAGNITUDE_TOO_LARGE": (
        "An inequality's coefficients times its variables' bounds are too "
        "large for the slack range to be computed exactly, so the constraint "
        "could be encoded wrongly; rescale the unit of the coefficients or "
        "the variables, or tighten the bounds."
    ),
    # Compilation
    "COMPILATION_FAILED": (
        "The problem passed validation but could not be compiled into a "
        "solver model, so no backend was invoked; fix the problem as "
        "described in the message, and report the case if the problem looks "
        "legitimate, because that indicates a mismatch between the validator "
        "and the compiler."
    ),
}

# Codes whose failure is transient by nature: the same request may succeed
# later without any change to the problem or configuration.
RETRYABLE_CODES: frozenset[str] = frozenset(
    {
        "CONCURRENCY_LIMIT",
        "REMOTE_TIMEOUT",
        "REMOTE_SOLVER_ERROR",
        "REMOTE_BUSY",
    }
)


def catalog_error(code: str, message: str, *, path: str | None = None) -> SolveError:
    """Build a SolveError whose retryable / recommended_action come from the catalog."""
    return SolveError(
        code=code,
        path=path,
        message=message,
        retryable=code in RETRYABLE_CODES,
        recommended_action=RECOMMENDED_ACTIONS.get(code),
    )
