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
        "such as simulated_annealing or exact."
    ),
    "REMOTE_DISABLED": (
        "Remote solving is disabled by server policy; ask the operator to "
        "enable remote backends, or use simulated_annealing or exact."
    ),
    "REMOTE_CREDENTIALS_MISSING": (
        "D-Wave credentials are not configured on the server; configure them "
        "via the standard D-Wave config mechanism, or use a local backend."
    ),
    "BACKEND_DISABLED_BY_POLICY": (
        "This backend is not in the server's enabled_backends list; choose "
        "an enabled backend or ask the operator to enable it."
    ),
    "EXACT_VARIABLE_LIMIT": (
        "The compiled problem has more variables than the exact solver "
        "limit; reduce the problem size or use simulated_annealing."
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
        "time_limit_seconds exceeds the server's limit for hybrid solving; "
        "lower it or omit it to use the sampler minimum."
    ),
    "CONCURRENCY_LIMIT": (
        "Too many solves are running concurrently on this server; retry "
        "after the current solves finish."
    ),
    "EMBEDDING_FAILED": (
        "Problem too dense or too large for QPU embedding; reduce "
        "variables/constraints, or use leap_hybrid_bqm or "
        "simulated_annealing."
    ),
    "REMOTE_AUTH_FAILED": (
        "The remote solver rejected the configured credentials; verify the "
        "D-Wave credentials on the server, or use a local backend."
    ),
    "REMOTE_TIMEOUT": (
        "The remote solve timed out; retry later, reduce the problem size, "
        "or use a local backend."
    ),
    "REMOTE_SOLVER_ERROR": (
        "The remote solver reported an error; retry later or use a local "
        "backend such as simulated_annealing."
    ),
    "REMOTE_RETRIES_DISABLED": (
        "Automatic retries are disabled for remote backends to protect "
        "quota; resubmit the request explicitly if a retry is intended."
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
        "A solver preference is outside its allowed range (values such as "
        "top_k, num_reads, num_sweeps, annealing_time_us, chain_strength and "
        "time_limit_seconds must be positive, and max_retries must not be "
        "negative); correct the value, or omit it to use the default."
    ),
    "TRIVIALLY_INFEASIBLE": (
        "A hard constraint cannot be satisfied by any assignment of the "
        "binary variables, judged from the reachable range of its combined "
        "left-hand side; correct the rhs, the operator or the coefficients, "
        "or make the constraint soft if it is only a preference."
    ),
    "NO_VARIABLES": (
        "The problem declares no variables, so there is nothing to optimize; "
        "declare at least one binary variable."
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
