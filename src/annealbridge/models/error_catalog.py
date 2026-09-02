"""Central catalog of error codes → fixed recommended_action text (Phase 2 spec §13.2).

Every error code used in ``SolveError.recommended_action`` gets its fixed
wording here, so agents can learn a stable vocabulary. The text is
categorical guidance only and must never contain configuration values.
"""

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
        "The D-Wave configuration exists but cannot be parsed; fix or "
        "remove the config on the server, or use a local backend."
    ),
    "BACKEND_UNAVAILABLE": (
        "The backend reported it is unavailable; see the message for the "
        "reason, or choose another backend."
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
