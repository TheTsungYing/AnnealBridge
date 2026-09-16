[← Back to README](../README.md) · [Documentation index](README.md)

# Errors and Warnings

This page is the complete catalog of the codes AnnealBridge can return: the
error codes that appear in `SolveResult.errors`, the warning codes that appear
in `SolveResult.warnings` and `ProblemValidationResult.warnings`, the reason
codes the backend recommender attaches to each ranked backend, and the exit
codes the CLI uses.

Domain failures are **returned, never raised**. A caller that gets a result
back always gets a `status` plus a structured explanation; nothing is left to a
stack trace.

## The SolveError structure

Errors and warnings share one shape.

| Field | Type | Meaning |
| --- | --- | --- |
| `code` | string | A stable code from this page. Codes are a vocabulary an agent can learn: the wording of a message may change, the code does not. |
| `path` | string \| null | JSON path into the submitted problem, e.g. `constraints[0].weight`, `variables[2]`, `solver.num_reads`. `null` when the failure is not attributable to one field. |
| `message` | string | What went wrong, with the concrete values involved. |
| `retryable` | boolean | Whether the identical request may succeed later with no change to the problem or the configuration. Always `false` for a warning. |
| `recommended_action` | string \| null | Fixed categorical guidance for this code. The text is the same for every occurrence and **never contains configuration values** — no limits, no hostnames, no credentials. |

Validation collects **all** errors in one pass, so a caller sees every problem
with the document at once rather than fixing them one round-trip at a time.

## Errors by status

`SolveResult.status` tells a caller what kind of failure it is; the code says
which one. The table headings below group each code under the status it is
reported with.

### `invalid_problem` — the submitted document is wrong

No backend is invoked and no quota is spent. Fix the document and resubmit.

| Code | What it means | Recommended action (summary) |
| --- | --- | --- |
| `NO_VARIABLES` | The problem declares no variables, so there is nothing to optimize. | Declare at least one variable. |
| `UNKNOWN_VARIABLE` | A term references a variable that is not declared. | Add the variable, or correct the name in the term. |
| `DUPLICATE_VARIABLE` | The same variable name is declared more than once. | Remove the duplicate declarations. |
| `RESERVED_VARIABLE_NAME` | A variable name starts with `__`, reserved for compiler-internal slack and encoding bits. | Rename it to a name that does not start with a double underscore. |
| `DUPLICATE_CONSTRAINT_ID` | Two or more constraints share the same `id`. | Give every constraint a unique id so results trace back to it. |
| `SELF_QUADRATIC_TERM` | A quadratic term multiplies a binary variable by itself, where `x·x = x`. | Express it as a linear term on that variable. (Legal for an integer variable.) |
| `NON_FINITE_COEFFICIENT` | A coefficient, `rhs`, `constant` or `weight` is NaN or infinite. | Replace it with a finite numeric value. |
| `EMPTY_CONSTRAINT` | A constraint has no terms and constrains nothing. | Add at least one term, or remove the constraint. |
| `NON_INTEGER_INEQUALITY` | A `<=` / `>=` constraint has non-integer coefficients or `rhs`, but slack encoding needs integers. | Scale coefficients and rhs by a common multiplier, or restate it as an equality. |
| `HARD_CONSTRAINT_HAS_WEIGHT` | A hard constraint carries a `weight`, but the server decides the hard penalty. | Remove the weight, or make the constraint soft if it is only a preference. |
| `SOFT_CONSTRAINT_MISSING_WEIGHT` | A soft constraint has no positive `weight`, so its violation cost is undefined. | Supply a positive weight in objective units, or make the constraint hard. |
| `INVALID_SOLVER_PREFERENCE` | A solver preference is outside its allowed range. | Correct the value, or omit it to use the default. Every numeric preference must be finite and `> 0`; a retry count may also be `0`. A `solver.seed` must lie within the `seed_min`–`seed_max` range the selected backend declares in its capabilities (currently only `simulated_annealing`, `0`–`2147483647`); a backend that does not support seeding raises the `SEED_IGNORED` warning for any seed instead. |
| `TRIVIALLY_INFEASIBLE` | A hard constraint cannot be satisfied by any assignment within the variables' bounds. | Correct the rhs, operator or coefficients, or make the constraint soft. |
| `INTEGER_BOUNDS_MISSING` | An integer variable is missing `lower_bound` or `upper_bound`. | Add both bounds, or make the variable binary. |
| `INTEGER_BOUNDS_INVALID` | `upper_bound` is not greater than `lower_bound`. | Equal bounds are a constant — fold it into the objective and constraints instead. |
| `BOUNDS_ON_BINARY` | A binary variable carries bounds. | Remove the bounds, or set `type` to `integer`. |
| `INTEGER_RANGE_TOO_LARGE` | An integer bound lies outside ±(2³¹ − 1). | Tighten the bounds or rescale the variable's unit. |
| `INTEGER_REQUIRES_VERSION_1_1` | The problem uses integer variables but does not declare `"version": "1.1"`. | Set `version` to `"1.1"`. |
| `INEQUALITY_MAGNITUDE_TOO_LARGE` | An inequality's coefficients times its variables' bounds exceed 2⁵³, so the slack range cannot be computed exactly. | Rescale the unit of the coefficients or the variables, or tighten the bounds. |
| `COMPILATION_FAILED` | The problem passed validation but could not be compiled; no backend was invoked. | Fix the problem as the message describes, and report the case if the problem looks legitimate — it would indicate a validator/compiler mismatch. |

### `backend_unavailable` — the requested backend cannot run

There is never a silent fallback to another backend.

| Code | What it means | Recommended action (summary) |
| --- | --- | --- |
| `UNKNOWN_BACKEND` | The backend name is not registered. | Use one of the backends listed by the capabilities tool. |
| `BACKEND_NOT_INSTALLED` | The backend's optional dependency is missing. | Install the corresponding extra (e.g. `dwave-system`), or choose a local backend. |
| `REMOTE_DISABLED` | Remote solving is off by server policy. | Ask the operator to enable remote backends, or use `simulated_annealing` / `exact`. |
| `REMOTE_CREDENTIALS_MISSING` | The remote solver's credentials are not configured on the server. | Ask the operator to configure them, or use a local backend. |
| `BACKEND_DISABLED_BY_POLICY` | The backend is not in the server's enabled-backends list. | Choose an enabled backend, or ask the operator to enable it. |
| `BACKEND_UNAVAILABLE` | The backend reported itself unavailable with no more specific reason. | See the message, or choose another backend. |

### `resource_limit_exceeded` — a ceiling was hit

Over-limit values are **refused, never silently clamped**, and the refusal
happens before any vendor call, so no quota is consumed. The ceilings come from
`ANNEALBRIDGE_*` settings — see [Configuration](configuration.md).

| Code | What it means | Recommended action (summary) |
| --- | --- | --- |
| `EXACT_VARIABLE_LIMIT` | The compiled problem has more variables than the exhaustive backend's limit; `solve` and `recommend` report it in the same wording. | Reduce the problem size, or use `simulated_annealing`. |
| `QPU_READS_LIMIT` | `num_reads` exceeds the server's QPU limit. | Lower `num_reads`. |
| `QPU_ANNEALING_TIME_LIMIT` | `annealing_time_us` exceeds the server's limit. | Lower it, or omit it to use the QPU default. |
| `REMOTE_TIME_LIMIT` | `time_limit_seconds` exceeds the server's limit for remote solving. | Lower it, or omit it to use the backend's default. |
| `LOCAL_READS_LIMIT` | `num_reads` exceeds the server's limit for local sampling. | Lower `num_reads`. |
| `SWEEPS_LIMIT` | `num_sweeps` exceeds the server's limit for local sampling. | Lower `num_sweeps`. |
| `RETRY_LIMIT` | `max_retries` exceeds the retry limit for this backend. Remote retries are bounded more tightly because each one is a billed submission. | Lower `max_retries`. |
| `TOP_K_LIMIT` | `top_k` exceeds the server's limit on returned solutions. | Lower `top_k`. |
| `PENALTY_OVERFLOW` | The hard penalty left the floating-point range before a feasible solution was found, so the retry ladder stopped. | Lower `penalty_multiplier` or `max_retries`, or rescale the problem's coefficients. |
| `CONCURRENCY_LIMIT` | Too many solves are running concurrently on this server. **Retryable.** | Retry after the current solves finish. |

### `configuration_error` — the server is misconfigured

The problem is fine; the operator has to act.

| Code | What it means | Recommended action (summary) |
| --- | --- | --- |
| `BACKEND_CONFIG_INVALID` | The backend's configuration is present but invalid (for example `FUJITSU_DA_URL` not using `https://`, or a vendor rejecting the request headers). | Fix or remove it on the server, or use a local backend. |
| `DWAVE_CONFIG_INVALID` | The D-Wave configuration cannot be parsed, or a value such as region, endpoint, profile, timeout or solver selection is rejected. | Fix or remove the config on the server, or use a local backend. |
| `NO_COMPILER_FOR_MODEL_TYPE` | The server has no compiler for any model type this backend accepts. | Report it — it is a server configuration error — or choose another backend. |

### `solver_error` — the backend failed while executing

| Code | What it means | Recommended action (summary) |
| --- | --- | --- |
| `SOLVER_ERROR` | The solver failed during execution, with no more specific mapping. | Check the message, adjust the problem or solver options, or try another backend. |
| `EMBEDDING_FAILED` | Minor-embedding onto the QPU topology failed. | Reduce variables/constraints, or use `leap_hybrid_bqm` or `simulated_annealing`. |
| `REMOTE_AUTH_FAILED` | The vendor rejected the configured credentials (HTTP 401/403). | Verify the remote solver's credentials on the server, or use a local backend. |
| `REMOTE_TIMEOUT` | The remote solve timed out. **Retryable.** | Retry later, reduce the problem size, or use a local backend. |
| `REMOTE_SOLVER_ERROR` | The vendor reported an error (the fallback for HTTP 400 / 413 / 5xx). **Retryable.** | Retry later, or use a local backend such as `simulated_annealing`. |
| `REMOTE_QUOTA_EXCEEDED` | The vendor's usage quota for this billing period is exhausted. | Wait for the next period, or use a local backend. |
| `REMOTE_BUSY` | The vendor has too many pending jobs for this account (HTTP 429). **Retryable.** | Retry later, or delete finished job results on the vendor portal. |

### Retryable codes

`retryable: true` means the identical request may succeed later without any
change to the problem or the server's configuration. Exactly four codes carry
it:

```text
CONCURRENCY_LIMIT   REMOTE_TIMEOUT   REMOTE_SOLVER_ERROR   REMOTE_BUSY
```

Every other error is deterministic: resubmitting it unchanged will fail the
same way. No warning is ever retryable.

## Warning codes

Warnings never block. They appear in `ProblemValidationResult.warnings` and in
`SolveResult.warnings`, use the same [SolveError](output-format.md#solveerror)
structure, and always carry `retryable: false`. `valid` is decided by errors
alone. `validate` and `solve` run the same advisory pass for the same backend,
so a solve result carries exactly the warnings a validate call would have
given — whatever its `status`, except `invalid_problem` — followed by the one
warning only a run can raise, `REMOTE_RETRIES_DISABLED`. The two advisories
`validate` alone reports, `UNKNOWN_BACKEND` and `NO_COMPILER_FOR_MODEL_TYPE`,
are errors on `solve`.

Warnings are only produced for a problem with no errors — an erroneous problem
has to be fixed first anyway, and the no-error gate is what makes the size
estimates well-defined. The backend-dependent ones (`EXACT_*`, `DENSE_FOR_QPU`,
`SEED_IGNORED`, `PARAMETER_IGNORED`) additionally require a backend to be
known.

| Code | When it is raised |
| --- | --- |
| `SOFT_WEIGHT_SMALL` | A soft constraint's weight is tiny compared with the objective's scale, so it will barely influence solutions. |
| `LARGE_SLACK_RANGE` | An inequality needs many slack bits on a BQM backend, enlarging the compiled model. |
| `EXACT_NEAR_LIMIT` | The estimated compiled size is close to the exhaustive backend's variable limit; expect noticeable run time and memory. |
| `EXACT_OVER_LIMIT` | The estimated compiled size already exceeds the exhaustive backend's limit; a solve would be refused with `EXACT_VARIABLE_LIMIT`. |
| `DENSE_FOR_QPU` | The problem is likely too dense or too large to minor-embed on a backend that requires embedding. |
| `SEED_IGNORED` | `solver.seed` was given but the selected backend does not support seeding. |
| `PARAMETER_IGNORED` | A preference has no effect on the selected backend or model path — for example `num_sweeps` on a backend that takes no sweeps, `penalty_multiplier` or `max_retries` on the CQM path, `max_retries` on an exhaustive backend such as `exact` (a retry can never surface new samples), or an option block belonging to a different backend. Only non-default values raise it. |
| `DUPLICATE_TERM_MERGED` | The objective repeats a term; the compiler sums the duplicates rather than rejecting them. |
| `REDUNDANT_CONSTRAINT` | An inequality is always satisfied over the declared bounds and adds nothing to the model. |
| `LARGE_INTEGER_RANGE` | An integer variable needs more than 10 encoding bits on a BQM backend. |
| `INTEGER_QUADRATIC_BLOWUP` | Binary-encoding the integer variables yields more than 2000 quadratic interactions on a BQM backend. |
| `SOFT_ALWAYS_VIOLATED` | A soft constraint can never be satisfied within the variables' bounds: every solution pays its weight. The hard counterpart is the `TRIVIALLY_INFEASIBLE` error. |
| `REMOTE_RETRIES_DISABLED` | Emitted during a solve, not validation: a remote solve on the BQM path found nothing feasible, the problem asked for retries (`max_retries > 0`), and server policy disables automatic remote retries, so only one attempt was made. With `max_retries: 0` nothing was blocked and no warning is emitted. |

Four of these — `LARGE_INTEGER_RANGE`, `INTEGER_QUADRATIC_BLOWUP`,
`PARAMETER_IGNORED` and `SEED_IGNORED` — are warnings, not errors. They never
appear in `SolveResult.errors` and never stop a solve.

## Recommendation reason codes

`recommend_backend` / `annealbridge recommend` attaches these to each ranked
backend in `BackendRecommendation.reasons`, in the order they were applied.
They are a separate vocabulary from the error catalog: they explain a ranking,
they never describe a failure.

| Code | Meaning |
| --- | --- |
| `R_UNUSABLE` | A solve on this backend would fail right now; see `blocking`. |
| `R_EXACT_FITS` | The estimated compiled variables fit the exhaustive backend limit, so it can prove optimality and infeasibility. |
| `R_EXACT_NEAR_LIMIT` | Within the exhaustive backend limit but close to it; expect noticeable run time and memory. |
| `R_EXACT_OVER_LIMIT` | The estimated compiled variables exceed the exhaustive backend limit; a solve would be refused. |
| `R_LOCAL_HEURISTIC` | Local heuristic: free and retryable, but optimality is not guaranteed. |
| `R_NATIVE_CONSTRAINTS` | Hard constraints are expressed natively in the model, with no penalty or slack variables. |
| `R_REMOTE` | Remote backend; consumes quota. |
| `R_SINGLE_SAMPLE` | Usually returns a single sample, so the effective `top_k` is at most 1. |
| `R_DENSE_FOR_QPU` | The problem is likely too dense or too large for minor-embedding. |
| `R_INTEGER_NATIVE` | Integer variables are passed to the model natively, with no binary encoding. |
| `R_INTEGER_ENCODED` | Integer variables are binary-encoded; the compiled size grows with the range. |
| `R_INTEGER_BLOWUP` | Binary-encoding the integer variables yields many quadratic interactions on this backend; a backend that takes integers natively ranks ahead of it. |

`R_INTEGER_NATIVE` and `R_INTEGER_ENCODED` are informational and do not change
the order. Of the integer codes only `R_INTEGER_BLOWUP` does, and only by
moving that backend behind the others. The ranking never rewrites
`problem.solver.backend`.

## CLI exit codes

`annealbridge solve`, `validate` and `recommend` share one convention.

| Exit code | Meaning |
| --- | --- |
| `0` | Success: `solve` produced a `success` result, `validate` found the problem valid, `recommend` produced a ranking. |
| `1` | A domain answer that is not success: a non-`success` `SolveResult`, an invalid problem for `validate` or `recommend`. The structured errors are printed either way. |
| `2` | The request never reached the service. |

Exit code `2` covers three situations:

- the input file cannot be read, is not valid JSON, or is not a valid
  optimization problem document (a schema error, as opposed to a semantic
  one — a wrong type, a missing required field, or a field the schema does
  not declare, reported as `<path>: unknown field`);
- `--backend` (available on `solve` and `validate` only) names a backend
  that is not one of the known names;
- an `ANNEALBRIDGE_*` environment variable holds an illegal value. The message
  names the variable only — values are never echoed back, in case one holds a
  secret. An `ANNEALBRIDGE_*` variable that is not recognised at all is *not*
  fatal: the setting keeps its default and the server logs one `WARNING` naming
  it.

`annealbridge capabilities` and `annealbridge export-schema` exit `0`; neither
performs any network I/O.

See [CLI](cli.md) for the commands themselves, and [Security](security.md) for
why limits are refused rather than clamped.
