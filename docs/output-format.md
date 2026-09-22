[← Back to README](../README.md) · [Documentation index](README.md)

# Output Format

This page is the field-by-field reference for everything AnnealBridge returns:
the `SolveResult` of a solve, the `ProblemValidationResult` of a dry-run
validation, the `BackendRecommendationResult` of a ranking, and the
`OptimizationCapabilities` document. The same models back the CLI's `--json`
output and the MCP tool results, so an agent and a human see identical data.

Two guarantees run through all of it:

- **`objective_value` and feasibility are always recomputed from the original
  problem**, never inferred from the compiled model. Every candidate the solver
  returns is decoded back to business variables and evaluated against the JSON
  the caller submitted, constraint by constraint.
- **`energy` is for debugging only.** It is the compiled model's energy, which
  includes hard penalties, slack terms and encoding bits. It is retained so a
  developer can reason about the compilation, and it is never used to judge
  feasibility, to compute the objective, or to rank solutions.

Domain failures are returned as structured results, never raised as exceptions.
For the codes that can appear in `errors` and `warnings`, see
[Errors and warnings](errors.md).

## SolveResult

The outcome of `solve_optimization` / `annealbridge solve`.

| Field | Type | Meaning |
| --- | --- | --- |
| `status` | [SolveStatus](#solvestatus) | The single verdict for the request. |
| `backend` | string \| null | Registry name of the backend that ran, or `null` when the request failed before a backend was chosen. |
| `objective_direction` | `"minimize"` \| `"maximize"` \| null | Echoed from the problem, so a consumer can interpret `objective_value` without re-reading the input. |
| `solutions` | array of [Solution](#solution) | Ranked feasible solutions, best first, at most `solver.top_k`. Empty unless `status` is `success`. |
| `attempts` | array of [SolveAttempt](#solveattempt) | One entry per compile/solve/validate attempt, in order. |
| `infeasibility_proven` | boolean | `true` only when an exhaustive backend actually enumerated every assignment and found none feasible. Defaults to `false`. |
| `infeasibility` | [InfeasibilityDiagnostics](#infeasibilitydiagnostics) \| null | Why the last attempt found nothing feasible. Present only when `status` is `infeasible` **and** that attempt had candidates to diagnose; `null` on every other status, and on an attempt that received no samples at all. |
| `optimality_proven` | boolean | `true` only on `success` when an exhaustive backend enumerated every assignment: rank 1 is then the global optimum of `ranking_score`, not merely the best candidate seen. Always `false` on a heuristic or remote backend. |
| `errors` | array of [SolveError](#solveerror) | Structured failures. Empty on success. |
| `warnings` | array of [SolveError](#solveerror) | Non-blocking advice, same structure as an error: the warnings `validate` gives for this backend, then any raised during the run. Present whatever the `status`, except `invalid_problem`. |
| `metadata` | [SolverExecutionMetadata](#solverexecutionmetadata) \| null | Sanitized execution facts. Present whenever an attempt actually completed, local backends included; `null` when the request failed before any solve finished. |
| `message` | string \| null | Human-readable one-line summary of the result. On `success`: which backend produced it, whether optimality is proven, the rank-1 objective with its direction (and its soft violation when non-zero), how many distinct candidates the attempt saw, how many were feasible and how many are returned, and the attempt number when a retry produced it. On `infeasible`: why nothing feasible was found. On a failure: the first error's message. Deterministic — it never contains timings. `null` when there is nothing to add. |
| `elapsed_ms` | number \| null | Wall-clock milliseconds measured by the service from entering `solve` to returning, problem validation and any wait for a concurrency slot included. Present whatever the `status`. Unrelated to `metadata.timing_us`, which is what a vendor reports about its own side. |
| `annealbridge_version` | string \| null | The installed package version that produced the result (`"unknown"` outside an installed distribution). |

### SolveStatus

Exactly one of seven values.

| Status | Meaning |
| --- | --- |
| `success` | At least one feasible solution was found and ranked; `solutions` is non-empty. |
| `infeasible` | The pipeline ran, but no candidate satisfied every hard constraint under independent re-validation. Check `infeasibility_proven` before concluding the problem has no solution. |
| `invalid_problem` | The problem failed validation. Every error found is in `errors`; no backend was invoked. |
| `resource_limit_exceeded` | A request parameter or the compiled size exceeded a server-side ceiling. Values are refused, never clamped. |
| `backend_unavailable` | The requested backend is not registered, not installed, disabled by policy, or missing credentials. There is never a silent fallback to another backend. |
| `configuration_error` | The *server's* configuration is wrong — not the problem. |
| `solver_error` | The backend failed while executing, or a remote vendor reported an error. |

`infeasible` is not the same as "no solution exists". Only
`infeasibility_proven: true` — which only an exhaustive backend that actually
returned samples can set — is a proof. On a heuristic or remote backend an
`infeasible` result means "not found under this configuration". Either way,
[`infeasibility`](#infeasibilitydiagnostics) says which hard constraints stood
in the way.

### Solution

| Field | Type | Meaning |
| --- | --- | --- |
| `rank` | integer | 1-based rank; 1 is the best `ranking_score`. |
| `variables` | object of string → integer | The business variables only. Slack and integer-encoding bits are stripped, and integers are decoded to plain `int` values inside their declared bounds. |
| `objective_value` | number | The objective recomputed from the original problem, including its `constant`. |
| `soft_violation_score` | number | `Σ weight × violation²` over **all** soft constraints, recomputed by the validator from the exact residual. The feasibility tolerance is deliberately not applied here, so this equals the soft energy the solver minimized. |
| `ranking_score` | number | `objective_value + soft_violation_score` when minimizing, `objective_value − soft_violation_score` when maximizing. The sort key. |
| `energy` | number \| null | The compiled model's energy. Debugging only. |
| `sample_count` | integer | How many rows of this attempt's raw solver output carried this business assignment, before deduplication. Not a confidence measure: on an exhaustive backend every business assignment is enumerated once per combination of the slack and integer-encoding bits, so the count only reflects how many internal variables the compiled model happened to have. |
| `hard_constraints_satisfied` | boolean | Always `true` for a returned solution — only feasible candidates are ranked. |
| `constraint_evaluations` | array of [ConstraintEvaluation](#constraintevaluation) | One entry per constraint, hard and soft. |

### InfeasibilityDiagnostics

Present on `result.infeasibility` only when `status` is `infeasible` and the
**last** attempt actually returned candidates. A backend that returned no
samples leaves it `null`: there is then nothing to be closest and no
denominator for a rate.

Everything here is recomputed by the validator from the *original* problem —
the same arithmetic that judges a ranked solution. The solver's energy and any
sampler-reported feasibility flag play no part, so the diagnosis cannot
contradict the `infeasible` verdict itself.

When a solve made several attempts, the diagnosis describes the last one, the
one that ran at the highest hard penalty. Earlier attempts are still listed in
`attempts`, but are not diagnosed.

| Field | Type | Meaning |
| --- | --- | --- |
| `closest_candidate` | [ClosestCandidate](#closestcandidate) | The candidate that came nearest to feasibility. |
| `hard_violation_rates` | array of [HardViolationRate](#hardviolationrate) | One entry per **hard** constraint, in the problem's constraint order. Soft constraints never appear: they cannot make a candidate infeasible. |

### ClosestCandidate

The only infeasible assignment a result ever exposes. It is chosen by the
smallest total hard violation — `Σ violation_amount` over the hard
constraints, in the constraints' own units — and on a tie by the candidate the
solver output listed first, so the choice is deterministic for a given raw
output. It is a diagnostic aid, not a solution: it violates at least one hard
constraint and must never be presented as an answer.

Because the units of different constraints are simply added, the total ranks
candidates rather than measuring them: it says which assignment is nearest,
not by how much in any single constraint's terms. For that, read
`constraint_evaluations`.

| Field | Type | Meaning |
| --- | --- | --- |
| `variables` | object of string → integer | The business variables only, like a [Solution](#solution)'s: slack and integer-encoding bits are stripped and integers are decoded. |
| `hard_violation_total` | number | `Σ violation_amount` over the hard constraints. Strictly positive — a zero total would be a feasible candidate. A hard constraint that holds within the feasibility tolerance contributes `0`. |
| `constraint_evaluations` | array of [ConstraintEvaluation](#constraintevaluation) | One entry per constraint, hard **and** soft, exactly as for a ranked solution. This is where the binding requirement shows up: the hard entries with `satisfied: false`. |

### HardViolationRate

How often one hard constraint failed across the last attempt's candidates.
A rate near `1` marks a constraint almost nothing could satisfy — usually the
one to relax or re-check first; a rate of `0` means that constraint was never
the obstacle by itself.

| Field | Type | Meaning |
| --- | --- | --- |
| `constraint_id` | string | The `id` from the problem. |
| `violated_candidates` | integer | How many candidates this constraint rejected, judged with the same feasibility tolerance as everywhere else. |
| `candidates` | integer | The attempt's deduplicated candidate count — the same value for every entry, and equal to `attempts[-1].unique_samples`. |
| `violated_fraction` | number | `violated_candidates / candidates`, between `0` and `1`. |

The rates are independent counts, one constraint at a time: a candidate that
breaks two constraints is counted in both, so they do not sum to 1 and a
constraint with a rate below 1 does not imply a feasible candidate exists.

### SolveAttempt

| Field | Type | Meaning |
| --- | --- | --- |
| `attempt` | integer | 1-based attempt number. |
| `penalty` | number \| null | The hard-constraint penalty λ used for this attempt. `null` on a path whose compiler uses no hard penalty (the CQM path). |
| `samples_received` | integer | Rows the backend returned. |
| `unique_samples` | integer | Distinct business assignments among those rows, **after decoding and deduplication** — the candidate count everything downstream works on. |
| `feasible_samples` | integer | How many of those **deduplicated** candidates satisfied every hard constraint under independent re-validation. Never larger than `unique_samples`. |
| `compiled_variables` | integer \| null | The compiled model's actual variable count (business variables plus slack and integer-encoding bits), as opposed to the estimate `validate` reports. |
| `compiled_interactions` | integer \| null | The compiled model's quadratic terms: the BQM's interactions, or on the CQM path the objective's plus every constraint's. |
| `compile_ms` | number \| null | Wall-clock milliseconds the compile stage took. |
| `solve_ms` | number \| null | Wall-clock milliseconds the backend call took, network round-trips included on a remote backend. |
| `validate_ms` | number \| null | Wall-clock milliseconds for decoding, deduplication, re-validation and ranking of the returned samples. |

An attempt is recorded even when it produced nothing feasible, so the retry
ladder is visible: attempt 2 carries double attempt 1's penalty.

Both sample counts are post-deduplication, so they can be compared with the
returned list directly: `len(solutions)` smaller than the last attempt's
`feasible_samples` means the list was truncated to `solver.top_k`, not that
candidates were lost.

The three `*_ms` timings are the service's own clock around each stage and
are measured for every backend, local ones included. They vary from run to
run and are the only fields of a result that do; `elapsed_ms` on the result
covers all attempts plus validation and bookkeeping, so it is never smaller
than their sum.

### ConstraintEvaluation

| Field | Type | Meaning |
| --- | --- | --- |
| `constraint_id` | string | The `id` from the problem, so a result traces back to it. |
| `constraint_type` | `"hard"` \| `"soft"` | Echoed from the problem. |
| `satisfied` | boolean | Whether this constraint holds for this solution. |
| `actual_value` | number | The left-hand side evaluated at this assignment. |
| `operator` | string | The constraint's operator (`==`, `<=`, `>=`). |
| `expected_value` | number | The constraint's `rhs`. |
| `violation_amount` | number | How far the constraint is from being satisfied. For a **hard** constraint it is `0` whenever the constraint holds. For a **soft** constraint it is always the exact residual (`\|actual − rhs\|`, or the one-sided excess/shortfall for an inequality), so it can be a tiny non-zero value while `satisfied` is still `true` — that is what the solver was charged for. |
| `weighted_penalty` | number \| null | `weight × violation_amount²` for a soft constraint; `null` for a hard one. |

### SolveError

The same structure carries both errors and warnings.

| Field | Type | Meaning |
| --- | --- | --- |
| `code` | string | A stable code from the catalog. See [Errors and warnings](errors.md). |
| `path` | string \| null | JSON path into the submitted problem, e.g. `constraints[0].weight` or `solver.num_reads`. |
| `message` | string | What went wrong, with the concrete values involved. |
| `retryable` | boolean | Whether the same request may succeed later without any change. `false` for every warning. |
| `recommended_action` | string \| null | Fixed categorical guidance for this code — stable wording, never containing configuration values. |

### SolverExecutionMetadata

Sanitized execution facts about one solver run. It describes the **last
completed attempt**, so timing and quota facts survive an `infeasible` result.
The local backends (`exact`, `simulated_annealing`, `tabu`,
`simulated_bifurcation`) report it as well, with the fields a local run can
fill: `backend`, `remote: false`, the `model_type` the service stamps, and — on
the three sampling backends, `simulated_annealing`, `tabu` and
`simulated_bifurcation` — the `num_reads_requested` asked of the
sampler. There is no vendor side to a local run, so `timing_us` is
empty and `solver_id`, `effective_time_limit_seconds` and the two QPU fields
are `null`.

The service never invents metadata: `metadata` is `null` when no attempt
completed — the request failed validation, no backend was available, or the
first solve errored.

| Field | Type | Meaning |
| --- | --- | --- |
| `backend` | string | The backend that produced this run. |
| `remote` | boolean | Whether the run left this machine. |
| `solver_id` | string \| null | Vendor-side solver identifier, e.g. `fujitsuDA3/v4`. |
| `num_reads_requested` | integer \| null | Reads actually requested of the sampler. |
| `effective_time_limit_seconds` | number \| null | The time limit actually used, after a sampler's own minimum was applied. |
| `timing_us` | object of string → number | Timing facts in microseconds; see the whitelist below. Empty when the backend reported none. |
| `average_chain_break_fraction` | number \| null | QPU only. |
| `embedding_max_chain_length` | integer \| null | QPU only. |
| `model_type` | `"bqm"` \| `"cqm"` \| null | Which compiler path ran. Filled in by the service, because the backend does not know it. |
| `sampler_reported_feasible` | integer \| null | How many samples the *sampler* called feasible. Informational only. |

**Timing whitelist.** `timing_us` is not the sampler's raw `info` dict. It is
an allow-list on output: only these keys can ever leave the solver layer, and
anything else a backend reports is silently dropped. This keeps vendor
diagnostics — and anything a vendor might embed in them — out of a tool
response. A whitelisted key whose value is not a finite number (NaN or
±infinity, an integer too large for a float, a Fujitsu DA millisecond
string such as `"NaN"` or `"inf"`, or a value that overflows when converted
to microseconds) is dropped on its own, and the
other keys are kept: JSON has no number for it, and publishing it as `null`
would break the `number` type of `timing_us`. For the same reason
`effective_time_limit_seconds` and `average_chain_break_fraction` are `null`
(not reported) when the vendor's value is not finite.

```text
qpu_access_time              qpu_sampling_time
qpu_anneal_time_per_sample   qpu_programming_time
total_post_processing_time   run_time
charge_time                  solve_time
total_elapsed_time
```

**`sampler_reported_feasible` is never trusted.** Only a constraint-model
backend fills it, and it records what the sampler claimed. Every sample is
still independently re-validated against the original problem, and this number
feeds neither feasibility nor ranking. It exists so a caller can notice a
disagreement between the vendor and the validator.

## Example: a full `SolveResult`

```bash
annealbridge solve examples/knapsack.json --json
```

(`examples/knapsack.json` is a file in a repository checkout — an installed
package has no such directory for the CLI to read. Any problem JSON saved
locally works the same.)

Ranks 3–5 are elided below; they continue the same pattern down to
`objective_value` 13.

```json
{
  "status": "success",
  "backend": "exact",
  "objective_direction": "maximize",
  "solutions": [
    {
      "rank": 1,
      "variables": {
        "item_a": 1,
        "item_b": 0,
        "item_c": 1,
        "item_d": 0
      },
      "objective_value": 17.0,
      "soft_violation_score": 0.0,
      "ranking_score": 17.0,
      "energy": -17.0,
      "sample_count": 16,
      "hard_constraints_satisfied": true,
      "constraint_evaluations": [
        {
          "constraint_id": "capacity",
          "constraint_type": "hard",
          "satisfied": true,
          "actual_value": 10.0,
          "operator": "<=",
          "expected_value": 10.0,
          "violation_amount": 0.0,
          "weighted_penalty": null
        }
      ]
    },
    {
      "rank": 2,
      "variables": {
        "item_a": 1,
        "item_b": 0,
        "item_c": 0,
        "item_d": 1
      },
      "objective_value": 16.0,
      "soft_violation_score": 0.0,
      "ranking_score": 16.0,
      "energy": -16.0,
      "sample_count": 16,
      "hard_constraints_satisfied": true,
      "constraint_evaluations": [
        {
          "constraint_id": "capacity",
          "constraint_type": "hard",
          "satisfied": true,
          "actual_value": 9.0,
          "operator": "<=",
          "expected_value": 10.0,
          "violation_amount": 0.0,
          "weighted_penalty": null
        }
      ]
    }
  ],
  "attempts": [
    {
      "attempt": 1,
      "penalty": 62.0,
      "samples_received": 256,
      "unique_samples": 16,
      "feasible_samples": 10,
      "compiled_variables": 8,
      "compiled_interactions": 28,
      "compile_ms": 1.2,
      "solve_ms": 3.4,
      "validate_ms": 0.8
    }
  ],
  "infeasibility_proven": false,
  "infeasibility": null,
  "optimality_proven": true,
  "errors": [],
  "warnings": [],
  "metadata": {
    "backend": "exact",
    "remote": false,
    "solver_id": null,
    "num_reads_requested": null,
    "effective_time_limit_seconds": null,
    "timing_us": {},
    "average_chain_break_fraction": null,
    "embedding_max_chain_length": null,
    "model_type": "bqm",
    "sampler_reported_feasible": null
  },
  "message": "exact proved optimality: rank 1 has objective 17 (maximize); 10 of 16 distinct candidates were feasible, 5 returned.",
  "elapsed_ms": 6.1,
  "annealbridge_version": "0.2.1"
}
```

Note the numbers. The `exact` backend enumerated all 256 assignments of the 8
compiled variables (4 items plus 4 slack bits); after decoding and dropping the
slack columns, 16 distinct business assignments remained, 10 of which satisfy
the capacity constraint. Each solution's `sample_count` is `16` for the same
reason: every business assignment was enumerated once per setting of the 4
slack bits, which is a fact about the encoding and not about the solution.
`energy` is `-17.0` because the compiled model
minimizes the negated objective — `objective_value` is the `17` the caller
asked about, recomputed from the original JSON. `metadata` says the run stayed
on this machine and took the `bqm` path; everything a vendor would report is
empty, and `num_reads_requested` is `null` because `exact` enumerates rather
than samples. `optimality_proven` is `true` because
`exact` enumerated every assignment, so the rank-1 objective of 17 is the
best any feasible assignment can reach. `message` restates exactly these
facts — the backend, the proof, the rank-1 objective, the 10 feasible out of
16 distinct candidates and the 5 solutions returned — in one sentence an
agent can relay. The `*_ms` values are illustrative: they are wall-clock
measurements and differ on every run, which is why `message` never mentions
them.

## ProblemValidationResult

Returned by `validate_optimization_problem` / `annealbridge validate`. Nothing
is compiled and no backend is invoked.

| Field | Type | Meaning |
| --- | --- | --- |
| `valid` | boolean | Decided by `errors` alone. Warnings never make a problem invalid. |
| `errors` | array of [SolveError](#solveerror) | Every error found in one pass. |
| `warnings` | array of [SolveError](#solveerror) | Advisory findings. Only produced when there are no errors. |
| `estimated_compiled_variables` | integer \| null | Compiled size without building a model: on the BQM path, binary variables + integer-encoding bits + slack bits; on the CQM path, variables + integer slacks. `null` when the problem is invalid. |
| `objective_scale` | number \| null | The upper bound on the objective's range used to size penalties and to judge soft weights. See [Soft constraint weights](problem-format.md#soft-constraint-weights). |
| `model_type` | `"bqm"` \| `"cqm"` \| null | Which compiler path the estimate assumed. |

Estimates and warnings are only computed for a problem with no errors: that
gate is what makes the slack arithmetic well-defined.

## BackendRecommendationResult

Returned by `recommend_backend` / `annealbridge recommend`. Deterministic, with
no network I/O, no solving, no concurrency slot and no quota consumed.

| Field | Type | Meaning |
| --- | --- | --- |
| `valid` | boolean | The problem's own validity. When `false`, `recommendations` is empty. An error only one backend's declaration raises — a `solver.seed` outside the range that backend declares — leaves `valid` true and blocks that backend instead. |
| `errors` | array of [SolveError](#solveerror) | The problem's errors when it is invalid. |
| `recommendations` | array of [BackendRecommendation](#backendrecommendation) | Every registered backend, ranked. |
| `advisory` | string | A fixed sentence restating that a solve always uses `problem.solver.backend` as given. |

### BackendRecommendation

| Field | Type | Meaning |
| --- | --- | --- |
| `rank` | integer | 1-based position in the ranking. |
| `backend` | string | The registry name — the value to put in `solver.backend`. |
| `usable` | boolean | Whether a solve on it right now would get past every gate and limit. |
| `model_type` | `"bqm"` \| `"cqm"` \| null | The compiler path it would take; `null` when the server has no compiler for it. |
| `reasons` | array of string | Fixed routing reason codes, in the order they were applied. See [Recommendation reason codes](errors.md#recommendation-reason-codes). |
| `blocking` | array of [SolveError](#solveerror) | Why a solve now would fail, e.g. `REMOTE_DISABLED`, or `INVALID_SOLVER_PREFERENCE` at `solver.seed` when the seed lies outside the range this backend declares. |
| `warnings` | array of [SolveError](#solveerror) | The validation warnings for this backend's path. |
| `estimated_compiled_variables` | integer \| null | The compiled size on this backend's path. `null` when there is no compiler path, or when the problem is invalid for this backend (see `blocking`). |

The ranking is **advisory only**: nothing feeds it back into a solve, and
`problem.solver.backend` is never rewritten.

## OptimizationCapabilities

Returned by `get_optimization_capabilities` / `annealbridge capabilities`. It
performs no network I/O, so it is safe to call before any credentials are
configured.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | The newest problem schema version this server accepts. |
| `schema_versions` | array of string | Every accepted version. Derived from the pydantic model, not hard-coded. |
| `supported_variable_types` | array of string | `binary` and `integer`. |
| `supported_constraint_operators` | array of string | `==`, `<=`, `>=`. |
| `supported_objective_terms` | array of string | `linear`, `quadratic`. |
| `inequality_requires_integer_coefficients` | boolean | Whether `<=` / `>=` constraints need integral coefficients and `rhs`. |
| `backends` | array of [BackendCapability](#backendcapability) | Every registered backend. |
| `problem_json_schema` | object \| null | The full `OptimizationProblem` JSON Schema, identical to `annealbridge export-schema`. `null` unless the MCP tool was called with `include_schema: true`; the CLI always fills it. |
| `annealbridge_version` | string \| null | The installed package version that produced this view (`"unknown"` outside an installed distribution). |

### BackendCapability

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | The **registry key**: the value to put in `solver.backend`, and the one `ANNEALBRIDGE_ENABLED_BACKENDS` is matched against. |
| `available` | boolean | Whether the backend can run right now (dependency installed, credentials present, configuration valid). |
| `enabled` | boolean | Whether server policy permits it: in the enabled-backends list, and remote execution allowed if it is remote. |
| `unavailable_reason` | string \| null | Categorical detail when `available` is `false`, e.g. "dwave-system not installed". Never contains configuration values. A backend whose availability check raises reads `availability check failed: unexpected <ExceptionClass>: <message>`, and one reporting an unknown category reads its detail (or `no reason reported`) followed by `(unknown availability category '<category>')`; both are redacted, and only that backend is listed as unavailable — the others are reported as usual. |
| `remote` | boolean | Whether it leaves this machine. |
| `heuristic` | boolean | Whether it may return a sub-optimal answer. |
| `exhaustive` | boolean | Whether it enumerates every assignment and can prove infeasibility. |
| `supports_seed` | boolean | Whether `solver.seed` has any effect. |
| `seed_min` | integer \| null | The smallest `solver.seed` this backend accepts, inclusive. A seed outside `seed_min`–`seed_max` is refused with `INVALID_SOLVER_PREFERENCE` before anything runs. `null` when the backend declares no seed range. |
| `seed_max` | integer \| null | The largest `solver.seed` this backend accepts, inclusive. `null` when the backend declares no seed range. |
| `returns_multiple_samples` | boolean | `false` means the effective `top_k` is at most 1. |
| `limits` | object of string → number | The policy ceilings that apply to this backend. |
| `description` | string | One-line description of the backend. |

`available` and `enabled` are independent: a backend can be installed and
working but switched off by policy, or allowed by policy but missing its
credentials. Both must be true for a solve to reach it. See
[Backends](backends.md) and [Configuration](configuration.md).
