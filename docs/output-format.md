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
| `errors` | array of [SolveError](#solveerror) | Structured failures. Empty on success. |
| `warnings` | array of [SolveError](#solveerror) | Non-blocking advice, same structure as an error: the warnings `validate` gives for this backend, then any raised during the run. Present whatever the `status`, except `invalid_problem`. |
| `metadata` | [SolverExecutionMetadata](#solverexecutionmetadata) \| null | Sanitized execution facts. `null` when the backend reported none — the local backends do not. |
| `message` | string \| null | Human-readable summary, mainly used to explain an `infeasible` result. |

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
`infeasible` result means "not found under this configuration".

### Solution

| Field | Type | Meaning |
| --- | --- | --- |
| `rank` | integer | 1-based rank; 1 is the best `ranking_score`. |
| `variables` | object of string → integer | The business variables only. Slack and integer-encoding bits are stripped, and integers are decoded to plain `int` values inside their declared bounds. |
| `objective_value` | number | The objective recomputed from the original problem, including its `constant`. |
| `soft_violation_score` | number | `Σ weight × violation²` over **all** soft constraints, recomputed by the validator from the exact residual. The feasibility tolerance is deliberately not applied here, so this equals the soft energy the solver minimized. |
| `ranking_score` | number | `objective_value + soft_violation_score` when minimizing, `objective_value − soft_violation_score` when maximizing. The sort key. |
| `energy` | number \| null | The compiled model's energy. Debugging only. |
| `hard_constraints_satisfied` | boolean | Always `true` for a returned solution — only feasible candidates are ranked. |
| `constraint_evaluations` | array of [ConstraintEvaluation](#constraintevaluation) | One entry per constraint, hard and soft. |

### SolveAttempt

| Field | Type | Meaning |
| --- | --- | --- |
| `attempt` | integer | 1-based attempt number. |
| `penalty` | number \| null | The hard-constraint penalty λ used for this attempt. `null` on a path whose compiler uses no hard penalty (the CQM path). |
| `samples_received` | integer | Rows the backend returned. |
| `unique_samples` | integer | Distinct assignments among them, after decoding. |
| `feasible_samples` | integer | How many satisfied every hard constraint under re-validation. |

An attempt is recorded even when it produced nothing feasible, so the retry
ladder is visible: attempt 2 carries double attempt 1's penalty.

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
The local backends (`exact`, `simulated_annealing`) report no metadata, and the
service never invents any: `metadata` is then `null`.

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
response.

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

(`examples/knapsack.json` is a file in a repository checkout — the installed
package does not ship it. Any problem JSON saved locally works the same.)

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
      "feasible_samples": 10
    }
  ],
  "infeasibility_proven": false,
  "errors": [],
  "warnings": [],
  "metadata": null,
  "message": null
}
```

Note the numbers. The `exact` backend enumerated all 256 assignments of the 8
compiled variables (4 items plus 4 slack bits); after decoding and dropping the
slack columns, 16 distinct business assignments remained, 10 of which satisfy
the capacity constraint. `energy` is `-17.0` because the compiled model
minimizes the negated objective — `objective_value` is the `17` the caller
asked about, recomputed from the original JSON. `metadata` is `null` because a
local backend reports no execution facts.

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
| `valid` | boolean | The problem's own validity. When `false`, `recommendations` is empty. |
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
| `blocking` | array of [SolveError](#solveerror) | Why a solve now would fail, e.g. `REMOTE_DISABLED`. |
| `warnings` | array of [SolveError](#solveerror) | The validation warnings for this backend's path. |
| `estimated_compiled_variables` | integer \| null | The compiled size on this backend's path. |

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
| `problem_json_schema` | object | The full `OptimizationProblem` JSON Schema, identical to `annealbridge export-schema`. |

### BackendCapability

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | The **registry key**: the value to put in `solver.backend`, and the one `ANNEALBRIDGE_ENABLED_BACKENDS` is matched against. |
| `available` | boolean | Whether the backend can run right now (dependency installed, credentials present, configuration valid). |
| `enabled` | boolean | Whether server policy permits it: in the enabled-backends list, and remote execution allowed if it is remote. |
| `unavailable_reason` | string \| null | Categorical detail when `available` is `false`, e.g. "dwave-system not installed". Never contains configuration values. |
| `remote` | boolean | Whether it leaves this machine. |
| `heuristic` | boolean | Whether it may return a sub-optimal answer. |
| `exhaustive` | boolean | Whether it enumerates every assignment and can prove infeasibility. |
| `supports_seed` | boolean | Whether `solver.seed` has any effect. |
| `returns_multiple_samples` | boolean | `false` means the effective `top_k` is at most 1. |
| `limits` | object of string → number | The policy ceilings that apply to this backend. |
| `description` | string | One-line description of the backend. |

`available` and `enabled` are independent: a backend can be installed and
working but switched off by policy, or allowed by policy but missing its
credentials. Both must be true for a solve to reach it. See
[Backends](backends.md) and [Configuration](configuration.md).
