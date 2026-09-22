[← Back to README](../README.md) · [Documentation index](README.md)

# Problem JSON Format

This page is the complete reference for the `OptimizationProblem` document —
the only thing an agent or a caller ever writes. It covers every top-level
field, the variable / objective / constraint models, the solver preference
block, the rules for bounded integer variables, and how soft weights and hard
penalties are interpreted.

The document is the whole public API. There is no QUBO matrix, no penalty λ,
no slack variable and no integer encoding anywhere in it: those are the
deterministic core's business, and the caller never sees them.

For what comes back, see [Output format](output-format.md); for the codes a
rejected document produces, see [Errors and warnings](errors.md).

## A minimal problem

A 0/1 knapsack with a capacity of 10 — the reduced form of
[examples/knapsack.json](../examples/knapsack.json), a file in a repository
checkout (the MCP server also serves it as `annealbridge://examples/knapsack`):

```json
{
  "version": "1.0",
  "name": "knapsack",
  "variables": [
    {"name": "item_a", "type": "binary"},
    {"name": "item_b", "type": "binary"},
    {"name": "item_c", "type": "binary"},
    {"name": "item_d", "type": "binary"}
  ],
  "objective": {
    "direction": "maximize",
    "linear_terms": [
      {"variable": "item_a", "coefficient": 10},
      {"variable": "item_b", "coefficient": 8},
      {"variable": "item_c", "coefficient": 7},
      {"variable": "item_d", "coefficient": 6}
    ]
  },
  "constraints": [
    {
      "id": "capacity",
      "type": "hard",
      "terms": [
        {"variable": "item_a", "coefficient": 6},
        {"variable": "item_b", "coefficient": 5},
        {"variable": "item_c", "coefficient": 4},
        {"variable": "item_d", "coefficient": 3}
      ],
      "operator": "<=",
      "rhs": 10
    }
  ],
  "solver": {"backend": "exact"}
}
```

A fuller document may also carry `quadratic_terms` and a `constant` on the
objective, soft constraints, and more solver preferences:

```json
{
  "constraints": [
    {
      "id": "prefer_item_b",
      "type": "soft",
      "weight": 5,
      "terms": [{"variable": "item_b", "coefficient": 1}],
      "operator": "==",
      "rhs": 1
    }
  ],
  "solver": {
    "backend": "simulated_annealing",
    "num_reads": 100,
    "num_sweeps": 1000,
    "seed": null,
    "top_k": 5,
    "max_retries": 3,
    "penalty_multiplier": 2.0
  }
}
```

## Top-level fields

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `version` | `"1.0"` \| `"1.1"` | no | `"1.0"` | Schema version. `1.1` is a superset of `1.0` and is required as soon as any variable is an integer. |
| `name` | string | **yes** | — | Problem name; echoed in CLI reports and logs. |
| `description` | string \| null | no | `null` | Free text for humans. Never sent to a remote vendor. |
| `variables` | array of [Variable](#variables) | **yes** | — | The decision variables. At least one is required (`NO_VARIABLES`). |
| `objective` | [Objective](#objective) | **yes** | — | What to minimize or maximize. |
| `constraints` | array of [Constraint](#constraints) | **yes** | — | Hard and soft constraints. May be empty (`[]`). |
| `solver` | [SolverPreferences](#solver-preferences) | no | all defaults | Backend choice and search parameters. |

Validation collects **all** errors in one pass and returns them as a
structured `invalid_problem` result; an invalid problem is never handed to a
solver, and no network call or quota is ever spent on one.

Numeric fields refuse a boolean or a string rather than coercing it: `true` is
a flag and `"10"` is text, so either one in a coefficient, a right-hand side, a
weight or a count is a schema error. An integer is still accepted where a float
is expected (`2` → `2.0`), and an integral float where an integer is expected
(`10.0` → `10`).

### Unknown fields are rejected

Every object in the document — the problem, a variable, a term, a constraint,
the solver block and its option blocks — accepts only the fields listed on
this page. A field the schema does not declare is a schema error naming its
path, never dropped: a `"variable3"` on a quadratic term, a `"cubic_terms"`
block, or a `"num_restarts"` in `solver` would otherwise vanish silently and
the server would solve a *different* problem that passes every check. The
published JSON Schema carries `additionalProperties: false` on every object
for the same reason.

On the CLI this is exit code `2`:

```console
$ annealbridge solve problem.json
Error: 'problem.json' is not a valid optimization problem:
  objective.cubic_terms: unknown field, not in the problem schema (see 'annealbridge export-schema')
```

Over MCP it is a tool error (not a `SolveResult`) whose text names the same
path, in the same channel as a boolean in a numeric field.

## Variables

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `name` | string | **yes** | — | Unique across the problem. Must not start with `__`, which is reserved for compiler-internal slack and integer-encoding bits. |
| `type` | `"binary"` \| `"integer"` | no | `"binary"` | `binary` is a 0/1 choice; `integer` is a bounded integer. |
| `lower_bound` | integer \| null | integer only | `null` | Inclusive lower bound. Must be absent for a binary variable. |
| `upper_bound` | integer \| null | integer only | `null` | Inclusive upper bound. Must be absent for a binary variable. |
| `description` | string \| null | no | `null` | Free text for humans. |

Rules enforced by the validator, each with its own error code:

- Duplicate names are rejected with `DUPLICATE_VARIABLE`; a name starting with
  a double underscore with `RESERVED_VARIABLE_NAME`; an empty `variables` list
  with `NO_VARIABLES`.
- A binary variable that carries bounds is rejected with `BOUNDS_ON_BINARY`.

### Integer variables (version 1.1)

A variable may be a bounded integer instead of a 0/1 choice. It is declared
with `type: "integer"` plus **both** `lower_bound` and `upper_bound`, and the
problem must then carry `"version": "1.1"` at its top level. This is the
variable block of
[examples/integer_knapsack.json](../examples/integer_knapsack.json) — how many
copies of each item to take:

```json
{
  "version": "1.1",
  "variables": [
    {"name": "item_a", "type": "integer", "lower_bound": 0, "upper_bound": 3},
    {"name": "item_b", "type": "integer", "lower_bound": 0, "upper_bound": 3},
    {"name": "item_c", "type": "integer", "lower_bound": 0, "upper_bound": 3},
    {"name": "item_d", "type": "integer", "lower_bound": 0, "upper_bound": 3}
  ]
}
```

Rules, each enforced by the validator with its own error code:

- Both bounds are required (`INTEGER_BOUNDS_MISSING`), both must be integers
  (a JSON `1.5` or a boolean is rejected at the schema), and
  `upper_bound > lower_bound` (`INTEGER_BOUNDS_INVALID` — equal bounds are a
  constant, not a variable). A negative `lower_bound` is fine.
- Each bound must satisfy `|bound| <= 2³¹ − 1` (`INTEGER_RANGE_TOO_LARGE`).
  Inside that range every decoded value, every sum and every product of two
  values fits 64-bit *integer* arithmetic without overflow. The compiled
  model's biases are float64, so a product larger than 2⁵³ can still be rounded
  there. Unbounded integers are not supported.
- Every inequality constraint must satisfy
  `Σ|coefficient| · max(|lower_bound|, |upper_bound|) + |rhs| <= 2⁵³`
  (`INEQUALITY_MAGNITUDE_TOO_LARGE`; a binary variable's bound counts as 1).
  See [Integer coefficients for inequality constraints](#integer-coefficients-for-inequality-constraints).
- A problem that declares any integer variable must say `"version": "1.1"`;
  with `"version": "1.0"` it is rejected with `INTEGER_REQUIRES_VERSION_1_1`.
  Version `1.1` is a superset of `1.0`: a `1.1` problem with only binary
  variables is legal, and every `1.0` document keeps its exact `1.0` behaviour
  (the compiled models, estimates and penalties for `1.0` problems are pinned
  bit for bit by a golden test — see [Testing](testing.md)).
- An objective term `x·x` is legal for an integer `x` (it is a genuine square);
  for a binary variable it is still rejected with `SELF_QUADRATIC_TERM`,
  because there `x·x = x`.

#### BQM path versus CQM path

What the compiler does with an integer depends on the model type of the chosen
backend, and the caller never sees either encoding:

- **BQM path** (`exact`, `simulated_annealing`, `tabu`,
  `simulated_bifurcation`, `dwave_qpu`, `leap_hybrid_bqm`, `fujitsu_da`): the
  integer is expanded into
  `(upper_bound − lower_bound).bit_length()` binary bits (`0..3` → 2 bits,
  `−3..4` → 3 bits) using the same binary expansion as inequality slack. These
  bits are compiler-internal (`__int_` prefix), they count towards the
  estimated number of compiled variables, and the solver's bit rows are decoded
  back into integer values before any candidate is validated or ranked. The
  four `0..3` integers of the integer knapsack example plus its slack bits
  compile to 15 variables, which `annealbridge validate` reports.
- **CQM path** (`leap_hybrid_cqm`): the integer is passed to the model natively
  as a `dimod` `INTEGER` variable with its bounds; no encoding bits are
  counted.

Solutions always report integers as plain `int` values inside their declared
bounds. `annealbridge recommend` labels the difference with the reason codes
`R_INTEGER_NATIVE` (CQM backend), `R_INTEGER_ENCODED` (BQM backend) and
`R_INTEGER_BLOWUP` (a BQM backend whose encoding triggers the
`INTEGER_QUADRATIC_BLOWUP` warning; such a backend is ranked after the others).
See [Limitations](limitations.md) for the cost of wide ranges.

## Objective

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `direction` | `"minimize"` \| `"maximize"` | **yes** | — | Optimization direction. |
| `linear_terms` | array of linear term | **yes** | — | `coefficient · variable`. May be empty. |
| `quadratic_terms` | array of quadratic term | no | `[]` | `coefficient · variable1 · variable2`. |
| `constant` | number | no | `0` | Added to every objective value. |

A **linear term** is `{"variable": <name>, "coefficient": <number>}`.
A **quadratic term** is
`{"variable1": <name>, "variable2": <name>, "coefficient": <number>}`.

- Every referenced name must be declared, or the term is rejected with
  `UNKNOWN_VARIABLE`.
- A coefficient or the constant that is NaN or infinite is rejected with
  `NON_FINITE_COEFFICIENT`.
- Duplicate terms are **not** an error: the compiler sums them and the
  validator emits the `DUPLICATE_TERM_MERGED` warning.
- `variable1 == variable2` is rejected with `SELF_QUADRATIC_TERM` for a binary
  variable and accepted as a genuine square for an integer one.

## Constraints

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `id` | string | **yes** | — | Unique per problem; results are traced back to it. |
| `description` | string \| null | no | `null` | Free text for humans. |
| `type` | `"hard"` \| `"soft"` | **yes** | — | `hard` must be satisfied; `soft` is a weighted preference. |
| `terms` | array of linear term | **yes** | — | The left-hand side. Must be non-empty. |
| `operator` | `"=="` \| `"<="` \| `">="` | **yes** | — | Comparison against `rhs`. |
| `rhs` | number | **yes** | — | The right-hand side. |
| `weight` | number \| null | soft only | `null` | Cost per unit of squared violation, in objective units. |

Constraints are linear in the declared variables; there is no quadratic
constraint form.

Rules enforced by the validator:

- An empty `terms` list is rejected with `EMPTY_CONSTRAINT`; two constraints
  sharing an `id` with `DUPLICATE_CONSTRAINT_ID`.
- A hard constraint that carries a `weight` is rejected with
  `HARD_CONSTRAINT_HAS_WEIGHT` — the penalty for hard constraints is decided by
  the server, never by the caller.
- A soft constraint without a positive `weight` is rejected with
  `SOFT_CONSTRAINT_MISSING_WEIGHT`.
- A hard constraint that no assignment within the declared bounds can satisfy
  is rejected with `TRIVIALLY_INFEASIBLE`. The soft constraint of the same
  shape is legal — its weight is simply always paid — and gets the
  `SOFT_ALWAYS_VIOLATED` warning instead.
- An inequality that is always satisfied over the declared bounds gets the
  `REDUNDANT_CONSTRAINT` warning; it is not an error.

### Integer coefficients for inequality constraints

Inequality constraints (`<=` / `>=`) are encoded into the BQM using binary
slack variables, which requires an exact integer range. Therefore **every
`coefficient` and the `rhs` of a `<=` / `>=` constraint must be an integer
value** — the JSON number may be written as `6` or `6.0`, but it must be
mathematically integral. Non-integer values are rejected with
`NON_INTEGER_INEQUALITY`. Equality (`==`) constraints are not subject to this
restriction, and no automatic scaling of fractional coefficients is performed.

The restriction also applies on the CQM path, where slack variables are not
used at all. It is kept deliberately conservative so that both paths accept
exactly the same problems. It applies to integer variables as well: their
bounds are integers, so an integer coefficient keeps the slack range exact.

Integrality alone is not enough for very large numbers. The slack range of an
inequality is computed in float64 from `coefficient × bound`, and beyond 2⁵³
those products are no longer exact: the range can come out too small, a
feasible assignment then has no encoding, and an exhaustive backend would
wrongly "prove" the problem infeasible. Hence the magnitude rule

```text
Σ|coefficient| · max(|lower_bound|, |upper_bound|) + |rhs|  <=  2⁵³
```

(`INEQUALITY_MAGNITUDE_TOO_LARGE`, a binary variable's bound counting as 1).
Rescale the unit of the coefficients or the variables, or tighten the bounds.

### Soft constraint weights

A soft constraint contributes `weight × (violation)²` to the solver's energy
and to the solution's `soft_violation_score`. The `weight` is expressed **in
objective units** and is **not normalized** — if your objective coefficients
are in the thousands, a weight of 5 has almost no influence.

The compile step exposes `objective_scale`, an upper bound on how much the
objective can vary (`max − min`) over the declared bounds:

```text
objective_scale = max(1.0, Σ_terms |coefficient| × width(term))
```

where `width` is the exact `max − min` of that single term over the variable
box: `upper − lower` for a linear term, the spread of the four corner products
for `x·y`, and the spread of the square for `x·x`. For an all-binary problem
every width is 1, so the formula reduces to `Σ|linear| + Σ|quadratic|`. Choose
weights relative to that scale; `objective_scale` never includes soft weights,
and the objective's `constant` does not affect it. A weight below 1 % of the
scale raises the `SOFT_WEIGHT_SMALL` warning.

`annealbridge validate` prints `objective_scale`, and
`ProblemValidationResult.objective_scale` carries it to a programmatic caller.

On the CQM path a soft constraint over binary variables only is submitted as a
native weighted constraint. A soft constraint that involves an integer variable
is instead folded into the objective as `weight × (violation)²` (with an
integer slack for inequalities), because `dimod` only offers a linear penalty
for integer variables and that would not be the formula the validator scores
with. Either way the solver's soft energy equals the validator's
`soft_violation_score`.

Keeping those two equal means a soft constraint is scored from its **exact**
residual: the feasibility tolerance that decides `satisfied` is not applied to
the score. A residual small enough to leave `satisfied: true` therefore still
contributes `weight × residual²`, because that is what the compiled model
charges for it — with a large enough weight a residual of 1e-9 is worth real
energy, and rounding it away would make the ranking prefer assignments the
solver was paying to avoid. Hard constraints are unaffected: a hard constraint
that holds within the tolerance reports `violation_amount: 0`.

### Hard constraint penalties

The hard-constraint penalty λ is computed by the penalty strategy, never taken
from the caller. It is sized against the whole non-penalty energy landscape:

```text
penalty_scale   = objective_scale + Σ_soft weight × D²
initial_penalty = penalty_scale × penalty_multiplier   (default multiplier 2)
retry           = previous × 2
```

where `D` is the largest absolute value the soft constraint's squared term can
reach over all assignments within the declared bounds (encoding and slack bits
included). Any assignment that violates a hard constraint therefore costs at
least `objective_min + λ`, while the best feasible assignment costs at most
`objective_max + Σ_soft weight × D²`, so a soft weight far above the objective
cannot drown a hard constraint: with `penalty_multiplier > 1` the lowest-energy
assignment of the compiled model is always feasible whenever one exists, for
any integer bounds (negative lower bounds included). Without soft constraints
`penalty_scale` equals `objective_scale`.

Two caveats remain. The argument assumes a violation costs at least one penalty
unit, i.e. integer coefficients — inequalities already require them, but an
equality with fractional coefficients or right-hand side can be violated by
less than one unit. And the guarantee is about the model's global minimum: a
non-exhaustive backend may not reach it, which is what the doubling retry is
for.

Soft weights are only used to *bound* the energy the penalty must dominate;
they are never used as, or substituted for, the hard penalty itself. A penalty
that would have to double past the floating-point range stops with a structured
`PENALTY_OVERFLOW` error rather than a solver error. On the CQM path there is
no penalty at all: hard constraints are submitted natively, `penalty` is
`null`, and there is never a retry.

### Ranking

Ranking accounts for soft violations. Solutions are ordered by `ranking_score`:

```text
ranking_score = objective_value + soft_violation_score   (minimize)
ranking_score = objective_value − soft_violation_score   (maximize)
```

with deterministic tie-breaking. Both components are reported separately so a
consumer can re-rank. Only solutions that satisfy every hard constraint under
independent re-validation are ranked at all.

## Solver preferences

The `solver` block is optional; every field has a default.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `backend` | enum | `"simulated_annealing"` | One of `simulated_annealing`, `exact`, `tabu`, `simulated_bifurcation`, `dwave_qpu`, `leap_hybrid_bqm`, `leap_hybrid_cqm`, `fujitsu_da`. See [Backends](backends.md). |
| `num_reads` | integer > 0 | `100` | Number of samples to request. Bounded by policy (`LOCAL_READS_LIMIT` / `QPU_READS_LIMIT`). |
| `num_sweeps` | integer > 0 | `1000` | Annealing sweeps per read on backends that take them — on `simulated_bifurcation`, integration steps. Bounded by policy (`SWEEPS_LIMIT`). |
| `seed` | integer \| null | `null` | Random seed. A backend that does not support seeding raises the `SEED_IGNORED` warning. A backend that declares a seed range — its own sampler's rule, so it differs per backend (`simulated_annealing`: `0`–`2147483647`; `tabu` and `simulated_bifurcation`: `0`–`4294967295`) — refuses a seed outside it with `INVALID_SOLVER_PREFERENCE`. |
| `top_k` | integer > 0 | `5` | Maximum number of ranked solutions to return. Bounded by policy (`TOP_K_LIMIT`). |
| `max_retries` | integer >= 0 | `3` | Additional attempts with a doubled hard penalty when no feasible solution was found. Bounded by policy (`RETRY_LIMIT`). |
| `penalty_multiplier` | finite number > 0 | `2.0` | Multiplier applied to `penalty_scale` for the first attempt. |
| `simulated_bifurcation` | object \| null | `null` | Options for the `simulated_bifurcation` backend. |
| `dwave_qpu` | object \| null | `null` | Options for the `dwave_qpu` backend. |
| `leap_hybrid_bqm` | object \| null | `null` | Options for the `leap_hybrid_bqm` backend. |
| `leap_hybrid_cqm` | object \| null | `null` | Options for the `leap_hybrid_cqm` backend. |
| `fujitsu_da` | object \| null | `null` | Options for the `fujitsu_da` backend. |

A value outside its allowed range is rejected with `INVALID_SOLVER_PREFERENCE`;
a value above a server-side ceiling is rejected with `resource_limit_exceeded`
and is **never silently clamped**. The ceilings are configured through
`ANNEALBRIDGE_*` environment variables — see [Configuration](configuration.md).

A parameter the chosen backend cannot use is not an error: it raises the
`PARAMETER_IGNORED` warning (for example `num_sweeps` on a backend that takes
no sweeps, `penalty_multiplier` / `max_retries` on the CQM path, which
applies no hard penalty, or `max_retries` on an exhaustive backend such as
`exact`, where a retry can never surface new samples). Filling in an option
block that does not belong to the selected backend raises the same warning.
Only a non-default value triggers it.

### Backend option blocks

Each block is named exactly like the backend it belongs to. An option left out
is not sent at all, so the backend's or the vendor's own default applies.

**`simulated_bifurcation`**

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `mode` | `"discrete"` \| `"ballistic"` | `"discrete"` | Which simulated bifurcation variant to run. `"discrete"` (dSB) drives the oscillators with the *signs* of their positions and is the stronger variant on dense problems; `"ballistic"` (bSB) uses the continuous positions and does better on some small penalty-dominated problems where dSB stalls. See [Backends](backends.md#simulated_bifurcation). |

The numerical constants of the dynamics are the paper's and are not exposed.

**`dwave_qpu`**

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `annealing_time_us` | finite number > 0 \| null | `null` | Annealing time per read in microseconds. Bounded by `ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US` (`QPU_ANNEALING_TIME_LIMIT`). |
| `chain_strength` | finite number > 0 \| null | `null` | Embedding chain strength. `null` uses Ocean's default. |
| `auto_scale` | boolean | `true` | Let the sampler auto-scale biases to the QPU range. |

**`leap_hybrid_bqm`** and **`leap_hybrid_cqm`**

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `time_limit_seconds` | finite number > 0 \| null | `null` | Hybrid solver time budget. `null` uses the sampler's minimum. Above `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` it is refused with `REMOTE_TIME_LIMIT`, never clamped. |

For `leap_hybrid_cqm` the sampler's minimum `time_limit_seconds` is 5 s: a
lower value is raised to the minimum and reported in
`metadata.effective_time_limit_seconds`.

**`fujitsu_da`**

| Option | Type | Range | Vendor default when omitted | Meaning |
| --- | --- | --- | --- | --- |
| `time_limit_seconds` | integer | 1–3600 | 10 | Annealing time budget. Compared against `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` and refused with `REMOTE_TIME_LIMIT` when above it. |
| `num_run` | integer | 1–1024 | 16 | Parallel annealing runs. |
| `num_group` | integer | 1–16 | 1 | Groups per run. |
| `num_output_solution` | integer | 1–1024 | 5 | Solutions returned per group. |

Only these four values are ever forwarded to Fujitsu; `num_reads`,
`num_sweeps` and `seed` are not, and are reported as `PARAMETER_IGNORED` /
`SEED_IGNORED`.

## Bundled examples

Four ready-to-run problems ship with the repository.

- [examples/knapsack.json](../examples/knapsack.json) — 0/1 knapsack, 4 items,
  capacity 10, one hard `<=` constraint. Global optimum selects items A and C
  for a total value of 17.
- [examples/assignment.json](../examples/assignment.json) — 3 workers × 3 tasks
  assignment with six hard equality (one-hot) constraints, minimizing total
  cost. Global optimum: alice=cook, bob=clean, carol=drive, total cost 8.
- [examples/tsp.json](../examples/tsp.json) — traveling salesman over 4 cities
  with symmetric distances, encoded as city × position binaries. The optimal
  cyclic tour a-b-c-d has total length 8.
- [examples/integer_knapsack.json](../examples/integer_knapsack.json) — a
  `version: "1.1"` bounded integer knapsack: four items, each taken 0–3 times,
  capacity 18, one hard `<=` constraint and one soft constraint (prefer at most
  two of B and C combined, weight 3). The unique global optimum is
  `item_a=0, item_b=1, item_c=1, item_d=3` with value 34; `annealbridge
  validate` reports 15 compiled variables on the BQM path.

All four declare a local backend; try `--backend simulated_annealing` to
compare against `exact` (simulated annealing is heuristic — without a fixed
`seed` and enough `num_reads` it may return a feasible but sub-optimal integer
knapsack). With remote execution configured, `--backend leap_hybrid_cqm` runs
the same problem down the CQM path, and `--backend fujitsu_da` sends the
compiled QUBO to the Digital Annealer.

## Exporting the JSON Schema

```bash
annealbridge export-schema
```

prints the complete JSON Schema for `OptimizationProblem` — the same schema an
agent can use for structured output, and the same one returned in the
`problem_json_schema` field of the `get_optimization_capabilities` MCP tool
(called with `include_schema: true`).
Because it is generated from the pydantic models, it never drifts from the
behaviour documented here: every field carries a `description`, and every
object declares `additionalProperties: false` (see
[Unknown fields are rejected](#unknown-fields-are-rejected)). See
[CLI](cli.md) and [MCP server](mcp.md).
