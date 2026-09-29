[← Back to README](../README.md) · [Documentation index](README.md)

# Problem JSON Format

This page is the complete reference for the `OptimizationProblem` document —
the only thing an agent or a caller ever writes. It covers every top-level
field, the variable / objective / constraint models, the cardinality
constraints of version 1.2, the index sets, parameters, variable families and
templates of version 1.3, the solver preference block, the rules for bounded
integer variables, and how soft weights and hard penalties are interpreted.

The document is the whole public API. There is no QUBO matrix, no penalty λ,
no slack variable and no integer encoding anywhere in it: those are the
deterministic core's business, and the caller never sees them.

For what comes back, see [Output format](output-format.md); for the codes a
rejected document produces, see [Errors and warnings](errors.md).

## A minimal problem

A 0/1 knapsack with a capacity of 10 — the reduced form of
[examples/knapsack.json](../examples/knapsack.json), a file in a repository
checkout (an installed package prints it with `annealbridge example knapsack`,
and the MCP server serves it as `annealbridge://examples/knapsack`):

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
| `version` | `"1.0"` \| `"1.1"` \| `"1.2"` \| `"1.3"` | no | `"1.0"` | Schema version. Each version is a superset of the one before: `1.1` or later is required as soon as any variable is an integer, `1.2` or later as soon as `cardinality_constraints` is non-empty, `1.3` as soon as any [template field](#templates-version-13) is non-empty. See [Schema versions](#schema-versions). |
| `name` | string | **yes** | — | Problem name; echoed in CLI reports and logs. |
| `description` | string \| null | no | `null` | Free text for humans. Never sent to a remote vendor. |
| `variables` | array of [Variable](#variables) | **yes** | — | The decision variables. At least one is required (`NO_VARIABLES`). |
| `objective` | [Objective](#objective) | **yes** | — | What to minimize or maximize. |
| `constraints` | array of [Constraint](#constraints) | **yes** | — | Hard and soft linear constraints. Required even when there are none: may be empty (`[]`). |
| `cardinality_constraints` | array of [CardinalityConstraint](#cardinality-constraints-version-12) | no | `[]` | Version 1.2: how many of a set of binary variables are chosen (exactly, at most or at least *k*). |
| `index_sets` | array of [IndexSet](#index_sets) | no | `[]` | Version 1.3: ordered sets of elements that template indices range over. |
| `parameters` | array of [Parameter](#parameters) | no | `[]` | Version 1.3: tables of numbers keyed by index set elements. |
| `variable_families` | array of [VariableFamily](#variable_families) | no | `[]` | Version 1.3: one variable per combination of index set elements. |
| `constraint_templates` | array of [ConstraintTemplate](#constraint_templates) | no | `[]` | Version 1.3: linear constraints repeated over index sets. |
| `cardinality_constraint_templates` | array of [CardinalityConstraintTemplate](#cardinality_constraint_templates) | no | `[]` | Version 1.3: cardinality constraints repeated over index sets. |
| `solver` | [SolverPreferences](#solver-preferences) | no | all defaults | Backend choice and search parameters. |

The objective carries the other two [template fields](#templates-version-13),
`linear_term_templates` and `quadratic_term_templates`. All seven are optional,
empty by default and left out of a serialized problem while empty, so a
document written for an older version dumps exactly as before.

### Schema versions

| Version | Adds | Relation to the one before |
| --- | --- | --- |
| `"1.0"` | binary variables, linear constraints | — |
| `"1.1"` | bounded [integer variables](#integer-variables-version-11) | superset of `1.0` |
| `"1.2"` | [cardinality constraints](#cardinality-constraints-version-12) | superset of `1.1`; may also declare integer variables |
| `"1.3"` | [index sets, parameters, variable families and templates](#templates-version-13) | superset of `1.2`; may also declare integer variables and cardinality constraints |

A newer version accepts everything an older one does, with the same meaning,
so a document can always be moved to a newer version by changing only its
`version`; the compiled models and estimates do not change, and the only
difference in validation is that from `1.2` on a hard linear at-most-one may
get the advisory
[`CARDINALITY_FORM_AVAILABLE`](#linear-at-most-one-constraints) warning. A
`1.3` document without templates behaves exactly like the same document
labelled `1.2`. The rule is a
minimum per feature, judged by what the document actually uses: an integer
variable needs `1.1` or later (`INTEGER_REQUIRES_VERSION_1_1` otherwise), a
non-empty `cardinality_constraints` list needs `1.2` or later
(`FEATURE_REQUIRES_NEWER_VERSION` otherwise), and a non-empty template field
needs `1.3` (`FEATURE_REQUIRES_NEWER_VERSION` too, its message naming every
template field the document uses). An empty `cardinality_constraints` list, or
an empty template field, is accepted by every version.
`get_optimization_capabilities` and
`annealbridge capabilities --json` list the accepted versions as
`schema_versions`, oldest first.

Validation collects **all** errors in one pass and returns them as a
structured `invalid_problem` result; an invalid problem is never handed to a
solver, and no network call or quota is ever spent on one.

Numeric fields refuse a boolean or a string rather than coercing it: `true` is
a flag and `"10"` is text, so either one in a coefficient, a right-hand side, a
weight or a count is a schema error (`INVALID_FIELD_VALUE`, with a message such
as `coefficient must be a number, not a boolean`). An integer is still accepted where a float
is expected (`2` → `2.0`), and an integral float where an integer is expected
(`10.0` → `10`). The one place a string is a legal value for a number is a
template's `coefficient`, `rhs` or `weight`, where it names a
[parameter](#parameters) (`"dist[i,j]"`); a number written as a string is
refused there too (`TEMPLATE_REFERENCE_INVALID`).

### Unknown fields are rejected

Every object in the document — the problem, a variable, a term, a constraint,
the index sets, parameters, families and templates of version 1.3, the solver
block and its option blocks — accepts only the fields listed on
this page. A field the schema does not declare is a schema error naming its
path, never dropped: a `"variable3"` on a quadratic term, a `"cubic_terms"`
block, or a `"num_restarts"` in `solver` would otherwise vanish silently and
the server would solve a *different* problem that passes every check. The
published JSON Schema carries `additionalProperties: false` on every object
for the same reason; even a parameter table is a typed list of rows, never a
free-form object.

A field that only a newer version has — `cardinality_constraints` before `1.2`,
a template field before `1.3` — is not an unknown field either. As an empty
list it is accepted by every version; a non-empty one is not a schema error but
`FEATURE_REQUIRES_NEWER_VERSION` at `version`, and one whose entries are
malformed gets the schema errors inside it first.

It is reported as `UNKNOWN_FIELD`, one of the three
[schema error](errors.md#schema-errors) codes, alongside `MISSING_FIELD` and
the `INVALID_FIELD_VALUE` a boolean in a numeric field gets; every schema
error in the document is listed at once. On the CLI this is exit code `2`:

```console
$ annealbridge solve problem.json
Error: 'problem.json' is not a valid optimization problem.

Schema errors (1):
  [UNKNOWN_FIELD] objective.cubic_terms: unknown field, not in the problem schema
    recommended action: The field is not part of the problem schema and unknown fields are never ignored; remove it, or use the field the schema declares for that purpose. Read the schema before inventing a field: get_optimization_capabilities with include_schema true, the annealbridge://schema resource, or annealbridge export-schema.
```

Over MCP it is a structured result, not a tool error: `status:
"invalid_problem"` from `solve_optimization` and `valid: false` from
`validate_optimization_problem` and `recommend_backend`, with the same code,
path and recommended action in `errors`. Semantic errors — an undeclared
variable, say — are reported once the document fits the schema.

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
problem must then carry `"version": "1.1"` or later at its top level. This is the
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
- A problem that declares any integer variable must say `"version": "1.1"`
  or later; with `"version": "1.0"` it is rejected with
  `INTEGER_REQUIRES_VERSION_1_1`. Version `1.1` is a superset of `1.0`: a
  `1.1` problem with only binary variables is legal, and every `1.0` document
  keeps its exact `1.0` behaviour (the compiled models, estimates and
  penalties for `1.0` problems are pinned bit for bit by a golden test, and
  the rest of the `1.0` and `1.1` behaviour by a compatibility golden — see
  [Testing](testing.md)).
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
| `id` | string | **yes** | — | Unique per problem, across `constraints` and `cardinality_constraints` together; results are traced back to it. |
| `description` | string \| null | no | `null` | Free text for humans. |
| `type` | `"hard"` \| `"soft"` | **yes** | — | `hard` must be satisfied; `soft` is a weighted preference. |
| `terms` | array of linear term | **yes** | — | The left-hand side. Must be non-empty. |
| `operator` | `"=="` \| `"<="` \| `">="` | **yes** | — | Comparison against `rhs`. |
| `rhs` | number | **yes** | — | The right-hand side. |
| `weight` | number \| null | soft only | `null` | Cost per unit of squared violation, in objective units. |

Constraints are linear in the declared variables; there is no quadratic
constraint form. A rule that only counts how many of a set of binary variables
are chosen — exactly one, at most one, at least *k* — is better declared in
[`cardinality_constraints`](#cardinality-constraints-version-12) (version 1.2):
it means the same as the linear constraint with every coefficient 1, and a hard
at-most-one compiles smaller there.

Rules enforced by the validator:

- An empty `terms` list is rejected with `EMPTY_CONSTRAINT`; two constraints
  sharing an `id` — two linear ones, two cardinality ones, or one of each —
  with `DUPLICATE_CONSTRAINT_ID`.
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

A soft [cardinality constraint](#cardinality-constraints-version-12) is scored
the same way, its violation being how many variables the count is off by — for
`<=` and `>=` only the excess or the shortfall: a soft `"<=", 1` with three
variables chosen pays `weight × 2²`. It compiles exactly like the equivalent
linear constraint, slack bits included, so on both paths the solver's soft
energy again equals the score. The slack-free pairwise encoding is reserved for
*hard* at-most-ones: for three or more chosen variables it would charge less
than `weight × violation²`, and the ranking would then prefer assignments the
solver was paying too little for.

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

A declared hard at-most-one — a [cardinality constraint](#cardinality-constraints-version-12)
with `"<="` and `rhs` 1 over two or more variables — is compiled on the BQM
path as the pairwise penalty `λ · Σ_{i<j} x_i·x_j` rather than with a slack
bit. It meets the two conditions the argument above rests on: it is zero for
every assignment that satisfies the constraint, and at least λ for every one
that violates it (`λ · k(k−1)/2` when *k* variables are chosen). It involves no
slack or encoding bit, so the guarantee and the `penalty_scale` formula are
unchanged. It does lack the negative linear term the slack form contributes,
so a variable's net linear bias can come out larger than under the slack
encoding, and a problem can reach the floating-point limit — and
`PENALTY_OVERFLOW` — sooner than its slack form would. That is always the
structured error below, never a silently wrong model.

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

## Cardinality constraints (version 1.2)

The most common rule in assignment, timetabling and selection problems only
counts: each exam takes exactly one slot, two clashing exams share at most one
slot, a team has at least two seniors. Version `1.2` declares such a rule
directly, in the top-level list `cardinality_constraints`. Each entry counts
how many of the binary variables it lists are chosen (take the value 1) and
compares that count with `rhs`:

| Rule | `operator` | `rhs` |
| --- | --- | --- |
| one-hot: exactly one | `"=="` | `1` |
| exactly *k* | `"=="` | *k* |
| at most *k* (at most one: *k* = 1) | `"<="` | *k* |
| at least *k* | `">="` | *k* |

An excerpt of
[examples/exam_timetabling.json](../examples/exam_timetabling.json), whose
rules are all cardinality constraints:

```json
{
  "version": "1.2",
  "constraints": [],
  "cardinality_constraints": [
    {
      "id": "math_once",
      "type": "hard",
      "variables": ["math_mon_am", "math_mon_pm", "math_tue_am"],
      "operator": "==",
      "rhs": 1
    },
    {
      "id": "chemistry_biology_apart_mon_am",
      "type": "hard",
      "variables": ["chemistry_mon_am", "biology_mon_am"],
      "operator": "<=",
      "rhs": 1
    },
    {
      "id": "small_hall_mon_am",
      "type": "soft",
      "variables": ["math_mon_am", "physics_mon_am", "chemistry_mon_am", "biology_mon_am"],
      "operator": "<=",
      "rhs": 1,
      "weight": 3
    }
  ]
}
```

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `id` | string | **yes** | — | Unique across `constraints` and `cardinality_constraints` together; results are traced back to it. |
| `description` | string \| null | no | `null` | Free text for humans. Never sent to a remote vendor. |
| `type` | `"hard"` \| `"soft"` | **yes** | — | As for a linear constraint. |
| `variables` | array of string | **yes** | — | The binary variables counted; each one that takes the value 1 counts once. Non-empty, each name at most once. |
| `operator` | `"=="` \| `"<="` \| `">="` | **yes** | — | How the count compares with `rhs`. |
| `rhs` | integer | **yes** | — | The count *k*, with \|`rhs`\| ≤ 2³¹ − 1. A fractional number, a string, a boolean or a larger integer is a schema error (`INVALID_FIELD_VALUE` at `cardinality_constraints[i].rhs`). |
| `weight` | number \| null | soft only | `null` | Cost per unit of squared violation, in objective units; the violation is how many variables the count is off by. |

**Meaning.** An entry means exactly what the linear constraint over the same
variables with every coefficient 1 means: the same feasibility, the same
`violation_amount`, the same soft score `weight × violation²`. A weighted sum,
or a sum over integer variables, stays a linear constraint in `constraints`,
which is still required (use `[]` when every rule is a cardinality one). The
field itself is optional; an empty list is the same as leaving it out, and it
is left out when a problem is serialized, so a `1.0` or `1.1` problem dumps
exactly as it always did.

**Validation.** Paths point at what was written, under
`cardinality_constraints[i]`, and the messages speak in counts
(`Hard cardinality constraint too_many counts 3 variables, so between 0 and 3
are chosen, but requires == 5`):

| Condition | Code | `path` |
| --- | --- | --- |
| a non-empty list in a `1.0` or `1.1` document | `FEATURE_REQUIRES_NEWER_VERSION` | `version` |
| `variables` is empty | `EMPTY_CONSTRAINT` | `cardinality_constraints[i]` |
| a name is not declared | `UNKNOWN_VARIABLE` | `cardinality_constraints[i].variables[j]` |
| a name is an integer variable | `CARDINALITY_VARIABLE_NOT_BINARY` | `cardinality_constraints[i].variables[j]` |
| a name is listed again | `DUPLICATE_CARDINALITY_VARIABLE` | `cardinality_constraints[i].variables[j]` (the repeat) |
| the `id` is already used, by either list | `DUPLICATE_CONSTRAINT_ID` | `cardinality_constraints[i]` |
| a hard entry carries a `weight` | `HARD_CONSTRAINT_HAS_WEIGHT` | `cardinality_constraints[i].weight` |
| a soft entry has no positive `weight` | `SOFT_CONSTRAINT_MISSING_WEIGHT` | `cardinality_constraints[i].weight` |
| the `weight` is NaN or infinite | `NON_FINITE_COEFFICIENT` | `cardinality_constraints[i].weight` |
| a hard entry no count from 0 to *n* satisfies (`"==", 5` over 3 variables, `"<=", -1`) | `TRIVIALLY_INFEASIBLE` | `cardinality_constraints[i]` |

Once an entry is free of those errors, the warnings a linear constraint gets
apply to it too, judged over the count range 0 to *n*: `SOFT_ALWAYS_VIOLATED`
for a soft entry no count satisfies, `REDUNDANT_CONSTRAINT` for one every count
satisfies (`"<=", n`, `">=", 0`, or an at-most-one over a single variable), and
`SOFT_WEIGHT_SMALL`. `FEATURE_REQUIRES_NEWER_VERSION` adds to its message that
`version` defaults to `"1.0"` when it was left out, and, for a `1.0` document
that also has integer variables, that `"1.2"` allows those too (the
`INTEGER_REQUIRES_VERSION_1_1` error is still reported beside it). See
[Errors and warnings](errors.md) for the recommended actions.

**Encoding.** The caller never sees it, but it decides the compiled size:

- **Hard at-most-one, BQM path.** A hard `"<="` with `rhs` 1 over two or more
  variables compiles to the pairwise penalty `λ · Σ_{i<j} x_i·x_j` — no slack
  variable, no linear term, no constant, the pairs added in the order of
  `variables`. It is zero while at most one variable is chosen and at least λ
  otherwise, the same zero set and the same smallest violation cost as the
  slack form, without the slack bit a linear `<= 1` needs (see
  [Hard constraint penalties](#hard-constraint-penalties)).
  `estimated_compiled_variables` counts no slack bit for it.
- **Every other form** compiles exactly like the linear constraint with every
  coefficient 1, bit for bit: `"=="` as `λ·(Σx − k)²`, `"<="` / `">="` with
  binary slack bits (or not at all when redundant), and every soft entry as
  `weight × violation²` with its slack bits — see
  [Soft constraint weights](#soft-constraint-weights) for why a soft
  at-most-one keeps its slack.
- **CQM path** (`leap_hybrid_cqm`). Each entry is added as the native linear
  constraint it means — hard ones as hard constraints, soft ones with their
  weight and a quadratic penalty — exactly as the equivalent linear constraint
  would be.

**Order.** Wherever constraints are listed or processed, the linear
`constraints` come first and the `cardinality_constraints` after them, each in
declaration order: in `constraint_evaluations`, in the infeasibility
diagnostics' `hard_violation_rates`, and in the compiled model. Each entry is
evaluated under its own `id`, with `actual_value` the number of its variables
chosen (see [Output format](output-format.md#constraintevaluation)).

#### Linear at-most-one constraints

The pairwise encoding is only ever applied to a *declared* cardinality
constraint. A linear `<= 1` in `constraints` keeps its slack encoding in every
version, so no `1.0` or `1.1` document compiles differently than before. In a
document of version `1.2` or later on a BQM backend, a hard linear constraint
that could be declared instead — operator `"<="`, `rhs` 1, every coefficient
exactly 1, over two or more distinct binary variables — gets the advisory
warning `CARDINALITY_FORM_AVAILABLE`, whose `path` names it. Moving it into
`cardinality_constraints` keeps its meaning and drops its slack bit; nothing
changes by itself.

#### `simulated_bifurcation` and at-most-one groups

The pairwise encoding was adopted after a comparison against the slack form on
six binary problems with `simulated_annealing`, `tabu` and
`simulated_bifurcation`, every result re-validated against the original
problem. `tabu` gained the most — it reached the proven optimum on problems
where the slack form rarely did — and `simulated_annealing` held its objective
or improved it slightly. `simulated_bifurcation` improved on two problems, but
on one weighted set-packing problem (maximize) its samples collapsed to nearly
all zeros, and the cause is not understood. An all-zero assignment satisfies
every at-most-one, so such a result is feasible but poor, never wrong. For a
packing-style maximize problem with many at-most-one groups, prefer
`simulated_annealing` or `tabu`, or turn on
[`repair_local_search`](#post-processing); see
[Limitations](limitations.md).

## Templates (version 1.3)

A problem with structure repeats itself: every city is visited at exactly one
position, every pair of consecutive positions pays the distance between its
two cities. Written out, a 4-city tour already takes 16 variables, 48
quadratic terms and 8 constraints, and the count grows with the cube of the
cities. Version `1.3` lets a document state each pattern once: an *index set*
is an ordered list of elements, a *parameter* a table of numbers keyed by
elements, a *variable family* one variable per combination of elements, and
every explicit list has a *template* sibling that repeats one entry for each
combination of bound indices, filtered by conditions.

| Explicit list | Template list |
| --- | --- |
| `variables` | `variable_families` |
| `objective.linear_terms` | `objective.linear_term_templates` |
| `objective.quadratic_terms` | `objective.quadratic_term_templates` |
| `constraints` | `constraint_templates` |
| `cardinality_constraints` | `cardinality_constraint_templates` |

Together with `index_sets` and `parameters` these are the seven **template
fields**. Before anything else reads the document, the server **expands** the
templates into an ordinary problem: the explicit entries first, then every
generated one. Validation, compilation, solving, re-validation and ranking only
ever see that expanded problem, exactly as if the document had listed every
entry itself; nothing downstream knows templates exist. Explicit and template
entries can be mixed, and an explicit entry may name a generated variable.

There is no expression language. A template string is one of two tiny
grammars — a reference with optional index shifts, or a comparison of two
operands — parsed once, never evaluated as code (see [The grammar](#the-grammar)).

### A complete example

[examples/tsp_template.json](../examples/tsp_template.json) is the 4-city
travelling salesman problem of [examples/tsp.json](../examples/tsp.json),
written with templates (the `description` strings are left out here):

```json
{
  "version": "1.3",
  "name": "tsp_4_cities_template",
  "index_sets": [
    {"name": "city", "elements": ["a", "b", "c", "d"]},
    {"name": "pos", "elements": [0, 1, 2, 3], "order": "cyclic"}
  ],
  "parameters": [
    {"name": "dist", "indices": ["city", "city"], "values": [
      {"key": ["a", "b"], "value": 1}, {"key": ["a", "c"], "value": 3},
      {"key": ["a", "d"], "value": 4}, {"key": ["b", "a"], "value": 1},
      {"key": ["b", "c"], "value": 2}, {"key": ["b", "d"], "value": 5},
      {"key": ["c", "a"], "value": 3}, {"key": ["c", "b"], "value": 2},
      {"key": ["c", "d"], "value": 1}, {"key": ["d", "a"], "value": 4},
      {"key": ["d", "b"], "value": 5}, {"key": ["d", "c"], "value": 1}
    ]}
  ],
  "variable_families": [
    {"name": "x", "indices": ["city", "pos"], "type": "binary"}
  ],
  "variables": [],
  "objective": {
    "direction": "minimize",
    "linear_terms": [],
    "quadratic_term_templates": [
      {"for_each": ["p in pos", "i in city", "j in city"], "where": ["i != j"],
       "coefficient": "dist[i,j]", "variable1": "x[i,p]", "variable2": "x[j,p+1]"}
    ]
  },
  "constraints": [],
  "cardinality_constraint_templates": [
    {"id": "city_once", "type": "hard", "for_each": ["i in city"],
     "variables": [{"for_each": ["p in pos"], "variable": "x[i,p]"}],
     "operator": "==", "rhs": 1},
    {"id": "position_once", "type": "hard", "for_each": ["p in pos"],
     "variables": [{"for_each": ["i in city"], "variable": "x[i,p]"}],
     "operator": "==", "rhs": 1}
  ],
  "solver": {"backend": "exact", "top_k": 5}
}
```

It expands to:

- 16 variables, `x[a,0]`, `x[a,1]`, … `x[d,3]`, city outermost;
- 48 quadratic terms `dist[i,j] · x[i,p] · x[j,p+1]`, one per position and
  ordered pair of different cities. `pos` is `cyclic`, so the position after
  `3` is `0` and the tour closes;
- 8 cardinality constraints, `city_once[a]` … `city_once[d]` and
  `position_once[0]` … `position_once[3]`.

That is `examples/tsp.json` term for term, with `x_a_0` renamed `x[a,0]`: the
same compiled model and the same optimal tour length, 8. `variables` and
`constraints` are still required, and empty here. The distance table has no
row for a city paired with itself: `where: ["i != j"]` never asks for one, and
the parameter deliberately has no `default`, which would have silently turned
a forgotten row into 0.

### Fields

Every object below accepts only the fields listed (see
[Unknown fields are rejected](#unknown-fields-are-rejected)). A *name* — of an
index set, a parameter, a family, a template id or an index — is an
identifier: ASCII letters, digits and underscores, not starting with a digit,
at most 64 characters, and never the word `in`.

#### `index_sets`

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `name` | string | **yes** | — | A name, unique among index sets, parameters and families. |
| `elements` | array of string \| integer | **yes** | — | The elements in order, at least one, each listed once, all strings or all integers. A string is 1 to 64 of the characters `A–Z a–z 0–9 _ . -`; an integer lies within ±(2³¹ − 1). They appear in generated names, e.g. `x[a,0]`. |
| `order` | `"none"` \| `"linear"` \| `"cyclic"` | no | `"none"` | Whether an index over this set may be shifted (`p+1`, `d-2`): `none` forbids shifts; `linear` allows them, and an entry whose shifted index runs past either end is left out; `cyclic` wraps around. |
| `description` | string \| null | no | `null` | Free text for humans. Never sent to a remote vendor. |

An element that is a boolean, a number written with a fraction or an
exponent (`2.0` included) or any other JSON type is a schema error
(`INVALID_FIELD_VALUE`, `index set elements must be strings or integers`);
every other rule on elements is `INDEX_SET_INVALID`, at
`index_sets[s].elements[k]` (at `elements` itself for an empty set).

A shift moves by **positions** in `elements`, not by numeric value: with
`"elements": [0, 5, 10]`, the element after `5` is `10`. Integers are written
in generated names in decimal (`x[a,-1]`).

#### `parameters`

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `name` | string | **yes** | — | A name, unique among index sets, parameters and families. |
| `indices` | array of string | **yes** | — | The index sets its keys range over, one to eight; a reference gives one bound index per entry, e.g. `dist[i,j]`. |
| `values` | array of row | no | `[]` | The table as rows `{"key": [e1, …], "value": n}`. |
| `default` | number \| null | no | `null` | A finite value for every key that has no row. |
| `description` | string \| null | no | `null` | Free text for humans. Never sent to a remote vendor. |

A **row** has exactly two fields: `key`, one element of each of the
parameter's index sets in the order of `indices`, and `value`, a finite JSON
number. A key element of the wrong JSON type, or a `value` that is a boolean
or a string, is a schema error, as for index set elements. Each key element must be an
element of its set, of the same type — in an integer set `1` is an element and
`"1"` is not — and each key may appear in one row only. Rows may leave keys
out: a table is allowed to be sparse. Violations are `PARAMETER_TABLE_INVALID`
at `parameters[p].values[r].key[d]`, `.key` or `.value`.

A key that has no row takes `default`. Without a default, a key that is
actually used is the error `PARAMETER_VALUE_MISSING`, whose `path` is the
field that used it (a `coefficient`, a `where[k]`, an `rhs` or a `weight`) and
whose message names the parameter and the key. **A default also hides a row
left out by mistake**, so give one only when most keys genuinely share the
value. A parameter used as a cardinality template's `rhs` must hold whole
numbers within ±(2³¹ − 1) in every row and in its default
(`PARAMETER_TABLE_INVALID` at the offending row).

There are no scalar parameters: a single number is written as a JSON number
where it is used.

#### `variable_families`

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `name` | string | **yes** | — | A name, unique among index sets, parameters and families. It must not start with `__` (`RESERVED_VARIABLE_NAME`) or equal the name of an explicit variable (`DUPLICATE_TEMPLATE_NAME`, since a template could then no longer tell the two apart). |
| `indices` | array of string | **yes** | — | The index sets, one to eight. One variable is generated for every combination of their elements, the first set outermost, named `name[e1,e2,…]`. |
| `type` | `"binary"` \| `"integer"` | no | `"binary"` | As for a [variable](#variables). |
| `lower_bound` | integer \| null | integer only | `null` | As for a variable. |
| `upper_bound` | integer \| null | integer only | `null` | As for a variable. |
| `description` | string \| null | no | `null` | Copied to every generated variable. Never sent to a remote vendor. |

The type and bounds are checked once per family, with the rules, codes and
messages of an explicit variable (`BOUNDS_ON_BINARY`,
`INTEGER_BOUNDS_MISSING`, `INTEGER_BOUNDS_INVALID`,
`INTEGER_RANGE_TOO_LARGE`, path `variable_families[f]`), and an integer family
needs version `1.3` like any template. A family always covers the full
product of its sets; a generated variable that nothing uses is
[left out](#unused-generated-variables-are-left-out).

#### `objective.linear_term_templates`

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `for_each` | array of string | no | `[]` | The indices to repeat the term over, outermost first, each `"index in set"`, at most eight. Without it the term is generated once. |
| `where` | array of string | no | `[]` | Conditions, at most eight, all of which must hold for a term to be generated. |
| `coefficient` | number \| string | **yes** | — | A JSON number, or a parameter reference such as `"cost[i]"`. |
| `variable` | string | **yes** | — | A family reference with one bound index per family index, e.g. `"x[i,p+1]"`, or the name of an explicit variable. |

#### `objective.quadratic_term_templates`

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `for_each` | array of string | no | `[]` | As for a linear term template. |
| `where` | array of string | no | `[]` | As for a linear term template. |
| `coefficient` | number \| string | **yes** | — | A JSON number, or a parameter reference. |
| `variable1` | string | **yes** | — | A family reference or the name of an explicit variable. |
| `variable2` | string | **yes** | — | A family reference or the name of an explicit variable. |

`variable1` and `variable2` written as the same reference on a binary family or
variable (`x[i,p]` twice) is `SELF_QUADRATIC_TERM`, reported once for the
template. Two references that only coincide for some bindings (`x[i,p]` and
`x[j,p]` without `where: ["i != j"]`) are reported by the validator for each
generated term it concerns.

#### `constraint_templates`

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `id` | string | **yes** | — | A name, unique among the templates and the explicit constraint ids. A generated constraint is named `id[e1,…]` by its `for_each` elements, or `id` itself without `for_each`. |
| `description` | string \| null | no | `null` | Copied to every generated constraint. |
| `type` | `"hard"` \| `"soft"` | **yes** | — | As for a constraint. |
| `for_each` | array of string | no | `[]` | The indices to repeat the constraint over; without it one constraint is generated. |
| `where` | array of string | no | `[]` | Conditions on the constraint's own indices. |
| `terms` | array of linear term template | **yes** | — | The left-hand side. Each entry is a `linear_term_templates` entry that contributes its terms for each of its own bindings, which may use the constraint's indices. |
| `operator` | `"=="` \| `"<="` \| `">="` | **yes** | — | As for a constraint. |
| `rhs` | number \| string | **yes** | — | A JSON number, or a parameter reference. |
| `weight` | number \| string \| null | soft only | `null` | A positive JSON number or a parameter reference; a hard template carries none. |

#### `cardinality_constraint_templates`

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `id` | string | **yes** | — | As for a constraint template. |
| `description` | string \| null | no | `null` | Copied to every generated constraint. |
| `type` | `"hard"` \| `"soft"` | **yes** | — | As for a constraint. |
| `for_each` | array of string | no | `[]` | As for a constraint template. |
| `where` | array of string | no | `[]` | As for a constraint template. |
| `variables` | array of string \| member | **yes** | — | The counted variables. A member `{"for_each": […], "where": […], "variable": "x[i,p]"}` repeats its variable over its own bindings (`for_each` and `where` optional); a plain string is a single reference, short for `{"variable": "…"}`. |
| `operator` | `"=="` \| `"<="` \| `">="` | **yes** | — | As for a cardinality constraint. |
| `rhs` | integer \| string | **yes** | — | An integer within ±(2³¹ − 1), or a reference to a parameter whose values are all whole numbers. |
| `weight` | number \| string \| null | soft only | `null` | As for a constraint template. |

A member must reference a binary family or a binary explicit variable
(`CARDINALITY_VARIABLE_NOT_BINARY` otherwise). An error in a member's
`variable` has the path `cardinality_constraint_templates[t].variables[m]`;
one in its `for_each` or `where` points at `…variables[m].for_each[k]` or
`…variables[m].where[k]`.

A hard template with a `weight`, a soft one without a positive literal
`weight`, and an inequality template whose literal coefficient or `rhs` is not
an integer are reported once, at the template, with the codes a constraint
gets (`HARD_CONSTRAINT_HAS_WEIGHT`, `SOFT_CONSTRAINT_MISSING_WEIGHT`,
`NON_INTEGER_INEQUALITY`). The same rules applied to a value taken from a
parameter are checked on each generated constraint.

### The grammar

```text
for_each item   index in set
reference       name    or    name[index, index, …]
index           a bound index name, optionally shifted: p+1, d-2
where condition operand CMP operand
operand         a bound index name | a parameter reference without shifts | a number
CMP             ==   !=   <   <=   >   >=
```

- Everything is ASCII: names are identifiers as above, a shift is a whole
  number of one to nine digits without leading zeros, a number is a JSON
  number (`3`, `-2`, `0.5`, `1e3`). Spaces may separate the tokens; `in` needs
  at least one on each side. Each string is at most 256 characters (longer is
  a schema error). Any other character is a syntax error whose message gives
  its position.
- Nothing is evaluated: no arithmetic, no function call, no code. A string
  that looks like one (`__import__('os')`) is simply a syntax error.
- `for_each`, `where` and `indices` hold at most eight entries each.
- **`in` is reserved**: it cannot name an index or an index set.
- **References.** A family reference gives one bound index per index set of
  the family, each bound to that very set (`x[i,p]` for `x` over `city, pos`).
  A parameter reference does the same for the parameter's `indices`. A name
  without brackets must be a declared explicit variable (`UNKNOWN_VARIABLE`
  otherwise); a family or parameter name without brackets is an error. An
  explicit variable whose name is not an identifier (`a-b`) cannot be
  referenced from a template.
- **Shifts** are allowed only on an index over a `linear` or `cyclic` set, and
  never inside a `where` condition.
- **No literal elements.** `x[a,0]` in a template, or `p == 0` in a `where`,
  is refused: bind an index with `for_each` and filter with a 0/1 parameter —
  `where: ["is_first[p] == 1"]`. The error message suggests exactly that.
- **`coefficient`, `rhs` and `weight`** take a JSON number as a literal, or a
  string that is a parameter reference. A number written as a string (`"2"`)
  is refused with `TEMPLATE_REFERENCE_INVALID`: write it as a JSON number.
- **Conditions.** Two indices must range over the same set: `==` and `!=`
  compare the elements, `<`, `<=`, `>` and `>=` their positions in `elements`
  (so `i < j` lists each unordered pair once, whatever the set's `order`). A
  number or a parameter value compares as a number, exactly, with no
  tolerance — it filters data, it does not judge a solution. Comparing an
  index with a number is an error.

Every one of these, and every unknown or mismatched name, is
`TEMPLATE_REFERENCE_INVALID` at the field that holds the string (for example
`objective.quadratic_term_templates[0].variable2` or
`constraint_templates[1].where[0]`), with a message that says what was
expected:

```text
[TEMPLATE_REFERENCE_INVALID] objective.quadratic_term_templates[0].variable2: Index p is shifted, but index set pos has no order; declare its order "linear" or "cyclic" to allow shifts
```

#### Scopes and indices

- A template's `for_each` binds the outer indices, first item outermost. The
  terms of a constraint template and the members of a cardinality template
  are inner scopes: their own `for_each` may use the outer indices but not
  bind the same name again. Two sibling terms or members may each bind the
  same name.
- An index name must differ from every index set, parameter and family name,
  and from the other indices in scope (`DUPLICATE_TEMPLATE_NAME`).
- **Every bound index must be used**: in a reference — a `variable`,
  `coefficient`, `rhs` or `weight` — of its scope or an inner one, or in the
  `where` of an inner scope. An index that only appears in its own scope's
  `where` would repeat the same entry once per element (an objective term
  counted several times, or identical constraints), so it is refused
  (`TEMPLATE_REFERENCE_INVALID` at its `for_each` item). Filtering an inner
  list by an outer index is fine: an outer `for_each: ["g in groups", "t in
  slots"]` with a member `{"for_each": ["e in exams"], "where":
  ["in_group[g,e] == 1"], "variable": "x[e,t]"}` counts the exams of group `g`
  placed in slot `t`, although `g` appears in no reference.

### How templates expand

**Order of evaluation.** For an objective template, each binding goes through:

1. the `where` conditions, left to right, stopping at the first that fails;
2. the shifts of its references: on a `linear` set, one that runs past either
   end leaves this term out;
3. the `coefficient` lookup;
4. the generated term.

For a constraint template, each binding of the outer indices goes through:

1. the outer `where` conditions, the same way;
2. the shifts in `rhs` and `weight`: past the end, the whole constraint is
   left out;
3. each term or member template in order, each binding of its own indices:
   its `where`, then the shifts of its references — the **first** that runs
   past the end leaves the **whole constraint** out, and nothing more of it is
   looked at;
4. the lookups of `rhs`, `weight` and every term coefficient, then the
   generated constraint.

A parameter value a `where` condition needs counts as *used* as soon as the
condition is evaluated, before any shift is checked: it must exist (or the
parameter have a default), or the binding is reported with
`PARAMETER_VALUE_MISSING`. A `coefficient`, `rhs` or `weight` is only looked
up for an entry that is actually generated, so an entry left out at a boundary
needs no row.

**Boundaries.** On a `cyclic` set a shift wraps around modulo the set's size.
On a `linear` set an objective term with an out-of-range reference is left
out on its own, but a constraint is left out **entirely**. Leaving out only
the member that fell off the end would quietly change the feasible set:
`x[d] + x[d+1] >= 1` would become `x[last] >= 1` on the last day and force it
to 1, and `x[p+1] - x[p] >= 0` would become `-x[last] >= 0` and force it to
0. Leaving the whole constraint out relaxes the rule at the boundary instead,
so it is always reported: one `TEMPLATE_BOUNDARY_SKIPPED` warning per
template, with the count. Declare the set `cyclic` if the rule should wrap
around.

**Names.** A generated variable is `family[e1,e2,…]`, its elements in the
order of the family's `indices`, separated by commas **without spaces**. A
generated constraint is `id[e1,…]`, its elements in the order of the
template's outer `for_each`, or plain `id` without `for_each`. Names and
elements cannot contain brackets or commas, so different combinations always
give different names. An explicit entry may use a generated name, but must
spell it exactly as generated: `x[a,0]`, not `x[a, 0]` — the
`UNKNOWN_VARIABLE` message points that out. (Inside a template string, spaces
between tokens are fine.) A generated name that equals an explicit one is
`DUPLICATE_VARIABLE` or `DUPLICATE_CONSTRAINT_ID`, as for two explicit
entries.

**Order.** Every expanded list holds the explicit entries first, then each
template's entries in the order of the template list; within a template, in
binding order (first `for_each` item outermost); within a generated
constraint, the terms or members in the order of their templates, each in its
own binding order. All of it comes from arrays — `for_each`, `elements` and
the lists themselves. A parameter table is only looked up, never iterated, so
the order of its rows does not matter. The same document always expands to
the same problem.

**Repeats and empty results.**

- A generated linear constraint that names the same variable more than once
  (typically a cyclic shift that wraps onto itself) keeps every term, and the
  compiler sums the coefficients, as for an explicit constraint; one
  `TEMPLATE_TERMS_MERGED` warning per template gives the count and the first
  example. An explicit constraint never gets this warning.
- Repeated objective terms get the usual `DUPLICATE_TERM_MERGED`, attributed
  to the template.
- A variable listed twice in one generated cardinality constraint is
  `DUPLICATE_CARDINALITY_VARIABLE`, at the member template.
- A generated constraint left with no terms or members (all filtered out by
  `where`) is `EMPTY_CONSTRAINT` — never silently dropped, since dropping an
  empty `== 1` would turn an infeasible problem into a feasible one.
- A template that generates nothing at all gets the
  `EMPTY_TEMPLATE_EXPANSION` warning.

### Unused generated variables are left out

A family generates the full product of its sets, and a generated variable no
objective term and no constraint references would be free in the model: a
solver could set it either way, and a reader could mistake its value for a
decision — a forbidden assignment "taken", say. So after expanding, every
generated variable that nothing references is **left out** of the problem, and
one `UNUSED_TEMPLATE_VARIABLES` warning per family gives how many and the
first five names. A generated variable with the same name as an explicit one
is kept, so the clash is still reported. Explicit variables are never left out,
used or not, and get no warning.

The consequences for results:

- Solutions carry only the variables that were kept, not every key of the
  family's product.
- For every assignment of the kept variables, feasibility, the objective and
  the soft score are exactly what they would be with the unused variables
  declared.
- The returned solutions and the order among equally ranked ones can still
  differ from a document that declares those variables explicitly: samples
  that differ only in a free variable are distinct solutions there and one
  solution here, ties are broken on the full assignment, and a smaller model
  changes a seeded sampler's path. `sample_count` and the denominators of
  `hard_violation_rates` change with it. Usually the result is better: no
  `top_k` slot and no share of `exact`'s variable limit is spent on a free
  variable.

When every generated variable is left out and nothing else is declared, the
problem fails with `NO_VARIABLES`, whose message then says why.

### Errors point at the template

Validation reports two kinds of errors for a document with templates.

- **Expansion errors** — a malformed name, index set, parameter row, family
  or template string, a missing parameter value, the size ceiling — point
  into the template itself: `constraint_templates[1].terms[0].variable`,
  `…for_each[2]`, `…where[0]`, `parameters[0].values[3].key[1]`. Every one
  found is reported at once. With any of them nothing further is validated
  and no warning is reported.
- **Errors and warnings on generated entries**, found by the ordinary
  validator on the expanded problem, are mapped back: `variables[k]` becomes
  `variable_families[f]`, a generated term the term template, a generated
  constraint `constraint_templates[t]` or `cardinality_constraint_templates[t]`
  (its terms, members, `rhs` and `weight` their template's), and the message
  ends with ` (generated by <template path>)`, plus the outer binding for a
  constraint:

  ```text
  [SOFT_CONSTRAINT_MISSING_WEIGHT] constraint_templates[0].weight: Soft constraint pref[i0] requires a weight > 0, got 0.0 (generated by constraint_templates[0] with i=i0)
  ```

  An issue on an explicit entry keeps its own path.

A template with a systematic mistake would otherwise produce one error per
generated entry, so the lists are bounded. Expansion errors stop at 20 per
source (one index set, parameter or top-level template) and 100 in all;
validator errors on generated entries stop at 20 per template. The last one
listed then ends with `; 10 more errors from constraint_templates[0] are not
listed (SOFT_CONSTRAINT_MISSING_WEIGHT ×10)`: every code is still named, with
its count. A warning on generated entries is reported once per code and
template path, ending with `; 6 generated entries from this template get
this warning` when it applies to more than one. As for any problem, warnings
(the expansion's own first) are only reported when there is no error at all.

### Size ceiling and concurrency

A template of a few hundred bytes can ask for hundreds of thousands of
entries, so expansion is bounded by the server setting
`ANNEALBRIDGE_MAX_TEMPLATE_BINDINGS` (default `250000`, reported as
`max_template_bindings` in every backend's capabilities `limits`). It is
checked twice, against the same number:

- **Before anything is generated**, the number of bindings the templates would
  iterate, U: for each family the product of its set sizes; for each
  objective template the product of its `for_each` set sizes; for each
  constraint template the product of its outer sets times one plus the sum,
  over its term or member templates, of the product of their own sets. It is
  computed without iterating, so an enormous request is refused at once.
- **While generating**, the work W: one unit per generated variable,
  objective term and constraint and per term or member of a generated
  constraint, plus, for each generated constraint, one unit per *pair* of its
  terms or members — a constraint over *k* variables costs *k*(*k* − 1)/2 on
  top. W catches the single huge constraint that a small U can still hide, and
  expansion stops the moment it passes the ceiling (a constraint is charged
  while its members are collected, so one later left out at a boundary keeps
  its charge).

Either way the answer is the single error `TEMPLATE_EXPANSION_LIMIT` at the
template being expanded — `resource_limit_exceeded` from `solve`,
`valid: false` from `validate` and `recommend` — and nothing is generated or
kept: a document is expanded in full or not at all, never truncated. The
static checks come first: a document with a malformed template gets those
errors, not the ceiling.

Expanding and validating a document with templates also takes a slot of its
own, one of as many as `ANNEALBRIDGE_MAX_CONCURRENT_SOLVES` allows, in
`validate`, `recommend` and `solve` alike. When every slot is taken the answer
is `CONCURRENCY_LIMIT` (retryable): `resource_limit_exceeded` from `solve`,
`valid: false` from the other two. A document without templates never takes
this slot. Expansion runs inside a solve's [wall-clock limit](#wall-clock-limit)
and its `elapsed_ms`, and cannot be interrupted. A library caller of
`annealbridge.validation.expand_problem`, `validate_problem` or
`validate_problem_full` is not gated; those take the ceiling as the keyword
`max_template_bindings`, `250000` unless given.

### Seeing the expanded problem

`annealbridge expand problem.json` prints the expanded document as JSON (see
[CLI](cli.md#expand)); `annealbridge.validation.expand_problem(problem)`
returns it to a Python caller, together with the expansion's errors and
warnings. Results always speak in generated names: a solution's variables are
`x[a,0]` … and a `constraint_evaluations` entry's `constraint_id` is
`city_once[a]` (see [Output format](output-format.md)).

### Not supported

- Scalar parameters: write the number itself.
- Sparse families: a family always covers the full product of its sets; the
  variables nothing uses are left out afterwards.
- Literal elements inside a template string: bind an index and filter with a
  0/1 parameter.
- Arithmetic of any kind, beyond shifting an index by a constant.
- Referencing from a template an explicit variable whose name is not an
  identifier.

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
| `postprocess` | `"none"` \| `"repair_local_search"` | `"none"` | Opt-in post-processing of each attempt's samples in the original variables: repair the infeasible ones, then move the feasible ones to a local optimum. Runs locally; ignored on an exhaustive backend. See [Post-processing](#post-processing). |
| `postprocess_candidates` | integer > 0 | `10` | How many distinct samples per attempt post-processing starts from, best first. Bounded by policy (`POSTPROCESS_LIMIT`) while `postprocess` is on; ignored while it is `"none"`. |
| `penalty_multiplier` | finite number > 0 | `2.0` | Multiplier applied to `penalty_scale` for the first attempt. |
| `wall_clock_limit_seconds` | finite number > 0 \| null | `null` | Upper bound on the solve's wall-clock time on this server, in seconds; `null` sets no limit. A ceiling that stops the solve early, not a budget to spend. Only accepted on a backend whose capabilities declare `supports_interrupt`. See [Wall-clock limit](#wall-clock-limit). |
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
applies no hard penalty, `max_retries` on an exhaustive backend such as
`exact`, where a retry can never surface new samples, `postprocess` on an
exhaustive backend, or `postprocess_candidates` while `postprocess` is
`"none"`). Filling in an option block that does not belong to the selected
backend raises the same warning. Only a non-default value triggers it.

### Post-processing

`"postprocess": "repair_local_search"` adds one step to every attempt, after
the samples are decoded and re-validated and before they are ranked. It works
on the business variables only — never on slack bits, encoding bits or the
hard penalty — and always runs on this machine, so on a remote backend it
spends CPU time here and no vendor quota.

1. **Select.** The attempt's distinct samples are ordered by their total hard
   violation (`Σ violation_amount` over the hard constraints, the same total
   the [infeasibility diagnostics](output-format.md#closestcandidate) use),
   then by ranking cost, then by the order the solver returned them. The first
   `postprocess_candidates` are taken.
2. **Repair** each selected infeasible sample by greedy descent: every step
   applies the move that most reduces the total hard violation, until the
   assignment is feasible. A repair that finds no move that reduces it, or
   reaches its step cap, is abandoned; the original sample stays a candidate.
3. **Local search** from each selected feasible sample and each successful
   repair: every step applies the move that most improves the ranking cost
   among the moves that keep every hard constraint satisfied, until no move
   improves it (a local optimum) or the step cap is reached. Every step stays
   feasible, so the point reached is kept either way.

A move changes one variable by ±1 inside its bounds (a flip, for a binary
variable), or two variables by ±1 each when both appear with a non-zero
coefficient in the same hard constraint, linear or cardinality — which covers
a swap inside a one-hot group and trading one item for another in a knapsack. A problem
without hard constraints only gets single-variable moves. Each repair and each
local search takes at most `4 · n` steps, `n` being the number of variables.
The search is deterministic — no random numbers, fixed tie-breaking, ceilings
that count evaluations rather than time — so with the same solver samples it
always produces the same assignments.

Every assignment it produces joins that attempt's candidates and is
re-validated and ranked exactly like a solver sample; the solver's energy
plays no part. A produced assignment is marked by
[`Solution.source`](output-format.md#solution), with `energy: null` and
`sample_count: 0`; one the solver also returned stays `source: "solver"`.
Rank 1 may therefore be an assignment the solver never returned. What the step
did in each attempt is in [`attempts[].postprocess`](output-format.md#postprocessstats).

- **Off by default.** With `"none"` nothing runs and every solution carries
  `source: "solver"`. A non-default `postprocess_candidates` then raises
  `PARAMETER_IGNORED` and is not checked against any ceiling.
- **Ignored on an exhaustive backend.** `exact` already returns every
  assignment, so there is nothing to repair or improve: `solver.postprocess`
  raises `PARAMETER_IGNORED` (and `postprocess_candidates` no separate
  warning), `attempts[].postprocess` stays `null`, and no ceiling is checked.
  The decision follows the backend's `exhaustive` capability, not its name.
- **A successful repair ends the retries.** With post-processing on, an
  attempt counts as feasible when a repair produced a feasible assignment even
  though none of the solver's samples was feasible, so no retry with a doubled
  penalty follows. With it off, retries behave exactly as before. Solutions
  always come from a single attempt; attempts are never merged.
- **Ceilings.** `postprocess_candidates` above
  `ANNEALBRIDGE_MAX_POSTPROCESS_CANDIDATES` (default `100`) is refused with
  `resource_limit_exceeded` / `POSTPROCESS_LIMIT` before anything runs. Each
  scan of the moves around one assignment is charged the elementary
  evaluations it performs, `2·(n + z) + 4·(p + s)`: `n` variables, `z`
  non-zero constraint coefficients, `p` variable pairs that share a hard
  constraint and `s` the sum over constraints of `k·(k−1)/2` for a constraint
  over `k` variables (the rows a pair move re-evaluates). Each attempt is
  also charged once for setting the pair moves up: for every such pair, the
  constraint entries of both variables. Both are charged against a
  per-attempt budget of `ANNEALBRIDGE_MAX_POSTPROCESS_EVALUATIONS` (default
  `20000000`, a few seconds of CPU), so the budget bounds time and scratch
  memory alike, dense constraints included. A problem whose setup plus a
  single scan already exceeds the budget is
  refused the same way before solving. Both refusals come from `solve`, and
  `recommend` lists them as blocking; `validate` does not check them. A budget
  that runs out part-way stops post-processing without failing the solve:
  what was finished is kept, a repair still in progress is dropped, a local
  search in progress keeps its current (feasible) point, and the
  `POSTPROCESS_LIMIT_REACHED` warning says so. The step cap is reported the
  same way (whenever a repair or local search uses all its steps, even if the
  last one happened to reach a local optimum). The budget applies to each attempt separately, so with retries the
  total can reach the number of attempts times the budget.
- **The wall-clock limit.** With
  [`wall_clock_limit_seconds`](#wall-clock-limit) set, post-processing is also
  checked before every start and every step, and stops when the limit has run
  out, keeping what was finished the same way as when the budget runs out. That
  stop is reported through `wall_clock_limit_reached` and
  `WALL_CLOCK_LIMIT_REACHED`, never in `limit_reached` or as
  `POSTPROCESS_LIMIT_REACHED`, and it is the one case in which post-processing
  depends on time.

Known limitations:

- An integer variable only ever moves by ±1 per step. With a wide range, the
  `4 · n` step cap can end a repair before it reaches feasibility, or a local
  search before it reaches a local optimum.
- Permutation constraints, as in a TSP (every city once, every position once),
  need at least four variables changed together to go from one feasible
  assignment to another. No move changes more than two, so local search
  cannot improve a feasible tour at all and only spends time.
- A local optimum is not a proof: `optimality_proven` stays `false`.

### Wall-clock limit

`"wall_clock_limit_seconds": 30` caps how long the server spends on the solve.
It is measured from the moment the service starts processing the request —
the same starting point as [`elapsed_ms`](output-format.md#solveresult) — and
covers every attempt and post-processing. It says *at most* how long, never
*how long*: a solve that finishes sooner is not held back, and one that
reaches the limit stops at its next checkpoint instead of running on.

When the limit runs out with work left, the solve returns what it has
completed: every sample it received is decoded, re-validated against the
original problem and ranked exactly as usual, and the status follows the
result — `success` when one of them is feasible, `infeasible` (with a message
saying none was found within the limit) when not. The result carries
`wall_clock_limit_reached: true` and a `WALL_CLOCK_LIMIT_REACHED` warning; see
[Output format](output-format.md#wall-clock-limit-and-reproducibility). It is
not an error, and it is only reported when work was actually left undone.

- **Validation.** The value must be finite and greater than zero; anything
  else is rejected with `INVALID_SOLVER_PREFERENCE`. On a backend whose
  capabilities do not declare `supports_interrupt` — currently `exact`,
  `dwave_qpu`, `leap_hybrid_bqm`, `leap_hybrid_cqm` and `fujitsu_da` — any
  value is rejected with `WALL_CLOCK_LIMIT_UNSUPPORTED` (`invalid_problem`)
  before anything runs, never silently ignored; `validate` and `recommend`
  report the same error. The supporting backends are `simulated_annealing`,
  `tabu` and `simulated_bifurcation`.
- **Every schema version.** It is a solver preference, so `"1.0"`, `"1.1"`,
  `"1.2"` and `"1.3"` accept it alike.
- **No server-side ceiling.** There is no policy key for it: a larger value
  only makes the limit less likely to fire.
- **It can be overrun.** Some stages cannot be interrupted and add their full
  duration on top of the limit: expanding a document's
  [templates](#templates-version-13), validating the problem, compiling it, decoding
  and re-validating the samples, the infeasibility diagnostics and setting up
  post-processing. Each backend also finishes the unit of work it is in when
  the limit runs out — a read, a shard of 25 reads for `tabu`, one integration
  step. The first attempt always starts even if the limit has already run out
  by then; its backend then returns no samples. The per-backend checkpoints and
  worst-case overruns are listed in
  [Backends](backends.md#wall-clock-limits-and-cancellation).
- **Reproducibility.** A solve the limit does not cut short returns exactly
  what it would return without the limit, bit for bit. A solve it does cut
  short depends on timing — machine speed and load, the worker counts, BLAS
  threads, the GPU — so the same seed can give a different result.

It is a different field from the `time_limit_seconds` of the remote option
blocks below:

| | `solver.wall_clock_limit_seconds` | `solver.<block>.time_limit_seconds` |
| --- | --- | --- |
| Where | top level of `solver` | inside a remote backend's option block |
| What it is | a ceiling: stop early when reached | a run-time budget handed to the vendor, which spends all of it |
| Measured | on this server, from the start of the solve | by the vendor, for its own run |
| Backends | those declaring `supports_interrupt` (local heuristics) | `leap_hybrid_bqm`, `leap_hybrid_cqm`, `fujitsu_da` |
| Server ceiling | none | `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` (`REMOTE_TIME_LIMIT`) |

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

Seven ready-to-run problems ship with the repository, and inside the
installed package as well: `annealbridge example` lists them and
`annealbridge example <name>` prints one (see [CLI](cli.md#example)), and the
MCP server serves the same files as `annealbridge://examples/<name>`.

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
- [examples/shift_scheduling.json](../examples/shift_scheduling.json) — 3
  people × 4 shifts over two days, one binary per person/shift pair,
  minimizing total dislike. Hard constraints: every shift covered by exactly
  one person (`== 1`), nobody on more than 2 shifts (`<= 2`), and nobody on
  Monday night followed by Tuesday day (`<= 1`); one soft preference (Ben
  would rather not work Monday night, weight 3). The global optimum,
  ann=mon_day and tue_day, cal=mon_night, ben=tue_night, has total dislike 7
  with the preference honoured. It compiles to 21 variables on the BQM path,
  close enough to the `exact` limit that validating or solving it on `exact`
  adds an `EXACT_NEAR_LIMIT` warning.
- [examples/exam_timetabling.json](../examples/exam_timetabling.json) — a
  `version: "1.2"` document whose rules are all
  [cardinality constraints](#cardinality-constraints-version-12): 4 exams ×
  3 slots, one binary per exam/slot pair, minimizing total inconvenience.
  Hard constraints: every exam in exactly one slot (`== 1`), at most one of
  math, physics and chemistry per slot and never chemistry beside biology
  (`<= 1`, which the BQM path encodes without slack); one soft preference (at
  most one exam in the small Monday-morning hall, weight 3). The unique
  global optimum, math=mon_am, physics=mon_pm, chemistry=tue_am,
  biology=mon_pm, has total inconvenience 6 with the preference honoured (the
  cheaper timetable with biology on Monday morning scores 4 + 3 = 7). It
  compiles to 13 variables on the BQM path: the 12 binaries plus one slack
  bit for the soft preference.
- [examples/tsp_template.json](../examples/tsp_template.json) — a
  `version: "1.3"` document: the 4-city traveling salesman problem of
  `tsp.json` written with [templates](#templates-version-13) — two index sets
  (`city`, and a cyclic `pos`), a distance table, one variable family and
  three templates instead of 16 variables, 48 terms and 8 constraints. It
  expands to exactly that model, with the variables named `x[a,0]` … `x[d,3]`
  and the constraints `city_once[a]` … `position_once[3]`, and the optimal
  tour length is again 8.

All seven declare a local backend; try `--backend simulated_annealing` to
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
