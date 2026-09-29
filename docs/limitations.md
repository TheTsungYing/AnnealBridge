[← Back to README](../README.md) · [Documentation index](README.md)

# Limitations and scope

Everything here is a deliberate boundary of the current implementation, not a
known defect. Where a limit is enforced, the error or warning code is named so
you can look it up in [Error reference](errors.md).

## Variables

- Only **binary** (`0/1`) and **bounded integer** variables are supported. Real
  variables and unbounded integers are not.
- An integer variable requires `"version": "1.1"` or later at the top of the
  problem and **both** bounds, each within `±(2³¹ − 1)`.
- On a BQM backend an integer costs
  `(upper_bound − lower_bound).bit_length()` encoding bits. A variable that
  needs more than 10 bits raises the `LARGE_INTEGER_RANGE` warning, and a
  problem whose encoding yields more than 2000 quadratic interactions raises
  `INTEGER_QUADRATIC_BLOWUP` — after which `recommend` ranks that backend
  behind every backend that does not carry the warning.
- Only **binary expansion** is implemented for integers. There is no one-hot or
  unary encoding option to choose from.

See [Problem format](problem-format.md) for the full rules on declaring
variables.

## Constraints

- Inequality constraints (`<=` / `>=`) must have **integer coefficients and an
  integer right-hand side** (`NON_INTEGER_INEQUALITY`). The JSON number may be
  written `6` or `6.0`, but it must be mathematically integral. No automatic
  scaling of fractional coefficients is performed. Equality constraints are not
  subject to this.
- The restriction applies to integer variables as well: their bounds are
  integers, so an integer coefficient keeps the slack range exact.
- An inequality must also stay inside the exactly representable float64 integer
  range: `Σ |coefficient| × max(|lower_bound|, |upper_bound|) + |rhs|` must be
  at most `2⁵³` (`INEQUALITY_MAGNITUDE_TOO_LARGE`; a binary variable's bound
  counts as 1). The slack range is computed from those products, and a rounded
  product could encode the constraint wrongly — rescale the units or tighten
  the bounds.
- The **CQM path keeps the same integer-coefficient requirement** even though
  it uses no slack variables at all. This is deliberately conservative, so both
  compilation paths accept exactly the same set of problems.
- **Cardinality constraints** (version `1.2`) count **binary** variables only,
  each listed once (`CARDINALITY_VARIABLE_NOT_BINARY`,
  `DUPLICATE_CARDINALITY_VARIABLE`); a weighted count or a sum over integer
  variables stays a linear constraint. Only a **hard at-most-one** gets the
  slack-free pairwise encoding: a soft one, an at-most-*k* with *k* ≥ 2 and
  an at-least-*k* keep the slack bits of the equivalent linear constraint (a
  quadratic model cannot express at-most-*k* without them, and a pairwise soft
  cost would under-charge the violation). A linear `<= 1` in `constraints` is
  never switched to the pairwise encoding automatically — that would change how
  a `1.0` document compiles — it only gets the `CARDINALITY_FORM_AVAILABLE`
  hint. On the CQM path a one-hot group is a plain linear equality: it is not
  marked as a discrete group for the solver.

## Templates (version 1.3)

- **A size ceiling, not a streaming expansion.** A document with templates is
  expanded in full before anything else happens, bounded by
  `ANNEALBRIDGE_MAX_TEMPLATE_BINDINGS` (default `250000`; about a 49-city TSP
  written as templates). Above it the document is refused whole with
  `TEMPLATE_EXPANSION_LIMIT` — never truncated or partly expanded — and the
  same model can then only be sent written out. Expanding and validating such
  documents also shares a limited number of slots
  (`ANNEALBRIDGE_MAX_CONCURRENT_SOLVES`); a full server answers the retryable
  `CONCURRENCY_LIMIT`.
- **The work count is an interaction bound for binary members only.** The
  ceiling charges every pair of terms or members of a generated constraint,
  which is exactly the interactions a binary constraint adds on the BQM path.
  An integer member is encoded into several bits and an inequality adds slack
  bits, so the real interaction count can be larger by roughly the square of
  the bits per variable — the same risk an explicit document already carries,
  flagged by `INTEGER_QUADRATIC_BLOWUP`. In the other direction the CQM path
  forms no pairs at all, so the count is conservative there and may refuse a
  template document the CQM path could handle; the same model written out is
  not subject to it.
- **A `linear` set relaxes a constraint template at its boundary.** A
  generated constraint whose shifted index runs past the end of a `linear`
  set is left out entirely, so a hard rule simply does not apply there; the
  only signal is the `TEMPLATE_BOUNDARY_SKIPPED` warning. If the rule should
  wrap around, declare the set `cyclic`.
- **What follows expansion is not bounded by it.** The response holds up to
  `top_k` solutions over every variable, each with an evaluation per
  constraint, and a heuristic's run time grows with the variables times
  `num_reads` and `num_sweeps`. A template document reaches those sizes in
  under 1 KB where the written-out form needed tens of megabytes; the read,
  sweep and `top_k` ceilings still apply as before.
- **Error lists are capped.** At most 20 errors are listed per template (20
  per source and 100 in all for the expansion's own errors); the rest are
  counted by code in the last one's message, so no code is hidden, but their
  details are not shown.
- **Not supported:** scalar parameters (write the number), sparse families (a
  family covers the full product of its sets; members nothing uses are left
  out afterwards, with `UNUSED_TEMPLATE_VARIABLES`), literal elements in a
  template string (use a 0/1 parameter), any arithmetic beyond shifting an
  index by a constant, and referencing from a template an explicit variable
  whose name is not an identifier (`a-b`). There is no MCP tool that returns
  the expanded document; use `annealbridge expand` or
  `annealbridge.validation.expand_problem`.

See [Templates](problem-format.md#templates-version-13) for the full rules.

## Soft constraints

- Soft constraint weights are expressed in **objective units** and are **not
  normalized**. If your objective coefficients run in the thousands, a weight
  of `5` has almost no influence. Size weights against the `objective_scale`
  the compile step reports.

## Backends

- **`simulated_annealing`, `tabu` and `simulated_bifurcation`** are heuristic.
  None of them guarantees a global optimum, and an `infeasible` result from any
  of them only means "not found under this configuration"
  (`infeasibility_proven: false`). `tabu` also takes no `num_sweeps` and no
  time limit: the only dial on its search effort is `num_reads`.
- **Stopping early is coarse on some backends.** `solver.wall_clock_limit_seconds`
  and cancellation are only supported by the three local heuristics, and a
  `tabu` shard of 25 reads cannot be stopped once it has started — seconds on
  a large problem. `exact` and the remote backends refuse a wall-clock limit,
  and a cancellation waits for their running call; a remote job already
  submitted is not cancelled on the vendor side and still consumes quota.
  Ctrl+C in the CLI does not stop a running shard early. See
  [Backends](backends.md#wall-clock-limits-and-cancellation).
- **`simulated_bifurcation` is weak on small penalty-dominated problems**, the
  shape most of the shipped examples have: it finds their optimum in only 1 to
  5 % of its reads, so it needs a large `num_reads` where the annealer does
  not. It is at its strongest on large dense problems instead. It also holds
  the couplings as a dense `N × N` matrix, so it is capped by
  `ANNEALBRIDGE_SB_MAX_VARIABLES` (`SB_VARIABLE_LIMIT`) rather than by time,
  and its answer is reproducible on one machine but may differ across CPUs or
  BLAS builds — see [Backends](backends.md#simulated_bifurcation).
- **`simulated_bifurcation` can collapse on packing problems with at-most-one
  groups.** With the pairwise encoding of hard cardinality at-most-ones, its
  samples on a weighted set-packing problem (maximize) fell to nearly all
  zeros, and the cause is not known; on other problems the same encoding
  helped it. An all-zero assignment satisfies every at-most-one, so the
  result is feasible but poor, never wrong. For such problems use
  `simulated_annealing` or `tabu`, or turn on `solver.postprocess:
  "repair_local_search"`; see
  [Problem format](problem-format.md#simulated_bifurcation-and-at-most-one-groups).
- **`exact`** is a testing and debugging backend, and a ground-truth benchmark
  for the annealer. The state space doubles with every variable, so it is
  limited to small problems.
- **Leap Hybrid** typically returns a single sample per solve, so there is no
  set of runners-up to rank.
- **QPU embedding may fail** for dense problems; the result then carries
  `EMBEDDING_FAILED`.

Per-backend detail lives in [Backends](backends.md).

## Model size

- **No interaction ceiling for binary problems.** A constraint over *n*
  variables — linear or cardinality, hard or soft — couples all of them
  pairwise in the BQM: `n(n−1)/2` quadratic interactions, which for a hard
  cardinality at-most-one are exactly its penalty pairs. The
  `INTEGER_QUADRATIC_BLOWUP` warning only fires for a problem with integer
  variables, so an all-binary problem with a large dense group raises no size
  warning and meets no interaction limit of its own; only the variable limits
  of `exact` and `simulated_bifurcation` and the vendors' own limits apply.
  `annealbridge validate` reports the compiled variable count, not the
  interaction count.
- **Backend recommendation enumerates coupled pairs.** To decide whether a
  backend's declared strength on large dense models applies, `recommend`
  lists every pair of variables a constraint couples — `O(n²)` time and
  memory for a constraint over *n* variables — once the problem is large
  enough and has no effective hard constraint. A single soft constraint over
  tens of thousands of variables makes `recommend` itself slow; `validate`
  and `solve` do not run this step.

## Fujitsu Digital Annealer

- The backend speaks the **QUBO API V4 with an API key only**: no OAuth access
  token, no V3c endpoints.
- **None of the annealer's native features are used** — not its inequality
  support, not one-hot, not penalty polynomials. The entire compiled QUBO
  (objective, hard-constraint penalties, slack and integer-encoding bits) is
  submitted as a single binary polynomial, exactly like every other BQM
  backend.
- The vendor's own limits — 100,000 bits per problem, at most 16 pending jobs
  per account, and the account's monthly quota — surface as
  `REMOTE_SOLVER_ERROR`, `REMOTE_BUSY` and `REMOTE_QUOTA_EXCEEDED` rather than
  being pre-checked locally.
- The vendor's `frequency` field is **not** expanded into duplicate samples:
  each returned solution becomes exactly one candidate row.

## Out of scope

Not planned for now:

- real (continuous) variables;
- unbounded integers;
- alternative integer encodings (one-hot, unary);
- nonlinear constraints;
- automatic soft-weight normalization.
