[← Back to README](../README.md) · [Documentation index](README.md)

# Limitations and scope

Everything here is a deliberate boundary of the current implementation, not a
known defect. Where a limit is enforced, the error or warning code is named so
you can look it up in [Error reference](errors.md).

## Variables

- Only **binary** (`0/1`) and **bounded integer** variables are supported. Real
  variables and unbounded integers are not.
- An integer variable requires `"version": "1.1"` at the top of the problem and
  **both** bounds, each within `±(2³¹ − 1)`.
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
- **`simulated_bifurcation` is weak on small penalty-dominated problems**, the
  shape most of the shipped examples have: it finds their optimum in only 1 to
  5 % of its reads, so it needs a large `num_reads` where the annealer does
  not. It is at its strongest on large dense problems instead. It also holds
  the couplings as a dense `N × N` matrix, so it is capped by
  `ANNEALBRIDGE_SB_MAX_VARIABLES` (`SB_VARIABLE_LIMIT`) rather than by time,
  and its answer is reproducible on one machine but may differ across CPUs or
  BLAS builds — see [Backends](backends.md#simulated_bifurcation).
- **`exact`** is a testing and debugging backend, and a ground-truth benchmark
  for the annealer. The state space doubles with every variable, so it is
  limited to small problems.
- **Leap Hybrid** typically returns a single sample per solve, so there is no
  set of runners-up to rank.
- **QPU embedding may fail** for dense problems; the result then carries
  `EMBEDDING_FAILED`.

Per-backend detail lives in [Backends](backends.md).

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
