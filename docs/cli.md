[← Back to README](../README.md) · [Documentation index](README.md)

# Command-line interface

This page documents the `annealbridge` command: its five subcommands, their
options and output, and the exit codes a script can rely on. The CLI is
installed by the core package — no extra required.

It is a presentation layer only. It parses arguments, loads the problem JSON,
calls the service, and formats the result; no optimization logic lives in it.

The five commands, as `annealbridge --help` describes them (the real output
is rendered in Rich panels; only the text is shown here):

```text
  solve          Solve an optimization problem loaded from a JSON file.
  validate       Validate an optimization problem without solving it.
  recommend      Rank the backends for a problem without solving it.
  capabilities   List backends with availability, policy status and limits.
  export-schema  Print the OptimizationProblem JSON schema.
```

All five commands read the `ANNEALBRIDGE_*` environment
([Configuration](configuration.md)) except `export-schema`, which needs no
configuration at all.

The `examples/…` paths in the commands below are files in a repository
checkout; the installed package does not ship them. Save any problem JSON
locally and pass its path instead.

## `solve`

Validate, compile, solve, re-validate and rank a problem.

```bash
annealbridge solve PROBLEM_FILE [--backend NAME] [--json]
```

| Option | Meaning |
| --- | --- |
| `--backend NAME` | Override `solver.backend` from the JSON |
| `--json` | Print the full `SolveResult` as JSON instead of the report |

```console
$ annealbridge solve examples/knapsack.json
Problem:   knapsack
Backend:   exact
Status:    success
Attempts:  1
Elapsed:   6.1 ms

Best solution (rank 1)
  objective (maximize):  17
  soft violation score:  0
  item_a = 1
  item_b = 0
  item_c = 1
  item_d = 0

Hard constraints: 1 / 1 satisfied
Soft constraints: 0 violations
Optimality proven: yes
```

`Elapsed` is the service's own wall clock for the whole call and varies from
run to run; `Optimality proven` is `yes` only on an exhaustive backend.

A `version: "1.1"` problem with bounded integer variables reads the same way;
integers come back as plain `int` values inside their declared bounds:

```console
$ annealbridge solve examples/integer_knapsack.json --backend exact
Problem:   integer_knapsack
Backend:   exact
Status:    success
Attempts:  1
Elapsed:   9.7 ms

Best solution (rank 1)
  objective (maximize):  34
  soft violation score:  0
  item_a = 0
  item_b = 1
  item_c = 1
  item_d = 3

Hard constraints: 1 / 1 satisfied
Soft constraints: 0 violations
```

What the report shows depends on `status`. `success` prints the best solution
as above; `infeasible` prints one line per attempt (penalty, samples received,
unique samples, feasible samples) and whether infeasibility was *proven*;
`invalid_problem`, `resource_limit_exceeded`, `backend_unavailable`,
`configuration_error` and `solver_error` print their structured errors as
`[CODE] path: message`, each with its recommended action. Warnings, when there
are any, are appended in the same form — the same warnings `validate` would
give for that backend, plus any raised during the run. With `"seed": 7` in
the solver block of the knapsack example:

```console
$ annealbridge solve examples/knapsack.json --backend exact
...
Warnings (1):
  [SEED_IGNORED] solver.seed: Backend exact does not support seeding; solver.seed will be ignored
    recommended action: This backend does not support seeding; remove solver.seed or use a backend that supports seeding if reproducibility is required.
```

`--json` prints the whole `SolveResult` — every ranked solution, every
per-constraint evaluation, every attempt, and the solver metadata. That is the
form to pipe into another tool; see [Output format](output-format.md).

## `validate`

Check a problem without solving it, and report the size it would compile to.
No compiling, no solving, no network I/O.

```bash
annealbridge validate PROBLEM_FILE [--backend NAME] [--json]
```

| Option | Meaning |
| --- | --- |
| `--backend NAME` | Override `solver.backend` from the JSON |
| `--json` | Print the full `ProblemValidationResult` as JSON |

```console
$ annealbridge validate examples/knapsack.json
Problem:   knapsack
Backend:   exact  (model type: bqm)
Valid:     yes
Estimated compiled variables: 8
Objective scale: 31
```

The estimate follows the model type the chosen backend compiles to, so the
same problem can report different sizes on different backends. On a bqm
backend it counts the slack bits of every inequality plus the binary-encoding
bits of every integer variable — which is why the bounded-integer knapsack
compiles to 15 variables rather than its 4 business variables:

```console
$ annealbridge validate examples/integer_knapsack.json
Problem:   integer_knapsack
Backend:   exact  (model type: bqm)
Valid:     yes
Estimated compiled variables: 15
Objective scale: 96
```

With `--json` the same result is machine-readable:

```console
$ annealbridge validate examples/knapsack.json --json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "estimated_compiled_variables": 8,
  "objective_scale": 31.0,
  "model_type": "bqm"
}
```

An invalid problem prints every error found — validation collects them all in
one pass — and exits `1`.

## `recommend`

Rank every registered backend for a problem, without solving it. Advisory
only, deterministic, no network I/O, no quota consumed.

```bash
annealbridge recommend PROBLEM_FILE [--json]
```

| Option | Meaning |
| --- | --- |
| `--json` | Print the full `BackendRecommendationResult` as JSON |

There is **no `--backend` option**: the command ranks all of them, and it
never rewrites `solver.backend`. Solving still uses the backend you asked for.

```console
$ annealbridge recommend examples/knapsack.json
Problem:   knapsack
Advisory:  recommendations only; `solve` uses solver.backend as given

Rank  Backend              Usable  Model  Reasons
1     exact                yes     bqm    R_EXACT_FITS
2     simulated_annealing  yes     bqm    R_LOCAL_HEURISTIC
3     leap_hybrid_cqm      no      cqm    R_UNUSABLE, R_NATIVE_CONSTRAINTS   [REMOTE_DISABLED]
4     dwave_qpu            no      bqm    R_UNUSABLE, R_REMOTE   [REMOTE_DISABLED]
5     leap_hybrid_bqm      no      bqm    R_UNUSABLE, R_REMOTE, R_SINGLE_SAMPLE   [REMOTE_DISABLED]
6     fujitsu_da           no      bqm    R_UNUSABLE, R_REMOTE   [REMOTE_DISABLED]
```

The trailing bracket holds the blocking error codes that make a backend
unusable right now — above, remote execution is simply switched off. `--json`
adds each entry's warnings and its estimated compiled variable count.

For a problem with integer variables the reasons also carry
`R_INTEGER_NATIVE` (a cqm backend), `R_INTEGER_ENCODED` (a bqm backend) or
`R_INTEGER_BLOWUP` (a bqm backend whose encoding triggers the
`INTEGER_QUADRATIC_BLOWUP` warning). Only the last one changes the order, and
only by moving that backend behind the others.

## `capabilities`

Show which backends are installed, permitted and under what limits.

```bash
annealbridge capabilities
```

No options. It performs **no network I/O**, so it is safe to run before any
credential is configured — availability is judged from the installed extras
and the environment alone.

```console
$ annealbridge capabilities
Backend              Available  Enabled  Remote  Limits
exact                yes        yes      no      max_variables=24, max_local_retries=10, max_top_k=1000
simulated_annealing  yes        yes      no      max_local_reads=100000, max_sweeps=100000, max_local_retries=10, max_top_k=1000
dwave_qpu            no         no       yes     max_reads=1000, max_annealing_time_us=2000, max_remote_retries=3, max_top_k=1000  (dwave-system not installed)
leap_hybrid_bqm      no         no       yes     max_time=300s, max_remote_retries=3, max_top_k=1000  (dwave-system not installed)
leap_hybrid_cqm      no         no       yes     max_time=300s, max_remote_retries=3, max_top_k=1000  (dwave-system not installed)
fujitsu_da           no         no       yes     max_time=300s, max_remote_retries=3, max_top_k=1000  (Fujitsu DA API key not configured)
```

- **Available** — the backend can run: its optional dependency is installed
  and its credentials are configured. The parenthesised text is the reason it
  cannot, in categorical form that never contains a configuration value.
- **Enabled** — server policy permits it: it is in
  `ANNEALBRIDGE_ENABLED_BACKENDS` (or the list is unset) *and*, for a remote
  backend, `ANNEALBRIDGE_ALLOW_REMOTE` is true. A remote backend the policy
  will not call is reported as not enabled, so nothing steers you into a
  guaranteed `REMOTE_DISABLED`.
- **Limits** — the ceilings this backend's requests are checked against,
  resolved from the current `ANNEALBRIDGE_*` environment. These are the same
  numbers `get_optimization_capabilities` reports to an agent.

Run it after each setup step in [Backends](backends.md#d-wave-setup) to
confirm the change took effect.

## `export-schema`

Print the `OptimizationProblem` JSON Schema to stdout.

```bash
annealbridge export-schema
annealbridge export-schema > problem.schema.json
```

No options, no configuration read, no network I/O. The output is the complete
JSON Schema of the problem document — the same schema an agent can use for
structured output, and the same one returned inside
`get_optimization_capabilities`. It is generated from the pydantic models, so
it can never drift from what the server actually accepts. See
[Problem format](problem-format.md) for the human-readable description.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success: `solve` returned `status: "success"`, `validate` found the problem valid, `recommend` produced a ranking, or `capabilities` / `export-schema` completed |
| `1` | A domain answer that is not success: a non-`success` `SolveResult` (including `infeasible`), or an invalid problem from `validate` / `recommend` |
| `2` | The command could not run at all |

Exit `2` covers three distinct situations, and all three print to stderr:

- **The input file cannot be used** — unreadable, not valid JSON, or not a
  valid `OptimizationProblem` document (each schema error is listed):

  ```console
  $ annealbridge solve nope.json
  Error: cannot read 'nope.json': [Errno 2] No such file or directory: 'nope.json'
  ```

- **`--backend` names a backend that does not exist** (`solve` and `validate`
  only):

  ```console
  $ annealbridge solve examples/knapsack.json --backend nope
  Error: unknown backend 'nope' (expected one of 'simulated_annealing', 'exact', 'dwave_qpu', 'leap_hybrid_bqm', 'leap_hybrid_cqm', 'fujitsu_da')
  ```

- **An `ANNEALBRIDGE_*` setting holds an invalid value**, or a registered
  backend declares a limit key the policy has no value for. The message names
  the variable and the reason but never echoes the value. This affects every
  command that builds the service — `solve`, `validate`, `recommend` and
  `capabilities`.

Note the difference between exit `1` and exit `2`: an *infeasible* problem, or
one the validator rejects, is a legitimate answer about your problem and exits
`1`. Exit `2` means the request never reached the optimizer.

## The same code path as MCP

The CLI and the MCP server are two adapters over one core. Both build their
service through the same composition root — settings → policy → registry →
service — and each command is a one-line delegation:

| CLI command | Service call | MCP tool |
| --- | --- | --- |
| `solve` | `OptimizationService.solve()` | `solve_optimization` |
| `validate` | `OptimizationService.validate()` | `validate_optimization_problem` |
| `recommend` | `OptimizationService.recommend()` | `recommend_backend` |
| `capabilities` | shared capabilities view | `get_optimization_capabilities` |

So the CLI and an agent always see the same answer for the same problem and
the same environment. That makes the CLI the natural way to debug what an
agent is getting: run `annealbridge validate --json` or
`annealbridge solve --json` on the exact problem document the agent sent, and
the result is byte-comparable with the tool's structured output. The
architecture that guarantees this is described in
[Architecture](architecture.md).
