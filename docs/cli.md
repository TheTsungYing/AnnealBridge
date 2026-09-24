[← Back to README](../README.md) · [Documentation index](README.md)

# Command-line interface

This page documents the `annealbridge` command: its seven subcommands, their
options and output, and the exit codes a script can rely on. The CLI is
installed by the core package — no extra required, though `mcp` needs the
`[mcp]` extra to do anything.

It is a presentation layer only. It parses arguments, loads the problem JSON,
calls the service, and formats the result; no optimization logic lives in it.

The seven commands, as `annealbridge --help` describes them (the real output
is rendered in Rich panels; only the text is shown here):

```text
  solve          Solve an optimization problem loaded from a JSON file.
  validate       Validate an optimization problem without solving it.
  recommend      Rank the backends for a problem without solving it.
  capabilities   List backends with availability, policy status and limits.
  example        List the shipped example problems, or print one as JSON.
  export-schema  Print the OptimizationProblem JSON schema.
  mcp            Run the MCP server (needs the "mcp" extra; same options as
                 annealbridge-mcp).
```

They all read the `ANNEALBRIDGE_*` environment
([Configuration](configuration.md)) except `example` and `export-schema`,
which need no configuration at all.

The `examples/…` paths in the commands below are files in a repository
checkout. An installed package carries the same documents:
`annealbridge example knapsack > knapsack.json` saves one locally, and
`annealbridge example knapsack | annealbridge solve -` runs it without a file
(see [`example`](#example) and [The problem file](#the-problem-file)).

## Global options

```bash
annealbridge --version
```

| Option | Meaning |
| --- | --- |
| `--version` | Print `annealbridge <version>` (the installed package version) and exit `0` |
| `--help` | Show the command list; after a command, that command's options |

`--version` is answered before any command runs, so it reads no
`ANNEALBRIDGE_*` setting and works even when one of them holds an invalid
value.

## Output encoding

Machine-readable output — `--json` on `solve`, `validate`, `recommend` and
`capabilities`, and `export-schema` — is always UTF-8 with `\n` newlines,
whether stdout is a terminal, a file or a pipe, and on Windows too: it does
not follow the console code page or `PYTHONIOENCODING`, so
`annealbridge capabilities --json > caps.json` can be read back as UTF-8
anywhere. `example NAME` prints the file's own UTF-8 bytes. The tables,
reports and the example list are for people and, like the messages on
stderr, use the terminal's encoding.

## The problem file

`solve`, `validate` and `recommend` take a `PROBLEM_FILE` argument: the path
of an `OptimizationProblem` JSON document, or `-` to read the document from
stdin. Either way it is decoded as UTF-8, with or without a byte-order mark.
Every message about a document read from stdin names it `<stdin>`:

```console
$ annealbridge example knapsack | annealbridge validate -
Problem:   knapsack
Backend:   exact  (model type: bqm)
Valid:     yes
Estimated compiled variables: 8
Objective scale: 31
$ echo 'nojson' | annealbridge validate -
Error: '<stdin>' is not valid JSON: Expecting value: line 1 column 1 (char 0)
```

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
as above, plus a `source:` line when `solver.postprocess` is on and the best
solution came from post-processing rather than from the solver; `infeasible`
prints one line per attempt (penalty, samples received, unique samples,
feasible samples) — followed, when post-processing ran in that attempt, by a
line summarising it — and whether infeasibility was *proven*;
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

Rank  Backend                Usable  Model  Reasons
1     exact                  yes     bqm    R_EXACT_FITS
2     simulated_annealing    yes     bqm    R_LOCAL_HEURISTIC
3     tabu                   yes     bqm    R_LOCAL_HEURISTIC
4     simulated_bifurcation  yes     bqm    R_LOCAL_HEURISTIC, R_PENALTY_WEAKNESS
5     leap_hybrid_cqm        no      cqm    R_UNUSABLE, R_NATIVE_CONSTRAINTS   [REMOTE_DISABLED]
6     dwave_qpu              no      bqm    R_UNUSABLE, R_REMOTE   [REMOTE_DISABLED]
7     leap_hybrid_bqm        no      bqm    R_UNUSABLE, R_REMOTE, R_SINGLE_SAMPLE   [REMOTE_DISABLED]
8     fujitsu_da             no      bqm    R_UNUSABLE, R_REMOTE   [REMOTE_DISABLED]
```

The trailing bracket holds the blocking error codes that make a backend
unusable right now — above, remote execution is simply switched off. `--json`
adds each entry's warnings and its estimated compiled variable count.

For a problem with integer variables the reasons also carry
`R_INTEGER_NATIVE` (a cqm backend), `R_INTEGER_ENCODED` (a bqm backend) or
`R_INTEGER_BLOWUP` (a bqm backend whose encoding triggers the
`INTEGER_QUADRATIC_BLOWUP` warning). Only the last one changes the order, and
only by moving that backend behind the others.

Two further codes reflect a structural preference a backend declares for
itself: `R_DENSE_STRENGTH` on a large dense problem with no effective hard
constraint, and `R_PENALTY_WEAKNESS` on a bqm-path problem that has one — as
the knapsack above does, which is why `simulated_bifurcation` falls to the
back of the local heuristics. Both reorder backends only *inside* the tier
they already share, so neither moves a heuristic past a fitting `exact` or
touches the remote backends. See
[Backends](backends.md#how-recommend-orders-the-local-heuristics).

## `capabilities`

Show which backends are installed, permitted and under what limits.

```bash
annealbridge capabilities [--json]
```

| Option | Meaning |
| --- | --- |
| `--json` | Print the capabilities as JSON instead of the table (see below) |

It performs **no network I/O**, so it is safe to run before any credential is
configured — availability is judged from the installed extras and the
environment alone.

The listing below is from a core install without the `dwave` extra. With
dwave-system installed but no credentials configured, the three D-Wave rows
read `(D-Wave credentials not configured)` instead.

```console
$ annealbridge capabilities
Backend                Available  Enabled  Remote  Limits
exact                  yes        yes      no      max_variables=24, max_local_retries=10, max_top_k=1000
simulated_annealing    yes        yes      no      max_local_reads=100000, max_sweeps=100000, max_local_retries=10, max_top_k=1000, max_postprocess_candidates=100, max_postprocess_evaluations=20000000
tabu                   yes        yes      no      max_local_reads=100000, max_local_retries=10, max_top_k=1000, max_postprocess_candidates=100, max_postprocess_evaluations=20000000
simulated_bifurcation  yes        yes      no      max_variables=10000, max_local_reads=100000, max_sweeps=100000, max_local_retries=10, max_top_k=1000, max_postprocess_candidates=100, max_postprocess_evaluations=20000000
dwave_qpu              no         no       yes     max_reads=1000, max_annealing_time_us=2000, max_remote_retries=3, max_top_k=1000, max_postprocess_candidates=100, max_postprocess_evaluations=20000000  (dwave-system not installed)
leap_hybrid_bqm        no         no       yes     max_time=300s, max_remote_retries=3, max_top_k=1000, max_postprocess_candidates=100, max_postprocess_evaluations=20000000  (dwave-system not installed)
leap_hybrid_cqm        no         no       yes     max_time=300s, max_remote_retries=3, max_top_k=1000, max_postprocess_candidates=100, max_postprocess_evaluations=20000000  (dwave-system not installed)
fujitsu_da             no         no       yes     max_time=300s, max_remote_retries=3, max_top_k=1000, max_postprocess_candidates=100, max_postprocess_evaluations=20000000  (Fujitsu DA API key not configured)
```

- **Available** — the backend can run: its optional dependency is installed
  and its credentials are configured. The parenthesised text is the reason it
  cannot, in categorical form that never contains a configuration value. A
  backend whose availability check itself fails shows `no` with that failure
  as its reason; the other rows are listed as usual and the command still
  exits `0`.
- **Enabled** — server policy permits it: it is in
  `ANNEALBRIDGE_ENABLED_BACKENDS` (or the list is unset) *and*, for a remote
  backend, `ANNEALBRIDGE_ALLOW_REMOTE` is true. A remote backend the policy
  will not call is reported as not enabled, so nothing steers you into a
  guaranteed `REMOTE_DISABLED`.
- **Limits** — the ceilings this backend's requests are checked against,
  resolved from the current `ANNEALBRIDGE_*` environment. These are the same
  numbers `get_optimization_capabilities` reports to an agent. The two
  post-processing ceilings are absent from `exact`: post-processing never runs
  on an exhaustive backend.

Run it after each setup step in [Backends](backends.md#d-wave-setup) to
confirm the change took effect.

`--json` prints the structure the MCP tool `get_optimization_capabilities`
returns by default (`include_schema: false`), field for field: the schema
versions, the supported variable types, constraint operators and objective
terms, whether inequalities need integer coefficients, one entry per backend
(availability, policy, seeding, limits and a description), and the package
version. `problem_json_schema` is always `null` here; `export-schema` prints
the schema on its own. Abridged:

```console
$ annealbridge capabilities --json
{
  "schema_version": "1.1",
  "schema_versions": [
    "1.0",
    "1.1"
  ],
  "supported_variable_types": [
    "binary",
    "integer"
  ],
  "supported_constraint_operators": [
    "==",
    "<=",
    ">="
  ],
  "supported_objective_terms": [
    "linear",
    "quadratic"
  ],
  "inequality_requires_integer_coefficients": true,
  "backends": [
    {
      "name": "exact",
      "available": true,
      "enabled": true,
      "unavailable_reason": null,
      "remote": false,
      "heuristic": false,
      "exhaustive": true,
      "supports_seed": false,
      "seed_min": null,
      "seed_max": null,
      "returns_multiple_samples": true,
      "limits": {
        "max_variables": 24,
        "max_local_retries": 10,
        "max_top_k": 1000
      },
      "description": "Local exhaustive solver enumerating every assignment; proves optimality and infeasibility but only suits small problems."
    },
    ...
  ],
  "problem_json_schema": null,
  "annealbridge_version": "0.3.0"
}
```

Without `--json` the output is the table above. The fields are described under
[`get_optimization_capabilities`](mcp.md#get_optimization_capabilities).

## `example`

List the example problems shipped inside the package, or print one.

```bash
annealbridge example [NAME]
```

No options, no configuration read, no network I/O. Without a name it lists
the examples, one line each:

```console
$ annealbridge example
knapsack          0/1 knapsack
integer_knapsack  bounded integer knapsack with a soft constraint
assignment        assignment (workers to tasks)
tsp               travelling salesman over four cities
shift_scheduling  shift scheduling (people to shifts)
```

With a name it prints that example's JSON to stdout exactly as the file holds
it, byte for byte, so it can be saved or piped straight into another command:

```bash
annealbridge example knapsack > problem.json
annealbridge example shift_scheduling | annealbridge solve -
```

An unknown name lists the available ones on stderr and exits `2`:

```console
$ annealbridge example nope
Error: unknown example 'nope'. Available: knapsack, integer_knapsack, assignment, tsp, shift_scheduling
```

The files are package data (`annealbridge/interfaces/examples/`), so the
command works on a core install without any extra. They are the same files
the MCP server serves as the resources `annealbridge://examples/<name>`, and
byte-for-byte copies of the repository's `examples/` directory; see
[Bundled examples](problem-format.md#bundled-examples).

## `export-schema`

Print the `OptimizationProblem` JSON Schema to stdout.

```bash
annealbridge export-schema
annealbridge export-schema > problem.schema.json
```

No options, no configuration read, no network I/O. The output is the complete
JSON Schema of the problem document — the same schema an agent can use for
structured output, and the same one `get_optimization_capabilities` returns
when called with `include_schema: true`. It is generated from the pydantic
models, so it can never drift from what the server actually accepts. See
[Problem format](problem-format.md) for the human-readable description.

## `mcp`

Run the MCP server, exactly as the `annealbridge-mcp` console script does.

```bash
annealbridge mcp [--transport {stdio,streamable-http}] [--host HOST] [--port PORT]
```

Every argument is handed to the server's own parser, `--help` and `--version`
included, so this subcommand and `annealbridge-mcp` answer identically; the
options are documented in [the MCP server page](mcp.md). Without the `[mcp]`
extra it prints the same one-line install hint on stderr and exits `2`.

It exists because the MCP Registry composes a package command as
`<runtimeHint> <runtimeArguments> <identifier> <packageArguments>`, and
`identifier` must be the PyPI project name (`annealbridge`) for the registry's
ownership check. A registry client therefore runs
`uvx --from=annealbridge[mcp] annealbridge mcp`. Use whichever entry point
suits you; a hand-written host configuration is shorter with
`annealbridge-mcp`.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success: `solve` returned `status: "success"`, `validate` found the problem valid, `recommend` produced a ranking, `capabilities` / `example` / `export-schema` completed, or `--version` printed the version |
| `1` | A domain answer that is not success: a non-`success` `SolveResult` (including `infeasible`), or an invalid problem from `validate` / `recommend` |
| `2` | The command could not run at all |

Exit `2` covers four distinct situations, and all four print to stderr:

- **The input file (or stdin) cannot be used** — unreadable, not valid JSON,
  or not a valid `OptimizationProblem` document:

  ```console
  $ annealbridge solve nope.json
  Error: cannot read 'nope.json': [Errno 2] No such file or directory: 'nope.json'
  ```

  A document that does not fit the schema lists every schema error at once,
  each as `[CODE] path: message` with the same `UNKNOWN_FIELD`,
  `MISSING_FIELD` and `INVALID_FIELD_VALUE` codes the MCP tools return, and
  its recommended action (see [Schema errors](errors.md#schema-errors)).
  The message never echoes the submitted value, and with `--json` stdout
  stays empty:

  ```console
  $ annealbridge validate broken.json
  Error: 'broken.json' is not a valid optimization problem.

  Schema errors (3):
    [MISSING_FIELD] objective: Field required
      recommended action: A field the problem schema requires is absent; add it at the reported path (...)
    [INVALID_FIELD_VALUE] constraints[0].operator: Input should be '==', '<=' or '>='
      recommended action: The value at the reported path has the wrong type or is not one of the allowed values, ...
    [UNKNOWN_FIELD] foo: unknown field, not in the problem schema
      recommended action: The field is not part of the problem schema and unknown fields are never ignored; ...
  ```

  Semantic errors (an undeclared variable, say) are reported, with exit `1`,
  once the document fits the schema.

- **`--backend` names a backend that does not exist** (`solve` and `validate`
  only):

  ```console
  $ annealbridge solve examples/knapsack.json --backend nope
  Error: unknown backend 'nope' (expected one of 'simulated_annealing', 'exact', 'tabu', 'simulated_bifurcation', 'dwave_qpu', 'leap_hybrid_bqm', 'leap_hybrid_cqm', 'fujitsu_da')
  ```

- **`example` names an example that does not exist** (shown under
  [`example`](#example)).

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

| CLI command | Service call | MCP counterpart |
| --- | --- | --- |
| `solve` | `OptimizationService.solve()` | `solve_optimization` |
| `validate` | `OptimizationService.validate()` | `validate_optimization_problem` |
| `recommend` | `OptimizationService.recommend()` | `recommend_backend` |
| `capabilities` | shared capabilities view | `get_optimization_capabilities` (`--json` is its default output) |
| `example` | shared registry of shipped examples | resources `annealbridge://examples/<name>` |
| `export-schema` | `OptimizationProblem` JSON schema | resource `annealbridge://schema`, or `include_schema: true` |

So the CLI and an agent always see the same answer for the same problem and
the same environment. That makes the CLI the natural way to debug what an
agent is getting: run `annealbridge validate --json` or
`annealbridge solve --json` on the exact problem document the agent sent, and
the result is byte-comparable with the tool's structured output. The
architecture that guarantees this is described in
[Architecture](architecture.md).

A core install without the `[mcp]` extra therefore still reaches what an
agent reads — the examples, the capabilities and the schema. Only the MCP
server has progress notifications during a solve and the three
[prompts](mcp.md#prompts).
