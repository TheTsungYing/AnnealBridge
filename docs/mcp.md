[← Back to README](../README.md) · [Documentation index](README.md)

# MCP server

This page covers the `annealbridge-mcp` server: how to install it, the four
tools it exposes, how to configure Claude Desktop and other MCP hosts, the
streamable-http transport, the MCP Inspector, and the order an agent should
call the tools in.

The server is a thin adapter. Each tool fetches the wired-up service, calls
into the core, and returns the core's model — the same code path the
[CLI](cli.md) uses, so an agent and the command line always see the same
answer.

## Installation

The MCP server lives behind an extra:

```bash
pip install "annealbridge[mcp]"
```

That adds the `mcp>=2,<3` dependency. The core install (and the
`annealbridge` CLI) does not depend on `mcp` at all.

The `annealbridge-mcp` command itself is installed either way, extra or not.
Without the extra it does not start: it prints one line on stderr,

```console
$ annealbridge-mcp
Error: the MCP server needs the optional "mcp" dependency; install it with: pip install "annealbridge[mcp]"
```

and ends with exit code `2`, leaving stdout empty — a missing extra is
reported, never a `ModuleNotFoundError` traceback. Install the extra and the
same command works.

Run it with no arguments for stdio, the default transport:

```bash
annealbridge-mcp
```

Configuration comes from the `ANNEALBRIDGE_*` environment
([Configuration](configuration.md)). The settings are read and validated
before the transport starts: an invalid value is reported on stderr and ends
the process with exit code `2` rather than a traceback, so a misconfigured
server fails at launch instead of on the first tool call. A missing `[mcp]`
extra ends the same way, with the same exit code. Logs go to stderr at
`INFO` level, which keeps stdout clean for the stdio protocol.

## Tools

| Tool | Input | Returns |
| --- | --- | --- |
| `get_optimization_capabilities` | none | `OptimizationCapabilities` |
| `validate_optimization_problem` | `problem` | `ProblemValidationResult` |
| `recommend_backend` | `problem` | `BackendRecommendationResult` |
| `solve_optimization` | `problem` | `SolveResult` |

`problem` is an `OptimizationProblem` document — see
[Problem format](problem-format.md). Every tool returns a structured result
derived from its return type, so a host gets a typed output schema rather than
free text; the shapes are described in [Output format](output-format.md).

The three tools that do real work run the synchronous, CPU-bound core in a
worker thread, so one large problem cannot freeze the event loop and with it
every other client of the server.

### Server instructions

At initialize the server hands the host a short instructions text alongside
its name and version (the installed package version). It carries what the
per-tool descriptions cannot: the recommended call order, the rules a first
document most often breaks (schema version `1.1` for integer variables,
integer coefficients on inequalities, unknown fields being rejected), and one
complete minimal problem, so an agent learns the document shape before its
first call rather than from its first error. The field descriptions in
`problem_json_schema` serve the same purpose one level down. Neither names a
configuration value or a limit; those come from
`get_optimization_capabilities`.

A `problem` argument carrying a field the schema does not declare — at any
level — is refused as a tool error naming the path, never dropped. See
[Unknown fields are rejected](problem-format.md#unknown-fields-are-rejected).

### `get_optimization_capabilities`

Describes what this server accepts and which backends are usable right now.

- **Input:** none.
- **Returns:** `OptimizationCapabilities` — the supported variable types,
  constraint operators and objective terms, the accepted schema versions,
  whether inequalities require integer coefficients, the full problem JSON
  schema, and one entry per backend with `available`, `enabled`,
  `unavailable_reason`, its flags and its resource limits.
- **When to call:** before formulating a problem. It performs no solving and
  no network requests.

`schema_version` is the newest accepted version and `schema_versions` lists
them all, newest last; `supported_variable_types` lists `binary` and
`integer`. All three are derived from the pydantic model rather than
hard-coded, as is the embedded JSON schema. Integer variables are only allowed
when the problem carries `"version": "1.1"` at its top level.

Each `backends[].name` is the **registry key** — the value to put in
`solver.backend`, and the one `ANNEALBRIDGE_ENABLED_BACKENDS` is matched
against. `enabled` already accounts for server policy, so a backend reported
as not enabled will not become usable by asking for it.

### `validate_optimization_problem`

Checks a problem without solving it.

- **Input:** `problem`.
- **Returns:** `ProblemValidationResult` — `valid`, all semantic errors (each
  with a `recommended_action`), advisory warnings, the estimated compiled
  variable count, the objective scale, and the model type the chosen backend
  would compile to.
- **When to call:** before `solve_optimization`, especially when the target is
  a remote backend — problems get fixed before quota is spent. Nothing is
  compiled or solved and no network requests are made.

Validation collects **all** errors in one pass, so one round trip is enough to
fix a problem. The size estimate follows the model type: on a bqm backend it
counts the slack bits of every inequality plus the binary-encoding bits of
every integer variable, so wider bounds cost more compiled variables; on a cqm
backend integers are native and no encoding bits are counted.

### `recommend_backend`

Ranks this server's backends for a given problem.

- **Input:** `problem`.
- **Returns:** `BackendRecommendationResult` — one entry per backend with its
  rank, whether it is `usable` right now, the model type it would compile to,
  deterministic reason codes, blocking errors, the warnings
  `validate_optimization_problem` would give for that backend, and the
  estimated compiled variable count.
- **When to call:** when choosing between backends. Deterministic, with no
  network I/O, no solving, no concurrency slot and no quota consumed.

**Advisory only.** It never rewrites `problem.solver.backend`;
`solve_optimization` always uses the backend the caller asked for. The ranking
is based on problem structure, backend capabilities and server policy — there
is no cost estimation.

A problem with integer variables adds reason codes: `R_INTEGER_NATIVE` when
the backend compiles to cqm and takes integers as they are,
`R_INTEGER_ENCODED` when it compiles to bqm and must binary-encode them, and
`R_INTEGER_BLOWUP` when that encoding also raises an
`INTEGER_QUADRATIC_BLOWUP` warning — a backend carrying that code is ranked
after the ones without it.

### `solve_optimization`

Validates, compiles, solves, re-validates and ranks.

- **Input:** `problem`.
- **Returns:** `SolveResult` — ranked feasible solutions with per-constraint
  evaluations and objective values, or a structured error with a
  `recommended_action`. Domain failures are results, not exceptions:
  `status` is one of `success`, `infeasible`, `invalid_problem`,
  `resource_limit_exceeded`, `backend_unavailable`, `configuration_error` or
  `solver_error`.
- **When to call:** only after translating the user's request into explicit
  binary or bounded-integer variables, an objective, and hard or soft linear
  constraints. Natural-language requirements do not belong in the tool input.

Things worth knowing before calling it:

- An integer variable needs `"type": "integer"` with both `lower_bound` and
  `upper_bound`, and the problem must carry `"version": "1.1"`. Integer values
  come back as `int`s inside their declared bounds.
- Inequality constraints (`<=`, `>=`) require integer coefficients and
  right-hand sides. Soft constraint weights are in objective units and are not
  normalized.
- Leave `solver.penalty_multiplier` at its default unless a previous result
  was infeasible on a remote backend; hard-constraint penalties are managed by
  the server.
- Every candidate is re-validated against the *original* problem, never judged
  by the solver's energy or by a sampler's own feasibility flag.
- On an `infeasible` result, read `infeasibility` to learn which candidate came
  closest to feasibility and how often each hard constraint was violated,
  instead of reporting only that nothing was found.
- Whatever the `status`, `warnings` holds the same advisory warnings
  `validate_optimization_problem` gives for that backend (`SEED_IGNORED`,
  `PARAMETER_IGNORED`, `LARGE_INTEGER_RANGE`, `SOFT_WEIGHT_SMALL`, ...),
  followed by any warning raised during the run (`REMOTE_RETRIES_DISABLED`).
  Skipping `validate` therefore never hides them; only an `invalid_problem`
  result carries none. Read them before trusting an answer that looks weaker
  than expected — a wide integer range on a heuristic backend, for example,
  can return a slightly sub-optimal value with `status: success`.

## Claude Desktop (stdio)

Add the server to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "annealbridge": {
      "command": "annealbridge-mcp"
    }
  }
}
```

If `annealbridge-mcp` is installed in a virtual environment that is not on the
host's `PATH` — the usual case — use its absolute path, and pass any
configuration through `env`:

```json
{
  "mcpServers": {
    "annealbridge": {
      "command": "/path/to/venv/bin/annealbridge-mcp",
      "env": {
        "ANNEALBRIDGE_ALLOW_REMOTE": "false"
      }
    }
  }
}
```

On Windows the path is `C:\\path\\to\\venv\\Scripts\\annealbridge-mcp.exe`
(escape the backslashes in JSON). Every value in an `env` block is a string,
including booleans and numbers. Restart the host after editing the file.

## Other MCP hosts

Any MCP host that speaks stdio — Codex, other desktop clients, or your own
client built on an MCP SDK — is configured the same way: run
`annealbridge-mcp` as the server command, with no arguments for stdio, and put
the `ANNEALBRIDGE_*` settings in whatever environment block the host provides.
Only the surrounding configuration file format differs.

## Streamable HTTP

For hosts that connect over HTTP instead of stdio:

```bash
annealbridge-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

| Argument | Default | Notes |
| --- | --- | --- |
| `--transport` | `stdio` | `stdio` or `streamable-http` |
| `--host` | `ANNEALBRIDGE_HTTP_HOST`, i.e. `127.0.0.1` | Non-empty, no whitespace. Ignored for stdio |
| `--port` | `ANNEALBRIDGE_HTTP_PORT`, i.e. `8000` | `1`–`65535` (`0` is refused). Ignored for stdio |

> **Security.** The server has **no authentication or authorization of any
> kind.** It binds `127.0.0.1` by default, and binding beyond localhost must
> be requested explicitly: an empty host is rejected as a configuration error,
> never read as a request to bind every interface. Do not expose it directly
> to a public network or bind it to `0.0.0.0`. If remote access is genuinely
> needed, put it behind an authenticating reverse proxy or on a private
> network. Read [Security](security.md) before changing the bind address.

## MCP Inspector

For interactive development, install a toolchain that ships the MCP CLI —
either the project's `[dev]` extra or `mcp[cli]` — and run the Inspector
against the server module:

```bash
pip install -e ".[dev]"          # or: pip install "mcp[cli]>=2,<3"
mcp dev src/annealbridge/interfaces/mcp/server.py
```

Alternatively, drive the installed entry point directly with the Node-based
Inspector:

```bash
npx @modelcontextprotocol/inspector annealbridge-mcp
```

Either way the Inspector lists the four tools, shows their generated
input/output schemas, and lets you submit a problem JSON by hand — the fastest
way to see what an agent will see.

## Typical agent workflow

The four tools are designed to be called in this order. Each step is cheap and
narrows what the next one has to guess.

1. **`get_optimization_capabilities`** — once, before formulating anything.
   It tells the agent which variable types and operators it may use, which
   schema version to declare, which backends are `available` *and* `enabled`
   on this server, and what limits they enforce. Formulating against a backend
   that is switched off wastes a whole round trip.
2. **`validate_optimization_problem`** — after writing the problem document,
   before spending anything. It returns *all* semantic errors at once, each
   with a recommended action, plus the estimated compiled size. This is where
   a malformed constraint or an integer bound that explodes into hundreds of
   encoding bits gets caught, at zero cost.
3. **`recommend_backend`** — when the choice of backend is not obvious. It
   ranks every backend for this specific problem with deterministic reason
   codes and the blocking errors that make one unusable, so the agent picks on
   evidence rather than on a name. It is advisory: the agent still writes the
   backend it wants into `solver.backend`.
4. **`solve_optimization`** — last, with a problem already known to be valid
   and a backend already known to be usable. Its result carries the ranked
   solutions and, on failure, a structured error with a recommended action.
   Its `warnings` are the same ones step 2 gave for that backend, so an agent
   that skipped step 2 still sees them.

Steps 1–3 perform no solving, no network requests and consume no vendor quota,
so an agent can iterate freely; only step 4 costs anything. If step 4 returns
`infeasible` on a remote backend, the usual next move is to re-read the
per-attempt data in the result rather than to blindly raise
`penalty_multiplier` — the server manages penalties itself.
