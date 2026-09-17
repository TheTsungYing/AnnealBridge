[← Back to README](../README.md) · [Documentation index](README.md)

# MCP server

This page covers the `annealbridge-mcp` server: how to install it, the four
tools it exposes, how to configure Claude Desktop and other MCP hosts, the
streamable-http transport, the MCP Inspector, and when an agent should call
each tool.

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

With [uv](https://docs.astral.sh/uv/) there is nothing to install by hand:
`uvx` resolves the package with its extra into an isolated, cached environment
and runs the console script from there. This is the form the host
configurations below use.

```bash
uvx --from "annealbridge[mcp]" annealbridge-mcp
```

The first `uvx` start has to resolve and download the dependency tree — numpy,
dimod, dwave-samplers and the rest — which can take tens of seconds. MCP hosts
put a timeout on starting a server, so that first start can be reported as
disconnected and look like a broken install. Run it once in a terminal before
pointing a host at it:

```bash
uvx --from "annealbridge[mcp]" annealbridge-mcp --version
```

That builds the environment and prints the version; every later start the host
makes reuses the cache and is quick.

`pipx install "annealbridge[mcp]"` is the equivalent that puts
`annealbridge-mcp` on the `PATH` permanently.

The same server is also reachable as a subcommand of the CLI:

```bash
annealbridge mcp
```

The two entry points are interchangeable — every argument, `--help` and
`--version` included, reaches the same parser, and a missing `[mcp]` extra
ends both the same way. The subcommand exists for the
[MCP Registry](#mcp-registry), whose command layout cannot express a console
script whose name differs from the PyPI project.

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
before the transport starts: an invalid value — or settings the service
refuses to be wired from, such as a registered backend declaring a limit key
the policy has no value for — is reported on stderr as one `Error: ...` line
and ends the process with exit code `2` rather than a traceback, so a
misconfigured server fails at launch instead of on the first tool call. A
missing `[mcp]` extra ends the same way, with the same exit code. Logs go to
stderr at `INFO` level, one line per record in the form
`<asctime> <LEVEL> <logger name>: <message>` — the unknown-variable `WARNING`
included — which keeps stdout clean for the stdio protocol. That format is
set up by the entry point, so it applies to `annealbridge-mcp` and
`python -m annealbridge.interfaces.mcp.server`; a use that bypasses it
(`mcp dev`, an embedding host, an in-memory `Client(mcp)`) keeps the MCP
SDK's own log handler and format.

Used without the entry point, the server builds its service on the first
tool call instead of at launch. An invalid setting then cannot end the
process: that call returns a tool error,
`Error executing tool <name>: <settings message>`, naming the variable but
never its value, the server logs it at `INFO` without a traceback,
and the next tool call tries again.

### Upgrading

`uvx` resolves the newest version on its first run and reuses that cached
environment afterwards, so a host configured as above keeps starting the
version it fetched the first time: a later release on PyPI does not reach it
on its own. Drop the cached environment and restart the host to move forward.

```bash
uv cache clean annealbridge
```

`uv`'s `--refresh` flag forces the same re-resolution on a single run. Putting
it in a host configuration re-resolves on every launch, which costs a network
round trip each time the host starts the server and fails when the machine is
offline, so it suits a one-off check rather than a permanent entry.

A pip or pipx install upgrades the usual way, with
`pip install --upgrade "annealbridge[mcp]"` or `pipx upgrade annealbridge`.

`annealbridge-mcp --version` prints the version that would actually run, and
every `solve_optimization` result carries the same string as
`annealbridge_version` — the two are the fastest way to tell a stale cached
environment from a current one.

### MCP Registry

The server is published to the [MCP Registry](https://registry.modelcontextprotocol.io)
as `io.github.TheTsungYing/annealbridge`, described by `server.json` in the
repository root. A registry client resolves that entry to:

```bash
uvx --from=annealbridge[mcp] annealbridge mcp
```

which is why the CLI carries an `mcp` subcommand: the registry composes a
package command as `<runtimeHint> <runtimeArguments> <identifier>
<packageArguments>`, and `identifier` must be the PyPI project name
(`annealbridge`) because the registry proves ownership by looking for an
`mcp-name: io.github.TheTsungYing/annealbridge` line in the package
description PyPI renders from `README.md`. That line must therefore survive in
the README, and a release that drops it breaks the next registry publish.

`server.json` pins a concrete version in two places — the server version and
the package version — and both must equal the `pyproject.toml` version of the
release being published. The release workflow checks all three against the tag
before it builds anything, then publishes to the registry over GitHub OIDC
once the new version is visible on PyPI, so the two stay in step without a
stored credential.

## Tools

| Tool | Input | Returns |
| --- | --- | --- |
| `get_optimization_capabilities` | `include_schema` (optional, default `false`) | `OptimizationCapabilities` |
| `validate_optimization_problem` | `problem` | `ProblemValidationResult` |
| `recommend_backend` | `problem` | `BackendRecommendationResult` |
| `solve_optimization` | `problem` | `SolveResult` |

`problem` is an `OptimizationProblem` document — see
[Problem format](problem-format.md). Every tool returns a structured result
derived from its return type, so a host gets a typed output schema rather than
free text; the shapes are described in [Output format](output-format.md).
Every field of those schemas carries a description, so an agent reading a
result does not have to guess what a field means — that `energy` is for
debugging only, or that `feasible_samples` counts deduplicated candidates.

The three tools that do real work run the synchronous, CPU-bound core in a
worker thread, so one large problem cannot freeze the event loop and with it
every other client of the server.

### Server instructions

At initialize the server hands the host a short instructions text alongside
its name and version (the installed package version). It carries what the
per-tool descriptions cannot: the everyday requests the server is for — which
items to take within a budget or capacity, how to assign people or jobs to
seats, shifts or machines, in which order to visit a handful of places, how to
split things into groups or pick a subset meeting several requirements at
once — so an agent reaches for it when the user never says "optimization";
when each tool is worth a call — on a local backend the agent may go straight
to `solve_optimization`, since an invalid document comes back as
`invalid_problem` carrying the same errors and recommended actions
`validate_optimization_problem` would give, while a remote backend or a large
problem is validated first and `get_optimization_capabilities` is for when the
backend list with its limits or the full schema is actually needed, the schema
only on request; a **choosing a backend** section; the rules a first document
most often breaks
(schema version `1.1` for integer variables, integer coefficients on
inequalities, unknown fields being rejected); and one complete minimal problem,
so an agent learns the document shape before its first call rather than from
its first error. The field descriptions in `problem_json_schema` (returned when
`get_optimization_capabilities` is called with `include_schema: true`) serve
the same purpose one level down. Neither names a configuration value or a
limit; those come from `get_optimization_capabilities`.

The backend section is the one piece of guidance that is about the agent's
own behaviour rather than the document: a backend the user named is used as
given and never substituted; otherwise the agent calls `recommend_backend`
and reads the reason codes — an exhaustive backend that fits proves
optimality, `R_DENSE_STRENGTH` marks a local heuristic that reaches the same
energy faster on this problem's shape and `R_PENALTY_WEAKNESS` one with a
lower hit rate on it, local backends are free while remote ones spend quota.
When more than one local
backend is usable and the user did not ask to be left out of the loop, the
agent is told to present the top entries with one-line reasons and **ask**
which to run, rather than to pick silently; either way the answer says which
backend ran and why.

A `problem` argument carrying a field the schema does not declare — at any
level — is refused as a tool error naming the path, never dropped. See
[Unknown fields are rejected](problem-format.md#unknown-fields-are-rejected).

### `get_optimization_capabilities`

Describes what this server accepts and which backends are usable right now.

- **Input:** `include_schema`, an optional boolean that defaults to `false`.
  The problem JSON schema is roughly three quarters of the full response, so
  it is left out unless it is asked for: the backend list an agent usually
  wants is a fraction of the size. Pass `true` when the schema itself is
  needed — an unfamiliar field, the exact shape of an integer variable, or
  anything the example in the server instructions does not cover.
- **Returns:** `OptimizationCapabilities` — the supported variable types,
  constraint operators and objective terms, the accepted schema versions,
  whether inequalities require integer coefficients, the installed package
  version (`annealbridge_version`), `problem_json_schema` (only when
  `include_schema` is `true`, otherwise `null`), and one entry per backend
  with `available`, `enabled`, `unavailable_reason`, its flags, the
  `seed_min` / `seed_max` range it accepts for `solver.seed` (`null` when it
  declares none) and its resource limits.
- **When to call:** when the agent needs the list of backends with their
  limits, or the full problem schema. Not before every problem — a small
  binary problem on a local backend can follow the example in the server
  instructions. Pass `include_schema: true` when the document needs more
  than the example in the server instructions shows. It performs no solving
  and no network requests.

`schema_version` is the newest accepted version and `schema_versions` lists
them all, newest last; `supported_variable_types` lists `binary` and
`integer`. All three are derived from the pydantic model rather than
hard-coded, as is the embedded JSON schema. Integer variables are only allowed
when the problem carries `"version": "1.1"` at its top level.

Each `backends[].name` is the **registry key** — the value to put in
`solver.backend`, and the one `ANNEALBRIDGE_ENABLED_BACKENDS` is matched
against. `solver.backend` is a closed schema that accepts only the eight
built-in names, so a backend has to be registered under its own
`capabilities.name` for a request to be able to name it at all. `enabled`
already accounts for server policy, so a backend reported as not enabled will
not become usable by asking for it.

A backend whose own availability check fails — it raises, or reports a
category the server does not know — does not fail the call: that backend is
listed with `available: false` and the redacted failure as its
`unavailable_reason`, and every other backend is listed as usual.

### `validate_optimization_problem`

Checks a problem without solving it.

- **Input:** `problem`.
- **Returns:** `ProblemValidationResult` — `valid`, all semantic errors (each
  with a `recommended_action`), advisory warnings, the estimated compiled
  variable count, the objective scale, and the model type the chosen backend
  would compile to.
- **When to call:** before `solve_optimization` when the target is a remote
  backend or the problem is large (many variables, wide integer ranges) —
  problems get fixed before quota is spent. On a local backend the agent may
  solve directly: an `invalid_problem` result carries the same errors and
  recommended actions. Nothing is compiled or solved and no network requests
  are made.

Validation collects **all** errors in one pass, so one round trip is enough to
fix a problem. That pass includes the one check that depends on the named
backend: a `solver.seed` outside the `seed_min` / `seed_max` range that
backend declares is an `INVALID_SOLVER_PREFERENCE` error. The size estimate
follows the model type: on a bqm backend it counts the slack bits of every
inequality plus the binary-encoding bits of
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

Two further reason codes order the **local heuristics among themselves**, from
a structural preference each backend declares and that was measured, not
guessed. Both are matched only on the **bqm path**: a backend that compiles to
cqm never carries either, whatever it declares.

`R_DENSE_STRENGTH` says the backend declares that on a large dense model with
no effective hard constraint it reaches the same energy as its peers in a
fraction of the time, and that this problem is one: at least 500 estimated
compiled variables (slack and integer encoding bits included), a density of at
least `0.5`, and no effective hard constraint anywhere in the document.
Density is the number of distinct variable pairs the model will couple divided
by `m(m−1)/2` for the `m` declared variables, counting a pair once however
many objective terms or constraints — hard or soft — contribute it.

`R_PENALTY_WEAKNESS` says the backend declares a lower hit rate on models
whose hard constraints compile to penalties, and that this problem is one: the
bqm path with at least one *effective* hard constraint, each of which becomes
a penalty term that dwarfs the objective. A hard constraint is effective when
it has a non-zero coefficient and is not redundant over the declared bounds —
an `x <= 1` on a binary variable compiles to no penalty and so does not count.

The codes sort *within* a tier — a `structure_fit` key applied after the
existing exhaustive / local / remote tiers and before registry order — so they
reorder peers and never make an unusable backend usable, never move a local
backend ahead of a fitting `exact`, and never touch the remote backends. In
practice, within the local heuristic tier (a fitting `exact` still ranks
first): a large dense problem with no effective hard constraint ranks `tabu`,
`simulated_bifurcation`, `simulated_annealing`; a problem with an effective
hard constraint (the shipped knapsack example) ranks `simulated_annealing`,
`tabu`, `simulated_bifurcation`; a small unconstrained one matches neither
shape and keeps registry order, `simulated_annealing`, `tabu`,
`simulated_bifurcation`.

The rule is conservative, and an agent should know where. A large dense
problem that carries one effective hard constraint is ranked
`simulated_annealing`, `tabu`, `simulated_bifurcation` even at a size where
simulated bifurcation might finish far sooner, because the speed advantage was
measured only on unconstrained instances. Faced with that combination, tell
the user the ranking is playing safe, or ask which trade they want, instead of
presenting the top entry as the only option. See
[Backends](backends.md#how-recommend-orders-the-local-heuristics) for the
measurements the thresholds come from.

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

Add the server to `claude_desktop_config.json`. With uv installed this is the
whole configuration: `uvx` fetches the package the first time the host starts
the server and reuses its cache afterwards — run the `--version` warm-up from
[Installation](#installation) once first, so that fetch does not happen inside
the host's start-up timeout. Configuration goes through `env`:

```json
{
  "mcpServers": {
    "annealbridge": {
      "command": "uvx",
      "args": ["--from", "annealbridge[mcp]", "annealbridge-mcp"],
      "env": {
        "ANNEALBRIDGE_ALLOW_REMOTE": "false"
      }
    }
  }
}
```

If you installed the package with pip instead, `"command"` is the
`annealbridge-mcp` executable. A virtual environment is not on the host's
`PATH`, so use the absolute path:

```json
{
  "mcpServers": {
    "annealbridge": {
      "command": "/path/to/venv/bin/annealbridge-mcp"
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

Claude Code registers the server from the command line:

```bash
claude mcp add annealbridge -- uvx --from "annealbridge[mcp]" annealbridge-mcp
```

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
| `--version` | — | Print `annealbridge <version>` and exit `0` without starting a transport. Read before any setting, so it works even when an `ANNEALBRIDGE_*` value is invalid |

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
against the package module:

```bash
pip install -e ".[dev]"          # or: pip install "mcp[cli]>=2,<3"
mcp dev src/annealbridge/interfaces/mcp/__init__.py
```

Point it at `__init__.py`, not `server.py`. `mcp dev` loads its target by file
path, which builds a *second* module object with its own server instance that
the tools were never registered on — the Inspector would then list no tools
at all, and a `:mcp` suffix does not change that. The package module imports
the canonical server and its tools, so loading it by path reaches the one
instance the four tools live on.

Alternatively, drive the installed entry point directly with the Node-based
Inspector:

```bash
npx @modelcontextprotocol/inspector annealbridge-mcp
```

Either way the Inspector lists the four tools, shows their generated
input/output schemas, and lets you submit a problem JSON by hand — the fastest
way to see what an agent will see.

## Typical agent workflow

Each of the four tools has its moment rather than a fixed place in a queue.
The full sequence below is what a careful run looks like; the fast path after
it is what a simple problem takes.

1. **`get_optimization_capabilities`** — when the agent needs the list of
   backends with their limits, or the full schema (`include_schema: true`).
   It tells it which variable types and operators it may use, which schema
   version to declare, which
   backends are `available` *and* `enabled` on this server, and what limits
   they enforce. It is worth the round trip when the document goes beyond the
   shape the server instructions already show, or when a backend's
   availability or limits matter before formulating — a large problem aimed at
   a backend that is switched off wastes the whole formulation.
2. **`validate_optimization_problem`** — before spending anything on a remote
   backend, and whenever the problem is large. It returns *all* semantic
   errors at once, each with a recommended action, plus the estimated compiled
   size. This is where a malformed constraint or an integer bound that
   explodes into hundreds of encoding bits gets caught, at zero cost.
3. **`recommend_backend`** — whenever the user did not name a backend. It
   ranks every backend for this specific problem with deterministic reason
   codes and the blocking errors that make one unusable, so the agent picks on
   evidence rather than on a name. It is advisory: the agent still writes the
   backend it wants into `solver.backend`, and a backend the user *did* name
   is used as given. When several local backends are usable and the user did
   not ask for an answer without being consulted, the server instructions tell
   the agent to show the top entries with their reasons and ask which to run
   instead of deciding alone.
4. **`solve_optimization`** — last, whether or not the steps above were taken.
   Its result carries the ranked solutions and, on failure, a structured error
   with a recommended action. Its `warnings` are the same ones step 2 gives for
   that backend, so an agent that skipped step 2 still sees them.

The fast path is shorter. For a small binary problem on a local backend, the
agent writes the document and calls `solve_optimization` directly — with
`recommend_backend` first when the user named no backend, and, when more than
one local backend is usable, still asking which to run as the server
instructions say. An invalid document does not cost a solve: it comes back as
`invalid_problem` with the same errors and `recommended_action`
`validate_optimization_problem` would have given, and the corrected document
is sent again. On a host that asks the user to approve every tool call, that
turns a simple problem from four prompts into one or two.

Steps 1–3 perform no solving, no network requests and consume no vendor quota,
so an agent can iterate freely, and a local solve is free as well — only a
step 4 on a remote backend spends anything. If step 4 returns `infeasible` on
a remote backend, the usual next move is to re-read the per-attempt data in
the result rather than to blindly raise `penalty_multiplier` — the server
manages penalties itself.
