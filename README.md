# AnnealBridge

[![CI](https://github.com/TheTsungYing/AnnealBridge/actions/workflows/ci.yml/badge.svg)](https://github.com/TheTsungYing/AnnealBridge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/TheTsungYing/AnnealBridge/blob/main/LICENSE)

English | [繁體中文](https://github.com/TheTsungYing/AnnealBridge/blob/main/README.zh-TW.md)

**Combinatorial optimization middleware for AI agents.** An agent describes
*what* to optimize as structured JSON; AnnealBridge decides *how* to encode
and solve it, checks every answer against the original problem, and returns
ranked, verified solutions over [MCP](https://modelcontextprotocol.io), a CLI,
or plain Python.

## Install

Requires Python 3.11 or newer. For an MCP host, [uv](https://docs.astral.sh/uv/)
is the only prerequisite: add the server to `claude_desktop_config.json` (or
your host's equivalent) and restart the host. `uvx` fetches the package into
its own cached environment the first time the host starts the server.

```json
{
  "mcpServers": {
    "annealbridge": {
      "command": "uvx",
      "args": ["--from", "annealbridge[mcp]", "annealbridge-mcp"]
    }
  }
}
```

Claude Code registers it in one line:

```bash
claude mcp add annealbridge -- uvx --from "annealbridge[mcp]" annealbridge-mcp
```

For the command line or the Python API, install the package into any
environment:

```bash
pip install annealbridge              # core: local backends + CLI
pip install "annealbridge[mcp]"       # + MCP server (annealbridge-mcp)
pip install "annealbridge[dwave]"     # + D-Wave cloud backends
pip install "annealbridge[all]"       # everything
```

`pipx`, a pip-installed server behind an absolute path, and the development
version from a checkout are covered in
[Installation in depth](#installation-in-depth).

## A conversation

Once the server is registered, optimizing is an ordinary chat. The agent
turns the request into a small JSON document; AnnealBridge solves it and
checks every answer against that document before anything comes back.

> **You:** I can carry 10 kg. Item A is worth 10 and weighs 6, B is worth 8
> and weighs 5, C is worth 7 and weighs 4, D is worth 6 and weighs 3. Which
> ones should I take?

Behind the reply, the agent calls three of the server's tools in order:

1. `get_optimization_capabilities` — which variable types and operators it
   may use, which backends are usable right now, and their limits.
2. `validate_optimization_problem` — its draft (four binary variables, a
   maximize objective, one `<= 10` hard constraint) comes back with every
   error at once, or clean. Nothing is solved yet and nothing is spent.
3. `solve_optimization` — ranked solutions, each re-validated against the
   original constraints, with `optimality_proven: true` because the
   exhaustive `exact` backend enumerated every combination.

> **Agent:** Take A and C: value 17 at exactly 10 kg. The runners-up are A
> and D (16, at 9 kg) and B and C (15, at 9 kg). This is the proven optimum;
> every combination was enumerated.

The wording is the agent's; the numbers are the tool result. The fourth tool,
`recommend_backend`, ranks the backends for a given problem and is advisory
only. Any stdio-capable MCP host works the same way, and a streamable-http
transport is available too; see [docs/mcp.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/mcp.md). The document
the agent sent is [the problem JSON at a glance](#the-problem-json-at-a-glance).

## Command line and Python

The same solve from a terminal or a script.

### Command line

Save [the problem JSON below](#the-problem-json-at-a-glance) as
`knapsack.json`, then:

```bash
annealbridge solve knapsack.json
```

```text
Problem:   knapsack
Backend:   exact
Status:    success
Attempts:  1
Elapsed:   2.7 ms

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
run to run.

Add `--json` for the full `SolveResult`, `--backend simulated_annealing` to
override the backend, or try `validate`, `recommend`, `capabilities` and
`export-schema`. See [docs/cli.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/cli.md).

### Python

```python
import json

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

with open("knapsack.json", encoding="utf-8") as f:
    problem = OptimizationProblem.model_validate(json.load(f))

result = OptimizationService().solve(problem)
print(result.status)                          # "success"
print(result.solutions[0].variables)          # {"item_a": 1, "item_b": 0, ...}
print(result.solutions[0].objective_value)    # 17.0
```

Domain failures come back as results, never as exceptions: `result.status`
is one of `success`, `infeasible`, `invalid_problem`,
`resource_limit_exceeded`, `backend_unavailable`, `configuration_error` or
`solver_error`. See [docs/output-format.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/output-format.md).

## The problem JSON at a glance

This is the document behind both examples above: a 0/1 knapsack with capacity
10, the reduced form of the repository's
[examples/knapsack.json](https://github.com/TheTsungYing/AnnealBridge/blob/main/examples/knapsack.json). Save it as `knapsack.json`
anywhere you like.

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

Integer variables (`"type": "integer"` with bounds, `"version": "1.1"`),
quadratic objective terms, soft constraints with weights, and per-backend
solver preferences are described in
[docs/problem-format.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/problem-format.md). `annealbridge export-schema`
prints the JSON Schema an agent can use for structured output.

Four ready-to-run examples live in the repository —
[knapsack](https://github.com/TheTsungYing/AnnealBridge/blob/main/examples/knapsack.json),
[assignment](https://github.com/TheTsungYing/AnnealBridge/blob/main/examples/assignment.json), [TSP](https://github.com/TheTsungYing/AnnealBridge/blob/main/examples/tsp.json) and
[integer knapsack](https://github.com/TheTsungYing/AnnealBridge/blob/main/examples/integer_knapsack.json). The installed wheel does
not ship them; take them from a checkout or from GitHub.

## Installation in depth

The core install needs no `mcp` and no `dwave-system`: the registry still
loads and simply reports the remote backends as unavailable. The Fujitsu
backend needs no extra at all; it talks to the vendor's HTTPS API through the
standard library and only waits for `FUJITSU_DA_API_KEY`.

`uvx` (above) runs the MCP server from an isolated, cached environment on
demand. [pipx](https://pipx.pypa.io/) is the equivalent that puts
`annealbridge-mcp` on your `PATH` permanently:

```bash
pipx install "annealbridge[mcp]"
```

A server installed with pip into a virtual environment is not on the host's
`PATH`: set `"command"` to the absolute path of `annealbridge-mcp` (on
Windows, `Scripts\annealbridge-mcp.exe`) instead of using `uvx`.
Configuration reaches the server through the host's `env` block; see
[docs/mcp.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/mcp.md).

To install the development version from a checkout:

```bash
pip install "annealbridge[all] @ git+https://github.com/TheTsungYing/AnnealBridge.git"
```

## How it works

An agent produces an `OptimizationProblem`: binary or bounded-integer
variables, a linear or quadratic objective, and hard or soft linear
constraints. Nothing else. AnnealBridge then, deterministically:

1. **validates** the problem and collects every error in one pass;
2. **compiles** it into a Binary Quadratic Model (BQM) or a Constrained
   Quadratic Model (CQM), computing penalties, slack and integer encodings
   itself;
3. **solves** it on a local or remote backend;
4. **re-validates** every candidate against the *original* JSON, never
   trusting solver energy;
5. **ranks** the feasible solutions and returns the top *K* with
   per-constraint evaluations.

The agent never writes a QUBO matrix, a penalty weight, a slack variable, or
an integer encoding. The Python core owns all of that, and every step is
testable without an AI, a network, or a vendor account.

Six backends sit behind one protocol: exhaustive `exact` and local
`simulated_annealing` for free, plus D-Wave QPU, Leap hybrid BQM, Leap hybrid
CQM, and the Fujitsu Digital Annealer for remote execution.

```text
  natural language ──► Agent ──► OptimizationProblem JSON
                                 (variables / objective / constraints only)
                                         │
                                         ▼
                  ┌──────────── AnnealBridge ────────────┐
                  │ validate → compile → solve →         │
                  │ re-validate against the ORIGINAL →   │
                  │ rank                                 │
                  └──────────────────┬───────────────────┘
                                     │
                       SolveResult: ranked, verified solutions
                                     │
                                     ▼
                                   Agent ──► natural-language answer
```

## Features

- **Business-level contract in both directions.** Input is variables,
  objective and constraints; output is ranked solutions with objective values
  and per-constraint evaluations. No solver internals leak either way.
- **Two compiler paths.** BQM (automatic hard-constraint penalties, binary
  slack, binary-encoded integers) for annealers, CQM (native constraints and
  integers) for the Leap hybrid CQM solver. The backend chooses the path by
  declaring what it supports.
- **Bounded integer variables** (`"version": "1.1"`) with the encoding hidden
  from the agent; `1.0` problems keep their exact behaviour, pinned by a
  golden test.
- **Structured failures, never exceptions.** Every outcome is a `SolveResult`
  with a `status`; every failure carries stable error codes from one catalog,
  each with a `recommended_action`. (`infeasible` is an answer, not a failure:
  it carries `infeasibility_proven` and a message instead of an error code;
  a `success` on an exhaustive backend carries `optimality_proven`.)
- **No silent decisions.** An unavailable backend is reported, never swapped
  for a local one. An over-limit parameter is rejected, never clamped. A field
  the schema does not declare is rejected, never ignored. A solve result
  carries the same advisory warnings `validate` gives, so an ignored seed or
  a wide integer range is never hidden behind `success`.
- **Safe by default.** Remote execution and remote retries are off until
  enabled; every resource limit is an environment variable; vendor credentials
  are redacted from results, logs and error messages, and a credential-leak
  test suite proves it.
- **Enforced architecture.** Import boundaries, "no backend names in the
  orchestration, validation or interface layers", and "a new backend plugs
  into the pipeline without touching it" are tests, not conventions.

## Solver backends

| Backend               | Kind   | Path | Notes                                                            |
| --------------------- | ------ | ---- | ---------------------------------------------------------------- |
| `exact`               | local  | BQM  | Enumerates every assignment; 24 compiled variables by default    |
| `simulated_annealing` | local  | BQM  | Heuristic; honours `num_reads`, `num_sweeps`, `seed`            |
| `dwave_qpu`           | remote | BQM  | D-Wave quantum annealer via `EmbeddingComposite`                |
| `leap_hybrid_bqm`     | remote | BQM  | D-Wave Leap hybrid BQM solver                                   |
| `leap_hybrid_cqm`     | remote | CQM  | D-Wave Leap hybrid CQM solver; native constraints               |
| `fujitsu_da`          | remote | BQM  | Fujitsu Digital Annealer, QUBO API V4 over HTTPS, no SDK        |

Remote backends need their vendor credential **and**
`ANNEALBRIDGE_ALLOW_REMOTE=true`; without both they report
`backend_unavailable`. `annealbridge recommend` ranks the backends for a
given problem without solving it and never changes the one you asked for.
Setup steps and per-backend behaviour: [docs/backends.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/backends.md).

## Documentation

The pages below live under [docs/](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/README.md).

| Page                                             | What it covers                                                        |
| ------------------------------------------------ | --------------------------------------------------------------------- |
| [docs/problem-format.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/problem-format.md) | The input JSON: variables, objective, constraints, solver preferences |
| [docs/output-format.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/output-format.md)   | `SolveResult` and every field it carries                              |
| [docs/errors.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/errors.md)                 | Error catalog, warning codes, reason codes, exit codes                |
| [docs/cli.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/cli.md)                       | The `annealbridge` command line                                       |
| [docs/mcp.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/mcp.md)                       | The MCP server, tools, host configuration, Inspector                  |
| [docs/backends.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/backends.md)             | The six backends, D-Wave and Fujitsu setup, adding a backend          |
| [docs/configuration.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/configuration.md)   | Every `ANNEALBRIDGE_*` variable and the vendor credentials            |
| [docs/architecture.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/architecture.md)     | Layers, package layout, design principles                             |
| [docs/security.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/security.md)             | Defaults, limits, credential redaction, what reaches a vendor         |
| [docs/testing.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/testing.md)               | Test layout, golden tests, live tests, CI                             |
| [docs/limitations.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/limitations.md)       | Known limits and what is out of scope                                 |

## Security in one paragraph

Remote execution is **off** by default, remote retries are **off** by
default, and every limit (`ANNEALBRIDGE_MAX_*`) is enforced as an error rather
than clamped, so a request can never quietly become several billed
submissions. The streamable-http transport binds `127.0.0.1` and has **no
authentication**; keep it behind a reverse proxy or a private network.
Credentials never appear in results, logs or error messages. Details in
[docs/security.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/security.md); reporting in [SECURITY.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/SECURITY.md).

## Development

```bash
git clone https://github.com/TheTsungYing/AnnealBridge.git
cd AnnealBridge
pip install -e ".[all,dev]"
pytest
```

`pytest` runs the full suite with no skip and no xfail and never touches the
network; the live vendor tests are opt-in (`pytest -m remote`). Architecture
rules, design principles and the pull-request checklist are in
[CONTRIBUTING.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/CONTRIBUTING.md).

## Status

Version 0.1.0. The problem contract (`1.0` / `1.1`), the six backends, the
CLI and the MCP tools are complete and covered by tests. Not supported, by
design for now: real-valued or unbounded variables, alternative integer
encodings (one-hot, unary), nonlinear constraints, automatic soft-weight
normalization, and the Fujitsu annealer's native inequality / one-hot
features. See [docs/limitations.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/docs/limitations.md) and
[CHANGELOG.md](https://github.com/TheTsungYing/AnnealBridge/blob/main/CHANGELOG.md).

## License

[MIT](https://github.com/TheTsungYing/AnnealBridge/blob/main/LICENSE)
