# AnnealBridge — Optimization Tool Middleware

A combinatorial optimization middleware designed to be called by AI agents over
[MCP](https://modelcontextprotocol.io). An agent produces a structured
`OptimizationProblem` JSON — variables, objective, constraints, nothing else —
and AnnealBridge deterministically validates it, compiles it into a Binary
Quadratic Model (BQM) or a Constrained Quadratic Model (CQM), solves it on a
local or D-Wave backend, re-validates every candidate against the *original*
problem, and returns ranked feasible business solutions.

The agent never writes a QUBO matrix, a penalty weight, or a slack variable.
The deterministic Python core owns all of that.

Specifications: [Phase 1](optimization_middleware_phase1_spec_v2.md),
[Phase 2](annealbridge_phase2_spec_v2.md) and
[Phase 3a](annealbridge_phase3a_spec_v1.md).

## Architecture

The package is split into a self-contained **core** and thin **interfaces**.
The CLI and the MCP server are adapters: they parse arguments, read the
environment, build an `OptimizationService`, and format results. No
optimization logic lives in either of them.

```text
  ┌───────────────────────────── interfaces ──────────────────────────────┐
  │                                                                       │
  │     annealbridge  (CLI)                annealbridge-mcp  (MCP)        │
  │     typer commands                     stdio / streamable-http        │
  │              │                                     │                  │
  │              └──────────────────┬──────────────────┘                  │
  │                                 │                                     │
  │                    composition root + config                          │
  │            ServerSettings (ANNEALBRIDGE_* env) → ExecutionPolicy      │
  │                                                                       │
  └─────────────────────────────────┼─────────────────────────────────────┘
                                    │  OptimizationService.solve(problem)
  ┌─────────────────────────────────▼──────────────── core ───────────────┐
  │                                                                       │
  │     models → validation → compiler {bqm, cqm} → penalty → solvers     │
  │                                                              │        │
  │                                                        orchestration  │
  │                                                                       │
  │     penalty is applied on the bqm path only.                          │
  │     solvers: exact, simulated_annealing, dwave_qpu,                   │
  │              leap_hybrid_bqm, leap_hybrid_cqm                         │
  │                                                                       │
  │     pydantic + dimod + dwave-samplers only.                           │
  │     Imports no mcp, no dwave.cloud, no config.                        │
  │     Usable as a plain Python library with zero extras installed.      │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘
```

Layering (lower layers never depend on higher ones):

```text
models ← validation ← compiler ← solvers ← orchestration ← CLI / MCP
```

These boundaries are not a convention — they are enforced by
`tests/architecture/`, which fails the build if `models/`, `validation/`,
`compiler/`, `penalty/`, `solvers/`, or `orchestration/` ever import `mcp`,
`dwave.cloud`, or `annealbridge.config`.

Key principles:

- **The agent never produces QUBO matrices, BQM biases, or slack variables.**
- **Feasibility and objective values are always recomputed from the original
  problem**, never inferred from BQM energy. Energy is retained for debugging.
- **Solvers are replaceable components** behind a small `SolverBackend`
  protocol, discovered through a registry.

Package layout under `src/annealbridge/`: `models/` (pydantic schema),
`validation/` (problem + solution validators), `compiler/` (BQM compiler with
slack encoding and CQM compiler), `penalty/` (penalty strategy), `solvers/`
(backends and registry), `orchestration/` (the `OptimizationService` pipeline
and `ExecutionPolicy`), `config/` (environment settings), and `interfaces/`
(`cli/`, `mcp/`).

## Installation

Requires Python >= 3.11.

```bash
# Core: pydantic, dimod, dwave-samplers, typer.
# Includes the local `simulated_annealing` and `exact` backends and the CLI.
pip install annealbridge

# + MCP server (mcp>=2,<3): adds the `annealbridge-mcp` entry point.
pip install "annealbridge[mcp]"

# + D-Wave cloud (dwave-system): enables the remote `dwave_qpu`,
#   `leap_hybrid_bqm` and `leap_hybrid_cqm` backends.
pip install "annealbridge[dwave]"

# Everything.
pip install "annealbridge[all]"
```

The core install has no dependency on `mcp` or `dwave-system`; the solver
registry still imports cleanly and simply reports the remote backends as
unavailable.

Two entry points are installed:

| Command            | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| `annealbridge`     | Command-line interface                           |
| `annealbridge-mcp` | MCP server (requires the `[mcp]` extra)          |

## Quick Start

Solve the bundled knapsack example from the command line:

```bash
annealbridge solve examples/knapsack.json
```

```text
Problem:   knapsack
Backend:   exact
Status:    success
Attempts:  1

Best solution (rank 1)
  objective (maximize):  17
  soft violation score:  0
  item_a = 1
  item_b = 0
  item_c = 1
  item_d = 0

Hard constraints: 1 / 1 satisfied
Soft constraints: 0 violations
```

Or use the library directly — no MCP, no extras required:

```python
import json

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

with open("examples/knapsack.json", encoding="utf-8") as f:
    problem = OptimizationProblem.model_validate(json.load(f))

result = OptimizationService().solve(problem)
print(result.status)                          # "success"
print(result.solutions[0].variables)          # {"item_a": 1, "item_b": 0, ...}
print(result.solutions[0].objective_value)    # 17.0
```

Domain failures are returned as structured results, never raised:
`SolveResult.status` is one of `success`, `infeasible`, `invalid_problem`,
`resource_limit_exceeded`, `backend_unavailable`, `configuration_error`, or
`solver_error`.

## Agent Flow

```text
  natural language
  "pick the most valuable items that fit in a 10 kg bag"
          │
          ▼
       Agent ─────────── get_optimization_capabilities (what is supported?)
          │              validate_optimization_problem (dry-run check)
          │              recommend_backend             (which backend fits?)
          │
          │  emits OptimizationProblem JSON
          │  (variables / objective / constraints only —
          │   no QUBO, no penalty λ, no slack variables)
          ▼
   MCP tool call: solve_optimization
          │
          ▼
  ┌───────────────── AnnealBridge ─────────────────┐
  │  1. validate      schema + semantics, all      │
  │                   errors collected in one pass │
  │  2. compile       objective + constraints →    │
  │                   BQM (penalty λ, slack) or    │
  │                   CQM (native constraints)     │
  │  3. solve         exact / SA / QPU / hybrid    │
  │  4. re-validate   every sample independently   │
  │                   against the ORIGINAL JSON,   │
  │                   never against BQM energy     │
  │  5. rank          feasible solutions, top-K    │
  └────────────────────────┬───────────────────────┘
          │
          │  SolveResult: ranked, validated solutions with
          │  per-constraint evaluations and objective values
          ▼
       Agent  ──►  natural-language answer
```

The contract in both directions is business-level. The agent describes *what*
to optimize; AnnealBridge decides *how* to encode and solve it, and proves the
answer against the problem the agent actually asked about.

## Tool Input Example

A minimal `solve_optimization` input — a 0/1 knapsack with a capacity of 10
(the reduced form of [examples/knapsack.json](examples/knapsack.json)):

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

## JSON Input Format

The document above is the whole public API. A fuller problem may also carry
`quadratic_terms` and a `constant` on the objective, soft constraints, and
solver preferences:

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

- Only **binary variables** are supported. Variable names must be unique and
  must not start with `__` (reserved for compiler-internal slack variables).
- `type: "hard"` constraints must be satisfied and must not carry a `weight`;
  `type: "soft"` constraints require a `weight > 0`.
- Validation collects **all** errors in one pass and returns them as a
  structured `invalid_problem` result; an invalid problem is never sent to a
  solver.

Run `annealbridge export-schema` for the complete JSON Schema — the same
schema an agent can use for structured output, and the same one returned by
the `get_optimization_capabilities` tool.

### Integer coefficients for inequality constraints

Inequality constraints (`<=` / `>=`) are encoded into the BQM using binary
slack variables, which requires an exact integer range. Therefore **every
`coefficient` and the `rhs` of a `<=` / `>=` constraint must be an integer
value** (the JSON number may be written as `6` or `6.0`, but it must be
mathematically integral). Non-integer values are rejected with the
`NON_INTEGER_INEQUALITY` validation error. Equality (`==`) constraints are not
subject to this restriction; no automatic scaling of fractional coefficients
is performed. This restriction also applies on the CQM path, where slack
variables are not used at all — it is kept deliberately conservative so that
both paths accept exactly the same problems (Phase 3a spec §21.3).

### Soft constraint weights

A soft constraint contributes `weight × (violation)²` to the solver's energy
and to the solution's `soft_violation_score`. The `weight` is expressed **in
objective units** and is **not normalized** — if your objective coefficients
are in the thousands, a weight of 5 has almost no influence. The compile step
exposes `objective_scale`
(`max(1.0, Σ|linear coefficients| + Σ|quadratic coefficients|)`), an upper
bound on the objective's range over binary variables; choose weights relative
to that scale. `objective_scale` never includes soft weights.

### Hard constraint penalties

The hard-constraint penalty λ is computed by the penalty strategy, never taken
from the agent. It is sized against the whole non-penalty energy landscape:

```text
penalty_scale   = objective_scale + Σ_soft weight × D²
initial_penalty = penalty_scale × penalty_multiplier   (default multiplier 2)
retry           = previous × 2
```

where `D` is the largest absolute value the soft constraint's squared term
can reach over all binary assignments (slack bits included). A soft weight far
above the objective therefore cannot drown a hard constraint: with
`multiplier > 1` the lowest-energy assignment is always feasible whenever one
exists (assuming integer coefficients, which inequalities already require).
Without soft constraints `penalty_scale` equals `objective_scale`. Soft
weights are only used to *bound* the energy the penalty must dominate; they
are never used as, or substituted for, the hard penalty itself.

Ranking accounts for soft violations: solutions are ordered by `ranking_score`
(`objective_value + soft_violation_score` when minimizing,
`objective_value − soft_violation_score` when maximizing), with deterministic
tie-breaking. Both components are reported separately so a consumer can
re-rank.

## CLI

```bash
# Human-readable solve report
annealbridge solve examples/knapsack.json

# Override the backend declared in the JSON
annealbridge solve examples/knapsack.json --backend exact

# Full SolveResult as JSON
annealbridge solve examples/knapsack.json --json

# Validate without solving; add --backend / --json
annealbridge validate examples/knapsack.json

# Rank the backends for a problem without solving it; advisory only
annealbridge recommend examples/knapsack.json

# Which backends are installed, enabled, and under what limits
annealbridge capabilities

# Export the OptimizationProblem JSON Schema
annealbridge export-schema
```

`annealbridge capabilities` prints a table of **Backend / Available / Enabled /
Remote / Limits**, resolved from the installed extras and the current
`ANNEALBRIDGE_*` environment. It performs no network I/O, so it is safe to run
before any credentials are configured.

Exit code is `0` on `success` and non-zero otherwise (`1` for a non-success
`SolveResult`, `2` for unreadable or malformed input files).

`annealbridge validate` exits `0` when the problem is valid, `1` when it is
invalid, and `2` for an unreadable or malformed file; `annealbridge recommend`
exits `0` on a successful ranking, `1` for an invalid problem, and `2` for a
file error. Both are one-line delegations to `OptimizationService.validate()`
and `OptimizationService.recommend()` — the same code paths the MCP tools use,
so the CLI and an agent always see the same answer. `recommend` never rewrites
`solver.backend`: it only ranks, and solving still uses the backend you asked
for.

## Solver Backends

| Backend               | Kind      | Notes                                                       |
| --------------------- | --------- | ----------------------------------------------------------- |
| `exact`               | local     | `dimod.ExactSolver`; enumerates every assignment            |
| `simulated_annealing` | local     | `dwave.samplers.SimulatedAnnealingSampler`; heuristic       |
| `dwave_qpu`           | remote    | `EmbeddingComposite(DWaveSampler())`                        |
| `leap_hybrid_bqm`     | remote    | `LeapHybridSampler`                                         |
| `leap_hybrid_cqm`     | remote    | `LeapHybridCQMSampler`; native constraints (CQM path)       |

- **`exact`** — a testing/debugging backend and a ground-truth benchmark for
  the annealer. The state space doubles with every variable, so it refuses
  problems with more than **24 variables** (including compiler-generated slack
  variables; configurable via `ANNEALBRIDGE_EXACT_MAX_VARIABLES`). It never
  retries, and when it finds no feasible solution the result carries
  `infeasibility_proven: true`.
- **`simulated_annealing`** — honors `num_reads`, `num_sweeps`, and `seed`
  (fixed seeds give reproducible sampling). If no feasible solution is found,
  the service retries with a doubled hard-constraint penalty, up to
  `max_retries` times. An `infeasible` result only means "not found under this
  configuration" (`infeasibility_proven: false`).
- **`dwave_qpu`**, **`leap_hybrid_bqm`** and **`leap_hybrid_cqm`** — require
  the `[dwave]` extra, D-Wave Leap credentials, and
  `ANNEALBRIDGE_ALLOW_REMOTE=true`. See [D-Wave Setup](#d-wave-setup). Without
  all three they report `backend_unavailable`; there is never a silent
  fallback to a local solver.
- **`leap_hybrid_cqm`** — the only backend on the **CQM path**. The problem is
  compiled into a `dimod.ConstrainedQuadraticModel`, so hard constraints are
  submitted natively: no penalty λ, no slack variables. Soft constraints become
  weighted constraints (`weight × violation²`, the same formula the validator
  uses for `soft_violation_score`). There is therefore always exactly one
  attempt, `SolveAttempt.penalty` is `null`, there is no penalty retry, and no
  `REMOTE_RETRIES_DISABLED` warning is emitted. The sampler's own `is_feasible`
  flag is **not trusted**: every sample is still re-validated against the
  original JSON, and the flag is only counted into
  `metadata.sampler_reported_feasible`. `metadata.model_type` records `"bqm"`
  or `"cqm"` for every solve. Preferences go in their own block:
  `"solver": {"backend": "leap_hybrid_cqm", "leap_hybrid_cqm":
  {"time_limit_seconds": 5}}`. The sampler's minimum `time_limit_seconds` is
  5 s — a lower value is raised to the minimum and recorded in
  `metadata.effective_time_limit_seconds`, while a value above
  `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` is rejected with
  `resource_limit_exceeded` / `REMOTE_TIME_LIMIT` rather than clamped. Leap's
  published ceilings — 5,000,000 variables and 100,000 constraints — are
  enforced by D-Wave, not by this server.

Which compiler runs is decided by the backend, not by a name check: each
backend declares its `supported_model_types`, and the service picks the first
declared model type it has a compiler for (`{bqm: BQMCompiler,
cqm: CQMCompiler}`). If none of the declared model types has a compiler, the
result is `configuration_error` / `NO_COMPILER_FOR_MODEL_TYPE`. `exact`,
`simulated_annealing`, `dwave_qpu` and `leap_hybrid_bqm` all take the BQM path,
with unchanged behaviour.

## MCP Server

Install the extra and the `annealbridge-mcp` entry point becomes available:

```bash
pip install "annealbridge[mcp]"
```

It exposes four tools:

| Tool                             | Purpose                                                        |
| -------------------------------- | -------------------------------------------------------------- |
| `get_optimization_capabilities`  | Supported types/operators, backend availability, JSON schema    |
| `validate_optimization_problem`  | Validate a problem without solving it                           |
| `recommend_backend`              | Rank the backends for a problem without solving it (advisory only) |
| `solve_optimization`             | Validate → compile → solve → re-validate → rank                 |

`recommend_backend` is **advisory only**. It ranks every backend for a given
problem — usable or not, model type, reason codes, blocking errors, warnings,
and the estimated number of compiled variables — deterministically, with no
network I/O, no solving, no concurrency slot, and no quota consumed. It never
rewrites `problem.solver.backend`; `solve_optimization` always uses the backend
the caller asked for.

### Claude Desktop (stdio)

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
host's `PATH`, use its absolute path, and pass any configuration through
`env`:

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

Restart the host after editing the file.

### Other MCP hosts

Any MCP host that speaks stdio — Codex, other desktop clients, or your own
client built on an MCP SDK — is configured the same way: run
`annealbridge-mcp` as the server command, with no arguments for stdio. Only
the surrounding configuration file format differs.

### Streamable HTTP

For hosts that connect over HTTP instead of stdio:

```bash
annealbridge-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

The transport defaults to `stdio`. In HTTP mode the server binds `127.0.0.1`
by default; see [Security](#security) before changing that.

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
input/output schemas, and lets you submit a problem JSON by hand.

## D-Wave Setup

Remote backends are opt-in in three separate steps.

**1. Install the extra:**

```bash
pip install "annealbridge[dwave]"
```

**2. Configure Leap credentials.** These are handled by Ocean's own
configuration, not by AnnealBridge settings. Either run the interactive
helper:

```bash
dwave config create
```

or set the environment variable that Ocean reads:

```bash
export DWAVE_API_TOKEN="DEV-xxxxxxxxxxxxxxxxxxxx"   # placeholder, not a real token
```

**3. Allow remote execution.** Even with valid credentials, AnnealBridge will
not reach the network until you say so:

```bash
export ANNEALBRIDGE_ALLOW_REMOTE=true
```

Verify with `annealbridge capabilities` — `dwave_qpu`, `leap_hybrid_bqm` and
`leap_hybrid_cqm` should now show as available and enabled.

### Environment variables

All settings use the `ANNEALBRIDGE_` prefix:

| Variable                            | Default     | Meaning                                        |
| ----------------------------------- | ----------- | ---------------------------------------------- |
| `ANNEALBRIDGE_ALLOW_REMOTE`         | `false`     | Enable the remote (D-Wave) backends            |
| `ANNEALBRIDGE_ALLOW_REMOTE_RETRIES` | `false`     | Allow penalty retries on remote backends       |
| `ANNEALBRIDGE_EXACT_MAX_VARIABLES`  | `24`        | Variable ceiling for the `exact` backend       |
| `ANNEALBRIDGE_MAX_QPU_READS`        | `1000`      | Upper bound on QPU `num_reads`                 |
| `ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US` | `2000` | Upper bound on QPU annealing time (µs)         |
| `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS`   | `300`  | Upper bound on hybrid solver time              |
| `ANNEALBRIDGE_MAX_CONCURRENT_SOLVES`     | `4`    | Concurrent solves allowed                      |
| `ANNEALBRIDGE_HTTP_HOST`            | `127.0.0.1` | Default bind host for streamable-http          |
| `ANNEALBRIDGE_HTTP_PORT`            | `8000`      | Default port for streamable-http               |
| `ANNEALBRIDGE_LIMITS`               | `{}`        | Generic policy limits as a JSON object, e.g. `{"iterations": 100}`; for backends that declare custom limit keys. Not needed by the built-in backends |

Every `ANNEALBRIDGE_*` variable above is retained with unchanged semantics.
`ANNEALBRIDGE_LIMITS` only adds *generic* limit keys for third-party backends
that register custom limit keys declaratively; the five built-in backends use
the dedicated variables instead. It must not contain a key that already has a
dedicated variable — `variables`, `reads`, `annealing_time_us` and
`time_seconds` are rejected, and have to be configured through the older
variables.

D-Wave credentials are deliberately **not** among them: they belong to Ocean's
configuration (`dwave config create` or `DWAVE_API_TOKEN`), so AnnealBridge
never has to read, store, or pass a token itself.

## Security

- **Remote execution is off by default.** `ANNEALBRIDGE_ALLOW_REMOTE` defaults
  to `false`; a request for `dwave_qpu`, `leap_hybrid_bqm` or
  `leap_hybrid_cqm` then returns
  `backend_unavailable` rather than silently falling back to a local solver.
  Nothing touches the network, and no quota is consumed, until you opt in.
- **Remote retries are off by default.** `ANNEALBRIDGE_ALLOW_REMOTE_RETRIES`
  defaults to `false`, so a `max_retries: 3` problem cannot turn into multiple
  billed QPU submissions. Resource limits (`MAX_QPU_READS`,
  `MAX_QPU_ANNEALING_TIME_US`, `MAX_REMOTE_TIME_SECONDS`) are enforced as
  errors, never silently clamped. The CQM path never retries at all: it always
  runs exactly one attempt.
- **Streamable HTTP binds `127.0.0.1` by default.** The server has **no
  authentication or authorization of any kind**. Do not expose it directly to
  a public network or bind it to `0.0.0.0`. If remote access is genuinely
  needed, put it behind an authenticating reverse proxy or a private network.
- **Credentials never appear in output.** Solver metadata is filtered through
  a timing whitelist rather than returned as-is, and results, error messages,
  logs, and metadata all pass through redaction, so an API token cannot leak
  into a tool response or a stack trace. This is covered by a dedicated
  credential-leak test.

## Examples

- [examples/knapsack.json](examples/knapsack.json) — 0/1 knapsack, 4 items,
  capacity 10, one hard `<=` constraint. Global optimum selects items A and C
  for a total value of 17.
- [examples/assignment.json](examples/assignment.json) — 3 workers × 3 tasks
  assignment with six hard equality (one-hot) constraints, minimizing total
  cost. Global optimum: alice=cook, bob=clean, carol=drive, total cost 8.
- [examples/tsp.json](examples/tsp.json) — traveling salesman over 4 cities
  with symmetric distances, encoded as city×position binaries. The optimal
  cyclic tour a-b-c-d has total length 8.

All three declare a local backend; try `--backend simulated_annealing` to
compare against `exact`. With remote execution configured,
`--backend leap_hybrid_cqm` runs the same problem down the CQM path instead.

## Testing

```bash
# Full suite. Never runs remote_live tests, never touches the network,
# never consumes D-Wave quota.
pytest
```

Remote live tests are opt-in and excluded by default:

```bash
# Requires DWAVE_API_TOKEN and CONSUMES REAL LEAP QUOTA.
# Now includes the Leap hybrid CQM live test
# (tests/remote_live/test_leap_hybrid_cqm_live.py, a 3-variable knapsack
# with a 5 s time limit).
pytest -m remote
```

`tests/unit/` covers models, validators, slack encoding, the BQM compiler, the
CQM compiler, the penalty strategy, the policy limits, backend routing, and the
solver backends; `tests/scenarios/` runs the full
JSON → validate → compile → solve → validate → rank pipeline;
`tests/remote_mock/` exercises the D-Wave backends against mocks;
`tests/mcp/` drives the four tools through an in-memory MCP client; and
`tests/architecture/` enforces the import boundaries described above, plus two
further rules: no backend-name constants in `orchestration/`, `validation/` or
the interfaces, and a fake fifth backend that plugs in with zero core changes.

## Limitations

```text
AnnealBridge currently supports binary-variable optimization problems only.
Inequality constraints require integer coefficients and right-hand sides.
Soft constraint weights are expressed in objective units and are not normalized.
The simulated annealing backend is heuristic and does not guarantee a global optimum.
The exact backend is for testing/debugging and is limited to small problems.
Leap Hybrid returns a single sample.
QPU embedding may fail for dense problems.
The CQM path still requires integer coefficients and right-hand sides for inequality constraints (kept deliberately conservative; spec §21.3).
Integer variables are planned for Phase 3b.
```

Out of scope for now: integer and real variables (Phase 3b), nonlinear
constraints, and automatic soft-weight normalization.
