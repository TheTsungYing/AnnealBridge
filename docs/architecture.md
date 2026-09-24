[← Back to README](../README.md) · [Documentation index](README.md)

# Architecture

## Overview

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
  │     models → validation → penalty → compiler {bqm, cqm} → solvers     │
  │                                                              │        │
  │                                                        orchestration  │
  │                                                                       │
  │     penalty is applied on the bqm path only.                          │
  │     integers are binary-encoded on the bqm path, native on cqm.       │
  │     solvers: exact, simulated_annealing, tabu,                        │
  │              simulated_bifurcation, dwave_qpu,                        │
  │              leap_hybrid_bqm, leap_hybrid_cqm, fujitsu_da             │
  │                                                                       │
  │     pydantic + dimod + dwave-samplers only.                           │
  │     Imports no mcp, no dwave.cloud, no config,                        │
  │     no third-party HTTP client.                                       │
  │     Usable as a plain Python library with zero extras installed.      │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘
```

Layering (lower layers never depend on higher ones):

```text
models ← validation ← penalty ← compiler ← solvers ← orchestration ← CLI / MCP
```

`penalty` is a sibling consumer of `validation.estimates`, not a step the
compiler calls: orchestration applies it on the BQM path, and the compiler
reads `compute_objective_scale` from `validation.estimates` directly.

Two modules sit at the top of the package, beside the layers rather than in
the chain, so every core layer can import them: `exceptions.py` and
`interrupt.py`. `interrupt.py` holds `CancelToken` and `Interrupt`, the stop
condition a solve's backends and post-processing poll (see
[Stopping a solve early](#stopping-a-solve-early)); it imports the standard
library only.

## Enforced boundaries

These boundaries are not a convention — `tests/architecture/` fails the build
when any of them is broken. The full list of rules:

| Rule | What fails the build |
| --- | --- |
| Core imports | Any file in `models/`, `validation/`, `compiler/`, `penalty/`, `solvers/` or `orchestration/` that imports `mcp`, `dwave.cloud`, `annealbridge.config` or `annealbridge.interfaces` |
| Validation depends downward only | `validation/` importing `penalty`, `compiler`, `solvers`, `orchestration`, `config` or `interfaces` |
| Compiler never imports penalty | `compiler/` importing `annealbridge.penalty` |
| Orchestration knows no concrete backend | `orchestration/` importing `solvers.exact`, `solvers.simulated_annealing`, `solvers.tabu`, `solvers.simulated_bifurcation`, `solvers.dwave_qpu`, `solvers.leap_hybrid_bqm`, `solvers.leap_hybrid_cqm` or `solvers.fujitsu_da` |
| Concrete compilers have one importer | `BQMCompiler` / `CQMCompiler` (in any import form) imported by an orchestration file other than `orchestration/optimizer.py`, which is where the default compiler list is assembled |
| Candidate, message and post-processing code knows no compiler | `orchestration/candidates.py`, `orchestration/messages.py` or `orchestration/postprocess.py` importing anything from `annealbridge.compiler`, `compiler.base` included |
| No third-party HTTP client | `requests`, `httpx` or `aiohttp` imported anywhere in the package, at module level or inside a function |
| No backend-name constants | A string constant whose *whole* value is a shipped backend name, appearing in `orchestration/`, `validation/`, `interrupt.py`, `interfaces/capabilities.py`, `interfaces/mcp/tools.py` or `interfaces/cli/main.py` |
| The interrupt module stays at the bottom | `interrupt.py` importing anything but the standard library — an `annealbridge` module (relative imports included) or a third-party package |
| A new backend plugs in by declaration alone | A fake backend registered next to the built-ins that cannot be routed, limited, warned about, recommended and redacted purely from its own declaration |

Two deliberate carve-outs:

- `dwave.system` is **not** banned. Remote backends lazy-import it inside a
  function so the core install stays free of the `[dwave]` extra.
- `solvers/ocean.py` may lazy-import `dwave.cloud.config` **inside a
  function**, because the active Ocean token has to be resolvable for
  redaction. A module-level import there is still a violation, and the
  exemption belongs to that one D-Wave-aware module — the shared
  `solvers/metadata.py` knows no vendor at all.

The backend-name rule matches whole values only: a message that *mentions* a
backend is fine, a string that *is* a backend name is a dispatch waiting to
happen. Docstrings are exempt (comments never reach the AST). The names are
allowed to appear in `models/problem.py` (the `backend` literal and the
option blocks), `models/error_catalog.py`, each backend's own module,
`solvers/registry.py`, the docs and the tests.

The "fifth backend" rule is the strongest of the set. A test-only backend is
registered alongside the eight shipped ones, and the suite proves that the
service, the validator, the capabilities view, the policy limits, the
recommendation ranking, the credential redaction and the wall-clock limit
(through `supports_interrupt`) all handle it from its declaration alone — then
asserts that `orchestration/optimizer.py`,
`orchestration/candidates.py`, `orchestration/messages.py`,
`orchestration/postprocess.py`,
`orchestration/limits.py`, `orchestration/policy.py`, `orchestration/routing.py`,
`interfaces/capabilities.py`, `validation/problem_validator.py`,
`solvers/metadata.py`, `solvers/base.py` and `interrupt.py` never mention its
name, its error code or its credential variables.

## Package layout

Everything lives under `src/annealbridge/`.

| Package | Responsibility |
| --- | --- |
| `models/` | Pydantic domain models — the `OptimizationProblem` IR, the result and capability models, and the error catalog |
| `validation/` | Problem validation, per-candidate solution validation, bounds-aware size estimates, backend recommendation, numeric tolerances |
| `penalty/` | The penalty strategy: objective scale, penalty scale, initial penalty and the doubling ladder |
| `compiler/` | The BQM compiler (slack and integer encoding), the CQM compiler, and the `decode` step that folds encoding bits back into integer values |
| `solvers/` | The eight solver backends, the registry, the `SolverBackend` protocol, the read sharding the two `dwave-samplers` backends share (`solvers/sharding.py` — not a backend), and metadata sanitisation / redaction |
| `orchestration/` | `OptimizationService`, `ExecutionPolicy`, model-type routing, candidate arrays (`candidates.py`), opt-in post-processing over the business variables (`postprocess.py`), result messages (`messages.py`), progress events; re-exports `CancelToken` and `SolveCancelled` for library callers |
| `interrupt.py` (module) | `CancelToken` and `Interrupt`: the wall-clock deadline and the caller's cancellation, polled at checkpoints. Standard library only |
| `exceptions.py` (module) | The core's exception types, `SolveCancelled` among them |
| `config/` | `ServerSettings` (the `ANNEALBRIDGE_*` environment) — importable by the interfaces only |
| `interfaces/` | `capabilities.py` and `composition.py` shared by both adapters, plus `cli/` and `mcp/` |

## Solve pipeline

`OptimizationService.solve(problem)` runs one deterministic pipeline:

1. **Validate** the problem against the schema and its semantics. All errors
   are collected in one pass; an invalid problem is returned as a structured
   `invalid_problem` result and never reaches a solver.
2. **Compile** it. The backend declares its `supported_model_types` and the
   service picks the first declared type it has a compiler for
   (`{bqm: BQMCompiler, cqm: CQMCompiler}`). No name check is involved. If no
   declared type has a compiler the result is `configuration_error` /
   `NO_COMPILER_FOR_MODEL_TYPE`. On the BQM path hard constraints become a
   penalty term and inequalities get binary slack; on the CQM path they are
   submitted natively.
3. **Solve** through the `SolverBackend` protocol. The service never touches a
   vendor SDK itself.
4. **Re-validate** every returned sample independently against the *original*
   problem — never against BQM energy, and never trusting a sampler's own
   feasibility flag. Encoded integer bits are decoded back into integer values
   first, so validation and ranking only ever see the agent's variables.
5. **Rank** the feasible solutions by `ranking_score` with deterministic
   tie-breaking, and keep the top `top_k`.
6. **Retry** with a doubled hard penalty when nothing feasible was found.
   Within one solve the compile work that does not depend on the hard
   penalty (integer and slack encoding, the objective, the soft penalties) is
   done once, through the compiler's optional `SupportsPrepare` stage, and a
   retry only re-expands the hard penalty terms; the model is bit for bit
   what a from-scratch compile at that penalty gives. Nothing is cached
   across solves.

`solve` also takes a keyword-only `on_progress` callback. It is called once
with a `SolveProgress` as the compile, solve and validate stage of each attempt
begins, on the thread running the solve and outside every per-attempt timing
window, so a listener never inflates an attempt's reported durations. Whatever
the callback raises is logged and dropped: it can never change the result. The
MCP server passes one to turn those events into progress notifications; the
CLI passes none.

The retry ladder is deliberately narrow. It runs at most once per attempt
budget, and the budget is `1` — a single attempt, no retry — whenever any of
these holds:

- the backend is exhaustive (`exact`): it already enumerated every state, so a
  larger penalty cannot surface a new feasible sample;
- the compiler does not use a hard penalty (the CQM path): there is no lever
  to turn, so a retry would repeat the identical submission;
- the backend is remote and `ANNEALBRIDGE_ALLOW_REMOTE_RETRIES` is off.

Otherwise the budget is `1 + max_retries`. A penalty that would have to double
past the floating-point range stops the ladder with a structured
`PENALTY_OVERFLOW` result rather than a solver error, keeping the attempts made
so far and calling no backend with a non-finite model.

### Stopping a solve early

`solve` also takes a keyword-only `cancel`, a `CancelToken`, and the problem
may set `solver.wall_clock_limit_seconds`. When either is present the service
builds one `Interrupt` for the solve: a deadline measured with the same clock
reading `elapsed_ms` starts from, the token, or both. When neither is present
no `Interrupt` exists and the solve runs exactly as it did before the feature.

The `Interrupt` is polled, never pushed: nothing is stopped from outside, so
every thread a solve with an `Interrupt` starts has returned before it does —
also when a shard fails, which then stops the shards not yet started and
waits for the running ones. A solve without an `Interrupt` handles a failed
shard the same way, so the failure is reported only after the shards already
running have ended, and the solve's concurrency slot is released with none of
them still using a CPU.

- **Service checkpoints** — before each attempt, after compiling, after the
  backend returns, before every post-processing start and step, and after the
  samples are processed. A cancelled token raises `SolveCancelled` at the next
  one (cancellation wins when both apply). A deadline that has passed stops
  post-processing, keeps the next retry from starting, and marks the result
  `wall_clock_limit_reached`; the first attempt always starts.
- **Backend checkpoints** — the `Interrupt` is handed to a backend only when
  its capabilities declare `supports_interrupt`, as the keyword `interrupt`.
  The backend stops between its own units of work and returns the reads that
  completed, flagged `RawSolverResult.interrupted`, rather than raising; the
  service reads that flag instead of the clock. Which backends declare it, and
  their checkpoints, are in
  [Backends](backends.md#wall-clock-limits-and-cancellation).
- **Not interruptible** — validation, compilation, decoding, re-validation
  and ranking, the infeasibility diagnostics and post-processing's setup.
  Every sample that did arrive is still re-validated in full.

`SolveCancelled` is the one exception `solve` raises by design. It is not an
`OptimizerError`: every domain failure is still a structured result, and a
cancellation is not a failure but the caller no longer wanting one. It is
raised only from the attempt loop, after the concurrency slot was taken, and
the slot is released on the way out; a request that fails earlier (invalid
problem, unavailable backend, full server) returns its result as usual. The
MCP server creates a token per `solve_optimization` call and cancels it when
the client cancels the request (see [MCP](mcp.md#cancellation)); the CLI
passes none.

## Design principles

These eight rules are the ones most easily lost. If a change seems to require
crossing one of them, stop and ask rather than working around it.

1. **The agent supplies *what*, never *how*.** The input JSON carries
   variables, an objective, constraints and soft weights. It never carries a Q
   matrix, a penalty λ, a slack variable or a BQM bias. When a test or an
   example needs a QUBO, that QUBO is compiler *output*, not input.
2. **Correctness is decided by re-checking the original problem, never by
   solver energy.** A sampler's energy is an internal number mixing penalties,
   slack and offsets. It may not decide feasibility and it may not be used as
   the business objective for ranking. Every candidate is re-evaluated against
   the original JSON. Taking only `.first`, or only the lowest-energy sample,
   is not acceptable — all candidates are validated.
3. **Hard penalty and soft weight are two different things.** The hard penalty
   is computed by the program so that hard constraints cannot be violated; the
   agent must not supply it. A soft weight is the agent's statement of how much
   a preference matters, expressed in objective units. Neither is derived from
   the other, they never share a field, and neither is a fallback for the
   other. Chain strength (a QPU physical parameter) is independent again.
4. **Solvers are replaceable parts, and the core knows none of them.** The
   models, the compiler and the validator know only the abstract
   `SolverBackend` protocol; every backend is a plugin discovered through the
   registry. Adding a vendor means adding a plugin, not editing the core. The
   classic drift is a pile of `if backend == "...":` special cases in
   orchestration.
5. **Never decide for the user silently.** An unavailable backend returns
   `backend_unavailable` — it does not quietly fall back to a local solver. A
   request above a limit is rejected — it is not clamped down to the ceiling. A
   failed remote solve returns a structured error — it is not resubmitted four
   times on the user's quota. An agent chooses its next step from the result;
   a silent downgrade would let it believe it received a quantum answer.
6. **Each layer does its own job.** CLI and MCP receive a request, convert
   formats and return a result — no optimization logic. The service sequences
   validate → compile → solve → validate → rank → retry. The compiler
   translates and does not solve. A solver solves and neither parses JSON nor
   validates answers. The validator checks and never modifies.
7. **No scaffolding for a future that has not arrived.** Exactly three
   abstractions earn their keep: `ModelCompiler`, `SolverBackend` and
   `PenaltyStrategy`. Everything else is written concretely — no factories, no
   repository layer, no plugin loader. (`SupportsPrepare` is an optional,
   staged form of `ModelCompiler`, not a fourth abstraction.)
8. **Tests are not to be worked around.** No `skip` or `xfail` to paper over a
   real bug. A scenario test must run the whole pipeline (JSON → service →
   result), not call an internal function and call itself an integration test.
   An MCP test must go through a real MCP client, not call the Python function
   underneath.

## Self-check list

Run through this after finishing a change — whether you are a contributor or
an AI agent working on the repository. Any "yes" means the change has drifted:

- Did I let a QUBO, a penalty λ or a slack variable appear in the input JSON?
- Did I use energy to decide feasibility, or to rank?
- Did I take only `.first`, or only the single lowest-energy sample?
- Did I put the hard penalty and a soft weight in the same field, or derive
  one from the other?
- Did I import a specific solver's package inside `orchestration/`,
  `compiler/` or `validation/`?
- Did I write optimization logic in the CLI or MCP layer?
- Did I silently substitute another backend when the requested one was
  unavailable?
- Did I silently clamp a user's parameter?
- Did I add a `skip` or an `xfail` to make a test pass?
- Did I rewrite existing core logic to make a new feature fit, instead of
  adding it as a declaration or a plugin?
- Did I let a credential reach any output, log line, error message or test
  file?

## Library use without the interfaces

One practical check on the whole design: **if you deleted `interfaces/cli/`
and `interfaces/mcp/` outright, would the remaining core still solve a
problem?** It must, and it does.

```python
import json

from annealbridge.models import OptimizationProblem
from annealbridge.orchestration import OptimizationService

with open("examples/knapsack.json", encoding="utf-8") as f:
    problem = OptimizationProblem.model_validate(json.load(f))

result = OptimizationService().solve(problem)
```

A library caller can cancel a solve from another thread with a `CancelToken`:

```python
import threading

from annealbridge.orchestration import CancelToken, SolveCancelled

token = CancelToken()
threading.Timer(2.0, token.cancel).start()  # e.g. a UI's Cancel button
try:
    result = OptimizationService().solve(problem, cancel=token)
except SolveCancelled:
    result = None  # stopped; no partial result, every thread has returned
```

A cancelled solve returns no partial result; for a partial result on a
deadline, set `solver.wall_clock_limit_seconds` instead (see
[Stopping a solve early](#stopping-a-solve-early)). `CancelToken` and
`SolveCancelled` are exported from `annealbridge.orchestration`; the top-level
`annealbridge` package exports nothing.

(`examples/knapsack.json` is a repository checkout path; an installed
package has no such directory for the CLI to read. Any problem JSON saved
locally works the same.)

The core depends on pydantic, dimod and dwave-samplers only. It needs no
`mcp`, no `dwave-system`, no configuration file and no environment variable —
`OptimizationService()` builds its own default registry, compilers, penalty
strategy and policy. Configuration is a property of the *adapters*: they read
`ANNEALBRIDGE_*` through `ServerSettings` and hand the core an
`ExecutionPolicy`, which is a plain object a library caller can construct
directly.

See also: [Testing](testing.md) for how these rules are checked, and
[Limitations](limitations.md) for what the design deliberately does not cover.
