[← Back to README](../README.md) · [Documentation index](README.md)

# Solver backends

This page describes the six solver backends AnnealBridge ships with: what each
one runs on, which compiler path it takes, which solver preferences it honours
and which it ignores, and what it needs before it can be used. It also covers
the setup steps for the D-Wave and Fujitsu backends, how the compiler path is
decided, and what it takes to add a backend of your own.

For the environment variables named here, see
[Configuration](configuration.md). For the error and warning codes, see
[Errors and warnings](errors.md).

## Overview

| Backend | Kind | Model path | Extra needed | Credential |
| --- | --- | --- | --- | --- |
| `exact` | local | bqm | none (core install) | none |
| `simulated_annealing` | local | bqm | none (core install) | none |
| `dwave_qpu` | remote | bqm | `[dwave]` | D-Wave Leap, via Ocean |
| `leap_hybrid_bqm` | remote | bqm | `[dwave]` | D-Wave Leap, via Ocean |
| `leap_hybrid_cqm` | remote | cqm | `[dwave]` | D-Wave Leap, via Ocean |
| `fujitsu_da` | remote | bqm | none (core install) | `FUJITSU_DA_API_KEY` |

Every remote backend additionally requires `ANNEALBRIDGE_ALLOW_REMOTE=true`.
Without it a request for a remote backend returns `backend_unavailable` /
`REMOTE_DISABLED`; there is never a silent fallback to a local solver, and
nothing touches the network.

The name in the first column is the **registry key** — the value to put in
`solver.backend`, the value `ANNEALBRIDGE_ENABLED_BACKENDS` is matched
against, and the `backends[].name` reported by
`get_optimization_capabilities` and `annealbridge capabilities`.

`annealbridge capabilities` prints the live view of this table for the current
install and environment (installed extras, credentials, policy, limits) and
performs no network I/O, so it is safe to run before any credential is
configured. See [CLI](cli.md#capabilities).

## `exact`

- Local exhaustive solver built on `dimod.ExactSolver`: it enumerates every
  assignment, so it proves optimality and — when it finds nothing feasible —
  proves infeasibility (`infeasibility_proven: true`).
- A testing and debugging backend, and a ground-truth benchmark for the
  annealer. The state space doubles with every variable, so the service
  refuses problems above `ANNEALBRIDGE_EXACT_MAX_VARIABLES` (default `24`)
  compiled variables — compiler-generated slack and integer-encoding bits
  included — with `resource_limit_exceeded` / `EXACT_VARIABLE_LIMIT`. The
  ceiling is enforced by the service policy, not inside the backend.
- It never retries: a penalty retry cannot change an exhaustive answer.
- Ignores `num_reads`, `num_sweeps`, `seed` and `max_retries`. A `seed`
  raises a `SEED_IGNORED` warning; the others raise `PARAMETER_IGNORED` when
  set to a non-default value, so a problem that simply leaves them alone gets
  no noise.
- Returns every assignment as a candidate row, so `top_k` really does yield
  the top K distinct feasible solutions.
- Reports `metadata` with `remote: false` and `model_type: "bqm"`. A local run
  has no vendor side, so `timing_us` is empty and the vendor fields are `null`
  — `num_reads_requested` included, since this backend does not sample.

## `simulated_annealing`

- Local heuristic sampler:
  `dwave.samplers.SimulatedAnnealingSampler`. The default backend when a
  problem does not name one.
- Honours `num_reads`, `num_sweeps` and `seed`. A fixed seed gives
  reproducible sampling; the seed is passed to the sampler only and global
  random state is never touched.
- Reads are sampled in shards of 25, up to `ANNEALBRIDGE_SA_WORKERS` of
  them concurrently (see [Configuration](configuration.md)). The shard
  layout and each shard's seed — derived from the request seed through
  `numpy.random.SeedSequence` — depend on `num_reads` and `seed` alone, so
  **the result is a function of `(problem, num_reads, num_sweeps, seed)` and
  is identical for any worker count or machine**; the worker count only
  changes wall time. Consequences: a request of at most 25 reads is a single
  sampler call with the seed passed straight through, exactly as before
  sharding existed; changing `num_reads` changes every read, not just the
  extra ones; and reproducibility holds for the same `dwave-samplers` and
  `numpy` versions, whose sampler and seed-derivation algorithms the result
  depends on.
- Sharding adds a fixed cost per sampler call, independent of `num_reads` and
  `num_sweeps`: every shard re-runs the sampler's default beta-range
  estimate. Measured with `ANNEALBRIDGE_SA_WORKERS=1` and 400 reads (16
  shards) on `dwave-samplers` 1.8.0 and an Intel Core i5-12500, that cost is
  about 15 ms per call for 300 variables / 13.6k interactions and 110 ms for
  800 variables / 96k interactions. Against a single sampler call this adds
  +9 % (300 variables) to +18 % (800 variables) wall time at the default 1000
  sweeps, and +52 % to +73 % at 200 sweeps. Raising
  `ANNEALBRIDGE_SA_WORKERS` so shards run in parallel offsets it, and the
  more sweeps a request asks for, the smaller its share.
- `num_reads` is bounded by `ANNEALBRIDGE_MAX_LOCAL_READS`
  (`LOCAL_READS_LIMIT`) and `num_sweeps` by `ANNEALBRIDGE_MAX_SWEEPS`
  (`SWEEPS_LIMIT`). An over-limit value is rejected, never clamped.
- If no feasible solution is found, the service retries with a doubled
  hard-constraint penalty, up to `max_retries` times (bounded by
  `ANNEALBRIDGE_MAX_LOCAL_RETRIES`).
- Being heuristic, an `infeasible` result only means "not found under this
  configuration": it carries `infeasibility_proven: false`.
- Reports `metadata` with `remote: false`, `model_type: "bqm"` and
  `num_reads_requested` set to the reads asked of the sampler — the whole
  request, not a shard. A local run has no vendor side, so `timing_us` is
  empty and the vendor fields are `null`.

## `dwave_qpu`

- The D-Wave quantum annealer, reached through
  `EmbeddingComposite(DWaveSampler())`. Minor-embedding and chain-break
  resolution (Ocean's default `majority_vote`) are Ocean's job; samples come
  back in logical variables.
- Requires the `[dwave]` extra, D-Wave Leap credentials and
  `ANNEALBRIDGE_ALLOW_REMOTE=true` — see [D-Wave setup](#d-wave-setup).
- Preferences go in their own block:

  ```json
  {"solver": {"backend": "dwave_qpu",
              "dwave_qpu": {"annealing_time_us": 20,
                            "chain_strength": null,
                            "auto_scale": true}}}
  ```

  `chain_strength: null` leaves Ocean's default
  (`uniform_torque_compensation`) in place.
- `num_reads` is bounded by `ANNEALBRIDGE_MAX_QPU_READS` (`QPU_READS_LIMIT`)
  and `dwave_qpu.annealing_time_us` by
  `ANNEALBRIDGE_MAX_QPU_ANNEALING_TIME_US` (`QPU_ANNEALING_TIME_LIMIT`).
- Does not support `seed` (`SEED_IGNORED`) or `num_sweeps` (a non-default
  value raises `PARAMETER_IGNORED`). A failed minor-embedding is
  reported as `EMBEDDING_FAILED`; a dense problem may also raise the
  `DENSE_FOR_QPU` warning ahead of time.

## `leap_hybrid_bqm`

- D-Wave's Leap cloud hybrid (classical + quantum) BQM solver, through
  `LeapHybridSampler`. Suited to large problems.
- Requires the `[dwave]` extra, Leap credentials and
  `ANNEALBRIDGE_ALLOW_REMOTE=true`.
- Typically returns a **single sample** per solve, so `top_k` effectively
  yields at most one solution. That is expected behaviour, not an error.
- The only preference it forwards is its time limit; `num_reads`,
  `num_sweeps` and `seed` are meaningless for the hybrid solver and are never
  sent. A `seed` raises `SEED_IGNORED`, and a non-default `num_reads` or
  `num_sweeps` raises `PARAMETER_IGNORED`.

  ```json
  {"solver": {"backend": "leap_hybrid_bqm",
              "leap_hybrid_bqm": {"time_limit_seconds": 5}}}
  ```

- The effective time limit is the caller's value floored at the sampler's own
  `min_time_limit` for the model (an unset value means the sampler minimum);
  it is recorded in `metadata.effective_time_limit_seconds`. A value above
  `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` is rejected with
  `resource_limit_exceeded` / `REMOTE_TIME_LIMIT` rather than clamped.

## `leap_hybrid_cqm`

The Leap hybrid constrained-quadratic-model solver, through
`LeapHybridCQMSampler`, and the only backend on the **CQM path**. It needs the
`[dwave]` extra, Leap credentials and `ANNEALBRIDGE_ALLOW_REMOTE=true` like the
other two D-Wave backends, but its solving model differs in several ways.

**Native constraints.** The problem is compiled into a
`dimod.ConstrainedQuadraticModel`, so hard constraints are submitted natively:
no penalty λ, no slack variables. Soft constraints become weighted constraints
(`weight × violation²`, the same formula the validator uses for
`soft_violation_score`).

**No retries.** Because there is no penalty to double, there is always exactly
one attempt, `SolveAttempt.penalty` is `null`, and no `REMOTE_RETRIES_DISABLED`
warning is emitted.

**Integers are native.** An integer variable is passed to the model as a
`dimod` `INTEGER` variable with its bounds; no encoding bits are added and none
are counted in the estimated compiled size. `recommend` marks this with the
`R_INTEGER_NATIVE` reason code.

**The sampler's feasibility flag is not trusted.** Every returned sample is
still re-validated against the original problem JSON; the sampler's own
`is_feasible` flag is only counted into `metadata.sampler_reported_feasible`.
`metadata.model_type` records `"bqm"` or `"cqm"` for every solve, on every
backend.

**Time limit.** Preferences go in their own block:

```json
{"solver": {"backend": "leap_hybrid_cqm",
            "leap_hybrid_cqm": {"time_limit_seconds": 5}}}
```

The effective value is the caller's value floored at the sampler's own minimum
for the model (5 s for the CQM sampler) and is reported in
`metadata.effective_time_limit_seconds`. A value above
`ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` is rejected with
`resource_limit_exceeded` / `REMOTE_TIME_LIMIT` — never clamped.

**Ignored preferences.** As on the other hybrid solver, `seed` raises
`SEED_IGNORED` and a non-default `num_reads` or `num_sweeps` raises
`PARAMETER_IGNORED`; because the CQM path applies no hard penalty and never
retries, a non-default `penalty_multiplier` or `max_retries` raises
`PARAMETER_IGNORED` as well.

**Vendor ceilings.** Leap's published limits — 5,000,000 variables and 100,000
constraints — are enforced by D-Wave, not by this server.

## `fujitsu_da`

The Fujitsu Digital Annealer through its **QUBO API V4** (`fujitsuDA3` solver
block), and the first backend that is not a D-Wave product. There is no vendor
SDK: the backend speaks the HTTPS JSON API through the standard library, so it
needs **no extra at all** and adds no dependency — it is part of the core
install and only waits for its API key. See
[Fujitsu setup](#fujitsu-setup).

### What is submitted

It takes the **BQM path**: the whole compiled QUBO — objective,
hard-constraint penalties, slack bits and integer-encoding bits — is submitted
as one `binary_polynomial`. The annealer's native inequality, one-hot and
penalty-polynomial features are not used. Integer variables are therefore
supported through the same binary encoding as on every other BQM backend.

The request body carries only variable indices and coefficients — no problem
name, no description, no other business string.

The vendor enforces a ceiling of 100,000 bits per problem and at most 16
pending jobs per account.

### Options

```json
{"solver": {"backend": "fujitsu_da",
            "fujitsu_da": {"time_limit_seconds": 10,
                           "num_run": 16,
                           "num_group": 1,
                           "num_output_solution": 5}}}
```

| Option | Range | Vendor default when unset |
| --- | --- | --- |
| `time_limit_seconds` | 1–3600 | 10 |
| `num_run` | 1–1024 | 16 |
| `num_group` | 1–16 | 1 |
| `num_output_solution` | 1–1024 | 5 |

An unset option is simply not sent, so the vendor default applies. Only these
four values are ever forwarded: `num_reads`, `num_sweeps` and `seed` are not.
A `seed` raises `SEED_IGNORED`, and a non-default `num_reads` or `num_sweeps`
raises `PARAMETER_IGNORED`.

The effective time limit (the option, or the 10 s vendor default) is compared
against `ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS` and, when above it, refused with
`resource_limit_exceeded` / `REMOTE_TIME_LIMIT` — never clamped.

### Job lifecycle and polling budget

The job is submitted asynchronously and polled until it reports `Done`; the
result is then deleted from the vendor side to free the job slot. A job is
polled for at most `time_limit_sec + 60` seconds; past that budget the backend
stops waiting, sends a best-effort cancel and then a best-effort delete so the
job does not keep occupying one of the account's slots, and returns
`REMOTE_TIMEOUT`.

Each individual HTTP request carries its own request timeout (30 s by
default), so the real worst case is the polling budget plus up to 30 s each for
the submit, the last status request, the cancel and the delete. The one job
that cannot be released is a submit that times out: the vendor may have created
it, but no `job_id` ever reached us, so the error says so and asks the operator
to check the account's job list.

### Results and errors

Every returned solution becomes one candidate row — the vendor's `frequency`
is not expanded into duplicate rows — and is re-validated against the original
problem. The annealer's `energy` and `penalty_energy` are never used to judge
feasibility.

Vendor errors map to catalog codes:

| Vendor response | Code | Status |
| --- | --- | --- |
| HTTP 401 / 403 | `REMOTE_AUTH_FAILED` | `solver_error` |
| HTTP 400, quota message | `REMOTE_QUOTA_EXCEEDED` | `solver_error` |
| HTTP 400, request-header rejection | `BACKEND_CONFIG_INVALID` | `configuration_error` |
| any other HTTP 400 / 413 / 5xx | `REMOTE_SOLVER_ERROR` | `solver_error` |
| HTTP 429 | `REMOTE_BUSY` (retryable) | `solver_error` |
| network timeout | `REMOTE_TIMEOUT` | `solver_error` |

A request-header rejection is deliberately a *configuration* error: the
server's own configuration is wrong, not the problem the agent sent.

`metadata.solver_id` is `fujitsuDA3/v4`, and the vendor's `solve_time` /
`total_elapsed_time` are the only timing fields kept (converted to
microseconds).

## How the compiler path is chosen

Which compiler runs is decided by the backend's **declaration**, not by a name
check. Each backend declares `supported_model_types` in preference order, and
the service picks the first declared model type it has a compiler for
(`{bqm: BQMCompiler, cqm: CQMCompiler}`). The same rule is used by `solve`,
`validate` and `recommend`, so all three agree on the path a problem would
take.

If none of the declared model types has a compiler, the result is
`configuration_error` / `NO_COMPILER_FOR_MODEL_TYPE`. (`validate` reports the
same code as a warning and leaves `model_type` unset, since it does not solve.)

`exact`, `simulated_annealing`, `dwave_qpu`, `leap_hybrid_bqm` and
`fujitsu_da` all declare `bqm`; `leap_hybrid_cqm` declares `cqm`. On the BQM
path hard constraints become penalty terms, inequalities get binary slack
variables, and integer variables are binary-encoded; on the CQM path
constraints and integers are native. See
[Problem format](problem-format.md) for what each path means for a problem, and
[Architecture](architecture.md) for where the compilers sit.

## D-Wave setup

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
`leap_hybrid_cqm` should now show as available and enabled. Until the extra is
installed they report `dwave-system not installed`; with the extra but no
credentials, `D-Wave credentials not configured`.

## Fujitsu setup

The `fujitsu_da` backend has no vendor SDK and no extra to install: it is part
of the core package and uses the standard library for HTTPS. Two steps make it
usable.

**1. Provide the API key.** AnnealBridge reads it from one environment
variable, inside the backend only, live on every call — it is never stored in
settings, a model, or metadata:

```bash
export FUJITSU_DA_API_KEY="da-xxxxxxxxxxxxxxxxxxxx"   # placeholder, not a real key
```

The base URL defaults to `https://api.aispf.global.fujitsu.com/da` and can be
overridden with `FUJITSU_DA_URL`. It must start with `https://`; any other
scheme makes the backend unavailable with `configuration_error` /
`BACKEND_CONFIG_INVALID`, and the reported detail never contains the value.
Only the V4 endpoints are supported, and only the permanent API key (no OAuth
access token).

**2. Allow remote execution**, exactly as for D-Wave:

```bash
export ANNEALBRIDGE_ALLOW_REMOTE=true
```

Verify with `annealbridge capabilities` — `fujitsu_da` shows
`Fujitsu DA API key not configured` until the key is set, and available and
enabled afterwards. The check is local; nothing is sent to Fujitsu until a
`solve` actually targets the backend.

Remember the polling budget above when choosing
`ANNEALBRIDGE_MAX_REMOTE_TIME_SECONDS`: a solve occupies one of the
(four, by default) concurrency slots for up to `time_limit_sec + 60` seconds
plus the per-request timeouts.

The opt-in live test for this backend consumes paid DA time; see
[Testing](testing.md).

## Adding a backend

A backend is a replaceable component. Adding one needs no change to the
optimization core: the architecture test
`tests/architecture/test_fifth_backend.py` plugs a fake backend into the
registry and exercises capabilities, routing, limits, recommendation and
redaction without a single edit to `orchestration/optimizer.py`,
`orchestration/limits.py`, `orchestration/policy.py`,
`orchestration/routing.py`, `interfaces/capabilities.py`,
`validation/problem_validator.py` or `solvers/metadata.py` — the test asserts
those files never mention the fake. The only schema-level edit a real new
backend needs is its name in the `SolverPreferences.backend` list (and its
option block, if it has one) in `models/problem.py`, so the public JSON API
still names exactly the backends this build ships.

**1. Implement the `SolverBackend` protocol**
(`annealbridge.solvers.base`). It is a structural `Protocol`, so no base class
is required:

- `capabilities` — a `SolverCapabilities` property (below).
- `is_available()` — returns an `AvailabilityStatus` with a `category`
  (`available`, `not_installed`, `credentials_missing`, `config_invalid`,
  `unavailable`), an optional categorical `detail` and an optional
  `error_code`. It must perform **no network I/O**, must not cache, and its
  `detail` must never contain a configuration value.
- `name` and `is_exhaustive` — aliases for the corresponding capability
  fields.
- `resolve_time_limit(compiled_problem, preferences)` — the effective time
  limit `solve` would submit, or `None` for a backend that declares none. It
  must not submit anything and must be deterministic, so the service can check
  it against policy before the solve runs.
- `solve(compiled_problem, preferences)` — returns a `RawSolverResult`
  (variable names, an integer sample matrix, an aligned energy vector).

The `BackendAliases` mixin in the same module supplies `name`,
`is_exhaustive` and the "no time limit" `resolve_time_limit` so a backend only
has to provide `capabilities`, `is_available` and `solve`; it is a
convenience, not a requirement.

**2. Declare `SolverCapabilities`.** The declaration is what the rest of the
system dispatches on — never the backend's name:

- `remote`, `heuristic`, `exhaustive`, `supports_seed`, `supports_num_reads`,
  `supports_num_sweeps`, `supports_time_limit`, `returns_multiple_samples`,
  `requires_embedding` — drive warnings, routing and the capabilities view.
- `supported_model_types` — the compiler path, in preference order (see
  [above](#how-the-compiler-path-is-chosen)).
- `parameter_limits` — each entry maps a dotted preference path (e.g.
  `"my_backend.time_limit_seconds"`) to a generic policy limit key and the
  catalog error code to report when the preference exceeds it. A custom key
  gets its value from `ANNEALBRIDGE_LIMITS`; the service refuses to start if a
  declared key has no value in the policy. The option block in
  `SolverPreferences` must be named exactly like the backend, because the
  service reads the preference by that dotted path.
- `credentials` — a `CredentialDeclaration` listing the backend's credential
  environment variables, HTTP header names and, if needed, value patterns.
  **Declaring is all that is required for redaction:** the declaration is fed
  to the shared redaction layer both by the backend's own `__init__` and by
  `SolverRegistry` when the backend is registered, so every error message, log
  line and metadata field masks those values from then on — including on a
  backend that was constructed directly, without a registry. (Declaring twice
  is idempotent: the layer keys declarations by backend name.) The shared
  layer knows no vendor, so a new backend is protected without touching the
  solver layer. A backend that carries credentials should therefore call
  `declare_credentials(self.capabilities.name, self.capabilities.credentials)`
  in its constructor, as the Fujitsu DA backend does. The three D-Wave backends
  get the same declaration, plus the Ocean config-file token source, from
  `solvers.ocean.ocean_sampler_holder`, the call that creates their lazy
  sampler; a new Ocean-based backend should build its sampler holder through
  it. See [Security](security.md).
- `description` — the text an agent sees in
  `get_optimization_capabilities`.

**3. Register it.** `SolverRegistry` maps a registry key to a backend
instance. The key is the name a request puts in `solver.backend` and the one
`ANNEALBRIDGE_ENABLED_BACKENDS` is matched against; the six built-in backends
register under their own `capabilities.name`.

Contributions are welcome — see [CONTRIBUTING](../CONTRIBUTING.md).
