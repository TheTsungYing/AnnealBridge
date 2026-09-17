[← Back to README](../README.md) · [Documentation index](README.md)

# Solver backends

This page describes the eight solver backends AnnealBridge ships with: what each
one runs on, which compiler path it takes, which solver preferences it honours
and which it ignores, and what it needs before it can be used. It also covers
the setup steps for the D-Wave and Fujitsu backends, how the compiler path is
decided, and what it takes to add a backend of your own.

For the environment variables named here, see
[Configuration](configuration.md). For the error and warning codes, see
[Errors and warnings](errors.md).

## Overview

| Backend | Kind | Model path | Extra needed | Credential | Best for |
| --- | --- | --- | --- | --- | --- |
| `exact` | local | bqm | none (core install) | none | small problems needing a proof |
| `simulated_annealing` | local | bqm | none (core install) | none | general purpose, small or hard-constrained |
| `tabu` | local | bqm | none (core install) | none | dense QUBOs; first choice once they are large |
| `simulated_bifurcation` | local | bqm | none (core install); optional `[gpu]` | none | large dense unconstrained |
| `dwave_qpu` | remote | bqm | `[dwave]` | D-Wave Leap, via Ocean | sparse problems small enough to embed |
| `leap_hybrid_bqm` | remote | bqm | `[dwave]` | D-Wave Leap, via Ocean | large BQMs, when quota is available |
| `leap_hybrid_cqm` | remote | cqm | `[dwave]` | D-Wave Leap, via Ocean | many hard constraints, kept native |
| `fujitsu_da` | remote | bqm | none (core install) | `FUJITSU_DA_API_KEY` | large QUBOs on vendor hardware |

**Best for** is a reading guide, not a switch. For the three local heuristics
it lines up with what `recommend` actually does, because each of them declares
its structural preference in its capabilities ([below](#how-recommend-orders-the-local-heuristics)).
The four remote rows are qualitative descriptions only: no declared field
corresponds to them, and `recommend` does not reorder the remote backends by
problem shape.

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
- `seed` must lie between `0` and `2147483647` inclusive (`0 <= seed <
  2^31`, the sampler's own rule), declared as `seed_min` / `seed_max` and
  reported by the capabilities view. A seed outside it is refused before
  anything runs: `validate` and `solve` report `INVALID_SOLVER_PREFERENCE`
  at `solver.seed` ("solver.seed must be an integer between 0 and 2147483647
  on backend simulated_annealing, got -1"), so `solve` returns
  `invalid_problem`, and `recommend` lists the same error in this backend's
  `blocking`.
- Reads are sampled in shards of 25, up to `ANNEALBRIDGE_SA_WORKERS` of
  them concurrently (see [Configuration](configuration.md)). The shard split,
  the derived shard seeds and the merge are shared code, used by
  [`tabu`](#tabu) on the same terms, so both `dwave-samplers` backends carry
  the same reproducibility contract. The shard
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
- **When to choose it.** The general-purpose local backend, and the one a
  problem gets when it names none. It has the best measured hit rate of the
  three on a penalty-dominated QUBO — the shape every problem with an
  effective hard constraint takes on the BQM path, because each such
  constraint becomes a penalty term orders of magnitude above the objective
  coefficients: about 7 % of its reads land on the shipped knapsack optimum,
  against 1–5 % for `simulated_bifurcation`. It declares no structural
  preference either way, so `recommend` ranks it ahead of the other two on a
  problem with an effective hard constraint and behind them on a large dense
  unconstrained one (see
  [below](#how-recommend-orders-the-local-heuristics)).

## `tabu`

- Local heuristic sampler: `dwave.samplers.TabuSampler`, a multistart tabu
  search. It is part of the core install — no extra, no credential — and is
  generally stronger than simulated annealing on dense QUBOs, where the
  annealer has to spend sweeps on moves tabu search rules out.
- Honours `num_reads` and `seed`. It has no notion of sweeps, so a non-default
  `num_sweeps` raises `PARAMETER_IGNORED` rather than being passed along and
  quietly ignored by the sampler.
- `seed` must lie between `0` and `4294967295` inclusive (`0 <= seed < 2^32`,
  the sampler's own rule, and **not** the simulated annealer's narrower
  range). It is declared as `seed_min` / `seed_max` and reported by the
  capabilities view; a seed outside it is refused before anything runs, with
  `INVALID_SOLVER_PREFERENCE` at `solver.seed` ("solver.seed must be an
  integer between 0 and 4294967295 on backend tabu, got -1"), so `solve`
  returns `invalid_problem` and `recommend` lists the same error in this
  backend's `blocking`.
- **Each read is one plain tabu search from one random starting point.** The
  backend always submits `timeout=None` and `num_restarts=0` instead of the
  vendor defaults; neither is reachable from a solver preference, which is why
  the backend declares no time limit: a wall-clock budget is not a dial on
  search quality here. The reasons:
  - The vendor default `timeout=20` is a 20 ms wall-clock budget **per read**,
    which makes the answer a function of the machine's speed and of whatever
    else is running on it. At 2000 variables the first search is still cut off
    at 20 ms and returns a worse best energy than the untimed search. Turning
    it off is what lets this backend return the same answer for the same seed
    on any machine.
  - With the wall clock gone, `num_restarts=0` is what bounds the work
    instead: the cost of a read is a *count* of variable updates
    (the sampler's `coefficient_z_first`), predictable in the problem size.
    Restarts are not traded away for quality — measured on dense ±1 SK
    instances at 300 and 600 variables, spending a fixed time budget on more
    restarts scored no better than spending it on more reads. Diversification
    is therefore left to `num_reads`, which the caller controls and policy
    caps.
- Measured single-threaded cost per read (`dwave-samplers` 1.8.0, Intel Core
  i5-12500): about 4 ms at 30 variables, 12 ms at 200, 88 ms at 800 and 330 ms
  at 2000. Multiply by `num_reads` to size a request.
- Reads are sampled in shards of 25, up to `ANNEALBRIDGE_TABU_WORKERS` of them
  concurrently — the same sharding the simulated annealer uses (see
  [Configuration](configuration.md)). Because the shard layout and each
  shard's seed depend on `num_reads` and `seed` alone, and the search itself
  is bounded by a count rather than a wall clock, **the result is a function
  of `(problem, num_reads, seed)` and is identical for any worker count or
  machine**; the worker count only changes wall time. A request of at most 25
  reads is a single sampler call with the seed passed straight through.
- `num_reads` is bounded by `ANNEALBRIDGE_MAX_LOCAL_READS`
  (`LOCAL_READS_LIMIT`). There is no sweeps ceiling and no time-limit ceiling,
  because the backend takes neither parameter. An over-limit value is
  rejected, never clamped.
- If no feasible solution is found, the service retries with a doubled
  hard-constraint penalty, up to `max_retries` times (bounded by
  `ANNEALBRIDGE_MAX_LOCAL_RETRIES`), exactly as on the other local heuristics.
- Being heuristic, it proves neither optimality nor infeasibility: an
  `infeasible` result carries `infeasibility_proven: false` and only means
  "not found under this configuration".
- There is **no `solver.tabu` option block**. Tenure and the rest of the
  search parameters stay at the sampler's own defaults.
- Reports `metadata` with `remote: false`, `model_type: "bqm"` and
  `num_reads_requested` set to the reads asked of the sampler — the whole
  request, not a shard. A local run has no vendor side, so `timing_us` is
  empty and the vendor fields are `null`.
- **When to choose it.** Dense QUBOs, and the first of the local heuristics
  once they are large. It matched the best energy the annealer found on the
  dense ±1 SK instances measured below in about a third of the wall time
  (11.6 s against 33 s at 1000 variables) — the difference measured was speed
  to the same energy, not a better answer. Wherever the annealer would spend
  sweeps on moves a tabu list rules out, the same advantage is *expected*, but
  only the SK instances above were measured. It therefore declares
  `strong_on_large_dense`, which puts it first among the local heuristics when
  the problem is large, dense and free of effective hard constraints. It
  declares no weakness: its hit rate on the penalty-dominated knapsack was
  never recorded, so none has been observed — which is not the same as having
  been measured and found equal. On every other shape it keeps registry order,
  behind `simulated_annealing`.

## `simulated_bifurcation`

- Local heuristic, implemented in this package on `numpy`: simulated
  bifurcation, the classical-mechanics heuristic of H. Goto, K. Endo,
  M. Suzuki, Y. Kanao, Y. Hamakawa, R. Hidaka, M. Yamasaki and K. Tatsumura,
  "High-performance combinatorial optimization based on classical mechanics",
  *Science Advances* **7**, eabe7953 (2021). Every spin is a continuous
  oscillator driven by the couplings and by a "pump" that grows from 0 to 1
  over the run; as the pump passes the bifurcation point each oscillator falls
  into `+1` or `-1`, and the signs are the answer. It needs no extra, no
  credential and no network: `numpy` is already a core dependency.
- **One step updates every variable of every read at once**, with a single
  matrix product against the dense `N × N` coupling matrix. That is what makes
  it a matrix routine rather than a loop over reads, why it is strongest on
  large dense problems, and why it has no thread setting of its own — the BLAS
  behind `numpy.matmul` already uses the cores (bound it with
  `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`).
- Honours `num_reads` (the number of parallel trajectories, the paper's
  "agents"), `num_sweeps` (the number of integration steps) and `seed`.
- `seed` must lie between `0` and `4294967295` inclusive (`0 <= seed < 2^32`,
  the same range the `tabu` backend declares). It is declared as `seed_min` /
  `seed_max` and reported by the capabilities view; a seed outside it is
  refused before anything runs, with `INVALID_SOLVER_PREFERENCE` at
  `solver.seed` ("solver.seed must be an integer between 0 and 4294967295 on
  backend simulated_bifurcation, got -1"), so `solve` returns
  `invalid_problem` and `recommend` lists the same error in this backend's
  `blocking`.
- Two variants of the dynamics are implemented, and which one wins is
  problem-dependent, so the choice is a solver option:

  ```json
  {"solver": {"backend": "simulated_bifurcation",
              "simulated_bifurcation": {"mode": "discrete"}}}
  ```

  - `"discrete"` (dSB, the default) — the force uses the *signs* of the
    positions. The stronger variant on the dense instances measured below.
  - `"ballistic"` (bSB) — the force uses the continuous positions. Weaker on
    those instances, but it finds the optimum of the shipped integer knapsack
    example, where dSB stalls at a local minimum three units above it.

  The numerical constants of the dynamics (the final pump value, the time
  step, the coupling gain, the initial amplitude) are the paper's and are not
  exposed: they are not a dial an agent can reason about.
- Measured best energy and wall time on dense ±1 SK instances, single
  precision, 1000 steps, 100 reads, on an Intel Core i5-12500 with OpenBLAS
  0.3.34:

  | Instance | `simulated_annealing` | `tabu` | bSB | dSB |
  | --- | --- | --- | --- | --- |
  | `N = 200` | −836 / 1.2 s | −836 / 1.1 s | −834 / 0.25 s | −836 / 0.4 s |
  | `N = 1000` | −9112 / 33 s | −9112 / 11.6 s | −9109 / 3.3 s | −9112 / 3.9 s |

  Cost per integration step on the same machine: 7 ms at 2000 variables with
  100 reads, 26 ms at 5000 variables with 100 reads, 39 ms at 10 000 variables
  with 32 reads. Multiply by `num_sweeps` to size a request.
- **The weak spot is the small penalty-dominated QUBO**, which is what the
  shipped examples are: with a hard-constraint penalty orders of magnitude
  above the objective coefficients, only 1 to 5 % of the reads land on the
  optimum in either mode, against about 7 % for simulated annealing — the
  knapsack example needs roughly 2000 reads to be reliable. Neither more
  integration steps nor a steepest-descent polish improved that when measured,
  so no polish is applied.
- `num_reads` is bounded by `ANNEALBRIDGE_MAX_LOCAL_READS`
  (`LOCAL_READS_LIMIT`) and `num_sweeps` by `ANNEALBRIDGE_MAX_SWEEPS`
  (`SWEEPS_LIMIT`). There is no time-limit ceiling, because there is no wall
  clock anywhere in the backend: the step count bounds the work. An over-limit
  value is rejected, never clamped.
- `ANNEALBRIDGE_SB_MAX_VARIABLES` (default `10000`) caps the compiled problem,
  including the internal slack and integer-encoding variables. The dynamics
  need the couplings as a dense single-precision `N × N` matrix — `4 N²`
  bytes, about 400 MB at the default — on top of the compiled `dimod` model
  itself, which on a dense problem holds every interaction and is the larger
  of the two. The cap is part of the backend's declaration, so the service
  refuses a problem above it *before compiling*, with `SB_VARIABLE_LIMIT`
  under `resource_limit_exceeded`, `recommend` lists the same refusal in
  `blocking`, and `capabilities` reports it as `max_variables` next to the
  policy ceilings; nothing is allocated or clamped. Reads are simulated in
  batches of a fixed 1024 columns to bound the state memory; the samples
  themselves are `num_reads × N` bytes.
- **Reproducibility is a slightly weaker promise than on the other two local
  backends.** The result is a function of `(problem, num_reads, num_sweeps,
  seed, mode, device)`: the initial states come from `numpy.random.SeedSequence`
  on every device, the batch layout is fixed, and nothing in the run depends on
  a clock. The BLAS thread count does not change the answer — measured
  bit-identical at 1 and at 6 threads. But a different CPU instruction set or a
  different BLAS build may round the matrix products differently and arrive at
  different samples, which the count-bounded `tabu` search does not do.
- If no feasible solution is found, the service retries with a doubled
  hard-constraint penalty, up to `max_retries` times (bounded by
  `ANNEALBRIDGE_MAX_LOCAL_RETRIES`), exactly as on the other local heuristics.
- Being heuristic, it proves neither optimality nor infeasibility: an
  `infeasible` result carries `infeasibility_proven: false` and only means
  "not found under this configuration".
- Reports `metadata` with `remote: false`, `model_type: "bqm"` and
  `num_reads_requested` set to the reads asked for — the whole request, not a
  batch. A local run has no vendor side, so `timing_us` is empty and the vendor
  fields are `null`.
- **When to choose it.** Large dense problems with no effective hard
  constraint, where the single matrix product per step pays off: it reached
  the same best energy as the other two on the 1000-variable SK instance in
  3.9 s against 11.6 s and 33 s — the same answer, sooner — and the GPU path
  keeps that lead as the size grows (see the timings under *Running it on a
  GPU*). Avoid it on models whose hard constraints compile to penalties, where
  its hit rate per read is the lowest of the three (1–5 % against about 7 %
  for the annealer on the shipped knapsack). It declares both preferences —
  `strong_on_large_dense` and `weak_on_penalty_dominated` — so `recommend`
  makes that trade for you on both shapes.

### Running it on a GPU

`ANNEALBRIDGE_SB_DEVICE` decides where the dynamics run: `cpu` (the default,
`numpy`) or `cuda` (PyTorch, imported lazily and only on that setting). It is a
server setting, not a policy limit and not something a request can choose —
the same problem, reads, steps, seed and mode are asked for either way.

Installing the GPU path takes two steps, because a pip extra cannot express
the first one, and the order matters:

```sh
# 1. On Windows (and anywhere the PyPI wheel is the CPU-only build), the CUDA
#    build of torch from PyTorch's own index, FIRST. Take the exact command
#    and the CUDA version tag from https://pytorch.org/get-started/.
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. The extra, which is deliberately NOT part of [all]. It finds torch
#    already satisfied and leaves the CUDA build in place.
pip install "annealbridge[gpu]"
```

The order matters because pip treats the CPU build as satisfying the `torch`
requirement: run in the other order, step 1 reports "already satisfied" and
the CPU build stays. If the CPU build is already installed, repeat step 1 with
`--force-reinstall`. Budget roughly 2.5 GB of download and 4–6 GB on disk for
the CUDA build. Step 2 alone is enough on a platform whose PyPI wheel already
carries CUDA.

**The backend never falls back to the CPU.** A silent fallback would hide a
misconfigured server behind an answer that merely arrived more slowly, so with
`ANNEALBRIDGE_SB_DEVICE=cuda` the backend reports itself unavailable instead,
in one of three states the capabilities view and `recommend` show:

| State | Reported as | Detail |
| --- | --- | --- |
| PyTorch installed and a CUDA device visible | available | — |
| PyTorch not installed | not installed | `PyTorch not installed (annealbridge[gpu] extra)` |
| PyTorch installed, no CUDA device | unavailable | `no CUDA device visible to PyTorch` |

Availability is re-checked on every call and never cached: a driver or an
install can change between requests.

Measured on an NVIDIA GeForce GTX 1660 SUPER (6 GB, Turing, no tensor cores;
PyTorch 2.14 + CUDA 12.6) against the same Intel Core i5-12500 / OpenBLAS CPU
path, dense ±1 SK instances, 1000 steps:

| Instance | `cpu`, 100 reads | `cuda`, 100 reads | `cuda`, 1000 reads | peak VRAM |
| --- | --- | --- | --- | --- |
| `N = 1000` | 2.3 s | 0.8 s | 3.4 s | 47 MB |
| `N = 5000` | 25.5 s | 6.4 s | 46 s | 278 MB |
| `N = 10 000` | — | 23 s | 203 s | 740 MB |

The GPU is three to four times faster at 100 reads and is not yet saturated
there: ten times the reads cost four to nine times the wall time. Below a few
hundred variables the fixed cost of launching the kernels dominates and the CPU
path is the better choice.

The reproducibility rule above is unchanged on `cuda`: the initial states are
still drawn by `numpy`, so the same request is a function of the same inputs,
with the device among them. Measured on this card, the same request twice
returned bit-identical samples at every size above. The `cuda` and `cpu`
samples for the same request are *not* the same — the two round the matrix
products differently — and both reach the same best energy on the `N = 1000`
instance and on the knapsack example; the device is part of the function.

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
microseconds). A value that does not parse, or is not finite once converted
(`"NaN"`, `"inf"`, an integer too large for a float, or a millisecond count
that overflows), is dropped on its own; the other field is kept.

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

`exact`, `simulated_annealing`, `tabu`, `simulated_bifurcation`, `dwave_qpu`,
`leap_hybrid_bqm` and `fujitsu_da` all declare `bqm`; `leap_hybrid_cqm`
declares `cqm`. On the BQM
path hard constraints become penalty terms, inequalities get binary slack
variables, and integer variables are binary-encoded; on the CQM path
constraints and integers are native. See
[Problem format](problem-format.md) for what each path means for a problem, and
[Architecture](architecture.md) for where the compilers sit.

### How `recommend` orders the local heuristics

`recommend` already separates the backends into tiers — an `exact` that fits
first, then the local heuristics, then the remote ones. Inside the local
heuristic tier the three backends used to come out in registry order, which
said nothing about the problem. A `structure_fit` key now orders them, applied
after the tier and before registry order, and driven — like everything else in
`recommend` — by what a backend **declares**, never by its name.

Two declarations exist, both `False` unless a backend opts in:

- `strong_on_large_dense` — the backend reaches the same energy as its peers
  in a fraction of the time on a large dense model with no effective hard
  constraint. `tabu` and `simulated_bifurcation` declare it.
- `weak_on_penalty_dominated` — the backend has a measured lower hit rate per
  read once hard constraints enter the model as penalty terms. Only
  `simulated_bifurcation` declares it.

Both are matched **only on the bqm path**. The shapes below describe a
compiled BQM, so a backend whose selected model type is `cqm` is never matched
even if it declares a flag: its hard constraints stay native and never become
penalties, and the density of a BQM says nothing about it.

The shapes are computed from the problem document and the estimate `validate`
already reports, so no compiling and no solving happens:

- **Size** — at least 500 estimated compiled variables, using the bqm estimate,
  which already includes slack variables and integer encoding bits.
- **Density** — the number of **distinct** variable pairs the model will
  couple, divided by `m(m−1)/2`, where `m` is the number of variables the
  document declares. A pair is counted from a quadratic objective term whose
  coefficients sum to something non-zero and that is not a variable squared
  with itself, and from the clique of every *effective* constraint — hard and
  soft alike, since both put their variables into the same quadratic block. A
  pair contributed twice, by two terms or by two constraints, counts once. The
  threshold is `0.5`.
- **Effective hard constraint** — a constraint whose type is `hard`, that has
  at least one non-zero coefficient, and that is not a redundant inequality.
  `x <= 1` on a binary variable is redundant: it holds over the declared
  bounds, raises `REDUNDANT_CONSTRAINT`, and the compiler emits no penalty for
  it, so it does not count here either. An all-zero constraint is skipped for
  the same reason.

The two shapes are then:

- **Large dense** — the bqm path, at or above both thresholds, and **no**
  effective hard constraint anywhere in the document.
- **Penalty-dominated** — the bqm path with **at least one** effective hard
  constraint. On that path each of them becomes a penalty weighted far above
  the objective, which is exactly the regime the weakness was measured in.
  Size and density are irrelevant here.

A backend then scores `0` when it declares the strength and the problem is
large dense (reason code `R_DENSE_STRENGTH`), `2` when it declares the weakness
and the problem is penalty-dominated (`R_PENALTY_WEAKNESS`), and `1` otherwise;
ties fall back to registry order. The result, for the three local heuristics:

| Problem | Order |
| --- | --- |
| Large, dense, no effective hard constraint | `tabu`, `simulated_bifurcation`, `simulated_annealing` |
| Any effective hard constraint on the bqm path (the shipped knapsack) | `simulated_annealing`, `tabu`, `simulated_bifurcation` |
| Small, no effective hard constraint (neither shape) | `simulated_annealing`, `tabu`, `simulated_bifurcation` |

Each of those orders is the order *within the local heuristic tier*. An
`exact` the problem fits still ranks first overall, ahead of all three.

The key only reorders backends that already share a tier: `exact` and the
remote backends are unaffected, and nothing here makes an unusable backend
usable or changes which errors are reported.

The two thresholds are calibrated on the measurements recorded in
[`CHANGELOG.md`](../CHANGELOG.md) and repeated in the backend sections above —
dense SK instances where all three reached the same energy (−9112 at 1000
variables) but dSB took 3.9 s against `tabu`'s 11.6 s and the annealer's 33 s,
and the penalty-dominated knapsack where simulated bifurcation hits the
optimum in 1–5 % of its reads against about 7 % for the annealer. They are
deliberately coarse: `recommend` still performs no cost estimation, runs no
benchmark and keeps no history, so the ordering stays deterministic and cheap.

#### What the rule gives up

The rule is deliberately conservative, and its worst case is worth stating
plainly. A problem that is **large and dense but carries one effective hard
constraint** is penalty-dominated, so it ranks `simulated_annealing`, `tabu`,
`simulated_bifurcation` — even at a size where simulated bifurcation might
finish many times sooner. That is intentional: the speed advantage was
measured only on unconstrained SK instances, and no measurement exists for
large models whose hard constraints compile to penalties. Rather than
extrapolate from one shape to the other, `recommend` declines to recommend
what it has no data for.

So the recommendation is a floor, not a ceiling. On a large dense problem that
does have a hard constraint, an agent should say so to the user — the ranking
is playing safe, and trying `simulated_bifurcation` may be much faster or may
miss the feasible region — or ask which trade the user wants, rather than
present the top entry as the only sensible choice.

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
- `strong_on_large_dense` / `weak_on_penalty_dominated` — optional structural
  preferences, both `False` by default, and **only matched on the bqm path**:
  a backend that compiles to cqm gains nothing by declaring either, because
  neither shape is computed for it. Declare the first only if the backend has
  a **measured** advantage on large dense models with no effective hard
  constraint, the second only if it has a measured disadvantage once hard
  constraints compile to penalties. They order `recommend` inside a tier and
  nothing else: they never change availability, limits or the compiler path,
  and a backend that declares neither simply keeps registry order. Record the
  measurement that justifies the claim, as the two heuristics that declare
  them do; see [above](#how-recommend-orders-the-local-heuristics).
- `seed_min` / `seed_max` — the inclusive range of `solver.seed` the backend
  itself accepts. Unlike `parameter_limits` this is the backend's own rule,
  not a policy ceiling: it takes no value from policy and no
  `ANNEALBRIDGE_*` setting changes it. Declare both or neither, only with
  `supports_seed=True`, and with `seed_min <= seed_max`; any other
  declaration fails when `SolverCapabilities` is constructed. With a range
  declared, validation refuses a seed outside it with
  `INVALID_SOLVER_PREFERENCE` at `solver.seed`, so `validate`, `solve` and
  `recommend` agree before the backend is called, and the capabilities view
  reports the range.
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
`ANNEALBRIDGE_ENABLED_BACKENDS` is matched against; the eight built-in
backends register under their own `capabilities.name`.

Contributions are welcome — see [CONTRIBUTING](../CONTRIBUTING.md).
